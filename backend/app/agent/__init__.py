"""Agentic workflow: an explicit, inspectable state machine.

States: PLAN -> (TOOL | APPROVAL_GATE) -> COMPOSE -> DONE

This is deliberately NOT a LangChain/LangGraph dependency: the graph is small
enough that an explicit state machine is easier to test, log, and reason about
than a framework DAG. Every transition is logged with the request ID.

Human-in-the-loop: TicketCreateTool.draft() stages a pending Approval instead
of executing. The agent response includes the approval card; a human resolves
it via /api/approvals/{id}/approve|reject. Approved approvals execute the
staged tool call exactly once (status flips to approved/approved atomically in
the resolver).

Planning is HYBRID (deliberate, documented):
- A deterministic intent router (the rule-based FakeLLMProvider.plan) plans
  EVERY turn first: order/ticket lookups, write phrasings, no-order-ID
  clarification, greetings, and KB search all route deterministically (100% on
  the 24-case eval set). A 1B local LLM proved unreliable at tool-calling JSON
  (malformed tool names like "order_lookup|null", dropped args, invented
  citations), so the LLM planner is only a fallback for messages the rules
  can't handle. The turn's reported provider is "rules" when the deterministic
  router planned it — honest, and visible per-turn in the UI.
- Composition: structured tool results (order/ticket lookups) and approval
  drafts always use deterministic templates so the agent can never contradict
  the database; the configured LLM (e.g. Ollama) composes KB answers and open
  chat in natural language.
- Deterministic safety rails then validate ANY plan: invalid tool names are
  repaired, missing order/ticket IDs are extracted from the raw message or
  turned into a clarification.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from ..errors import APIError, PROVIDER_UNAVAILABLE
from ..logging_setup import Timer, log_event
from ..providers import FakeLLMProvider, Plan, TICKET_ID_RE, normalize_order_id
from ..tools import TOOLS

logger = logging.getLogger("copilot.agent")


async def plan_with_retry(provider, fallback, message: str, history: list[dict],
                          max_retries: int, backoff_base: float):
    """Call provider.plan with exponential backoff; fall back to the fake provider.

    Returns (plan, provider_used_name). Raises APIError(PROVIDER_UNAVAILABLE)
    only if every provider (including the fake fallback) fails.
    """
    attempt = 0
    primary = provider
    for prov in (primary, fallback):
        attempt_label = f"{prov.name}"
        for retry in range(max_retries if prov is primary else 1):
            try:
                timer = Timer()
                plan = await prov.plan(message, history)
                log_event(logger, "plan_ok", provider=prov.name,
                          tool=plan.tool, latency_ms=round(timer.elapsed_ms(), 1))
                return plan, prov.name
            except Exception as e:  # noqa: BLE001 - provider failure is expected
                wait = backoff_base * (2 ** retry)
                log_event(logger, "provider_plan_failed", provider=prov.name,
                          retry=retry, error=str(e)[:200])
                await asyncio.sleep(wait)
        log_event(logger, "provider_exhausted", provider=attempt_label)
    raise APIError(PROVIDER_UNAVAILABLE[0],
                   "All LLM providers failed (see server logs). Please try again shortly.",
                   PROVIDER_UNAVAILABLE[1])


async def compose_with_retry(provider, fallback, **kwargs):
    """Same retry/fallback wrapper for the compose step."""
    for prov in (provider, fallback):
        try:
            return await prov.compose(**kwargs)
        except Exception as e:  # noqa: BLE001
            log_event(logger, "provider_compose_failed", provider=prov.name,
                      error=str(e)[:200])
    raise APIError(PROVIDER_UNAVAILABLE[0],
                   "All LLM providers failed (see server logs). Please try again shortly.",
                   PROVIDER_UNAVAILABLE[1])


async def run_turn(session, provider, fallback, retriever, message: str,
                   history: list[dict], settings) -> dict:
    """Run one agent turn. Returns a serializable result dict."""
    timer = Timer()

    # ── STATE: PLAN ──────────────────────────────────────────────────────
    # Deterministic intent override: the rule-based router is 100% accurate on
    # all covered intents (see module docstring). It plans every turn; the LLM
    # planner remains only as a fallback for messages the rules can't handle
    # (e.g. empty input). "rules" is reported honestly as the turn's provider.
    det_plan = await FakeLLMProvider().plan(message, history)
    if det_plan.routed or det_plan.reply_draft:
        plan, provider_used = det_plan, "rules"
        log_event(logger, "plan_override", tool=plan.tool,
                  reason="deterministic intent routing")
    else:
        plan, provider_used = await plan_with_retry(
            provider, fallback, message, history,
            settings.PROVIDER_MAX_RETRIES, settings.PROVIDER_BACKOFF_BASE)

    tool_calls: list[dict] = []
    tool_result_data: dict | None = None
    approval = None
    kb_snippets: list[dict] = []
    final_tool = None

    # ── Deterministic safety rails on the planner output ─────────────────
    # These hold for ANY provider (rule-based or LLM): a planner can hallucinate
    # a tool call with missing/empty args, so the agent layer validates before
    # dispatching. Never let a lookup run without its ID.
    if plan.tool and plan.tool not in TOOLS:
        # Small models sometimes emit malformed tool strings ("order_lookup|null").
        # Try to recover a valid tool name; otherwise treat as no tool.
        import re as _re
        m = _re.search(r"order_lookup|ticket_status|ticket_create|kb_search", plan.tool)
        log_event(logger, "plan_repaired", tool=plan.tool,
                  reason="invalid tool name" + (f" -> {m.group(0)}" if m else " -> none"))
        plan.tool = m.group(0) if m else None
        plan.routed = plan.tool is not None
    if plan.tool == "order_lookup":
        # ID extraction is deterministic: prefer the planner's arg, but fall
        # back to scanning the raw message (small models often drop args).
        oid = (normalize_order_id(str(plan.args.get("order_id") or ""))
               or normalize_order_id(message))
        if not oid:
            log_event(logger, "plan_repaired", tool="order_lookup",
                      reason="missing order_id -> clarification")
            plan = Plan(tool=None, reply_draft=(
                "I'd be happy to check that — could you share your order ID? "
                "It's the 4-digit number in your confirmation email "
                "(format ORD-1234; demo orders run ORD-1001 to ORD-1005)."),
                routed=False)
        else:
            plan.args["order_id"] = oid
    if plan.tool == "ticket_status":
        tid = None
        m = TICKET_ID_RE.search(str(plan.args.get("ticket_id") or "")) or TICKET_ID_RE.search(message)
        if m:
            tid = m.group(0).upper()
        if not tid:
            log_event(logger, "plan_repaired", tool="ticket_status",
                      reason="missing ticket_id -> clarification")
            plan = Plan(tool=None, reply_draft=(
                "I can check that — what's the ticket ID? It looks like TKT-2001."),
                routed=False)
        else:
            plan.args["ticket_id"] = tid

    # ── STATE: TOOL / APPROVAL_GATE ────────────────────────────────────────
    if plan.tool and plan.tool in TOOLS:
        tool = TOOLS[plan.tool]
        record = {"tool": tool.name, "args": plan.args}
        if tool.needs_approval:
            # APPROVAL_GATE: stage only, never execute.
            result = tool.draft(session, plan.args)
            approval = {
                "id": result.data["approval_id"],
                "action": result.data["action"],
                "status": "pending",
                "payload": result.data["payload"],
            }
            tool_result_data = {"awaiting_approval": True, "message": result.message}
            record["status"] = "awaiting_approval"
            record["approval_id"] = result.data["approval_id"]
            log_event(logger, "approval_staged", approval_id=result.data["approval_id"],
                      action="ticket_create")
        else:
            try:
                result = tool.execute(session, plan.args)
                if tool.name == "kb_search":
                    kb_snippets = result.data.get("snippets", [])
                else:
                    tool_result_data = result.data
                record["status"] = "ok"
            except APIError as e:
                # Tool-level errors (e.g. order not found) are recoverable:
                # the agent explains them to the user instead of crashing.
                record["status"] = "error"
                record["error"] = e.message
                tool_result_data = {"found": False, "id": plan.args.get("order_id") or plan.args.get("ticket_id"),
                                    "error": e.message}
                log_event(logger, "tool_error", tool=tool.name, code=e.code)
        tool_calls.append(record)
        final_tool = tool.name
    elif plan.tool:
        tool_calls.append({"tool": plan.tool, "status": "unknown_tool"})

    # ── STATE: COMPOSE ───────────────────────────────────────────────────
    if not plan.routed and not plan.tool:
        # General chat: the provider already drafted the reply.
        reply = plan.reply_draft or "How can I help you today?"
        log_event(logger, "turn_done", provider=provider_used, tool=None,
                  latency_ms=round(timer.elapsed_ms(), 1))
        return {
            "reply": reply,
            "tool_calls": tool_calls,
            "approval": approval,
            "sources": [],
            "provider": provider_used,
        }

    # Structured tool results (order/ticket lookups) are rendered with the
    # deterministic template composer — never the LLM. A small local model can
    # contradict the database or invent citations; templates can't. The LLM
    # still composes KB answers and general chat. Approval drafts always use
    # the template path too (awaiting_approval short-circuits in compose()).
    if final_tool in ("order_lookup", "ticket_status") and tool_result_data is not None:
        reply = await FakeLLMProvider().compose(
            message=message, history=history, tool_name=final_tool,
            tool_result=tool_result_data, kb_snippets=kb_snippets)
    else:
        # Natural-language composition always tries the configured provider
        # first (e.g. Ollama for KB answers), then the fake fallback —
        # regardless of which planner produced the plan.
        reply = await compose_with_retry(
            provider, fallback,
            message=message, history=history, tool_name=final_tool,
            tool_result=tool_result_data, kb_snippets=kb_snippets)

    log_event(logger, "turn_done", provider=provider_used, tool=final_tool,
              approval=approval["id"] if approval else None,
              latency_ms=round(timer.elapsed_ms(), 1))
    return {
        "reply": reply,
        "tool_calls": tool_calls,
        "approval": approval,
        "sources": [{"doc_id": s["doc_id"], "title": s["title"], "chunk_id": s["chunk_id"],
                     "score": s["score"]} for s in kb_snippets],
        "provider": provider_used,
    }


def resolve_approval(session, approval_id: str, approve: bool, reason: str | None = None) -> dict:
    """Resolve a pending approval. On approve, execute the staged tool call once."""
    from ..db import Approval as ApprovalModel

    ap = session.get(ApprovalModel, approval_id)
    if ap is None:
        from ..errors import APPROVAL_NOT_FOUND
        raise APIError(APPROVAL_NOT_FOUND[0], f"Approval {approval_id} not found.",
                       APPROVAL_NOT_FOUND[1])
    if ap.status != "pending":
        from ..errors import APPROVAL_ALREADY_RESOLVED
        raise APIError(APPROVAL_ALREADY_RESOLVED[0],
                       f"Approval {approval_id} is already {ap.status}.",
                       APPROVAL_ALREADY_RESOLVED[1])
    ap.status = "approved" if approve else "rejected"
    ap.reason = reason
    from datetime import datetime, timezone
    ap.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)

    executed = None
    if approve:
        payload = json.loads(ap.payload_json)
        if ap.action == "ticket_create":
            result = TOOLS["ticket_create"].execute(session, payload)
            executed = result.data
    session.commit()
    log_event(logger, "approval_resolved", approval_id=approval_id, approved=approve)
    return {"id": approval_id, "status": ap.status, "executed": executed}
