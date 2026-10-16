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
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from ..errors import APIError, PROVIDER_UNAVAILABLE
from ..logging_setup import Timer, log_event
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
    plan, provider_used = await plan_with_retry(
        provider, fallback, message, history,
        settings.PROVIDER_MAX_RETRIES, settings.PROVIDER_BACKOFF_BASE)

    tool_calls: list[dict] = []
    tool_result_data: dict | None = None
    approval = None
    kb_snippets: list[dict] = []
    final_tool = None

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

    reply = await compose_with_retry(
        provider if provider_used == provider.name else fallback, fallback,
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
