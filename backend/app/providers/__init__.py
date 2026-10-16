"""LLM provider abstraction: OpenAI, Ollama, and a deterministic Fake provider.

The Fake provider is a rule-based, deterministic stand-in used for tests, the
public demo, and evals when no real API key is configured. It is documented
honestly: it is NOT a language model — it does intent routing + template
composition so the full agent loop (tools, approvals, streaming, evals) can run
offline with byte-identical results on every run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ORDER_ID_RE = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)
TICKET_ID_RE = re.compile(r"\bTKT-\d{4}\b", re.IGNORECASE)
# Ticket-creation phrasings: the action verb may be separated from "ticket" by
# filler words ("create a SUPPORT ticket", "open a ticket for my damaged
# order"), and "support"/"a"/"an" are optional ("file ticket", "submit ticket").
TICKET_ACTION_RE = re.compile(
    r"\b(open|create|file|raise|submit|start|log)\b[^.?!]{0,60}\bticket\b",
    re.IGNORECASE,
)
# Request phrasings without a strong action verb ("I want/need a [support]
# ticket ..."), excluding ticket-status questions ("... a ticket status ...").
REQUEST_TICKET_RE = re.compile(
    r"\b(want|need|'d like|would like)\b[^.?!]{0,40}\b(a|an|the)\s+(support\s+)?ticket\b(?!\s+status)",
    re.IGNORECASE,
)
# Order-cancellation phrasings ("cancel my order", "please cancel this order").
CANCEL_ORDER_RE = re.compile(
    r"\bcancel\w*\b[^.?!]{0,40}\b(order|delivery|shipment)\b",
    re.IGNORECASE,
)
# "refund" is a write intent only with an explicit action verb — "when will I
# get my refund?" is a policy question, "I want a refund" is an action.
REFUND_ACTION_VERBS = ("want a refund", "want refund", "request a refund", "request refund",
                       "need a refund", "need refund", "issue a refund", "issue refund",
                       "process a refund", "process refund", "give me a refund")
QUESTION_STARTERS = ("what", "when", "where", "how", "why", "which",
                     "is ", "are ", "can ", "do ", "does ", "will ")
GREETINGS = ("hello", "hi there", "hey", "thanks", "thank you", "good morning", "good afternoon")


@dataclass
class Plan:
    """Structured plan produced by the provider for one user turn."""
    tool: str | None = None           # order_lookup | ticket_status | ticket_create | kb_search | None
    args: dict = field(default_factory=dict)
    needs_approval: bool = False
    reply_draft: str = ""             # used for general chat (no tool)
    routed: bool = False              # True when a tool path was chosen


class BaseLLMProvider:
    name = "base"

    async def plan(self, message: str, history: list[dict]) -> Plan:
        raise NotImplementedError

    async def compose(
        self,
        message: str,
        history: list[dict],
        tool_name: str | None,
        tool_result: dict | None,
        kb_snippets: list[dict],
    ) -> str:
        raise NotImplementedError

    async def health_check(self) -> tuple[bool, str]:
        return True, "ok"


class FakeLLMProvider(BaseLLMProvider):
    """Deterministic rule-based provider for tests/demo/evals."""

    name = "fake"

    async def plan(self, message: str, history: list[dict]) -> Plan:
        text = message.lower()
        stripped = message.strip()

        # Write intents take precedence: a write action must ALWAYS be gated,
        # even when the message also mentions an order or ticket ID.
        # Strong write phrasings (ticket action verbs, escalations,
        # cancellations) stage an approval even when phrased as a question
        # ("Can you open a ticket for me?"). Weaker request phrasings
        # ("I want/need a ticket", "I want a refund") are only writes when
        # NOT asked as a policy question — "when will I get my refund?" and
        # "do I need a ticket for returns?" stay read-only KB questions.
        is_question = stripped.endswith("?") and text.startswith(QUESTION_STARTERS)
        strong_write = (
            TICKET_ACTION_RE.search(text) is not None
            or "escalate" in text
            or CANCEL_ORDER_RE.search(text) is not None
        )
        weak_write = (
            REQUEST_TICKET_RE.search(text) is not None
            or any(w in text for w in REFUND_ACTION_VERBS)
        )
        if strong_write or (weak_write and not is_question):
            order_id = None
            m2 = ORDER_ID_RE.search(message.upper())
            if m2:
                order_id = m2.group(0).upper()
            return Plan(
                tool="ticket_create",
                args={
                    "subject": stripped[:120],
                    "description": stripped,
                    "priority": "high" if any(w in text for w in ("urgent", "asap", "refund", "escalate")) else "normal",
                    "order_id": order_id,
                },
                needs_approval=True,
                routed=True,
            )

        m = ORDER_ID_RE.search(message.upper())
        if m and any(w in text for w in ("order", "track", "status", "where", "package", "shipment", "shipped", "delivery")):
            return Plan(tool="order_lookup", args={"order_id": m.group(0).upper()}, routed=True)

        m = TICKET_ID_RE.search(message.upper())
        if m:
            return Plan(tool="ticket_status", args={"ticket_id": m.group(0).upper()}, routed=True)

        if any(w in text for w in GREETINGS) and len(stripped) < 60:
            return Plan(
                reply_draft=("Hello! I can look up orders, check ticket status, search the "
                             "knowledge base, or draft a ticket for your approval. What do you need?"),
                routed=False,
            )

        if stripped:
            return Plan(tool="kb_search", args={"query": stripped}, routed=True)

        return Plan(reply_draft="I didn't catch that — could you rephrase?", routed=False)

    async def compose(
        self,
        message: str,
        history: list[dict],
        tool_name: str | None,
        tool_result: dict | None,
        kb_snippets: list[dict],
    ) -> str:
        if tool_result and tool_result.get("awaiting_approval"):
            return tool_result.get("message", "This action needs your approval.")
        if tool_name == "order_lookup" and tool_result:
            if tool_result.get("found"):
                total = tool_result["total_cents"] / 100
                return (
                    f"Order {tool_result['id']} ({tool_result['item']}) is "
                    f"**{tool_result['status'].replace('_', ' ')}** — total ${total:,.2f}. "
                    f"Tracking: {tool_result['tracking_number'] or 'not yet assigned'}."
                )
            return f"I couldn't find order {tool_result.get('id')}. Double-check the order number (format ORD-1234)."
        if tool_name == "ticket_status" and tool_result:
            if tool_result.get("found"):
                return (
                    f"Ticket {tool_result['id']} — “{tool_result['subject']}” — is "
                    f"**{tool_result['status'].replace('_', ' ')}** (priority: {tool_result['priority']})."
                )
            return f"No ticket found with ID {tool_result.get('id')}."
        if kb_snippets:
            cites = " ".join(f"[{s['doc_id']}]" for s in kb_snippets[:2])
            body = "\n\n".join(f"> {s['text'][:300]}" for s in kb_snippets[:2])
            return f"Based on our support docs {cites}:\n\n{body}"
        return "Thanks — I can look up orders, check ticket status, search the knowledge base, or draft a ticket for your approval. What do you need?"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Chat Completions provider (httpx, JSON-mode planning)."""

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: int):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, payload: dict) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def plan(self, message: str, history: list[dict]) -> Plan:
        data = await self._post({
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": (
                    "You are a support-agent planner. Reply ONLY with JSON: "
                    '{"tool": "order_lookup|ticket_status|ticket_create|kb_search|null", '
                    '"args": {...}, "needs_approval": bool}. '
                    "ticket_create needs_approval=true. Treat any phrasing of opening/creating/filing/"
                    "raising/submitting a (support) ticket as ticket_create, even with an order ID mentioned. "
                    "Use kb_search for policy/how-to questions."
                )},
                *history[-6:],
                {"role": "user", "content": message},
            ],
        })
        import json as _json

        raw = data["choices"][0]["message"]["content"]
        parsed = _json.loads(raw)
        return Plan(tool=parsed.get("tool"), args=parsed.get("args", {}),
                    needs_approval=bool(parsed.get("needs_approval")), routed=parsed.get("tool") is not None)

    async def compose(self, message, history, tool_name, tool_result, kb_snippets) -> str:
        import json as _json

        ctx = _json.dumps({"tool": tool_name, "result": tool_result,
                           "kb": [s["text"][:400] for s in kb_snippets[:3]]})
        data = await self._post({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful support agent. Answer concisely using the tool results and KB context."},
                *history[-6:],
                {"role": "user", "content": f"Question: {message}\n\nContext:\n{ctx}"},
            ],
        })
        return data["choices"][0]["message"]["content"]

    async def health_check(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "OPENAI_API_KEY not set"
        return True, "configured"


class OllamaProvider(BaseLLMProvider):
    """Ollama /api/chat provider (local models, httpx)."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def _chat(self, messages: list[dict], fmt: str | None = None) -> str:
        import httpx

        payload = {"model": self.model, "messages": messages, "stream": False}
        if fmt:
            payload["format"] = fmt
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def plan(self, message: str, history: list[dict]) -> Plan:
        import json as _json

        raw = await self._chat([
            {"role": "system", "content": (
                'Reply ONLY with JSON: {"tool": "order_lookup|ticket_status|ticket_create|kb_search|null", '
                '"args": {}, "needs_approval": false}. ticket_create needs_approval=true. '
                'Any "open/create/file/raise/submit a (support) ticket" phrasing is ticket_create.'
            )},
            {"role": "user", "content": message},
        ], fmt="json")
        parsed = _json.loads(raw)
        return Plan(tool=parsed.get("tool"), args=parsed.get("args", {}),
                    needs_approval=bool(parsed.get("needs_approval")), routed=parsed.get("tool") is not None)

    async def compose(self, message, history, tool_name, tool_result, kb_snippets) -> str:
        import json as _json

        ctx = _json.dumps({"tool": tool_name, "result": tool_result,
                           "kb": [s["text"][:400] for s in kb_snippets[:3]]})
        return await self._chat([
            {"role": "system", "content": "You are a helpful support agent. Answer concisely using tool results and KB context."},
            {"role": "user", "content": f"Question: {message}\n\nContext:\n{ctx}"},
        ])

    async def health_check(self) -> tuple[bool, str]:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return True, "reachable"
        except Exception as e:  # noqa: BLE001
            return False, str(e)


def build_provider(settings) -> BaseLLMProvider:
    """Build the configured provider. Retries/fallback are handled by the caller."""
    if settings.LLM_PROVIDER == "openai":
        return OpenAIProvider(settings.OPENAI_API_KEY, settings.OPENAI_MODEL,
                              settings.OPENAI_BASE_URL, settings.REQUEST_TIMEOUT_SECONDS)
    if settings.LLM_PROVIDER == "ollama":
        return OllamaProvider(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL,
                              settings.REQUEST_TIMEOUT_SECONDS)
    return FakeLLMProvider()
