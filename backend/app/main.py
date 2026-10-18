"""AgentDesk API."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .agent import resolve_approval, run_turn
from .config import settings
from .db import Approval, ChatMessage, Order, Ticket, get_engine, init_db, session_factory
from .embeddings import CachedEmbeddings, FakeEmbeddings, OpenAIEmbeddings, TTLCache
from .errors import APIError, APPROVAL_NOT_FOUND, ORDER_NOT_FOUND, TICKET_NOT_FOUND, register_error_handlers
from .logging_setup import Timer, get_request_id, log_event, new_request_id, set_request_id, setup_logging
from .providers import FakeLLMProvider, build_provider
from .rag import Retriever
from .seed import seed_db
from .security import require_api_key
from .tools import register_tools

logger = setup_logging(settings.LOG_LEVEL)

# Shared singletons (built once at startup; replaced in tests)
_engine = None
_SessionLocal = None
_provider = None
_fallback_provider = None
_retriever = None
_embedding_cache = None
_retrieval_cache = None


def build_runtime(database_url: str | None = None, provider_name: str | None = None):
    """Construct engine/session/provider/retriever. Used by app startup and tests."""
    global _engine, _SessionLocal, _provider, _fallback_provider, _retriever
    global _embedding_cache, _retrieval_cache

    url = database_url or settings.DATABASE_URL
    if url.startswith("sqlite") and url != "sqlite:///:memory:":
        os.makedirs(os.path.dirname(os.path.abspath(url.split("sqlite:///")[-1])), exist_ok=True)
    _engine = get_engine(url)
    init_db(_engine)
    _SessionLocal = session_factory(_engine)

    with _SessionLocal() as s:
        seed_db(s)

    pname = (provider_name or settings.LLM_PROVIDER or "fake").lower()
    if settings.DEMO_MODE:
        pname = "fake"

    class _S:  # minimal settings view for build_provider
        pass
    s_view = _S()
    s_view.LLM_PROVIDER = pname
    s_view.OPENAI_API_KEY = settings.OPENAI_API_KEY
    s_view.OPENAI_MODEL = settings.OPENAI_MODEL
    s_view.OPENAI_BASE_URL = settings.OPENAI_BASE_URL
    s_view.OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
    s_view.OLLAMA_MODEL = settings.OLLAMA_MODEL
    s_view.REQUEST_TIMEOUT_SECONDS = settings.REQUEST_TIMEOUT_SECONDS
    _provider = build_provider(s_view)
    _fallback_provider = FakeLLMProvider()

    if pname == "openai" and settings.OPENAI_API_KEY:
        inner = OpenAIEmbeddings(settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL,
                                timeout=settings.REQUEST_TIMEOUT_SECONDS)
    else:
        inner = FakeEmbeddings()
    _embedding_cache = TTLCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
    _retrieval_cache = TTLCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
    embeddings = CachedEmbeddings(inner, _embedding_cache)
    _retriever = Retriever(embeddings, cache=_retrieval_cache)
    register_tools(_retriever)
    log_event(logger, "runtime_built", provider=_provider.name, db=url.split("://")[0])
    return _engine


def get_session() -> Session:
    s = _SessionLocal()
    try:
        yield s
    finally:
        s.close()


# ── Schemas ──────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=64)


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]
    approval: dict | None
    sources: list[dict]
    provider: str
    request_id: str


class ApproveRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


class TicketCreateRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    priority: str = Field(default="normal", max_length=20)
    order_id: str | None = Field(default=None, max_length=16)


# ── App factory ──────────────────────────────────────────────────────────
def create_app(database_url: str | None = None, provider_name: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        build_runtime(database_url, provider_name)
        yield

    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    register_error_handlers(app)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_request_id()
        set_request_id(rid)
        timer = Timer()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log_event(logger, "http", method=request.method, path=request.url.path,
                  status=response.status_code, latency_ms=round(timer.elapsed_ms(), 1))
        return response

    # ── Health / readiness (no auth) ─────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": settings.APP_NAME}

    @app.get("/ready")
    async def ready():
        ok, detail = await _provider.health_check()
        return {"ready": ok, "provider": _provider.name, "provider_detail": detail,
                "model": getattr(_provider, "model", None),
                "demo_mode": settings.DEMO_MODE,
                "db": "ok", "cache": _retrieval_cache.stats()}

    # ── Chat ─────────────────────────────────────────────────────────────
    def _history(session: Session, session_id: str) -> list[dict]:
        rows = (session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id.desc()).limit(10).all())
        return [{"role": r.role, "content": r.content} for r in reversed(rows)]

    @app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    async def chat(req: ChatRequest, session: Session = Depends(get_session)):
        history = _history(session, req.session_id)
        session.add(ChatMessage(session_id=req.session_id, role="user", content=req.message))
        session.commit()
        result = await run_turn(session, _provider, _fallback_provider, _retriever,
                                req.message, history, settings)
        session.add(ChatMessage(session_id=req.session_id, role="assistant", content=result["reply"]))
        session.commit()
        return ChatResponse(**result, request_id=get_request_id())

    @app.post("/api/chat/stream", dependencies=[Depends(require_api_key)])
    async def chat_stream(req: ChatRequest, session: Session = Depends(get_session)):
        history = _history(session, req.session_id)
        session.add(ChatMessage(session_id=req.session_id, role="user", content=req.message))
        session.commit()

        async def event_gen():
            result = await run_turn(session, _provider, _fallback_provider, _retriever,
                                    req.message, history, settings)
            yield f"event: meta\ndata: {json.dumps({'tool_calls': result['tool_calls'], 'approval': result['approval'], 'sources': result['sources'], 'provider': result['provider']})}\n\n"
            # Stream the reply in word chunks (fake provider has no token stream;
            # real providers' streaming can replace this generator).
            for word in result["reply"].split(" "):
                yield f"event: token\ndata: {json.dumps({'text': word + ' '})}\n\n"
            yield "event: done\ndata: {}\n\n"
            session.add(ChatMessage(session_id=req.session_id, role="assistant", content=result["reply"]))
            session.commit()

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # ── Orders ───────────────────────────────────────────────────────────
    @app.get("/api/orders/{order_id}", dependencies=[Depends(require_api_key)])
    async def get_order(order_id: str, session: Session = Depends(get_session)):
        order = session.get(Order, order_id.upper())
        if order is None:
            raise APIError(ORDER_NOT_FOUND[0], f"Order {order_id} not found.", ORDER_NOT_FOUND[1])
        return {"id": order.id, "customer_name": order.customer_name, "item": order.item,
                "total_cents": order.total_cents, "status": order.status,
                "tracking_number": order.tracking_number}

    # ── Tickets ──────────────────────────────────────────────────────────
    @app.get("/api/tickets/{ticket_id}", dependencies=[Depends(require_api_key)])
    async def get_ticket(ticket_id: str, session: Session = Depends(get_session)):
        ticket = session.get(Ticket, ticket_id.upper())
        if ticket is None:
            raise APIError(TICKET_NOT_FOUND[0], f"Ticket {ticket_id} not found.", TICKET_NOT_FOUND[1])
        return {"id": ticket.id, "subject": ticket.subject, "description": ticket.description,
                "status": ticket.status, "priority": ticket.priority, "order_id": ticket.order_id}

    @app.post("/api/tickets", dependencies=[Depends(require_api_key)], status_code=202)
    async def create_ticket(req: TicketCreateRequest, session: Session = Depends(get_session)):
        """Direct ticket creation always goes through the approval gate."""
        from .tools import TOOLS
        result = TOOLS["ticket_create"].draft(session, req.model_dump())
        return {"approval_id": result.data["approval_id"], "status": "pending",
                "message": result.message, "payload": result.data["payload"]}

    # ── Approvals (human-in-the-loop) ────────────────────────────────────
    @app.get("/api/approvals", dependencies=[Depends(require_api_key)])
    async def list_approvals(status: str | None = None, session: Session = Depends(get_session)):
        q = session.query(Approval).order_by(Approval.created_at.desc())
        if status:
            q = q.filter(Approval.status == status)
        return [{"id": a.id, "action": a.action, "status": a.status,
                 "payload": json.loads(a.payload_json), "reason": a.reason,
                 "created_at": a.created_at.isoformat() if a.created_at else None,
                 "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None}
                for a in q.limit(100).all()]

    @app.get("/api/approvals/{approval_id}", dependencies=[Depends(require_api_key)])
    async def get_approval(approval_id: str, session: Session = Depends(get_session)):
        ap = session.get(Approval, approval_id.upper())
        if ap is None:
            raise APIError(APPROVAL_NOT_FOUND[0], f"Approval {approval_id} not found.",
                           APPROVAL_NOT_FOUND[1])
        return {"id": ap.id, "action": ap.action, "status": ap.status,
                "payload": json.loads(ap.payload_json), "reason": ap.reason}

    @app.post("/api/approvals/{approval_id}/approve", dependencies=[Depends(require_api_key)])
    async def approve(approval_id: str, req: ApproveRequest, session: Session = Depends(get_session)):
        return resolve_approval(session, approval_id.upper(), True, req.reason)

    @app.post("/api/approvals/{approval_id}/reject", dependencies=[Depends(require_api_key)])
    async def reject(approval_id: str, req: ApproveRequest, session: Session = Depends(get_session)):
        return resolve_approval(session, approval_id.upper(), False, req.reason)

    # ── Knowledge base ───────────────────────────────────────────────────
    @app.get("/api/kb/search", dependencies=[Depends(require_api_key)])
    async def kb_search(q: str, top_k: int = 3):
        if not q or len(q) > 500:
            raise APIError("VALIDATION_ERROR", "Query 'q' must be 1-500 characters.", 422)
        return {"results": _retriever.search(q, top_k=min(top_k, 10)),
                "cache": _retrieval_cache.stats()}

    @app.get("/api/cache/stats", dependencies=[Depends(require_api_key)])
    async def cache_stats():
        return {"embeddings": _embedding_cache.stats(), "retrieval": _retrieval_cache.stats()}

    # ── Serve built frontend (single-URL deployment) ─────────────────────
    dist = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.isdir(dist):
        app.mount("/", StaticFiles(directory=dist, html=True), name="static")

    return app


app = create_app()
