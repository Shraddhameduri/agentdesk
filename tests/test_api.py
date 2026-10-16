"""API tests: chat, streaming, auth, validation, error envelopes, health."""
from __future__ import annotations

from .conftest import HEADERS


def test_chat_order_lookup(api):
    r = api.post("/api/chat", headers=HEADERS, json={"message": "Where is my order ORD-1002?"})
    assert r.status_code == 200
    body = r.json()
    assert "ORD-1002" in body["reply"]
    assert body["tool_calls"][0]["tool"] == "order_lookup"
    assert body["provider"] == "fake"
    assert body["request_id"]


def test_chat_kb_with_sources(api):
    r = api.post("/api/chat", headers=HEADERS, json={"message": "What is your returns policy?"})
    assert r.status_code == 200
    body = r.json()
    assert any(s["doc_id"] == "RET-001" for s in body["sources"])
    assert "[RET-001]" in body["reply"]


def test_chat_stream_sse(api):
    with api.stream("POST", "/api/chat/stream", headers=HEADERS,
                    json={"message": "Where is my order ORD-1001?"}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = [line for line in r.iter_lines() if line.startswith("event:")]
    kinds = [e.split("event: ")[1] for e in events]
    assert "meta" in kinds and "done" in kinds and "token" in kinds


def test_auth_missing_key_rejected(api):
    r = api.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert "request_id" in body["error"]


def test_auth_wrong_key_rejected(api):
    r = api.post("/api/chat", headers={"X-API-Key": "wrong"}, json={"message": "hi"})
    assert r.status_code == 401


def test_validation_error_envelope(api):
    r = api.post("/api/chat", headers=HEADERS, json={"wrong_field": 1})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_order_not_found_envelope(api):
    r = api.get("/api/orders/ORD-9999", headers=HEADERS)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "ORDER_NOT_FOUND"


def test_health_and_ready_no_auth(api):
    assert api.get("/health").json()["status"] == "ok"
    ready = api.get("/ready").json()
    assert ready["ready"] is True and ready["provider"] == "fake"


def test_request_id_header_echoed(api):
    r = api.post("/api/chat", headers={**HEADERS, "X-Request-ID": "test-req-123"},
                 json={"message": "hi"})
    assert r.headers.get("X-Request-ID") == "test-req-123"


def test_kb_search_endpoint(api):
    r = api.get("/api/kb/search", headers=HEADERS, params={"q": "warranty"})
    assert r.status_code == 200
    assert len(r.json()["results"]) > 0
