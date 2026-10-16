"""Approval flow: draft -> pending -> approve (executes) / reject (discards)."""
from __future__ import annotations

from .conftest import HEADERS


def _draft_ticket(api):
    r = api.post("/api/tickets", headers=HEADERS, json={
        "subject": "Broken item", "description": "Arrived broken", "priority": "high"})
    assert r.status_code == 202
    return r.json()["approval_id"]


def test_direct_ticket_create_returns_pending_approval(api):
    approval_id = _draft_ticket(api)
    r = api.get(f"/api/approvals/{approval_id}", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "pending"


def test_approve_executes_ticket_creation(api):
    approval_id = _draft_ticket(api)
    r = api.post(f"/api/approvals/{approval_id}/approve", headers=HEADERS, json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved"
    assert body["executed"]["id"].startswith("TKT-")
    # ticket is retrievable
    t = api.get(f"/api/tickets/{body['executed']['id']}", headers=HEADERS)
    assert t.status_code == 200 and t.json()["subject"] == "Broken item"


def test_reject_discards_without_creating(api):
    approval_id = _draft_ticket(api)
    r = api.post(f"/api/approvals/{approval_id}/reject", headers=HEADERS,
                 json={"reason": "duplicate"})
    assert r.status_code == 200
    assert r.json()["status"] == "rejected" and r.json()["executed"] is None


def test_double_resolve_is_conflict(api):
    approval_id = _draft_ticket(api)
    api.post(f"/api/approvals/{approval_id}/approve", headers=HEADERS, json={})
    r = api.post(f"/api/approvals/{approval_id}/approve", headers=HEADERS, json={})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "APPROVAL_ALREADY_RESOLVED"


def test_approval_not_found_envelope(api):
    r = api.get("/api/approvals/APR-9999", headers=HEADERS)
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "APPROVAL_NOT_FOUND"
    assert "request_id" in body["error"]


def test_chat_write_intent_stages_approval(api):
    r = api.post("/api/chat", headers=HEADERS, json={"message": "Open a ticket, my order never arrived"})
    assert r.status_code == 200
    body = r.json()
    assert body["approval"] is not None and body["approval"]["status"] == "pending"
    assert "nothing has been created" in body["reply"].lower() or "approval" in body["reply"].lower()
