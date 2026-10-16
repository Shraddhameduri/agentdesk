"""Tool unit tests: read tools execute, write tools stage approvals."""
from __future__ import annotations

import pytest

from app.db import Approval, Ticket
from app.errors import APIError
from app.tools import TOOLS


def _session(api):
    import app.main as main_mod

    return main_mod._SessionLocal()


def test_order_lookup_found(api):
    s = _session(api)
    r = TOOLS["order_lookup"].execute(s, {"order_id": "ORD-1001"})
    assert r.ok and r.data["found"] and r.data["status"] == "shipped"
    s.close()


def test_order_lookup_not_found_raises_api_error(api):
    s = _session(api)
    with pytest.raises(APIError) as ei:
        TOOLS["order_lookup"].execute(s, {"order_id": "ORD-9999"})
    assert ei.value.code == "ORDER_NOT_FOUND"
    s.close()


def test_ticket_status_found(api):
    s = _session(api)
    r = TOOLS["ticket_status"].execute(s, {"ticket_id": "TKT-2001"})
    assert r.ok and r.data["status"] == "in_progress"
    s.close()


def test_ticket_status_not_found(api):
    s = _session(api)
    with pytest.raises(APIError) as ei:
        TOOLS["ticket_status"].execute(s, {"ticket_id": "TKT-9999"})
    assert ei.value.code == "TICKET_NOT_FOUND"
    s.close()


def test_ticket_create_draft_creates_approval_not_ticket(api):
    s = _session(api)
    before = s.query(Ticket).count()
    r = TOOLS["ticket_create"].draft(s, {"subject": "x", "description": "y", "priority": "normal"})
    assert r.awaiting_approval
    assert s.query(Ticket).count() == before  # nothing created
    ap = s.get(Approval, r.data["approval_id"])
    assert ap is not None and ap.status == "pending"
    s.close()


def test_kb_search_returns_snippets(api):
    s = _session(api)
    r = TOOLS["kb_search"].execute(s, {"query": "returns policy"})
    assert r.ok and len(r.data["snippets"]) >= 1
    assert r.data["snippets"][0]["doc_id"] == "RET-001"
    s.close()
