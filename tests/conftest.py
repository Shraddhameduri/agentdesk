"""Shared fixtures: FastAPI TestClient over an in-memory SQLite DB + fake provider."""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["API_KEY"] = "test-key"

from app.main import create_app  # noqa: E402

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture(scope="session")
def client():
    app = create_app(database_url="sqlite:///:memory:", provider_name="fake")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def api(client):
    """Fresh DB per test: drop + recreate via a new app would be slow;
    instead truncate mutable tables between tests."""
    import app.main as main_mod

    with main_mod._SessionLocal() as s:
        from app.db import Approval, ChatMessage, Ticket
        s.query(Approval).delete()
        s.query(Ticket).filter(Ticket.id.like("TKT-20%")).delete()
        # re-seed the 3 stock tickets
        from app.seed import TICKETS
        for row in TICKETS:
            if s.get(Ticket, row["id"]) is None:
                s.add(Ticket(**row))
        s.query(ChatMessage).delete()
        s.commit()
    return client
