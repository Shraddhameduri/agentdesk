"""Agent routing: the planner must pick the right tool for each intent."""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.providers import FakeLLMProvider  # noqa: E402


def plan_for(message: str):
    return asyncio.run(FakeLLMProvider().plan(message, []))


def test_order_lookup_routing():
    p = plan_for("Where is my order ORD-1002?")
    assert p.tool == "order_lookup" and p.args["order_id"] == "ORD-1002"


def test_order_tracking_routing():
    p = plan_for("Track package ORD-1001")
    assert p.tool == "order_lookup"


def test_ticket_status_routing():
    p = plan_for("What is the status of ticket TKT-2001?")
    assert p.tool == "ticket_status" and p.args["ticket_id"] == "TKT-2001"


def test_ticket_create_write_intent():
    p = plan_for("I want to open a ticket about my broken headphones")
    assert p.tool == "ticket_create" and p.needs_approval


def test_write_intent_beats_order_id():
    # Safety: even when an order ID is present, a write intent must be gated.
    p = plan_for("I want a refund for order ORD-1005")
    assert p.tool == "ticket_create" and p.needs_approval
    assert p.args["order_id"] == "ORD-1005"


def test_refund_question_is_kb_search():
    p = plan_for("When will I get my refund?")
    assert p.tool == "kb_search" and not p.needs_approval


def test_policy_question_is_kb_search():
    p = plan_for("What is your returns policy?")
    assert p.tool == "kb_search"


def test_greeting_is_general_chat():
    p = plan_for("Thanks for the help!")
    assert p.tool is None and not p.routed
    assert "look up orders" in p.reply_draft


# ── Regression: write-intent detection must not be phrasing-sensitive ──────
# Live-demo bug: "file a ticket" staged an approval, but "create a support
# ticket" fell through to kb_search and "open a support ticket" to order_lookup.


def test_ticket_create_support_ticket_phrasing():
    p = plan_for("Create a support ticket: my order arrived damaged")
    assert p.tool == "ticket_create" and p.needs_approval


def test_ticket_create_open_support_ticket_with_order_id():
    p = plan_for("I want to open a support ticket for my damaged order ORD-1002")
    assert p.tool == "ticket_create" and p.needs_approval
    assert p.args["order_id"] == "ORD-1002"


def test_ticket_create_verb_variants():
    for msg in (
        "Please raise a support ticket about my late delivery",
        "Submit a ticket for my broken charger",
        "I need to file a ticket about ORD-1003",
        "Can you open a ticket for me?",
        "I want a support ticket for my defective monitor",
        "Please start a ticket about the missing keycaps",
    ):
        p = plan_for(msg)
        assert p.tool == "ticket_create" and p.needs_approval, msg


def test_cancel_order_variants_still_write():
    for msg in (
        "Cancel my order ORD-1005",
        "Please cancel this order",
        "I want to cancel order ORD-1003",
    ):
        p = plan_for(msg)
        assert p.tool == "ticket_create" and p.needs_approval, msg


# ── Guards: read intents must NOT stage tickets ────────────────────────────


def test_order_question_does_not_stage_ticket():
    p = plan_for("Where is my order ORD-1002?")
    assert p.tool == "order_lookup" and not p.needs_approval


def test_policy_question_does_not_stage_ticket():
    for msg in (
        "What is your returns policy?",
        "When will I get my refund?",
        "How do I return an item?",
    ):
        p = plan_for(msg)
        assert p.tool == "kb_search" and not p.needs_approval, msg


def test_ticket_status_read_stays_read():
    for msg in (
        "What is the status of ticket TKT-2001?",
        "I want to check the status of my ticket TKT-2003",
        "I need a ticket status update on TKT-2003",
    ):
        p = plan_for(msg)
        assert p.tool == "ticket_status" and not p.needs_approval, msg
