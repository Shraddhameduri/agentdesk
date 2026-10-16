"""Regression tests for issues Shraddha found on the live demo (2026-10-15).

1. "where is my order" (no order ID) dumped KB docs — must ask for the ID.
2. "ord 234?" fell through to KB — loose IDs must route to order_lookup.
3. "can i return my order no ord5678" dumped KB — return+ID must look up the
   order (with a returns note when found), not dump KB docs.
4. KB compose() stitched two unrelated chunks with mid-sentence truncation —
   must end at sentence boundaries and prefer a single on-topic snippet.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.providers import (  # noqa: E402
    FakeLLMProvider,
    normalize_order_id,
    truncate_at_sentence,
)


def plan_for(message: str):
    return asyncio.run(FakeLLMProvider().plan(message, []))


def compose_for(message, tool_name, tool_result, kb_snippets=None):
    return asyncio.run(
        FakeLLMProvider().compose(message, [], tool_name, tool_result, kb_snippets or [])
    )


# ── 1. "where is my order" (no ID) -> clarification, not KB ────────────────

def test_where_is_my_order_no_id_asks_for_id():
    p = plan_for("where is my order")
    assert p.tool is None and not p.routed
    draft = p.reply_draft.lower()
    assert "order id" in draft and "ord-1234" in draft
    assert "kb" not in p.tool if p.tool else True


def test_where_is_my_order_no_id_never_kb_search():
    for msg in ("where is my order", "where's my package", "Where is my order?"):
        p = plan_for(msg)
        assert p.tool != "kb_search", msg


def test_where_is_my_order_with_id_still_looks_up():
    p = plan_for("where is my order ORD-1002")
    assert p.tool == "order_lookup" and p.args["order_id"] == "ORD-1002"


def test_how_to_track_package_still_kb_search():
    # A how-to question stays a policy question (matches golden eval).
    p = plan_for("How do I track my package?")
    assert p.tool == "kb_search"


# ── 2. Loose order IDs normalize and route to order_lookup ─────────────────

def test_loose_order_id_forms_normalize():
    assert normalize_order_id("ord 234?") == "ORD-0234"
    assert normalize_order_id("ord234") == "ORD-0234"
    assert normalize_order_id("ORD-5678") == "ORD-5678"
    assert normalize_order_id("ord5678") == "ORD-5678"
    assert normalize_order_id("no order id here") is None


def test_loose_id_ord_234_routes_to_lookup():
    p = plan_for("ord 234?")
    assert p.tool == "order_lookup"
    assert p.args["order_id"] == "ORD-0234"


def test_loose_id_ord5678_routes_to_lookup():
    p = plan_for("ord5678")
    assert p.tool == "order_lookup"
    assert p.args["order_id"] == "ORD-5678"


def test_loose_id_ord_dash_1002_routes_to_lookup():
    p = plan_for("ord-1002")
    assert p.tool == "order_lookup"
    assert p.args["order_id"] == "ORD-1002"


def test_ticket_create_still_extracts_loose_id():
    p = plan_for("I need to file a ticket about ORD-1003")
    assert p.tool == "ticket_create" and p.needs_approval
    assert p.args["order_id"] == "ORD-1003"


# ── 3. Return intent + order ID -> lookup (+ returns note when found) ────────

def test_return_with_unknown_order_id_looks_up_not_kb():
    p = plan_for("can i return my order no ord5678")
    assert p.tool == "order_lookup"
    assert p.args["order_id"] == "ORD-5678"


def test_return_with_found_order_adds_returns_note():
    reply = compose_for(
        "can i return my order no ord1002",
        "order_lookup",
        {"found": True, "id": "ORD-1002", "item": "Mechanical Keyboard (TKL)",
         "total_cents": 12999, "status": "in_transit", "tracking_number": "1Z8842WQ1002"},
    )
    assert "ORD-1002" in reply
    assert "30 days" in reply and "[RET-001]" in reply


def test_return_with_unknown_order_gives_not_found_with_hint():
    reply = compose_for(
        "can i return my order no ord5678",
        "order_lookup",
        {"found": False, "id": "ORD-5678"},
    )
    assert "ORD-5678" in reply
    assert "ORD-1001" in reply  # demo ID range hint


def test_unknown_id_not_found_hints_demo_range():
    reply = compose_for(
        "ord 234?",
        "order_lookup",
        {"found": False, "id": "ORD-0234"},
    )
    assert "couldn't find" in reply.lower()
    assert "ORD-1001" in reply


def test_return_without_id_stays_policy_kb_search():
    # A bare return question is still a policy question (correct behavior).
    p = plan_for("How do I return an item?")
    assert p.tool == "kb_search"


# ── 4. KB compose: sentence-boundary truncation, single best snippet ─────────

def test_truncate_at_sentence_no_mid_word_cut():
    long_text = "x" * 100 + ". " + "Second sentence keeps going and going. " * 20
    out = truncate_at_sentence(long_text, 120)
    assert len(out) <= 120
    assert out.endswith(".")
    # Sentence boundary won even though the first sentence is very short
    assert out == "x" * 100 + "."


def test_truncate_prefers_sentence_boundary_over_word_break():
    text = ("Our returns policy: you can return most items within 30 days of delivery "
            "for a full refund. Items must be unused and in original packaging.")
    out = truncate_at_sentence(text, 100)
    assert out.endswith(".")
    assert "unused" not in out or out.endswith("packaging.")


def test_compose_prefers_single_snippet_on_different_topics():
    snippets = [
        {"doc_id": "SHP-001", "text": "Orders placed before 2 PM ET ship the same business day. Standard shipping takes 3 to 5 business days.", "score": 0.95},
        {"doc_id": "RET-001", "text": "Our returns policy: you can return most items within 30 days of delivery for a full refund.", "score": 0.60},
    ]
    reply = compose_for("How long does shipping take?", "kb_search", None, snippets)
    assert "[SHP-001]" in reply
    assert "[RET-001]" not in reply
    # No mid-sentence truncation: body sentences end with periods
    assert reply.rstrip().endswith(".")


def test_compose_keeps_citations_and_footer_format():
    snippets = [
        {"doc_id": "ACC-001", "text": "You can view your orders and their statuses at any time. Order IDs look like ORD-1234.", "score": 0.9},
        {"doc_id": "ACC-001", "text": "Keep your order confirmation email — it contains your order ID and tracking number.", "score": 0.85},
    ]
    reply = compose_for("What do order IDs look like?", "kb_search", None, snippets)
    assert "[ACC-001]" in reply
    assert "Based on our support docs" in reply
