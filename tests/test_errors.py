"""Error-handling tests: provider failure -> retry/backoff -> fallback -> envelope."""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest  # noqa: E402

import app.main as main_mod  # noqa: E402
from app.agent import plan_with_retry  # noqa: E402
from app.errors import APIError  # noqa: E402
from app.providers import BaseLLMProvider, FakeLLMProvider, Plan  # noqa: E402


class ExplodingProvider(BaseLLMProvider):
    name = "exploding"

    async def plan(self, message, history):
        raise RuntimeError("simulated provider outage")

    async def compose(self, message, history, tool_name, tool_result, kb_snippets):
        raise RuntimeError("simulated provider outage")


class ExplodingFallback(FakeLLMProvider):
    name = "exploding-fallback"

    async def plan(self, message, history):
        raise RuntimeError("fallback also down")


def test_provider_failure_falls_back_to_fake():
    plan, used = asyncio.run(plan_with_retry(
        ExplodingProvider(), FakeLLMProvider(), "Where is my order ORD-1001?", [],
        max_retries=2, backoff_base=0.01))
    assert used == "fake"
    assert plan.tool == "order_lookup"


def test_total_provider_failure_returns_503_envelope():
    with pytest.raises(APIError) as ei:
        asyncio.run(plan_with_retry(
            ExplodingProvider(), ExplodingFallback(), "hi", [],
            max_retries=1, backoff_base=0.01))
    assert ei.value.code == "PROVIDER_UNAVAILABLE"
    assert ei.value.status_code == 503


def test_no_unhandled_500_stack_trace_leaks():
    # Exercise the global uncaught-exception handler directly: an unexpected
    # exception must become the standard envelope with no internals leaking.
    import asyncio
    import json

    from fastapi import Request

    from app.errors import unhandled_error_handler
    from app.logging_setup import set_request_id

    set_request_id("test-req-999")
    scope = {"type": "http", "method": "GET", "path": "/__boom__", "headers": [],
             "query_string": b"", "server": ("test", 80), "scheme": "http"}
    resp = asyncio.run(unhandled_error_handler(Request(scope), RuntimeError("kaboom")))
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "kaboom" not in body["error"]["message"]  # no internals leak
    assert body["error"]["request_id"] == "test-req-999"
