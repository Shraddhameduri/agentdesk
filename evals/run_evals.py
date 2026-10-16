"""Offline eval suite. Runs the golden dataset through the agent with the
deterministic fake provider and reports:
  - tool-call accuracy: fraction of cases where the chosen tool == expected_tool
  - approval-gate compliance: fraction of write-intent cases that produced a
    pending approval AND created zero tickets before approval
  - groundedness: fraction of kb_search cases whose reply cites the expected
    doc id AND contains the expected key phrase
Usage: PYTHONPATH=backend python evals/run_evals.py [--json-out path]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.agent import run_turn  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import Ticket, get_engine, init_db, session_factory  # noqa: E402
from app.embeddings import CachedEmbeddings, FakeEmbeddings, TTLCache  # noqa: E402
from app.providers import FakeLLMProvider  # noqa: E402
from app.rag import Retriever  # noqa: E402
from app.seed import seed_db  # noqa: E402
from app.tools import register_tools  # noqa: E402


async def run_case(session, provider, retriever, case: dict) -> dict:
    tickets_before = session.query(Ticket).count()
    result = await run_turn(session, provider, provider, retriever,
                            case["message"], [], settings)
    tickets_after = session.query(Ticket).count()

    actual_tool = result["tool_calls"][0]["tool"] if result["tool_calls"] else None
    tool_ok = actual_tool == case["expected_tool"]

    approval_ok = True
    if case["expect_approval"]:
        approval_ok = (
            result["approval"] is not None
            and result["approval"]["status"] == "pending"
            and tickets_after == tickets_before  # nothing created before approval
        )

    grounded_ok = True
    if case["expected_tool"] == "kb_search":
        reply = result["reply"].lower()
        grounded_ok = all(phrase.lower() in reply for phrase in case["must_mention"])

    return {
        "id": case["id"],
        "expected_tool": case["expected_tool"],
        "actual_tool": actual_tool,
        "tool_ok": tool_ok,
        "approval_ok": approval_ok,
        "grounded_ok": grounded_ok,
        "reply_preview": result["reply"][:120],
    }


async def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    SessionLocal = session_factory(engine)
    session = SessionLocal()
    seed_db(session)

    provider = FakeLLMProvider()
    retriever = Retriever(CachedEmbeddings(FakeEmbeddings(), TTLCache()))
    register_tools(retriever)

    with open(os.path.join(os.path.dirname(__file__), "golden.json")) as f:
        cases = json.load(f)

    results = [await run_case(session, provider, retriever, c) for c in cases]
    n = len(results)
    tool_acc = sum(r["tool_ok"] for r in results) / n
    write_cases = [r for r, c in zip(results, cases) if c["expect_approval"]]
    approval_acc = sum(r["approval_ok"] for r in write_cases) / len(write_cases)
    kb_cases = [r for r, c in zip(results, cases) if c["expected_tool"] == "kb_search"]
    grounded_acc = sum(r["grounded_ok"] for r in kb_cases) / len(kb_cases)

    summary = {
        "n_cases": n,
        "tool_call_accuracy": round(tool_acc, 4),
        "approval_gate_compliance": round(approval_acc, 4),
        "groundedness": round(grounded_acc, 4),
        "failures": [r for r in results if not (r["tool_ok"] and r["approval_ok"] and r["grounded_ok"])],
    }

    print(f"cases:                  {n}")
    print(f"tool-call accuracy:     {summary['tool_call_accuracy']:.1%}  ({sum(r['tool_ok'] for r in results)}/{n})")
    print(f"approval-gate ok:       {summary['approval_gate_compliance']:.1%}  ({sum(r['approval_ok'] for r in write_cases)}/{len(write_cases)})")
    print(f"groundedness:           {summary['groundedness']:.1%}  ({sum(r['grounded_ok'] for r in kb_cases)}/{len(kb_cases)})")
    for r in summary["failures"]:
        print(f"  FAIL {r['id']}: expected={r['expected_tool']} actual={r['actual_tool']} "
              f"tool_ok={r['tool_ok']} approval_ok={r['approval_ok']} grounded_ok={r['grounded_ok']}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    asyncio.run(main())
