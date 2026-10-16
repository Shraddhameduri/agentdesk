"""Benchmark: p50/p95 latency of the local API with the fake provider.

Starts uvicorn on a temp SQLite DB, warms up, then runs N chat requests and
N repeated KB searches (to demonstrate the retrieval-cache effect). Prints
real measured numbers as JSON. No API keys needed.

Usage: .venv/bin/python scripts/bench.py [--requests 60]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def wait_for(port: int, timeout: int = 30) -> bool:
    # trust_env=False: the sandbox egress proxy var has no scheme and makes
    # httpx raise InvalidURL for localhost URLs.
    deadline = time.time() + timeout
    with httpx.Client(trust_env=False, timeout=2) as probe:
        while time.time() < deadline:
            try:
                r = probe.get(f"http://127.0.0.1:{port}/health")
                if r.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="copilot-bench-")
    env = dict(os.environ, API_KEY="bench-key", LLM_PROVIDER="fake",
               DATABASE_URL=f"sqlite:///{tmp}/bench.db", LOG_LEVEL="WARNING")
    proc = subprocess.Popen(
        [VENV_PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(args.port)],
        cwd=os.path.join(ROOT, "backend"), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for(args.port):
            print(json.dumps({"error": "server did not start"}))
            sys.exit(1)
        headers = {"X-API-Key": "bench-key"}
        base = f"http://127.0.0.1:{args.port}"

        prompts = [
            "Where is my order ORD-1002?",
            "What is your returns policy?",
            "Track order ORD-1001 please",
            "How long does shipping take?",
            "What is the status of ticket TKT-2001?",
        ]
        with httpx.Client(timeout=30, trust_env=False) as client:
            # warmup (also populates caches)
            for p in prompts:
                client.post(f"{base}/api/chat", headers=headers, json={"message": p})

            chat_lat, chat_cold, chat_warm = [], [], []
            for i in range(args.requests):
                p = prompts[i % len(prompts)]
                t0 = time.perf_counter()
                r = client.post(f"{base}/api/chat", headers=headers, json={"message": p})
                chat_lat.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200

            # cache effect: first (cold) vs repeated (warm) identical KB query
            q = "warranty coverage for electronics manufacturing defects"
            t0 = time.perf_counter()
            client.get(f"{base}/api/kb/search", headers=headers, params={"q": q})
            chat_cold.append((time.perf_counter() - t0) * 1000)
            for _ in range(10):
                t0 = time.perf_counter()
                client.get(f"{base}/api/kb/search", headers=headers, params={"q": q})
                chat_warm.append((time.perf_counter() - t0) * 1000)

            cache = client.get(f"{base}/api/cache/stats", headers=headers).json()

        result = {
            "requests": args.requests,
            "chat_latency_ms": {
                "p50": round(percentile(chat_lat, 50), 2),
                "p95": round(percentile(chat_lat, 95), 2),
                "mean": round(sum(chat_lat) / len(chat_lat), 2),
                "max": round(max(chat_lat), 2),
            },
            "kb_search_cold_ms": round(chat_cold[0], 2),
            "kb_search_warm_p50_ms": round(percentile(chat_warm, 50), 2),
            "cache_stats": cache,
            "provider": "fake",
        }
        print(json.dumps(result, indent=2))
        with open(os.path.join(ROOT, "evals", "benchmark.json"), "w") as f:
            json.dump(result, f, indent=2)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
