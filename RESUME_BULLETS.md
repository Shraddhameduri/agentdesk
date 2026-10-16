# Resume bullets — Agentic Support Copilot

All metrics below were measured from this build (pytest + `evals/run_evals.py` + `scripts/bench.py`). Nothing is estimated.

- Built an agentic support copilot (FastAPI, React/TypeScript, PostgreSQL/SQLite) where the agent **acts via tools** — order lookup, ticket status, knowledge-base search — instead of only chatting; deployed live with the UI and API on one URL.
- Implemented a **human-in-the-loop approval gate**: write actions (ticket creation) are staged as pending approvals and execute only after explicit approve; verified by tests — 6/6 write intents gated, zero tickets created pre-approval.
- Added a RAG pipeline (document chunking, embeddings, cosine retrieval with caching) over a seeded support knowledge base; answers carry `[DOC-ID]` citations. Offline eval suite of 24 golden cases: **100% tool-call accuracy, 100% approval-gate compliance, 100% groundedness**.
- Hardened for production: standardized error envelope (`code`, `message`, `request_id`) on all failures, LLM provider retry with exponential backoff plus graceful fallback (503, never a leaked stack trace), request-ID structured logging, `/health` + `/ready` endpoints; **38/38 pytest tests pass**.
- Optimized latency: async endpoints, pooled DB connections, two-level TTL caching for embeddings and retrieval; measured **p50 21.7 ms / p95 49.2 ms** on chat turns, KB search **15.3 ms cold → 4.6 ms warm** (~3.3×) over 60 benchmarked requests.
