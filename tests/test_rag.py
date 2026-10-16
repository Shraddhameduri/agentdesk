"""RAG tests: chunking, retrieval relevance, caching behavior."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import app.main as main_mod  # noqa: E402
from app.rag import chunk_text  # noqa: E402


def test_chunking_produces_overlap():
    chunks = chunk_text("word " * 300, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # overlap: tail of chunk 0 appears at head of chunk 1
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]


def test_retrieval_returns_policy_for_returns_question(api):
    results = main_mod._retriever.search("What is your returns policy?", top_k=3)
    assert results[0]["doc_id"] == "RET-001"
    assert results[0]["score"] > 0


def test_retrieval_returns_warranty_doc(api):
    results = main_mod._retriever.search("electronics warranty manufacturing defects", top_k=3)
    assert results[0]["doc_id"] == "WAR-001"


def test_retrieval_cache_hit_on_repeat(api):
    main_mod._retrieval_cache.clear()
    main_mod._retriever.search("shipping delivery tracking", top_k=3)
    stats_after_first = main_mod._retrieval_cache.stats()
    main_mod._retriever.search("shipping delivery tracking", top_k=3)
    stats_after_second = main_mod._retrieval_cache.stats()
    assert stats_after_first["misses"] == 1
    assert stats_after_second["hits"] == 1  # second identical query served from cache


def test_embedding_cache_hit(api):
    main_mod._embedding_cache.clear()
    main_mod._retriever.embeddings.embed("unique cache probe phrase xyz")
    main_mod._retriever.embeddings.embed("unique cache probe phrase xyz")
    stats = main_mod._embedding_cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
