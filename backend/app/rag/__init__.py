"""RAG knowledge base: seeded support docs, chunking, in-memory retrieval.

Retrieval is cosine similarity over cached embeddings. With Postgres +
pgvector this layer is swapped for server-side vector search (see README
trade-offs); the interface is identical so the agent code does not change.
"""
from __future__ import annotations

import math

DOCS = [
    {
        "id": "RET-001",
        "title": "Returns Policy",
        "text": (
            "Our returns policy: you can return most items within 30 days of delivery for a full refund. "
            "Items must be unused and in original packaging. To start a return, open a support ticket and "
            "our team will email you a prepaid shipping label. Refunds are issued to the original payment "
            "method within 5 to 7 business days after we receive the item. Final-sale items marked at "
            "checkout cannot be returned."
        ),
    },
    {
        "id": "SHP-001",
        "title": "Shipping & Delivery",
        "text": (
            "Orders placed before 2 PM ET ship the same business day. Standard shipping takes 3 to 5 business "
            "days; express shipping takes 1 to 2 business days. Every order includes a tracking number that "
            "you can check with the order lookup tool using your order ID (format ORD-1234). If a package "
            "shows delivered but you cannot find it, wait 24 hours, then contact support to open a ticket."
        ),
    },
    {
        "id": "REF-001",
        "title": "Refunds",
        "text": (
            "Refunds are processed after the returned item arrives at our warehouse. The amount is credited "
            "to the original payment method within 5 to 7 business days. If you paid with a gift card, the "
            "refund is issued as store credit. To request a refund, first open a support ticket; refunds over "
            "$100 require a human agent's approval before they are issued."
        ),
    },
    {
        "id": "WAR-001",
        "title": "Warranty & Repairs",
        "text": (
            "All electronics include a 1-year limited warranty covering manufacturing defects. The warranty "
            "does not cover accidental damage, water damage, or wear and tear. To make a warranty claim, "
            "open a support ticket with your order ID and a description of the defect. Approved claims are "
            "repaired or replaced within 10 business days."
        ),
    },
    {
        "id": "ACC-001",
        "title": "Account & Orders",
        "text": (
            "You can view your orders and their statuses at any time. Order IDs look like ORD-1234. Statuses "
            "are: in_transit, shipped, delivered, or cancelled. If an order shows cancelled and you did not "
            "cancel it, contact support immediately. Keep your order confirmation email — it contains your "
            "order ID and tracking number."
        ),
    },
    {
        "id": "TRB-001",
        "title": "Troubleshooting Common Issues",
        "text": (
            "Device won't turn on? Charge it for at least 30 minutes with the included cable, then hold the "
            "power button for 10 seconds. Connection issues? Restart both the device and your router. If the "
            "problem persists after these steps, open a support ticket describing what you tried and include "
            "your order ID so we can check warranty coverage."
        ),
    },
]


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Sliding-window character chunker over word boundaries."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # overlap: keep the tail of the chunk
            tail: list[str] = []
            tail_len = 0
            for w in reversed(current):
                tail.append(w)
                tail_len += len(w) + 1
                if tail_len >= overlap:
                    break
            current = list(reversed(tail))
            current_len = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Retriever:
    """In-memory RAG retriever: pre-embeds doc chunks once, ranks by cosine sim."""

    def __init__(self, embeddings, cache=None):
        self.embeddings = embeddings
        self.cache = cache  # TTLCache for retrieval results
        self._index: list[dict] = []
        self._build_index()

    def _build_index(self) -> None:
        for doc in DOCS:
            for i, chunk in enumerate(chunk_text(doc["text"])):
                vec = self.embeddings.embed(chunk)
                self._index.append({
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "chunk_id": f"{doc['id']}#c{i}",
                    "text": chunk,
                    "vector": vec,
                })

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        cache_key = f"rag:{self.embeddings.name}:{query}:{top_k}"
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached  # type: ignore[return-value]
        qvec = self.embeddings.embed(query)
        ranked = sorted(
            self._index,
            key=lambda item: cosine_similarity(qvec, item["vector"]),
            reverse=True,
        )[:top_k]
        results = [
            {"doc_id": r["doc_id"], "title": r["title"], "chunk_id": r["chunk_id"],
             "text": r["text"], "score": round(cosine_similarity(qvec, r["vector"]), 4)}
            for r in ranked
        ]
        if self.cache is not None:
            self.cache.set(cache_key, results)
        return results

    def chunk_count(self) -> int:
        return len(self._index)
