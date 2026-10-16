"""Embeddings abstraction.

FakeEmbeddings is a deterministic hash-based embedding (128 dims) used for
tests, evals, and the demo. OpenAIEmbeddings is the production path.
An LRU-style TTL cache wraps embed() so repeated KB queries don't re-embed.
"""
from __future__ import annotations

import hashlib
import math
import time


class TTLCache:
    def __init__(self, ttl_seconds: int = 300, max_items: int = 2048):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: dict[str, tuple[float, object]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires, value = entry
        if time.time() > expires:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: object) -> None:
        if len(self._store) >= self.max_items:
            # evict an arbitrary (oldest-inserted) item
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]
        self._store[key] = (time.time() + self.ttl, value)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._store)}

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


class BaseEmbeddings:
    name = "base"
    dims = 0

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def cache_key(self, text: str) -> str:
        return f"{self.name}:{hashlib.sha256(text.encode()).hexdigest()[:32]}"


STOPWORDS = frozenset("""
a an the and or but if then else when where which who whom whose what how why is are was were be been
being have has had do does did will would shall should may might must can could of in on at to for
with by from as it its this that these those i you he she we they them his her our their your my me
us him not no yes so such than too very just about into over after before between through during
there here all any both each few more most other some only own same d there s t re ve ll don isn
aren wasn weren haven hasn hadn doesn didn won wouldn couldn shouldn your you re ll ve
""".split())

class FakeEmbeddings(BaseEmbeddings):
    """Deterministic 128-dim hash embeddings.

    Each token hashes to several buckets; cosine similarity then correlates
    with lexical overlap, which is exactly what we want for offline RAG tests.
    Stopwords are dropped so content words dominate the similarity.
    """

    name = "fake"
    dims = 128
    _BUCKETS_PER_TOKEN = 3

    def embed(self, text: str) -> list[float]:
        import re

        vec = [0.0] * self.dims
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in STOPWORDS:
                continue
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            for i in range(self._BUCKETS_PER_TOKEN):
                vec[(h >> (i * 11)) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbeddings(BaseEmbeddings):
    name = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1", timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dims = 1536

    def embed(self, text: str) -> list[float]:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text[:8000]},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


class CachedEmbeddings:
    """Wraps any embeddings with a TTL cache. Cache stats are observable."""

    def __init__(self, inner: BaseEmbeddings, cache: TTLCache):
        self.inner = inner
        self.cache = cache

    def embed(self, text: str) -> list[float]:
        key = self.inner.cache_key(text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        vec = self.inner.embed(text)
        self.cache.set(key, vec)
        return vec

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dims(self) -> int:
        return self.inner.dims
