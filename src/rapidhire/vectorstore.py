"""Vector store: embeddings (sentence-transformers) + ANN storage (ChromaDB).

This is deliberately plain Python. Agents reach it only through the tools in
``rapidhire.tools``; nothing here knows about CrewAI or LangChain.

Two things worth knowing before you touch this file:

* ``PersistentClient`` is single-writer. It's SQLite underneath and SQLite
  serializes writes, so two processes pointing at the same ``chroma_path`` will
  eventually hit "database is locked". For anything multi-process (Docker,
  multiple workers) run a Chroma server and use ``chroma_mode="http"``.
* A collection's embedding dimension is fixed on the first write. Switching
  ``embedding_model`` from MiniLM (384) to mpnet (768) against an existing
  collection raises ``InvalidDimensionException``. Changing models means deleting
  the persist dir (or bumping the collection name) and re-indexing.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, get_settings
from .logging import get_logger
from .models import EvidenceChunk

log = get_logger(__name__)

CRITERIA_COLLECTION = "hiring_criteria"
CANDIDATE_COLLECTION = "candidate_pool"

# MiniLM truncates silently past 256 word-pieces. Word count is a rough proxy for
# word-piece count (English averages ~1.3 pieces/word), so ~180 words keeps us
# comfortably under the limit for typical prose without a tokenizer round-trip.
_MAX_WORDS = 180
_OVERLAP = 30


def chunk_text(text: str, max_words: int = _MAX_WORDS, overlap: int = _OVERLAP) -> list[str]:
    """Split text into overlapping word windows.

    Overlap keeps a sentence that straddles a boundary from being cut in half in
    both chunks. Returns a single-element list for short inputs.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []
    step = max_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if window:
            chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return chunks


# One model instance per process. Loading MiniLM is ~1-2s and a few hundred MB of
# torch; doing it per request would dominate the "sub-100ms" match budget.
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def load_model(name: str) -> SentenceTransformer:
    if name not in _MODEL_CACHE:
        log.info("loading embedding model %s", name)
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


class VectorStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model = load_model(self.settings.embedding_model)
        self._client = self._make_client()
        # embedding_function=None: we always hand Chroma pre-computed vectors so the
        # single cached model above is the only thing that ever embeds anything.
        self.criteria = self._client.get_or_create_collection(
            name=CRITERIA_COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        self.candidates = self._client.get_or_create_collection(
            name=CANDIDATE_COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def _make_client(self):
        if self.settings.chroma_mode == "http":
            return chromadb.HttpClient(
                host=self.settings.chroma_host, port=self.settings.chroma_port
            )
        Path(self.settings.chroma_path).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=self.settings.chroma_path)

    # --- embedding ----------------------------------------------------------
    def embed(self, texts: list[str]) -> np.ndarray:
        # normalize so a dot product is cosine similarity for the direct JD<->profile
        # comparison in matching. Chroma computes its own cosine distance regardless.
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two documents, each chunked then mean-pooled."""
        va = self._pooled(a)
        vb = self._pooled(b)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb))

    def _pooled(self, text: str) -> np.ndarray | None:
        chunks = chunk_text(text)
        if not chunks:
            return None
        vecs = self.embed(chunks)
        pooled = vecs.mean(axis=0)
        norm = np.linalg.norm(pooled)
        return pooled / norm if norm else pooled

    # --- writes -------------------------------------------------------------
    def index_criteria(self, docs: dict[str, str]) -> int:
        """Index role/criteria documents. ``docs`` maps a source name to its text.

        Chunks are stored with a ``source`` metadata field so retrieved evidence can
        be traced back to the document it came from. Re-indexing the same source
        overwrites (ids are deterministic), so a re-run doesn't duplicate.
        """
        ids, chunks, metas = [], [], []
        for source, text in docs.items():
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{source}::{i}")
                chunks.append(chunk)
                metas.append({"source": source})
        if not chunks:
            return 0
        self.criteria.upsert(
            ids=ids, documents=chunks, metadatas=metas, embeddings=self.embed(chunks).tolist()
        )
        log.info("indexed %d criteria chunks from %d docs", len(chunks), len(docs))
        return len(chunks)

    def add_candidates(self, profiles: dict[str, str]) -> int:
        """Add candidate profiles to the pool so future matches can find near-peers."""
        ids = list(profiles.keys())
        docs = list(profiles.values())
        if not docs:
            return 0
        self.candidates.upsert(
            ids=ids,
            documents=docs,
            metadatas=[{"source": i} for i in ids],
            embeddings=self.embed(docs).tolist(),
        )
        return len(docs)

    # --- reads --------------------------------------------------------------
    def query(self, text: str, k: int | None = None, *, pool: bool = False) -> list[EvidenceChunk]:
        """Return the k nearest chunks to ``text`` as evidence, cosine-filtered.

        Set ``pool=True`` to search the candidate pool instead of the criteria docs.
        Chunks below ``relevance_threshold`` are dropped so a query with no good
        match returns an empty list rather than noise.
        """
        collection = self.candidates if pool else self.criteria
        if collection.count() == 0:
            return []
        k = k or self.settings.top_k
        res = collection.query(
            query_embeddings=self.embed([text]).tolist(),
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        out: list[EvidenceChunk] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], strict=False
        ):
            # cosine space -> distance is (1 - cosine similarity); invert it back.
            score = 1.0 - float(dist)
            if score < self.settings.relevance_threshold:
                continue
            out.append(EvidenceChunk(text=doc, source=str(meta.get("source", "?")), score=score))
        return out

    # --- seeding ------------------------------------------------------------
    def seed_criteria_if_empty(self) -> int:
        """Load ``criteria_dir`` into the store the first time we run.

        Idempotent by construction: once the collection has chunks we do nothing,
        and even a forced re-seed upserts by deterministic id.
        """
        if self.criteria.count() > 0:
            return 0
        directory = Path(self.settings.criteria_dir)
        if not directory.exists():
            log.warning("criteria dir %s does not exist; nothing to seed", directory)
            return 0
        docs = {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(directory.glob("*"))
            if p.suffix.lower() in {".md", ".txt"}
        }
        return self.index_criteria(docs)


# A single VectorStore per process. The tools and the Streamlit app all go through
# this so the embedding model and Chroma client are opened exactly once, and the
# criteria seed runs on the first access rather than on every tool call.
_STORE: VectorStore | None = None


def get_vectorstore() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
        _STORE.seed_criteria_if_empty()
    return _STORE


def reset_vectorstore() -> None:
    """Drop the cached store. Only useful in tests that swap settings."""
    global _STORE
    _STORE = None
