"""Vector-store tests.

These use the real embedding model and a real (temp) Chroma collection, so no API
key is needed: embeddings run locally. The module is skipped if the heavy deps aren't
installed, and the model-backed tests skip individually if the embedding model can't
be fetched (e.g. an offline CI box), so the suite stays green everywhere.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("chromadb")

from rapidhire.vectorstore import VectorStore, chunk_text  # noqa: E402


@pytest.fixture
def store(settings):
    # Building the store loads the embedding model. If that download can't happen
    # (no network to the model host), skip rather than fail; the code path is fine,
    # the environment just can't exercise it.
    try:
        return VectorStore(settings)
    except OSError as exc:
        pytest.skip(f"embedding model unavailable (offline?): {exc}")


def test_chunk_text_short_is_single():
    assert chunk_text("just a few words") == ["just a few words"]


def test_chunk_text_long_overlaps():
    words = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(words, max_words=100, overlap=20)
    assert len(chunks) > 1
    # Consecutive chunks should share their overlap region.
    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_similarity_identical_vs_different(store):
    same = store.similarity("backend python engineer", "backend python engineer")
    different = store.similarity("backend python engineer", "marine biology fieldwork")
    assert same > 0.99
    assert same > different


def test_index_and_query_returns_scored_evidence(store):
    store.index_criteria(
        {
            "backend.md": "Python API design, databases, testing, distributed systems.",
            "frontend.md": "React, CSS, accessibility, browser performance.",
        }
    )
    hits = store.query("experienced python backend developer", k=3)
    assert hits
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    # The backend doc should be the top match for a backend query.
    assert hits[0].source == "backend.md"


def test_query_threshold_filters_weak_matches(store, settings):
    store.index_criteria({"backend.md": "Python API design and databases."})
    # Bump the floor so only a strong match would survive; an unrelated query returns none.
    settings.relevance_threshold = 0.9
    assert store.query("competitive figure skating history") == []


def test_seed_is_idempotent(store):
    first = store.seed_criteria_if_empty()
    second = store.seed_criteria_if_empty()
    assert first > 0
    assert second == 0
