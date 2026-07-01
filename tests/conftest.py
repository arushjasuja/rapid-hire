"""Shared fixtures.

The important bits: every test gets a clean settings cache and a fresh vector-store
singleton, and anything touching the store points at a temp directory so runs don't
pollute each other or the real ./data/chroma.
"""

from __future__ import annotations

import pytest

from rapidhire.config import get_settings
from rapidhire.models import (
    CandidateProfile,
    CategoryScore,
    Recommendation,
    ScoreCard,
    Seniority,
)
from rapidhire.vectorstore import reset_vectorstore


@pytest.fixture(autouse=True)
def _clean_caches():
    # Settings and the store are process-cached; reset both so env changes in one
    # test don't leak into the next.
    get_settings.cache_clear()
    reset_vectorstore()
    yield
    get_settings.cache_clear()
    reset_vectorstore()


@pytest.fixture
def criteria_dir(tmp_path):
    d = tmp_path / "criteria"
    d.mkdir()
    (d / "backend.md").write_text(
        "Backend engineer criteria. Strong Python and API design. Experience with "
        "databases, testing, and distributed systems. Communication matters for on-call.",
        encoding="utf-8",
    )
    (d / "general.md").write_text(
        "General hiring bar. Ownership, clear writing, and evidence of shipping real "
        "software. Degree is not required if experience is strong.",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def settings(tmp_path, criteria_dir, monkeypatch):
    monkeypatch.setenv("RAPIDHIRE_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("RAPIDHIRE_CRITERIA_DIR", str(criteria_dir))
    monkeypatch.setenv("RAPIDHIRE_ENABLE_PANEL", "false")
    monkeypatch.delenv("RAPIDHIRE_OPENAI_API_KEY", raising=False)
    return get_settings()


@pytest.fixture
def sample_profile():
    return CandidateProfile(
        name="Dana Lopez",
        email="dana@example.com",
        target_role="Backend Engineer",
        seniority=Seniority.senior,
        skills=["Python", "PostgreSQL", "FastAPI", "AWS"],
        summary="Senior backend engineer with eight years building APIs.",
        source="dana.txt",
    )


def make_card(name: str, overall: float) -> ScoreCard:
    """Build a plausible ScoreCard for tests that don't exercise scoring itself."""
    from rapidhire.tools.scoring import RUBRIC

    cats = [
        CategoryScore(category=c, score=overall, weight=w, justification="test") for c, w in RUBRIC
    ]
    return ScoreCard(
        candidate_name=name,
        overall_score=overall,
        recommendation=Recommendation.hold,
        rationale="placeholder",
        categories=cats,
    )
