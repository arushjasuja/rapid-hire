"""Tool tests.

Parsing is exercised for real. Matching, scoring, and intake are exercised with
their LLM/store dependencies injected as fakes, so no network or API key is needed
and the deterministic parts (arithmetic, schema parsing) are what's under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapidhire.models import EvidenceChunk
from rapidhire.tools.parsing import extract_text


# --- parsing ---------------------------------------------------------------
def test_extract_text_from_path(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_text("Jane Doe\nPython engineer, 5 years.", encoding="utf-8")
    assert "Python engineer" in extract_text(p)


def test_extract_text_from_bytes_needs_filename():
    with pytest.raises(ValueError):
        extract_text(b"some bytes")
    assert "hello world" in extract_text(b"hello world here", filename="note.md")


def test_extract_text_docx(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "cv.docx"
    doc = docx.Document()
    doc.add_paragraph("Alex Kim")
    doc.add_paragraph("Staff engineer, distributed systems.")
    doc.save(path)
    text = extract_text(path)
    assert "Alex Kim" in text and "distributed systems" in text


# --- matching --------------------------------------------------------------
class _FakeStore:
    def __init__(self, sim, evidence, nearest=None):
        self._sim = sim
        self._evidence = evidence
        self._nearest = nearest or []

    def similarity(self, a, b):
        return self._sim

    def query(self, text, k=None, *, pool=False):
        return self._nearest if pool else self._evidence


def test_run_match_builds_result():
    from rapidhire.tools.matching import run_match

    store = _FakeStore(
        sim=0.72,
        evidence=[EvidenceChunk(text="python api design", source="backend.md", score=0.8)],
    )
    result = run_match("backend role", "Dana Lopez", "python, apis", store=store)
    assert result.candidate_name == "Dana Lopez"
    assert result.role_similarity == pytest.approx(0.72, abs=1e-6)
    assert result.evidence[0].source == "backend.md"


# --- scoring ---------------------------------------------------------------
def _fake_draft():
    from rapidhire.tools.scoring import ScoreDraft

    return ScoreDraft(
        skills_score=90,
        skills_reason="strong python",
        experience_score=80,
        experience_reason="8 years",
        education_score=60,
        education_reason="bachelors",
        role_alignment_score=85,
        role_alignment_reason="direct match",
        communication_score=70,
        communication_reason="clear resume",
        summary="Solid senior backend candidate.",
    )


class _FakeChain:
    def invoke(self, _inputs):
        return _fake_draft()


def test_score_against_rubric_weighted_overall(settings):
    from rapidhire.tools.scoring import RUBRIC, score_against_rubric

    store = _FakeStore(sim=0.0, evidence=[])
    card = score_against_rubric(
        "backend role",
        "Dana Lopez",
        "python, apis, 8 years",
        store=store,
        settings=settings,
        scorer=_FakeChain(),
    )
    expected = 90 * 0.30 + 80 * 0.30 + 60 * 0.10 + 85 * 0.20 + 70 * 0.10
    assert card.overall_score == pytest.approx(round(expected, 1))
    assert len(card.categories) == len(RUBRIC)
    assert card.weighted_check == card.overall_score
    # 80.0 is in the [50, 70)? No -- it's >= 70, so interview.
    assert card.recommendation.value == "interview"


# --- intake ----------------------------------------------------------------
def test_extract_profile_parses_strict_toolcall(settings):
    # The client is faked, so no real call happens; the key just satisfies the gate.
    settings.openai_api_key = "sk-test"
    from rapidhire.agents.intake import extract_profile

    arguments = (
        '{"name":"Dana Lopez","email":"dana@example.com","phone":null,'
        '"target_role":"Backend Engineer","seniority":"senior",'
        '"skills":["Python","FastAPI"],"work_history":[],"education":[],'
        '"summary":"Senior backend engineer."}'
    )
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=arguments))
    message = SimpleNamespace(tool_calls=[tool_call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
    )
    profile = extract_profile(
        "resume text here", source="dana.txt", settings=settings, client=fake_client
    )
    assert profile.name == "Dana Lopez"
    assert profile.seniority.value == "senior"
    assert profile.source == "dana.txt"
    assert profile.phone is None
