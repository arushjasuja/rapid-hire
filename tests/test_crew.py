"""Crew wiring and batch behavior.

Two things are worth testing without an API key: that the crew is assembled with
the right agents, tasks, and output types; and that ``run`` ranks correctly, skips
a candidate that errors, and only sends hold-band cards to the panel. The heavy
part -- the actual LLM crew -- is monkeypatched. What's under test is the plumbing
around it.
"""

from __future__ import annotations

import pytest

# The crew module imports CrewAI and LangChain at module load (agents and tools
# pull them in at the top level), so skip cleanly if the heavy stack isn't present.
pytest.importorskip("crewai")
pytest.importorskip("langchain_openai")

from conftest import make_card  # plain helper, not a fixture
from rapidhire import crew as crew_mod
from rapidhire.crew import _Summary, build_candidate_crew, run
from rapidhire.models import (
    CandidateProfile,
    MatchResult,
    Recommendation,
    ScoreCard,
)
from rapidhire.panel import PanelResult


def _tool_names(agent) -> set[str]:
    return {t.name for t in getattr(agent, "tools", []) or []}


def test_build_candidate_crew_structure(settings):
    crew = build_candidate_crew(settings)

    assert len(crew.agents) == 4
    assert len(crew.tasks) == 4

    # Task outputs are typed, and the order is intake -> match -> score -> summary.
    output_types = [t.output_pydantic for t in crew.tasks]
    assert output_types == [CandidateProfile, MatchResult, ScoreCard, _Summary]

    intake, matcher, screener, coordinator = crew.agents
    # Each worker agent carries exactly the tool it needs; the coordinator writes
    # prose and holds none.
    assert "extract_candidate_profile" in _tool_names(intake)
    assert "match_candidate" in _tool_names(matcher)
    assert "score_candidate" in _tool_names(screener)
    assert _tool_names(coordinator) == set()


def test_run_ranks_best_first_and_numbers_them(settings, monkeypatch):
    scores = {"alice resume": 55.0, "bob resume": 82.0, "carol resume": 40.0}

    def fake_analyze(job_description, resume_text, source="", settings=None):
        name = source.replace(".txt", "")
        return make_card(name, scores[resume_text])

    monkeypatch.setattr(crew_mod, "analyze_candidate", fake_analyze)

    resumes = [
        ("alice resume", "alice.txt"),
        ("bob resume", "bob.txt"),
        ("carol resume", "carol.txt"),
    ]
    cards = run("some job", resumes, settings)

    assert [c.candidate_name for c in cards] == ["bob", "alice", "carol"]
    assert [c.overall_score for c in cards] == [82.0, 55.0, 40.0]
    assert [c.rank for c in cards] == [1, 2, 3]


def test_run_skips_a_candidate_that_errors(settings, monkeypatch):
    def fake_analyze(job_description, resume_text, source="", settings=None):
        if "broken" in resume_text:
            raise RuntimeError("intake blew up")
        return make_card(source.replace(".txt", ""), 70.0)

    monkeypatch.setattr(crew_mod, "analyze_candidate", fake_analyze)

    resumes = [
        ("good one", "ok.txt"),
        ("broken one", "bad.txt"),
        ("good two", "fine.txt"),
    ]
    cards = run("some job", resumes, settings)

    names = [c.candidate_name for c in cards]
    assert names == ["ok", "fine"]  # the failed resume is dropped, not fatal
    assert [c.rank for c in cards] == [1, 2]


def test_panel_runs_only_for_hold_band(tmp_path, criteria_dir, monkeypatch):
    # Fresh settings with the panel switched on. The autouse cache reset means no
    # earlier get_settings() call is cached, so this env is what gets read.
    from rapidhire.config import get_settings

    monkeypatch.setenv("RAPIDHIRE_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("RAPIDHIRE_CRITERIA_DIR", str(criteria_dir))
    monkeypatch.setenv("RAPIDHIRE_ENABLE_PANEL", "true")
    monkeypatch.delenv("RAPIDHIRE_OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    panel_settings = get_settings()
    assert panel_settings.enable_panel is True

    def card_with(name, overall, rec):
        card = make_card(name, overall)
        card.recommendation = rec
        return card

    canned = {
        "clear yes": card_with("Yes", 82.0, Recommendation.interview),
        "on the fence": card_with("Fence", 60.0, Recommendation.hold),
        "clear no": card_with("No", 40.0, Recommendation.reject),
    }

    def fake_analyze(job_description, resume_text, source="", settings=None):
        return canned[resume_text]

    monkeypatch.setattr(crew_mod, "analyze_candidate", fake_analyze)

    seen: list[str] = []

    def fake_deliberate(candidate_name, job_description, card, settings=None):
        seen.append(candidate_name)
        return PanelResult(consensus=Recommendation.interview, summary="panel leaned yes")

    monkeypatch.setattr(crew_mod, "deliberate", fake_deliberate)

    resumes = [("clear yes", "y.txt"), ("on the fence", "f.txt"), ("clear no", "n.txt")]
    cards = run("some job", resumes, panel_settings)

    # Only the hold-band candidate is deliberated.
    assert seen == ["Fence"]

    by_name = {c.candidate_name: c for c in cards}
    # The panel pushed the borderline candidate to a yes and left a summary.
    assert by_name["Fence"].recommendation is Recommendation.interview
    assert by_name["Fence"].panel_summary == "panel leaned yes"
    # The clear cases are untouched by the panel.
    assert by_name["Yes"].recommendation is Recommendation.interview
    assert by_name["Yes"].panel_summary is None
    assert by_name["No"].recommendation is Recommendation.reject
    assert by_name["No"].panel_summary is None
