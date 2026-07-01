"""Crew assembly and the single entry point the app calls.

``build_candidate_crew`` wires the four agents into a sequential CrewAI crew.
``analyze_candidate`` runs it for one resume and returns a ScoreCard;
``run`` does a batch, ranks the results, and (optionally) sends borderline
candidates to the panel.

One deliberate choice worth flagging: the authoritative numbers come from the
scoring task's output, and the overall score plus the interview/hold/reject label
are recomputed in Python after the crew finishes. The LLM writes the rationale; it
does not get to drift the arithmetic. See docs/system_design.md for why the crew
is shaped this way and where a CrewAI Flow would be leaner.
"""

from __future__ import annotations

from crewai import Crew, Process, Task
from pydantic import BaseModel

from .agents import (
    build_intake_agent,
    build_matching_agent,
    build_orchestrator_agent,
    build_screening_agent,
)
from .config import Settings, get_settings
from .logging import configure_logging, get_logger
from .models import CandidateProfile, MatchResult, Recommendation, ScoreCard, recommend
from .panel import deliberate

log = get_logger(__name__)


class _Summary(BaseModel):
    """Small relay type for the orchestrator's output: prose plus its own read of
    the verdict. The verdict is advisory; ``analyze_candidate`` re-derives the
    authoritative one from the score."""

    rationale: str
    recommendation: Recommendation


def build_candidate_crew(settings: Settings | None = None) -> Crew:
    settings = settings or get_settings()
    intake = build_intake_agent(settings)
    matcher = build_matching_agent(settings)
    screener = build_screening_agent(settings)
    coordinator = build_orchestrator_agent(settings)

    intake_task = Task(
        description="Extract a complete, structured profile from this resume:\n\n{resume_text}",
        expected_output="A structured candidate profile with skills, work history, and education.",
        agent=intake,
        output_pydantic=CandidateProfile,
    )
    match_task = Task(
        description=(
            "Using the candidate profile from the previous step, measure how well the "
            "candidate fits this role and collect the supporting evidence.\n\nRole:\n{job_description}"
        ),
        expected_output="A match result: a role_similarity score and the evidence behind it.",
        agent=matcher,
        context=[intake_task],
        output_pydantic=MatchResult,
    )
    score_task = Task(
        description=(
            "Score the candidate against the role on the weighted rubric, using retrieved "
            "hiring criteria as context. Give each category a score and a one-line "
            "justification.\n\nRole:\n{job_description}"
        ),
        expected_output="A scorecard: per-category scores with justifications and an overall score.",
        agent=screener,
        context=[intake_task, match_task],
        output_pydantic=ScoreCard,
    )
    summary_task = Task(
        description=(
            "Write the recruiter-facing rationale for this candidate in two or three "
            "sentences, grounded in the rubric scores and the retrieved evidence. State a "
            "recommendation of interview, hold, or reject consistent with the overall score."
        ),
        expected_output="A short rationale and a recommendation.",
        agent=coordinator,
        context=[score_task, match_task],
        output_pydantic=_Summary,
    )

    return Crew(
        agents=[intake, matcher, screener, coordinator],
        tasks=[intake_task, match_task, score_task, summary_task],
        process=Process.sequential,
        verbose=False,
    )


def analyze_candidate(
    job_description: str,
    resume_text: str,
    source: str = "",
    settings: Settings | None = None,
) -> ScoreCard:
    """Run the crew for one resume and return a reconciled ScoreCard."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    crew = build_candidate_crew(settings)
    result = crew.kickoff(inputs={"job_description": job_description, "resume_text": resume_text})

    card = _pick(result, ScoreCard)
    if card is None:
        raise RuntimeError("crew run did not produce a ScoreCard")

    summary = _pick(result, _Summary)
    if summary and summary.rationale.strip():
        card.rationale = summary.rationale.strip()

    match = _pick(result, MatchResult)
    if match is not None:
        card.match = match
        if not card.candidate_name:
            card.candidate_name = match.candidate_name

    return _reconcile(card, settings)


def run(
    job_description: str,
    resumes,
    settings: Settings | None = None,
) -> list[ScoreCard]:
    """Score a batch of resumes, ranked best-first.

    ``resumes`` is an iterable of ``(text, source)`` pairs (or bare strings). Parse
    files to text before calling this -- the crew only handles text. A single
    candidate that errors out is logged and skipped rather than sinking the batch.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    cards: list[ScoreCard] = []
    for item in resumes:
        text, source = _unpack(item)
        try:
            cards.append(analyze_candidate(job_description, text, source, settings))
        except Exception:
            log.exception("could not analyze %s; skipping", source or "a resume")

    cards.sort(key=lambda c: c.overall_score, reverse=True)
    for rank, card in enumerate(cards, start=1):
        card.rank = rank

    if settings.enable_panel:
        _apply_panel(job_description, cards, settings)

    return cards


def _apply_panel(job_description: str, cards: list[ScoreCard], settings: Settings) -> None:
    for card in cards:
        if card.recommendation is not Recommendation.hold:
            continue
        outcome = deliberate(card.candidate_name, job_description, card, settings)
        card.panel_summary = outcome.summary or None
        # The panel can push a borderline candidate off the fence; honor its verdict.
        if outcome.consensus is not None:
            card.recommendation = outcome.consensus


def _reconcile(card: ScoreCard, settings: Settings) -> ScoreCard:
    # Recompute the overall from the category scores and re-derive the label. Keeps a
    # relayed number honest and the label always consistent with it.
    if card.categories:
        card.overall_score = card.weighted_check
    card.recommendation = recommend(
        card.overall_score, settings.borderline_low, settings.borderline_high
    )
    return card


def _pick(result, model_type):
    """Return the first task output that is an instance of ``model_type``."""
    top = getattr(result, "pydantic", None)
    if isinstance(top, model_type):
        return top
    for task_output in getattr(result, "tasks_output", []) or []:
        candidate = getattr(task_output, "pydantic", None)
        if isinstance(candidate, model_type):
            return candidate
    return None


def _unpack(item) -> tuple[str, str]:
    if isinstance(item, (tuple, list)):
        text = item[0]
        source = item[1] if len(item) > 1 else ""
        return text, source
    return item, ""
