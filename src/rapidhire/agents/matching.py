"""Matching agent: a thin wrapper around the semantic-match tool.

The real work is deterministic (embeddings + vector lookup), so this agent runs on
the cheap model. It exists to fit the matching step into the crew, not to reason.
"""

from __future__ import annotations

from crewai import Agent

from ..config import Settings, get_settings
from ..tools.matching import MatchingTool


def build_matching_agent(settings: Settings | None = None) -> Agent:
    settings = settings or get_settings()
    return Agent(
        role="Semantic Matcher",
        goal="Measure how well a candidate fits the role and surface the evidence.",
        backstory="You translate resumes and roles into vectors and trust the numbers "
        "over gut feel.",
        tools=[MatchingTool()],
        llm=settings.intake_model,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
