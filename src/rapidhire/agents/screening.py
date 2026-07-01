"""Screening agent: holds the rubric-scoring tool (the RAG step).

The scoring tool does its own retrieval and its own LLM call on the reasoning
model, so the agent itself stays light -- it calls the tool and passes the result
on. The redundancy (an agent turn wrapping a tool that itself calls an LLM) is a
consequence of running everything through CrewAI; see docs/system_design.md.
"""

from __future__ import annotations

from crewai import Agent

from ..config import Settings, get_settings
from ..tools.scoring import RubricScoringTool


def build_screening_agent(settings: Settings | None = None) -> Agent:
    settings = settings or get_settings()
    return Agent(
        role="Screening Analyst",
        goal="Score the candidate against the role rubric with clear justifications.",
        backstory="You grade fit against explicit criteria and won't hand-wave a score "
        "you can't defend in a sentence.",
        tools=[RubricScoringTool()],
        llm=settings.intake_model,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
