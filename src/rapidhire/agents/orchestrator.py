"""Orchestrator agent: turns the scored result into a recruiter-facing summary.

This is the only agent doing genuine LLM reasoning rather than relaying a tool
result, so it runs on the reasoning model. It has no tools -- it reads the scoring
and matching output from the task context and writes the rationale. The numeric
overall and the interview/hold/reject label are recomputed in Python afterwards
(see crew.analyze_candidate); the agent's job is the prose, not the arithmetic.
"""

from __future__ import annotations

from crewai import Agent

from ..config import Settings, get_settings


def build_orchestrator_agent(settings: Settings | None = None) -> Agent:
    settings = settings or get_settings()
    return Agent(
        role="Hiring Coordinator",
        goal="Summarize the screening result into a clear recommendation and rationale.",
        backstory="You brief hiring managers. You're concise, you cite the evidence, and "
        "you flag when a candidate is a genuine maybe rather than forcing a verdict.",
        llm=settings.reasoning_model,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
