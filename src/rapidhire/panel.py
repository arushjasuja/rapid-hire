"""Optional interview panel: a short AutoGen debate for borderline candidates.

Isolated on purpose. The rest of the pipeline never imports autogen; this module
is only touched when ``enable_panel`` is on and a candidate lands in the hold band.
If AutoGen isn't installed or the panel is off, nothing here runs.

The autogen-agentchat API (v0.4+) is async top to bottom. The call site in the
crew is sync, so ``deliberate`` bridges the two: it runs the coroutine directly
when there's no event loop (the normal case under Streamlit and the CLI), and
falls back to a worker thread with its own loop if one is already running.
"""

from __future__ import annotations

import asyncio
import re
import threading

from pydantic import BaseModel

from .config import Settings, get_settings, require_api_key
from .logging import get_logger
from .models import Recommendation, ScoreCard

log = get_logger(__name__)

_CONSENSUS_RE = re.compile(r"CONSENSUS:\s*(interview|hold|reject)", re.IGNORECASE)


class PanelResult(BaseModel):
    consensus: Recommendation | None = None
    summary: str = ""


_ROLES = {
    "CEO": "You weigh business impact and hiring risk. You care about whether this "
    "person moves the company forward and what a bad hire would cost.",
    "CTO": "You judge technical depth and whether the candidate can do the actual work. "
    "You're skeptical of buzzwords without substance.",
    "HR": "You focus on communication, collaboration, and fairness. You watch for bias in "
    "the reasoning and push back if a judgment isn't grounded in evidence.",
}


def _panel_prompt(candidate_name: str, job_description: str, card: ScoreCard) -> str:
    lines = [f"- {c.category}: {c.score:.0f} ({c.justification})" for c in card.categories]
    return (
        f"Candidate '{candidate_name}' scored {card.overall_score:.0f}/100 for this role, "
        f"which is borderline. Decide together whether to interview, hold, or reject.\n\n"
        f"Role:\n{job_description}\n\n"
        f"Rubric scores:\n" + "\n".join(lines) + "\n\n"
        f"Draft rationale: {card.rationale}\n\n"
        "Discuss briefly, then have one member post a final line exactly in the form "
        "'CONSENSUS: interview' (or hold, or reject)."
    )


async def run_panel(
    candidate_name: str, job_description: str, card: ScoreCard, settings: Settings
) -> PanelResult:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    require_api_key(settings)
    client = OpenAIChatCompletionClient(model=settings.reasoning_model, temperature=0.3)
    agents = [
        AssistantAgent(name, model_client=client, system_message=msg)
        for name, msg in _ROLES.items()
    ]
    # Stop as soon as someone declares consensus, or hard-cap the turns so a stuck
    # debate can't burn tokens forever.
    termination = TextMentionTermination("CONSENSUS:") | MaxMessageTermination(
        settings.panel_max_messages
    )
    team = RoundRobinGroupChat(
        agents, termination_condition=termination, max_turns=settings.panel_max_messages
    )

    try:
        result = await team.run(task=_panel_prompt(candidate_name, job_description, card))
    finally:
        # Close the HTTP client regardless of how the run ended.
        await client.close()

    transcript: list[str] = []
    consensus: Recommendation | None = None
    for message in result.messages:
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        source = getattr(message, "source", "?")
        transcript.append(f"{source}: {content.strip()}")
        found = _CONSENSUS_RE.search(content)
        if found:
            consensus = Recommendation(found.group(1).lower())

    summary = _summarize(transcript, consensus)
    log.info("panel finished for %s: consensus=%s", candidate_name, consensus)
    return PanelResult(consensus=consensus, summary=summary)


def _summarize(transcript: list[str], consensus: Recommendation | None) -> str:
    # Keep the last few exchanges; the early turns are usually preamble.
    tail = transcript[-3:] if len(transcript) > 3 else transcript
    body = "\n".join(tail)
    verdict = consensus.value if consensus else "no clear consensus"
    return f"Panel verdict: {verdict}.\n{body}"


def deliberate(
    candidate_name: str,
    job_description: str,
    card: ScoreCard,
    settings: Settings | None = None,
) -> PanelResult:
    """Run the panel synchronously. Returns an empty result if the panel is off."""
    settings = settings or get_settings()
    if not settings.enable_panel:
        return PanelResult()

    def make_coro():
        return run_panel(candidate_name, job_description, card, settings)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread: the straightforward path.
        return asyncio.run(make_coro())
    # Something already owns the loop here; give the coroutine its own loop on a
    # separate thread and block until it's done.
    return _run_in_thread(make_coro)


def _run_in_thread(make_coro) -> PanelResult:
    box: dict[str, PanelResult] = {}

    def worker():
        loop = asyncio.new_event_loop()
        try:
            box["result"] = loop.run_until_complete(make_coro())
        finally:
            loop.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    return box.get("result", PanelResult())
