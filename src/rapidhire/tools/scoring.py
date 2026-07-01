"""Scoring tool: the RAG step and the only place retrieval feeds the LLM.

This is where LangChain earns its keep: a prompt template, an LLM bound to a
structured-output schema, and LCEL (`prompt | llm`) to wire them. The retrieval
boundary lives here and nowhere else -- role criteria are pulled from Chroma,
filtered by cosine relevance, and dropped into the prompt as context.

The rubric weights are fixed in code, not decided by the model. The LLM scores
each category and writes a justification; the weighted overall is plain arithmetic
afterwards, because a language model doing a weighted average is asking for
trouble.
"""

from __future__ import annotations

from crewai.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config import Settings, get_settings, require_api_key
from ..logging import get_logger
from ..models import CategoryScore, EvidenceChunk, ScoreCard, recommend
from ..vectorstore import VectorStore, get_vectorstore
from . import llm_retry

log = get_logger(__name__)

# (category, weight). Weights sum to 1.0. Skills and experience carry the most
# signal for a first-pass screen; education is a weak signal for most roles.
RUBRIC: list[tuple[str, float]] = [
    ("skills", 0.30),
    ("experience", 0.30),
    ("education", 0.10),
    ("role_alignment", 0.20),
    ("communication", 0.10),
]


class ScoreDraft(BaseModel):
    """What the LLM returns: a 0-100 score and a one-line reason per category."""

    skills_score: int = Field(ge=0, le=100)
    skills_reason: str
    experience_score: int = Field(ge=0, le=100)
    experience_reason: str
    education_score: int = Field(ge=0, le=100)
    education_reason: str
    role_alignment_score: int = Field(ge=0, le=100)
    role_alignment_reason: str
    communication_score: int = Field(ge=0, le=100)
    communication_reason: str
    summary: str = Field(description="Two or three sentences a recruiter would read.")


_SYSTEM = (
    "You are a hiring analyst scoring a candidate against a role. Score each rubric "
    "category from 0 to 100 and give a short, specific justification grounded in the "
    "resume and the retrieved hiring criteria. Do not inflate scores; a mid-level "
    "candidate for a senior role should score accordingly. Judge communication from "
    "how the resume itself is written."
)

_HUMAN = """Job description:
{job_description}

Retrieved hiring criteria (may be partial):
{criteria}

Candidate profile:
{candidate}

Score the candidate on: skills, experience, education, role_alignment, communication."""


def _build_scorer(settings: Settings):
    """Return the LCEL chain that turns a prompt into a ScoreDraft.

    Retries are handled by our own decorator on the caller, so the client's own
    retry is off to avoid stacking two backoff loops.
    """
    require_api_key(settings)
    llm = ChatOpenAI(
        model=settings.reasoning_model,
        temperature=0,
        timeout=settings.request_timeout,
        max_retries=0,
    )
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
    return prompt | llm.with_structured_output(ScoreDraft)


def _retrieve_criteria(
    vs: VectorStore, job_description: str, candidate_summary: str
) -> list[EvidenceChunk]:
    # Single retrieval call for the whole scoring step. Querying on JD + candidate
    # keeps the criteria relevant to this specific pairing.
    return vs.query(f"{job_description}\n{candidate_summary}")


def score_against_rubric(
    job_description: str,
    candidate_name: str,
    candidate_summary: str,
    *,
    role_similarity: float = 0.0,
    store: VectorStore | None = None,
    settings: Settings | None = None,
    scorer=None,
) -> ScoreCard:
    """Retrieve criteria, score the candidate, and assemble a ScoreCard.

    ``scorer`` is injectable so tests can supply a fake chain and skip the API call;
    in normal use it defaults to the LangChain LCEL chain above.
    """
    settings = settings or get_settings()
    vs = store or get_vectorstore()
    chunks = _retrieve_criteria(vs, job_description, candidate_summary)
    criteria_text = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks) or "(no criteria indexed)"

    chain = scorer or _build_scorer(settings)
    invoke = llm_retry(settings.max_retries)(chain.invoke)
    draft: ScoreDraft = invoke(
        {
            "job_description": job_description,
            "criteria": criteria_text,
            "candidate": candidate_summary,
        }
    )

    raw = {
        "skills": (draft.skills_score, draft.skills_reason),
        "experience": (draft.experience_score, draft.experience_reason),
        "education": (draft.education_score, draft.education_reason),
        "role_alignment": (draft.role_alignment_score, draft.role_alignment_reason),
        "communication": (draft.communication_score, draft.communication_reason),
    }
    categories = [
        CategoryScore(category=name, score=raw[name][0], weight=weight, justification=raw[name][1])
        for name, weight in RUBRIC
    ]
    overall = round(sum(c.score * c.weight for c in categories), 1)

    return ScoreCard(
        candidate_name=candidate_name,
        overall_score=overall,
        recommendation=recommend(overall, settings.borderline_low, settings.borderline_high),
        rationale=draft.summary,
        categories=categories,
    )


class _ScoreArgs(BaseModel):
    job_description: str = Field(description="The full job description text.")
    candidate_name: str = Field(description="Candidate's name.")
    candidate_summary: str = Field(description="The candidate's profile as text.")


class RubricScoringTool(BaseTool):
    name: str = "score_candidate"
    description: str = (
        "Score a candidate against the role on a weighted rubric (skills, experience, "
        "education, role alignment, communication), using retrieved hiring criteria as "
        "context. Returns per-category scores with justifications and an overall score. "
        "Call once per candidate."
    )
    args_schema: type[BaseModel] = _ScoreArgs

    def _run(self, job_description: str, candidate_name: str, candidate_summary: str) -> str:
        card = score_against_rubric(job_description, candidate_name, candidate_summary)
        return card.model_dump_json()
