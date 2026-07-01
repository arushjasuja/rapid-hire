"""Matching tool: how close is this candidate to the role, and on what evidence.

Plain function first (``run_match``), CrewAI wrapper second. The function does no
LLM work at all -- it's embeddings and a vector lookup, which is the part of the
pipeline that actually hits the sub-100ms budget once the model is warm.
"""

from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..models import MatchResult
from ..vectorstore import VectorStore, get_vectorstore


def run_match(
    job_description: str,
    candidate_name: str,
    candidate_summary: str,
    store: VectorStore | None = None,
) -> MatchResult:
    """Score role fit by cosine similarity and pull supporting criteria chunks."""
    vs = store or get_vectorstore()
    similarity = vs.similarity(job_description, candidate_summary)

    # Retrieve against the JD plus the candidate blob so the evidence reflects the
    # overlap, not just the role text. This is criteria evidence, not the RAG step
    # that feeds scoring -- that lives in scoring.py.
    evidence = vs.query(f"{job_description}\n{candidate_summary}")

    # Only meaningful once a candidate pool has been indexed; empty otherwise.
    nearest = vs.query(candidate_summary, pool=True)

    return MatchResult(
        candidate_name=candidate_name,
        role_similarity=round(similarity, 4),
        evidence=evidence,
        nearest_candidates=nearest,
    )


class _MatchArgs(BaseModel):
    job_description: str = Field(description="The full job description text.")
    candidate_name: str = Field(description="Candidate's name, for labelling the result.")
    candidate_summary: str = Field(
        description="The candidate's profile as text: skills, roles, and summary."
    )


class MatchingTool(BaseTool):
    name: str = "match_candidate"
    description: str = (
        "Compute semantic similarity between a candidate and a role, and return the "
        "similarity score plus the hiring-criteria snippets that back it up. Call this "
        "once per candidate with the job description and the candidate's profile text."
    )
    args_schema: type[BaseModel] = _MatchArgs

    def _run(self, job_description: str, candidate_name: str, candidate_summary: str) -> str:
        result = run_match(job_description, candidate_name, candidate_summary)
        return result.model_dump_json()
