"""Domain models.

These are the typed objects that move between pipeline stages. Passing Pydantic
models rather than dicts means a malformed hand-off fails loudly at the boundary
instead of three steps later with a KeyError.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Seniority(str, Enum):
    intern = "intern"
    junior = "junior"
    mid = "mid"
    senior = "senior"
    staff = "staff"
    lead = "lead"
    unknown = "unknown"


class Recommendation(str, Enum):
    interview = "interview"
    hold = "hold"
    reject = "reject"


class WorkExperience(BaseModel):
    company: str
    title: str
    # Free-form on purpose: resumes write dates every which way and normalizing
    # them reliably isn't worth it for a screening pass.
    duration: str = ""
    responsibilities: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str = ""
    field: str = ""
    year: str = ""


class CandidateProfile(BaseModel):
    """Structured resume, produced by the intake step."""

    name: str
    email: str | None = None
    phone: str | None = None
    target_role: str = ""
    seniority: Seniority = Seniority.unknown
    skills: list[str] = Field(default_factory=list)
    work_history: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    summary: str = ""
    source: str = ""  # filename or "pasted text"; handy for the UI, never logged at INFO


class EvidenceChunk(BaseModel):
    """A retrieved snippet plus how close it was to the query (cosine, 0-1)."""

    text: str
    source: str
    score: float

    def short(self, n: int = 160) -> str:
        body = self.text.strip().replace("\n", " ")
        return body if len(body) <= n else body[: n - 1] + "\u2026"


class MatchResult(BaseModel):
    """Output of the matching step: similarity signal + the evidence behind it."""

    candidate_name: str
    # Cosine between the whole JD and the whole profile (both chunked, then averaged).
    role_similarity: float
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    # Populated only when a candidate pool is loaded into the store.
    nearest_candidates: list[EvidenceChunk] = Field(default_factory=list)


class CategoryScore(BaseModel):
    category: str
    score: float = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    justification: str


class ScoreCard(BaseModel):
    """The full recruiter-facing result for one candidate."""

    candidate_name: str
    overall_score: float = Field(ge=0, le=100)
    recommendation: Recommendation
    rationale: str
    categories: list[CategoryScore] = Field(default_factory=list)
    match: MatchResult | None = None
    panel_summary: str | None = None  # set only when the borderline debate ran
    rank: int | None = None  # assigned after all candidates are scored

    @property
    def weighted_check(self) -> float:
        """Recompute the overall from categories; used in tests as a sanity check."""
        if not self.categories:
            return self.overall_score
        return round(sum(c.score * c.weight for c in self.categories), 1)


def recommend(overall: float, low: float, high: float) -> Recommendation:
    """Map an overall score to a recommendation using the borderline band.

    At or above ``high`` is a clear interview; below ``low`` is a clear reject; the
    band between them is a hold (and, if enabled, what triggers the panel debate).
    Kept as a plain function so the label is always a deterministic function of the
    number -- the LLM never decides this.
    """
    if overall >= high:
        return Recommendation.interview
    if overall < low:
        return Recommendation.reject
    return Recommendation.hold
