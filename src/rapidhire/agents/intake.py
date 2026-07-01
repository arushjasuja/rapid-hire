"""Intake: raw resume text into a structured CandidateProfile.

This is the one place that uses the OpenAI SDK directly rather than through
LangChain or CrewAI, to show the strict structured-output path end to end: a
forced tool call with ``strict: true`` guarantees the arguments match the schema,
so the JSON always parses into a CandidateProfile without defensive cleanup.

Everything else in the file is the CrewAI wiring: a tool over ``extract_profile``
and the agent that holds it.
"""

from __future__ import annotations

import json

import openai
from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..config import Settings, get_settings, require_api_key
from ..logging import get_logger, redact
from ..models import CandidateProfile, Seniority
from ..tools import llm_retry

log = get_logger(__name__)

# Hand-written strict schema. It mirrors CandidateProfile, but strict mode has its
# own rules: every property must be listed in `required`, `additionalProperties`
# must be false, and an "optional" field is expressed as a nullable type rather
# than by omission. So email/phone are string-or-null and still required.
_PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "email": {"type": ["string", "null"]},
        "phone": {"type": ["string", "null"]},
        "target_role": {"type": "string", "description": "Role the candidate is aiming for."},
        "seniority": {"type": "string", "enum": [s.value for s in Seniority]},
        "skills": {"type": "array", "items": {"type": "string"}},
        "work_history": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "duration": {"type": "string"},
                    "responsibilities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["company", "title", "duration", "responsibilities"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": "string"},
                    "field": {"type": "string"},
                    "year": {"type": "string"},
                },
                "required": ["institution", "degree", "field", "year"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": [
        "name",
        "email",
        "phone",
        "target_role",
        "seniority",
        "skills",
        "work_history",
        "education",
        "summary",
    ],
}

_EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_candidate_profile",
        "description": "Return the candidate's details extracted from their resume. Infer "
        "seniority and target role from context; don't leave them blank if the resume "
        "implies an answer.",
        "strict": True,
        "parameters": _PROFILE_SCHEMA,
    },
}


def extract_profile(
    resume_text: str,
    source: str = "",
    *,
    settings: Settings | None = None,
    client: openai.OpenAI | None = None,
) -> CandidateProfile:
    """Extract a CandidateProfile from resume text via a strict tool call.

    ``client`` is injectable for tests. Note we log only a redacted fingerprint of
    the resume, never its contents -- that text is candidate PII.
    """
    settings = settings or get_settings()
    require_api_key(settings)
    client = client or openai.OpenAI(timeout=settings.request_timeout)

    log.info("intake: extracting profile from %s", redact(resume_text, keep=0))

    @llm_retry(settings.max_retries)
    def _call():
        return client.chat.completions.create(
            model=settings.intake_model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract structured data from the resume. Be faithful to "
                    "the text but infer obvious implications (e.g. seniority from years of "
                    "experience).",
                },
                {"role": "user", "content": resume_text},
            ],
            tools=[_EXTRACTION_TOOL],
            tool_choice={"type": "function", "function": {"name": "emit_candidate_profile"}},
            temperature=0,
        )

    response = _call()
    call = response.choices[0].message.tool_calls[0]
    data = json.loads(call.function.arguments)
    return CandidateProfile(source=source, **data)


class _IntakeArgs(BaseModel):
    resume_text: str = Field(description="The full plain text of one resume.")


class ResumeExtractionTool(BaseTool):
    name: str = "extract_candidate_profile"
    description: str = (
        "Parse one resume's text into a structured candidate profile (name, contact, "
        "target role, seniority, skills, work history, education). Call once per resume."
    )
    args_schema: type[BaseModel] = _IntakeArgs

    def _run(self, resume_text: str) -> str:
        profile = extract_profile(resume_text)
        return profile.model_dump_json()


def build_intake_agent(settings: Settings | None = None) -> Agent:
    settings = settings or get_settings()
    return Agent(
        role="Resume Intake Specialist",
        goal="Convert a raw resume into a complete, faithful candidate profile.",
        backstory="You've screened thousands of resumes and read between the lines of "
        "sparse ones without inventing credentials.",
        tools=[ResumeExtractionTool()],
        llm=settings.intake_model,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
