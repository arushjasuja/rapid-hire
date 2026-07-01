"""Streamlit front end.

Upload or paste a job description and some resumes, hit run, and get a ranked,
explainable result per candidate. The heavy objects (embedding model, Chroma
client) are cached across reruns so only the first load pays the startup cost.

Kept intentionally thin: this file handles input, calls ``crew.run``, and renders.
No scoring logic lives here.
"""

from __future__ import annotations

import streamlit as st

from rapidhire.config import MissingAPIKeyError, get_settings, require_api_key
from rapidhire.crew import run
from rapidhire.logging import configure_logging
from rapidhire.models import Recommendation
from rapidhire.tools.parsing import UnsupportedResumeError, extract_text
from rapidhire.vectorstore import get_vectorstore

st.set_page_config(page_title="RapidHire", page_icon=None, layout="wide")

_BADGE = {
    Recommendation.interview: st.success,
    Recommendation.hold: st.warning,
    Recommendation.reject: st.error,
}


@st.cache_resource(show_spinner="Loading embedding model and criteria...")
def _warm_store():
    # Seeds the criteria on first call; cached so reruns reuse the same instance.
    return get_vectorstore()


def _sidebar(settings):
    with st.sidebar:
        st.header("Setup")
        try:
            require_api_key()
            st.caption("OpenAI key detected.")
        except MissingAPIKeyError:
            st.error("No OpenAI API key. Set RAPIDHIRE_OPENAI_API_KEY or OPENAI_API_KEY.")
        settings.enable_panel = st.toggle(
            "Panel debate for borderline scores",
            value=settings.enable_panel,
            help="Runs a short AutoGen CEO/CTO/HR debate on candidates in the hold band.",
        )
        st.caption(f"Reasoning model: {settings.reasoning_model}")
        st.caption(f"Intake model: {settings.intake_model}")
        st.caption(f"Embeddings: {settings.embedding_model}")


def _collect_job() -> str:
    st.subheader("Job description")
    uploaded = st.file_uploader("Upload a JD", type=["txt", "md"], key="job_file")
    pasted = st.text_area("...or paste it", height=180, key="job_text")
    if uploaded is not None:
        try:
            return extract_text(uploaded)
        except UnsupportedResumeError as exc:
            st.error(str(exc))
    return pasted.strip()


def _collect_resumes() -> list[tuple[str, str]]:
    st.subheader("Resumes")
    files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="resume_files",
    )
    resumes: list[tuple[str, str]] = []
    for f in files or []:
        try:
            text = extract_text(f)
        except (UnsupportedResumeError, ValueError) as exc:
            st.warning(f"Skipped {f.name}: {exc}")
            continue
        if text:
            resumes.append((text, f.name))

    pasted = st.text_area("...or paste a single resume", height=150, key="resume_text")
    if pasted.strip():
        resumes.append((pasted.strip(), "pasted resume"))
    return resumes


def _render(cards) -> None:
    st.subheader("Results")
    if not cards:
        st.info("No candidates were scored.")
        return
    for card in cards:
        header = f"#{card.rank} - {card.candidate_name} - {card.overall_score:.1f}/100"
        with st.expander(header, expanded=card.rank == 1):
            _BADGE[card.recommendation](f"Recommendation: {card.recommendation.value}")
            st.write(card.rationale)

            rows = [
                {
                    "Category": c.category,
                    "Score": round(c.score, 1),
                    "Weight": c.weight,
                    "Why": c.justification,
                }
                for c in card.categories
            ]
            if rows:
                st.table(rows)

            if card.panel_summary:
                with st.expander("Panel debate"):
                    st.text(card.panel_summary)

            if card.match and card.match.evidence:
                with st.expander("Retrieved evidence"):
                    for chunk in card.match.evidence:
                        st.markdown(f"**{chunk.source}** ({chunk.score:.2f}) - {chunk.short(240)}")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    _warm_store()

    st.title("RapidHire")
    st.caption(
        "Parse, match, and score candidates against a role. A technical demo, not a hiring decision."
    )

    _sidebar(settings)

    left, right = st.columns(2)
    with left:
        job = _collect_job()
    with right:
        resumes = _collect_resumes()

    if st.button("Run screening", type="primary", disabled=not (job and resumes)):
        try:
            require_api_key()
        except MissingAPIKeyError as exc:
            st.error(str(exc))
            return
        with st.spinner(f"Screening {len(resumes)} candidate(s)..."):
            cards = run(job, resumes, settings)
        # Keep only the rendered results in session state, not the raw resume text.
        st.session_state["results"] = cards

    if "results" in st.session_state:
        _render(st.session_state["results"])


main()
