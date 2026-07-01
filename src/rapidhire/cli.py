"""Command-line entry point.

Handy for scoring a batch without the UI, and for smoke-testing a real run:

    rapidhire data/sample_jobs/backend_engineer.md data/sample_resumes/*.txt

Needs an API key in the environment (or .env), same as the app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import MissingAPIKeyError, get_settings, require_api_key
from .crew import run
from .logging import configure_logging
from .models import ScoreCard
from .tools.parsing import extract_text
from .vectorstore import get_vectorstore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rapidhire", description="Screen resumes against a job description."
    )
    parser.add_argument("job", type=Path, help="Job description file (.md/.txt).")
    parser.add_argument("resumes", type=Path, nargs="+", help="Resume files (.pdf/.docx/.txt/.md).")
    parser.add_argument("--json", action="store_true", help="Emit full results as JSON.")
    parser.add_argument(
        "--panel", action="store_true", help="Enable the AutoGen panel for borderline scores."
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.panel:
        settings.enable_panel = True
    configure_logging(settings.log_level)

    try:
        require_api_key()
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    get_vectorstore()  # seed criteria before scoring so retrieval has context

    job_text = extract_text(args.job)
    resumes = [(extract_text(path), path.name) for path in args.resumes]
    cards = run(job_text, resumes, settings)

    if args.json:
        print(json.dumps([c.model_dump(mode="json") for c in cards], indent=2))
    else:
        _print_table(cards)
    return 0


def _print_table(cards: list[ScoreCard]) -> None:
    if not cards:
        print("No candidates scored.")
        return
    width = max(len(c.candidate_name) for c in cards)
    print(f"\n{'#':>2}  {'Candidate':<{width}}  {'Score':>5}  Recommendation")
    print("-" * (width + 22))
    for card in cards:
        print(
            f"{card.rank:>2}  {card.candidate_name:<{width}}  "
            f"{card.overall_score:>5.1f}  {card.recommendation.value}"
        )
    print()
    for card in cards:
        print(f"{card.candidate_name} - {card.rationale}")


if __name__ == "__main__":
    raise SystemExit(main())
