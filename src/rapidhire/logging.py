"""Logging setup.

One place configures the root handler; everything else just calls ``get_logger``.
The important rule here is a policy one, not a technical one: resume text and
contact details are candidate PII and must never land in INFO logs. Helpers below
exist so call sites can log *that* something happened and *how big* it was without
logging *what it said*.
"""

from __future__ import annotations

import hashlib
import logging

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Chroma and httpx are chatty at INFO; we only want their warnings.
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def redact(text: str, keep: int = 0) -> str:
    """Return a short, non-reversible tag for a blob of text.

    Use in logs instead of the text itself. ``keep`` optionally leaves a few
    leading characters so a human debugging can tell two documents apart.
    """
    digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:8]
    prefix = text[:keep].replace("\n", " ") if keep else ""
    return f"{prefix}<{len(text)}chars sha1:{digest}>"
