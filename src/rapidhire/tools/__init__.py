"""Agent-facing tools.

Each tool is a thin CrewAI wrapper over a plain function that does the actual
work. The split matters for testing: the plain functions are called directly in
the test suite (mock the LLM, feed a fixture), while the CrewAI wrappers only get
exercised when the crew runs for real.

Imports here are deliberately lazy. Parsing has no LLM dependency, so importing
this package shouldn't drag in openai/tenacity just to read a text file.
"""

from __future__ import annotations


def llm_retry(attempts: int = 3):
    """Backoff decorator for functions that make an LLM call.

    Exponential wait capped at ~8s so a brief rate-limit blip doesn't abort a whole
    screening run, without hanging on a genuine outage. Only transient failures are
    retried; a 400/401 fails fast because it never succeeds on retry.
    """
    import openai
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    retryable = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(retryable),
        reraise=True,
    )
