"""RapidHire: multi-agent candidate screening.

The public surface most callers want is ``run`` (score a batch of resumes against
a job description) and the domain models. Everything else is an implementation
detail of the pipeline.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Kept intentionally short. Importing crew here would pull in crewai/langchain at
# package import time, which slows down anything that only needs the models.

__all__ = ["__version__"]
