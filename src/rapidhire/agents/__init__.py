"""Agent builders, one per pipeline stage.

Each returns a configured CrewAI ``Agent``. They're functions rather than
module-level singletons so a fresh crew can be built per run with the current
settings, and so importing this package doesn't construct anything eagerly.
"""

from __future__ import annotations

from .intake import build_intake_agent
from .matching import build_matching_agent
from .orchestrator import build_orchestrator_agent
from .screening import build_screening_agent

__all__ = [
    "build_intake_agent",
    "build_matching_agent",
    "build_screening_agent",
    "build_orchestrator_agent",
]
