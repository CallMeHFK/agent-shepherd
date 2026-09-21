"""Tier 0 deterministic detectors.

These run on every event, cost nothing, and decide *when* the LLM judge wakes
up. The philosophy (learned from MINDLAS): deterministic signals decide when
to wake the LLM; the LLM only produces low-friction, actionable guidance.
"""

from .detectors import (
    BindingDriftDetector,
    ContextRotDetector,
    CUSUMDriftDetector,
    Detector,
    LoopDetector,
    OffSpecDetector,
    RegressionDetector,
)
from .rulebook import Rulebook, adherence_after_nudge
from .signals import FAILED, OK, UNKNOWN, classify

__all__ = [
    "FAILED",
    "OK",
    "UNKNOWN",
    "BindingDriftDetector",
    "CUSUMDriftDetector",
    "ContextRotDetector",
    "Detector",
    "LoopDetector",
    "OffSpecDetector",
    "RegressionDetector",
    "Rulebook",
    "adherence_after_nudge",
    "classify",
]