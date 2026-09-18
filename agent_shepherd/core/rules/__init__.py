"""Tier 0 deterministic detectors.

These run on every event, cost nothing, and decide *when* the LLM judge wakes
up. The philosophy (learned from MINDLAS): deterministic signals decide when
to wake the LLM; the LLM only produces low-friction, actionable guidance.
"""

from .detectors import (
    ContextRotDetector,
    Detector,
    LoopDetector,
    OffSpecDetector,
    RegressionDetector,
)

__all__ = [
    "ContextRotDetector",
    "Detector",
    "LoopDetector",
    "OffSpecDetector",
    "RegressionDetector",
]