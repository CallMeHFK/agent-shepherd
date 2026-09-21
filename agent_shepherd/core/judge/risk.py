"""Risk-controlled admission thresholds for interventions.

The policy config exposed ``nudge_threshold`` / ``block_threshold`` /
``max_guidance_tokens`` as user-facing knobs, but nothing read them: a judge
verdict was applied verbatim and its confidence was written to the ledger and
otherwise ignored. The literature is direct about why a hand-picked score
threshold is the wrong instrument here — *Calibration Is Not Control*
(arXiv 2606.21399) argues oversight should be controlled on the *effect of the
intervention*, and verbalized LLM confidence is a poor score to threshold on
its own (arXiv 2412.14737).

So the thresholds are calibrated instead of guessed, using split conformal risk
control: given labeled (score, was-this-really-drift) examples, pick the most
permissive threshold whose empirical false-intervention rate stays within a
budget, with the finite-sample (n+1)/n correction so the bound is a bound and
not a point estimate. With no calibration data the configured defaults stand in,
and the config is honest about that.

The guidance budget is enforced here too, because injected text is the actual
cost the agent pays.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _home() -> Path:
    return Path(os.environ.get("SHEPHERD_HOME", str(Path.home() / ".shepherd")))


@dataclass
class CalibrationSet:
    """Labeled scores: how the judge scored, and whether it was right."""

    drift_scores: list[float] = field(default_factory=list)  # label = True
    clean_scores: list[float] = field(default_factory=list)  # label = False

    def add(self, score: float, is_drift: bool) -> None:
        (self.drift_scores if is_drift else self.clean_scores).append(float(score))

    @property
    def n_negative(self) -> int:
        return len(self.clean_scores)

    def empirical_fpr(self, threshold: float) -> float:
        """Fraction of clean examples that would cross ``threshold``."""
        if not self.clean_scores:
            return 0.0
        return sum(1 for s in self.clean_scores if s >= threshold) / len(self.clean_scores)

    def true_positive_rate(self, threshold: float) -> float:
        if not self.drift_scores:
            return 0.0
        return sum(1 for s in self.drift_scores if s >= threshold) / len(self.drift_scores)


def calibrate(
    samples: CalibrationSet,
    target_fpr: float,
    fallback: float,
    min_samples: int = 20,
    grid: tuple[float, float, float] = (0.0, 1.0, 0.01),
) -> float:
    """Most permissive threshold whose finite-sample-corrected FPR fits the budget.

    Risk is monotone non-increasing in the threshold, so scanning the grid from
    the permissive end and stopping at the first threshold that fits is exactly
    "the most sensitive detector the budget allows". The (n+1)/n correction is
    the standard conformal adjustment: with few clean samples a nominal 0 hits
    still cannot certify a zero false-alarm rate, and pretending otherwise is
    how a supervisor ends up trusting an uncalibrated number.
    """
    if samples.n_negative < min_samples:
        return fallback
    lo, hi, step = grid
    correction = (samples.n_negative + 1) / samples.n_negative
    n = samples.n_negative
    threshold = hi
    current = lo
    while current <= hi + 1e-9:
        hits = sum(1 for s in samples.clean_scores if s >= current)
        if (hits / n) * correction <= target_fpr:
            threshold = current
            break
        current += step
    return round(threshold, 4)


@dataclass
class RiskModel:
    """Per-(agent, source) calibrated admission thresholds, persisted on disk."""

    target_fpr: float = 0.05
    min_samples: int = 20
    defaults: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    samples: dict[str, CalibrationSet] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, defaults: dict[str, float], target_fpr: float, min_samples: int = 20) -> RiskModel:
        path = _home() / "risk.json"
        model = cls(target_fpr=target_fpr, min_samples=min_samples, defaults=dict(defaults), path=path)
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return model
            model.thresholds = {k: float(v) for k, v in (raw.get("thresholds") or {}).items()}
            for key, rec in (raw.get("samples") or {}).items():
                cs = CalibrationSet()
                cs.drift_scores = [float(s) for s in rec.get("drift", [])]
                cs.clean_scores = [float(s) for s in rec.get("clean", [])]
                model.samples[key] = cs
        return model

    def threshold_for(self, key: str) -> float:
        """Calibrated threshold for ``key``; the configured default until enough data."""
        if key in self.thresholds:
            return self.thresholds[key]
        return self.defaults.get(key, 1.0)

    def observe(self, key: str, score: float, is_drift: bool) -> None:
        samples = self.samples.setdefault(key, CalibrationSet())
        samples.add(score, is_drift)
        # Keep the persisted record bounded; newest labels matter most.
        for bucket in (samples.drift_scores, samples.clean_scores):
            if len(bucket) > 2000:
                del bucket[: len(bucket) - 2000]

    def refit(self, key: str) -> float:
        samples = self.samples.get(key) or CalibrationSet()
        fallback = self.defaults.get(key, 1.0)
        self.thresholds[key] = calibrate(samples, self.target_fpr, fallback, self.min_samples)
        return self.thresholds[key]

    def save(self) -> None:
        if self.path is None:
            return
        payload: dict[str, Any] = {
            "target_fpr": self.target_fpr,
            "thresholds": self.thresholds,
            "samples": {
                k: {"drift": v.drift_scores[-500:], "clean": v.clean_scores[-500:]}
                for k, v in self.samples.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)


def truncate_guidance(text: str | None, max_tokens: int) -> str | None:
    """Enforce the guidance budget.

    No tokenizer dependency: ~4 characters per token is the usual English/code
    average, and this is a ceiling on injected context, not a billing figure, so
    an estimate with a hard word-level cut is the right trade for zero deps.
    """
    if not text or max_tokens <= 0:
        return text
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text
    cut = text[:char_budget]
    # Do not end mid-word: back off to the last whitespace boundary.
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    if boundary > char_budget // 2:
        cut = cut[:boundary]
    return re.sub(r"\s+$", "", cut) + " …[guidance truncated]"
