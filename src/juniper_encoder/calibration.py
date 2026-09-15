"""Calibration utilities with explicit pre-calibration state."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .errors import EncoderError, numerical_error
from .evaluation import brier_score, expected_calibration_error


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int], *, lower: float = 0.05, upper: float = 20.0) -> dict:
    if len(logits) != len(labels) or not logits:
        raise EncoderError("INVALID_INPUT", "temperature calibration needs matched nonempty logits and labels")
    if lower <= 0 or upper < lower:
        raise EncoderError("INVALID_INPUT", "temperature search interval is invalid")

    def nll(temperature: float) -> float:
        total = 0.0
        for row, label in zip(logits, labels):
            values = [float(value) / temperature for value in row]
            if not all(math.isfinite(value) for value in values) or not 0 <= label < len(values):
                raise numerical_error("nonfinite calibration logit or invalid label")
            maximum = max(values)
            total += math.log(sum(math.exp(value - maximum) for value in values)) + maximum - values[label]
        return total / len(labels)

    # Deterministic bounded grid plus local refinement. This is a protocol
    # implementation, not an assertion that calibration improves argmax.
    grid = [lower * (upper / lower) ** (index / 200) for index in range(201)]
    best = min(grid, key=nll)
    for _ in range(24):
        step = (upper - lower) / 200
        candidates = [max(lower, best - step), best, min(upper, best + step)]
        best = min(candidates, key=nll)
        step /= 2

    def probabilities(temperature: float) -> list[list[float]]:
        result: list[list[float]] = []
        for row in logits:
            values = [float(value) / temperature for value in row]
            maximum = max(values)
            exponentials = [math.exp(value - maximum) for value in values]
            total = math.fsum(exponentials)
            result.append([value / total for value in exponentials])
        return result

    pre_probabilities = probabilities(1.0)
    post_probabilities = probabilities(best)
    pre_ece = expected_calibration_error(pre_probabilities, labels, bins=15)
    post_ece = expected_calibration_error(post_probabilities, labels, bins=15)
    return {
        "temperature": best, "pre_nll": nll(1.0), "post_nll": nll(best),
        "pre_brier": brier_score(pre_probabilities, labels), "post_brier": brier_score(post_probabilities, labels),
        "pre_ece": pre_ece["ece"], "post_ece": post_ece["ece"],
        "pre_reliability_bins": pre_ece["bins"], "post_reliability_bins": post_ece["bins"],
        "ece_bins": 15, "interval": [lower, upper], "converged": True, "status": "CALIBRATED",
    }


def select_thresholds(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select conditional-routing thresholds from threshold-calibration rows.

    Rows contain retrieval ``score``, ``margin``, a boolean ``skip_safe`` and
    optionally an ``always_correct``/``conditional_correct`` result.  The
    deterministic observed grid never inspects sealed-test rows.
    """
    if not samples:
        raise EncoderError("INVALID_INPUT", "threshold calibration needs nonempty samples")
    for row in samples:
        if not all(key in row for key in ("score", "margin", "skip_safe")):
            raise EncoderError("INVALID_INPUT", "threshold rows need score, margin, and skip_safe")
        if not math.isfinite(float(row["score"])) or not math.isfinite(float(row["margin"])):
            raise numerical_error("threshold calibration contains a nonfinite score")
    scores = sorted({float(row["score"]) for row in samples})
    margins = sorted({float(row["margin"]) for row in samples})
    # Include finite endpoints so equality behavior is evaluated explicitly.
    score_grid = [scores[0] - 1.0, *scores, scores[-1] + 1.0]
    margin_grid = [margins[0] - 1.0, *margins, margins[-1] + 1.0]
    best: tuple[float, float, int, int] | None = None
    for score_threshold in score_grid:
        for margin_threshold in margin_grid:
            skipped = [row for row in samples if float(row["score"]) >= score_threshold and float(row["margin"]) >= margin_threshold]
            safe = sum(bool(row["skip_safe"]) for row in skipped)
            # Maximize safe skips, then prefer the stricter/higher thresholds.
            candidate = (safe / len(samples), score_threshold, margin_threshold, len(skipped))
            key = (candidate[0], candidate[1], candidate[2], -candidate[3])
            if best is None or key > (best[0], best[1], best[2], -best[3]):
                best = (candidate[0], score_threshold, margin_threshold, len(skipped))
    assert best is not None
    return {"status": "CALIBRATED", "score_threshold": best[1], "margin_threshold": best[2], "safe_skip_rate": best[0], "calibration_support": len(samples), "skipped_support": best[3], "search": "observed-score-grid-with-finite-endpoints"}


def calibration_manifest(*, temperature: dict[str, Any], thresholds: dict[str, Any], checkpoint_identity: str, tokenizer_identity: str, threshold_subset_identity: str, temperature_subset_identity: str, registry_snapshot_identity: str, implementation_sha: str) -> dict[str, Any]:
    if temperature.get("status") != "CALIBRATED" or thresholds.get("status") != "CALIBRATED":
        raise EncoderError("BLOCKED_CALIBRATION", "both threshold and temperature calibration must complete")
    return {
        "status": "CALIBRATED", "temperature": temperature["temperature"],
        "tau_score": thresholds["score_threshold"], "tau_margin": thresholds["margin_threshold"],
        "pre_nll": temperature["pre_nll"], "post_nll": temperature["post_nll"],
        "pre_brier": temperature["pre_brier"], "post_brier": temperature["post_brier"],
        "pre_ece": temperature["pre_ece"], "post_ece": temperature["post_ece"],
        "pre_reliability_bins": temperature["pre_reliability_bins"], "post_reliability_bins": temperature["post_reliability_bins"],
        "temperature_converged": bool(temperature.get("converged")),
        "checkpoint_identity": checkpoint_identity, "tokenizer_identity": tokenizer_identity,
        "threshold_subset_identity": threshold_subset_identity, "temperature_subset_identity": temperature_subset_identity,
        "registry_snapshot_identity": registry_snapshot_identity, "implementation_sha": implementation_sha,
    }
