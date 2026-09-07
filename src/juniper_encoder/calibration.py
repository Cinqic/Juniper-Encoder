"""Calibration utilities with explicit pre-calibration state."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .errors import EncoderError, numerical_error


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
    return {"temperature": best, "pre_nll": nll(1.0), "post_nll": nll(best), "interval": [lower, upper], "status": "CALIBRATED"}
