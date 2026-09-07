"""Dependency-free metric definitions for auditable small evaluation fixtures."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def macro_f1(predictions: Sequence[str], labels: Sequence[str], classes: Sequence[str] = ("CALL", "NO_CALL", "CLARIFY")) -> dict:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have equal length")
    per_class: dict[str, dict[str, float]] = {}
    for label in classes:
        tp = sum(prediction == label and truth == label for prediction, truth in zip(predictions, labels))
        fp = sum(prediction == label and truth != label for prediction, truth in zip(predictions, labels))
        fn = sum(prediction != label and truth == label for prediction, truth in zip(predictions, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return {"macro_f1": sum(item["f1"] for item in per_class.values()) / len(classes), "per_class": per_class, "support": len(labels)}


def false_call_rate(predictions: Sequence[str], labels: Sequence[str]) -> float:
    non_call = sum(label != "CALL" for label in labels)
    return sum(prediction == "CALL" and label != "CALL" for prediction, label in zip(predictions, labels)) / non_call if non_call else 0.0


def recall_at_k(rankings: Sequence[Sequence[str]], relevant: Sequence[set[str]], k: int = 8) -> float:
    values = []
    for ranking, gold in zip(rankings, relevant):
        if not gold:
            continue
        values.append(len(set(ranking[:k]) & gold) / len(gold))
    return sum(values) / len(values) if values else float("nan")


def ndcg(ranking: Sequence[str], relevance: dict[str, int], k: int = 10) -> float:
    actual = sum((2 ** relevance.get(item, 0) - 1) / math.log2(index + 2) for index, item in enumerate(ranking[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    maximum = sum((2 ** value - 1) / math.log2(index + 2) for index, value in enumerate(ideal))
    return actual / maximum if maximum else float("nan")
