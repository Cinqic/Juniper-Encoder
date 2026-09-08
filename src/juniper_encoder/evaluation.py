"""Dependency-free metric definitions for auditable small evaluation fixtures."""

from __future__ import annotations

import hashlib
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


def mrr(ranking: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for index, item in enumerate(ranking[:k], 1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def brier_score(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must be matched and nonempty")
    total = 0.0
    for row, label in zip(probabilities, labels):
        values = [float(value) for value in row]
        if not 0 <= label < len(values) or not values or not all(math.isfinite(value) for value in values):
            raise ValueError("probabilities contain an invalid row or label")
        if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("Brier score requires rows of nonnegative probabilities summing to one")
        total += sum((value - (1.0 if index == label else 0.0)) ** 2 for index, value in enumerate(values))
    return total / len(labels)


def expected_calibration_error(probabilities: Sequence[Sequence[float]], labels: Sequence[int], bins: int = 15) -> dict[str, object]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must be matched and nonempty")
    counts = [0] * bins
    confidence = [0.0] * bins
    accuracy = [0.0] * bins
    for row, label in zip(probabilities, labels):
        values = [float(value) for value in row]
        if not values or not all(math.isfinite(value) for value in values) or not 0 <= label < len(values):
            raise ValueError("probabilities contain an invalid row or label")
        if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("calibration error requires rows of nonnegative probabilities summing to one")
        prediction = max(range(len(values)), key=lambda index: (values[index], -index))
        confidence_value = values[prediction]
        bucket = min(bins - 1, int(max(0.0, min(1.0, confidence_value)) * bins))
        counts[bucket] += 1
        confidence[bucket] += confidence_value
        accuracy[bucket] += float(prediction == label)
    result_bins = []
    ece = 0.0
    for index, support in enumerate(counts):
        item = {"bin": index, "lower": index / bins, "upper": (index + 1) / bins, "support": support, "confidence": confidence[index] / support if support else 0.0, "accuracy": accuracy[index] / support if support else 0.0}
        result_bins.append(item)
        ece += abs(item["confidence"] - item["accuracy"]) * support / len(labels)
    return {"ece": ece, "bins": result_bins, "support": len(labels)}


def classification_report(predictions: Sequence[str], labels: Sequence[str], classes: Sequence[str] = ("CALL", "NO_CALL", "CLARIFY")) -> dict[str, object]:
    report = macro_f1(predictions, labels, classes)
    report["call_recall"] = report["per_class"].get("CALL", {}).get("recall", 0.0)
    report["false_call_rate"] = false_call_rate(predictions, labels)
    report["call_support"] = sum(label == "CALL" for label in labels)
    report["non_call_support"] = sum(label != "CALL" for label in labels)
    return report


def evaluate_predictions(rows: Sequence[dict[str, object]], *, ece_bins: int = 15, _include_slices: bool = True) -> dict[str, object]:
    """Compute the auditable metric bundle from materialized raw predictions.

    Each row may contain ``ranking``, ``relevant``, ``relevance``, ``prediction``,
    ``label``, ``logits``, and ``exact``. Missing optional component fields are
    reported as unavailable instead of silently changing denominators.
    """
    result: dict[str, object] = {"support": len(rows), "denominators": {"all_rows": len(rows)}}
    retrieval_rows = [row for row in rows if isinstance(row.get("ranking"), list) and isinstance(row.get("relevant"), (list, set, tuple))]
    if retrieval_rows:
        rankings = [row["ranking"] for row in retrieval_rows]
        relevant = [set(row["relevant"]) for row in retrieval_rows]
        result["shortlist_recall_at_8"] = recall_at_k(rankings, relevant, 8)
        result["retrieval_recall_at_10"] = recall_at_k(rankings, relevant, 10)
        result["any_hit_at_8"] = sum(bool(set(ranking[:8]) & gold) for ranking, gold in zip(rankings, relevant)) / len(retrieval_rows)
        result["denominators"]["retrieval"] = len(retrieval_rows)
        ndcg_values, mrr_values = [], []
        for row, gold in zip(retrieval_rows, relevant):
            relevance = row.get("relevance", {item: int(item in gold) for item in row["ranking"]})
            ndcg_values.append(ndcg(row["ranking"], relevance, 10))
            mrr_values.append(mrr(row["ranking"], gold, 10))
        result["retrieval_ndcg_at_10"] = sum(ndcg_values) / len(ndcg_values)
        result["retrieval_mrr_at_10"] = sum(mrr_values) / len(mrr_values)
        reranked_rows = [row for row in retrieval_rows if isinstance(row.get("reranked_ranking"), list)]
        if reranked_rows:
            rerank_ndcg = []
            rerank_mrr = []
            for row in reranked_rows:
                gold = set(row["relevant"])
                relevance = row.get("reranker_relevance", {item: int(item in gold) for item in row["reranked_ranking"]})
                rerank_ndcg.append(ndcg(row["reranked_ranking"], relevance, 10))
                rerank_mrr.append(mrr(row["reranked_ranking"], gold, 10))
            result["reranker_ndcg_at_10"] = sum(rerank_ndcg) / len(rerank_ndcg)
            result["reranker_mrr_at_10"] = sum(rerank_mrr) / len(rerank_mrr)
            result["denominators"]["reranker"] = len(reranked_rows)
    classification_rows = [row for row in rows if isinstance(row.get("prediction"), str) and isinstance(row.get("label"), str)]
    if classification_rows:
        predictions = [row["prediction"] for row in classification_rows]
        labels = [row["label"] for row in classification_rows]
        result.update({"decision_macro_f1": classification_report(predictions, labels)["macro_f1"], "per_class": classification_report(predictions, labels)["per_class"], "call_recall": classification_report(predictions, labels)["call_recall"], "false_call_rate": classification_report(predictions, labels)["false_call_rate"]})
        result["denominators"]["classification"] = len(classification_rows)
    probability_rows = [row for row in rows if isinstance(row.get("logits"), (list, tuple)) and isinstance(row.get("label_index"), int)]
    if probability_rows:
        probabilities = [row["probabilities"] for row in probability_rows if "probabilities" in row]
        labels = [row["label_index"] for row in probability_rows]
        probability_values = [list(row["probabilities"]) for row in probability_rows] if all(isinstance(row.get("probabilities"), (list, tuple)) for row in probability_rows) else _softmax_rows([row["logits"] for row in probability_rows])
        calibration = expected_calibration_error(probability_values, labels, bins=ece_bins)
        result.update({"nll": _nll(probabilities, labels), "brier": brier_score(probability_values, labels), "ece": calibration["ece"], "ece_bins": calibration["bins"]})
        result["denominators"]["calibration"] = len(probability_rows)
    exact_rows = [row for row in rows if isinstance(row.get("exact"), bool)]
    if exact_rows:
        result["end_to_end_exact_accuracy"] = sum(bool(row["exact"]) for row in exact_rows) / len(exact_rows)
        result["end_to_end_accuracy"] = result["end_to_end_exact_accuracy"]
        result["denominators"]["exact"] = len(exact_rows)
    path_rows = [row for row in rows if isinstance(row.get("rerank_invoked"), bool)]
    if path_rows:
        result["rerank_invocation_rate"] = sum(bool(row["rerank_invoked"]) for row in path_rows) / len(path_rows)
        result["denominators"]["path"] = len(path_rows)
    paired_rows = [row for row in rows if isinstance(row.get("conditional_exact"), bool) and isinstance(row.get("always_rerank_exact"), bool)]
    if paired_rows:
        conditional_accuracy = sum(bool(row["conditional_exact"]) for row in paired_rows) / len(paired_rows)
        always_accuracy = sum(bool(row["always_rerank_exact"]) for row in paired_rows) / len(paired_rows)
        conditional_false_calls = [row.get("conditional_false_call") for row in paired_rows if isinstance(row.get("conditional_false_call"), bool)]
        always_false_calls = [row.get("always_rerank_false_call") for row in paired_rows if isinstance(row.get("always_rerank_false_call"), bool)]
        result["fast_path_exact_accuracy"] = conditional_accuracy
        result["always_rerank_exact_accuracy"] = always_accuracy
        result["conditional_accuracy_difference"] = conditional_accuracy - always_accuracy
        if len(conditional_false_calls) == len(paired_rows) and len(always_false_calls) == len(paired_rows):
            result["conditional_false_call_difference"] = (sum(conditional_false_calls) - sum(always_false_calls)) / len(paired_rows)
        result["denominators"]["paired_paths"] = len(paired_rows)
    if _include_slices:
        slice_rows: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            declared = row.get("slices", row.get("slice", []))
            if isinstance(declared, str):
                declared = [declared]
            if isinstance(declared, (list, tuple, set)):
                for name in declared:
                    if isinstance(name, str) and name:
                        slice_rows.setdefault(name, []).append(row)
        if slice_rows:
            result["slices"] = {name: evaluate_predictions(values, ece_bins=ece_bins, _include_slices=False) for name, values in sorted(slice_rows.items())}
    return result


def _softmax_rows(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    output = []
    for row in rows:
        maximum = max(float(value) for value in row)
        values = [math.exp(float(value) - maximum) for value in row]
        total = sum(values)
        output.append([value / total for value in values])
    return output


def _nll(rows: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    total = 0.0
    for row, label in zip(rows, labels):
        values = [float(value) for value in row]
        maximum = max(values)
        total += math.log(sum(math.exp(value - maximum) for value in values)) + maximum - values[label]
    return total / len(labels)


def bootstrap_mean(values: Sequence[float], *, groups: Sequence[str] | None = None, samples: int = 1000, seed: int = 1729) -> dict[str, float | int]:
    if not values:
        raise ValueError("bootstrap requires nonempty values")
    if groups is None:
        groups = [str(index) for index in range(len(values))]
    if len(groups) != len(values):
        raise ValueError("bootstrap groups and values must match")
    grouped: dict[str, list[float]] = {}
    for group, value in zip(groups, values):
        grouped.setdefault(group, []).append(float(value))
    keys = sorted(grouped)
    if len(keys) < 2:
        mean = sum(values) / len(values)
        return {"mean": mean, "lower": mean, "upper": mean, "groups": len(keys), "samples": samples}
    import random

    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [keys[rng.randrange(len(keys))] for _ in keys]
        sampled = [value for key in selected for value in grouped[key]]
        estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    lower_index = max(0, int(0.025 * len(estimates)) - 1)
    upper_index = min(len(estimates) - 1, int(0.975 * len(estimates)))
    return {"mean": sum(values) / len(values), "lower": estimates[lower_index], "upper": estimates[upper_index], "groups": len(keys), "samples": samples}
