"""Matched INT8 gate calculations. A missing measurement is never a pass."""

from __future__ import annotations

from typing import Any


def compare_metrics(floating: dict[str, float], int8: dict[str, float], gates: dict[str, float]) -> dict[str, Any]:
    metric_losses = {}
    failures = []
    for metric, tolerance in (
        ("shortlist_recall_at_8", 0.01), ("retrieval_recall_at_10", 0.01), ("retrieval_ndcg_at_10", 0.01),
        ("reranker_ndcg_at_10", 0.01), ("reranker_mrr_at_10", 0.01), ("decision_macro_f1", 0.01),
        ("call_recall", 0.01), ("end_to_end_accuracy", 0.01),
    ):
        if metric not in floating or metric not in int8:
            failures.append({"metric": metric, "reason": "missing matched measurement"})
            continue
        loss = floating[metric] - int8[metric]
        metric_losses[metric] = loss
        if loss > tolerance:
            failures.append({"metric": metric, "loss": loss, "tolerance": tolerance})
    if "false_call_rate" in floating and "false_call_rate" in int8:
        increase = int8["false_call_rate"] - floating["false_call_rate"]
        metric_losses["false_call_rate_increase"] = increase
        if increase > 0.005:
            failures.append({"metric": "false_call_rate", "increase": increase, "tolerance": 0.005})
    else:
        failures.append({"metric": "false_call_rate", "reason": "missing matched measurement"})
    if "ece" in floating and "ece" in int8:
        increase = int8["ece"] - floating["ece"]
        metric_losses["ece_increase"] = increase
        if increase > 0.01:
            failures.append({"metric": "ece", "increase": increase, "tolerance": 0.01})
    else:
        failures.append({"metric": "ece", "reason": "missing matched measurement"})
    for metric, ratio_limit in (("warm_p50_latency_ratio", 1.05), ("warm_p95_latency_ratio", 1.05), ("artifact_size_ratio", 0.65), ("peak_memory_ratio", 1.0)):
        if metric not in int8:
            failures.append({"metric": metric, "reason": "missing matched measurement"})
        elif (metric in {"artifact_size_ratio", "peak_memory_ratio"} and int8[metric] > ratio_limit) or (metric.startswith("warm_") and int8[metric] > ratio_limit):
            failures.append({"metric": metric, "ratio": int8[metric], "limit": ratio_limit})
    if int8.get("numerical_failures", 1) != 0:
        failures.append({"metric": "numerical_validity", "reason": "one or more numerical failures"})
    return {"status": "INT8_ACCEPTED" if not failures else "INT8_REJECTED", "losses": metric_losses, "failures": failures}
