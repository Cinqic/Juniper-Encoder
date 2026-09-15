from juniper_encoder.quantization import compare_metrics


def test_missing_int8_measurement_rejects_qualification():
    result = compare_metrics({"shortlist_recall_at_8": 1.0}, {}, {})
    assert result["status"] == "INT8_REJECTED"
    assert result["failures"]


def test_int8_performance_regression_is_not_hidden():
    floating = {key: 1.0 for key in ("shortlist_recall_at_8", "retrieval_recall_at_10", "retrieval_ndcg_at_10", "reranker_ndcg_at_10", "reranker_mrr_at_10", "decision_macro_f1", "call_recall", "end_to_end_accuracy")}
    floating.update({"false_call_rate": 0.0, "ece": 0.0})
    int8 = dict(floating)
    int8.update({"false_call_rate": 0.006, "ece": 0.011, "warm_p50_latency_ratio": 1.06, "warm_p95_latency_ratio": 1.0, "artifact_size_ratio": 0.65, "peak_memory_ratio": 1.0, "numerical_failures": 0})
    result = compare_metrics(floating, int8, {})
    assert result["status"] == "INT8_REJECTED"


def test_int8_gate_requires_all_required_quality_metrics():
    values = {key: 1.0 for key in ("shortlist_recall_at_8", "retrieval_recall_at_10", "retrieval_ndcg_at_10", "reranker_ndcg_at_10", "reranker_mrr_at_10", "decision_macro_f1", "call_recall", "end_to_end_accuracy")}
    values.update({"false_call_rate": 0.0, "ece": 0.0})
    result = compare_metrics(values, {"shortlist_recall_at_8": 1.0}, {})
    assert result["status"] == "INT8_REJECTED"
    assert any(item["reason"] == "missing matched measurement" for item in result["failures"])
