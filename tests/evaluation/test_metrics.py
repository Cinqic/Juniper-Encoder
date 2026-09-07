from juniper_encoder.evaluation import evaluate_predictions


def test_evaluation_reports_component_denominators_and_conditional_paths():
    result = evaluate_predictions([
        {
            "ranking": ["a", "b"],
            "reranked_ranking": ["a", "b"],
            "relevant": ["a"],
            "prediction": "CALL",
            "label": "CALL",
            "logits": [3.0, 0.0, -1.0],
            "label_index": 0,
            "exact": True,
            "rerank_invoked": False,
            "conditional_exact": True,
            "always_rerank_exact": True,
            "conditional_false_call": False,
            "always_rerank_false_call": False,
            "slices": ["CALL", "tool"],
        }
    ])
    assert result["denominators"] == {"all_rows": 1, "retrieval": 1, "reranker": 1, "classification": 1, "calibration": 1, "exact": 1, "path": 1, "paired_paths": 1}
    assert result["end_to_end_exact_accuracy"] == 1.0
    assert result["rerank_invocation_rate"] == 0.0
    assert result["fast_path_exact_accuracy"] == 1.0
    assert result["slices"]["CALL"]["support"] == 1
