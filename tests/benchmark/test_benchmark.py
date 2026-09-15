from juniper_encoder.benchmark import measure_case, required_cases


def test_required_benchmark_matrix_has_both_variants_and_all_shapes():
    config = {"batch_sizes": [1, 8], "padded_lengths": [128, 512, 1024]}
    assert len(required_cases(config, "reference")) == 42
    assert len(required_cases(config, "int8")) == 42


def test_measure_case_uses_nearest_rank_and_preserves_raw_samples():
    result = measure_case(lambda: None, warmup_runs=2, measured_runs=5)
    assert result["warmup_runs"] == 2
    assert result["measured_runs"] == 5
    assert len(result["raw_latency_ms"]) == 5
    assert result["p50_ms"] <= result["p95_ms"]
