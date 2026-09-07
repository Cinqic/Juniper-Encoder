"""Required benchmark matrix description and honest completeness checks."""

from __future__ import annotations

import resource
import statistics
import time
from pathlib import Path
from typing import Any

from .errors import EncoderError
from .utils import load_config, sha256_file, write_json


def required_cases(config: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    components = ["retrieval_encoding", "capability_encoding", "reranker", "decision", "fast_full", "reranked_full", "conditional_full"]
    return [{"variant": variant, "batch_size": batch, "padded_length": length, "component": component}
            for batch in config["batch_sizes"] for length in config["padded_lengths"] for component in components]


def check_coverage(config_path: str | Path, results_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    required = required_cases(config, "reference") + required_cases(config, "int8")
    if results_path is None or not Path(results_path).exists():
        return {"status": "BLOCKED_BENCHMARK", "required_cases": len(required), "missing_raw_samples": len(required), "warmup_runs": config["warmup_runs"], "measured_runs": config["measured_runs"]}
    import json
    actual = json.loads(Path(results_path).read_text(encoding="utf-8"))
    completed = {(item.get("variant"), item.get("batch_size"), item.get("padded_length"), item.get("component")) for item in actual.get("cases", [])}
    missing = [case for case in required if (case["variant"], case["batch_size"], case["padded_length"], case["component"]) not in completed]
    return {"status": "PASS" if not missing else "BLOCKED_BENCHMARK", "required_cases": len(required), "missing": missing, "cases": len(actual.get("cases", []))}


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sample")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(q * len(ordered) + 0.9999999999) - 1))
    return ordered[index]


def measure_case(callable_, *, warmup_runs: int = 20, measured_runs: int = 200, synchronize=None) -> dict[str, Any]:
    """Measure a real callable, synchronizing CUDA when supplied by the caller."""
    for _ in range(warmup_runs):
        callable_()
        if synchronize:
            synchronize()
    samples: list[float] = []
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(measured_runs):
        start = time.perf_counter()
        callable_()
        if synchronize:
            synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mean = statistics.fmean(samples)
    return {"raw_latency_ms": samples, "warmup_runs": warmup_runs, "measured_runs": measured_runs, "p50_ms": _quantile(samples, 0.50), "p95_ms": _quantile(samples, 0.95), "mean_ms": mean, "throughput_per_second": 1000.0 / mean, "process_max_rss_kib": max(rss_before, rss_after)}


def execute_benchmark(config_path: str | Path, runners: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Execute every configured variant/shape/component and preserve failures."""
    config = load_config(config_path)
    cases: list[dict[str, Any]] = []
    for case in required_cases(config, "reference") + required_cases(config, "int8"):
        runner = runners.get(case["variant"])
        item = dict(case)
        if runner is None:
            item.update({"status": "UNSUPPORTED", "error": "runner unavailable"})
        else:
            try:
                item.update(measure_case(lambda: runner(case), warmup_runs=int(config["warmup_runs"]), measured_runs=int(config["measured_runs"]), synchronize=getattr(runner, "synchronize", None)))
                item["status"] = "MEASURED"
            except Exception as exc:
                item.update({"status": "FAILED", "error": repr(exc)})
        cases.append(item)
    result = {"status": "MEASURED" if all(case["status"] == "MEASURED" for case in cases) else "BENCHMARK_INCOMPLETE", "config_sha256": sha256_file(config_path), "cases": cases}
    write_json(output, result)
    return result


def make_model_runner(model: Any, *, device: str = "cpu") -> Any:
    """Build a benchmark callable that exercises the selected model artifact."""
    import torch

    selected_device = torch.device(device)
    model = model.to(selected_device).eval()
    cache: dict[tuple[int, int], Any] = {}

    def runner(case: dict[str, Any]) -> None:
        shape = (int(case["batch_size"]), int(case["padded_length"]))
        if shape not in cache:
            # ID 7 is the first ordinary byte token. Keeping every position
            # ordinary makes this a valid pooled request at every required
            # padded length while retaining the exact benchmark shape.
            cache[shape] = torch.full(shape, 7, dtype=torch.long, device=selected_device)
        input_ids = cache[shape]
        component = case["component"]
        with torch.inference_mode():
            if component in {"retrieval_encoding", "capability_encoding"}:
                model.retrieval_vector(input_ids)
            elif component == "reranker":
                model.rerank(input_ids)
            elif component == "decision":
                model.classify(input_ids)
            elif component == "fast_full":
                model.retrieval_vector(input_ids)
                model.classify(input_ids)
            elif component == "reranked_full":
                model.retrieval_vector(input_ids)
                model.rerank(input_ids)
                model.classify(input_ids)
            elif component == "conditional_full":
                # The shape-only benchmark exercises the fast conditional
                # branch; path decision quality is evaluated separately on
                # paired request/registry rows.
                model.retrieval_vector(input_ids)
                model.classify(input_ids)
            else:
                raise ValueError(f"unknown benchmark component: {component}")

    if selected_device.type == "cuda":
        runner.synchronize = torch.cuda.synchronize
    return runner
