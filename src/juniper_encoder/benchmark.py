"""Required benchmark matrix description and honest completeness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import EncoderError
from .utils import load_config, write_json


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
    return {"status": "PASS" if not missing else "BLOCKED_BENCHMARK", "required_cases": len(required), "missing": missing}
