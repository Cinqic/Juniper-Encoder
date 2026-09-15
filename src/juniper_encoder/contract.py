"""Machine-readable contract checks, parameter accounting, and traceability."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .constants import PARAMETER_COUNTS, STRUCTURAL_IDS, VOCAB_SIZE
from .model import ModelConfig, actual_parameter_counts, require_torch
from .utils import canonical_json_bytes, sha256_file, write_json


def verify_machine_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "vocab_size": VOCAB_SIZE,
        "structural_ids": STRUCTURAL_IDS,
        "parameter_counts": PARAMETER_COUNTS,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"machine contract field {key} differs from frozen values")
    config = ModelConfig(**payload["model"])
    config.validate()
    root = contract_path.parent
    pinned_files = {"frozen_spec_sha256": root / "PROPOSAL_B_FROZEN.md", "implementation_brief_sha256": root / "IMPLEMENTATION_REVIEW_BRIEF.md"}
    for field, candidate in pinned_files.items():
        if field in payload:
            if not candidate.exists() or sha256_file(candidate) != payload[field]:
                raise ValueError(f"machine contract field {field} does not match the pinned repository bytes")
    return {"valid": True, "contract_sha256": sha256_file(path), "fields": sorted(payload)}


def audit_model(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = ModelConfig()
    config.validate()
    try:
        from .model import DeploymentModel

        model = DeploymentModel(config)
        counts = actual_parameter_counts(model)
        expected = {
            "shared_backbone": PARAMETER_COUNTS["shared_backbone"],
            "retrieval_participating": PARAMETER_COUNTS["retrieval_participating"],
            "decision_participating": PARAMETER_COUNTS["decision_participating"],
            "reranking_participating": PARAMETER_COUNTS["reranking_participating"],
            "fast_pipeline_unique": PARAMETER_COUNTS["fast_pipeline_unique"],
            "reranked_pipeline_deployed_total": PARAMETER_COUNTS["reranked_pipeline_deployed_total"],
        }
        mismatches = {key: {"actual": counts[key], "expected": value} for key, value in expected.items() if counts[key] != value}
        result = {"status": "PASS" if not mismatches else "FAIL", "counts": counts, "expected": expected, "mismatches": mismatches}
    except Exception as exc:
        if getattr(exc, "code", None) == "BLOCKED_ENVIRONMENT":
            result = {"status": "BLOCKED_ENVIRONMENT", "message": str(exc), "expected": PARAMETER_COUNTS}
        else:
            raise
    write_json(output, result)
    return result

def verify_traceability(path: str | Path, *, spec_sha256: str | None = None) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    required = {"requirement_id", "proposal_section", "exact_text", "implementation_symbol", "config_field", "test_id", "expected_result", "evidence_path_hash", "status"}
    if rows and set(rows[0]) != required:
        raise ValueError("traceability columns do not match the frozen audit schema")
    ids = [row["requirement_id"] for row in rows]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    missing = [row["requirement_id"] for row in rows if not all(row.get(field) for field in ("implementation_symbol", "test_id", "expected_result", "status"))]
    result = {"status": "PASS" if not duplicate_ids and not missing else "FAIL", "rows": len(rows), "duplicate_ids": duplicate_ids, "missing_fields": missing, "spec_sha256": spec_sha256}
    if duplicate_ids or missing:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result
