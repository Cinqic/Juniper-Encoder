"""Safe deployment export bookkeeping; never pretends an absent checkpoint exists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import make_sha256sums
from .constants import PARAMETER_COUNTS
from .errors import EncoderError
from .utils import load_config, write_json


def export_checkpoint(checkpoint: str | Path, output: str | Path, variant: str) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise EncoderError("BLOCKED_CHECKPOINT", "selected checkpoint is not materialized", {"checkpoint": str(checkpoint_path)})
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    result = {
        "artifact_type": "juniper-deployment-export",
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_identity": metadata.get("checkpoint_identity"),
        "status": "EXPORTED_METADATA_ONLY",
        "parameter_count": PARAMETER_COUNTS["reranked_pipeline_deployed_total"],
        "weights": "blocked-until-safe-tensor-payload-is-materialized",
    }
    write_json(destination / "export.json", result)
    make_sha256sums(destination, destination / "SHA256SUMS")
    return result


def verify_export(directory: str | Path) -> dict[str, Any]:
    export_path = Path(directory) / "export.json"
    if not export_path.exists():
        raise EncoderError("BLOCKED_CHECKPOINT", "export metadata is absent")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    if payload.get("status") != "EXPORTED_METADATA_ONLY":
        raise ValueError("unexpected export status")
    return {"status": "BLOCKED_EXPORT_PAYLOAD", "reason": "safe tensor weights are not materialized"}
