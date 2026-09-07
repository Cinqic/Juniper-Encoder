"""Materialize and verify a non-executable deployment export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import make_sha256sums, verify_sha256sums
from .constants import PARAMETER_COUNTS
from .errors import EncoderError
from .model import DeploymentModel, actual_parameter_counts, require_torch
from .utils import canonical_json_bytes, load_config, sha256_file, write_json


def _resolve_checkpoint(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path)
    if source.is_file():
        payload = json.loads(source.read_text(encoding="utf-8"))
        directory = payload.get("checkpoint_directory") or payload.get("directory")
        if directory is None:
            directory = source.parent / payload.get("checkpoint", "")
        source = Path(directory)
    metadata_path = source / "metadata.json"
    if not metadata_path.exists():
        raise EncoderError("BLOCKED_CHECKPOINT", "checkpoint metadata is absent", {"checkpoint": source.as_posix()})
    return source, json.loads(metadata_path.read_text(encoding="utf-8"))


def export_checkpoint(checkpoint: str | Path, output: str | Path, variant: str, *, tokenizer_directory: str | Path | None = None, model_config: str | Path | None = None, calibration: str | Path | None = None) -> dict[str, Any]:
    require_torch()
    try:
        from safetensors import torch as safe_torch
    except ImportError as exc:
        raise EncoderError("BLOCKED_ENVIRONMENT", "safe tensor serialization is required for deployment export", {"dependency": "safetensors"}) from exc
    checkpoint_dir, checkpoint_payload = _resolve_checkpoint(checkpoint)
    checkpoint_metadata = checkpoint_payload.get("metadata", checkpoint_payload)
    metadata_file = checkpoint_dir / "metadata.json"
    if metadata_file.exists():
        metadata_wrapper = json.loads(metadata_file.read_text(encoding="utf-8"))
        checkpoint_metadata = metadata_wrapper.get("metadata", metadata_wrapper)
    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.exists():
        raise EncoderError("BLOCKED_CHECKPOINT", "checkpoint does not contain safe deployment weights", {"checkpoint": checkpoint_dir.as_posix()})
    state = safe_torch.load_file(str(weights_path), device="cpu")
    if not any(key.startswith("backbone.") for key in state):
        raise EncoderError("BLOCKED_CHECKPOINT", "selected checkpoint contains backbone-only or MLM weights; downstream heads are required")
    model = DeploymentModel(seed=1729)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise EncoderError("BLOCKED_CHECKPOINT", "selected checkpoint state keys do not match the deployment model", {"reason": str(exc)}) from exc
    model.eval()
    counts = actual_parameter_counts(model)
    if counts["physical_named_parameters"] != PARAMETER_COUNTS["reranked_pipeline_deployed_total"]:
        raise EncoderError("BLOCKED_CHECKPOINT", "selected checkpoint parameter count is not the frozen deployment count", counts)
    if not tokenizer_directory or not (Path(tokenizer_directory) / "tokenizer.json").exists():
        raise EncoderError("BLOCKED_CHECKPOINT", "deployment export requires the promoted tokenizer directory")
    if calibration and not Path(calibration).exists():
        raise EncoderError("BLOCKED_CALIBRATION", "calibration manifest is absent")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    deployment_state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    safe_torch.save_file(deployment_state, str(destination / "model.safetensors"), metadata={"format": "juniper-deployment-v2", "variant": variant})
    config = model.backbone.config
    config_payload = load_config(model_config) if model_config else {"vocab_size": config.vocab_size, "blocks": config.blocks, "width": config.width, "q_heads": config.q_heads, "kv_heads": config.kv_heads, "head_dim": config.head_dim, "geglu_width": config.geglu_width, "rope_theta": config.rope_theta, "layer_norm_epsilon": config.layer_norm_epsilon, "dropout": config.dropout, "native_context": config.native_context}
    write_json(destination / "model_config.json", config_payload)
    if tokenizer_directory:
        tokenizer_source = Path(tokenizer_directory)
        shutil.copytree(tokenizer_source, destination / "tokenizer", dirs_exist_ok=True)
    else:
        raise EncoderError("BLOCKED_CHECKPOINT", "deployment export requires the promoted tokenizer directory")
    if calibration:
        calibration_source = Path(calibration)
        shutil.copy2(calibration_source, destination / "calibration.json")
    else:
        write_json(destination / "calibration.json", {"status": "NOT_CALIBRATED"})
    metadata = {
        "artifact_type": "juniper-deployment-export", "format_version": "juniper-export-v2", "variant": variant,
        "checkpoint_identity": checkpoint_payload.get("checkpoint_identity"), "checkpoint_directory": checkpoint_dir.as_posix(),
        "weights_sha256": sha256_file(destination / "model.safetensors"), "parameter_counts": counts,
        "tokenizer_identity": checkpoint_metadata.get("tokenizer_identity"),
        "status": "EXPORTED", "safe_weights": True,
    }
    write_json(destination / "export.json", metadata)
    make_sha256sums(destination, destination / "SHA256SUMS")
    return metadata


def verify_export(directory: str | Path) -> dict[str, Any]:
    require_torch()
    try:
        from safetensors import torch as safe_torch
    except ImportError as exc:
        raise EncoderError("BLOCKED_ENVIRONMENT", "safe tensor serialization is required for export verification", {"dependency": "safetensors"}) from exc
    root = Path(directory)
    export_path = root / "export.json"
    weights_path = root / "model.safetensors"
    if not export_path.exists() or not weights_path.exists():
        raise EncoderError("BLOCKED_EXPORT_PAYLOAD", "deployment export requires export.json and model.safetensors")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    if payload.get("status") != "EXPORTED" or payload.get("safe_weights") is not True:
        raise EncoderError("BLOCKED_EXPORT_PAYLOAD", "export is not a verified safe deployment payload")
    if payload.get("weights_sha256") != sha256_file(weights_path):
        raise EncoderError("BLOCKED_EXPORT_PAYLOAD", "deployment weights checksum mismatch")
    state = safe_torch.load_file(str(weights_path), device="cpu")
    model = DeploymentModel(seed=1729)
    model.load_state_dict(state, strict=True)
    counts = actual_parameter_counts(model)
    if counts["pad_row_max_abs"] != 0.0 or counts["physical_named_parameters"] != PARAMETER_COUNTS["reranked_pipeline_deployed_total"]:
        raise EncoderError("BLOCKED_EXPORT_PAYLOAD", "deployment payload failed parameter or PAD checks", counts)
    if not (root / "tokenizer/tokenizer.json").exists():
        raise EncoderError("BLOCKED_EXPORT_PAYLOAD", "deployment payload has no tokenizer")
    sums = root / "SHA256SUMS"
    if not sums.exists():
        raise EncoderError("BLOCKED_EXPORT_PAYLOAD", "deployment payload has no checksum manifest")
    verify_sha256sums(root, sums)
    return {"status": "PASS", "variant": payload.get("variant"), "parameter_counts": counts, "files": len(safe_torch.load_file(str(weights_path), device="cpu"))}


def load_float_export(directory: str | Path, *, device: str = "cpu") -> Any:
    """Load a verified floating export as an executable benchmark runner."""
    verify_export(directory)
    from safetensors.torch import load_file
    from .benchmark import make_model_runner

    model = DeploymentModel(seed=1729)
    model.load_state_dict(load_file(str(Path(directory) / "model.safetensors"), device="cpu"), strict=True)
    return make_model_runner(model, device=device)
