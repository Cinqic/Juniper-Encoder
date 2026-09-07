"""Safe-tensor checkpoints with explicit, validated resume state.

No canonical checkpoint path uses ``torch.save`` or an executable pickle. Model
and optimizer tensors are stored in safetensors; scalar and RNG metadata is
stored as canonical JSON. This makes the resume contract inspectable and keeps
untrusted checkpoint bytes from being imported as Python objects.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

from ..errors import EncoderError
from ..utils import canonical_json_bytes, sha256_bytes, sha256_file, write_json
from .protocol import require_safetensors


@dataclasses.dataclass(frozen=True)
class CheckpointMetadata:
    format_version: str
    source_sha: str
    config_hashes: dict[str, str]
    rng_states: dict[str, Any]
    successful_updates: int
    attempted_updates: int
    skipped_updates: int
    consumed_tokens: int
    sampler_identity: str
    tokenizer_identity: str
    data_identity: str
    predecessor_identity: str | None
    epoch: int = 0
    cursor: int = 0
    batch_state: dict[str, Any] = dataclasses.field(default_factory=dict)
    accumulation_state: dict[str, Any] = dataclasses.field(default_factory=dict)
    precision: dict[str, Any] = dataclasses.field(default_factory=dict)
    kernel_settings: dict[str, Any] = dataclasses.field(default_factory=dict)
    environment_lock_identity: str | None = None

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def identity(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def validate_checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "format_version", "source_sha", "config_hashes", "rng_states", "successful_updates",
        "attempted_updates", "skipped_updates", "consumed_tokens", "sampler_identity",
        "tokenizer_identity", "data_identity", "predecessor_identity",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise EncoderError("INVALID_INPUT", "checkpoint metadata is incomplete", {"missing": missing})
    for field in ("successful_updates", "attempted_updates", "skipped_updates", "consumed_tokens"):
        if not isinstance(payload[field], int) or payload[field] < 0:
            raise EncoderError("INVALID_INPUT", "checkpoint counters must be nonnegative integers", {"field": field})
    if payload["successful_updates"] + payload["skipped_updates"] > payload["attempted_updates"]:
        raise EncoderError("INVALID_INPUT", "checkpoint counters are inconsistent")
    return {"status": "VALID", "identity": sha256_bytes(canonical_json_bytes(payload))}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"unsupported metadata value {type(value).__name__}")


def _capture_rng() -> dict[str, Any]:
    result: dict[str, Any] = {"python": _jsonable(random.getstate())}
    try:
        import numpy as np

        state = np.random.get_state()
        result["numpy"] = {"algorithm": state[0], "keys": state[1].tolist(), "position": int(state[2]), "has_gauss": int(state[3]), "cached_gaussian": float(state[4])}
    except ImportError:
        result["numpy"] = None
    return result


def _restore_rng(states: dict[str, Any], tensors: dict[str, Any]) -> None:
    if states.get("python") is not None:
        random.setstate(_to_tuple(states["python"]))
    if states.get("numpy") is not None:
        try:
            import numpy as np

            value = states["numpy"]
            np.random.set_state((value["algorithm"], np.asarray(value["keys"], dtype=np.uint32), int(value["position"]), int(value["has_gauss"]), float(value["cached_gaussian"])))
        except ImportError:
            pass
    import torch

    if "rng/cpu" in tensors:
        torch.set_rng_state(tensors["rng/cpu"].cpu())
    if torch.cuda.is_available():
        cuda_states = [tensors[key].cpu() for key in sorted(tensors) if key.startswith("rng/cuda/")]
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


def _to_tuple(value: Any) -> Any:
    return tuple(_to_tuple(item) for item in value) if isinstance(value, list) else value


def _parameter_names(module: Any) -> dict[Any, str]:
    return {parameter: name for name, parameter in module.named_parameters()}


def save_checkpoint(
    directory: str | Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    metadata: CheckpointMetadata,
    *,
    sampler_state: dict[str, Any] | None = None,
    precision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_torch = require_safetensors()
    import torch

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    model_tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items() if hasattr(tensor, "detach")}
    safe_torch.save_file(model_tensors, str(destination / "model.safetensors"), metadata={"format": "juniper-model-v2"})

    names = _parameter_names(model)
    optimizer_tensors: dict[str, Any] = {}
    optimizer_entries: list[dict[str, Any]] = []
    for index, (parameter, state) in enumerate(optimizer.state.items()):
        name = names.get(parameter)
        if name is None:
            raise EncoderError("INVALID_INPUT", "optimizer contains a parameter absent from the model")
        entry = {"index": index, "parameter": name, "tensor_keys": []}
        for state_name, value in state.items():
            if hasattr(value, "detach"):
                tensor_key = f"optimizer/{index}/{state_name}"
                optimizer_tensors[tensor_key] = value.detach().cpu().contiguous()
                entry["tensor_keys"].append(tensor_key)
            else:
                entry.setdefault("scalars", {})[state_name] = _jsonable(value)
        optimizer_entries.append(entry)
    if optimizer_tensors:
        safe_torch.save_file(optimizer_tensors, str(destination / "optimizer.safetensors"), metadata={"format": "juniper-optimizer-v2"})

    rng_tensors = {"rng/cpu": torch.get_rng_state().cpu()}
    if torch.cuda.is_available():
        rng_tensors.update({f"rng/cuda/{index}": state.cpu() for index, state in enumerate(torch.cuda.get_rng_state_all())})
    safe_torch.save_file(rng_tensors, str(destination / "rng.safetensors"), metadata={"format": "juniper-rng-v2"})

    payload = metadata.payload()
    payload["rng_states"] = _capture_rng()
    payload["sampler_state"] = sampler_state or {}
    payload["precision"] = precision or payload.get("precision", {})
    payload["optimizer_entries"] = optimizer_entries
    payload["scheduler_state"] = _jsonable(scheduler.state_dict()) if scheduler is not None else None
    payload["scaler_state"] = _jsonable(scaler.state_dict()) if scaler is not None else None
    payload["weights_sha256"] = sha256_file(destination / "model.safetensors")
    payload["optimizer_sha256"] = sha256_file(destination / "optimizer.safetensors") if optimizer_tensors else None
    payload["rng_sha256"] = sha256_file(destination / "rng.safetensors")
    validation = validate_checkpoint_metadata(payload)
    result = {"metadata": payload, "checkpoint_identity": validation["identity"], "weights_format": "safetensors", "directory": destination.as_posix()}
    write_json(destination / "metadata.json", result)
    return result


def load_checkpoint(directory: str | Path, model: Any, optimizer: Any | None = None, scheduler: Any | None = None, scaler: Any | None = None, *, restore_rng: bool = True) -> dict[str, Any]:
    safe_torch = require_safetensors()
    import torch

    source = Path(directory)
    wrapper = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    payload = wrapper.get("metadata", wrapper)
    validate_checkpoint_metadata(payload)
    if payload.get("weights_sha256") != sha256_file(source / "model.safetensors"):
        raise EncoderError("INVALID_INPUT", "checkpoint model tensor checksum mismatch")
    model.load_state_dict(safe_torch.load_file(str(source / "model.safetensors"), device="cpu"), strict=True)
    if optimizer is not None and (source / "optimizer.safetensors").exists():
        tensors = safe_torch.load_file(str(source / "optimizer.safetensors"), device="cpu")
        names = _parameter_names(model)
        by_name = {name: parameter for parameter, name in names.items()}
        optimizer.state.clear()
        for entry in payload.get("optimizer_entries", []):
            parameter = by_name.get(entry["parameter"])
            if parameter is None:
                raise EncoderError("INVALID_INPUT", "checkpoint optimizer parameter is absent from model", {"parameter": entry["parameter"]})
            state: dict[str, Any] = {key.rsplit("/", 1)[-1]: tensors[key].to(parameter.device) for key in entry.get("tensor_keys", [])}
            state.update(entry.get("scalars", {}))
            optimizer.state[parameter] = state
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng and (source / "rng.safetensors").exists():
        _restore_rng(payload.get("rng_states", {}), safe_torch.load_file(str(source / "rng.safetensors"), device="cpu"))
    return {"status": "LOADED", "checkpoint_identity": wrapper.get("checkpoint_identity"), "metadata": payload}


def write_metadata(path: str | Path, metadata: CheckpointMetadata) -> dict[str, Any]:
    payload = metadata.payload()
    result = {"metadata": payload, "checkpoint_identity": metadata.identity(), "weights_format": "safetensors"}
    write_json(path, result)
    return result
