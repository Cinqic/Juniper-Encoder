"""Matched INT8 gate calculations. A missing measurement is never a pass."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import make_sha256sums, verify_sha256sums
from .errors import EncoderError
from .utils import load_config, sha256_file, write_json


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
    limits = {"warm_p50_latency_ratio": gates.get("p50_ratio", 1.05), "warm_p95_latency_ratio": gates.get("p95_ratio", 1.05), "artifact_size_ratio": gates.get("artifact_size_ratio", 0.65), "peak_memory_ratio": gates.get("peak_memory_ratio", 1.0)}
    for metric, ratio_limit in limits.items():
        if metric not in int8:
            failures.append({"metric": metric, "reason": "missing matched measurement"})
        elif (metric in {"artifact_size_ratio", "peak_memory_ratio"} and int8[metric] > ratio_limit) or (metric.startswith("warm_") and int8[metric] > ratio_limit):
            failures.append({"metric": metric, "ratio": int8[metric], "limit": ratio_limit})
    if int8.get("numerical_failures", 1) != 0:
        failures.append({"metric": "numerical_validity", "reason": "one or more numerical failures"})
    return {"status": "INT8_ACCEPTED" if not failures else "INT8_REJECTED", "losses": metric_losses, "failures": failures}


def _require_torch() -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise EncoderError("BLOCKED_ENVIRONMENT", "INT8 construction requires PyTorch") from exc
    return torch, nn


def _quantize_weight(weight: Any) -> tuple[Any, Any]:
    torch, _ = _require_torch()
    scales = weight.detach().float().abs().amax(dim=1).clamp_min(1e-12) / 127.0
    quantized = torch.round(weight.detach().float() / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    return quantized.cpu().contiguous(), scales.cpu().contiguous()


class Int8Linear:
    """Portable post-training INT8 linear with explicit per-output scales.

    The module stores weights as int8 and dequantizes immediately before the
    matmul. This is a genuine INT8 payload and a deterministic fallback for
    runtimes without a validated fused backend; its qualification report makes
    the lack of fused INT8 execution visible.
    """

    def __new__(cls, linear: Any) -> Any:
        torch, nn = _require_torch()

        class _Impl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                quantized, scales = _quantize_weight(linear.weight)
                self.register_buffer("weight_int8", quantized)
                self.register_buffer("weight_scale", scales)
                if linear.bias is not None:
                    self.register_buffer("bias", linear.bias.detach().cpu().contiguous())
                else:
                    self.bias = None

            def forward(self, inputs: Any) -> Any:
                import torch.nn.functional as F

                weight = self.weight_int8.float() * self.weight_scale.float().unsqueeze(1)
                return F.linear(inputs.float(), weight, self.bias.float() if self.bias is not None else None).to(dtype=inputs.dtype)

        return _Impl()


def quantize_model(model: Any) -> tuple[Any, list[dict[str, Any]]]:
    torch, nn = _require_torch()
    quantized = copy.deepcopy(model).cpu().eval()
    operators: list[dict[str, Any]] = []
    for block_index, block in enumerate(quantized.backbone.blocks):
        for name in ("query", "key", "value", "output", "ffn_in", "down"):
            module = getattr(block, name)
            if not isinstance(module, nn.Linear):
                raise EncoderError("BLOCKED_INT8", "eligible backbone operator is not a linear module", {"module": f"backbone.blocks.{block_index}.{name}"})
            setattr(block, name, Int8Linear(module))
            module_name = f"backbone.blocks.{block_index}.{name}"
            operators.append({"module": module_name, "weight_dtype": "int8", "activation_dtype": "float32", "axis": 0, "scale_storage": f"{module_name}.weight_scale", "backend": "portable-dequantized-linear"})
    return quantized, operators


def quantize_export(floating_directory: str | Path, output: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    torch, _ = _require_torch()
    try:
        from safetensors import torch as safe_torch
        from .model import DeploymentModel
    except ImportError as exc:
        raise EncoderError("BLOCKED_ENVIRONMENT", "INT8 export requires safetensors and PyTorch") from exc
    source = Path(floating_directory)
    weights = source / "model.safetensors"
    if not weights.exists() or not (source / "export.json").exists():
        raise EncoderError("BLOCKED_INT8", "floating deployment export is absent")
    from .export import verify_export

    verify_export(source)
    model = DeploymentModel(seed=1729)
    model.load_state_dict(safe_torch.load_file(str(weights), device="cpu"), strict=True)
    quantized, operators = quantize_model(model)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in quantized.state_dict().items()}
    safe_torch.save_file(tensors, str(destination / "model.safetensors"), metadata={"format": "juniper-int8-v1"})
    for name in ("model_config.json", "calibration.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    if (source / "tokenizer").exists():
        shutil.copytree(source / "tokenizer", destination / "tokenizer", dirs_exist_ok=True)
    payload = {"artifact_type": "juniper-deployment-export", "format_version": "juniper-int8-v1", "variant": "int8", "status": "EXPORTED", "source_export_sha256": sha256_file(source / "model.safetensors"), "weights_sha256": sha256_file(destination / "model.safetensors"), "operators": operators, "excluded_modules": ["embedding", "LayerNorm", "softmax", "mean_pooling", "l2_normalization", "retrieval", "classifier", "reranker"], "backend": "portable-dequantized-linear", "calibration_subset": load_config(config_path).get("calibration_subset") if config_path else "quantization-calibration"}
    write_json(destination / "export.json", payload)
    make_sha256sums(destination, destination / "SHA256SUMS")
    return payload


def _load_quantized_model(directory: str | Path) -> Any:
    from safetensors.torch import load_file
    from .model import DeploymentModel

    model = DeploymentModel(seed=1729)
    model, _ = quantize_model(model)
    model.load_state_dict(load_file(str(Path(directory) / "model.safetensors"), device="cpu"), strict=True)
    return model


def verify_int8_export(directory: str | Path) -> dict[str, Any]:
    """Verify the portable INT8 payload and run one finite inference probe."""
    torch, _ = _require_torch()
    root = Path(directory)
    export_path = root / "export.json"
    weights_path = root / "model.safetensors"
    if not export_path.exists() or not weights_path.exists():
        raise EncoderError("BLOCKED_INT8", "INT8 export requires export.json and model.safetensors")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    if payload.get("status") != "EXPORTED" or payload.get("format_version") != "juniper-int8-v1" or payload.get("variant") != "int8":
        raise EncoderError("BLOCKED_INT8", "payload is not a Juniper INT8 export")
    if payload.get("weights_sha256") != sha256_file(weights_path):
        raise EncoderError("BLOCKED_INT8", "INT8 weights checksum mismatch")
    sums_path = root / "SHA256SUMS"
    if not sums_path.exists():
        raise EncoderError("BLOCKED_INT8", "INT8 export checksum manifest is absent")
    verify_sha256sums(root, sums_path)
    from safetensors.torch import load_file

    tensors = load_file(str(weights_path), device="cpu")
    operators = payload.get("operators", [])
    missing = [item["scale_storage"] for item in operators if item.get("scale_storage") not in tensors or tensors[item["scale_storage"]].dtype != torch.float32]
    int8_keys = [item["module"] + ".weight_int8" for item in operators]
    missing.extend(key for key in int8_keys if key not in tensors or tensors[key].dtype != torch.int8)
    if missing:
        raise EncoderError("BLOCKED_INT8", "INT8 payload is missing declared quantization tensors", {"missing": missing[:20]})
    model = _load_quantized_model(root).eval()
    with torch.inference_mode():
        outputs = (model.retrieval_vector(torch.full((1, 8), 7, dtype=torch.long)), model.classify(torch.full((1, 8), 7, dtype=torch.long)), model.rerank(torch.full((1, 8), 7, dtype=torch.long)))
    if not all(bool(torch.isfinite(output).all()) for output in outputs):
        raise EncoderError("BLOCKED_INT8", "INT8 finite-output probe failed")
    return {"status": "PASS", "variant": "int8", "operators": len(operators), "int8_tensors": len(int8_keys), "backend": payload.get("backend"), "finite_probe": True}


def load_int8_export(directory: str | Path, *, device: str = "cpu") -> Any:
    """Load a verified INT8 export as an executable benchmark runner."""
    verify_int8_export(directory)
    from .benchmark import make_model_runner

    return make_model_runner(_load_quantized_model(directory), device=device)
