"""Resolved training protocol; expensive execution is never silently faked."""

from __future__ import annotations

import dataclasses
import random
from typing import Any, Sequence

from ..constants import ORDINARY_TOKEN_MIN, ORDINARY_TOKEN_MAX
from ..errors import EncoderError
from ..utils import canonical_json_bytes, sha256_bytes


@dataclasses.dataclass(frozen=True)
class MaskingProtocol:
    seed: int = 1729
    mask_rate: float = 0.15

    def mask(self, token_ids: Sequence[int], example_id: str, epoch: int) -> tuple[list[int], list[int]]:
        eligible = [i for i, token_id in enumerate(token_ids) if ORDINARY_TOKEN_MIN <= token_id <= ORDINARY_TOKEN_MAX]
        rng = random.Random(f"{self.seed}:{example_id}:{epoch}")
        chosen = [i for i in eligible if rng.random() < self.mask_rate]
        if eligible and not chosen:
            chosen = [eligible[0]]
        output = list(token_ids)
        targets: list[int] = []
        for position in chosen:
            targets.append(token_ids[position])
            branch = rng.random()
            if branch < 0.8:
                output[position] = 4
            elif branch < 0.9:
                output[position] = rng.randint(ORDINARY_TOKEN_MIN, ORDINARY_TOKEN_MAX)
        return output, targets


@dataclasses.dataclass(frozen=True)
class ResolvedProtocol:
    stage: str
    seed: int
    max_consumed_tokens: int
    successful_update_budget: int | None
    config_hash: str
    workers: int = 0

    @classmethod
    def from_config(cls, stage: str, config: dict[str, Any]) -> "ResolvedProtocol":
        if config.get("workers", 0) != 0:
            raise ValueError("reference protocol starts with zero data-loader workers")
        payload = {"stage": stage, "config": config}
        return cls(stage, int(config.get("seed", 1729)), int(config.get("max_consumed_tokens", 0)), config.get("successful_update_budget"), sha256_bytes(canonical_json_bytes(payload)), 0)


def require_training_runtime() -> None:
    try:
        import torch
    except ImportError as exc:
        raise EncoderError("BLOCKED_ENVIRONMENT", "training requires the verified PyTorch runtime", {"dependency": "torch"}) from exc
    if not hasattr(torch, "optim"):
        raise EncoderError("BLOCKED_ENVIRONMENT", "PyTorch optimizer support is unavailable")


def run_mechanical_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Run a real finite forward/backward check when the verified runtime exists."""
    require_training_runtime()
    import torch
    from ..model import DeploymentModel, actual_parameter_counts

    torch.manual_seed(int(config.get("seed", 1729)))
    model = DeploymentModel(seed=int(config.get("seed", 1729)))
    model.train()
    input_ids = torch.tensor([[7, 8, 9, 0], [10, 11, 0, 0]], dtype=torch.long)
    vectors = model.retrieval_vector(input_ids)
    loss = vectors.float().square().mean()
    loss.backward()
    pad_grad = model.backbone.embedding.weight.grad[0].abs().max().item()
    model.backbone.enforce_pad_row()
    finite = bool(torch.isfinite(loss)) and bool(torch.isfinite(vectors).all()) and bool(torch.isfinite(model.backbone.embedding.weight.grad).all())
    if not finite or pad_grad != 0.0:
        raise EncoderError("NUMERICAL_ERROR", "mechanical smoke produced nonfinite values or a PAD gradient")
    return {"status": "SMOKE_TESTED", "loss": float(loss.item()), "pad_gradient_max_abs": pad_grad, "counts": actual_parameter_counts(model)}
