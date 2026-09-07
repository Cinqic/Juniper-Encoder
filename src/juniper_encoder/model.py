"""Proposal B model definition and a transparent PyTorch oracle.

PyTorch is intentionally optional for contract-only installs. Training and model
execution fail with an explicit environment blocker until the verified lock
materializes it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

from .constants import *  # noqa: F403 - the frozen values are the model contract.
from .errors import EncoderError, numerical_error

try:  # pragma: no cover - exercised on FLOWBOX after the locked runtime is installed.
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - the CPU contract suite does not need torch.
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    blocks: int = BLOCKS
    width: int = HIDDEN_WIDTH
    q_heads: int = Q_HEADS
    kv_heads: int = KV_HEADS
    head_dim: int = HEAD_DIM
    geglu_width: int = GEGLU_WIDTH
    rope_theta: float = ROPE_THETA
    layer_norm_epsilon: float = LN_EPS
    dropout: float = DROPOUT_P
    native_context: int = 1_024

    def validate(self) -> None:
        expected = {
            "vocab_size": VOCAB_SIZE,
            "blocks": BLOCKS,
            "width": HIDDEN_WIDTH,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "geglu_width": GEGLU_WIDTH,
            "rope_theta": ROPE_THETA,
            "layer_norm_epsilon": LN_EPS,
            "dropout": DROPOUT_P,
        }
        for field, frozen in expected.items():
            if getattr(self, field) != frozen:
                raise ValueError(f"{field} diverges from frozen model contract")
        if self.q_heads != self.kv_heads or self.width != self.q_heads * self.head_dim:
            raise ValueError("head shape is not the frozen 512 = 8 x 64 layout")


def require_torch() -> Any:
    if torch is None:
        raise EncoderError(
            "BLOCKED_ENVIRONMENT",
            "PyTorch is not installed in the verified environment",
            {"install": "bash scripts/bootstrap.sh --lock requirements/flowbox.lock"},
        )
    return torch


def _zero_pad(x: Tensor, input_ids: Tensor) -> Tensor:
    return x.masked_fill((input_ids == 0).unsqueeze(-1), 0.0)


def truncated_normal_(tensor: Tensor, *, generator: Any | None = None) -> Tensor:
    """Fill from N(0, .02²) rejected outside [-.04, .04]; never clip."""
    require_torch()
    with torch.no_grad():
        pending = torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)
        while bool(pending.any()):
            sample = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype) * INIT_STD
            accepted = pending & (sample >= -INIT_ABS_BOUND) & (sample <= INIT_ABS_BOUND)
            tensor[accepted] = sample[accepted]
            pending &= ~accepted
    return tensor


def zero_pad_gradient_and_optimizer_state(model: Any, optimizer: Any | None = None) -> None:
    """Keep the PAD row and its Adam moments zero at every training boundary."""
    require_torch()
    embedding = model.backbone.embedding.weight
    with torch.no_grad():
        embedding[0].zero_()
    if embedding.grad is not None:
        embedding.grad[0].zero_()
    if optimizer is not None:
        state = optimizer.state.get(embedding, {})
        for value in state.values():
            if hasattr(value, "ndim") and value.ndim == embedding.ndim and value.shape[0] == embedding.shape[0]:
                value[0].zero_()


if torch is not None:  # pragma: no branch

    class ExactLayerNorm(nn.Module):
        def __init__(self, width: int, eps: float = LN_EPS) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(width))
            self.bias = nn.Parameter(torch.zeros(width))
            self.eps = eps

        def forward(self, x: Tensor) -> Tensor:
            xf = x.float()
            mean = xf.mean(dim=-1, keepdim=True)
            variance = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
            normalized = (xf - mean) * torch.rsqrt(variance + self.eps)
            return (normalized * self.scale.float() + self.bias.float()).to(dtype=x.dtype)


    class AdjacentPairRoPE(nn.Module):
        def __init__(self, head_dim: int = HEAD_DIM, theta: float = ROPE_THETA) -> None:
            super().__init__()
            if head_dim % 2:
                raise ValueError("head dimension must have adjacent pairs")
            inverse = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            self.register_buffer("inverse_frequency", inverse, persistent=True)

        def forward(self, x: Tensor, positions: Tensor) -> Tensor:
            # x: batch, sequence, heads, head_dim. The convention is the
            # positive rotation x0' = x0 cos - x1 sin.
            angles = positions.float().unsqueeze(-1) * self.inverse_frequency
            cosine = angles.cos().unsqueeze(0).unsqueeze(2)
            sine = angles.sin().unsqueeze(0).unsqueeze(2)
            first = x[..., 0::2]
            second = x[..., 1::2]
            rotated_first = first * cosine - second * sine
            rotated_second = first * sine + second * cosine
            result = torch.empty_like(x)
            result[..., 0::2] = rotated_first
            result[..., 1::2] = rotated_second
            return result


    class EncoderBlock(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.ln_attention = ExactLayerNorm(config.width, config.layer_norm_epsilon)
            self.query = nn.Linear(config.width, config.width, bias=False)
            self.key = nn.Linear(config.width, config.width, bias=False)
            self.value = nn.Linear(config.width, config.width, bias=False)
            self.output = nn.Linear(config.width, config.width, bias=False)
            self.ln_geglu = ExactLayerNorm(config.width, config.layer_norm_epsilon)
            self.gate = nn.Linear(config.width, config.geglu_width, bias=False)
            self.value_branch = nn.Linear(config.width, config.geglu_width, bias=False)
            self.down = nn.Linear(config.geglu_width, config.width, bias=False)
            self.rope = AdjacentPairRoPE(config.head_dim, config.rope_theta)
            self.config = config

        def forward(self, x: Tensor, input_ids: Tensor, positions: Tensor, training: bool) -> Tensor:
            batch, length, _ = x.shape
            head_count = self.config.q_heads
            residual = x
            normalized = self.ln_attention(x)
            q = self.query(normalized).view(batch, length, head_count, self.config.head_dim)
            k = self.key(normalized).view(batch, length, head_count, self.config.head_dim)
            v = self.value(normalized).view(batch, length, head_count, self.config.head_dim)
            if not bool(torch.isfinite(q).all() and torch.isfinite(k).all() and torch.isfinite(v).all()):
                raise numerical_error("nonfinite attention projection")
            q = self.rope(q, positions)
            k = self.rope(k, positions)
            scores = torch.einsum("bshd,bthd->bhst", q.float(), k.float()) * ROPE_SCALE
            key_is_real = input_ids.ne(0).unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~key_is_real, float("-inf"))
            if not bool(torch.isfinite(scores.masked_fill(~key_is_real, 0.0)).all()):
                raise numerical_error("nonfinite attention score")
            probabilities = torch.softmax(scores, dim=-1)
            probabilities = F.dropout(probabilities, p=DROPOUT_P, training=training)
            attended = torch.einsum("bhst,bthd->bshd", probabilities, v.float()).reshape(batch, length, -1)
            attended = self.output(attended.to(dtype=x.dtype))
            attended = F.dropout(attended, p=DROPOUT_P, training=training)
            x = _zero_pad(residual + attended, input_ids)

            residual = x
            normalized = self.ln_geglu(x)
            geglu = F.gelu(self.gate(normalized), approximate="none") * self.value_branch(normalized)
            feed_forward = self.down(geglu)
            feed_forward = F.dropout(feed_forward, p=DROPOUT_P, training=training)
            return _zero_pad(residual + feed_forward, input_ids)


    class JuniperBackbone(nn.Module):
        def __init__(self, config: ModelConfig | None = None, *, seed: int = 1729) -> None:
            super().__init__()
            self.config = config or ModelConfig()
            self.config.validate()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.embedding = nn.Embedding(self.config.vocab_size, self.config.width)
            self.blocks = nn.ModuleList(EncoderBlock(self.config) for _ in range(self.config.blocks))
            self.final_norm = ExactLayerNorm(self.config.width, self.config.layer_norm_epsilon)
            self.reset_parameters(generator)

        def reset_parameters(self, generator: Any) -> None:
            truncated_normal_(self.embedding.weight, generator=generator)
            for module in self.modules():
                if module is self or module is self.embedding:
                    continue
                if isinstance(module, nn.Linear):
                    truncated_normal_(module.weight, generator=generator)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, ExactLayerNorm):
                    nn.init.ones_(module.scale)
                    nn.init.zeros_(module.bias)
            self.enforce_pad_row()

        @torch.no_grad()
        def enforce_pad_row(self) -> None:
            self.embedding.weight[0].zero_()

        def forward(self, input_ids: Tensor, *, training: bool | None = None) -> tuple[Tensor, Tensor, Tensor]:
            if input_ids.ndim != 2 or input_ids.dtype not in (torch.int64, torch.int32):
                raise ValueError("input_ids must be a rank-2 integer tensor")
            if input_ids.numel() == 0 or bool(input_ids.eq(0).all(dim=-1).any()):
                raise invalid_input("all-PAD inputs are invalid")
            if bool((input_ids < 0).any()) or bool((input_ids >= VOCAB_SIZE).any()):
                raise invalid_input("input token ID is outside the vocabulary")
            training = self.training if training is None else training
            positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.float32)
            hidden = self.embedding(input_ids)
            hidden = F.dropout(hidden, p=DROPOUT_P, training=training)
            hidden = _zero_pad(hidden, input_ids)
            if not bool(torch.isfinite(hidden).all()):
                raise numerical_error("nonfinite embedding activation")
            for block in self.blocks:
                hidden = block(hidden, input_ids, positions, training)
                if not bool(torch.isfinite(hidden).all()):
                    raise numerical_error("nonfinite block activation")
            hidden = _zero_pad(self.final_norm(hidden), input_ids)
            if not bool(torch.isfinite(hidden).all()):
                raise numerical_error("nonfinite final activation")
            ordinary = input_ids.ge(7) & input_ids.le(LAST_TOKEN_ID)
            denominator = ordinary.sum(dim=-1, keepdim=True)
            if bool(denominator.eq(0).any()):
                raise invalid_input("input has no ordinary tokens for pooling")
            pooled = (hidden.float() * ordinary.unsqueeze(-1)).sum(dim=1) / denominator.float()
            return hidden, pooled, ordinary


    class DeploymentModel(nn.Module):
        """Backbone plus exactly the three frozen deployment heads."""
        def __init__(self, config: ModelConfig | None = None, *, seed: int = 1729) -> None:
            super().__init__()
            self.backbone = JuniperBackbone(config, seed=seed)
            self.retrieval = nn.Linear(HIDDEN_WIDTH, RETRIEVAL_DIM, bias=False)
            self.classifier = nn.Linear(HIDDEN_WIDTH, CLASS_COUNT, bias=True)
            self.reranker = nn.Linear(HIDDEN_WIDTH, 1, bias=True)
            generator = torch.Generator(device="cpu").manual_seed(seed + 1)
            truncated_normal_(self.retrieval.weight, generator=generator)
            truncated_normal_(self.classifier.weight, generator=generator)
            truncated_normal_(self.reranker.weight, generator=generator)
            nn.init.zeros_(self.classifier.bias)
            nn.init.zeros_(self.reranker.bias)

        def encode_pool(self, input_ids: Tensor) -> Tensor:
            _, pooled, _ = self.backbone(input_ids)
            return pooled

        def retrieval_vector(self, input_ids: Tensor) -> Tensor:
            projected = self.retrieval(self.encode_pool(input_ids).float())
            norm = projected.norm(dim=-1, keepdim=True)
            if bool((norm < PROJECTION_NORM_EPS).any()) or not bool(torch.isfinite(norm).all()):
                raise numerical_error("retrieval projection norm is invalid")
            return (projected / norm).float()

        def classify(self, input_ids: Tensor) -> Tensor:
            logits = self.classifier(self.encode_pool(input_ids).float()).float()
            if not bool(torch.isfinite(logits).all()):
                raise numerical_error("classifier produced nonfinite logits")
            return logits

        def rerank(self, input_ids: Tensor) -> Tensor:
            score = self.reranker(self.encode_pool(input_ids).float()).squeeze(-1).float()
            if not bool(torch.isfinite(score).all()):
                raise numerical_error("reranker produced nonfinite scores")
            return score


else:

    class ExactLayerNorm:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class AdjacentPairRoPE:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class EncoderBlock:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class JuniperBackbone:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


    class DeploymentModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()


def actual_parameter_counts(model: Any) -> dict[str, int]:
    """Count actual named tensors, with no arithmetic from configuration labels."""
    require_torch()
    physical = {name: int(parameter.numel()) for name, parameter in model.named_parameters()}
    total = sum(physical.values())
    backbone_total = sum(parameter.numel() for parameter in model.backbone.parameters())
    retrieval_total = backbone_total + sum(parameter.numel() for parameter in model.retrieval.parameters())
    decision_total = backbone_total + sum(parameter.numel() for parameter in model.classifier.parameters())
    reranking_total = backbone_total + sum(parameter.numel() for parameter in model.reranker.parameters())
    return {
        "physical_named_parameters": total,
        "shared_backbone": backbone_total,
        "retrieval_participating": retrieval_total,
        "decision_participating": decision_total,
        "reranking_participating": reranking_total,
        "fast_pipeline_unique": retrieval_total + sum(parameter.numel() for parameter in model.classifier.parameters()),
        "reranked_pipeline_deployed_total": total,
        "state_dict_keys": len(model.state_dict()),
        "pad_row_max_abs": float(model.backbone.embedding.weight[0].abs().max().item()),
    }
