"""Actual deterministic MLM and downstream training primitives.

The runner is intentionally small and explicit: it consumes already-frozen
records, performs real optimizer updates, and emits safe-tensor checkpoints.
It does not invent labels or silently turn a missing data manifest into a
successful run.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..constants import CLASS_COUNT, ORDINARY_TOKEN_MAX, ORDINARY_TOKEN_MIN
from ..errors import EncoderError, numerical_error
from ..model import DeploymentModel, JuniperBackbone, zero_pad_gradient_and_optimizer_state
from ..utils import canonical_json_bytes, sha256_bytes, sha256_file, write_json
from .checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from .protocol import MaskingProtocol, require_safetensors, require_training_runtime


@dataclasses.dataclass(frozen=True)
class TokenExample:
    example_id: str
    token_ids: tuple[int, ...]

    @classmethod
    def from_record(cls, record: dict[str, Any], tokenizer: Any) -> "TokenExample":
        text = record.get("text", record.get("content"))
        if not isinstance(text, str) or not text:
            raise ValueError("training records need nonempty text/content")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("training records need immutable record_id")
        tokens = tuple(int(token) for token in tokenizer.encode(text))
        if not tokens or any(token < ORDINARY_TOKEN_MIN or token > ORDINARY_TOKEN_MAX for token in tokens):
            raise ValueError("training examples must contain ordinary tokenizer IDs")
        return cls(record_id, tokens)


class DeterministicSampler:
    def __init__(self, examples: Sequence[TokenExample], *, seed: int = 1729) -> None:
        self.examples = tuple(examples)
        self.seed = seed
        self.epoch = 0
        self.cursor = 0

    def order(self) -> list[int]:
        generator = random.Random(f"{self.seed}:{self.epoch}")
        result = list(range(len(self.examples)))
        generator.shuffle(result)
        return result

    def next_batch(self, batch_size: int) -> list[TokenExample]:
        if not self.examples:
            raise EncoderError("BLOCKED_DATA", "training sampler has no examples")
        order = self.order()
        if self.cursor >= len(order):
            self.epoch += 1
            self.cursor = 0
            order = self.order()
        indices = order[self.cursor:self.cursor + batch_size]
        self.cursor += len(indices)
        return [self.examples[index] for index in indices]

    def state(self) -> dict[str, Any]:
        return {"seed": self.seed, "epoch": self.epoch, "cursor": self.cursor, "example_ids": [item.example_id for item in self.examples]}

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("seed") != self.seed or state.get("example_ids") != [item.example_id for item in self.examples]:
            raise EncoderError("INVALID_INPUT", "sampler state does not match the training examples")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])


def _pad_batch(examples: Sequence[TokenExample], masking: MaskingProtocol, epoch: int, device: Any) -> tuple[Any, Any, Any, int]:
    import torch

    corrupted: list[list[int]] = []
    positions: list[tuple[int, int]] = []
    targets: list[int] = []
    width = max(len(item.token_ids) for item in examples)
    for row, example in enumerate(examples):
        masked, selected, expected = masking.mask_detailed(example.token_ids, example.example_id, epoch)
        corrupted.append(masked + [0] * (width - len(masked)))
        positions.extend((row, index) for index in selected)
        targets.extend(expected)
    if not targets:
        raise EncoderError("BLOCKED_DATA", "MLM batch contains no supervised ordinary tokens")
    ids = torch.tensor(corrupted, dtype=torch.long, device=device)
    selected = torch.tensor(positions, dtype=torch.long, device=device)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return ids, selected, target_tensor, len(targets)


class MLMTrainingModel:
    """Temporary tied decoder used only while the foundation objective runs."""

    def __init__(self, backbone: Any) -> None:
        import torch.nn as nn

        self.module = nn.Module()
        self.module.add_module("backbone", backbone)

    def parameters(self) -> Iterable[Any]:
        return self.module.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def train(self, mode: bool = True) -> None:
        self.module.train(mode)

    def eval(self) -> None:
        self.module.eval()

    def to(self, device: Any) -> "MLMTrainingModel":
        self.module.to(device)
        return self

    def __call__(self, input_ids: Any, selected: Any) -> Any:
        import torch.nn.functional as F

        hidden, _, _ = self.module.backbone(input_ids)
        rows, columns = selected[:, 0], selected[:, 1]
        selected_hidden = hidden[rows, columns].float()
        # Weight tying is by reference: the temporary decoder adds no deployed
        # parameters and is never exported.
        return F.linear(selected_hidden, self.module.backbone.embedding.weight.float())


def _optimizer_for(module: Any, *, backbone_lr: float, head_lr: float | None = None) -> Any:
    import torch

    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "ln_" in name or "final_norm" in name else decay).append(parameter)
    groups = [{"params": decay, "lr": backbone_lr, "weight_decay": 0.01}, {"params": no_decay, "lr": backbone_lr, "weight_decay": 0.0}]
    if head_lr is not None:
        groups.append({"params": [], "lr": head_lr, "weight_decay": 0.01})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def _amp_context(device: Any, enabled: bool) -> Any:
    import torch

    if enabled and getattr(device, "type", str(device)) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_mlm(
    examples: Sequence[TokenExample],
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    source_sha: str,
    tokenizer_identity: str,
    data_identity: str,
    device: str | None = None,
    max_updates: int | None = None,
    checkpoint_every: int = 100,
    resume_dir: str | Path | None = None,
) -> dict[str, Any]:
    require_training_runtime()
    require_safetensors()
    import torch

    seed = int(config.get("seed", 1729))
    torch.manual_seed(seed)
    random.seed(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone = JuniperBackbone(seed=seed).to(selected_device)
    model = MLMTrainingModel(backbone).to(selected_device)
    model.train()
    optimizer = _optimizer_for(model.module, backbone_lr=float(config.get("peak_learning_rate", 1e-4)))
    total_updates = int(max_updates if max_updates is not None else config.get("successful_update_budget") or 1)
    warmup = max(1, int(total_updates * float(config.get("warmup_fraction", 0.06))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup) * max(0.0, 1.0 - max(0, step - warmup) / max(1, total_updates - warmup)))
    amp_enabled = bool(config.get("amp", "").startswith("validate-fp16")) and selected_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    masking = MaskingProtocol(seed=seed, mask_rate=float(config.get("mask_rate", 0.15)))
    sampler = DeterministicSampler(examples, seed=seed)
    successful = skipped = attempted = consumed = 0
    predecessor_identity = None
    if resume_dir:
        loaded = load_checkpoint(resume_dir, model.module, optimizer, scheduler, scaler, restore_rng=True)
        previous = loaded["metadata"]
        if previous.get("tokenizer_identity") != tokenizer_identity or previous.get("data_identity") != data_identity:
            raise EncoderError("INVALID_INPUT", "resume checkpoint does not match tokenizer/data identities")
        sampler.load_state(previous.get("sampler_state", {}))
        successful = int(previous["successful_updates"])
        attempted = int(previous["attempted_updates"])
        skipped = int(previous["skipped_updates"])
        consumed = int(previous["consumed_tokens"])
        predecessor_identity = loaded.get("checkpoint_identity")
    losses: list[float] = []
    started = time.monotonic()
    while successful < total_updates:
        attempted += 1
        batch = sampler.next_batch(max(1, int(config.get("batch_size", 1))))
        ids, selected, targets, target_count = _pad_batch(batch, masking, sampler.epoch, selected_device)
        consumed += int(sum(len(item.token_ids) for item in batch))
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(selected_device, amp_enabled):
            logits = model(ids, selected)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum") / target_count
        if not bool(torch.isfinite(loss).item()):
            skipped += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            optimizer.zero_grad(set_to_none=True)
            # unscale_ registered the inf checks; close this scaler iteration
            # even when the explicit finite-gradient gate rejects the update.
            if amp_enabled:
                scaler.update()
            skipped += 1
            continue
        scaler.step(optimizer)
        scaler.update()
        zero_pad_gradient_and_optimizer_state(model.module, optimizer)
        scheduler.step()
        successful += 1
        losses.append(float(loss.detach().cpu().item()))
        if checkpoint_every and successful % checkpoint_every == 0:
            metadata = CheckpointMetadata(
                format_version="juniper-checkpoint-v2", source_sha=source_sha,
                config_hashes={"mlm": sha256_bytes(canonical_json_bytes(config))}, rng_states={},
                successful_updates=successful, attempted_updates=attempted, skipped_updates=skipped,
                consumed_tokens=consumed, sampler_identity=sha256_bytes(canonical_json_bytes(sampler.state())),
                tokenizer_identity=tokenizer_identity, data_identity=data_identity, predecessor_identity=predecessor_identity,
                epoch=sampler.epoch, cursor=sampler.cursor,
            )
            save_checkpoint(Path(output_dir) / f"step-{successful:08d}", model.module, optimizer, scheduler, scaler, metadata, sampler_state=sampler.state(), precision={"device": str(selected_device), "amp_fp16": amp_enabled})
    final_dir = Path(output_dir) / f"step-{successful:08d}"
    if not final_dir.exists():
        metadata = CheckpointMetadata(
            format_version="juniper-checkpoint-v2", source_sha=source_sha,
            config_hashes={"mlm": sha256_bytes(canonical_json_bytes(config))}, rng_states={},
            successful_updates=successful, attempted_updates=attempted, skipped_updates=skipped,
            consumed_tokens=consumed, sampler_identity=sha256_bytes(canonical_json_bytes(sampler.state())),
            tokenizer_identity=tokenizer_identity, data_identity=data_identity, predecessor_identity=predecessor_identity,
            epoch=sampler.epoch, cursor=sampler.cursor,
        )
        save_checkpoint(final_dir, model.module, optimizer, scheduler, scaler, metadata, sampler_state=sampler.state(), precision={"device": str(selected_device), "amp_fp16": amp_enabled})
    result = {"status": "TRAINED", "stage": "mlm", "checkpoint": final_dir.as_posix(), "successful_updates": successful, "attempted_updates": attempted, "skipped_updates": skipped, "consumed_tokens": consumed, "mean_loss": sum(losses) / len(losses) if losses else None, "elapsed_seconds": time.monotonic() - started, "device": str(selected_device), "amp_fp16": amp_enabled, "weights_sha256": sha256_file(final_dir / "model.safetensors")}
    write_json(Path(output_dir) / "run.json", result)
    return result


def retrieval_contrastive_loss(query_vectors: Any, candidate_vectors: Any, *, temperature: float = 0.05) -> Any:
    """Compute the frozen FP32 temperature-scaled retrieval loss.

    A two-dimensional candidate tensor uses in-batch negatives and assumes the
    diagonal is positive. A three-dimensional tensor is shaped ``[B,K,D]``;
    candidate zero is the positive and the remaining columns are explicit
    negatives for that query.
    """
    import torch
    import torch.nn.functional as F

    if temperature != 0.05 or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("retrieval contrastive temperature is frozen at 0.05")
    query = query_vectors.float()
    candidates = candidate_vectors.float()
    if query.ndim != 2 or candidates.ndim not in (2, 3):
        raise ValueError("retrieval vectors must be rank-2 or rank-3 candidate batches")
    if candidates.ndim == 2:
        if query.shape != candidates.shape:
            raise ValueError("in-batch retrieval candidates must match query shape")
        logits = query @ candidates.transpose(0, 1) / temperature
        targets = torch.arange(query.shape[0], device=query.device)
    else:
        if candidates.shape[0] != query.shape[0] or candidates.shape[2] != query.shape[1] or candidates.shape[1] < 1:
            raise ValueError("explicit retrieval candidates have incompatible shape")
        logits = torch.einsum("bd,bkd->bk", query, candidates) / temperature
        targets = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, targets)


def reranker_listwise_loss(scores: Any, target_indices: Any) -> Any:
    """Compute listwise cross-entropy over one query's candidate group."""
    import torch.nn.functional as F

    values = scores.float()
    targets = target_indices.long()
    if values.ndim != 2 or targets.ndim != 1 or values.shape[0] != targets.shape[0] or bool((targets < 0).any()) or bool((targets >= values.shape[1]).any()):
        raise ValueError("reranker listwise inputs must be [batch,candidates] and valid target indices")
    return F.cross_entropy(values, targets)


def _flatten_candidate_inputs(inputs: Any, forward: Any) -> Any:
    if getattr(inputs, "ndim", 0) != 3:
        return forward(inputs)
    batch, candidates, length = inputs.shape
    return forward(inputs.reshape(batch * candidates, length)).reshape(batch, candidates)


def train_head(model: Any, batches: Iterable[tuple[Any, Any]], *, objective: str, optimizer: Any, device: Any, updates: int) -> dict[str, Any]:
    """Train a retrieval, listwise reranker, or classifier over caller data.

    The caller owns labels and candidate provenance. Missing labels are not
    synthesized by this primitive.
    """
    import torch
    import torch.nn.functional as F

    if objective not in {"retrieval", "reranker", "classifier"}:
        raise ValueError("unsupported downstream objective")
    model.train()
    losses: list[float] = []
    iterator = iter(batches)
    for _ in range(updates):
        try:
            inputs, labels = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            inputs, labels = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        if objective == "classifier":
            logits = model.classify(inputs.to(device))
            loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
        elif objective == "reranker":
            scores = _flatten_candidate_inputs(inputs.to(device), model.rerank)
            target = labels.to(device)
            if scores.ndim == 2:
                if target.ndim == 2:
                    target = target.argmax(dim=-1)
                loss = reranker_listwise_loss(scores, target)
            else:
                # Pairwise datasets are supported only when their labels are
                # explicitly supplied; the default configured objective is
                # listwise and therefore takes the rank-2 path above.
                loss = F.binary_cross_entropy_with_logits(scores, target.float())
        else:
            vectors = model.retrieval_vector(inputs.to(device))
            candidate_vectors = labels["candidates"] if isinstance(labels, dict) and "candidates" in labels else labels
            candidate_vectors = candidate_vectors.to(device)
            loss = retrieval_contrastive_loss(vectors, candidate_vectors)
        if not bool(torch.isfinite(loss).item()):
            raise numerical_error("downstream training produced a nonfinite loss", objective=objective)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        zero_pad_gradient_and_optimizer_state(model, optimizer)
        losses.append(float(loss.detach().cpu().item()))
    return {"status": "TRAINED", "objective": objective, "successful_updates": updates, "mean_loss": sum(losses) / len(losses) if losses else None}


def initialize_deployment_from_foundation(foundation_directory: str | Path, output_directory: str | Path, *, source_sha: str, tokenizer_identity: str, data_identity: str) -> dict[str, Any]:
    """Promote a foundation backbone into a real deployment checkpoint.

    Heads are freshly initialized here and must subsequently be trained by the
    downstream objectives. The resulting artifact is therefore explicitly
    marked ``DOWNSTREAM_PENDING`` until those objectives finish.
    """
    require_training_runtime()
    require_safetensors()
    import torch
    from safetensors import torch as safe_torch

    foundation = Path(foundation_directory)
    state = safe_torch.load_file(str(foundation / "model.safetensors"), device="cpu")
    if not state or any(key.startswith(("retrieval.", "classifier.", "reranker.")) for key in state):
        raise EncoderError("INVALID_INPUT", "foundation checkpoint contains deployment head weights")
    if all(key.startswith("backbone.") for key in state):
        backbone_state = state
    else:
        backbone_state = {f"backbone.{key}": value for key, value in state.items()}
    model = DeploymentModel(seed=1729)
    model.load_state_dict(backbone_state, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metadata = CheckpointMetadata(
        format_version="juniper-checkpoint-v2", source_sha=source_sha, config_hashes={}, rng_states={},
        successful_updates=0, attempted_updates=0, skipped_updates=0, consumed_tokens=0,
        sampler_identity="downstream-pending", tokenizer_identity=tokenizer_identity,
        data_identity=data_identity, predecessor_identity=json.loads((foundation / "metadata.json").read_text(encoding="utf-8")).get("checkpoint_identity"),
    )
    result = save_checkpoint(output_directory, model, optimizer, None, None, metadata, precision={"device": "cpu", "amp_fp16": False})
    result["status"] = "DOWNSTREAM_PENDING"
    write_json(Path(output_directory) / "promotion.json", result)
    return result
