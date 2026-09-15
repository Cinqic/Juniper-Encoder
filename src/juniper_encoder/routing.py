"""Deterministic conditional reranking and three-class routing semantics."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .constants import CLASS_NAMES, CLASS_CALL, CLASS_CLARIFY, CLASS_NO_CALL
from .errors import EncoderError, index_mismatch, invalid_input, numerical_error
from .formatting import (
    FormattedRequest,
    RegistryRecord,
    format_decision,
    format_generic_pair,
    format_request,
    validate_decision,
)
from .registry import RegistryIndex, RegistrySnapshot
from .tokenizer import RawByteBPE
from .utils import require_finite


@dataclasses.dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    score_threshold: float | None = None
    margin_threshold: float | None = None
    calibrated: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise numerical_error("temperature must be finite and positive")
        for value in (self.score_threshold, self.margin_threshold):
            if value is not None and not math.isfinite(value):
                raise numerical_error("calibration threshold must be finite")
        if self.calibrated and (self.score_threshold is None or self.margin_threshold is None):
            raise numerical_error("calibrated routing requires both deployment thresholds")


@dataclasses.dataclass(frozen=True)
class RoutingResult:
    decision: str
    capability_id: str | None
    metadata: dict[str, Any]

    def decision_json(self) -> dict[str, Any]:
        return {"decision": self.decision, "capability_id": self.capability_id}


@dataclasses.dataclass(frozen=True)
class RoutingCallbacks:
    encode_query: Callable[[FormattedRequest], Sequence[float]]
    rerank_pair: Callable[[Sequence[int], RegistryRecord], float]
    classify: Callable[[Sequence[int]], Sequence[float]]


def _classify(logits: Sequence[float], temperature: float) -> int:
    values = tuple(float(value) for value in logits)
    if len(values) != 3:
        raise numerical_error("classifier must return exactly three logits")
    require_finite(values, "classifier logits")
    scaled = tuple(value / temperature for value in values)
    # Argmax is invariant to positive T; explicit lowest-index tie handling is required.
    return min(range(3), key=lambda index: (-scaled[index], index))


def _class_probabilities(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    values = tuple(float(value) / temperature for value in logits)
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    total = math.fsum(exponentials)
    if not math.isfinite(total) or total <= 0:
        raise numerical_error("classifier probability normalization failed")
    return tuple(value / total for value in exponentials)


def route(
    request: Mapping[str, Any],
    snapshot: RegistrySnapshot,
    index: RegistryIndex,
    tokenizer: RawByteBPE,
    callbacks: RoutingCallbacks,
    *,
    calibration: Calibration | None = None,
) -> RoutingResult:
    # Formatting and compatibility are performed even for an empty registry.
    formatted = format_request(request, tokenizer)
    if tokenizer.identity() != snapshot.tokenizer_identity:
        raise index_mismatch("active tokenizer does not match the registry snapshot", expected=snapshot.tokenizer_identity, actual=tokenizer.identity())
    index.verify_compatible(snapshot, tokenizer_identity=snapshot.tokenizer_identity, encoder_variant_identity=snapshot.encoder_variant_identity)
    calibration = calibration or Calibration()
    calibration.validate()
    if not snapshot.records:
        return RoutingResult("NO_CALL", None, {"dropped_history_count": formatted.dropped_history_count, "path": "empty-registry", "neural_passes": 0})

    query_vector = tuple(float(value) for value in callbacks.encode_query(formatted))
    require_finite(query_vector, "query embedding")
    ranked = index.search(query_vector, snapshot, limit=8)
    if not ranked:
        raise numerical_error("nonempty registry produced an empty shortlist")
    path = "singleton" if len(snapshot.records) == 1 else "conditional"
    reranked = False
    final = ranked
    if len(snapshot.records) >= 2:
        score = ranked[0][1]
        margin = ranked[0][1] - ranked[1][1]
        can_skip = (
            calibration.calibrated
            and calibration.score_threshold is not None
            and calibration.margin_threshold is not None
            and score >= calibration.score_threshold
            and margin >= calibration.margin_threshold
        )
        if not can_skip:
            reranked = True
            rescored = []
            for record, retrieval_score in ranked[: min(8, len(ranked))]:
                pair = format_generic_pair(formatted.serialized, record, tokenizer, allow_document_truncation=False)
                score_value = float(callbacks.rerank_pair(pair.token_ids, record))
                if not math.isfinite(score_value):
                    raise numerical_error("reranker produced a nonfinite score", capability_id=record.id)
                rescored.append((record, score_value, retrieval_score))
            rescored.sort(key=lambda item: (-item[1], item[0].id.encode("ascii")))
            final = [(record, rerank_score) for record, rerank_score, _ in rescored]
        else:
            path = "fast"

    candidates = [record for record, _ in final[:2]]
    decision_input = format_decision(formatted.serialized, candidates, tokenizer)
    logits = tuple(float(value) for value in callbacks.classify(decision_input.token_ids))
    class_index = _classify(logits, calibration.temperature)
    probabilities = _class_probabilities(logits, calibration.temperature)
    decision = CLASS_NAMES[class_index]
    capability_id = candidates[0].id if decision == "CALL" else None
    validate_decision({"decision": decision, "capability_id": capability_id}, {record.id for record in snapshot.records})
    return RoutingResult(
        decision,
        capability_id,
        {
            "dropped_history_count": formatted.dropped_history_count,
            "path": path,
            "rerank_invocations": len(candidates) if reranked else 0,
            "neural_passes": 1 + (len(candidates) if reranked else 1),
            "temperature": calibration.temperature,
            "probabilities_calibrated": calibration.calibrated,
            "decision_probabilities": {name: probabilities[index] for index, name in enumerate(("CALL", "NO_CALL", "CLARIFY"))},
            "retrieval_scores": {record.id: score for record, score in ranked},
            "rerank_scores": {record.id: score for record, score in final} if reranked else {},
        },
    )


def compare_routing_paths(request: Mapping[str, Any], snapshot: RegistrySnapshot, index: RegistryIndex, tokenizer: RawByteBPE, callbacks: RoutingCallbacks, *, calibration: Calibration | None = None) -> dict[str, Any]:
    """Run paired conditional and always-rerank paths on the same snapshot."""
    conditional = route(request, snapshot, index, tokenizer, callbacks, calibration=calibration)
    always = route(request, snapshot, index, tokenizer, callbacks, calibration=Calibration(temperature=(calibration.temperature if calibration else 1.0), calibrated=False))
    return {"conditional": conditional.decision_json(), "always_rerank": always.decision_json(), "same_decision": conditional.decision_json() == always.decision_json(), "conditional_metadata": conditional.metadata, "always_metadata": always.metadata}
