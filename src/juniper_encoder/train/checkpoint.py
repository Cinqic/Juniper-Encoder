"""Safe checkpoint metadata and identity checks."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ..errors import EncoderError
from ..utils import canonical_json_bytes, sha256_bytes, write_json


@dataclasses.dataclass(frozen=True)
class CheckpointMetadata:
    format_version: str
    source_sha: str
    config_hashes: dict[str, str]
    rng_states: dict[str, str]
    successful_updates: int
    attempted_updates: int
    skipped_updates: int
    consumed_tokens: int
    sampler_identity: str
    tokenizer_identity: str
    data_identity: str
    predecessor_identity: str | None

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def identity(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def validate_checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"format_version", "source_sha", "config_hashes", "rng_states", "predecessor_identity"}
    missing = sorted(required - set(payload))
    if missing:
        raise EncoderError("INVALID_INPUT", "checkpoint metadata is incomplete", {"missing": missing})
    return {"status": "VALID", "identity": sha256_bytes(canonical_json_bytes(payload))}


def write_metadata(path: str | Path, metadata: CheckpointMetadata) -> dict[str, Any]:
    payload = metadata.payload()
    result = {"metadata": payload, "checkpoint_identity": metadata.identity(), "weights_format": "safe-tensor-payload-required"}
    write_json(path, result)
    return result
