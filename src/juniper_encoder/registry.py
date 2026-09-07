"""Immutable registry snapshots and exact FP32 normalized-dot-product indexes."""

from __future__ import annotations

import dataclasses
import math
from array import array
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .errors import index_mismatch, invalid_registry, numerical_error
from .formatting import RegistryRecord, parse_registry_json
from .utils import canonical_json_bytes, require_finite, sha256_bytes, write_json


@dataclasses.dataclass(frozen=True)
class RegistrySnapshot:
    records: tuple[RegistryRecord, ...]
    tokenizer_identity: str
    encoder_variant_identity: str
    snapshot_identity: str

    @classmethod
    def create(
        cls,
        records: Iterable[RegistryRecord],
        *,
        tokenizer_identity: str,
        encoder_variant_identity: str,
    ) -> "RegistrySnapshot":
        ordered = tuple(records)
        ids = [record.id for record in ordered]
        if len(ids) != len(set(ids)):
            raise invalid_registry("registry IDs must be unique")
        identity_payload = {
            "records": [record.as_mapping() for record in ordered],
            "tokenizer_identity": tokenizer_identity,
            "encoder_variant_identity": encoder_variant_identity,
        }
        identity = sha256_bytes(canonical_json_bytes(identity_payload))
        return cls(ordered, tokenizer_identity, encoder_variant_identity, identity)

    @classmethod
    def from_json(cls, path: str | Path, *, tokenizer_identity: str, encoder_variant_identity: str) -> "RegistrySnapshot":
        records = parse_registry_json(Path(path).read_bytes())
        return cls.create(records, tokenizer_identity=tokenizer_identity, encoder_variant_identity=encoder_variant_identity)

    def as_payload(self) -> dict:
        return {
            "records": [record.as_mapping() for record in self.records],
            "tokenizer_identity": self.tokenizer_identity,
            "encoder_variant_identity": self.encoder_variant_identity,
            "snapshot_identity": self.snapshot_identity,
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.as_payload())

@dataclasses.dataclass(frozen=True)
class RegistryIndex:
    snapshot_identity: str
    tokenizer_identity: str
    encoder_variant_identity: str
    dimension: int
    vectors: tuple[tuple[float, ...], ...]
    index_identity: str

    @classmethod
    def build(
        cls,
        snapshot: RegistrySnapshot,
        encoder: Callable[[RegistryRecord], Sequence[float]],
        *,
        dimension: int = 256,
    ) -> "RegistryIndex":
        vectors: list[tuple[float, ...]] = []
        for record in snapshot.records:
            raw = tuple(float(value) for value in encoder(record))
            if len(raw) != dimension:
                raise index_mismatch("registry vector dimension mismatch", capability_id=record.id, expected=dimension, actual=len(raw))
            require_finite(raw, f"registry vector {record.id}")
            norm = math.sqrt(math.fsum(value * value for value in raw))
            if norm < 1e-12:
                raise numerical_error("registry vector norm is below threshold", capability_id=record.id)
            vectors.append(tuple(value / norm for value in raw))
        payload = {
            "snapshot_identity": snapshot.snapshot_identity,
            "tokenizer_identity": snapshot.tokenizer_identity,
            "encoder_variant_identity": snapshot.encoder_variant_identity,
            "dimension": dimension,
            "vectors": [list(vector) for vector in vectors],
        }
        return cls(
            snapshot.snapshot_identity,
            snapshot.tokenizer_identity,
            snapshot.encoder_variant_identity,
            dimension,
            tuple(vectors),
            sha256_bytes(canonical_json_bytes(payload)),
        )

    def verify_compatible(self, snapshot: RegistrySnapshot, *, tokenizer_identity: str, encoder_variant_identity: str) -> None:
        if (
            self.snapshot_identity != snapshot.snapshot_identity
            or self.tokenizer_identity != tokenizer_identity
            or self.encoder_variant_identity != encoder_variant_identity
            or len(self.vectors) != len(snapshot.records)
        ):
            raise index_mismatch(
                "index does not match the active registry/tokenizer/encoder variant",
                index_snapshot=self.snapshot_identity,
                active_snapshot=snapshot.snapshot_identity,
            )
        for vector in self.vectors:
            if len(vector) != self.dimension:
                raise index_mismatch("index contains a vector with the wrong dimension")
            require_finite(vector, "index vector")

    def search(self, query: Sequence[float], snapshot: RegistrySnapshot, *, limit: int = 8) -> list[tuple[RegistryRecord, float]]:
        self.verify_compatible(snapshot, tokenizer_identity=self.tokenizer_identity, encoder_variant_identity=self.encoder_variant_identity)
        query_values = tuple(float(value) for value in query)
        if len(query_values) != self.dimension:
            raise index_mismatch("query vector dimension mismatch", expected=self.dimension, actual=len(query_values))
        require_finite(query_values, "query vector")
        norm = math.sqrt(math.fsum(value * value for value in query_values))
        if norm < 1e-12:
            raise numerical_error("query vector norm is below threshold")
        normalized = tuple(value / norm for value in query_values)
        scored = [
            (snapshot.records[i], math.fsum(normalized[j] * self.vectors[i][j] for j in range(self.dimension)))
            for i in range(len(snapshot.records))
        ]
        require_finite((score for _, score in scored), "search score")
        scored.sort(key=lambda item: (-item[1], item[0].id.encode("ascii")))
        return scored[: min(limit, len(scored))]

    def as_payload(self) -> dict:
        return {
            "snapshot_identity": self.snapshot_identity,
            "tokenizer_identity": self.tokenizer_identity,
            "encoder_variant_identity": self.encoder_variant_identity,
            "dimension": self.dimension,
            "vectors": [list(vector) for vector in self.vectors],
            "index_identity": self.index_identity,
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.as_payload())
