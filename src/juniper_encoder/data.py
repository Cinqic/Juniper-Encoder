"""Lawful data-manifest, provenance, deterministic split, and leakage checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .errors import EncoderError
from .utils import canonical_json_bytes, load_config, sha256_file, write_json


REQUIRED_SOURCE_FIELDS = (
    "source_id", "revision", "url", "upstream_sha256", "retrieval_date", "rights_evidence",
    "license", "attribution", "permitted_use", "redistribution_status", "language", "domain", "estimated_bytes",
)


def validate_source_manifest(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("approved source manifest must be an array")
    for source in payload:
        if not isinstance(source, dict) or any(not source.get(field) and source.get(field) != 0 for field in REQUIRED_SOURCE_FIELDS):
            raise ValueError("every source needs immutable identity and reviewed rights evidence")
        if source["rights_evidence"] in ("unknown", "conflicting") or source["redistribution_status"] == "unknown":
            raise EncoderError("BLOCKED_DATA_RIGHTS", "source rights are unknown or conflicting", {"source_id": source.get("source_id")})
    return payload


def plan(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_path = Path(config["source_manifest"])
    sources = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else []
    result = {"status": "PLANNED", "config": str(config_path), "config_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(), "sources": sources}
    write_json(output, result)
    return result


def acquire(manifest_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_source_manifest(payload)
    if not payload:
        raise EncoderError("BLOCKED_DATA", "no source is approved for acquisition")
    return {"status": "ACQUISITION_AUTHORIZED", "source_count": len(payload), "source_ids": [item["source_id"] for item in payload]}


def deterministic_split(records: Iterable[dict[str, Any]], ratios: dict[str, float], seed: int = 1729) -> list[dict[str, Any]]:
    import random
    values = [dict(record) for record in records]
    for record in values:
        if "record_id" not in record:
            raise ValueError("split records need immutable record_id")
    # Family grouping is upstream-owned; the stable hash is deterministic and does not
    # expose test records to training code.
    boundaries: list[tuple[str, float]] = []
    cursor = 0.0
    for split, fraction in ratios.items():
        cursor += fraction
        boundaries.append((split, cursor))
    for record in sorted(values, key=lambda item: hashlib.sha256(f"{seed}:{item['record_id']}".encode()).hexdigest()):
        point = int(hashlib.sha256(f"{seed}:{record['record_id']}".encode()).hexdigest()[:12], 16) / float(16**12)
        split = next(name for name, boundary in boundaries if point < boundary)
        record["split"] = split
    return values


def audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(records)
    ids: dict[str, str] = {}
    leaks: list[dict[str, str]] = []
    for record in items:
        content_hash = record.get("content_sha256")
        if content_hash and content_hash in ids and ids[content_hash] != record.get("split"):
            leaks.append({"content_sha256": content_hash, "first_split": ids[content_hash], "split": record.get("split", "")})
        if content_hash:
            ids[content_hash] = record.get("split", "")
    return {"status": "FAIL" if leaks else "PASS", "records": len(items), "leaks": leaks}
