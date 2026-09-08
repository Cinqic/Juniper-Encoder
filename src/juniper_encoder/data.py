"""Bounded, provenance-preserving data acquisition and deterministic splitting.

The data pipeline deliberately has no implicit network source. A source must
be present in the reviewed manifest, carry rights evidence, and identify an
immutable byte payload. The implementation accepts JSONL, JSON arrays, plain
UTF-8 text, and safely extracts common archives without following links or
allowing archive members to escape the destination directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import EncoderError
from .utils import canonical_json_bytes, sha256_bytes, sha256_file, write_json


REQUIRED_SOURCE_FIELDS = (
    "source_id", "revision", "url", "upstream_sha256", "retrieval_date", "rights_evidence",
    "license", "attribution", "permitted_use", "redistribution_status", "language", "domain", "estimated_bytes",
)
DEFAULT_MAX_BYTES = 1_073_741_824
DEFAULT_MAX_EXTRACTED_BYTES = 5_368_709_120
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def validate_source_manifest(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("approved source manifest must be an array")
    seen: set[str] = set()
    for source in payload:
        if not isinstance(source, dict) or any(not source.get(field) and source.get(field) != 0 for field in REQUIRED_SOURCE_FIELDS):
            raise ValueError("every source needs immutable identity and reviewed rights evidence")
        source_id = source["source_id"]
        if not isinstance(source_id, str) or source_id in seen:
            raise ValueError("source IDs must be unique nonempty strings")
        seen.add(source_id)
        if not isinstance(source["estimated_bytes"], int) or source["estimated_bytes"] < 0:
            raise ValueError("estimated_bytes must be a nonnegative integer")
        if source["rights_evidence"] in ("unknown", "conflicting") or source["redistribution_status"] == "unknown":
            raise EncoderError("BLOCKED_DATA_RIGHTS", "source rights are unknown or conflicting", {"source_id": source_id})
        digest = source["upstream_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("upstream_sha256 must be a 64-character hexadecimal digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("upstream_sha256 must be hexadecimal") from exc
    return payload


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    from .utils import load_config

    config = load_config(config_path)
    source_path = Path(config["source_manifest"])
    sources = validate_source_manifest(_load_json(source_path)) if source_path.exists() else []
    result = {
        "status": "PLANNED" if sources else "BLOCKED_DATA",
        "config": str(config_path),
        "config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path) if source_path.exists() else None,
        "sources": sources,
        "reason": None if sources else "no approved source is present",
    }
    write_json(output, result)
    return result


def _download(url: str, destination: Path, *, max_bytes: int, retries: int = 3, timeout: int = 60) -> int:
    if not url.startswith(("http://", "https://")):
        raise EncoderError("BLOCKED_DATA", "source URL must use HTTP(S)", {"url": url})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "juniper-encoder/0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > max_bytes:
                    raise EncoderError("BLOCKED_DATA_SIZE", "source exceeds acquisition size limit", {"content_length": int(content_length), "max_bytes": max_bytes})
                written = 0
                with destination.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > max_bytes:
                            raise EncoderError("BLOCKED_DATA_SIZE", "source exceeded acquisition size limit", {"bytes": written, "max_bytes": max_bytes})
                        handle.write(block)
                return written
        except EncoderError:
            raise
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise EncoderError("BLOCKED_DATA_ACQUISITION", "source download failed after retries", {"url": url, "reason": repr(last_error)})


def _safe_member_path(root: Path, member_name: str) -> Path:
    if not member_name or Path(member_name).is_absolute():
        raise EncoderError("BLOCKED_DATA_ARCHIVE", "archive contains an absolute member path", {"member": member_name})
    destination = (root / member_name).resolve()
    if root.resolve() != destination and root.resolve() not in destination.parents:
        raise EncoderError("BLOCKED_DATA_ARCHIVE", "archive member escapes extraction root", {"member": member_name})
    return destination


def _safe_extract(archive: Path, destination: Path, *, max_bytes: int) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    names: list[str] = []
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if member.is_dir():
                    continue
                path = _safe_member_path(destination, member.filename)
                extracted += member.file_size
                if extracted > max_bytes:
                    raise EncoderError("BLOCKED_DATA_ARCHIVE", "archive exceeds extracted-size limit", {"max_extracted_bytes": max_bytes})
                path.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                names.append(path.relative_to(destination).as_posix())
        return names
    if not tarfile.is_tarfile(archive):
        return []
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise EncoderError("BLOCKED_DATA_ARCHIVE", "archive links are not permitted", {"member": member.name})
            if not member.isfile():
                continue
            path = _safe_member_path(destination, member.name)
            extracted += member.size
            if extracted > max_bytes:
                raise EncoderError("BLOCKED_DATA_ARCHIVE", "archive exceeds extracted-size limit", {"max_extracted_bytes": max_bytes})
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = handle.extractfile(member)
            if stream is None:
                raise EncoderError("BLOCKED_DATA_ARCHIVE", "archive member could not be read", {"member": member.name})
            with stream, path.open("wb") as target:
                shutil.copyfileobj(stream, target, length=1024 * 1024)
            names.append(path.relative_to(destination).as_posix())
    return names


def acquire(manifest_path: str | Path, *, output_root: str | Path = "data/raw", max_bytes: int = DEFAULT_MAX_BYTES, max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES) -> dict[str, Any]:
    payload = validate_source_manifest(_load_json(manifest_path))
    if not payload:
        raise EncoderError("BLOCKED_DATA", "no source is approved for acquisition")
    destination_root = Path(output_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for source in payload:
        source_dir = destination_root / source["source_id"]
        source_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(source["url"].split("?", 1)[0]).suffix or ".bin"
        upstream_path = source_dir / f"upstream{suffix}"
        local_path = source.get("local_path")
        if local_path:
            source_bytes = Path(local_path).read_bytes()
            if len(source_bytes) > max_bytes:
                raise EncoderError("BLOCKED_DATA_SIZE", "local source exceeds acquisition size limit", {"source_id": source["source_id"]})
            upstream_path.write_bytes(source_bytes)
        else:
            _download(source["url"], upstream_path, max_bytes=max_bytes)
        digest = sha256_file(upstream_path)
        if digest != source["upstream_sha256"]:
            raise EncoderError("BLOCKED_DATA_CHECKSUM", "source checksum does not match the reviewed manifest", {"source_id": source["source_id"], "expected": source["upstream_sha256"], "actual": digest})
        extracted_root = source_dir / "extracted"
        extracted = _safe_extract(upstream_path, extracted_root, max_bytes=max_extracted_bytes) if upstream_path.name.endswith(_ARCHIVE_SUFFIXES) else []
        artifacts.append({
            "source_id": source["source_id"], "revision": source["revision"], "url": source["url"],
            "upstream_sha256": digest, "path": upstream_path.as_posix(), "size": upstream_path.stat().st_size,
            "role": source.get("role", "general"),
            "extracted_root": extracted_root.as_posix() if extracted else None, "extracted_files": extracted,
        })
    result = {"status": "ACQUIRED", "manifest_sha256": sha256_file(manifest_path), "artifacts": artifacts}
    write_json(destination_root / "acquisition.json", result)
    return result


def _iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"acquisition.json", "prepared.json", "splits.json"}:
            continue
        yield path


def _records_from_file(path: Path) -> Iterator[tuple[str, str]]:
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value.get("text", value.get("content")) if isinstance(value, dict) else value
            if isinstance(text, str):
                yield f"{path.as_posix()}:{line_number}", text
        return
    if path.suffix.lower() == ".json":
        value = _load_json(path)
        values = value if isinstance(value, list) else [value]
        for index, item in enumerate(values):
            text = item.get("text", item.get("content")) if isinstance(item, dict) else item
            if isinstance(text, str):
                yield f"{path.as_posix()}:{index}", text
        return
    text = path.read_text(encoding="utf-8", errors="strict")
    if text:
        yield path.as_posix(), text


def _family_key(source_id: str, origin: str, text: str) -> tuple[str, str, str, str]:
    content_hash = sha256_bytes(text.encode("utf-8"))
    # JSONL rows are authored independent records; plain files yield one
    # record. The immutable origin (including line number when available) is
    # the conservative lineage boundary in the absence of upstream metadata.
    document_family = sha256_bytes(origin.encode("utf-8"))
    return source_id, document_family, document_family, content_hash


def prepare(acquisition_manifest: str | Path, output: str | Path, *, max_records: int | None = None) -> dict[str, Any]:
    acquisition = _load_json(acquisition_manifest)
    if acquisition.get("status") != "ACQUIRED":
        raise EncoderError("BLOCKED_DATA", "prepare requires a successful acquisition manifest")
    records: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    dropped: list[dict[str, Any]] = []
    for artifact in acquisition.get("artifacts", []):
        source_id = artifact["source_id"]
        root = Path(artifact.get("extracted_root") or artifact["path"])
        for path in _iter_files(root):
            try:
                source_records = list(_records_from_file(path))
            except (UnicodeError, json.JSONDecodeError) as exc:
                dropped.append({"path": path.as_posix(), "reason_code": "INVALID_ENCODING_OR_JSON", "detail": str(exc)})
                continue
            for origin, raw_text in source_records:
                text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
                if not text.strip():
                    dropped.append({"path": origin, "reason_code": "EMPTY_OR_WHITESPACE"})
                    continue
                content_hash = sha256_bytes(text.encode("utf-8"))
                family = _family_key(source_id, origin, text)
                if content_hash in seen_hashes:
                    dropped.append({"path": origin, "reason_code": "EXACT_DUPLICATE", "duplicate_of": seen_hashes[content_hash]})
                    continue
                seen_hashes[content_hash] = origin
                records.append({
                    "record_id": sha256_bytes(f"{source_id}:{origin}:{content_hash}".encode("utf-8")),
                    "source_id": source_id, "source_role": artifact.get("role", "general"),
                    "source_lineage": family[1], "document_family": family[1],
                    "conversation_family": family[2], "capability_family": family[3],
                    "origin": origin, "content_sha256": content_hash, "text": text,
                    "cleaning_reason_codes": ["CRLF_NORMALIZED"] if raw_text != text else [],
                })
                if max_records is not None and len(records) >= max_records:
                    break
            if max_records is not None and len(records) >= max_records:
                break
        if max_records is not None and len(records) >= max_records:
            break
    result = {
        "status": "PREPARED" if records else "BLOCKED_DATA",
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest),
        "records": records, "record_count": len(records), "dropped": dropped,
        "dropped_count": len(dropped), "content_hash_algorithm": "sha256-utf8-clean-text-v1",
    }
    if not records:
        result["reason"] = "no usable records remained after deterministic cleaning"
    write_json(output, result)
    return result


def deterministic_split(records: Iterable[dict[str, Any]], ratios: dict[str, float], seed: int = 1729) -> list[dict[str, Any]]:
    values = [dict(record) for record in records]
    if not values:
        return []
    if any("record_id" not in record or not isinstance(record["record_id"], str) for record in values):
        raise ValueError("split records need immutable record_id")
    if not ratios or any(float(value) < 0 for value in ratios.values()) or abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must be nonnegative and sum to one")
    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    group_fields = ("source_lineage", "document_family", "conversation_family", "capability_family")
    # Build connected components over every declared family/lineage key. If
    # two records share any relation, they are assigned as one unit before a
    # split boundary is sampled. This prevents a later field from undoing an
    # earlier family grouping.
    parent = list(range(len(values)))
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    owners: dict[tuple[str, str], int] = {}
    for index, record in enumerate(values):
        for field in group_fields:
            value = record.get(field)
            if value:
                relation = (field, json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                if relation in owners:
                    union(index, owners[relation])
                else:
                    owners[relation] = index
    for index, record in enumerate(values):
        components[find(index)].append(record)
    groups: dict[str, list[dict[str, Any]]] = {}
    for component in components.values():
        member_ids = sorted(record["record_id"] for record in component)
        family_key = sha256_bytes(canonical_json_bytes(["family-v1", member_ids]))
        groups[family_key] = component
    boundaries: list[tuple[str, float]] = []
    cursor = 0.0
    split_names = list(ratios)
    for split, fraction in ratios.items():
        cursor += float(fraction)
        boundaries.append((split, cursor))
    for key in sorted(groups):
        digest = sha256_bytes(canonical_json_bytes([seed, key]))
        point = int(digest[:16], 16) / float(16 ** 16)
        split = next(name for name, boundary in boundaries if point < boundary or name == split_names[-1])
        for record in groups[key]:
            record["family_key"] = key
            record["split"] = split
    return sorted(values, key=lambda item: item["record_id"])


def freeze(records_manifest: str | Path, output: str | Path, ratios: dict[str, float], seed: int = 1729) -> dict[str, Any]:
    payload = _load_json(records_manifest)
    records = payload.get("records", payload) if isinstance(payload, (dict, list)) else []
    excluded_roles = {"tokenizer_training_only", "training_only", "foundation_training_only"}
    excluded = [record for record in records if record.get("source_role") in excluded_roles]
    split_records = deterministic_split(
        [record for record in records if record.get("source_role") not in excluded_roles],
        ratios,
        seed,
    )
    result = {
        "status": "FROZEN" if split_records else "BLOCKED_DATA",
        "source_manifest_sha256": sha256_file(records_manifest),
        "records": split_records,
        "excluded_source_role_counts": {
            role: sum(record.get("source_role") == role for record in excluded)
            for role in sorted(excluded_roles)
            if any(record.get("source_role") == role for record in excluded)
        },
        "ratios": ratios,
        "seed": seed,
    }
    leakage = audit(split_records)
    result["leakage_audit"] = leakage
    if leakage["status"] != "PASS":
        raise EncoderError("BLOCKED_DATA_LEAKAGE", "frozen splits contain cross-split family leakage", leakage)
    write_json(output, result)
    return result


def audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(records)
    leaks: list[dict[str, Any]] = []
    for field in ("content_sha256", "source_lineage", "document_family", "conversation_family", "capability_family", "family_key"):
        owners: dict[str, str] = {}
        for record in items:
            value = record.get(field)
            split = record.get("split", "")
            if value is None:
                continue
            key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key in owners and owners[key] != split:
                leaks.append({"field": field, "value": key, "first_split": owners[key], "split": split})
            else:
                owners[key] = split
    return {"status": "FAIL" if leaks else "PASS", "records": len(items), "leaks": leaks}
