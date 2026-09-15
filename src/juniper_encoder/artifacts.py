"""Content-addressed manifests and safe JSON/tensor artifact checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import canonical_json_bytes, sha256_file, sha256_tree, write_json


def make_sha256sums(root: str | Path, output: str | Path) -> dict[str, str]:
    files = sha256_tree(root, exclude={Path(output).name})
    lines = [f"{digest}  {name}" for name, digest in sorted(files.items())]
    Path(output).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return files


def verify_sha256sums(root: str | Path, sums_path: str | Path) -> dict[str, Any]:
    expected: dict[str, str] = {}
    for line in Path(sums_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError("invalid SHA256SUMS line")
        digest, relative = parts
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("checksum path escapes artifact root")
        expected[relative] = digest
    actual = {relative: sha256_file(Path(root) / relative) for relative in expected}
    mismatches = {relative: {"expected": expected[relative], "actual": actual[relative]} for relative in expected if expected[relative] != actual[relative]}
    if mismatches:
        raise ValueError(json.dumps(mismatches, sort_keys=True))
    return {"status": "PASS", "files": len(expected)}


def candidate_identity(payload: dict[str, Any]) -> str:
    from .utils import sha256_bytes

    return sha256_bytes(canonical_json_bytes(payload))


def package_descriptor(output: str | Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    result = dict(descriptor)
    result["candidate_id"] = candidate_identity(descriptor)
    write_json(output, result)
    return result


def package_candidate(output: str | Path, *, source_sha: str, checkpoint: str | Path, export_directories: list[str | Path], manifests: dict[str, str | Path] | None = None, status: str = "CANDIDATE_PROVISIONAL_NOT_RELEASED") -> dict[str, Any]:
    """Create a content-addressed candidate directory from existing artifacts."""
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    import shutil

    sources = [Path(checkpoint), *(Path(item) for item in export_directories)]
    if manifests:
        sources.extend(Path(item) for item in manifests.values())
    for source in sources:
        if not source.exists():
            raise ValueError(f"candidate input does not exist: {source}")
        target_name = source.name if source.is_file() else source.name
        target = destination / target_name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        files = sorted(item for item in target.rglob("*") if item.is_file()) if target.is_dir() else [target]
        for path in files:
            copied[path.relative_to(destination).as_posix()] = sha256_file(path)
    descriptor = {"artifact_type": "juniper-encoder-candidate", "status": status, "source_sha": source_sha, "files": copied}
    descriptor["candidate_id"] = candidate_identity(descriptor)
    write_json(destination / "candidate.json", descriptor)
    make_sha256sums(destination, destination / "SHA256SUMS")
    return descriptor
