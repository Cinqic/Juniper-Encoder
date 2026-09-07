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
        digest, relative = line.split("  ", 1)
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
