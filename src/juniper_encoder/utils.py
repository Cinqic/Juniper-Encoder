"""Small dependency-free helpers used for identity and safe serialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .errors import numerical_error


def canonical_json_bytes(value: Any) -> bytes:
    """The single JSON convention used for all semantic payloads."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8", "strict")


def strict_json_loads(text: str | bytes) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=hook, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: str | os.PathLike[str], *, exclude: Iterable[str] = ()) -> dict[str, str]:
    root_path = Path(root)
    excluded = set(exclude)
    result: dict[str, str] = {}
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        relative = path.relative_to(root_path).as_posix()
        if relative not in excluded:
            result[relative] = sha256_file(path)
    return result


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_finite(values: Iterable[float], label: str) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(float(value)):
            raise numerical_error(f"nonfinite {label}", index=index, value=repr(value))


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load JSON-compatible YAML files without silently accepting loose syntax."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        value = strict_json_loads(raw)
    except (ValueError, json.JSONDecodeError):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(f"{path} is not JSON and PyYAML is not installed") from exc
        value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError(f"config {path} must contain an object")
    return value


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")
