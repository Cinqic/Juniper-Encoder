"""Minimal dependency-free PEP 517 backend for the contract-only package."""

from __future__ import annotations

import base64
import hashlib
import os
import zipfile
from pathlib import Path


NAME = "juniper_encoder"
VERSION = "0.1.0"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _write_metadata(Path(metadata_directory))


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _write_metadata(Path(metadata_directory))


def _metadata_text() -> str:
    return "Metadata-Version: 2.1\nName: juniper-encoder\nVersion: 0.1.0\nSummary: Contract-first Juniper request router\nRequires-Python: >=3.12,<3.13\n"


def _write_metadata(root: Path) -> str:
    destination = root / DIST_INFO
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (destination / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: juniper-local-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n", encoding="utf-8")
    return DIST_INFO


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    wheel_path = Path(wheel_directory) / f"{NAME}-{VERSION}-py3-none-any.whl"
    source = str((Path(__file__).parent / "src").resolve())
    files = {
        f"{NAME}.pth": source + "\n",
        f"{DIST_INFO}/METADATA": _metadata_text(),
        f"{DIST_INFO}/WHEEL": "Wheel-Version: 1.0\nGenerator: juniper-local-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    return _write_wheel(wheel_path, files)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    wheel_path = Path(wheel_directory) / f"{NAME}-{VERSION}-py3-none-any.whl"
    files: dict[str, bytes | str] = {
        f"{DIST_INFO}/METADATA": _metadata_text(),
        f"{DIST_INFO}/WHEEL": "Wheel-Version: 1.0\nGenerator: juniper-local-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    for path in sorted((Path(__file__).parent / "src" / NAME).rglob("*.py")):
        files[f"{NAME}/{path.relative_to(Path(__file__).parent / 'src' / NAME).as_posix()}"] = path.read_bytes()
    return _write_wheel(wheel_path, files)


def _write_wheel(path: Path, files: dict[str, bytes | str]) -> str:
    records: list[str] = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(files.items()):
            data = value.encode("utf-8") if isinstance(value, str) else value
            archive.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            records.append(f"{name},sha256={digest},{len(data)}")
        records.append(f"{DIST_INFO}/RECORD,,")
        archive.writestr(f"{DIST_INFO}/RECORD", "\n".join(records) + "\n")
    return path.name
