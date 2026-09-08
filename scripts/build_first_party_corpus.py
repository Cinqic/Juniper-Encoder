#!/usr/bin/env python3
"""Materialize the reproducible, first-party tokenizer corpus snapshot.

The snapshot is deliberately generated from repository-owned technical text;
it is not a scrape, model distillation output, or evaluation-label source.
The fixed path list and canonical JSONL encoding make repeat builds auditable.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOTS = ("src", "docs", "configs", "scripts", "schemas", "spec")
ROOT_FILES = ("README.md", "CHANGELOG.md", "LICENSE")
SUFFIXES = frozenset({".py", ".md", ".yaml", ".json", ".sh", ".csv"})


def source_paths(root: Path) -> list[Path]:
    paths = [root / item for item in ROOT_FILES if (root / item).is_file()]
    for name in ROOTS:
        directory = root / name
        paths.extend(
            path
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUFFIXES and "__pycache__" not in path.parts
        )
    return sorted(paths)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/first_party/repository-corpus.jsonl")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in source_paths(Path.cwd()):
        text = path.read_text(encoding="utf-8", errors="strict")
        row = {
            "text": text,
            "source_path": path.as_posix(),
            "material_kind": "first_party_repository_technical_text",
            "corpus_role": "tokenizer_training_only",
        }
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(json.dumps({"status": "BUILT", "documents": len(lines), "bytes": destination.stat().st_size, "output": destination.as_posix()}, separators=(",", ":")))


if __name__ == "__main__":
    main()
