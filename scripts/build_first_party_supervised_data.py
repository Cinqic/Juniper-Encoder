#!/usr/bin/env python3
"""Build auditable first-party routing supervision and model-input rows.

All text and labels in this fixture are authored in this file from the frozen
registry/decision semantics. No frontier-model output, hidden distillation, or
sealed-test feedback is used. Held-out capability families are emitted only to
development/calibration/test partitions and never to the training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from juniper_encoder.constants import CLASS_NAMES, STRUCTURAL_IDS
from juniper_encoder.formatting import RegistryRecord, format_decision, format_generic_pair, format_request
from juniper_encoder.model import DeploymentModel
from juniper_encoder.registry import RegistrySnapshot
from juniper_encoder.tokenizer import RawByteBPE
from juniper_encoder.utils import canonical_json_bytes, sha256_file, write_json


CAPABILITIES = (
    {"id": "calendar", "type": "tool", "name": "Calendar", "purpose": "create and inspect calendar events and schedules", "supported_requests": ["schedule a meeting", "check an appointment", "move an event"], "exclusions": ["general time questions without an event"], "required_inputs": ["date or time", "event details"]},
    {"id": "code-review", "type": "skill", "name": "Code Review", "purpose": "inspect source code and explain defects or safe changes", "supported_requests": ["review code", "find a bug", "explain a compiler error"], "exclusions": ["executing arbitrary code"], "required_inputs": ["source code or error"]},
    {"id": "email", "type": "tool", "name": "Email", "purpose": "draft, search, and organize email messages", "supported_requests": ["find an email", "draft a message", "summarize a thread"], "exclusions": ["sending without confirmation"], "required_inputs": ["recipient or search terms"]},
    {"id": "file-search", "type": "tool", "name": "File Search", "purpose": "locate and inspect files in connected storage", "supported_requests": ["find a document", "search files", "locate a report"], "exclusions": ["deleting files"], "required_inputs": ["file name or search terms"]},
    {"id": "maps", "type": "agent", "name": "Maps", "purpose": "find places and plan routes between locations", "supported_requests": ["find a place", "plan a route", "estimate travel time"], "exclusions": ["booking transportation"], "required_inputs": ["place or origin and destination"]},
    {"id": "weather", "type": "tool", "name": "Weather", "purpose": "provide current and forecast weather for a location", "supported_requests": ["check weather", "get a forecast", "compare temperatures"], "exclusions": ["climate research reports"], "required_inputs": ["location"]},
    {"id": "music", "type": "skill", "name": "Music", "purpose": "search music and assemble playback queues", "supported_requests": ["find a song", "make a playlist", "play an album"], "exclusions": ["music licensing advice"], "required_inputs": ["artist, song, or mood"]},
    {"id": "shopping", "type": "agent", "name": "Shopping", "purpose": "compare products and track purchase options", "supported_requests": ["compare products", "find an item", "check a price"], "exclusions": ["placing an order without confirmation"], "required_inputs": ["product or constraints"]},
)

CALL_TEMPLATES = (
    "Please {verb} {subject}.",
    "Can you help me {verb} {subject}?",
    "I need the registered capability to {verb} {subject}.",
    "Use the appropriate tool to {verb} {subject}.",
)

CALL_SUBJECTS = {
    "calendar": ("schedule a meeting tomorrow at 3 PM", "check my appointment for Friday", "move the team event to Monday"),
    "code-review": ("review this function for defects", "explain why this test fails", "inspect the patch for a safe fix"),
    "email": ("find the message from Alex", "draft a reply about the project", "summarize the latest thread"),
    "file-search": ("find the quarterly report", "search connected files for the design note", "locate the deployment checklist"),
    "maps": ("find a pharmacy near downtown", "plan a route from home to the office", "estimate travel time to the station"),
    "weather": ("check the weather in Detroit", "get tomorrow's forecast for Toronto", "compare temperatures in two cities"),
    "music": ("find a jazz playlist", "make a quiet study queue", "play the latest album by that artist"),
    "shopping": ("compare two noise-cancelling headphones", "find a durable travel backpack", "check the current price of a monitor"),
}

NO_CALL_TEXT = (
    "Hello, how are you today?",
    "Explain what a byte is in simple terms.",
    "Give me a short reflective writing prompt.",
    "What is the difference between a tool and a skill?",
    "Tell me a harmless fact about trees.",
    "I would like to think through my priorities.",
)

CLARIFY_TEXT = (
    "Please schedule it.",
    "Find the thing I mentioned earlier.",
    "Send the message to them.",
    "Check the forecast there.",
    "Compare the best options.",
    "Review this and fix it.",
)


def _record_id(prefix: str, index: int) -> str:
    return hashlib.sha256(f"first-party-routing-v1:{prefix}:{index}".encode("utf-8")).hexdigest()


def _raw_rows() -> tuple[list[dict[str, Any]], list[RegistryRecord]]:
    records = [RegistryRecord.from_mapping(item) for item in CAPABILITIES]
    rows: list[dict[str, Any]] = []
    index = 0
    for capability in records:
        subjects = CALL_SUBJECTS[capability.id]
        for template_index, template in enumerate(CALL_TEMPLATES):
            subject = subjects[template_index % len(subjects)]
            text = template.format(verb="handle the request to", subject=subject)
            split = "train" if capability.id in {item["id"] for item in CAPABILITIES[:6]} and template_index < 3 else "development"
            if capability.id in {"music", "shopping"}:
                split = "test"
            rows.append({"record_id": _record_id("call", index), "request": {"history": [], "user": text}, "label": "CALL", "capability_id": capability.id, "candidate_ids": [capability.id, records[(records.index(capability) + 1) % len(records)].id], "capability_family": capability.id, "split": split, "material_kind": "first_party_authored_routing_example"})
            index += 1
    for offset, text in enumerate(NO_CALL_TEXT):
        rows.append({"record_id": _record_id("no-call", offset), "request": {"history": [], "user": text}, "label": "NO_CALL", "capability_id": None, "candidate_ids": ["calendar", "weather"], "capability_family": "none", "split": "train" if offset < 4 else "development", "material_kind": "first_party_authored_routing_example"})
    for offset, text in enumerate(CLARIFY_TEXT):
        candidate_pair = ["calendar", "email"] if offset % 2 == 0 else ["weather", "maps"]
        rows.append({"record_id": _record_id("clarify", offset), "request": {"history": [], "user": text}, "label": "CLARIFY", "capability_id": None, "candidate_ids": candidate_pair, "capability_family": candidate_pair[0], "split": "train" if offset < 4 else "development", "material_kind": "first_party_authored_routing_example"})
    return rows, records


def _load_model(checkpoint: str | None) -> DeploymentModel | None:
    if not checkpoint:
        return None
    import torch
    from safetensors.torch import load_file

    model = DeploymentModel(seed=1729)
    model.load_state_dict(load_file(str(Path(checkpoint) / "model.safetensors"), device="cpu"), strict=True)
    return model.eval()


def _pad(rows: list[list[int]], width: int | None = None) -> list[list[int]]:
    width = width or max(len(row) for row in rows)
    return [row + [STRUCTURAL_IDS["[PAD]"]] * (width - len(row)) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    tokenizer = RawByteBPE.load(args.tokenizer)
    rows, records = _raw_rows()
    by_id = {record.id: record for record in records}
    model = _load_model(args.checkpoint)
    encoder_variant_identity = f"checkpoint:{sha256_file(Path(args.checkpoint) / 'model.safetensors')}" if args.checkpoint else "foundation-pending"
    snapshot = RegistrySnapshot.create(records, tokenizer_identity=tokenizer.identity(), encoder_variant_identity=encoder_variant_identity)
    retrieval_targets: dict[str, list[float]] = {}
    if model is not None:
        import torch
        with torch.inference_mode():
            for record in records:
                tokens = [STRUCTURAL_IDS["[CLS]"], STRUCTURAL_IDS["[DOCUMENT]"], *tokenizer.encode(record.serialized().decode("utf-8")), STRUCTURAL_IDS["[SEP]"]]
                retrieval_targets[record.id] = model.retrieval_vector(torch.tensor([tokens], dtype=torch.long))[0].tolist()
    classifier_rows: list[dict[str, Any]] = []
    reranker_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for row in rows:
        formatted = format_request(row["request"], tokenizer)
        candidates = [by_id[identifier] for identifier in row["candidate_ids"]]
        decision = format_decision(formatted.serialized, candidates, tokenizer)
        classifier_rows.append({**row, "input_ids": list(decision.token_ids), "label_index": CLASS_NAMES.index(row["label"]), "tokenizer_identity": tokenizer.identity(), "registry_snapshot_identity": snapshot.snapshot_identity})
        pairs = [format_generic_pair(formatted.serialized, candidate, tokenizer, allow_document_truncation=False).token_ids for candidate in candidates]
        if row["label"] == "CALL":
            reranker_rows.append({**row, "input_ids": [list(pair) for pair in pairs], "target_index": row["candidate_ids"].index(row["capability_id"]), "tokenizer_identity": tokenizer.identity(), "registry_snapshot_identity": snapshot.snapshot_identity})
        if model is not None and row["label"] == "CALL":
            positive = retrieval_targets[row["capability_id"]]
            negatives = [retrieval_targets[candidate.id] for candidate in candidates if candidate.id != row["capability_id"]]
            retrieval_rows.append({**row, "input_ids": list(formatted.token_ids), "positive_vector": positive, "negative_vectors": negatives, "tokenizer_identity": tokenizer.identity(), "registry_snapshot_identity": snapshot.snapshot_identity})
    for collection, field in ((classifier_rows, "input_ids"), (retrieval_rows, "input_ids")):
        if collection:
            width = max(len(row[field]) for row in collection)
            for row in collection:
                row[field] = row[field] + [STRUCTURAL_IDS["[PAD]"]] * (width - len(row[field]))
    if reranker_rows:
        pair_width = max(len(pair) for row in reranker_rows for pair in row["input_ids"])
        for row in reranker_rows:
            row["input_ids"] = _pad(row["input_ids"], pair_width)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "registry.json", [record.as_mapping() for record in records])
    write_json(destination / "requests.json", {"status": "AUTHORED", "records": rows, "registry_snapshot_identity": snapshot.snapshot_identity})
    for name, values in (("classifier", classifier_rows), ("reranker", reranker_rows), ("retrieval", retrieval_rows)):
        write_json(destination / f"{name}.json", {"status": "TOKENIZED", "records": values, "tokenizer_identity": tokenizer.identity(), "registry_snapshot_identity": snapshot.snapshot_identity, "source_sha256": sha256_file(__file__)})
    write_json(destination / "manifest.json", {"status": "AUTHORED", "material_kind": "first_party_authored_routing_fixture", "record_count": len(rows), "training_records": sum(row["split"] == "train" for row in rows), "development_records": sum(row["split"] == "development" for row in rows), "test_records": sum(row["split"] == "test" for row in rows), "held_out_capabilities": ["music", "shopping"], "tokenizer_identity": tokenizer.identity(), "registry_snapshot_identity": snapshot.snapshot_identity, "encoder_variant_identity": encoder_variant_identity, "checkpoint_identity": sha256_file(Path(args.checkpoint) / "model.safetensors") if args.checkpoint else None, "generator_sha256": sha256_file(__file__)})
    print(json.dumps({"status": "BUILT", "records": len(rows), "training": sum(row["split"] == "train" for row in rows), "development": sum(row["split"] == "development" for row in rows), "test": sum(row["split"] == "test" for row in rows), "tokenizer_identity": tokenizer.identity(), "registry_snapshot_identity": snapshot.snapshot_identity}, separators=(",", ":")))


if __name__ == "__main__":
    main()
