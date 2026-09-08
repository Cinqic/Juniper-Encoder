#!/usr/bin/env python3
"""Evaluate a materialized DeploymentModel through the frozen routing path.

The evaluator deliberately owns no labels: it consumes the first-party authored
request manifest, runs the real model callbacks, and writes raw predictions plus
the development-only threshold rows needed by the calibration CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from juniper_encoder.constants import CLASS_NAMES, STRUCTURAL_IDS
from juniper_encoder.formatting import RegistryRecord, format_decision, format_generic_pair, format_request
from juniper_encoder.registry import RegistryIndex, RegistrySnapshot
from juniper_encoder.routing import Calibration, RoutingCallbacks, compare_routing_paths, route
from juniper_encoder.tokenizer import RawByteBPE
from juniper_encoder.utils import sha256_file, write_json


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(path: str | Path) -> list[dict[str, Any]]:
    payload = _json(path)
    values = payload.get("records", payload.get("predictions", [])) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("request manifest must contain records")
    return values


def _slices(row: dict[str, Any], records: dict[str, RegistryRecord]) -> list[str]:
    names = [str(row["label"]), f"family:{row.get('capability_family', 'none')}"]
    capability_id = row.get("capability_id")
    if isinstance(capability_id, str) and capability_id in records:
        names.append(records[capability_id].type)
        if capability_id in {"music", "shopping"}:
            names.append("held-out-capability")
    if row.get("label") == "CLARIFY":
        names.append("missing-input")
    return names


def _load_model(checkpoint: str | Path, device: Any, *, quantized: bool = False) -> Any:
    import torch
    if quantized:
        from juniper_encoder.quantization import _load_quantized_model

        model = _load_quantized_model(checkpoint)
    else:
        from safetensors.torch import load_file
        from juniper_encoder.model import DeploymentModel

        model = DeploymentModel(seed=1729)
        model.load_state_dict(load_file(str(Path(checkpoint) / "model.safetensors"), device="cpu"), strict=True)
    return model.to(device).eval()


def _calibration(path: str | None, *, checkpoint_identity: str | None, tokenizer_identity: str, snapshot_identity: str) -> Calibration | None:
    if not path:
        return None
    payload = _json(path)
    expected = {"tokenizer_identity": tokenizer_identity, "registry_snapshot_identity": snapshot_identity}
    if checkpoint_identity is not None:
        expected["checkpoint_identity"] = checkpoint_identity
    mismatches = {key: {"expected": value, "actual": payload.get(key)} for key, value in expected.items() if payload.get(key) != value}
    if mismatches:
        raise ValueError(f"calibration identity mismatch: {mismatches}")
    return Calibration(
        temperature=float(payload["temperature"]),
        score_threshold=float(payload["tau_score"]),
        margin_threshold=float(payload["tau_margin"]),
        calibrated=True,
    )


def _sorted_scores(values: dict[str, Any]) -> list[str]:
    return [item[0] for item in sorted(values.items(), key=lambda item: (-float(item[1]), item[0].encode("ascii")))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = RawByteBPE.load(args.tokenizer)
    checkpoint_identity = sha256_file(Path(args.checkpoint) / "model.safetensors")
    model = _load_model(args.checkpoint, device, quantized=args.quantized)
    registry_payload = _json(args.registry)
    records = tuple(RegistryRecord.from_mapping(item) for item in registry_payload)
    by_id = {record.id: record for record in records}
    variant_identity = f"int8:{checkpoint_identity}" if args.quantized else f"checkpoint:{checkpoint_identity}"
    snapshot = RegistrySnapshot.create(records, tokenizer_identity=tokenizer.identity(), encoder_variant_identity=variant_identity)

    def tensor(ids: Any) -> Any:
        return torch.tensor([list(ids)], dtype=torch.long, device=device)

    def encode_record(record: RegistryRecord) -> list[float]:
        ids = (STRUCTURAL_IDS["[CLS]"], STRUCTURAL_IDS["[DOCUMENT]"], *tokenizer.encode(record.serialized().decode("utf-8")), STRUCTURAL_IDS["[SEP]"])
        with torch.inference_mode():
            return model.retrieval_vector(tensor(ids))[0].detach().cpu().tolist()

    index = RegistryIndex.build(snapshot, encode_record)
    calibration = _calibration(args.calibration, checkpoint_identity=None if args.quantized else checkpoint_identity, tokenizer_identity=tokenizer.identity(), snapshot_identity=snapshot.snapshot_identity)
    rows = _records(args.requests)
    predictions: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for row in rows:
        request = row["request"]
        formatted = format_request(request, tokenizer)
        probe: dict[str, Any] = {}

        def encode_query(value: Any) -> list[float]:
            with torch.inference_mode():
                return model.retrieval_vector(tensor(value.token_ids))[0].detach().cpu().tolist()

        def rerank_pair(ids: Any, record: RegistryRecord) -> float:
            del record
            with torch.inference_mode():
                return float(model.rerank(tensor(ids))[0].item())

        def classify(ids: Any) -> list[float]:
            with torch.inference_mode():
                values = model.classify(tensor(ids))[0].detach().cpu().tolist()
            probe["logits"] = values
            return values

        callbacks = RoutingCallbacks(encode_query=encode_query, rerank_pair=rerank_pair, classify=classify)
        with torch.inference_mode():
            result = route(request, snapshot, index, tokenizer, callbacks, calibration=calibration)
            main_logits = list(probe.get("logits", []))
            paired = compare_routing_paths(request, snapshot, index, tokenizer, callbacks, calibration=calibration)
            query_vector = encode_query(formatted)
        ranked = index.search(query_vector, snapshot, limit=8)
        ranking = [record.id for record, _ in ranked]
        retrieval_scores = {record.id: float(score) for record, score in ranked}
        relevant = [row["capability_id"]] if row.get("label") == "CALL" and isinstance(row.get("capability_id"), str) else []
        expected = {"decision": row["label"], "capability_id": row.get("capability_id")}
        actual = result.decision_json()
        prediction = {
            "record_id": row["record_id"],
            "split": row.get("split", "train"),
            "request": request,
            "label": row["label"],
            "label_index": CLASS_NAMES.index(row["label"]),
            "prediction": result.decision,
            "capability_id": result.capability_id,
            "expected_capability_id": row.get("capability_id"),
            "exact": actual == expected,
            "logits": main_logits,
            "ranking": ranking if relevant else None,
            "relevant": relevant if relevant else None,
            "relevance": {identifier: int(identifier in relevant) for identifier in ranking} if relevant else None,
            "reranked_ranking": _sorted_scores(result.metadata.get("rerank_scores", {})) if result.metadata.get("rerank_scores") else None,
            "reranker_relevance": {identifier: int(identifier in relevant) for identifier in result.metadata.get("rerank_scores", {})} if relevant and result.metadata.get("rerank_scores") else None,
            "retrieval_scores": retrieval_scores,
            "rerank_scores": result.metadata.get("rerank_scores", {}),
            "rerank_invoked": bool(result.metadata.get("rerank_invocations", 0)),
            "path": result.metadata.get("path"),
            "decision_probabilities": result.metadata.get("decision_probabilities", {}),
            "slices": _slices(row, by_id),
            "checkpoint_identity": checkpoint_identity,
            "tokenizer_identity": tokenizer.identity(),
            "registry_snapshot_identity": snapshot.snapshot_identity,
        }
        prediction = {key: value for key, value in prediction.items() if value is not None}
        prediction.update({
            "conditional_exact": paired["conditional"] == expected,
            "always_rerank_exact": paired["always_rerank"] == expected,
            "conditional_false_call": paired["conditional"]["decision"] == "CALL" and row["label"] != "CALL",
            "always_rerank_false_call": paired["always_rerank"]["decision"] == "CALL" and row["label"] != "CALL",
        })
        predictions.append(prediction)

        if row.get("split") == "development":
            top_score = float(ranked[0][1]) if ranked else 0.0
            margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else top_score
            fast = route(
                request,
                snapshot,
                index,
                tokenizer,
                callbacks,
                calibration=Calibration(score_threshold=top_score, margin_threshold=margin, calibrated=True),
            )
            always = route(request, snapshot, index, tokenizer, callbacks, calibration=Calibration())
            threshold_rows.append({
                "record_id": row["record_id"],
                "split": "development",
                "score": top_score,
                "margin": margin,
                "skip_safe": fast.decision_json() == always.decision_json(),
                "conditional_correct": fast.decision_json() == expected,
                "always_correct": always.decision_json() == expected,
                "checkpoint_identity": checkpoint_identity,
                "tokenizer_identity": tokenizer.identity(),
                "registry_snapshot_identity": snapshot.snapshot_identity,
            })

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "checkpoint_identity": checkpoint_identity,
        "tokenizer_identity": tokenizer.identity(),
        "registry_snapshot_identity": snapshot.snapshot_identity,
        "index_identity": index.index_identity,
        "device": str(device),
        "calibration": args.calibration,
    }
    snapshot.save(destination / "registry_snapshot.json")
    index.save(destination / "registry_index.json")
    write_json(destination / "predictions.json", {"status": "RAW_PREDICTIONS", "sealed": False, **identity, "records": predictions})
    for split in ("train", "development", "test"):
        split_rows = [row for row in predictions if row.get("split") == split]
        write_json(destination / f"{split}.json", {"status": "RAW_PREDICTIONS", "sealed": False, "split": split, **identity, "records": split_rows})
    write_json(destination / "threshold-calibration.json", {"status": "RAW_THRESHOLD_ROWS", "sealed": False, **identity, "records": threshold_rows})
    write_json(destination / "manifest.json", {"status": "EVALUATED", "sealed": False, **identity, "record_count": len(predictions), "split_counts": {split: sum(row.get("split") == split for row in predictions) for split in ("train", "development", "test")}, "threshold_support": len(threshold_rows)})
    print(json.dumps({"status": "EVALUATED", "records": len(predictions), "threshold_support": len(threshold_rows), "checkpoint_identity": checkpoint_identity, "registry_snapshot_identity": snapshot.snapshot_identity}, separators=(",", ":")))


if __name__ == "__main__":
    main()
