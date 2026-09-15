"""The single machine-readable command-line entrypoint."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import constants
from .artifacts import make_sha256sums, package_candidate, package_descriptor, verify_sha256sums
from .benchmark import check_coverage, execute_benchmark
from .calibration import calibration_manifest, fit_temperature, select_thresholds
from .contract import audit_model, verify_machine_contract, verify_traceability
from .data import acquire, audit, deterministic_split, freeze, plan, prepare, validate_source_manifest
from .errors import EncoderError
from .evaluation import evaluate_predictions, false_call_rate, macro_f1, ndcg, recall_at_k
from .export import export_checkpoint, verify_export
from .formatting import RegistryRecord, format_request
from .model import ModelConfig, require_torch
from .registry import RegistryIndex, RegistrySnapshot
from .routing import Calibration, RoutingCallbacks, route
from .tokenizer import RawByteBPE, TokenizerConfig, validate_round_trip
from .train.checkpoint import validate_checkpoint_metadata
from .train.protocol import MaskingProtocol, ResolvedProtocol, require_training_runtime, run_mechanical_smoke
from .train.runner import TokenExample, train_mlm
from .utils import canonical_json_bytes, load_config, sha256_bytes, sha256_file, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m juniper_encoder", description="Juniper Encoder frozen-contract CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    env = commands.add_parser("env", help="verify the locked runtime")
    env.add_argument("action", choices=["verify"])
    env.add_argument("--lock", required=True)

    spec = commands.add_parser("spec", help="verify the machine contract")
    spec.add_argument("action", choices=["verify"])
    spec.add_argument("--contract", required=True)

    model = commands.add_parser("model", help="audit the actual model")
    model.add_argument("action", choices=["audit"])
    model.add_argument("--config", required=True)
    model.add_argument("--output", required=True)

    tokenizer = commands.add_parser("tokenizer", help="build and verify raw byte BPE")
    tokenizer.add_argument("action", choices=["corpus", "train", "compare", "conformance"])
    tokenizer.add_argument("--config")
    tokenizer.add_argument("--output")
    tokenizer.add_argument("--left")
    tokenizer.add_argument("--right")
    tokenizer.add_argument("--artifact")

    data = commands.add_parser("data", help="data provenance and split manifests")
    data.add_argument("action", choices=["plan", "acquire", "prepare", "split", "audit", "freeze"])
    data.add_argument("--config")
    data.add_argument("--manifest")
    data.add_argument("--output")
    data.add_argument("--output-root", default="data/raw")
    data.add_argument("--max-bytes", type=int, default=1_073_741_824)
    data.add_argument("--max-extracted-bytes", type=int, default=5_368_709_120)
    data.add_argument("--fail-on-leakage", action="store_true")

    routing = commands.add_parser("routing", help="routing conformance")
    routing.add_argument("action", choices=["conformance"])
    routing.add_argument("--fixtures", required=True)

    train = commands.add_parser("train", help="training protocol and gated execution")
    train.add_argument("action", choices=["smoke", "overfit", "profile", "resume-test", "run", "promote", "downstream", "gate", "mine-negatives", "build-decisions", "select"])
    train.add_argument("--stage")
    train.add_argument("--config")
    train.add_argument("--lengths", nargs="*")
    train.add_argument("--run")
    train.add_argument("--policy")
    train.add_argument("--split")
    train.add_argument("--data")
    train.add_argument("--tokenizer")
    train.add_argument("--output")
    train.add_argument("--device")
    train.add_argument("--resume")
    train.add_argument("--checkpoint-every", type=int, default=100)

    evaluate = commands.add_parser("evaluate", help="component and end-to-end evaluation")
    evaluate.add_argument("action", nargs="?", choices=["adversarial", "compare-paths", "gates"])
    evaluate.add_argument("--split")
    evaluate.add_argument("--config")
    evaluate.add_argument("--policy")
    evaluate.add_argument("--sealed", action="store_true")
    evaluate.add_argument("--predictions")
    evaluate.add_argument("--output")

    calibrate = commands.add_parser("calibrate", help="calibration metadata")
    calibrate.add_argument("action", choices=["thresholds", "temperature", "freeze", "verify"])
    calibrate.add_argument("--config")
    calibrate.add_argument("--split")
    calibrate.add_argument("--manifest")
    calibrate.add_argument("--output")
    calibrate.add_argument("--temperature-artifact")
    calibrate.add_argument("--thresholds-artifact")
    calibrate.add_argument("--checkpoint-identity")
    calibrate.add_argument("--tokenizer-identity")
    calibrate.add_argument("--threshold-subset-identity")
    calibrate.add_argument("--temperature-subset-identity")
    calibrate.add_argument("--registry-snapshot-identity")
    calibrate.add_argument("--implementation-sha")

    export = commands.add_parser("export", help="safe deployment export")
    export.add_argument("action", nargs="?", choices=["create", "verify"], default="create")
    export.add_argument("--checkpoint")
    export.add_argument("--variant")
    export.add_argument("--output")
    export.add_argument("--tokenizer")
    export.add_argument("--config")
    export.add_argument("--calibration")

    quantize = commands.add_parser("quantize", help="INT8 qualification")
    quantize.add_argument("action", nargs="?", choices=["create", "gates", "verify"], default="create")
    quantize.add_argument("--config", required=True)
    quantize.add_argument("--input")
    quantize.add_argument("--output")
    quantize.add_argument("--floating-metrics")
    quantize.add_argument("--int8-metrics")

    index = commands.add_parser("index", help="registry index lifecycle")
    index.add_argument("action", choices=["rebuild"])
    index.add_argument("--variant", required=True)
    index.add_argument("--config", required=True)
    index.add_argument("--registry")
    index.add_argument("--tokenizer")
    index.add_argument("--output")

    benchmark = commands.add_parser("benchmark", help="required target benchmark matrix")
    benchmark.add_argument("action", nargs="?", choices=["run", "compare", "coverage"])
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--variant")
    benchmark.add_argument("--results")
    benchmark.add_argument("--output")
    benchmark.add_argument("--reference-export")
    benchmark.add_argument("--int8-export")
    benchmark.add_argument("--device")

    package = commands.add_parser("package", help="candidate descriptor and artifact verification")
    package.add_argument("action", nargs="?", choices=["verify"], default="create")
    package.add_argument("--checkpoint")
    package.add_argument("--output")
    package.add_argument("--directory")
    package.add_argument("--candidate")
    package.add_argument("--export", action="append", default=[])

    review = commands.add_parser("review", help="independent review entrypoint")
    review.add_argument("--candidate", required=True)
    review.add_argument("--reviewer", choices=["sol", "astra"], required=True)
    return parser


def _status_result(status: str, **payload: Any) -> dict[str, Any]:
    return {"status": status, **payload}


def _toy_tokenizer() -> RawByteBPE:
    # Fixture-only tokenizer. It is never promoted as the frozen deployment tokenizer.
    return RawByteBPE.from_merges([], toy=True)


def _source_sha() -> str:
    explicit = os.environ.get("GIT_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records_payload(path: str | Path) -> list[dict[str, Any]]:
    payload = _json_payload(path)
    if isinstance(payload, dict):
        records = payload.get("records", payload.get("predictions", []))
    else:
        records = payload
    if not isinstance(records, list):
        raise EncoderError("INVALID_INPUT", "manifest must contain a records/predictions array")
    return records


def _load_training_examples(
    data_path: str | Path,
    tokenizer_path: str | Path,
    split: str = "train",
    *,
    allowed_source_roles: set[str] | None = None,
    max_sequence_length: int = 1_024,
) -> tuple[list[TokenExample], str]:
    tokenizer = RawByteBPE.load(tokenizer_path)
    payload = _json_payload(data_path)
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise EncoderError("BLOCKED_DATA", "training manifest does not contain records")
    selected = [
        record
        for record in records
        if record.get("split", split) == split
        and (not allowed_source_roles or record.get("source_role", "general") in allowed_source_roles)
    ]
    examples: list[TokenExample] = []
    for record in selected:
        text = record.get("text", record.get("content"))
        record_id = record.get("record_id")
        if not isinstance(text, str) or not isinstance(record_id, str):
            raise EncoderError("BLOCKED_DATA", "training records need text and immutable record_id")
        token_ids = tuple(int(token) for token in tokenizer.encode(text))
        if not token_ids:
            continue
        for offset in range(0, len(token_ids), max_sequence_length):
            chunk = token_ids[offset:offset + max_sequence_length]
            examples.append(TokenExample(f"{record_id}#chunk-{offset // max_sequence_length:05d}", chunk))
    if not examples:
        raise EncoderError("BLOCKED_DATA", "training split contains no usable examples", {"split": split})
    return examples, tokenizer.identity()


def _documents_from_glob(pattern: str) -> list[str]:
    documents: list[str] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        file_path = Path(path)
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() == ".jsonl":
            for line in file_path.read_text(encoding="utf-8", errors="strict").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                text = value.get("text", value.get("content")) if isinstance(value, dict) else value
                if isinstance(text, str) and text:
                    documents.append(text)
        elif file_path.suffix.lower() == ".json":
            value = _json_payload(file_path)
            values = value if isinstance(value, list) else [value]
            for item in values:
                text = item.get("text", item.get("content")) if isinstance(item, dict) else item
                if isinstance(text, str) and text:
                    documents.append(text)
        else:
            documents.append(file_path.read_text(encoding="utf-8", errors="strict"))
    return documents


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "env":
        lock = Path(args.lock)
        if not lock.exists():
            raise EncoderError("BLOCKED_ENVIRONMENT", "lock file does not exist", {"lock": str(lock)})
        payload: dict[str, Any] = {"python": sys.version, "platform": platform.platform(), "lock_sha256": sha256_file(lock)}
        try:
            import torch
            payload.update({"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())})
            if torch.cuda.is_available():
                payload["cuda_device"] = torch.cuda.get_device_name(0)
                payload["torch_cuda_build"] = torch.version.cuda
                payload["bf16_native"] = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                try:
                    probe = torch.nn.Linear(4, 2, device="cuda")
                    scaler = torch.amp.GradScaler("cuda")
                    probe_input = torch.ones((2, 4), device="cuda")
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        probe_loss = probe(probe_input).float().square().mean()
                    scaler.scale(probe_loss).backward()
                    torch.cuda.synchronize()
                    payload["fp16_amp_probe"] = bool(torch.isfinite(probe_loss).item())
                except Exception as exc:
                    raise EncoderError("BLOCKED_ENVIRONMENT", "FP16 AMP probe failed", {"reason": str(exc), **payload}) from exc
        except ImportError:
            raise EncoderError("BLOCKED_ENVIRONMENT", "PyTorch is absent from the locked environment", payload)
        return _status_result("PASS", **payload)

    if args.command == "spec":
        result = verify_machine_contract(args.contract)
        spec_path = Path("spec/PROPOSAL_B_FROZEN.md")
        actual_spec_sha = sha256_file(spec_path)
        contract_payload = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        if contract_payload.get("frozen_spec_sha256") != actual_spec_sha:
            raise ValueError("machine contract does not pin the frozen specification bytes")
        result["frozen_spec_sha256"] = actual_spec_sha
        result["traceability"] = verify_traceability("spec/traceability.csv", spec_sha256=actual_spec_sha)
        if "BLOCKED_FROZEN_SPEC" in spec_path.read_text(encoding="utf-8", errors="strict"):
            return _status_result("BLOCKED_FROZEN_SPEC", **result, reason="complete 27-section Proposal B is not materialized")
        return _status_result("PASS", **result)

    if args.command == "model":
        return audit_model(args.config, args.output)

    if args.command == "tokenizer":
        if args.action == "corpus":
            config = load_config(args.config)
            paths = sorted(glob.glob(config.get("training_glob", ""), recursive=True))
            if not paths:
                return _status_result("BLOCKED_DATA", reason="no approved tokenizer-training corpus is materialized")
            stats = {"files": [], "document_count": 0, "byte_count": 0, "document_byte_count": 0, "language_scope": config.get("language_scope", ["en"]), "domain_scope": config.get("domain_scope", ["technical", "capability"])}
            documents: list[str] = []
            for path in paths:
                file_path = Path(path)
                if not file_path.is_file():
                    continue
                data = file_path.read_bytes()
                file_info = {"path": path, "size": len(data), "sha256": sha256_file(file_path)}
                stats["files"].append(file_info)
                stats["byte_count"] += len(data)
                documents.extend(_documents_from_glob(path))
            stats["document_count"] = len(documents)
            stats["document_byte_count"] = sum(len(document.encode("utf-8")) for document in documents)
            result = _status_result("CORPUS_READY", config=config, **stats)
            if args.output:
                write_json(args.output, result)
            return result
        if args.action == "train":
            config = load_config(args.config)
            documents = config.get("documents", [])
            if not documents:
                documents = _documents_from_glob(config.get("training_glob", ""))
            if not documents:
                raise EncoderError("BLOCKED_DATA", "tokenizer training requires an approved, training-only corpus manifest")
            tokenizer = RawByteBPE.train(documents, config=TokenizerConfig(**{key: value for key, value in config.items() if key in {"merge_count", "vocab_size", "implementation_version", "unicode_database", "tie_break", "overlap_rule"}}), require_full=bool(config.get("require_full_vocabulary", True)))
            if not args.output:
                raise EncoderError("INVALID_INPUT", "tokenizer train requires --output")
            corpus_files = [
                {"path": path, "sha256": sha256_file(path), "size": Path(path).stat().st_size}
                for path in sorted(glob.glob(config.get("training_glob", ""), recursive=True))
                if Path(path).is_file()
            ]
            corpus_hash = sha256_bytes(canonical_json_bytes(corpus_files))
            return tokenizer.save(args.output, source_sha=_source_sha(), corpus_hash=corpus_hash)
        if args.action == "compare":
            left, right = RawByteBPE.load(args.left), RawByteBPE.load(args.right)
            return _status_result("PASS" if left.compare_payload(right) else "FAIL", left=left.identity(), right=right.identity())
        if args.action == "conformance":
            tokenizer = RawByteBPE.load(args.artifact)
            validate_round_trip(tokenizer, ["", "hello", "e\u0301", "🙂", "\x00", "[PAD]", "\u05e9\u05dc\u05d5\u05dd", "\u2028", "🚀", "\t\n\r"])
            try:
                tokenizer.encode("\ud800")
            except EncoderError:
                pass
            else:
                raise AssertionError("surrogate was not rejected")
            return _status_result("PASS", tokenizer_identity=tokenizer.identity())

    if args.command == "data":
        if args.action == "plan":
            return plan(args.config, args.output)
        if args.action == "acquire":
            return acquire(args.manifest, output_root=args.output_root, max_bytes=args.max_bytes, max_extracted_bytes=args.max_extracted_bytes)
        if args.action == "prepare":
            if not args.manifest or not args.output:
                raise EncoderError("INVALID_INPUT", "data prepare requires --manifest acquisition.json and --output")
            return prepare(args.manifest, args.output)
        if args.action == "freeze":
            if not args.manifest or not args.output:
                raise EncoderError("INVALID_INPUT", "data freeze requires --manifest and --output")
            split_config = load_config(args.config or "configs/splits.yaml")
            return freeze(args.manifest, args.output, split_config["ratios"], int(split_config.get("seed", 1729)))
        if args.action == "split":
            config = load_config(args.config)
            records = config.get("records", _records_payload(config["manifest"]) if config.get("manifest") else [])
            if not records:
                raise EncoderError("BLOCKED_DATA", "split requires an immutable processed-record manifest")
            result = deterministic_split(records, config["ratios"], int(config.get("seed", 1729)))
            write_json(args.output, {"status": "FROZEN", "records": result})
            return _status_result("FROZEN", records=len(result), output=args.output)
        if args.action == "audit":
            payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
            result = audit(payload.get("records", []))
            if args.fail_on_leakage and result["leaks"]:
                raise EncoderError("BLOCKED_DATA_LEAKAGE", "cross-split duplicate families were found", result)
            return result

    if args.command == "routing":
        tokenizer = _toy_tokenizer()
        records = tuple(RegistryRecord.from_mapping({"id": item, "type": "tool", "name": item, "purpose": item, "supported_requests": [item], "exclusions": [], "required_inputs": []}) for item in ("alpha", "beta"))
        snapshot = RegistrySnapshot.create(records, tokenizer_identity=tokenizer.identity(), encoder_variant_identity="fixture")
        index = RegistryIndex.build(snapshot, lambda record: [1.0 if record.id == "alpha" else 0.0, 1.0 if record.id == "beta" else 0.0] + [0.0] * 254)
        calls = {"query": 0, "rerank": 0, "classify": 0}
        def encode_query(_request):
            calls["query"] += 1
            return [1.0] + [0.0] * 255
        def rerank_pair(_tokens, _record):
            calls["rerank"] += 1
            return 0.0
        def classify(_tokens):
            calls["classify"] += 1
            return [1.0, 0.0, 0.0]
        result = route({"history": [], "user": "hello"}, snapshot, index, tokenizer, RoutingCallbacks(encode_query, rerank_pair, classify))
        if result.decision != "CALL" or result.capability_id != "alpha":
            raise AssertionError("fixture routing did not produce the controlled CALL")
        return _status_result("PASS", decision=result.decision_json(), metadata=result.metadata, calls=calls)

    if args.command == "train":
        if args.action == "smoke":
            return run_mechanical_smoke(load_config(args.config))
        if args.action == "overfit":
            tokenizer = _toy_tokenizer()
            examples = [TokenExample(str(index), tuple(tokenizer.encode(text))) for index, text in enumerate(("alpha alpha", "beta beta", "alpha beta", "beta alpha"))]
            config = load_config(args.config or "configs/mlm.yaml")
            return train_mlm(examples, config, output_dir=args.output or "artifacts/experiments/overfit", source_sha=_source_sha(), tokenizer_identity=tokenizer.identity(), data_identity="first-party-overfit-fixture", device=args.device, max_updates=int(config.get("overfit_updates", 1)), checkpoint_every=args.checkpoint_every)
        if args.action == "run":
            if not args.data or not args.tokenizer or not args.output or not args.config:
                raise EncoderError("INVALID_INPUT", "train run requires --config, --data, --tokenizer, and --output")
            config = load_config(args.config)
            examples, tokenizer_identity = _load_training_examples(
                args.data,
                args.tokenizer,
                allowed_source_roles=set(config.get("training_source_roles", [])),
                max_sequence_length=int(config.get("max_sequence_length", 1_024)),
            )
            return train_mlm(examples, config, output_dir=args.output, source_sha=_source_sha(), tokenizer_identity=tokenizer_identity, data_identity=sha256_file(args.data), device=args.device, checkpoint_every=args.checkpoint_every, resume_dir=args.resume)
        if args.action == "profile":
            if not args.config:
                raise EncoderError("INVALID_INPUT", "train profile requires --config")
            config = load_config(args.config)
            return _status_result("PROFILED", config_sha256=sha256_file(args.config), requested_lengths=config.get("native_lengths", []), runtime="ready")
        if args.action == "resume-test":
            if not args.data or not args.tokenizer or not args.config or not args.output:
                raise EncoderError("INVALID_INPUT", "train resume-test requires --config, --data, --tokenizer, and --output")
            config = load_config(args.config)
            examples, tokenizer_identity = _load_training_examples(
                args.data,
                args.tokenizer,
                allowed_source_roles=set(config.get("training_source_roles", [])),
                max_sequence_length=int(config.get("max_sequence_length", 1_024)),
            )
            first = train_mlm(examples, config, output_dir=Path(args.output) / "uninterrupted", source_sha=_source_sha(), tokenizer_identity=tokenizer_identity, data_identity=sha256_file(args.data), device=args.device, max_updates=2, checkpoint_every=1)
            interrupted = train_mlm(examples, config, output_dir=Path(args.output) / "interrupted", source_sha=_source_sha(), tokenizer_identity=tokenizer_identity, data_identity=sha256_file(args.data), device=args.device, max_updates=1, checkpoint_every=1)
            resumed = train_mlm(examples, config, output_dir=Path(args.output) / "resumed", source_sha=_source_sha(), tokenizer_identity=tokenizer_identity, data_identity=sha256_file(args.data), device=args.device, max_updates=2, checkpoint_every=1, resume_dir=interrupted["checkpoint"])
            matched = first.get("weights_sha256") == resumed.get("weights_sha256")
            return _status_result("RESUME_TESTED" if matched else "RESUME_MISMATCH", uninterrupted=first, interrupted=interrupted, resumed=resumed, bitwise_match=matched, bitwise_reference="same-environment deterministic runner")
        if args.action == "promote":
            if not args.run or not args.output or not args.tokenizer or not args.data:
                raise EncoderError("INVALID_INPUT", "train promote requires --run, --data, --tokenizer, and --output")
            from .train.runner import initialize_deployment_from_foundation
            run_payload = _json_payload(args.run)
            foundation = run_payload.get("checkpoint")
            if not foundation:
                raise EncoderError("BLOCKED_TRAINING", "foundation run has no checkpoint")
            tokenizer = RawByteBPE.load(args.tokenizer)
            return initialize_deployment_from_foundation(foundation, args.output, source_sha=_source_sha(), tokenizer_identity=tokenizer.identity(), data_identity=sha256_file(args.data))
        if args.action == "downstream":
            if not args.stage or args.stage not in {"retrieval", "reranker", "classifier"} or not args.resume or not args.data or not args.output or not args.config or not args.tokenizer:
                raise EncoderError("INVALID_INPUT", "train downstream requires --stage retrieval|reranker|classifier, --resume, --data, --tokenizer, --config, and --output")
            import torch
            from .model import DeploymentModel
            from .train.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
            from .train.runner import train_head
            tokenizer = RawByteBPE.load(args.tokenizer)
            source_checkpoint = Path(args.resume)
            state_path = source_checkpoint / "model.safetensors"
            if not state_path.exists():
                raise EncoderError("BLOCKED_CHECKPOINT", "downstream training requires a safe deployment checkpoint")
            from safetensors.torch import load_file
            state = load_file(str(state_path), device="cpu")
            if not all(any(key.startswith(prefix) for key in state) for prefix in ("backbone.", "retrieval.", "classifier.", "reranker.")):
                raise EncoderError("BLOCKED_CHECKPOINT", "downstream training requires a promoted deployment checkpoint with all task heads")
            config = load_config(args.config)
            selected_device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model = DeploymentModel(seed=1729).to(selected_device)
            model.load_state_dict(state, strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("head_learning_rate", 1e-4)))
            loaded = load_checkpoint(source_checkpoint, model, optimizer, restore_rng=True)
            rows = [row for row in _records_payload(args.data) if row.get("split", "train") == "train"]
            if not rows:
                raise EncoderError("BLOCKED_DATA", "downstream manifest has no labeled training rows")
            if args.stage == "classifier":
                if any(not isinstance(row.get("input_ids"), list) or not isinstance(row.get("label_index"), int) for row in rows):
                    raise EncoderError("BLOCKED_DATA", "classifier rows need input_ids and label_index")
                inputs = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long, device=selected_device)
                labels = torch.tensor([row["label_index"] for row in rows], dtype=torch.long, device=selected_device)
            elif args.stage == "reranker":
                if any(not isinstance(row.get("input_ids"), list) or not row["input_ids"] or not isinstance(row.get("target_index"), int) for row in rows):
                    raise EncoderError("BLOCKED_DATA", "reranker rows need grouped input_ids and target_index")
                inputs = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long, device=selected_device)
                labels = torch.tensor([row["target_index"] for row in rows], dtype=torch.long, device=selected_device)
            else:
                if any(not isinstance(row.get("input_ids"), list) or not isinstance(row.get("positive_vector"), list) for row in rows):
                    raise EncoderError("BLOCKED_DATA", "retrieval rows need input_ids and positive_vector")
                negatives = [row.get("negative_vectors", []) for row in rows]
                candidate_counts = {1 + len(value) for value in negatives if isinstance(value, list)}
                if len(candidate_counts) != 1:
                    raise EncoderError("BLOCKED_DATA", "retrieval rows must have the same explicit candidate count")
                candidates = [ [row["positive_vector"], *row.get("negative_vectors", [])] for row in rows ]
                inputs = torch.tensor([row["input_ids"] for row in rows], dtype=torch.long, device=selected_device)
                labels = {"candidates": torch.tensor(candidates, dtype=torch.float32, device=selected_device)}
            updates = int(config.get("updates", config.get("downstream_updates", config.get("downstream_passes", 1))))
            training = train_head(model, [(inputs, labels)], objective=args.stage, optimizer=optimizer, device=selected_device, updates=updates)
            successful = int(loaded["metadata"].get("successful_updates", 0)) + int(training["successful_updates"])
            destination = Path(args.output) / f"step-{successful:08d}"
            metadata = CheckpointMetadata(
                format_version="juniper-checkpoint-v2", source_sha=_source_sha(), config_hashes={args.stage: sha256_file(args.config)}, rng_states={},
                successful_updates=successful, attempted_updates=successful, skipped_updates=0, consumed_tokens=int(inputs.numel()),
                sampler_identity="downstream-explicit-labeled-batch-v1", tokenizer_identity=tokenizer.identity(), data_identity=sha256_file(args.data),
                predecessor_identity=loaded.get("checkpoint_identity"),
            )
            saved = save_checkpoint(destination, model, optimizer, None, None, metadata, precision={"device": str(selected_device), "amp_fp16": False})
            result = {"status": "DOWNSTREAM_TRAINED", "stage": args.stage, "checkpoint": destination.as_posix(), "training": training, "checkpoint_identity": saved["checkpoint_identity"]}
            write_json(Path(args.output) / "run.json", result)
            return result
        if args.action == "gate":
            if not args.run or not Path(args.run).exists():
                raise EncoderError("BLOCKED_TRAINING", "training gate needs a real checkpoint/run manifest")
            run_payload = _json_payload(args.run)
            checkpoint = Path(run_payload.get("checkpoint", ""))
            if run_payload.get("status") != "TRAINED" or not (checkpoint / "metadata.json").exists():
                raise EncoderError("BLOCKED_TRAINING", "training run is not a materialized trained checkpoint")
            validate_checkpoint_metadata(_json_payload(checkpoint / "metadata.json").get("metadata", {}))
            return _status_result("PASS", checkpoint=checkpoint.as_posix(), training=run_payload)
        if args.action in {"mine-negatives", "build-decisions", "select"}:
            if not args.data or not args.output:
                raise EncoderError("INVALID_INPUT", f"train {args.action} requires --data and --output")
            records = _records_payload(args.data)
            if args.action == "mine-negatives":
                result = {"status": "MINED", "records": [{"query_id": record.get("record_id"), "positive_id": record.get("record_id"), "negative_ids": [other.get("record_id") for other in records if other.get("record_id") != record.get("record_id")][:7], "source_data_sha256": sha256_file(args.data)} for record in records]}
            elif args.action == "build-decisions":
                result = {"status": "BUILT", "records": [{"record_id": record.get("record_id"), "label": record.get("label", "NO_CALL"), "candidate_ids": record.get("candidate_ids", [])} for record in records], "source_data_sha256": sha256_file(args.data)}
            else:
                result = {"status": "SELECTED", "selection_rule": "development-only-first-trained-checkpoint", "source_data_sha256": sha256_file(args.data), "checkpoint": records[0].get("checkpoint") if records else None}
            write_json(args.output, result)
            return result
        require_training_runtime()
        raise EncoderError("BLOCKED_TRAINING", "this training action requires a materialized data/checkpoint workflow", {"action": args.action, "stage": args.stage})

    if args.command == "evaluate":
        source = args.predictions or args.split
        if args.action == "gates":
            policy = load_config(args.policy)
            required = ("minimum_recall_at_8", "minimum_macro_f1", "minimum_call_recall", "minimum_end_to_end_accuracy", "maximum_false_call_rate", "maximum_ece")
            if any(policy.get(key) is None for key in required):
                raise EncoderError("BLOCKED_QUALITY_POLICY", "absolute project quality thresholds require independent protocol review")
            if not source:
                raise EncoderError("BLOCKED_EVALUATION", "quality gates require evaluated metrics")
            metrics = _json_payload(source)
            failures = []
            checks = (("shortlist_recall_at_8", ">=", policy["minimum_recall_at_8"]), ("decision_macro_f1", ">=", policy["minimum_macro_f1"]), ("call_recall", ">=", policy["minimum_call_recall"]), ("end_to_end_exact_accuracy", ">=", policy["minimum_end_to_end_accuracy"]), ("false_call_rate", "<=", policy["maximum_false_call_rate"]), ("ece", "<=", policy["maximum_ece"]))
            for key, operator, threshold in checks:
                if key not in metrics or ((operator == ">=" and metrics[key] < threshold) or (operator == "<=" and metrics[key] > threshold)):
                    failures.append({"metric": key, "operator": operator, "threshold": threshold, "actual": metrics.get(key)})
            return _status_result("PASS" if not failures else "FAIL", metrics=metrics, failures=failures)
        if not source:
            raise EncoderError("BLOCKED_EVALUATION", "evaluation requires a materialized raw-predictions or evaluation manifest")
        source_payload = _json_payload(source)
        if args.sealed and not (isinstance(source_payload, dict) and source_payload.get("sealed") is True):
            raise EncoderError("BLOCKED_EVALUATION", "sealed evaluation requires a manifest explicitly marked sealed")
        rows = _records_payload(source)
        result = evaluate_predictions(rows, ece_bins=int(load_config(args.config)["ece_bins"]) if args.config else 15)
        if set(result) == {"support", "denominators"}:
            raise EncoderError("BLOCKED_EVALUATION", "evaluation manifest contains no raw predictions or component labels")
        result["source_sha256"] = sha256_file(source)
        result["sealed"] = bool(args.sealed)
        if args.output:
            write_json(args.output, result)
        return result

    if args.command == "calibrate":
        if args.action == "verify":
            if not args.manifest or not Path(args.manifest).exists():
                raise EncoderError("BLOCKED_CALIBRATION", "calibration metadata is absent")
            payload = _json_payload(args.manifest)
            required = {"temperature", "tau_score", "tau_margin", "checkpoint_identity", "tokenizer_identity", "pre_nll", "post_nll", "pre_brier", "post_brier", "pre_ece", "post_ece", "temperature_converged"}
            missing = sorted(required - set(payload))
            if missing:
                raise EncoderError("BLOCKED_CALIBRATION", "calibration metadata is incomplete", {"missing": missing})
            return _status_result("PASS", calibration_identity=sha256_file(args.manifest), fields=sorted(payload))
        if args.action == "freeze":
            required = (args.temperature_artifact, args.thresholds_artifact, args.checkpoint_identity, args.tokenizer_identity, args.threshold_subset_identity, args.temperature_subset_identity, args.registry_snapshot_identity, args.implementation_sha, args.output)
            if any(value is None for value in required):
                raise EncoderError("INVALID_INPUT", "calibrate freeze requires both fitted artifacts, all identities, and --output")
            manifest = calibration_manifest(
                temperature=_json_payload(args.temperature_artifact),
                thresholds=_json_payload(args.thresholds_artifact),
                checkpoint_identity=args.checkpoint_identity,
                tokenizer_identity=args.tokenizer_identity,
                threshold_subset_identity=args.threshold_subset_identity,
                temperature_subset_identity=args.temperature_subset_identity,
                registry_snapshot_identity=args.registry_snapshot_identity,
                implementation_sha=args.implementation_sha,
            )
            write_json(args.output, manifest)
            return manifest
        if not args.manifest or not args.output:
            raise EncoderError("INVALID_INPUT", "calibration requires --manifest input rows and --output")
        rows = _records_payload(args.manifest)
        if args.action == "temperature":
            logits = [row["logits"] for row in rows if "logits" in row]
            labels = [int(row.get("label_index", row.get("label"))) for row in rows if "logits" in row]
            fitted = fit_temperature(logits, labels)
            write_json(args.output, fitted)
            return fitted
        if args.action == "thresholds":
            fitted = select_thresholds(rows)
            write_json(args.output, fitted)
            return fitted

    if args.command == "export":
        if args.action == "verify":
            return verify_export(args.output or "artifacts/reference")
        return export_checkpoint(args.checkpoint, args.output, args.variant or "reference", tokenizer_directory=args.tokenizer, model_config=args.config, calibration=args.calibration)

    if args.command == "quantize":
        if args.action == "gates":
            if not args.floating_metrics or not args.int8_metrics:
                raise EncoderError("BLOCKED_INT8", "matched floating and INT8 metric manifests are required")
            from .quantization import compare_metrics
            config = load_config(args.config)
            result = compare_metrics(_json_payload(args.floating_metrics), _json_payload(args.int8_metrics), config.get("gates", config))
            if args.output:
                write_json(args.output, result)
            return result
        if args.action == "verify":
            if not args.input:
                raise EncoderError("BLOCKED_INT8", "INT8 verification requires --input export directory")
            from .quantization import verify_int8_export
            return verify_int8_export(args.input)
        if not args.input or not args.output:
            raise EncoderError("BLOCKED_INT8", "quantization requires a floating export and output directory")
        from .quantization import quantize_export
        return quantize_export(args.input, args.output, args.config)

    if args.command == "index":
        if not args.registry or not args.tokenizer or not args.output:
            raise EncoderError("BLOCKED_INDEX", "index rebuild requires --registry, --tokenizer, and --output")
        tokenizer = RawByteBPE.load(args.tokenizer)
        snapshot = RegistrySnapshot.from_json(args.registry, tokenizer_identity=tokenizer.identity(), encoder_variant_identity=args.variant)
        export_dir = Path(load_config(args.config).get("export_directory", ""))
        if not export_dir or not (export_dir / "model.safetensors").exists():
            raise EncoderError("BLOCKED_INDEX", "selected encoder export is absent")
        from safetensors.torch import load_file
        import torch
        from .model import DeploymentModel
        model = DeploymentModel(seed=1729)
        model.load_state_dict(load_file(str(export_dir / "model.safetensors"), device="cpu"), strict=True)
        model.eval()
        def encode(record):
            tokens = [constants.STRUCTURAL_IDS["[CLS]"], constants.STRUCTURAL_IDS["[DOCUMENT]"], *tokenizer.encode(record.serialized().decode("utf-8")), constants.STRUCTURAL_IDS["[SEP]"]]
            with torch.no_grad():
                return model.retrieval_vector(torch.tensor([tokens], dtype=torch.long))[0].tolist()
        index = RegistryIndex.build(snapshot, encode)
        snapshot.save(Path(args.output) / "snapshot.json")
        index.save(Path(args.output) / "index.json")
        return _status_result("INDEX_BUILT", snapshot_identity=snapshot.snapshot_identity, index_identity=index.index_identity, records=len(snapshot.records))

    if args.command == "benchmark":
        if args.action == "run":
            if not args.reference_export or not args.int8_export or not args.output:
                raise EncoderError("BLOCKED_BENCHMARK", "benchmark run requires --reference-export, --int8-export, and --output")
            from .quantization import load_int8_export
            from .export import load_float_export
            runners = {"reference": load_float_export(args.reference_export, device=args.device or "cpu"), "int8": load_int8_export(args.int8_export, device=args.device or "cpu")}
            return execute_benchmark(args.config, runners, args.output)
        return check_coverage(args.config, args.results)

    if args.command == "package":
        if args.action == "verify":
            directory = Path(args.directory or Path(args.candidate).parent)
            sums = directory / "SHA256SUMS"
            if not sums.exists():
                raise EncoderError("BLOCKED_PACKAGE", "candidate checksum manifest is absent")
            return verify_sha256sums(directory, sums)
        if not args.checkpoint or not Path(args.checkpoint).exists():
            raise EncoderError("BLOCKED_PACKAGE", "candidate package requires an immutable selected checkpoint")
        if not args.output:
            raise EncoderError("INVALID_INPUT", "package creation requires --output")
        return package_candidate(args.output, source_sha=_source_sha(), checkpoint=args.checkpoint, export_directories=args.export)

    if args.command == "review":
        return _status_result("AWAITING_INDEPENDENT_REVIEW", reviewer=args.reviewer, candidate=args.candidate, note="This entrypoint records scope for an independent reviewer; it does not approve.")

    raise AssertionError(f"unhandled command {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=False, separators=(",", ":")))
        status = str(result.get("status", ""))
        blocked = {"FAIL", "BLOCKED_FROZEN_SPEC", "BLOCKED_ENVIRONMENT", "BLOCKED_DATA", "BLOCKED_DATA_RIGHTS", "BLOCKED_DATA_SIZE", "BLOCKED_DATA_CHECKSUM", "BLOCKED_DATA_ARCHIVE", "BLOCKED_DATA_ACQUISITION", "BLOCKED_DATA_LEAKAGE", "BLOCKED_TRAINING", "BLOCKED_EVALUATION", "BLOCKED_CALIBRATION", "BLOCKED_INT8", "BLOCKED_INDEX", "BLOCKED_PACKAGE", "BLOCKED_QUALITY_POLICY", "BLOCKED_CHECKPOINT", "BLOCKED_EXPORT_PAYLOAD", "BLOCKED_BENCHMARK", "BENCHMARK_INCOMPLETE", "INT8_REJECTED", "RESUME_MISMATCH"}
        return 0 if status not in blocked else 20
    except EncoderError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=False, separators=(",", ":")))
        return 20
    except Exception as exc:
        print(json.dumps({"error": "INVALID_INPUT", "message": str(exc)}, ensure_ascii=False, sort_keys=False, separators=(",", ":")))
        return 2
