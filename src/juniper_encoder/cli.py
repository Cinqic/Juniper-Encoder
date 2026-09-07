"""The single machine-readable command-line entrypoint."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from . import constants
from .artifacts import make_sha256sums, package_descriptor, verify_sha256sums
from .benchmark import check_coverage
from .calibration import fit_temperature
from .contract import audit_model, verify_machine_contract, verify_traceability
from .data import acquire, audit, deterministic_split, plan, validate_source_manifest
from .errors import EncoderError
from .evaluation import false_call_rate, macro_f1, ndcg, recall_at_k
from .export import export_checkpoint, verify_export
from .formatting import RegistryRecord, format_request
from .model import ModelConfig, require_torch
from .registry import RegistryIndex, RegistrySnapshot
from .routing import Calibration, RoutingCallbacks, route
from .tokenizer import RawByteBPE, TokenizerConfig, validate_round_trip
from .train.checkpoint import validate_checkpoint_metadata
from .train.protocol import MaskingProtocol, ResolvedProtocol, require_training_runtime, run_mechanical_smoke
from .utils import canonical_json_bytes, load_config, sha256_file, write_json


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
    data.add_argument("--fail-on-leakage", action="store_true")

    routing = commands.add_parser("routing", help="routing conformance")
    routing.add_argument("action", choices=["conformance"])
    routing.add_argument("--fixtures", required=True)

    train = commands.add_parser("train", help="training protocol and gated execution")
    train.add_argument("action", choices=["smoke", "overfit", "profile", "resume-test", "run", "gate", "mine-negatives", "build-decisions", "select"])
    train.add_argument("--stage")
    train.add_argument("--config")
    train.add_argument("--lengths", nargs="*")
    train.add_argument("--run")
    train.add_argument("--policy")
    train.add_argument("--split")

    evaluate = commands.add_parser("evaluate", help="component and end-to-end evaluation")
    evaluate.add_argument("action", nargs="?", choices=["adversarial", "compare-paths", "gates"])
    evaluate.add_argument("--split")
    evaluate.add_argument("--config")
    evaluate.add_argument("--policy")
    evaluate.add_argument("--sealed", action="store_true")

    calibrate = commands.add_parser("calibrate", help="calibration metadata")
    calibrate.add_argument("action", choices=["thresholds", "temperature", "verify"])
    calibrate.add_argument("--config")
    calibrate.add_argument("--split")
    calibrate.add_argument("--manifest")

    export = commands.add_parser("export", help="safe deployment export")
    export.add_argument("action", nargs="?", choices=["verify"], default="create")
    export.add_argument("--checkpoint")
    export.add_argument("--variant")
    export.add_argument("--output")

    quantize = commands.add_parser("quantize", help="INT8 qualification")
    quantize.add_argument("action", nargs="?", choices=["gates"], default="create")
    quantize.add_argument("--config", required=True)

    index = commands.add_parser("index", help="registry index lifecycle")
    index.add_argument("action", choices=["rebuild"])
    index.add_argument("--variant", required=True)
    index.add_argument("--config", required=True)

    benchmark = commands.add_parser("benchmark", help="required target benchmark matrix")
    benchmark.add_argument("action", nargs="?", choices=["compare", "coverage"])
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--variant")
    benchmark.add_argument("--results")

    package = commands.add_parser("package", help="candidate descriptor and artifact verification")
    package.add_argument("action", nargs="?", choices=["verify"], default="create")
    package.add_argument("--checkpoint")
    package.add_argument("--output")
    package.add_argument("--directory")
    package.add_argument("--candidate")

    review = commands.add_parser("review", help="independent review entrypoint")
    review.add_argument("--candidate", required=True)
    review.add_argument("--reviewer", choices=["sol", "astra"], required=True)
    return parser


def _status_result(status: str, **payload: Any) -> dict[str, Any]:
    return {"status": status, **payload}


def _toy_tokenizer() -> RawByteBPE:
    # Fixture-only tokenizer. It is never promoted as the frozen deployment tokenizer.
    return RawByteBPE.from_merges([], toy=True)


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
        return _status_result("PASS", **result)

    if args.command == "model":
        return audit_model(args.config, args.output)

    if args.command == "tokenizer":
        if args.action == "corpus":
            config = load_config(args.config)
            paths = sorted(glob.glob(config.get("training_glob", ""), recursive=True))
            return _status_result("PLANNED", config=config, files=paths, file_count=len(paths)) if paths else _status_result("BLOCKED_DATA", reason="no approved tokenizer-training corpus is materialized")
        if args.action == "train":
            config = load_config(args.config)
            documents = config.get("documents", [])
            if not documents:
                raise EncoderError("BLOCKED_DATA", "tokenizer training requires an approved, training-only corpus manifest")
            tokenizer = RawByteBPE.train(documents, config=TokenizerConfig(**{key: value for key, value in config.items() if key in {"merge_count", "vocab_size", "implementation_version", "unicode_database", "tie_break", "overlap_rule"}}), require_full=bool(config.get("require_full_vocabulary", True)))
            return tokenizer.save(args.output, source_sha=os.environ.get("GIT_COMMIT"), corpus_hash=sha256_file(args.config))
        if args.action == "compare":
            left, right = RawByteBPE.load(args.left), RawByteBPE.load(args.right)
            return _status_result("PASS" if left.compare_payload(right) else "FAIL", left=left.identity(), right=right.identity())
        if args.action == "conformance":
            tokenizer = RawByteBPE.load(args.artifact)
            validate_round_trip(tokenizer, ["", "hello", "e\u0301", "🙂", "\x00", "[PAD]", "\u05e9\u05dc\u05d5\u05dd", "\u2028"])
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
            return acquire(args.manifest)
        if args.action in {"prepare", "freeze"}:
            raise EncoderError("BLOCKED_DATA", "processed data is not materialized from approved immutable sources")
        if args.action == "split":
            config = load_config(args.config)
            records = config.get("records", [])
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
        if args.action == "gate":
            if not args.run or not Path(args.run).exists():
                raise EncoderError("BLOCKED_TRAINING", "training gate needs a real checkpoint/run manifest")
            return _status_result("BLOCKED_TRAINING", reason="no trained checkpoint may be inferred from a manifest path")
        require_training_runtime()
        raise EncoderError("BLOCKED_TRAINING", "this command requires a materialized lawful corpus and a resolved training runner", {"action": args.action, "stage": args.stage})

    if args.command == "evaluate":
        if args.action == "gates":
            policy = load_config(args.policy)
            if any(policy.get(key) is None for key in ("minimum_recall_at_8", "minimum_macro_f1", "minimum_call_recall", "minimum_end_to_end_accuracy", "maximum_false_call_rate", "maximum_ece")):
                raise EncoderError("BLOCKED_QUALITY_POLICY", "absolute project quality thresholds require independent protocol review")
        raise EncoderError("BLOCKED_EVALUATION", "immutable evaluation records and a selected checkpoint are not materialized")

    if args.command == "calibrate":
        if args.action == "verify":
            raise EncoderError("BLOCKED_CALIBRATION", "calibration metadata is absent")
        raise EncoderError("BLOCKED_CALIBRATION", "calibration subsets and selected checkpoint are not materialized")

    if args.command == "export":
        if args.action == "verify":
            return verify_export(args.output or "artifacts/reference")
        return export_checkpoint(args.checkpoint, args.output, args.variant or "reference")

    if args.command == "quantize":
        raise EncoderError("BLOCKED_INT8", "floating reference export and matched measurements are not materialized")

    if args.command == "index":
        raise EncoderError("BLOCKED_INDEX", "selected encoder export and immutable registry snapshot are not materialized")

    if args.command == "benchmark":
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
        descriptor = {"source_sha": os.environ.get("GIT_COMMIT", "unknown"), "checkpoint": str(args.checkpoint), "status": "CANDIDATE_PROVISIONAL_NOT_RELEASED"}
        return package_descriptor(Path(args.output) / "candidate.json", descriptor)

    if args.command == "review":
        return _status_result("AWAITING_INDEPENDENT_REVIEW", reviewer=args.reviewer, candidate=args.candidate, note="This entrypoint records scope for an independent reviewer; it does not approve.")

    raise AssertionError(f"unhandled command {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _dispatch(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=False, separators=(",", ":")))
        status = str(result.get("status", ""))
        return 0 if status not in {"FAIL", "BLOCKED_ENVIRONMENT", "BLOCKED_DATA", "BLOCKED_DATA_LEAKAGE", "BLOCKED_TRAINING", "BLOCKED_EVALUATION", "BLOCKED_CALIBRATION", "BLOCKED_INT8", "BLOCKED_INDEX", "BLOCKED_PACKAGE", "BLOCKED_QUALITY_POLICY", "BLOCKED_CHECKPOINT", "BLOCKED_EXPORT_PAYLOAD", "INT8_REJECTED"} else 20
    except EncoderError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=False, separators=(",", ":")))
        return 20
    except Exception as exc:
        print(json.dumps({"error": "INVALID_INPUT", "message": str(exc)}, ensure_ascii=False, sort_keys=False, separators=(",", ":")))
        return 2
