# Reproduce

> Historical document. Juniper Encoder was retired on 2026-09-15; see [RETIRED.md](../RETIRED.md) for the canonical final status. The text below describes the project as it stood during development.

All commands run from the repository root and should be executed at the exact candidate source SHA. Before any expensive work, capture the host and runtime with `bash scripts/preflight.sh --output reports/preflight/flowbox.json`.

The canonical environment reconstruction is:

```bash
bash scripts/bootstrap.sh --lock requirements/flowbox.lock
source .venv/bin/activate
python -m pip check
python -m juniper_encoder env verify --lock requirements/flowbox.lock
```

The present FLOWBOX lock contains the verified Python 3.12 CUDA 12.8 PyTorch wheel and all resolved dependency hashes. `env verify` reports CUDA availability and the FP16 AMP probe; native BF16 is recorded as unsupported on the RTX 2060. This is an environment result, not a successful training or deployment result.

Contract-only checks are runnable without PyTorch:

```bash
python -m juniper_encoder spec verify --contract spec/proposal_b.contract.json
python -m juniper_encoder routing conformance --fixtures tests/fixtures/routing
python -m pytest -q
```

The data inputs can be acquired and frozen with:

```bash
python -m juniper_encoder data acquire --manifest manifests/sources/approved.json --output-root data/raw
python -m juniper_encoder data prepare --manifest data/raw/acquisition.json --output data/processed/prepared.json
python -m juniper_encoder data freeze --config configs/splits.yaml --manifest data/processed/prepared.json --output manifests/data/frozen.json
```

The repository tokenizer snapshot is generated deterministically and is explicitly training-only:

```bash
python scripts/build_first_party_corpus.py
python -m juniper_encoder tokenizer corpus --config configs/tokenizer.yaml --output reports/tokenizer-corpus.json
python -m juniper_encoder tokenizer train --config configs/tokenizer.yaml --output artifacts/tokenizer-vN
python -m juniper_encoder tokenizer conformance --artifact artifacts/tokenizer-vN
```

The supervised routing fixture is generated separately with `scripts/build_first_party_supervised_data.py`; it is not mixed into tokenizer training or sealed evaluation. Training, calibration, export, target benchmarks, and INT8 qualification must consume immutable manifests and a selected checkpoint. Their CLI commands fail with a machine-readable blocker when those inputs are absent. Local prototype artifacts do not constitute a selected candidate or release.
