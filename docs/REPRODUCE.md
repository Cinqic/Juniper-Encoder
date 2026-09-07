# Reproduce

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

Training, calibration, export, target benchmarks, and INT8 qualification must consume immutable manifests and a selected checkpoint. Their CLI commands fail with a machine-readable blocker when those inputs are absent.
