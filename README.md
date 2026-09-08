# Juniper Encoder

Juniper Encoder is a contract-first, encoder-only routing model for registered agents, tools, and skills. This repository contains the implementation and reproducibility controls for the supplied frozen Proposal B instructions.

## Status

`IMPLEMENTED / TESTED / NOT RELEASED`

The repository contains one small, first-party, training-only routing fixture (32 records), a deterministic repository-technical tokenizer/foundation snapshot, and a reproducible frozen split manifest. The local FLOWBOX worktree has produced provisional tokenizer, model, calibration, floating, and INT8 artifacts, but no selected checkpoint, durable qualified deployment export, quality qualification, independent Sol approval, or Astra approval exists. Consequently this branch makes no production quality, training-completion, or release claim.

The frozen authority is [spec/PROPOSAL_B_FROZEN.md](spec/PROPOSAL_B_FROZEN.md), SHA-256 `ce881075b84e36a28f673e1a41199e91cf0426baf66f788de880d86c156109b2`. The machine-readable contract is [spec/proposal_b.contract.json](spec/proposal_b.contract.json).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps -e .
python -m juniper_encoder spec verify --contract spec/proposal_b.contract.json
python -m juniper_encoder routing conformance --fixtures tests/fixtures/routing
python -m pytest -q
```

The PyTorch/CUDA runtime is pinned in `requirements/flowbox.lock` from the official CUDA 12.8 index and has been imported on FLOWBOX. The RTX 2060 reports native BF16 unsupported; the FP16 AMP probe passes. Training and deployment still require the separate data/checkpoint gates below.

The checked-in first-party fixture is documented in [manifests/sources/approved.json](manifests/sources/approved.json) and [manifests/data/frozen.json](manifests/data/frozen.json). It supports provenance, split, tokenizer, and overfit-pipeline verification only; it must not be presented as a production corpus.

## Reproduction and evidence

See [docs/REPRODUCE.md](docs/REPRODUCE.md), [docs/TRAINING.md](docs/TRAINING.md), [docs/DATA.md](docs/DATA.md), and [docs/HANDOFF.md](docs/HANDOFF.md). Heavy data, checkpoints, indexes, and exports must be referenced by immutable external artifact identity; they do not belong in ordinary Git history.

The Apache-2.0 software license in [LICENSE](LICENSE) does not grant rights to any future corpus, tokenizer source material, or model artifacts.
