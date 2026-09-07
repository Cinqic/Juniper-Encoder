# Juniper Encoder

Juniper Encoder is a contract-first, encoder-only routing model for registered agents, tools, and skills. This repository contains the implementation and reproducibility controls for the supplied frozen Proposal B instructions.

## Status

`IMPLEMENTED / SMOKE_TESTED / NOT RELEASED`

The repository currently has no approved training sources, selected checkpoint, calibrated metadata, durable model artifacts, independent Sol approval, or Astra approval. Consequently it makes no quality, training-completion, GPU/INT8, or release claim. Those are explicit gates, not inferred from the presence of code.

The frozen authority is [spec/PROPOSAL_B_FROZEN.md](spec/PROPOSAL_B_FROZEN.md), SHA-256 `9363e892bef02d7b26856c31282bf8ca3902d1a140700e970d808057d6e77d1b`. The machine-readable contract is [spec/proposal_b.contract.json](spec/proposal_b.contract.json).

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

## Reproduction and evidence

See [docs/REPRODUCE.md](docs/REPRODUCE.md), [docs/TRAINING.md](docs/TRAINING.md), [docs/DATA.md](docs/DATA.md), and [docs/HANDOFF.md](docs/HANDOFF.md). Heavy data, checkpoints, indexes, and exports must be referenced by immutable external artifact identity; they do not belong in ordinary Git history.

The Apache-2.0 software license in [LICENSE](LICENSE) does not grant rights to any future corpus, tokenizer source material, or model artifacts.
