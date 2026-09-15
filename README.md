# Juniper Encoder

A small encoder-only model for routing user requests to registered agents, tools, and skills.

## RETIRED — 2026-09-15

`RETIRED / RESEARCH COMPLETE / NOT RELEASED / NOT PRODUCTION-QUALIFIED`

**This project is finished and this repository is an archive.** It is not under development, not awaiting review, and not on a release path. See **[RETIRED.md](RETIRED.md)** for the canonical retirement record.

The source, specifications, manifests, and measured evidence remain published for historical and research purposes. Nothing here is a maintained dependency or a supported model.

### Final outcome

The frozen Proposal B design was implemented end to end and executed: contract-exact model construction and parameter audit, tokenizer training, a CUDA foundation run, chained downstream training, calibration, evaluation, floating and INT8 exports, and the full 84-case benchmark matrix. The measured routing quality was inadequate for qualification — held-out exact accuracy `0.00`, development false-CALL rate `1.0`, aggregate decision macro-F1 `~0.2807`. The matched INT8 export was rejected on latency. No production checkpoint was selected, no independent review occurred, and no model was released.

Full measurements, including the favorable ones, are in [RETIRED.md](RETIRED.md).

## Repository contents

The frozen authority is [spec/PROPOSAL_B_FROZEN.md](spec/PROPOSAL_B_FROZEN.md), SHA-256 `ce881075b84e36a28f673e1a41199e91cf0426baf66f788de880d86c156109b2`. The machine-readable contract is [spec/proposal_b.contract.json](spec/proposal_b.contract.json).

The checked-in first-party fixture is documented in [manifests/sources/approved.json](manifests/sources/approved.json) and [manifests/data/frozen.json](manifests/data/frozen.json). It supports provenance, split, tokenizer, and overfit-pipeline verification only; it was never a production corpus.

## Running the archived checks

The contract-only checks still run without PyTorch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps -e .
python -m juniper_encoder spec verify --contract spec/proposal_b.contract.json
python -m juniper_encoder routing conformance --fixtures tests/fixtures/routing
python -m pytest -q
```

The PyTorch/CUDA runtime is pinned in `requirements/flowbox.lock` from the official CUDA 12.8 index. The RTX 2060 used for this work reports native BF16 unsupported; the FP16 AMP probe passes.

## Historical documentation

[docs/REPRODUCE.md](docs/REPRODUCE.md), [docs/TRAINING.md](docs/TRAINING.md), [docs/DATA.md](docs/DATA.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md), and [docs/DECISIONS.md](docs/DECISIONS.md) describe the work as it stood during development. [docs/HANDOFF.md](docs/HANDOFF.md) is a superseded historical handoff. Heavy data, checkpoints, indexes, and exports were referenced by immutable external artifact identity and are not in Git history.

The Apache-2.0 software license in [LICENSE](LICENSE) does not grant rights to any corpus, tokenizer source material, or model artifacts.
