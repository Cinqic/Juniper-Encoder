# Juniper Encoder — retirement record

Retirement date: 2026-09-15

## Status

`RETIRED / RESEARCH COMPLETE / NOT RELEASED / NOT PRODUCTION-QUALIFIED`

Retirement is final. Juniper Encoder is no longer under active development, is not awaiting Sol or Astra review work, is not on an active release path, and is not waiting on additional training or data as a planned next step. The repository is public and read-only, retained for research history and reproducibility.

This document and `README.md` are the current authority for project status. Everything else in the repository is historical evidence from the period in which it was written.

## Project

Juniper Encoder was a small encoder-only model intended to route user requests among registered agents, tools, and skills, implemented against the frozen Proposal B design (`spec/PROPOSAL_B_FROZEN.md`, SHA-256 `ce881075b84e36a28f673e1a41199e91cf0426baf66f788de880d86c156109b2`) and its machine-readable contract (`spec/proposal_b.contract.json`, SHA-256 `34616027034eec33b728b8408485c10939e04575ad62077c85e177c270610ac7`).

## What was completed

- Exact dense encoder implementation and a physical parameter-count audit against the frozen contract.
- Raw UTF-8 byte BPE tokenizer with a deterministic lossless artifact format and conformance checks.
- Strict registry/index handling, request formatting, and conditional three-class routing (`CALL` / `NO_CALL` / `CLARIFY`) with deterministic error precedence.
- Bounded data acquisition, provenance tracking, deterministic family splits, and leakage auditing, over two first-party sources.
- A real token-budgeted MLM foundation stage plus chained retrieval, reranker, and classifier training on CUDA.
- Resumable safetensors checkpoints with validated RNG, optimizer, scheduler, and lineage metadata.
- Temperature calibration and threshold selection.
- Model-driven evaluation with explicit denominators and slices.
- Floating-point deployment export and a matched portable INT8 quantized export.
- Full execution of the configured target benchmark matrix.
- Dependency-free packaging, pinned environment locks, reproduction documentation, a test suite, and candidate CI that pins an exact reviewed SHA.

## Final measured results

| Item | Result |
| --- | --- |
| Deployed parameters | 49,441,796 |
| Shared backbone parameters | 49,308,672 |
| Contract verification / traceability | PASS, 27 rows |
| Exact model parameter audit | PASS, zero PAD row |
| Tokenizer | 16,121 merges, 16,384 vocab; two independent builds matched at identity `d7ecca67…` |
| Tokenizer corpus | 85 documents, 445,464 bytes, SHA-256 `a76a31fb…` |
| Foundation run | 5,202 successful updates, 2,000,637 tokens consumed, 4 skipped FP16 overflow attempts, budget reached |
| Downstream training | classifier, reranker, retrieval — three updates each |
| Routing evaluation set | 44 authored requests: 26 train, 10 development, 8 held-out test |
| Development exact accuracy | ~0.10 |
| Held-out test exact accuracy | 0.00 |
| Development false-CALL rate | 1.0 |
| Aggregate decision macro-F1 | ~0.2807 |
| End-to-end exact accuracy | ~0.0909 |
| Calibration | temperature 2.68744694; development NLL 1.2896 → 0.9484; ECE 0.3085 → 0.0039 |
| Floating export | PASS |
| Portable INT8 export | PASS technically — 72 operators/tensors, finite probe |
| Benchmark coverage | 84/84 configured cases measured (20 warmups, 200 measurements) |
| INT8 matched latency | p50 1.1581x, p95 1.1078x against a 1.05x gate |
| INT8 artifact size ratio | 0.38345x |
| INT8 isolated CUDA peak memory | unavailable (process RSS only) |
| INT8 disposition | `INT8_REJECTED` |
| Quality gate policy | thresholds intentionally null — `BLOCKED_QUALITY_POLICY` |
| Selected production checkpoint | `BLOCKED_CHECKPOINT`, none selected |
| Sol review | never performed |
| Astra review | never performed |
| Release status | `NOT RELEASED` — no model, package, or GitHub Release |

Calibration improved the measured development NLL and ECE. That is a calibration result on a tiny fixture and does not establish production quality.

## What was not achieved

- Production routing quality was not demonstrated. Held-out exact accuracy was 0.00 and the development false-CALL rate was 1.0.
- No qualified production checkpoint exists; `manifests/checkpoints/selected.json` remains `BLOCKED_CHECKPOINT`.
- No production model was released. No GitHub Release, model package, or distributed checkpoint exists.
- No Sol approval was obtained.
- No Astra approval was obtained.
- INT8 did not satisfy the project deployment gate; it was rejected on p50 and p95 latency ratios.
- Isolated CUDA peak-memory qualification was unavailable; the benchmark recorded cumulative process RSS instead.
- The tiny first-party supervision and evaluation fixture is not production-scale evidence and was never independently reviewed as production supervision.
- Absolute quality thresholds were never set, because setting them required the independent protocol review that never occurred. Quality qualification was therefore structurally blocked, not merely failed.
- Native BF16 was unsupported on the available RTX 2060; FP16 AMP was the measured fallback rather than the specified reference precision.
- The trained weights, tokenizer artifacts, and exports were local ignored artifacts on the FLOWBOX workstation. They were never promoted to an immutable durable artifact store, and they are not part of this repository.

## Interpretation

The implementation and reproducibility machinery were completed far enough to test the idea. The contract, tokenizer, model, training path, calibration, evaluation, export, quantization, and benchmark stages all executed end to end and produced real measurements. Those measurements did not justify qualification or further release work: the available data and model experiment produced routing quality far below anything deployable, and the deployment gate failed independently on latency.

This is a usable negative result. The architecture and pipeline were shown to be implementable and reproducible; the data and training program required to make them useful was never built, and the project was ended before it was.

## Reason for retirement

Cinqic has chosen to focus its AI research effort on AAA and broader adaptive and autonomous model research rather than continuing a standalone routing encoder as an independent model project.

Juniper Encoder did not become AAA, and AAA does not inherit Juniper Encoder's scientific results. Individual lessons or components may inform later work, but Juniper Encoder itself is finished.

## Preservation policy

- Git history is preserved, including the complete Proposal B implementation ancestry merged from PR #1.
- Historical pull-request discussion is preserved, including the immutable candidate descriptor posted during review.
- Frozen specifications are preserved byte-for-byte and were not modified during retirement.
- Source, tests, configs, manifests, and reports are preserved.
- Negative and incomplete evidence is preserved. Failed gates, blocked states, and unreviewed statuses were not rewritten into passes.
- Historical documents were not rewritten to pretend retirement was known in advance. Documents that record an earlier state continue to do so.
- This archive is not a maintained dependency and not a supported model.

### Preserved historical inconsistencies

These are intentional. They are artifacts of chronology, not errors to be corrected:

- `reports/failures/ledger.json` is a snapshot from an early stage of the work. Its `BLOCKED_TOKENIZER_CORPUS` entry describes the 3,113-byte seed fixture yielding 1,025 merges. A later commit added the first-party repository-technical corpus, and the full 16,121-merge tokenizer was subsequently built. The ledger was left unmodified so its recorded hashes remain valid. Every blocker it lists became a terminal outcome when the project ended.
- The candidate descriptor in the PR #1 conversation was computed at source commit `a6c1ecb`, before the complete 27-section Proposal B was supplied. Its `frozen_spec_sha256` of `9363e892…` refers to the earlier implementation brief, which is now preserved separately at `spec/IMPLEMENTATION_REVIEW_BRIEF.md`. The final frozen specification identity is `ce881075…`. The descriptor was not recomputed or corrected.
- `docs/HANDOFF.md`, `manifests/tokenizer/provisional.json`, `manifests/checkpoints/selected.json`, and `reports/failures/ledger.json` continue to record `AWAITING_SOL`, `AWAITING_ASTRA`, `PROVISIONAL_NOT_RELEASED`, and `BLOCKED_CHECKPOINT`. Those were the true final states. The project ended with those gates unresolved; they were not resolved, and they were not falsified into approvals.

## Final state

Final retirement tag: `retired-2026-09-15`

The repository was placed into GitHub's archived read-only state after this record was merged and tagged.
