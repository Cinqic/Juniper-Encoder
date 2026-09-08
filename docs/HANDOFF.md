# Candidate handoff

## Current status

`IMPLEMENTED / TESTED / NOT RELEASED / AWAITING_SOL`

Repository: `https://github.com/Cinqic/Juniper-Encoder`

Branch: `work/luna-encoder-proposal-b`
Frozen spec SHA-256: `ce881075b84e36a28f673e1a41199e91cf0426baf66f788de880d86c156109b2`

The implementation branch contains the complete frozen spec, deterministic contract/tokenizer/routing/data/checkpoint code, a first-party routing fixture, a reproducible repository-technical tokenizer corpus, and a verified FLOWBOX CUDA runtime lock. The current worktree has also exercised a real full-vocabulary tokenizer, foundation/downstream checkpoint chain, model-driven evaluation, calibration, floating export, and portable INT8 export. Those generated weights and reports are local ignored artifacts, not an immutable candidate package.

The small authored dataset fails the project quality gates, the required target benchmark matrix is not complete, and independent Sol/Astra review has not occurred. Therefore this remains a reviewable implementation handoff, not a release candidate.

## Required next action

Obtain a sufficiently large lawful corpus and reviewed downstream labels, then rerun the gated tokenizer, foundation, downstream, calibration, evaluation, export, benchmark, and resume workflows at one immutable source SHA. A separate Sol clean clone must review the exact resulting candidate; Astra may review only after Sol approves that same immutable candidate. Do not merge, auto-merge, publish, or release.
