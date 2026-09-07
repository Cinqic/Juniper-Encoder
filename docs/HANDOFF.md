# Candidate handoff

## Current status

`IMPLEMENTED / TESTED / NOT RELEASED / AWAITING_SOL`

Repository: `https://github.com/Cinqic/Juniper-Encoder`

Branch: `work/luna-encoder-proposal-b`
Frozen spec SHA-256: `ce881075b84e36a28f673e1a41199e91cf0426baf66f788de880d86c156109b2`

The implementation branch contains the complete frozen spec, deterministic contract/tokenizer/routing/data/checkpoint code, a first-party 32-record fixture, CPU-side tests, and a verified FLOWBOX CUDA runtime lock. No complete deployment tokenizer, foundation/downstream checkpoint, calibration, target benchmark, INT8 qualification, candidate descriptor, Sol approval, or Astra approval exists. Therefore this is a reviewable implementation handoff, not a release candidate.

## Required next action

Obtain a sufficiently large lawful corpus and reviewed downstream labels, then run the gated tokenizer, foundation, downstream, calibration, evaluation, export, and resume workflows. A separate Sol clean clone must review the exact resulting candidate; Astra may review only after Sol approves that same immutable candidate. Do not merge, auto-merge, publish, or release.
