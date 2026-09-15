# Candidate handoff (SUPERSEDED — HISTORICAL)

> **This handoff is superseded and must not be acted on.**
> Juniper Encoder was retired on 2026-09-15. It is not under development and no one is expected to continue it.
> The canonical current status is [RETIRED.md](../RETIRED.md).
>
> The text below is preserved unchanged as the historical handoff written while the project was still active.
> Its status line and "Required next action" describe the state at that time, not the final state.

---

## Historical handoff (as written during development)

## Current status

`IMPLEMENTED / TESTED / NOT RELEASED / AWAITING_SOL`

Repository: `https://github.com/Cinqic/Juniper-Encoder`

Branch: `work/luna-encoder-proposal-b`
Frozen spec SHA-256: `ce881075b84e36a28f673e1a41199e91cf0426baf66f788de880d86c156109b2`

The implementation branch contains the complete frozen spec, deterministic contract/tokenizer/routing/data/checkpoint code, a first-party routing fixture, a reproducible repository-technical tokenizer corpus, and a verified FLOWBOX CUDA runtime lock. The current worktree has also exercised a real full-vocabulary tokenizer, foundation/downstream checkpoint chain, model-driven evaluation, calibration, floating export, and portable INT8 export. Those generated weights and reports are local ignored artifacts, not an immutable candidate package.

The small authored dataset fails the project quality gates. The exact target benchmark matrix is complete at 84/84 measured cases, but the INT8 gate rejects the matched export on p50/p95 latency and isolated CUDA peak memory is unavailable. Independent Sol/Astra review has not occurred. Therefore this remains a reviewable implementation handoff, not a release candidate.

## Required next action

Obtain a sufficiently large lawful corpus and reviewed downstream labels, then rerun the gated tokenizer, foundation, downstream, calibration, evaluation, export, benchmark, and resume workflows at one immutable source SHA. A separate Sol clean clone must review the exact resulting candidate; Astra may review only after Sol approves that same immutable candidate. Do not merge, auto-merge, publish, or release.
