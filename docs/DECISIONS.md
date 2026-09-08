# Decisions

- `FROZEN_CONTRACT`: Proposal B bytes are preserved in `spec/PROPOSAL_B_FROZEN.md` and pinned by SHA-256.
- `IMPLEMENTATION_CONVENTION`: semantic JSON uses UTF-8, `ensure_ascii=False`, no optional whitespace, fixed field order, strict finite JSON, and no slash escaping.
- `IMPLEMENTATION_CONVENTION`: content uses raw strict UTF-8 byte BPE; no normalization or regex pretokenization.
- `QUALIFICATION_POLICY`: absolute project thresholds remain null until independent Sol protocol review. A null threshold blocks quality qualification.
- `FROZEN_SPEC_UPDATE`: the complete 27-section Proposal B was supplied and is preserved byte-for-byte at `spec/PROPOSAL_B_FROZEN.md`; its SHA-256 is pinned in the machine contract. The earlier 18-section pasted artifact remains preserved as `spec/PROPOSAL_B_PARTIAL_FROZEN.md` for audit history.
- `DATA_SCOPE`: the checked-in sources are a small first-party routing fixture plus a repository-technical snapshot marked `training_only`; the snapshot can support tokenizer/foundation engineering but is not a reviewed production corpus. Separately authored routing labels remain a small pipeline fixture, so production quality claims stay blocked.
