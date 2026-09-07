# Decisions

- `FROZEN_CONTRACT`: Proposal B bytes are preserved in `spec/PROPOSAL_B_FROZEN.md` and pinned by SHA-256.
- `IMPLEMENTATION_CONVENTION`: semantic JSON uses UTF-8, `ensure_ascii=False`, no optional whitespace, fixed field order, strict finite JSON, and no slash escaping.
- `IMPLEMENTATION_CONVENTION`: content uses raw strict UTF-8 byte BPE; no normalization or regex pretokenization.
- `QUALIFICATION_POLICY`: absolute project thresholds remain null until independent Sol protocol review. A null threshold blocks quality qualification.
- `BLOCKER`: the supplied artifact contains 18 visible numbered sections while it refers to a complete 27-section Proposal B. No missing sections are invented; the preserved bytes remain authoritative pending any separately supplied frozen text.
