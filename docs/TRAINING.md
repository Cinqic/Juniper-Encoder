# Training protocol

The resolved protocol is controlled by the locked configs in `configs/locks/` and the supplied frozen instructions. The initial seed is 1729; 1730 and 1731 are pilot-variability seeds only. The intended foundation budgets are 2,000,000-token bounded pilot and 25,000,000-token first foundation stage, not claims that either is sufficient.

Training is from random initialization. The temporary MLM decoder is tied to the embedding only during the MLM stage and must not appear in a deployment export. Checkpoints must contain safe tensor payloads plus validated metadata for RNG, sampler, optimizer, scheduler, source/config/tokenizer/data identities, counters, and predecessor lineage.

The repository now materializes two distinct first-party inputs: a small general routing fixture and a deterministic repository-technical snapshot used only for tokenizer/foundation training. The latter is excluded before routing split assignment. The supervised routing rows are separately authored, identity-split, and explicitly marked as a small engineering fixture rather than reviewed production labels.

The FLOWBOX run produced a real full-vocabulary tokenizer, a CUDA foundation update, chained retrieval/reranker/classifier updates, calibration metadata, a floating export, and a portable INT8 export. These local artifacts are ignored from Git and remain provisional: the authored data is too small, the measured quality gates fail, the exact 84-case benchmark matrix is complete but INT8 latency gates fail and isolated CUDA peak memory is unavailable, and no `QUALITY_QUALIFIED` or release label is valid.
