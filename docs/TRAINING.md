# Training protocol

The resolved protocol is controlled by the locked configs in `configs/locks/` and the supplied frozen instructions. The initial seed is 1729; 1730 and 1731 are pilot-variability seeds only. The intended foundation budgets are 2,000,000-token bounded pilot and 25,000,000-token first foundation stage, not claims that either is sufficient.

Training is from random initialization. The temporary MLM decoder is tied to the embedding only during the MLM stage and must not appear in a deployment export. Checkpoints must contain safe tensor payloads plus validated metadata for RNG, sampler, optimizer, scheduler, source/config/tokenizer/data identities, counters, and predecessor lineage.

The repository now materializes a lawful first-party fixture and a frozen split manifest, but the fixture is not large enough to produce the required 16,121 merges and has no reviewed downstream routing labels. A separate toy-tokenizer overfit run exercised real FP16 CUDA MLM updates and safe-tensor checkpoint emission; it is an experiment, not a foundation or deployment checkpoint. No `QUALITY_QUALIFIED` label is valid.
