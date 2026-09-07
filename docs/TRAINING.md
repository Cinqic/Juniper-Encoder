# Training protocol

The resolved protocol is controlled by the locked configs in `configs/locks/` and the supplied frozen instructions. The initial seed is 1729; 1730 and 1731 are pilot-variability seeds only. The intended foundation budgets are 2,000,000-token bounded pilot and 25,000,000-token first foundation stage, not claims that either is sufficient.

Training is from random initialization. The temporary MLM decoder is tied to the embedding only during the MLM stage and must not appear in a deployment export. Checkpoints must contain safe tensor payloads plus validated metadata for RNG, sampler, optimizer, scheduler, source/config/tokenizer/data identities, counters, and predecessor lineage.

This candidate has not trained: the verified PyTorch runtime is available, but no lawful approved corpus, processed split manifest, or reviewed labels are materialized. No `TRAINED`, `EVALUATED`, or `QUALITY_QUALIFIED` label is valid.
