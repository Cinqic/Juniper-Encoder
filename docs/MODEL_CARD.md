# Juniper Encoder model card

**Status: NOT RELEASED.** No trained model or qualified deployment artifact is present.

The frozen architecture targets 49,441,796 deployed parameters and 49,308,672 shared-backbone parameters. Mode totals are retrieval 49,439,744, decision 49,310,211, reranking 49,309,185, and fast pipeline 49,441,283. These are contract counts, not evidence that weights exist.

Training is specified as random initialization with a temporary MLM stage followed by retrieval, reranker, classifier, and optional joint stabilization. The repository contains only a 3,113-byte first-party fixture: enough for data/provenance tests, not enough for the frozen tokenizer or a production model. A toy-tokenizer overfit experiment exists as pipeline evidence, but actual production tokens, downstream steps, reviewed labels, calibration values, metrics, critical slices, target precision, INT8 results, failure history, and release artifact IDs are not available. No target is supported or released.
