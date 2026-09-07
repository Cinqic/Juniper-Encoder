# Data and rights

`manifests/sources/approved.json` contains one immutable, first-party fixture owned by Cinqic. Public availability is not treated as a license. The fixture has a reviewed checksum, rights evidence, attribution, permitted use, redistribution status, language/domain, and exclusions. It is a small 3,113-byte research fixture, not a production corpus.

The materialized frozen manifest contains 32 provenance-preserving records with deterministic cleaning, duplicate-family grouping, and disjoint assignments: train 27, development 3, calibration 0, test 2. The zero calibration count is an explicit insufficiency, not a quality result. Held-out capability identities and their example families must remain outside tokenizer training, MLM, supervised training, hard-negative mining, and synthetic generation. The Apache-2.0 repository license does not label a mixed training corpus.
