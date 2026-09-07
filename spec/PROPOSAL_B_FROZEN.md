# Juniper Encoder — end-to-end implementation and independent review instructions

**Implementer:** Codex GPT-5.6 Luna  
**First independent reviewer:** GPT-5.6 Sol  
**Final independent reviewer:** GPT-6 Astra  
**Developer:** Cinqic  
**Training workstation:** FLOWBOX  
**Authority:** The complete 27-section Juniper Encoder Proposal B supplied with this task, status FROZEN.  
**Terminal deliverable:** An evidence-backed, immutable candidate and independent review decisions. **NOT RELEASED.**

## 1. Mission, authority, and boundaries

Luna: build Juniper Encoder from random initialization through tokenizer creation, lawful curated data preparation, foundation pretraining, routing training, calibration, evaluation, deployment export, INT8 qualification, reproducible packaging, and exact-candidate handoff. Continue through all authorized phases when their gates pass. Do not stop at scaffolding, a toy training run, passing unit tests, or a plan. Do not claim that sufficient training or quality is guaranteed by a token budget.

Proposal B is the sole architectural and routing authority. Save its complete supplied text verbatim as `spec/PROPOSAL_B_FROZEN.md`, hash its bytes, and commit it before implementation. This brief incorporates that entire document by reference; summaries here never replace it. Preserve all 27 sections, every count, formula, boundary, token ID, error, tie rule, benchmark, and INT8 gate. Never substitute Proposal A, Juniper Router, an earlier memory, a pretrained model, or library defaults. Formatting artifacts in the supplied equations must be interpreted consistently with their explicit shapes and prose; any genuinely conflicting contract requires a recorded blocker, not a guessed revision.

Classify each decision as:

1. `FROZEN_CONTRACT`: copied from Proposal B; cannot change.
2. `TRAINING_PROTOCOL`: unspecified training choice, documented with rationale, evidence, and version; cannot change deployment semantics or parameterization.
3. `IMPLEMENTATION_CONVENTION`: deterministic detail needed to realize the contract, such as JSON escaping; documented, tested, and shared by all artifacts.
4. `QUALIFICATION_POLICY`: additional candidate-quality or resource gates; explicitly not a new Proposal B requirement.

Never weaken a frozen gate to accommodate hardware, data, time, or a failed experiment. Never invent approval, benchmark measurements, hardware access, training completion, licensing permission, or review results. A blocked expensive phase does not prevent unrelated authorized implementation work, but no dependent gate may pass.

Luna implements, tests, preserves evidence, commits, pushes implementation branches, and prepares draft candidate PRs. Luna never approves, merges, enables auto-merge, creates a release, publishes a model as released, or impersonates a reviewer. Sol independently reproduces and audits the exact candidate from a clean clone. Astra performs a separate final audit after Sol approves that same candidate. If either reviewer authors a material fix, a separate independent review of that fix is required; prefer routing fixes back to Luna.

Any blocker or disagreement means `REQUEST_CHANGES / NOT RELEASED`. A new source SHA, changed weights, tokenizer, data, calibration, export, runtime lock, or artifact hash invalidates affected approvals and requires renewed review. Both approvals permit a later separately authorized merge/release process; they do not themselves perform it. If the named reviewer is unavailable, prepare their complete handoff and record `AWAITING_SOL` or `AWAITING_ASTRA`; never substitute Luna's self-review.

## 2. Repository discovery and canonical structure

GitHub is canonical for source, frozen specifications, small manifests, issues/PRs, qualification policies, and review evidence. Heavy datasets and checkpoints belong in an authorized durable artifact store, referenced by immutable revision/object identity, byte length, and SHA-256 from GitHub. Do not put bulk training data, caches, credentials, or large weights into ordinary Git history. Git LFS is acceptable only after verifying quotas, retention, and clean-clone materialization. Expiring CI artifacts alone are not archival storage.

Context checked when writing this brief: a GitHub search returned Cinqic/Juniper-App and Cinqic/Juniper-Router but no Juniper Encoder repository. The Router README describes a different ~135M SmolLM2-based conversation/orchestration project. The App README describes a separate local application and runtime. These were targeted context checks, not full-repository audits. Recheck live state when starting. Do not rename, overwrite, archive, merge into, or repurpose either project for Encoder. Sources: [Juniper Router README](https://github.com/Cinqic/Juniper-Router/blob/main/README.md), [Juniper App README](https://github.com/Cinqic/Juniper-App/blob/main/README.md).

Run these read-only discovery commands first, recording exit codes as well as output:

```bash
set -euo pipefail
gh auth status
gh repo list Cinqic --limit 200 --json name,nameWithOwner,isArchived,url
gh repo view Cinqic/Juniper-Encoder --json nameWithOwner,url,defaultBranchRef,visibility
```

Do not treat an authentication/network failure as proof of absence. Resolve the canonical repository and existing `AGENTS.md`, contribution, security, licensing, and review rules. If Encoder does not exist, prepare the project in its own directory and document the exact proposed repository name. Use existing session authority to create it if repository creation and visibility are already authorized; otherwise finish the reviewable local scaffold and report that narrow external blocker before creation. Do not silently choose public visibility or a license. Source-code, weights, tokenizer, and data licensing must each be established; do not inherit Router's or App's license by assumption.

After the remote is verified:

```bash
export JE_REMOTE='https://github.com/Cinqic/Juniper-Encoder.git'
export JE_ROOT="$PWD/Juniper-Encoder"
git clone "$JE_REMOTE" "$JE_ROOT"
cd "$JE_ROOT"
git status --porcelain=v1
rg --files -g AGENTS.md -g '*CONTRIBUT*' -g '*LICENSE*' -g '*SECURITY*'
git switch -c work/luna-encoder-proposal-b
```

If an existing directory is present, inspect it and preserve user work; do not rerun clone over it or reset it. Never use force push, destructive reset, or automatic deletion as recovery.

Create this minimum repository layout, adapting names only when established repository rules require it and updating every command consistently:

```text
spec/PROPOSAL_B_FROZEN.md
spec/proposal_b.contract.json
spec/traceability.csv
schemas/{registry,request,output,metadata,error,checkpoint,artifact}.schema.json
configs/{model,tokenizer,data,splits,mlm,retrieval,reranker,classifier,joint}.yaml
configs/{calibration,evaluation,quality-gates,benchmarks,quantization,resources}.yaml
configs/locks/                         # fully resolved run configs, no moving defaults
src/juniper_encoder/{model,tokenizer,formatting,registry,index,routing,errors}.py
src/juniper_encoder/{train,calibrate,evaluate,quantize,export,cli}/
tests/{contract,numerics,tokenizer,data,training,routing,export,integration}/
tests/fixtures/                        # compact lawful deterministic fixtures
scripts/{bootstrap,preflight,verify-candidate,review-candidate}.sh
requirements/{bootstrap,flowbox,cpu-review}.lock
pyproject.toml
.github/workflows/{ci,candidate}.yml
manifests/{sources,tokenizer,data,checkpoints,artifacts}/
reports/{preflight,conformance,data,experiments,evaluation,benchmarks,failures}/
reviews/{sol,astra}/
docs/{REPRODUCE,TRAINING,DATA,DECISIONS,SECURITY,MODEL_CARD,HANDOFF}.md
README.md
CHANGELOG.md
```

The CLI below is an interface Luna must implement, document, and test; it is not claimed to exist today. `python -m juniper_encoder` is the single entrypoint. Every listed subcommand must have strict schema validation, `--help`, meaningful nonzero failure exit codes, a resolved-config record, and machine-readable output. Never build a command that writes a hard-coded PASS. All aliases such as `selected` must resolve once to immutable identities in the run manifest. Shell commands assume execution from repository root in the locked environment.

## 3. FLOWBOX preflight and environment lock

Expected hardware, to verify rather than assume: Linux Mint, Ryzen 7 5700G, 16 GB RAM, RTX 2060 with 6 GB VRAM, and constrained disk capacity. This remote instruction-writing environment is not FLOWBOX. Do not report its hardware as FLOWBOX measurements.

Capture:

```bash
uname -a
cat /etc/os-release
lscpu
free -b
lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
df -B1 .
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv
python3 --version
git --version
gh --version
```

If a query field is unsupported, record the failed command and use a documented supported query; do not discard the failure. Record CPU threads, GPU capability, driver, temperature/power limits, current competing processes, disk paths, filesystem, and available memory. Avoid logging secrets or unrelated private process arguments. Never upgrade the host driver, repartition disks, kill unrelated applications, or modify global Python merely to make the project work.

Resolve and test a compatible Python/PyTorch/CUDA-runtime combination supporting the actual GPU. A CUDA version shown by `nvidia-smi` is not proof that a selected wheel, kernel, or local toolkit works. Prefer a project virtual environment and verified official distributions. Pin Python patch version, every direct/transitive dependency and wheel hash, index/source locations, build tools, runtime providers, and OS libraries that affect numerical behavior. Preserve installable wheel artifacts or an immutable retrieval manifest. Maintain a separate explicit CPU-review lock. Do not insert invented current package versions into the lock.

Implement `scripts/bootstrap.sh` so the canonical reconstruction is:

```bash
bash scripts/bootstrap.sh --lock requirements/flowbox.lock
source .venv/bin/activate
python -m pip check
bash scripts/preflight.sh --output reports/preflight/flowbox.json
python -m juniper_encoder env verify --lock requirements/flowbox.lock
```

Bootstrap must install using pinned/hash-verified requirements, then install the local project without silently resolving new dependencies. Record the resolved environment and lock hash. Reconstruct it in a second fresh environment before training.

Use native BF16 only when hardware and actual forward/backward kernels validate it. Probe `torch.cuda.is_bf16_supported(including_emulation=False)` when supported by the pinned API; do not mistake emulation for native support. Expect to validate FP16 AMP with GradScaler on this RTX 2060; FP32 is the numerical/debug fallback. BF16 preference is not permission to force an unsupported dtype. Reference deployment remains validated BF16 or FP16 per Proposal B; an FP32 oracle is an additional diagnostic, not a substituted reference for convenient size ratios. See [PyTorch BF16 capability API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html).

Set reproducibility variables before importing torch:

```bash
export PYTHONHASHSEED=1729
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

Seed Python, NumPy, CPU torch, CUDA, samplers, and masking. Set deterministic algorithms strictly, disable cuDNN benchmarking and TF32 for the reference protocol, and pin attention kernel dispatch. Begin with zero data-loader workers; add workers only with an explicit reproducible sampling/resume proof. Record thread settings. Same-environment determinism and cross-platform numerical equivalence are different claims: PyTorch does not guarantee identical results across releases or platforms. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Implement configurable resource guards with initial engineering defaults: at least 1 GiB available host RAM reserve beyond measured process peak; at least 512 MiB VRAM reserve; free disk exceeding planned download/extraction/cache growth plus two full checkpoint sizes plus 5 GiB. Measure actual checkpoints, including optimizer state. Stream or memory-map data and bound caches. No sustained swap-dependent training. Preflight must estimate wall time from measured tokens/sec and actual budgets; no unsupported completion promises. Set a recorded finite run budget and checkpoint policy before unattended training.

**Gate:** compatible locked install reconstructed; CPU and GPU forward/backward finite; allocation/throughput measurements obtained; sufficient resources; approved data/artifact locations exist. Otherwise `BLOCKED_ENVIRONMENT` or `BLOCKED_RESOURCES`, preserve logs and fix only the affected environment/configuration.

## 4. Frozen-contract implementation and traceability

Extract every atomic requirement into `spec/traceability.csv`: requirement ID, Proposal B section and exact text, implementation symbol, config field, test ID, expected result, evidence path/hash, status. Cross-reference all 27 sections and numerical gates. An independent script verifies coverage, duplicate IDs, missing tests, changed spec hashes, and undocumented config keys. A coverage row is not evidence until its test actually executes.

Machine-readable configuration must lock 12 blocks, width 512, 8 Q and 8 KV heads, head dimension 64, GeGLU branches 1,536/1,536, fused gate-then-value projection, exact erf GELU, full bidirectional attention with 1/8 scaling, two Pre-LayerNorms and final LayerNorm with bias/scale and population variance epsilon 1e-5, full 64-dimensional adjacent-pair RoPE theta 10,000, the exact four dropout sites at 0.1 during training, and all explicit absences in Proposal B. No architectural override flag is allowed in deployment.

Implement a transparent unfused FP32 oracle first. LayerNorm statistics and attention softmax are FP32 even under AMP; perform pooling sums and L2 normalization in FP32, classifier probabilities in FP32, and return FP32 retrieval vectors and reranker scalars. Prove cast placement rather than assuming autocast does it. Verify Q/K-only adjacent-coordinate RoPE, positive rotation convention, continuous pair positions, PAD positions zero, correct mathematical/storage matrix transposes, and masks. Generic framework RoPE often uses a different pairing layout: do not substitute it.

PAD is zeroed after embedding dropout, each residual addition, and final normalization. Key masking is negative infinity for PAD only, with no causal mask. The pooling mask selects ordinary token IDs only. Reject all-PAD/malformed inputs through the correct layer of the API; do not let all-masked softmax fabricate a routing output. Real structural tokens participate in attention. Reject nonfinite weights/activations/scores and projection norms below 1e-12; exactly 1e-12 is not below the threshold. Prevent a NaN from being converted into a valid ranking or decision.

Initialize embeddings and linear weights from an underlying N(0,0.02²), rejection-truncated at ±0.04, not clipped and not variance-corrected to a post-truncation standard deviation of 0.02. LayerNorm scales are one; all specified biases zero. Fix PAD row zero after initialization, gradients, optimizer updates, restoration, and export. Mask its gradient and optimizer moments, including the temporary tied MLM decoder path. Count its 512 entries anyway.

Compute parameter counts from actual named tensors, independent of configuration labels. Verify tensor shapes, storage aliases, presence/absence of biases, state-dict keys, and unique deployed parameters:

| Component/mode | Exact count |
|---|---:|
| Embedding table | 8,388,608 |
| Attention per block | 1,048,576 |
| GeGLU per block | 2,359,296 |
| LayerNorms per block | 2,048 |
| Complete block | 3,409,920 |
| Twelve blocks | 40,919,040 |
| Final LayerNorm | 1,024 |
| Shared backbone | 49,308,672 |
| Retrieval head | 131,072 |
| Decision classifier | 1,539 |
| Reranker | 513 |
| Retrieval participating | 49,439,744 |
| Decision participating | 49,310,211 |
| Reranking participating | 49,309,185 |
| Fast pipeline unique | 49,441,283 |
| Reranked pipeline/deployed total | 49,441,796 |

Verify participating counts by the actual module/tensor dependency set used in each mode, not only arithmetic from constants. Repeated passes do not multiply unique parameters. No tied deployed tensors. Quantization buffers, index vectors, tokenizer bytes, T, and thresholds are metadata rather than learned parameters.

Implement independent numerical fixtures and meaningful tests for forward formulas, train/eval dropout placement, initialization bounds, PAD invariance, bidirectionality, no document-position reset, batch/padding invariance, pooling exclusions, finite outputs/gradients, error precedence, ties, and heads. Optimized kernels must pass differential tests against the oracle with predeclared dtype tolerances and unchanged contract behavior. No approximate GELU, hidden extra dropout, causal packing, or fused normalization shortcut that changes the specification.

```bash
python -m juniper_encoder spec verify --contract spec/proposal_b.contract.json
python -m juniper_encoder model audit --config configs/model.yaml --output reports/conformance/parameters.json
python -m pytest -q tests/contract tests/numerics
```

**Gate:** all exact counts and structural/behavioral tests pass; traceability has no unexplained gaps. A count mismatch is a stop, not permission to resize a head or vocabulary.

## 5. Tokenizer: corpus, deterministic training, conformance

Train from scratch. Do not import a pretrained tokenizer or rely on default byte-level tokenizer behavior. First establish data partitions and an evaluation exclusion inventory so tokenizer training cannot consume calibration/test examples or reserved capability identities/descriptions.

Use approved training-only natural-language, dialogue, technical, and capability-style content with language/domain/source proportions in `configs/tokenizer.yaml`. Bound corpus size after a deterministic pilot; report bytes, documents, languages, domains, and compression/fertility on development data. Select files and records by pinned provenance and stable hash ordering, never unseeded streaming shuffle. No evaluation-guided corpus additions. If multilingual coverage cannot be meaningfully evaluated, state the supported/evaluated scope honestly; byte coverage alone does not establish multilingual routing quality.

Implement raw UTF-8 byte BPE with exactly seven structural IDs, bytes 7–262, and exactly 16,121 learned merges producing IDs 263–16,383 in learned order. Train each content field independently. No regex pretokenization, normalization, folding, whitespace cleanup, prefix space, or cross-field/document merges. Input must consist of valid Unicode scalar sequences: reject lone surrogates and invalid UTF-8 rather than replacement decoding. Noncharacters that are valid scalar values must not be rejected merely because they are uncommon.

Document deterministic pair-frequency counting, overlap treatment, stable tie breaking, and merge application. Initial convention: highest frequency first; equal frequencies ordered by numeric `(left_token_id,right_token_id)`; replacement left-to-right, non-overlapping; contiguous new IDs in learned order. Ensure each learned token has a unique byte representation and compatible merge history; test deterministic exclusion/handling of duplicate byte results. If the corpus cannot yield all 16,121 legitimate learned merge tokens, improve the lawful training corpus and retrain; never pad the vocabulary with fabricated unused entries. All training conventions are versioned and reproducible.

Literal `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, `[QUERY]`, and `[DOCUMENT]` in text encode as ordinary bytes. Only the formatter inserts structural IDs. Normal inference content contains no PAD/UNK/MASK structural IDs. Keep tokenizer APIs for ordinary content separate from formatter APIs.

Release artifacts: byte map, full vocabulary with lossless byte representation, ordered merges, configuration, implementation version/source SHA, corpus manifest/hash, conformance fixtures, and SHA-256 manifest. Decode ordinary token sequences by concatenating bytes and strict UTF-8 decoding; individual tokens can contain partial UTF-8 sequences. Verify exact full-string round-trip before truncation. Do not decode partial tokens independently and replace malformed fragments.

Test all 256 byte mappings as bytes; valid Unicode spanning 1–4 UTF-8 bytes; combining characters; emoji; whitespace; NUL/control characters; literal structural strings; JSON escapes; right-to-left text; long fields; surrogate rejection; no UNK for valid Unicode; boundaries; repeat-build identity; fresh-process encoding parity. Pin the Unicode whitespace classification/version used by request validation.

```bash
python -m juniper_encoder data plan --config configs/data.yaml --output manifests/sources/plan.json
python -m juniper_encoder tokenizer corpus --config configs/tokenizer.yaml
python -m juniper_encoder tokenizer train --config configs/tokenizer.yaml --output artifacts/tokenizer-build-a
python -m juniper_encoder tokenizer train --config configs/tokenizer.yaml --output artifacts/tokenizer-build-b
python -m juniper_encoder tokenizer compare --left artifacts/tokenizer-build-a --right artifacts/tokenizer-build-b
python -m juniper_encoder tokenizer conformance --artifact artifacts/tokenizer-build-a
```

Compare canonical payload bytes and hashes, excluding separately stored timestamps. Promote only identical payloads. Every subsequent dataset/checkpoint/index/export must pin the tokenizer identity and fail on mismatch. Correcting a tokenizer bug creates a new tokenizer/data lineage and invalidates downstream artifacts; it is not a drop-in repair.

## 6. Registry, request formatting, and routing before training

Implement schemas and deterministic formatting without waiting for learned weights. Test with controlled mock scores, then exercise the same code using actual model tensors. Mocks establish contract correctness, not routing quality.

Registry records have exactly the seven semantic fields, in frozen order: id, type, name, purpose, supported_requests, exclusions, required_inputs. Strictly validate strings/arrays/types, duplicate JSON keys, exact allowed types, nonempty ASCII capability IDs matching `[A-Za-z0-9._-]+`, unique case-sensitive IDs, and the 256-ordinary-token serialized-record limit. No extra semantic fields. Snapshot version/hash and runtime schemas belong outside each record. Reject oversized records; do not edit descriptions to fit during inference.

Adopt one documented compact JSON encoder: UTF-8, no optional whitespace, no normalization, field order fixed, minimal required escaping, `ensure_ascii=False`, no nonstandard numbers/NaN. Pin treatment of control escapes and slash escaping with byte fixtures. Serialize request keys history/user, history entries role/content, classifier keys request/capabilities. Validate before encoding. Budget the whole serialized request, not just the user string, and the whole serialized capability record. JSON wrappers/escaping consume ordinary tokens.

Drop oldest complete history entries until `{"history":[],"user":"…"}` plus retained history fits 256 ordinary tokens. Never truncate the current user or split a history entry. Reject empty/Unicode-whitespace-only user requests and over-budget user-alone serialized requests. Preserve the exact retained request bytes throughout retrieval, pairs, and classification. Disclose dropped-history count in separate metadata.

Formatter templates must match Proposal B exactly. Generic pairs have five structural tokens and at most 1,019 content tokens, at most 256 query tokens, with remaining capacity assigned to the document. Keep generic truncation distinct from registered routing, where both records have already passed strict validation and cannot be truncated. Decision content has at most 1,021 ordinary tokens; if oversized, remove the second capability, reserialize/re-tokenize, and then return INPUT_TOO_LONG if still oversized. Position IDs include structural tokens and never reset at a pair boundary. Right pad only.

Build immutable snapshot objects and exact FP32 normalized-dot-product search. Include capability ID, description hash, tokenizer identity, and encoder-variant artifact identity in cache keys. Validate index completeness, dimensions, finite values, and compatibility. Different descriptions invalidate their own embeddings; new encoder/tokenizer/quantized variant rebuilds the entire index. Test snapshot replacement during a request without mixing versions. Cache/index data never adds learned model parameters.

Implement all branch behavior: empty registry validates then returns NO_CALL with no neural passes; singleton skips reranking; N≥2 skips only when both frozen inequalities meet or exceed thresholds; otherwise every member of min(8,N) is cross-encoded. Sort exact-score ties by ascending ASCII capability ID. Classifier consumes up to top two final records, returns argmax of FP32 temperature-scaled probabilities with lowest-index ties, and can only accept/reject the top candidate. Positive T does not change mathematical argmax and cannot repair incorrect routing. No generic confidence-to-CLARIFY override. Do not embed external execution/permission policy into Encoder.

Keep decision JSON exactly as specified, with metadata separate. Validate any returned capability ID against the same snapshot. Implement all five frozen errors: INVALID_INPUT, INPUT_TOO_LONG, INVALID_REGISTRY, INDEX_MISMATCH, NUMERICAL_ERROR. Document a deterministic error precedence for multiply invalid inputs without adding success fallbacks or new public codes. Dependency/environment failures are operational failures, never fabricated routing decisions. Compatibility checks must cover even empty-registry requests where applicable without invoking the encoder.

```bash
python -m pytest -q tests/tokenizer tests/routing tests/integration
python -m juniper_encoder routing conformance --fixtures tests/fixtures/routing
```

**Gate:** boundary and adversarial contract fixtures pass, including exactly-at/one-over budgets, both threshold equalities, all ranking/class ties, classifier overflow, registry mutations, invalid Unicode, and injected numerical faults.

## 7. Curated data acquisition, labeling, deduplication, and splits

Before downloading substantial data, produce `manifests/sources/approved.json` containing each exact source/revision/URL, upstream file checksums, acquisition script, retrieval date, rights/license evidence, attribution, permitted intended use, redistribution status, language/domain, estimated size, and exclusions. No source is approved solely because it is public, mirrored, or on Hugging Face. Verify source-level obligations; do not label the mixed corpus under the code license. Unknown or conflicting rights mean quarantine, not training. Do not use private user conversations, secrets, paid APIs, or synthetic-provider output without applicable authorization and usage rights.

Start with a modest, auditable mixture of natural text and task-relevant technical/dialogue material, plus first-party capability records and labeled routing examples. External retrieval/reranking datasets are optional auxiliaries only if their labels, domains, leakage exposure, and rights are verified. Luna must record actual selected sources; this brief grants no blanket dataset license. Store download/extraction limits and prevent archive path traversal/decompression exhaustion.

Cleaning is deterministic and separately versioned: strict decoding; source/record provenance; removal or quarantine of corrupted, irrelevant, low-quality, unsafe-to-store private data and boilerplate under explicit rules; document segmentation; language identification if used; exact dedupe and near-duplicate clustering. Preserve original hashes and reason codes. Do not conflate data-cleaning policy with the frozen tokenizer's prohibition on normalization. Any matching-only normalization is recorded and must not rewrite inference/tokenizer behavior.

Partition duplicate/derivative families before splitting. Group by original document, conversation, template/generation family, source lineage, and capability family. Use exact byte/content hashes plus versioned near-duplicate methods and thresholds; audit sampled false matches and missed matches. Paraphrases and synthetic derivatives inherit the source's split. Mining negatives or deriving new prompts may use training only.

Create disjoint `train`, `development`, `calibration`, and sealed `test` partitions with immutable IDs and hashes. Conservative starting family-level proportions are 80/10/5/5, adjusted before training only for adequate class/slice support; publish the final assignment and counts. Keep separate general-text development/test partitions too. Calibration has disjoint threshold-selection and temperature-fitting portions; quantization calibration uses a separately identified calibration subset, never test. Record parent split ownership to prevent accidental reuse.

Reserve entire capability identities and related descriptions/example families for calibration/development/test novelty tracks. Held-out test capability identities and their labeled request families must never be consumed by tokenizer training, MLM, supervised training, development selection, calibration, hard-negative mining, or synthetic generation. They may be indexed at evaluation time, because dynamic registration is the behavior under test. Keep raw reserved data outside training globs and check access manifests. Public pretrained benchmark contamination cannot be proven absent just by renaming IDs; record searches/exclusion evidence and remaining limits.

Routing labels must distinguish:

- CALL: top candidate appropriate and sufficient user-provided information; valid explicit preference is sufficient for selection as Proposal B states.
- NO_CALL: direct conversation, acknowledgments, unsupported/unavailable capabilities, or no appropriate current recipient.
- CLARIFY: a specific missing/ambiguous user detail prevents selection or establishing required inputs; annotate the missing detail. Never use CLARIFY as a catch-all for uncertain annotators or weak predictions.

Record acceptable capability sets/relevance grades and a rationale, not just a single opaque ID. Define treatment of genuinely equivalent capabilities before evaluation. For exact routing, predicted IDs must satisfy the frozen end-to-end correctness rule and the prespecified gold policy. Ambiguous labels are adjudicated or quarantined. Include negative cases with strong lexical overlap, requested unavailable tools, stale context, conflicting intent, exclusions, missing inputs, explicit preferences, and multi-step requests that may validly route to an orchestration agent. Split tool/agent/skill coverage deliberately.

Have an independent annotator/reviewer audit a stratified sample and all disputed label policies; preserve disagreement rates and decisions. Synthetic labels require independent quality audit and are never sufficient alone to establish real-world quality. Freeze evaluation metric definitions and critical slices before candidate selection. Commit an evaluation hash without exposing sealed examples to training code.

```bash
python -m juniper_encoder data acquire --manifest manifests/sources/approved.json
python -m juniper_encoder data prepare --config configs/data.yaml
python -m juniper_encoder data split --config configs/splits.yaml
python -m juniper_encoder data audit --config configs/data.yaml --fail-on-leakage
python -m juniper_encoder data freeze --output manifests/data/frozen.json
```

**Gate:** retrievable immutable source identities, reviewed rights, deterministic rebuilt processed hashes, no known cross-split leakage, reviewed label rules, adequate counts per declared slice. Missing evidence blocks scale. Correcting leakage invalidates affected training and evaluation lineage; preserve the contaminated results as invalidated evidence.

## 8. Training protocol and evidence-before-scale

Write the resolved protocol before each run. Initial conservative choices below are training choices, not claims of optimality or changes to Proposal B. Every change gets a new config hash/run ID and a stated development-only reason.

Use seed 1729 for the initial full run and 1730/1731 for affordable pilot variability checks. Do not claim multi-seed full-training stability if only one full seed ran. Initialize the complete deployed model once with the frozen rules. No pretrained weights, hidden teacher checkpoint, architecture search, adapters, extra deployed heads, or distillation dependency.

Initial optimizer: AdamW, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01 on embedding/linear weights, no decay on LayerNorm parameters or biases. Keep FP32 optimizer states and master weights under AMP. Enforce zero PAD row and moments explicitly. Clip global gradient norm to 1.0 after unscaling and before the optimizer step. Scheduler advances on successful optimizer updates only; count attempted/skipped updates and consumed tokens separately. Initial MLM peak LR 1e-4; downstream backbone LR 2e-5 and task-head LR 1e-4. Use linear warmup for the first 6% of the predetermined successful-update budget, then linear decay to zero. Document optimizer reinitialization when changing objectives; keep each stage's predecessor checkpoint immutable.

Initial effective MLM update target: 8,192 non-PAD tokens, assembled into deterministic batches with actual token counts recorded. Start microbatch 1 and use gradient accumulation; normalize summed loss by the actual supervised-token count across the effective update, not an average of uneven microbatch means. For routing losses normalize by actual query groups/examples. Supervised effective batch target is 32 query groups; actual negatives per group are separately pinned. Gradient accumulation does not automatically provide in-batch contrastive negatives: use explicit per-query negatives initially.

Temporary objective: masked language modeling over ordinary content tokens only, no next-sentence head. Select 15% of eligible positions with a deterministic per-example/epoch mask rule; selected positions are 80% MASK, 10% a uniformly sampled ordinary token ID, 10% unchanged. Guarantee at least one target for a nonempty training sequence using a documented rounding rule. Never mask structural/PAD tokens or predict them as targets. A conventional MLM objective is supported by [the BERT paper](https://arxiv.org/abs/1810.04805); only the objective convention is borrowed, not BERT's architecture or tokenizer.

Use a minimal temporary bias-free tied decoder `H_masked @ E.T`; it adds zero unique trainable parameters when tied to the input embedding and introduces no extra transform/LayerNorm. Compute cross-entropy over ordinary-token logits only (IDs 7–16,383); targets map consistently. Materialize logits only for selected positions to control memory. All three deployed task heads remain present but unused/untrained during this stage. Record physical state-dict tensors versus unique aliases. The temporary decoder module/alias must be removed on export; the deployed model has no tying. If another temporary head is justified, count every additional parameter separately and obtain protocol review before scale.

Use valid template structures and unmasked structural tokens during pretraining, with training-only request-like and document-like content. Foundation chunking is a training preprocessing operation with lineage; it must not weaken inference's no-truncation rule. Do not pack unrelated documents with a block-diagonal attention mask that contradicts full attention. Initially use one document/content example per sequence with right padding and deterministic chunk boundaries.

Run this progression:

1. **Mechanical smoke:** forward/backward, serialization, PAD row, finite optimizer states, exact counts, and deliberately injected failures.
2. **Tiny overfit:** fixed small lawful training fixtures with fixed masks; demonstrate substantial MLM loss reduction and memorization of a small unambiguous routing set. Record initial/final losses and predictions. This is a debugging gate, never a quality claim.
3. **Length/memory pilot:** 128, 512, and 1,024 native lengths; microbatch and accumulation measurements; no context-extension redesign. Qualify activation checkpointing with preserved RNG and prove reference/dropout/resume equivalence.
4. **Bounded learning pilot:** initially cap at 2 million consumed ordinary tokens with checkpoints and development evaluation. Run a longer pilot only if the shorter one is too noisy and resources permit; record that reason.
5. **Foundation stage:** first planned budget 25 million consumed ordinary tokens after the pilot gate; select a length curriculum using measured padding efficiency, including explicit 1,024-token training exposure. A conservative initial mix is 50% at max length 128, 35% at 512, 15% at 1,024 by consumed content tokens. Freeze the exact schedule, token-to-update handling, and total successful-update schedule before starting. Larger budgets require a new evidence-backed training plan, not blind continuation.

A token quota is an experiment boundary, not “trained enough.” Continue additional foundation stages only when development MLM and downstream retrieval probes justify compute and the recorded resource ceiling permits it. Preserve all pilot checkpoints/metrics needed to explain decisions. Do not use test results to justify more training.

```bash
python -m juniper_encoder train smoke --config configs/mlm.yaml
python -m juniper_encoder train overfit --config configs/mlm.yaml
python -m juniper_encoder train profile --config configs/mlm.yaml --lengths 128 512 1024
python -m juniper_encoder train resume-test --config configs/mlm.yaml
python -m juniper_encoder train run --stage mlm --config configs/locks/mlm-pilot.yaml
python -m juniper_encoder train gate --run manifests/checkpoints/mlm-pilot.json
python -m juniper_encoder train run --stage mlm --config configs/locks/mlm-foundation.yaml
```

**Gate:** correct learning signal; no unexplained numerical faults; independent validation masks; measured resource fit; exact-resume test passed; finite projected runtime/disk budget; development evidence supports scale. A plateau or insufficient capability is a result to report, not a reason to rewrite the architecture.

## 9. Checkpoints, interruption, and recovery

A resumable checkpoint contains model weights, temporary-head specification, optimizer tensors/parameter mapping, scheduler, GradScaler, global successful/attempted/skipped updates, consumed tokens/examples, epoch and exact sampler cursor/permutation identity, batch/accumulation state, masking state, Python/NumPy/CPU/CUDA RNG states, precision/kernel settings, source SHA/dirty status, environment lock, config hashes, tokenizer/data manifests, and predecessor checkpoint identity. Do not serialize executable arbitrary objects for normal deployment; keep tensor payloads in a safe tensor format and structured metadata in validated JSON.

Prefer checkpoints at optimizer-update boundaries, at least every 100 successful updates or 10 minutes, whichever occurs first, with signal handling that finishes a safe boundary. If checkpointing mid-accumulation, save partial gradients and all normalization/count state; otherwise explicitly discard the incomplete update and resume from the last committed boundary, replaying the same data. SIGKILL recovery uses the last complete checkpoint. Save atomically to a temporary path, flush/fsync as appropriate, verify bytes/tensors, then rename and write a content-addressed manifest. Never overwrite the sole known-good checkpoint.

Keep at least the latest two valid recovery checkpoints, all selected/best/milestone checkpoints, and failure-relevant evidence. Bounded retention may delete only declared reproducible intermediate checkpoints after durable preservation requirements are met; never erase failed runs, negative results, or the only recoverable lineage to save space.

Resume equivalence test: on the same hardware/locked software, compare uninterrupted K+M updates against K updates, save, terminate process, reconstruct in a fresh process, resume M. Compare exact sample order, masks, losses, LR/scaler state, update counters, optimizer state, and tensor hashes at the same boundaries. Require bitwise identity in the designated deterministic path. Test activation-checkpointing dropout RNG, accumulation, epoch boundary, and an interruption before an update. Separately define dtype-aware numerical tolerances for cross-device/backend inference; do not silently relabel a failed same-environment resume as “close enough.”

Recovery rules:

- OOM: preserve failure context; restart from the last valid checkpoint; reduce microbatch and adjust accumulation while preserving effective objective/counts; prove equivalence or label a changed execution protocol as a new run branch. Never shrink model/vocabulary/native context.
- Nonfinite gradient: record it; GradScaler may skip a bounded transient overflow. Halt after 3 consecutive skipped updates or >1% skipped updates in a 200-attempt window, initial engineering policy. Nonfinite parameters, evaluation outputs, or optimizer state halt immediately. Investigate AMP/casts/data, test FP32, and fork a versioned corrected run.
- Disk/checkpoint corruption: reject partial/bad-hash checkpoint; restore the previous verified one; never silently load partial tensors.
- Lost source/environment/data identity: no exact-resume claim; reconstruct the pinned identities or mark blocked.
- Plateau: compare development baselines and error slices, stop the exhausted experiment, document next training-only hypothesis. No endless automatic restarts.

## 10. Retrieval, reranker, and decision training

All tasks share one backbone. Every backbone change invalidates cached embeddings and may degrade previously trained heads; re-evaluate all components after each stage. Never evaluate a new backbone using an old index.

**Retrieval.** Use the frozen masked-mean → bias-free 512→256 → L2 head and fixed contrastive temperature 0.05. Train query-to-capability relevance using labeled positives and explicit negatives. Begin with up to 7 deterministic negatives per query, mixing random valid negatives and reviewed lexical confounders. For multiple acceptable capabilities, use a documented multi-positive loss or exclude other positives from negatives. NO_CALL/CLARIFY examples without an appropriate positive cannot be assigned an invented positive. Add mined hard negatives only from training snapshots with the checkpoint/index hashes recorded. Compute similarities/loss stably in FP32. Gradient accumulation alone does not increase the negative pool. Compare against deterministic lexical and random/initial-weight baselines on development data.

**Reranker.** Use the exact pair template and scalar head with bias; return raw FP32 scores in inference. Train a listwise softmax relevance loss over a per-query candidate group, or a prespecified pairwise logistic ranking loss when only pairwise labels exist. Record which one is used; initial default is listwise relevance cross-entropy over up to eight training candidates with valid positive mass. Training loss normalization does not turn inference scores into probabilities. Include overlapping capabilities, exclusions, explicit preference, and missing-input distinctions. Keep retrieval-top-8 training lists separate from diagnostic top-10 evaluation pools. Preserve training-only mined candidate records and score provenance.

**Decision classifier.** Generate classifier examples using actual final-ranked top candidates from train-only pipelines, covering both retrieval order and reranker order. Do not train exclusively on gold top candidates. Annotate the correct class conditional on the presented top record: a retrieval mistake is not automatically CLARIFY. Use the frozen three-class indices and ordinary cross-entropy initially, no head dropout or unapproved confidence threshold. Include genuine natural class proportions; if sampling is balanced, record weights and separately retain representative development/calibration distributions. Unused heads should have no optimizer updates.

**Joint stabilization.** After individual stages, run a bounded mixed-task stage if needed to address shared-backbone forgetting, with a fixed cycle of retrieval/reranker/classifier batches and declared loss normalization. Initial cycle 1:1:1 and backbone LR 1e-5; head LR 5e-5. Evaluate all components at every checkpoint. Select using a predeclared development-only rule: first satisfy false-call and component gates, then maximize end-to-end exact accuracy, then use smaller compute/earlier checkpoint as the deterministic tie-break. Never select the checkpoint with best test score. Freeze the final backbone and all heads before final index rebuild and calibration.

```bash
python -m juniper_encoder train run --stage retrieval --config configs/locks/retrieval.yaml
python -m juniper_encoder train mine-negatives --config configs/locks/retrieval.yaml --split train
python -m juniper_encoder train run --stage reranker --config configs/locks/reranker.yaml
python -m juniper_encoder train build-decisions --config configs/classifier.yaml --split train
python -m juniper_encoder train run --stage classifier --config configs/locks/classifier.yaml
python -m juniper_encoder evaluate --split development --config configs/evaluation.yaml
python -m juniper_encoder train run --stage joint --config configs/locks/joint.yaml
python -m juniper_encoder train select --policy configs/quality-gates.yaml --split development
```

Do not run joint training mechanically if evidence says it degrades the model; record a skipped/not-needed stage with results. Every stage needs max updates, eval cadence, LR schedule, actual corpus/negative IDs, and stopping rule. Initial downstream cap is 3 passes over its frozen training examples with evaluation every 250 successful updates and at stage end; extend only on development evidence under a new protocol. Report confidence intervals and limited sample support. Weak training data means `QUALITY_NOT_ESTABLISHED`, not “routing ready.”

## 11. Calibration: T, τ_score, τ_margin

Only calibration data fits deployment calibration metadata. Before fitting, T=1 and metadata explicitly says uncalibrated. Do not deploy arbitrary guessed thresholds as calibrated values. Use diagnostic always-rerank execution while thresholds are unavailable, clearly labeled as a reference mode rather than a qualified conditional candidate.

Freeze the selected checkpoint, tokenizer, formats, registry calibration snapshots, metric definitions, critical slices, and a deterministic threshold-search protocol. Use the threshold-selection calibration subset to evaluate a bounded grid of score/margin threshold pairs derived from its observed scores, including no-skip and maximal-skip endpoints represented by finite stored thresholds. For every pair run the entire pipeline, including classifier candidate context. N=0 and N=1 retain their frozen special cases. Maximize rerank skipping only subject to the prespecified calibration quality/noninferiority constraints; break ties conservatively toward more reranking, then deterministic threshold ordering. Never choose thresholds solely from retrieval margin without end-to-end and false-CALL evidence.

Fit positive scalar T on the separate temperature calibration subset by minimizing decision NLL using a bounded scalar optimizer in log(T). Initial search interval T∈[0.05,20] is an optimization policy, not a Proposal B constraint; boundary optima trigger diagnosis and a documented calibration-only extension. Use conditional-pipeline logits with frozen thresholds. Record pre/post NLL, Brier score, 15-bin equal-width top-label ECE, reliability-bin counts, fit convergence, and exact FP32 serialization. Report classwise calibration additionally where supported. T cannot improve argmax F1 and must not be advertised as doing so.

Store exact τ_score, τ_margin, T, checkpoint/tokenizer hashes, calibration-subset/snapshot hashes, fitting code SHA, objective, search space, and selected-result evidence in immutable metadata. Validate finite thresholds and T>0. No automatic production adaptation. After any weight/tokenizer change, recalibrate a new candidate. INT8 comparison uses identical T and thresholds; do not recalibrate INT8 separately to hide drift.

```bash
python -m juniper_encoder calibrate thresholds --config configs/calibration.yaml --split calibration-thresholds
python -m juniper_encoder calibrate temperature --config configs/calibration.yaml --split calibration-temperature
python -m juniper_encoder calibrate verify --manifest manifests/checkpoints/selected.json
```

**Gate:** no calibration/test leakage, optimizer converged, metadata hashes pinned, representative calibration support disclosed, conditional policy satisfies its predeclared constraints. If no useful skip policy passes, a conservatively calibrated near-always-rerank policy is legitimate; claim no unmeasured acceleration.

## 12. Evaluation definitions, coverage, and quality gates

Write `configs/evaluation.yaml` and `configs/quality-gates.yaml` before sealed evaluation. Proposal B gives exact INT8 tolerances but no absolute minimum baseline accuracy or conditional noninferiority margin. Do not invent a frozen minimum. Establish additional project qualification thresholds using training/development evidence and pre-test Sol protocol review. Required numeric fields include minimum Recall@8, macro-F1, CALL recall, end-to-end accuracy; maximum false-CALL and ECE; conditional allowed accuracy/selection losses and false-CALL increase; and per-slice minimum support. A missing numeric policy blocks `QUALITY_QUALIFIED`. Reviewable software/experimental candidates may still be handed off with `QUALITY_NOT_ESTABLISHED / NOT RELEASED`.

Every final candidate is evaluated on sealed, versioned held-out records and immutable registry snapshots, both componentwise and end to end. Report raw predictions and gold labels/IDs (where redistribution is lawful), numerators, denominators, exclusions, sample counts, confidence intervals, and confusion matrices. Bootstrap intervals should resample independent request/capability families, not pretend paraphrases are independent. Use paired comparisons on the same requests/snapshots for variants. Report insufficient-support slices as inconclusive, never pass them by omission.

Define the metrics explicitly:

- Shortlist Recall@8: mean relevant-set recall in the first min(8,N) exact-search results over queries with at least one gold relevant capability; also report any-hit@8 when multiple positives exist.
- Retrieval Recall@10 and nDCG@10: separate top-min(10,N) retrieval evaluation over the full snapshot. This diagnostic never changes production top-8.
- Reranker nDCG@10 and MRR@10: score a fixed, prespecified evaluation pool of up to ten candidates per request, e.g. frozen reference retrieval top-10, shared identically across floating and INT8 variants. State candidate recall and relevance grades; if no relevant candidate is present, score zero when the gold query has relevance, rather than dropping it. Supplement with production top-8 reranking metrics. Never label eight candidates as an actual ten-candidate experiment.
- nDCG uses recorded graded judgments, gain 2^rel−1, discount log2(rank+1), and explicit ideal-pool policy. MRR uses the first relevant rank. For no-relevance queries report these ranking metrics as not applicable, with counts; evaluate their routing decisions normally.
- Decision macro-F1 over the frozen three classes; per-class precision/recall and support.
- CALL recall: correct CALL class predictions / gold CALL requests; additionally report ID-correct CALL recall to expose selection errors.
- False CALL rate: predicted CALL among gold non-CALL requests / gold non-CALL requests. Also report wrong-ID CALL frequency and total erroneous calls per request, because the first definition alone misses misrouting.
- Calibration error: 15-bin equal-width top-label ECE on [0,1], with frozen bin-edge handling; publish bin counts and supplement with NLL/Brier.
- End-to-end exact routing accuracy: correct decision and correct gold-acceptable capability ID for CALL, and correct decision/null ID for NO_CALL or CLARIFY. Contract errors have their own expected-error suite; unexpected errors on valid routing inputs are failures and count against accuracy, never vanish from the denominator.
- Rerank invocation rate, fast-path exact routing accuracy, and path-specific class/slice metrics. Fast path means neural routing that skips reranking; report singleton and N≥2 separately. Empty-registry deterministic returns are separate from neural fast-path statistics.

Prespecify critical slices: agent/tool/skill; CALL/NO_CALL/CLARIFY; held-out capability identities/families; similar/overlapping descriptions; required-input missingness; contextual ambiguity; unsupported capabilities; adversarial requests/descriptions. Add all frozen coverage: additions, removals, description changes, empty/singleton registries, context reduction, exact limits/input errors, numerical failures, fast and reranked paths. Include valid explicit preferences; forged IDs; literal structural strings; instructions embedded inside descriptions; Unicode edge cases; snapshot/index mismatch; and history that tries to override the current request.

Adaptability evaluation adds entirely unseen identities/descriptions to a post-training snapshot, embeds them without training, and measures behavior. Removal/replacement tests must prove stale cache entries cannot leak. Do not train on the “new” capability first and call that dynamic adaptation.

```bash
python -m juniper_encoder evaluate --split test --config configs/evaluation.yaml --sealed
python -m juniper_encoder evaluate adversarial --config configs/evaluation.yaml
python -m juniper_encoder evaluate compare-paths --config configs/evaluation.yaml --split test
python -m juniper_encoder evaluate gates --policy configs/quality-gates.yaml
```

Always-rerank reference cross-encodes the complete production shortlist for N≥2, keeping empty/singleton special cases; any diagnostic singleton pass is separately labeled. Compare it with conditional reranking using identical weights, inputs, snapshots, formats, and T. Report skip percentage, paired end-to-end accuracy difference, false-CALL difference, capability-selection accuracy difference with fixed gold-CALL denominator, median encoder-pass reduction, and p50/p95 latency reduction. Record absolute values and signed deltas, not just percentages. Report whether constraints pass globally and by critical slice.

If final test fails, preserve the result and mark REQUEST_CHANGES. Training or thresholds revised in response require a new candidate and an untouched confirmation set; previously inspected tests become known evaluation/development evidence, not a fresh seal. Do not run repeated secret test-guided selection.

## 13. Export, runtime correctness, and INT8 qualification

Export the selected shared backbone and exactly three deployment heads. Remove the temporary MLM module and any tied decoder aliases. Verify deployed key list, tensor shapes, unique parameter count, PAD zeros, initialization/training lineage, RoPE constants, dtypes, templates, tokenizer compatibility, and frozen calibration metadata. Save weights in a safe non-executable format, with validated config and tokenizer payloads. No arbitrary code execution is required to load an untrusted registry or model metadata.

First qualify a BF16/FP16 reference variant on each declared target/backend. Use FP32 oracle comparisons for diagnostics, with predeclared per-output tolerances and routing identity tests on non-boundary fixtures. Precision differences near ties must be measured in full evaluation. Export does not permit approximate GELU or silently altered LayerNorm/softmax precision. Verify exported graph operators and actual provider dispatch. No unmeasured CPU/GPU fallback may be called accelerated execution.

Initial mandatory device scope: FLOWBOX CPU and CUDA GPU reference inference with the corresponding supported backend; target-specific INT8 attempts where a compatible backend exists. Any additional promised device, OS, mobile target, or backend becomes a required benchmark target before claiming support. Do not infer Android readiness from FLOWBOX. If a backend cannot satisfy reference precision/ops or a required case, report unsupported/failed honestly and block that target's qualification.

Try a conservative post-training INT8 variant first, using a verified supported runtime. ONNX Runtime is a candidate backend, not a guaranteed compatible solution; test exact graph semantics and provider support. Dynamic quantization and static quantization have different activation and calibration behavior. [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

Quantize eligible backbone operators only initially. Keep LayerNorm, softmax, mean pooling, L2 normalization, retrieval head, decision head, and reranking head floating point. Preserve required FP32 accumulations/outputs. Record actual per-operator weight and activation precisions, axes, scales/zero points, exclusions, graph optimizations, runtime/version, provider, and hardware. Decide explicitly whether embedding weights are quantized; measure the effect. Merely storing INT8 weights while executing floating GEMMs is not evidence of INT8 acceleration.

Static activation-range fitting uses the identified quantization calibration subset only. Training-aware quantization is a new training experiment/checkpoint and must return through full checkpoint selection/calibration/reference comparison; do not quietly apply it to rescue a failed post-training artifact. For the frozen acceptance comparison, use the same source checkpoint, request data, snapshots, tokenizer, formats, and identical T/thresholds. Build each variant's registry embeddings using that variant's encoder. Also compare the reranker on the shared diagnostic pool so candidate-pool changes cannot hide head degradation.

Apply all quality tolerances to global and prespecified critical slices, including held-out capabilities:

| Metric | INT8 gate relative to matching floating reference |
|---|---|
| Shortlist Recall@8 | loss ≤0.01 absolute |
| Retrieval Recall@10 | loss ≤0.01 |
| Retrieval nDCG@10 | loss ≤0.01 |
| Reranker nDCG@10 | loss ≤0.01 |
| Reranker MRR@10 | loss ≤0.01 |
| Decision macro-F1 | loss ≤0.01 |
| CALL recall | loss ≤0.01 |
| End-to-end exact routing accuracy | loss ≤0.01 |
| False CALL rate | increase ≤0.005 |
| Calibration ECE | increase ≤0.01 |
| Numerical validity | zero failures |
| Warm p50 latency | ≤1.05× reference |
| Warm p95 latency | ≤1.05× reference |
| Model artifact size | ≤0.65× reference |
| Peak inference memory | ≤reference |

All quality metrics are on [0,1]. Do not round before gate comparison. Publish measured deltas, sample support, and uncertainty; a point estimate within tolerance is not permission to conceal an inconclusive critical slice. Use matched thread/device/runtime conditions. Artifact-size comparison counts the complete usable deployment payload, including quantization metadata and required weight/graph files under the same inclusion policy; also report raw tensor bytes. Do not inflate the reference with training checkpoints or duplicate graphs to pass the ratio. Registry/index memory and size are reported separately and included consistently in full-process memory measurements.

```bash
python -m juniper_encoder export --checkpoint manifests/checkpoints/selected.json --variant reference
python -m juniper_encoder export verify --variant reference
python -m juniper_encoder quantize --config configs/quantization.yaml
python -m juniper_encoder index rebuild --variant reference --config configs/evaluation.yaml
python -m juniper_encoder index rebuild --variant int8 --config configs/evaluation.yaml
python -m juniper_encoder evaluate quantization --config configs/evaluation.yaml
python -m juniper_encoder quantize gates --config configs/quantization.yaml
```

**Gate:** every applicable frozen gate passes for the specific target/backend. Otherwise preserve `INT8_REJECTED` evidence and retain qualified floating deployment on that device. Do not weaken a gate or call the entire model INT8-qualified because one target passed. A floating candidate may remain eligible if its own complete gates pass, with INT8 failure plainly disclosed.

## 14. Complete benchmark matrix

For every declared target device/backend and qualified reference/INT8 variant, benchmark retrieval encoding, capability encoding, reranker, decision classifier, fast full routing, reranked full routing, and mixed conditional full routing. Include the always-rerank comparison and separately record empty/singleton behavior.

The required cross-product is batch size 1 and 8 × padded sequence length 128, 512, and 1,024. Each case uses at least 20 warmup runs and at least 200 measured runs. Distinguish per-batch latency, per-request latency/throughput, and request counts. Fix thread/device configuration, process isolation, clock/power conditions where controllable, seeds, inputs, and synchronization. Report actual conditions, background contention, and cache state. Synchronize GPU measurements at the correct boundaries; do not measure only asynchronous enqueue time.

Registered routing requests/documents remain within their 256-ordinary-token limits. Exercise larger padded lengths with valid content plus right padding or generic component fixtures where appropriate; do not violate routing budgets to fabricate a 1,024-content-token production input. Classifier/pair templates have their own real lengths. Publish actual unpadded lengths for every encoder call, and a padding policy that fits all stages. For 128-length pipeline fixtures, use sufficiently short valid records so each stage fits. Production natural-padding benchmarks are additional, not substitutes for the fixed matrix.

Freeze meaningful snapshot sizes, initially N=0,1,8,100,1000 where lawful fixtures permit. Record real/constructed data provenance and distinguish synthetic performance registries from quality evaluation registries. Record shortlist size, physical model forward invocations, encoded-sequence equivalents, rerank invocation rate, batch packing, and registry-search implementation. With N≥1, a fast route normally uses one query encoder pass plus one classifier pass; reranking adds up to eight encoded pairs, possibly one batched invocation. Reusing query hidden states for classifier context is invalid because classifier input differs. Offline index generation is excluded from route latency.

Full-pipeline timing includes tokenization, formatting/validation under the documented boundary, exact search, conditional gate, reranking, classifier, and postprocessing. Exclude model loading and offline indexing as required. Publish cold-start/loading/index-build costs separately if measured. Measure p50/p95 end to end directly; do not add component percentiles. Record peak process RSS and relevant device allocated/reserved memory with baseline/incremental definitions, plus absolute peak inference footprint. Report caches including precomputed registry vectors and whether request caches are disabled; never use repeated-response caching to create false speedups.

Run in fresh benchmark processes with warm resident models. Preserve raw per-run latency samples and summarize with fixed quantile definitions. Compare paired variants with stable interleaved or randomized run order where thermal drift is material. An OOM or unsupported case remains a failed/unsupported matrix entry; do not silently lower batch/length, omit it, or mark it passed. Batch 8 may be challenging on 6 GB; measured failure restricts qualification rather than authorizing architecture changes.

```bash
python -m juniper_encoder benchmark --config configs/benchmarks.yaml --variant reference
python -m juniper_encoder benchmark --config configs/benchmarks.yaml --variant int8
python -m juniper_encoder benchmark compare --config configs/benchmarks.yaml
python -m juniper_encoder benchmark coverage --contract spec/proposal_b.contract.json
```

**Gate:** matrix complete for every claimed target/backend; at least the frozen warmup/measurement counts; raw samples present; no hidden cases or invalid comparisons. Apply INT8 performance gates per matched required case, not a convenient average that hides regressions.

## 15. CI, immutable packaging, and exact candidate identity

CI must validate the exact source commit under review. Pin GitHub Actions by commit SHA, dependencies by hashes, and requested source checkout by a full commit SHA. Do not confuse GitHub's synthetic PR merge SHA with the candidate head SHA. Record both when applicable. Workflow checks must output the checked-out SHA and fail if it differs from the requested candidate. Never execute untrusted PR code on a privileged FLOWBOX runner or expose credentials through pull-request-target execution.

Fast CPU CI: syntax/lint/type checks where applicable; schema validation; frozen-spec hash/traceability; computed parameter/state-key counts; tokenizer conformance/deterministic rebuild on a lawful fixture; routing/errors/ties/budgets; data-manifest integrity/leakage fixture tests; safe artifact loader; export smoke; miniature reproducible train/resume; manifest/hash verification. CPU CI does not establish real GPU/INT8 quality or performance. Required hardware jobs/evidence validate actual target precision, full benchmarks, and checkpoint resume. Missing required hardware evidence is blocked, never a skipped pass.

Full candidate validation materializes immutable weights/tokenizer/data/evaluation payloads from their manifests, verifies hashes before use, and runs the declared evaluation/benchmark commands under the exact source SHA and environment. Large training jobs need not run on every PR; deterministic short-run reproduction is required in CI, full training reconstruction remains runnable and documented, and reviewers independently replay meaningful learning/resume segments and selected evaluations. Do not claim a full from-scratch training reproduction when only a smoke run was replayed.

Define an immutable candidate descriptor outside the source tree it identifies, avoiding a self-referential commit/hash loop. It contains:

- Candidate source SHA and clean worktree assertion; training source SHA(s) and lineage; review/package code SHA if different and justified.
- Frozen spec and machine-contract hashes; all resolved configs and environment lock identities.
- Source/data/split/tokenizer manifests and hashes; exact checkpoints and safe tensor hashes.
- Reference and each quantized export hash, byte size, graph/operator manifest, and target scope.
- Calibration metadata and subset hashes; registry snapshot/index hashes for each encoder variant.
- Evaluation/benchmark scripts, raw results, gate output, failure ledger, exclusions, and unsupported targets.
- Model card, licenses/notices, reproducibility commands, CI run IDs/URLs and exact checked SHAs.

The descriptor's content hash is the candidate ID. Store the descriptor and evidence in a durable append-only artifact namespace; reference that ID in the PR and later evidence records. If an evidence-only Git commit is used, identify it separately from the tested source SHA. Never claim that adding an evidence file retroactively tested a later commit. Changes to executable files/configs always require a fresh exact-commit run.

Generate `SHA256SUMS` for payload files with deterministic path ordering; the checksum file does not hash itself. Hash/sign or attest the manifest through a separate envelope if supported. Verify bytes after upload/download from the durable store. Do not publish expiring signed URLs, secrets, or private data in GitHub; store stable identifiers with authorized retrieval instructions.

The model card must state frozen purpose/scope, 49,441,796 deployed/49,308,672 backbone parameters, all mode counts, random-init training history and actual tokens/steps, corpus sources/rights, evaluated language scope, decision semantics, native context/input budgets, calibration values, metrics with slices/support, conditional-vs-always comparison, target-specific precision/INT8 results, failure/negative results, limitations, unsupported targets, exact artifact IDs, license distinctions, and `NOT RELEASED`. Training code that works is not proof the model is useful.

```bash
python -m juniper_encoder package --checkpoint manifests/checkpoints/selected.json --output artifacts/candidate
python -m juniper_encoder package verify --directory artifacts/candidate
python -m pytest -q
bash scripts/verify-candidate.sh --candidate artifacts/candidate/candidate.json

git status --porcelain=v1
git add spec schemas configs src tests scripts requirements pyproject.toml .github manifests reports reviews docs README.md CHANGELOG.md
git diff --cached --check
git diff --cached --stat
git commit -m 'Build frozen Juniper Encoder candidate with reproducible evidence'
git push -u origin work/luna-encoder-proposal-b
```

Before staging, scan staged content for secrets, unintended data, and oversized artifacts; stage only existing intended paths and adjust the exact list to the repository. Candidate packaging done before this commit is provisional: capture the new SHA, run final clean-clone validation on it, and generate the final immutable descriptor referencing that SHA. Never present provisional precommit checks as exact-candidate CI.

Implement the candidate workflow so these commands are real and documented:

```bash
JE_SHA="$(git rev-parse HEAD)"
gh workflow run candidate.yml --ref work/luna-encoder-proposal-b -f candidate_sha="$JE_SHA"
gh run list --workflow candidate.yml --commit "$JE_SHA"
```

Resolve the returned run ID and inspect jobs/logs/conclusions, not just the existence of a run. Create a draft PR using a prepared file:

```bash
gh pr create --draft --base main --head work/luna-encoder-proposal-b \
  --title 'Juniper Encoder: frozen Proposal B candidate for independent review' \
  --body-file docs/HANDOFF.md
```

Use the verified default base branch if it is not main. PR body: concrete scope, source SHA, candidate ID, artifacts, gates and failures, exact reproduction commands, target matrix, untested claims, and explicit request for Sol then Astra. Do not auto-merge or release.

## 16. Sol: independent clean-clone reproduction and audit

Sol must begin independently, not approve by reading Luna's summary. Fetch the candidate descriptor through its immutable identity, verify hashes, and use a new directory without borrowing Luna's virtual environment, untracked files, caches, or mutable local weights.

```bash
export JE_REVIEW_ROOT="$PWD/Juniper-Encoder-sol-review"
git clone 'https://github.com/Cinqic/Juniper-Encoder.git' "$JE_REVIEW_ROOT"
cd "$JE_REVIEW_ROOT"
git fetch origin "$JE_CANDIDATE_SHA"
git checkout --detach "$JE_CANDIDATE_SHA"
test "$(git rev-parse HEAD)" = "$JE_CANDIDATE_SHA"
test -z "$(git status --porcelain=v1)"
bash scripts/bootstrap.sh --lock requirements/flowbox.lock
source .venv/bin/activate
bash scripts/review-candidate.sh --candidate "$JE_CANDIDATE_MANIFEST" --reviewer sol
```

`JE_CANDIDATE_SHA` and `JE_CANDIDATE_MANIFEST` must come from the verified handoff; never paste guessed values. Use CPU-review lock only for explicitly CPU-scoped checks, and disclose the remaining FLOWBOX requirements.

Sol independently checks every traceability row; manually recomputes representative architecture totals and validates actual full counts; audits tokenizer byte/merge rules and repeat-build hashes; source rights and split leakage; labeling and metric denominators; no test-guided tuning; temporary-head removal; precision/cast behavior; deterministic resume in a new process; artifact/export/index compatibility; conditional gate semantics and comparison; every INT8 tolerance; benchmark matrix raw samples; exact CI SHA; candidate/weights/config provenance; model card truthfulness; and all preserved failed experiments.

Sol must reproduce at least complete contract/tokenizer tests, processed-data determinism on a meaningful frozen subset plus whole-manifest checks, learning/resume on a pinned training segment, held-out candidate evaluation from immutable inputs, export/load inference parity, and required target qualification measurements. If data access or hardware prevents any required result, record the blocked scope; no full approval. Full-from-scratch replay is required if source/checkpoint lineage cannot otherwise be independently established; distinguish it explicitly from segment reproduction.

Use independently constructed adversarial fixtures and inspect implementation internals, not only tests written by Luna. Record independent raw outputs and hashes. Approval names the exact source SHA and candidate descriptor hash, scope, executed commands, limitations, and reviewer identity. Blockers produce `REQUEST_CHANGES`; send them to Luna with reproduction steps and expected contracts. Luna fixes on the implementation branch, creates a new candidate, and Sol re-reviews affected areas plus regressions. Do not approve the old hash for the new artifact.

## 17. Astra: separate final audit after Sol

Only after Sol approves the exact immutable candidate, Astra independently clones/checks out that same SHA, reconstructs the locked environment, verifies the candidate hash and Sol's evidence, and runs the same review entrypoint with `--reviewer astra`. Astra must form an independent conclusion, not treat Sol's approval as correctness evidence by itself.

Prioritize hidden contract drift, tokenizer/JSON boundary bugs, incorrect parameter aliases/counts, leakage through tokenizer/pretraining/mining, unsupported CLARIFY semantics, calibration misuse, top-8 versus @10 evaluation confusion, shared-backbone/index staleness, false-CALL denominator tricks, INT8 metadata mismatch, near-boundary numerical behavior, benchmark exclusions, incomplete device scope, source/artifact identity mismatches, and missing failure history. Independently rerun contract/loader/hash tests and a meaningful target-backed evaluation/benchmark confirmation; expand reproduction where Sol's evidence or Astra's findings leave risk unresolved.

Astra records `APPROVE` or `REQUEST_CHANGES` with the exact SHA/candidate ID, commands/results, evidence hashes, and limitations. If Astra finds a blocker, send it to Luna; Sol reviews the repaired candidate again before Astra's renewed final audit. Disagreement is not resolved by majority vote, reviewer substitution, or hiding a limitation. It stays `REQUEST_CHANGES / NOT RELEASED` until resolved with evidence.

## 18. Required terminal handoff and status discipline

Luna's final report must give repository/branch/PR URLs; exact source SHA and candidate hash; phase statuses; actual training tokens/steps/seeds/time; frozen count verification; tokenizer/data/provenance identities; calibrated T/τ values; global and critical-slice metrics; conditional-vs-always deltas; complete device/backend/precision matrix; INT8 acceptance/rejection reasons; clean-clone/CI/resume evidence; unresolved issues; Sol status; Astra status; and the next concrete authorized action.

Use precise labels: `IMPLEMENTED`, `SMOKE_TESTED`, `TRAINED` with actual budget, `EVALUATED`, `QUALITY_QUALIFIED`, `INT8_ACCEPTED` or `INT8_REJECTED` per target, `CANDIDATE_READY`, `AWAITING_SOL`, `AWAITING_ASTRA`, `REQUEST_CHANGES`, and **`NOT RELEASED`**. Labels are separate dimensions, not a single progress bar. No review, dataset-access, hardware, metric, or evidence gap may be relabeled PASS or hidden as not applicable.

Finish all feasible authorized implementation and evidence work. If blocked, preserve a reproducible checkpoint/handoff and name the exact missing resource or decision, its origin, affected gates, and recovery command. Do not promise to keep working after the session has ended. Do not manufacture a trained model or successful review to satisfy the requested scope.

**The job is complete when the frozen implementation, real training/evaluation lineage, immutable candidate, and independently recorded reviews are concrete and verifiable—or when an explicit external blocker prevents a required gate and all feasible work plus recovery evidence has been completed. Neither outcome authorizes Luna to merge or release.**
