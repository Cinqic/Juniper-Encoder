# Juniper Encoder — Proposal B

**Developer:** Cinqic  
**Status:** FROZEN  
**Architecture:** Dense, text-only, encoder-only Transformer  
**Deployed parameters:** 49,441,796  
**Shared backbone:** 49,308,672  
**Native context:** 1,024 tokens per encoder input  

## 1. Purpose and Scope

Juniper Encoder routes user requests to registered capabilities within the Juniper ecosystem.

A capability may be an:

- Agent
- Tool
- Skill

Juniper Encoder is responsible for:

- Understanding a user request using relevant retained conversation context.
- Retrieving relevant registered capabilities.
- Reranking ambiguous capability candidates when necessary.
- Selecting one capability when appropriate.
- Returning `NO_CALL` or `CLARIFY` when appropriate.

Juniper Encoder does **not**:

- Answer users.
- Generate prose.
- Generate execution arguments.
- Perform complex planning.
- Execute actions.
- Enforce permissions.
- Replace worker models or agents.

Capabilities are stored in an external dynamic registry. Capability identities are not fixed classifier outputs.

Adding a new capability does not change Juniper Encoder's architecture or parameter count.

---

# 2. Backbone

| Property | Frozen specification |
|---|---|
| Hidden dimension | 512 |
| Layers | 12 identical dense Transformer encoder blocks |
| Attention | Full bidirectional self-attention |
| Query heads | 8 |
| KV heads | 8 |
| Head dimension | 64 |
| Attention scaling | 1/√64 = 1/8 |
| QK normalization | None |
| Attention pattern | Global dense attention in every layer |
| FFN | GeGLU |
| Gate branch width | 1,536 |
| Value branch width | 1,536 |
| Expansion | 3× hidden dimension per branch |
| Fused input projection | 512 → 3,072, ordered `[gate, value]` |
| FFN output projection | 1,536 → 512 |
| GELU | Exact erf formulation |
| Normalization | Two Pre-LayerNorms per block + final LayerNorm |
| LayerNorm parameters | Learned scale and bias |
| LayerNorm epsilon | 1e-5 |
| LayerNorm variance | Population variance over 512 features |
| Position encoding | RoPE |
| RoPE theta | 10,000 |
| Rotary dimensions | All 64 dimensions of every Q/K head |
| RoPE scaling | None |
| Residuals | Standard additive |
| Residual scaling | None |
| Attention bias | None |
| FFN bias | None |
| Token embeddings | Learned 16,384 × 512 |
| Embedding scaling | None |
| Absolute position embeddings | None |
| Segment embeddings | None |
| MoE | None |
| Generative deployment head | None |
| Softcapping | None |

A 2,048-token version requires a separate specification revision, continued training, and independent evaluation.

---

# 3. Parameter Counts

| Component | Parameters |
|---|---:|
| Token embeddings | 8,388,608 |
| Attention per block | 1,048,576 |
| GeGLU per block | 2,359,296 |
| LayerNorms per block | 2,048 |
| Complete block | 3,409,920 |
| Twelve blocks | 40,919,040 |
| Final LayerNorm | 1,024 |
| **Shared backbone** | **49,308,672** |
| Retrieval head | 131,072 |
| Decision classifier | 1,539 |
| Capability reranker | 513 |
| **Deployed total** | **49,441,796** |

### Mode-Specific Participating Parameters

| Mode | Parameters |
|---|---:|
| Retrieval | 49,439,744 |
| Decision classification | 49,310,211 |
| Capability reranking | 49,309,185 |
| Fast routing pipeline, unique parameters | 49,441,283 |
| Reranked routing pipeline, unique parameters | 49,441,796 |
| Total deployed | 49,441,796 |

Counts include the complete embedding table, including the fixed-zero PAD row.

Repeated encoder passes do not multiply unique parameter count.

Registry embeddings, calibration metadata, tokenizer artifacts, and quantization metadata are not model parameters.

---

# 4. Transformer Forward Computation

Matrices use `[input, output]` mathematical orientation.

Let `P` zero padded rows.

Let `Dₚ` be independent elementwise inverted dropout with `p = 0.1` during training and identity during inference.

\[
X_0=P(D_p(E[\mathrm{input\_ids}]))
\]

For each block \(l=1,\ldots,12\):

\[
U=\operatorname{LN}_{l,1}(X_{l-1})
\]

\[
Q=\operatorname{RoPE}(UW_Q)
\]

\[
K=\operatorname{RoPE}(UW_K)
\]

\[
V=UW_V
\]

After reshaping into eight heads:

\[
A=D_p\left(
\operatorname{softmax}_{keys}(QK^\top/8+M)
\right)
\]

\[
Y=P\left(
X_{l-1}
+
D_p(\operatorname{ConcatHeads}(AV)W_O)
\right)
\]

\[
Z=\operatorname{LN}_{l,2}(Y)
\]

\[
X_l=P\left(
Y+
D_p(
[\operatorname{GELU}(ZW_g)\odot ZW_v]W_d
)
\right)
\]

After block 12:

\[
H=P(\operatorname{LN}_{final}(X_{12}))
\]

Exact GELU:

\[
\operatorname{GELU}(x)
=
\frac{x}{2}
\left[
1+\operatorname{erf}(x/\sqrt2)
\right]
\]

### Matrix Shapes

| Matrix | Shape |
|---|---|
| WQ | 512 × 512 |
| WK | 512 × 512 |
| WV | 512 × 512 |
| WO | 512 × 512 |
| Wg | 512 × 1,536 |
| Wv | 512 × 1,536 |
| Wd | 1,536 × 512 |

`M = 0` for real keys and `−∞` for PAD keys.

No causal mask is used.

Structural tokens participate normally in attention.

Padded query rows may compute attention but their resulting hidden rows are zeroed.

---

# 5. Dropout

| Location | Training | Inference |
|---|---:|---:|
| Embeddings | 0.1 | 0.0 |
| Attention probabilities | 0.1 | 0.0 |
| Attention residual branch | 0.1 | 0.0 |
| FFN residual branch | 0.1 | 0.0 |

Attention dropout occurs after softmax.

Attention-output and FFN-output dropout occurs before residual addition.

There is no additional:

- Head dropout
- Pooling dropout
- Post-residual dropout
- Extra branch dropout

---

# 6. RoPE and Position Handling

For coordinate pairs \((2j,2j+1)\), where \(j=0,\ldots,31\):

\[
\theta_{p,j}=p\,10000^{-2j/64}
\]

For pair `(a,b)`:

\[
(a,b)\mapsto
(a\cos\theta-b\sin\theta,\;
a\sin\theta+b\cos\theta)
\]

RoPE applies only to:

- Q
- K

RoPE does not apply to:

- V
- Token embeddings
- Attention outputs

Position IDs begin at zero and increase across the complete unpadded sequence, including structural tokens.

Query-document pair positions are continuous and do not reset at the document boundary.

Right padding only.

PAD position IDs are zero.

No RoPE interpolation or scaling is used.

---

# 7. Tokenizer

**Type:** Byte-level BPE  
**Vocabulary:** Exactly 16,384 tokens  
**Base byte coverage:** Complete 256-byte alphabet  

### Structural Tokens

| ID | Token |
|---:|---|
| 0 | `[PAD]` |
| 1 | `[UNK]` |
| 2 | `[CLS]` |
| 3 | `[SEP]` |
| 4 | `[MASK]` |
| 5 | `[QUERY]` |
| 6 | `[DOCUMENT]` |

Byte value `b ∈ [0,255]` maps to:

\[
ID=7+b
\]

IDs `263–16,383` contain exactly **16,121 learned BPE merge tokens**.

Encoding rules:

- Input Unicode is encoded as UTF-8.
- No Unicode normalization.
- No case folding.
- No whitespace cleanup.
- No prefix-space insertion.
- No regex pretokenization.
- BPE operates separately within each template content field.
- Structural boundaries cannot be merged across.
- Structural IDs are inserted only by the formatter.
- Literal strings such as `[CLS]` in user text encode as ordinary bytes.
- Valid Unicode must never require UNK.
- Invalid Unicode is rejected.
- PAD, UNK, and MASK are prohibited inside normal inference content.

Ordinary-token decoding must reconstruct the original UTF-8 input exactly before truncation.

The released tokenizer artifact must include:

- Byte mapping
- Vocabulary
- Ordered merges
- Configuration
- Implementation version
- Conformance tests
- SHA-256 hashes

Every checkpoint references its tokenizer manifest and rejects incompatible tokenizer artifacts.

---

# 8. External Capability Registry

Each registered capability contains exactly these semantic fields in order:

| Field | Type |
|---|---|
| id | String |
| type | String |
| name | String |
| purpose | String |
| supported_requests | String array |
| exclusions | String array |
| required_inputs | String array |

`type` must be:

- `agent`
- `tool`
- `skill`

Capability IDs must:

- Be unique.
- Be nonempty.
- Use case-sensitive ASCII letters, digits, `.`, `_`, or `-`.

Registry descriptions are descriptive data, not executable instructions.

Each record is serialized as compact UTF-8 JSON using the field order above.

No optional whitespace or Unicode normalization is applied.

A serialized capability record may contain at most **256 ordinary tokens**.

Oversized records are rejected.

Invocation schemas, executable implementations, credentials, permissions, and runtime authority remain external.

Each routing request uses one immutable registry snapshot from start to finish.

---

# 9. Capability Embeddings

Capability descriptions are pre-embedded using the deployed retrieval encoder.

Cached embeddings are keyed by:

- Capability ID
- Capability-description hash
- Tokenizer identity
- Encoder artifact identity

Changing a capability description requires re-embedding that capability.

Changing the encoder or tokenizer requires rebuilding the registry index.

The reference implementation uses **exact normalized dot-product search**.

Approximate nearest-neighbor indexing may be introduced separately when justified by measured registry scale.

---

# 10. Request Context

Request content uses compact JSON:

```text
{"history":[],"user":"…"}
```

`history` contains chronological objects containing:

```text
role
content
```

Allowed roles:

- `user`
- `assistant`
- `tool`

The request-context budget is **256 ordinary tokens**.

Remove the oldest complete history entries until the retained request fits.

The current user request must never be silently truncated.

If the current user request alone exceeds 256 ordinary tokens:

```text
INPUT_TOO_LONG
```

Empty or Unicode-whitespace-only current requests are rejected.

Exactly the same retained request representation is used throughout retrieval, reranking, and classification.

---

# 11. Input Templates

### Request Retrieval

```text
[CLS][QUERY] request [SEP]
```

### Capability Indexing

```text
[CLS][DOCUMENT] capability [SEP]
```

### Capability Reranking

```text
[CLS][QUERY] request [SEP][DOCUMENT] capability [SEP]
```

### Decision Classification

```text
[CLS][QUERY] decision_context [SEP]
```

The generic pair formatter permits **1,019 content tokens**.

For generic pair inputs:

- Retain at most 256 query tokens.
- Give remaining content capacity to the document.

Registered routing inputs already enforce stricter 256-token request and capability limits.

Structural tokens count toward the 1,024-token context.

---

# 12. Routing Pipeline

The frozen pipeline is:

```text
Validate
   ↓
Retrieve
   ↓
Top-8 Shortlist
   ↓
Retrieval Ambiguity Gate
   ├── Clear → Fast Path
   └── Ambiguous → Capability Reranking
   ↓
Decision Classification
   ↓
CALL / NO_CALL / CLARIFY
```

## Step 1 — Validate

Validate:

- Request
- Registry snapshot
- Tokenizer compatibility
- Encoder/index compatibility

## Step 2 — Retrieve

Encode the retained request once.

Score all registered capability embeddings using normalized dot product.

## Step 3 — Shortlist

Retain:

\[
\min(8,N)
\]

capabilities, where `N` is registry size.

Retrieval ties are resolved by ascending capability ID.

## Step 4 — Conditional Reranking

Reranking is **not mandatory for every request**.

Let:

- `s₁` = top retrieval score
- `s₂` = second-best retrieval score
- `Δ = s₁ − s₂`

Two deployment thresholds are stored as nontrainable checkpoint metadata:

- `τ_score`
- `τ_margin`

For registries with at least two candidates, the fast path may skip reranking only when:

\[
s_1 \ge \tau_{score}
\]

and:

\[
\Delta \ge \tau_{margin}
\]

Otherwise, cross-encode and rerank the complete top-eight shortlist.

If only one capability exists, reranking is skipped because ranking cannot change.

The thresholds are fitted using held-out routing calibration data and frozen with the checkpoint.

They must not adapt automatically during production inference.

When reranking occurs:

- Every shortlisted capability is scored.
- Higher reranker score ranks first.
- Exact-score ties use ascending capability ID.

When reranking is skipped:

- Retrieval order becomes the final candidate ranking.

This design preserves high-precision reranking for ambiguous requests while avoiding unnecessary repeated encoder passes for clearly separated candidates.

## Step 5 — Classify

The classifier receives the retained request and up to the top two final-ranked capability records.

Final ranking means:

- Reranker ordering when reranking occurred.
- Retrieval ordering when fast-path reranking was skipped.

Classifier context:

```text
{"request":{"history":[],"user":"…"},"capabilities":[…]}
```

If context exceeds 1,021 content tokens:

1. Remove the second-ranked capability.
2. Recheck.
3. If still oversized, return `INPUT_TOO_LONG`.

Do not truncate individual capability descriptions or the retained request.

## Step 6 — Return

The system outputs one of:

- CALL
- NO_CALL
- CLARIFY

For CALL, return only the highest-ranked capability ID.

The classifier may accept or reject the top candidate but does not choose another capability.

The reranking/retrieval subsystem determines candidate order.

The routing pipeline returns at most **one capability per user request**.

Complex requests may route to an agent whose purpose includes task decomposition or orchestration.

---

# 13. Empty Registry

An empty capability registry produces:

```text
NO_CALL
```

after input validation and without neural routing inference.

---

# 14. Retrieval Head

Pipeline:

```text
Final hidden states
→ masked mean pooling
→ bias-free 512→256 projection
→ L2 normalization
```

The pooling mask includes ordinary content tokens only.

It excludes:

- PAD
- UNK
- CLS
- SEP
- MASK
- QUERY
- DOCUMENT

\[
m=
\frac{\sum_i r_iH_i}
{\sum_i r_i}
\]

\[
z=mW_R
\]

where:

\[
W_R\in\mathbb R^{512\times256}
\]

Normalize:

\[
e=
\frac{z}
{\max(\|z\|_2,10^{-12})}
\]

Return 256-dimensional FP32 embeddings.

Reject:

- Empty pooling masks.
- Nonfinite projections.
- Projection norms below \(10^{-12}\).

Inference similarity:

\[
e_q^\top e_d
\]

No learned inference scaling is used.

Contrastive-training temperature:

\[
0.05
\]

---

# 15. Capability Reranker

Input:

```text
[CLS][QUERY] request [SEP][DOCUMENT] capability [SEP]
```

Score:

\[
s=H_{CLS}W_S+b_S
\]

where:

```text
512 → 1
```

The head includes bias.

Return one FP32 scalar relevance score.

Higher is better.

No sigmoid.

No softmax.

The score is not interpreted as probability.

Scores are compared only among capabilities for the same request.

---

# 16. Decision Classifier

\[
logits=H_{CLS}W_C+b_C
\]

Projection:

```text
512 → 3
```

with bias.

| Index | Decision | Frozen semantics |
|---:|---|---|
| 0 | CALL | The top-ranked registered capability is appropriate for the user's request and sufficient user-provided information exists to select it. Explicit user preference for that capability is sufficient when otherwise valid. |
| 1 | NO_CALL | No suitable available capability should currently receive the request. This includes direct conversation, acknowledgments, unsupported capability requests, and cases where no registered capability applies. |
| 2 | CLARIFY | A specific missing or materially ambiguous user-provided detail prevents reliable capability selection or establishing the capability's required inputs. |

`CLARIFY` represents semantic missing information.

It is **not** a generic low-confidence state.

Low model confidence is handled separately through calibration and runtime policy.

The classifier does not:

- Select another candidate.
- Generate arguments.
- Authorize actions.
- Execute capabilities.
- Enforce permissions.

---

# 17. Decision Calibration

Apply:

\[
p=
\operatorname{softmax}(logits/T)
\]

where scalar:

\[
T>0
\]

is fitted using held-out calibration data.

`T` is nontrainable deployment metadata.

Before calibration:

```text
T = 1
```

and probabilities must be marked uncalibrated.

Prediction uses argmax.

Exact ties choose the lowest class index.

---

# 18. Output Contract

Successful CALL:

```json
{"decision":"CALL","capability_id":"registered.id"}
```

Successful NO_CALL:

```json
{"decision":"NO_CALL","capability_id":null}
```

Successful CLARIFY:

```json
{"decision":"CLARIFY","capability_id":null}
```

`capability_id` must originate from the immutable registry snapshot.

Juniper Encoder never generates capability identifiers.

Decision probabilities, reranking information, fast-path/reranking status, and context reduction are returned separately as metadata.

Reranker scores are never confidence probabilities.

---

# 19. Error Contract

Input, compatibility, registry, or numerical failures produce explicit errors rather than fabricated routing decisions.

Frozen error codes:

```text
INVALID_INPUT
INPUT_TOO_LONG
INVALID_REGISTRY
INDEX_MISMATCH
NUMERICAL_ERROR
```

The external Juniper application or worker system handles:

- User-facing clarification.
- Answers.
- Argument construction.
- Permission checks.
- Authentication.
- Execution.
- Complex planning.

---

# 20. Initialization

| Component | Initialization |
|---|---|
| Embeddings | Normal mean 0, underlying σ 0.02, rejection-truncated to [-0.04,+0.04] |
| Linear matrices | Same |
| LayerNorm scale | 1 |
| LayerNorm bias | 0 |
| Classifier bias | 0 |
| Reranker bias | 0 |
| PAD embedding row | Fixed zero |
| Depth-dependent residual rescaling | None |

The stated `0.02` standard deviation refers to the underlying normal distribution before truncation.

The PAD row remains zero during:

- Training
- Optimizer updates
- Export
- Deployment

Its 512 stored entries remain included in parameter counts.

---

# 21. Weight Tying and Pretraining Heads

The deployed Juniper Encoder contains no weight tying.

A temporary language-pretraining head may exist during foundation pretraining.

A temporary MLM decoder may share the input embedding matrix.

Any temporary pretraining-only parameters:

- Must be counted separately.
- Are not part of the 49,441,796 deployed parameters.
- Must be removed before deployment.

---

# 22. Precision

| Operation | Precision |
|---|---|
| Training | BF16 preferred |
| Reference inference | BF16 or FP16 |
| LayerNorm statistics | FP32 |
| Attention softmax | FP32 |
| Pooling sums | FP32 |
| L2 norm | FP32 |
| Classifier probabilities | FP32 |
| Retrieval embedding output | FP32 |
| Reranker score output | FP32 |
| Primary quantized deployment | Validated INT8 |

Reference inference must pass finite-output validation.

---

# 23. Evaluation Contract

Juniper Encoder must be evaluated both per component and end to end.

Required coverage includes:

- Agent routing.
- Tool routing.
- Skill routing.
- Correct NO_CALL behavior.
- Correct CLARIFY behavior.
- Unnecessary-call restraint.
- Unsupported requests.
- Similar and overlapping capabilities.
- Missing required inputs.
- Contextual ambiguity.
- Capabilities introduced after model training.
- Registry additions.
- Registry removals.
- Description changes.
- Empty registries.
- Misleading or adversarial text inside requests or descriptions.
- Context reduction.
- Input-limit failures.
- Numerical failures.
- Fast-path routing.
- Reranked routing.
- Fast-path versus always-rerank accuracy.

Entire capability identities must be held out when evaluating adaptability.

Required metrics:

```text
Shortlist Recall@8
Retrieval Recall@10
Retrieval nDCG@10
Reranker nDCG@10
Reranker MRR@10
Decision macro-F1
CALL recall
False CALL rate
Calibration error
End-to-end exact routing accuracy
Rerank invocation rate
Fast-path exact routing accuracy
```

A CALL counts as end-to-end correct only when both:

1. The CALL decision is correct.
2. The capability ID is correct.

---

# 24. Conditional-Reranking Validation

The conditional fast path must be evaluated against an always-rerank reference pipeline.

Report:

- Percentage of requests skipping reranking.
- End-to-end routing accuracy difference.
- False CALL difference.
- Capability-selection accuracy difference.
- Median encoder-pass reduction.
- p50 latency reduction.
- p95 latency reduction.

`τ_score` and `τ_margin` are checkpoint-specific calibration metadata, not learned model parameters.

Any checkpoint must publish its exact frozen thresholds.

---

# 25. INT8 Acceptance Criteria

INT8 is the primary quantized deployment target.

Keep the following operations floating point unless separately validated:

- LayerNorm
- Softmax
- Mean pooling
- L2 normalization
- Retrieval head
- Decision head
- Reranking head

Each quantized artifact records:

- Weight precision per operator.
- Activation precision per operator.
- Quantization axes.
- Quantization scales.
- Excluded operators.
- Runtime and version.
- Target hardware.

Quantized evaluation must use:

- The same checkpoint.
- Identical request data.
- Identical registry snapshots.
- Identical tokenizer.
- Identical formatting.
- Identical calibration metadata.

Registry embeddings must be generated using the corresponding deployed encoder variant.

Calibration data must remain separate from evaluation data.

### Required Gates

| Metric | Maximum degradation |
|---|---:|
| Shortlist Recall@8 | ≤0.01 absolute |
| Retrieval Recall@10 | ≤0.01 |
| Retrieval nDCG@10 | ≤0.01 |
| Reranker nDCG@10 | ≤0.01 |
| Reranker MRR@10 | ≤0.01 |
| Decision macro-F1 | ≤0.01 |
| CALL recall | ≤0.01 |
| End-to-end exact routing accuracy | ≤0.01 |
| False CALL rate | Increase ≤0.005 |
| Calibration ECE | Increase ≤0.01 |
| Numerical validity | No failures |
| Warm p50 latency | ≤1.05× reference |
| Warm p95 latency | ≤1.05× reference |
| Model artifact size | ≤0.65× reference |
| Peak inference memory | ≤ floating-point reference |

Quality metrics use the `[0,1]` scale.

Apply quality gates both globally and to prespecified critical slices, including held-out capabilities.

---

# 26. Deployment Benchmark Contract

Benchmark every inference mode on every target device/backend.

Test:

```text
Batch sizes:
1
8

Padded sequence lengths:
128
512
1,024
```

Use:

- Identical thread configuration.
- Identical device configuration.
- At least 20 warmup runs.
- At least 200 measured runs per case.

Record:

- p50 latency
- p95 latency
- Peak memory
- Registry size
- Shortlist size
- Number of encoder passes
- Reranking invocation rate
- Cache state

Complete-pipeline measurements include:

- Tokenization
- Formatting
- Retrieval search
- Conditional reranking
- Decision classification
- Postprocessing

Exclude:

- Model loading
- Offline registry indexing

Devices that fail INT8 acceptance retain floating-point deployment.

---

# 27. Frozen Architecture Summary

```text
Juniper Encoder Proposal B

49,441,796 deployed parameters
49,308,672 shared-backbone parameters

Dense encoder-only Transformer
Text only

12 Transformer blocks
d_model = 512

8 query heads
8 KV heads
head_dim = 64

Full global bidirectional attention
Attention scale = 1/8
No QK normalization

GeGLU
Gate width = 1,536
Value width = 1,536
Exact GELU

Pre-LayerNorm
Final LayerNorm
epsilon = 1e-5

RoPE
theta = 10,000
100% Q/K rotary dimensions
No scaling

16,384-token byte-level BPE
Complete 256-byte coverage

Native context = 1,024

Retrieval:
Masked mean pooling
512 → 256 bias-free projection
L2 normalization

Capability retrieval:
Exact cosine/dot-product search
Top-8 shortlist

Reranking:
Conditional
Joint query-capability cross-encoding
512 → 1 scalar head

Decision:
CALL
NO_CALL
CLARIFY

Capabilities:
Agents
Tools
Skills

Capability registry:
External
Dynamic
Versioned
Description-driven

No fixed capability output classes
No generation
No argument generation
No complex planning
No execution
No permission enforcement
No MoE

Primary deployment target:
Validated INT8
```

## Frozen Core Principle

> **Juniper Encoder does not determine how to complete a task. It determines which registered agent, tool, or skill should receive the request.**

**Proposal B is frozen.**