# Juniper Encoder — Proposal B

**Developer:** Cinqic  
**Status:** Frozen architecture and routing specification  
**Architecture:** Dense, text-only, encoder-only Transformer  
**Deployed parameters:** **49,441,796**  
**Native context:** **1,024 tokens per encoder input**

## 1. Purpose and Scope

Juniper Encoder routes user requests to registered capabilities within the Juniper ecosystem. A capability may be an **agent, tool, or skill**.

Its responsibilities are:

- Understand the request using relevant conversation context.
- Retrieve and rank available capabilities.
- Select one capability when appropriate.
- Return `NO_CALL` or `CLARIFY` when appropriate.

Juniper Encoder does not generate answers or prose, construct execution arguments, perform complex planning, execute actions, enforce permissions, or replace worker models and agents.

Capability discovery uses an external registry and standardized descriptions. Capability identities are not fixed classifier outputs. Registering a new capability does not change the architecture or parameter count.

## 2. Backbone

| Property | Frozen specification |
|---|---|
| Hidden dimension | 512 |
| Layers | 12 identical dense Transformer encoder blocks |
| Attention | Full bidirectional self-attention in every layer |
| Query heads / KV heads | 8 / 8 |
| Head dimension | 64 |
| Attention scaling | \(1/\sqrt{64}=1/8\) |
| QK normalization | None |
| FFN | GeGLU |
| Gate / value branch widths | 1,536 / 1,536 |
| Expansion | 3× hidden dimension per branch |
| Fused input projection | 512 → 3,072, ordered `[gate, value]` |
| FFN output projection | 1,536 → 512 |
| GELU | Exact erf formulation |
| Normalization | Two pre-LayerNorms per block; one final LayerNorm |
| LayerNorm parameters | Learned scale and bias |
| LayerNorm epsilon | \(10^{-5}\) |
| LayerNorm variance | Population variance across 512 features |
| Positional encoding | RoPE; theta 10,000 |
| Rotary dimensions | All 64 dimensions of each Q/K head |
| RoPE scaling | None |
| Residuals | Additive; no residual scaling |
| Attention / FFN biases | None |
| Token embeddings | Learned 16,384 × 512 matrix |
| Embedding scaling | None |
| Absolute position / segment embeddings | None |
| MoE / experts / internal expert router | None |
| Generative deployment head | None |
| Softcapping | None |

A future 2,048-token version requires a separate specification revision, continued training, and independent evaluation.

## 3. Parameter Counts

| Component | Calculation | Parameters |
|---|---:|---:|
| Token embeddings | \(16,384\times512\) | 8,388,608 |
| Attention per block | \(4\times512^2\) | 1,048,576 |
| GeGLU per block | \(3\times512\times1,536\) | 2,359,296 |
| LayerNorms per block | \(2\times2\times512\) | 2,048 |
| Complete block | Attention + GeGLU + LayerNorms | 3,409,920 |
| Twelve blocks | \(12\times3,409,920\) | 40,919,040 |
| Final LayerNorm | \(2\times512\) | 1,024 |
| **Shared backbone** | Embeddings + blocks + final LayerNorm | **49,308,672** |
| Retrieval head | \(512\times256\) | 131,072 |
| Decision classifier | \(512\times3+3\) | 1,539 |
| Capability reranker | \(512\times1+1\) | 513 |
| **Deployed total** | Backbone + three heads | **49,441,796** |

| Mode | Participating parameters |
|---|---:|
| Retrieval | **49,439,744** |
| Decision classification | **49,310,211** |
| Capability reranking | **49,309,185** |
| Complete routing pipeline: unique parameters | **49,441,796** |

Counts include the complete embedding table and its fixed-zero PAD row. They describe participating modules, not individual embedding rows accessed. Repeated encoder passes do not multiply the unique parameter count.

Inactive heads remain resident unless omitted from the loaded artifact. Registry embeddings, calibration metadata, and quantization scales are separate from model parameters.

## 4. Exact Forward Computation

Matrices use mathematical `[input, output]` orientation. Implementations may store transposes.

Let \(P\) zero padded rows. Let \(D_p\) apply independent elementwise inverted dropout with \(p=0.1\) during training and identity during inference.

\[
X_0=P(D_p(E[\mathrm{input\_ids}]))
\]

For each block \(l=1,\ldots,12\):

\[
U=\operatorname{LN}_{l,1}(X_{l-1})
\]

\[
Q=\operatorname{RoPE}(UW_Q),\qquad
K=\operatorname{RoPE}(UW_K),\qquad
V=UW_V
\]

After reshaping Q, K, and V into eight heads:

\[
A=D_p\left(
\operatorname{softmax}_{\mathrm{keys}}(QK^\top/8+M)
\right)
\]

\[
Y=P\left(
X_{l-1}+D_p(\operatorname{ConcatHeads}(AV)W_O)
\right)
\]

\[
Z=\operatorname{LN}_{l,2}(Y)
\]

\[
X_l=P\left(
Y+D_p\left(
[\operatorname{GELU}(ZW_g)\odot ZW_v]W_d
\right)
\right)
\]

\[
H=P(\operatorname{LN}_{final}(X_{12}))
\]

\[
\operatorname{GELU}(x)
=\frac{x}{2}\left[1+\operatorname{erf}(x/\sqrt2)\right]
\]

| Matrix | Shape |
|---|---|
| \(W_Q,W_K,W_V,W_O\) | \(512\times512\) each |
| \(W_g,W_v\) | \(512\times1,536\) each |
| \(W_d\) | \(1,536\times512\) |

**Masking:** \(M=0\) for real keys and \(-\infty\) for PAD keys. No causal mask. Structural tokens participate in attention. Padded queries may compute attention, but their resulting rows are zeroed. Accepted sequences always contain real tokens.

**Dropout:** Applied only to embeddings, attention probabilities after softmax, attention output before residual addition, and FFN output before residual addition.

| Location | Training | Inference |
|---|---:|---:|
| Embeddings | 0.1 | 0.0 |
| Attention probabilities | 0.1 | 0.0 |
| Attention residual branch | 0.1 | 0.0 |
| FFN residual branch | 0.1 | 0.0 |

No additional branch, head, pooling, or post-residual dropout.

## 5. RoPE and Position Handling

For adjacent coordinate pairs \((2j,2j+1)\), \(j=0,\ldots,31\):

\[
\theta_{p,j}=p\,10000^{-2j/64}
\]

\[
(a,b)\mapsto
(a\cos\theta-b\sin\theta,\;
a\sin\theta+b\cos\theta)
\]

- Apply identical adjacent-pair conventions to Q and K after projection and before attention scoring.
- Do not rotate V, embeddings, or attention outputs.
- Position IDs start at zero and increase across the complete unpadded sequence, including structural tokens.
- Pair inputs do not reset positions at the document boundary.
- Right-padding only; PAD position IDs are zero.
- No interpolation or RoPE scaling.

## 6. Tokenizer

**Type:** Byte-level BPE  
**Vocabulary:** Exactly 16,384 tokens  
**Base coverage:** All 256 bytes

| ID | Structural token |
|---:|---|
| 0 | `[PAD]` |
| 1 | `[UNK]` |
| 2 | `[CLS]` |
| 3 | `[SEP]` |
| 4 | `[MASK]` |
| 5 | `[QUERY]` |
| 6 | `[DOCUMENT]` |

- Byte value \(b\in[0,255]\) maps to token ID \(7+b\).
- IDs 263–16,383 contain exactly 16,121 learned BPE merge tokens.
- Encode valid Unicode as UTF-8 without normalization, case folding, whitespace cleanup, prefix-space insertion, or regex pretokenization.
- Apply BPE separately to each template content field. Select the lowest-rank merge; break equal-rank occurrence ties left-to-right.
- Never merge across structural-token boundaries.
- Structural IDs are inserted exclusively by the formatter. Literal structural-token spellings in text encode as ordinary bytes.
- Valid input must remain representable without UNK substitution. Reject invalid Unicode.
- PAD, UNK, and MASK are forbidden inside normal inference content.
- Ordinary-token decoding must reconstruct the original UTF-8 text exactly before truncation.

The tokenizer release includes the byte mapping, vocabulary, ordered merges, configuration, implementation version, conformance tests, and SHA-256 hashes. Each checkpoint references this manifest and rejects tokenizer mismatches.

## 7. External Capability Registry

Each registered capability provides the following fields in this order:

| Field | Type | Contract |
|---|---|---|
| `id` | String | Unique, stable capability identifier |
| `type` | String | `agent`, `tool`, or `skill` |
| `name` | String | Human-readable name |
| `purpose` | String | Concise responsibility |
| `supported_requests` | String array | Supported intents or representative requests |
| `exclusions` | String array | Explicit unsupported uses |
| `required_inputs` | String array | Information necessary to request use |

IDs use case-sensitive ASCII letters, digits, `.`, `_`, or `-` and must be nonempty. Duplicate IDs are rejected. Registry text is descriptive data, not executable instructions.

Serialize each complete record as compact UTF-8 JSON, preserving the field order above and array order. Use standard JSON escaping, no optional whitespace, and no Unicode normalization.

Each serialized record must fit **256 ordinary tokens**. Reject oversized records rather than silently deleting capability constraints. Invocation schemas, implementations, credentials, and permission state remain external.

The application supplies the available registry snapshot. Each request uses one immutable snapshot throughout retrieval, reranking, and classification.

Precompute description embeddings with the deployed retrieval encoder. Cache entries are keyed by capability ID, description hash, tokenizer identity, and encoder artifact identity. Description changes require re-embedding; encoder or tokenizer changes require rebuilding the index.

## 8. Request Context and Templates

Request content is compact JSON with fixed field order:

```json
{"history":[],"user":"…"}
```

`history` contains chronological objects with `role` followed by `content`. Roles are `user`, `assistant`, or `tool`; content and current user values are strings.

Drop oldest history entries until serialized request content fits **256 ordinary tokens**. If the current user request alone exceeds this budget, return `INPUT_TOO_LONG`. Do not silently truncate the current request. Reject empty or Unicode-whitespace-only current requests.

Use the same retained request content for retrieval, reranking, and classification. Report removed history entries in metadata.

| Mode | Template |
|---|---|
| Request retrieval | `[CLS][QUERY] request [SEP]` |
| Capability indexing | `[CLS][DOCUMENT] capability [SEP]` |
| Capability reranking | `[CLS][QUERY] request [SEP][DOCUMENT] capability [SEP]` |
| Decision classification | `[CLS][QUERY] decision_context [SEP]` |

The general pair formatter permits 1,019 content tokens: retain at most the first 256 query tokens and then at most \(1,019-|q|\) document tokens. Valid registry-routing inputs already satisfy stricter 256-token limits per field and therefore require no pair truncation.

Structural tokens count toward the 1,024-token native limit. No formatter may remove structural tokens or silently truncate classifier context.

## 9. Routing Pipeline

1. **Validate:** Validate request, registry snapshot, tokenizer, and index compatibility.
2. **Retrieve:** Encode the retained request and score every registered description by normalized dot product.
3. **Shortlist:** Retain the top \(\min(8,N)\) capabilities, where \(N\) is registry size.
4. **Rerank:** Cross-encode the request with each shortlisted capability.
5. **Classify:** Evaluate `CALL`, `NO_CALL`, or `CLARIFY` using the request and the highest-ranked capability descriptions.
6. **Return:** Emit the decision and, for `CALL`, the highest-ranked capability ID.

Retrieval uses exact dot-product search for the reference implementation. Ties in retrieval or reranking are resolved by ascending capability ID.

An empty registry returns `NO_CALL` after request validation, without model inference.

The classifier receives up to the top **two** reranked records in descending relevance order. Build compact JSON with this field order:

```json
{"request":{"history":[],"user":"…"},"capabilities":[…]}
```

If this exceeds 1,021 content tokens, remove the second-ranked record and recheck. If the remaining context still exceeds the budget, return `INPUT_TOO_LONG`. Do not truncate individual records or the retained request.

The classifier assesses whether the top-ranked capability is appropriate. It cannot select another candidate directly. Reranking determines selection; classification determines whether to use it.

The pipeline returns at most one capability per request. Requests requiring complex decomposition may route to a registered agent whose description covers that responsibility.

## 10. Task Heads

All heads consume the final LayerNorm output. No additional CLS pooler, activation, or shared head weights.

### 10.1 Retrieval Head

**Pipeline:** Masked mean pooling → bias-free 512 → 256 projection → L2 normalization.

The pooling mask \(r_i\) includes ordinary content positions and excludes all seven structural IDs.

\[
m=\frac{\sum_i r_iH_i}{\sum_i r_i}
\]

\[
z=mW_R,\qquad W_R\in\mathbb R^{512\times256}
\]

\[
e=\frac{z}{\max(\|z\|_2,10^{-12})}
\]

Return FP32 embeddings. Reject an empty pooling mask, nonfinite projection, or projection norm below \(10^{-12}\). Do not index or score invalid outputs.

Inference similarity is \(e_q^\top e_d\), without learned scaling. Contrastive-training temperature is fixed at **0.05** and recorded in training configuration.

### 10.2 Capability Reranker

\[
s=H_{\mathrm{CLS}}W_S+b_S
\]

**Projection:** 512 → 1, with bias.

Return one FP32 relevance score. Higher is better. No sigmoid, softmax, or probability interpretation. Scores rank capabilities for the same request.

### 10.3 Decision Classifier

\[
\mathrm{logits}=H_{\mathrm{CLS}}W_C+b_C
\]

**Projection:** 512 → 3, with bias.

| Index | Label | Semantics |
|---:|---|---|
| 0 | `CALL` | The top-ranked registered capability is appropriate, is needed, and has sufficient request information to be selected |
| 1 | `NO_CALL` | No suitable available capability is needed or supported by the supplied candidates |
| 2 | `CLARIFY` | Missing or ambiguous user information prevents reliable capability selection or establishing required inputs |

Use `CLARIFY` when a specific user-provided detail can resolve the routing uncertainty. Unsupported requests and the absence of a suitable capability map to `NO_CALL`. Greetings and acknowledgments map to `NO_CALL` unless a registered capability is explicitly needed.

The classifier assesses information sufficiency from descriptions and context. Actual argument validation and authorization remain external.

Apply:

\[
p=\operatorname{softmax}(\mathrm{logits}/T)
\]

Fit scalar \(T>0\) on held-out calibration data and store it as nontrainable metadata. Predict argmax; ties select the lowest label index. Before calibration, use \(T=1\) and mark probabilities uncalibrated.

Training uses the same serialization, candidate ordering, registry descriptions, and context budgets as deployment.

## 11. Output Contract

Successful routing produces:

```json
{"decision":"CALL","capability_id":"registered.id"}
```

or:

```json
{"decision":"NO_CALL","capability_id":null}
```

or:

```json
{"decision":"CLARIFY","capability_id":null}
```

`capability_id` must come from the request’s registry snapshot. The model never generates an identifier.

Return decision probabilities and context-reduction information as separate metadata. Reranker scores are not confidence probabilities.

Input, registry, compatibility, or numerical failures produce an explicit error result, not a fabricated routing label. Error codes are:

- `INVALID_INPUT`
- `INPUT_TOO_LONG`
- `INVALID_REGISTRY`
- `INDEX_MISMATCH`
- `NUMERICAL_ERROR`

An external application or worker handles user-facing clarification, answers, argument construction, permissions, and execution.

## 12. Initialization and Weight Tying

| Component | Initialization |
|---|---|
| Embeddings and all linear matrices | Independent zero-mean normal; underlying standard deviation 0.02; rejection-truncated to \([-0.04,0.04]\) |
| LayerNorm scale | 1 |
| LayerNorm bias | 0 |
| Classifier / reranker biases | 0 |
| PAD embedding row | Fixed zero |
| Depth-dependent rescaling | None |

The initializer standard deviation describes the distribution before truncation. Keep the PAD embedding row zero through training, optimizer updates, and export. Its 512 stored entries remain included in parameter counts.

No deployed weight tying. A temporary pretraining MLM decoder may share input embeddings. Report MLM-specific parameters separately and remove them before deployment.

## 13. Precision

| Operation | Precision |
|---|---|
| Training | BF16 preferred |
| Reference inference | BF16 or FP16, with finite-output validation |
| LayerNorm statistics | FP32 |
| Attention softmax | FP32 |
| Pooling sums and L2 norms | FP32 |
| Classifier probabilities | FP32 |
| Retrieval embeddings | FP32 |
| Reranker scores | FP32 |
| Quantized deployment target | Validated INT8 |

## 14. Evaluation Contract

Evaluate the complete routing pipeline and each head separately.

Required evaluation coverage:

- Correct capability selection across agents, tools, and skills.
- Appropriate `NO_CALL` and `CLARIFY` decisions.
- Unnecessary calls and unsupported requests.
- Similar capabilities with overlapping descriptions.
- Missing required inputs and contextual ambiguity.
- Capabilities added after model training.
- Registry additions, removals, description updates, and empty registries.
- Description and request text containing misleading instructions.
- Context reduction, input limits, and numerical failures.

Hold out complete capability identities during evaluation of adaptability. No-call restraint and correct clarification remain separate metrics from capability ranking.

Report shortlist Recall@8, retrieval Recall@10 and nDCG@10, reranker nDCG@10 and MRR@10, decision macro-F1, CALL recall, false CALL rate, calibration, and end-to-end exact routing accuracy.

For end-to-end exact routing accuracy, a `CALL` is correct only when both the decision and capability ID are correct. Ranking metrics use the available list length when fewer than ten candidates exist; the operational shortlist remains eight.

## 15. INT8 Acceptance Criteria

Each artifact records per-operator weight and activation precision, quantization axes and scales, excluded operators, runtime/version, and hardware. Keep LayerNorm, softmax, pooling, normalization, and task heads in floating point.

Compare the same checkpoint against its floating-point reference using identical requests, registry snapshots, tokenizer, formatting, and classifier calibration. Build each variant’s registry embeddings using its corresponding encoder.

Freeze evaluation datasets and metrics before testing. Quantization calibration data must be disjoint from evaluation data.

| Gate | Acceptance threshold |
|---|---|
| Shortlist Recall@8 | Decrease ≤0.01 absolute |
| Retrieval Recall@10 / nDCG@10 | Each decrease ≤0.01 absolute |
| Reranker nDCG@10 / MRR@10 | Each decrease ≤0.01 absolute |
| Decision macro-F1 / CALL recall | Each decrease ≤0.01 absolute |
| End-to-end exact routing accuracy | Decrease ≤0.01 absolute |
| False CALL rate among non-CALL examples | Increase ≤0.005 absolute |
| Calibration | ECE with 15 equal-width bins increases ≤0.01 |
| Numerical validity | No NaN/Inf, invalid embeddings, or input-contract changes |
| Warm latency | p50 and p95 each ≤1.05× reference |
| Model artifact size | ≤0.65× reference |
| Peak inference memory | ≤reference |

Quality metrics use the \([0,1]\) scale. Apply gates to overall results and prespecified critical slices, including held-out capabilities.

Benchmark each inference mode on each target device/backend at batch sizes **1 and 8**, padded sequence lengths **128, 512, and 1,024**, identical threading and device settings, at least **20 warmups**, and **200 timed runs** per case.

Also benchmark the complete routing pipeline with identical registry snapshots and request sets. Record registry size, shortlist size, encoder passes, cache state, and peak memory. Include tokenization, formatting, registry search, and postprocessing; exclude model loading and offline registry indexing.

Targets that fail retain floating-point deployment.