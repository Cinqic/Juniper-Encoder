"""Frozen Proposal B constants shared by every implementation path."""

VOCAB_SIZE = 16_384
STRUCTURAL_IDS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "[QUERY]": 5,
    "[DOCUMENT]": 6,
}
BYTE_OFFSET = 7
BYTE_TOKEN_MIN = BYTE_OFFSET
BYTE_TOKEN_MAX = BYTE_OFFSET + 255
FIRST_MERGE_ID = 263
MERGE_COUNT = 16_121
LAST_TOKEN_ID = VOCAB_SIZE - 1
ORDINARY_TOKEN_MIN = BYTE_TOKEN_MIN
ORDINARY_TOKEN_MAX = LAST_TOKEN_ID
MAX_ORDINARY_REQUEST_TOKENS = 256
MAX_ORDINARY_PAIR_CONTENT_TOKENS = 1_019
MAX_ORDINARY_DECISION_CONTENT_TOKENS = 1_021
HIDDEN_WIDTH = 512
BLOCKS = 12
Q_HEADS = 8
KV_HEADS = 8
HEAD_DIM = 64
GEGLU_WIDTH = 1_536
ROPE_THETA = 10_000.0
ROPE_SCALE = 1.0 / 8.0
DROPOUT_P = 0.1
LN_EPS = 1e-5
INIT_STD = 0.02
INIT_ABS_BOUND = 0.04
PROJECTION_NORM_EPS = 1e-12
RETRIEVAL_DIM = 256
CONTRASTIVE_TEMPERATURE = 0.05
CLASS_COUNT = 3
CLASS_CALL = 0
CLASS_NO_CALL = 1
CLASS_CLARIFY = 2
CLASS_NAMES = ("CALL", "NO_CALL", "CLARIFY")

PARAMETER_COUNTS = {
    "embedding_table": 8_388_608,
    "attention_per_block": 1_048_576,
    "geglu_per_block": 2_359_296,
    "layer_norms_per_block": 2_048,
    "complete_block": 3_409_920,
    "twelve_blocks": 40_919_040,
    "final_layer_norm": 1_024,
    "shared_backbone": 49_308_672,
    "retrieval_head": 131_072,
    "decision_classifier": 1_539,
    "reranker": 513,
    "retrieval_participating": 49_439_744,
    "decision_participating": 49_310_211,
    "reranking_participating": 49_309_185,
    "fast_pipeline_unique": 49_441_283,
    "reranked_pipeline_deployed_total": 49_441_796,
}

FROZEN_ERROR_CODES = (
    "INVALID_INPUT",
    "INPUT_TOO_LONG",
    "INVALID_REGISTRY",
    "INDEX_MISMATCH",
    "NUMERICAL_ERROR",
)

# This is an implementation convention, not a new routing policy.
ERROR_PRECEDENCE = FROZEN_ERROR_CODES
UNICODE_DATABASE_VERSION = "python-unicode-database"
