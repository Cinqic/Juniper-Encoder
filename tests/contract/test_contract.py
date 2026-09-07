from pathlib import Path

from juniper_encoder.constants import PARAMETER_COUNTS, STRUCTURAL_IDS, VOCAB_SIZE
from juniper_encoder.contract import verify_machine_contract, verify_traceability
from juniper_encoder.utils import sha256_file


ROOT = Path(__file__).parents[2]


def test_machine_contract_and_frozen_hash():
    result = verify_machine_contract(ROOT / "spec/proposal_b.contract.json")
    assert result["valid"] is True
    assert sha256_file(ROOT / "spec/PROPOSAL_B_FROZEN.md") == "9363e892bef02d7b26856c31282bf8ca3902d1a140700e970d808057d6e77d1b"
    assert STRUCTURAL_IDS["[PAD]"] == 0
    assert VOCAB_SIZE == 16384


def test_traceability_has_unique_implemented_rows():
    result = verify_traceability(ROOT / "spec/traceability.csv")
    assert result["status"] == "PASS"
    assert result["rows"] >= 18


def test_parameter_contract_arithmetic_is_explicit():
    assert PARAMETER_COUNTS["shared_backbone"] == PARAMETER_COUNTS["twelve_blocks"] + PARAMETER_COUNTS["embedding_table"] + PARAMETER_COUNTS["final_layer_norm"]
    assert PARAMETER_COUNTS["reranked_pipeline_deployed_total"] == PARAMETER_COUNTS["shared_backbone"] + PARAMETER_COUNTS["retrieval_head"] + PARAMETER_COUNTS["decision_classifier"] + PARAMETER_COUNTS["reranker"]
