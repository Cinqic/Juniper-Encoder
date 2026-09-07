import pytest

from juniper_encoder.data import audit, deterministic_split
from juniper_encoder.errors import EncoderError


def test_split_is_repeatable_and_leakage_is_visible():
    records = [{"record_id": f"r{index}", "content_sha256": f"h{index}"} for index in range(20)]
    first = deterministic_split(records, {"train": 0.8, "development": 0.1, "calibration": 0.05, "test": 0.05})
    second = deterministic_split(records, {"train": 0.8, "development": 0.1, "calibration": 0.05, "test": 0.05})
    assert first == second
    assert audit(first)["status"] == "PASS"
    leaked = first + [{"record_id": "leaked", "content_sha256": first[0]["content_sha256"], "split": "test" if first[0]["split"] != "test" else "train"}]
    assert audit(leaked)["status"] == "FAIL"


def test_split_assignment_is_independent_of_input_order():
    records = [{"record_id": f"r{index}", "content_sha256": f"h{index}"} for index in range(20)]
    ratios = {"train": 0.8, "development": 0.1, "calibration": 0.05, "test": 0.05}
    first = {record["record_id"]: record["split"] for record in deterministic_split(records, ratios)}
    second = {record["record_id"]: record["split"] for record in deterministic_split(list(reversed(records)), ratios)}
    assert first == second


def test_split_requires_immutable_record_ids():
    with pytest.raises(ValueError):
        deterministic_split([{"text": "missing id"}], {"train": 1.0})
