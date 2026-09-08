import pytest

from juniper_encoder.data import audit, deterministic_split, freeze
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


def test_freeze_excludes_training_only_records_before_partitioning(tmp_path):
    manifest = tmp_path / "prepared.json"
    output = tmp_path / "frozen.json"
    manifest.write_text(
        '{"records":['
        '{"record_id":"general","content_sha256":"general-hash","source_role":"general"},'
        '{"record_id":"training","content_sha256":"training-hash","source_role":"training_only"}'
        ']}',
        encoding="utf-8",
    )
    result = freeze(manifest, output, {"train": 1.0})
    assert [record["record_id"] for record in result["records"]] == ["general"]
    assert result["excluded_source_role_counts"] == {"training_only": 1}
    assert result["leakage_audit"]["status"] == "PASS"
