import pytest

from juniper_encoder.errors import EncoderError
from juniper_encoder.formatting import RegistryRecord, format_request, parse_registry_json
from juniper_encoder.registry import RegistryIndex, RegistrySnapshot
from juniper_encoder.routing import Calibration, RoutingCallbacks, route
from juniper_encoder.tokenizer import RawByteBPE


def record(capability_id: str) -> RegistryRecord:
    return RegistryRecord.from_mapping({"id": capability_id, "type": "tool", "name": capability_id, "purpose": capability_id, "supported_requests": [capability_id], "exclusions": [], "required_inputs": []})


def test_registry_exact_fields_and_duplicate_json_keys():
    with pytest.raises(EncoderError) as error:
        RegistryRecord.from_mapping({"id": "x", "type": "tool"})
    assert error.value.code == "INVALID_REGISTRY"
    with pytest.raises(EncoderError) as error:
        parse_registry_json(b'[{"id":"x","id":"y"}]')
    assert error.value.code == "INVALID_REGISTRY"


def test_ascii_tie_break_and_index_identity():
    tokenizer = RawByteBPE.from_merges([], toy=True)
    snapshot = RegistrySnapshot.create([record("z"), record("a")], tokenizer_identity=tokenizer.identity(), encoder_variant_identity="fixture")
    index = RegistryIndex.build(snapshot, lambda _record: [1.0] + [0.0] * 255)
    ranked = index.search([1.0] + [0.0] * 255, snapshot)
    assert [item[0].id for item in ranked] == ["a", "z"]


def test_request_budget_drops_oldest_complete_entries():
    tokenizer = RawByteBPE.from_merges([], toy=True)
    request = {"history": [{"role": "user", "content": "x" * 80} for _ in range(5)], "user": "hello"}
    formatted = format_request(request, tokenizer)
    assert formatted.dropped_history_count > 0
    assert formatted.history[-1].content == "x" * 80


def test_empty_registry_validates_then_returns_no_call():
    tokenizer = RawByteBPE.from_merges([], toy=True)
    snapshot = RegistrySnapshot.create([], tokenizer_identity=tokenizer.identity(), encoder_variant_identity="fixture")
    index = RegistryIndex.build(snapshot, lambda _record: [1.0] + [0.0] * 255)
    callbacks = RoutingCallbacks(lambda _request: [1.0] * 256, lambda _tokens, _record: 0.0, lambda _tokens: [0.0, 1.0, 0.0])
    result = route({"history": [], "user": "hello"}, snapshot, index, tokenizer, callbacks)
    assert result.decision_json() == {"decision": "NO_CALL", "capability_id": None}
    assert result.metadata["neural_passes"] == 0


def test_conditional_path_reranks_when_thresholds_do_not_pass():
    tokenizer = RawByteBPE.from_merges([], toy=True)
    records = [record("alpha"), record("beta")]
    snapshot = RegistrySnapshot.create(records, tokenizer_identity=tokenizer.identity(), encoder_variant_identity="fixture")
    index = RegistryIndex.build(snapshot, lambda item: [1.0 if item.id == "alpha" else 0.0, 1.0 if item.id == "beta" else 0.0] + [0.0] * 254)
    counts = {"query": 0, "rerank": 0, "classify": 0}
    callbacks = RoutingCallbacks(
        lambda _request: counts.__setitem__("query", counts["query"] + 1) or [1.0] + [0.0] * 255,
        lambda _tokens, _record: counts.__setitem__("rerank", counts["rerank"] + 1) or 0.0,
        lambda _tokens: counts.__setitem__("classify", counts["classify"] + 1) or [1.0, 0.0, 0.0],
    )
    result = route({"history": [], "user": "hello"}, snapshot, index, tokenizer, callbacks)
    assert result.decision == "CALL"
    assert result.metadata["path"] == "conditional"
    assert counts["rerank"] == 2


def test_positive_temperature_does_not_change_argmax():
    tokenizer = RawByteBPE.from_merges([], toy=True)
    records = [record("alpha")]
    snapshot = RegistrySnapshot.create(records, tokenizer_identity=tokenizer.identity(), encoder_variant_identity="fixture")
    index = RegistryIndex.build(snapshot, lambda _item: [1.0] + [0.0] * 255)
    callbacks = RoutingCallbacks(lambda _request: [1.0] + [0.0] * 255, lambda _tokens, _record: 0.0, lambda _tokens: [0.1, 0.2, 0.0])
    assert route({"history": [], "user": "hello"}, snapshot, index, tokenizer, callbacks, calibration=Calibration(temperature=20.0)).decision == "NO_CALL"
