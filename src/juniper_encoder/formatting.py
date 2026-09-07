"""Strict registry/request validation and deterministic routing templates."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import (
    CLASS_NAMES,
    MAX_ORDINARY_DECISION_CONTENT_TOKENS,
    MAX_ORDINARY_PAIR_CONTENT_TOKENS,
    MAX_ORDINARY_REQUEST_TOKENS,
    STRUCTURAL_IDS,
    UNICODE_DATABASE_VERSION,
)
from .errors import EncoderError, input_too_long, invalid_input, invalid_registry
from .tokenizer import RawByteBPE
from .utils import canonical_json_bytes, sha256_bytes, strict_json_loads


REGISTRY_FIELDS = (
    "id",
    "type",
    "name",
    "purpose",
    "supported_requests",
    "exclusions",
    "required_inputs",
)
ALLOWED_TYPES = frozenset(("agent", "tool", "skill"))
CAPABILITY_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclasses.dataclass(frozen=True)
class RegistryRecord:
    id: str
    type: str
    name: str
    purpose: str
    supported_requests: tuple[str, ...]
    exclusions: tuple[str, ...]
    required_inputs: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RegistryRecord":
        if not isinstance(value, Mapping):
            raise invalid_registry("capability record must be an object")
        if tuple(value.keys()) != REGISTRY_FIELDS:
            raise invalid_registry("capability record must contain exactly the seven fields in frozen order", fields=list(value.keys()))
        for field in REGISTRY_FIELDS[:4]:
            if not isinstance(value[field], str) or not value[field]:
                raise invalid_registry(f"record field {field} must be a nonempty string")
        capability_id = value["id"]
        if not capability_id.isascii() or not CAPABILITY_ID_RE.fullmatch(capability_id):
            raise invalid_registry("capability id must be nonempty ASCII matching [A-Za-z0-9._-]+", id=capability_id)
        if value["type"] not in ALLOWED_TYPES:
            raise invalid_registry("record type is not allowed", type=value["type"])
        arrays: dict[str, tuple[str, ...]] = {}
        for field in REGISTRY_FIELDS[4:]:
            raw = value[field]
            if not isinstance(raw, list) or any(not isinstance(item, str) or not item for item in raw):
                raise invalid_registry(f"record field {field} must be an array of nonempty strings")
            arrays[field] = tuple(raw)
        return cls(
            id=capability_id,
            type=value["type"],
            name=value["name"],
            purpose=value["purpose"],
            **arrays,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "purpose": self.purpose,
            "supported_requests": list(self.supported_requests),
            "exclusions": list(self.exclusions),
            "required_inputs": list(self.required_inputs),
        }

    def serialized(self) -> bytes:
        return canonical_json_bytes(self.as_mapping())

    def description_hash(self) -> str:
        return sha256_bytes(self.serialized())


def parse_registry_json(raw: str | bytes) -> list[RegistryRecord]:
    try:
        value = strict_json_loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise invalid_registry("registry is not strict JSON", reason=str(exc)) from exc
    if not isinstance(value, list):
        raise invalid_registry("registry must be an array")
    records = [RegistryRecord.from_mapping(item) for item in value]
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise invalid_registry("capability IDs must be unique and case-sensitive")
    return records


@dataclasses.dataclass(frozen=True)
class HistoryEntry:
    role: str
    content: str

    def as_mapping(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclasses.dataclass(frozen=True)
class FormattedRequest:
    history: tuple[HistoryEntry, ...]
    user: str
    serialized: bytes
    dropped_history_count: int
    token_ids: tuple[int, ...]


def _validate_request_shape(request: Mapping[str, Any]) -> tuple[list[HistoryEntry], str]:
    if not isinstance(request, Mapping) or tuple(request.keys()) != ("history", "user"):
        raise invalid_input("request must contain exactly history and user in frozen order")
    history = request["history"]
    user = request["user"]
    if not isinstance(history, list):
        raise invalid_input("history must be an array")
    if not isinstance(user, str) or not user or user.isspace():
        raise invalid_input("user must not be empty or Unicode-whitespace-only", unicode_database=UNICODE_DATABASE_VERSION)
    entries: list[HistoryEntry] = []
    for item in history:
        if not isinstance(item, Mapping) or tuple(item.keys()) != ("role", "content"):
            raise invalid_input("each history entry must contain role and content in frozen order")
        if item["role"] not in {"user", "assistant", "tool"}:
            raise invalid_input("history role must be one of user, assistant, or tool")
        if not isinstance(item["content"], str):
            raise invalid_input("history content must be a string")
        entries.append(HistoryEntry(item["role"], item["content"]))
    return entries, user


def serialize_request(history: Sequence[HistoryEntry], user: str) -> bytes:
    return canonical_json_bytes({
        "history": [entry.as_mapping() for entry in history],
        "user": user,
    })


def format_request(request: Mapping[str, Any], tokenizer: RawByteBPE) -> FormattedRequest:
    history, user = _validate_request_shape(request)
    current = serialize_request([], user)
    user_token_count = len(tokenizer.encode(current.decode("utf-8")))
    if user_token_count > MAX_ORDINARY_REQUEST_TOKENS:
        raise input_too_long("serialized user-alone request exceeds the ordinary-token budget", token_count=user_token_count)
    retained = list(history)
    dropped = 0
    serialized = serialize_request(retained, user)
    while len(tokenizer.encode(serialized.decode("utf-8"))) > MAX_ORDINARY_REQUEST_TOKENS:
        if not retained:
            raise input_too_long("serialized request exceeds the ordinary-token budget")
        retained.pop(0)
        dropped += 1
        serialized = serialize_request(retained, user)
    token_ids = tuple(tokenizer.encode(serialized.decode("utf-8")))
    return FormattedRequest(tuple(retained), user, serialized, dropped, token_ids)


def _structural(name: str) -> int:
    return STRUCTURAL_IDS[name]


@dataclasses.dataclass(frozen=True)
class PairInput:
    token_ids: tuple[int, ...]
    query_tokens: tuple[int, ...]
    document_tokens: tuple[int, ...]
    content_tokens: int


def format_generic_pair(
    query_serialized: bytes,
    record: RegistryRecord,
    tokenizer: RawByteBPE,
    *,
    allow_document_truncation: bool = True,
) -> PairInput:
    try:
        query_text = query_serialized.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise invalid_input("request bytes are not valid UTF-8", reason=str(exc)) from exc
    query_tokens = tokenizer.encode(query_text)
    if len(query_tokens) > MAX_ORDINARY_REQUEST_TOKENS:
        raise input_too_long("serialized query exceeds the query-token budget", token_count=len(query_tokens))
    document_tokens = tokenizer.encode(record.serialized().decode("utf-8"))
    remaining = MAX_ORDINARY_PAIR_CONTENT_TOKENS - len(query_tokens)
    if remaining < 0:
        raise input_too_long("query leaves no capacity for a generic pair")
    if len(document_tokens) > remaining and not allow_document_truncation:
        raise input_too_long("registered capability record cannot be truncated in routing", capability_id=record.id)
    document_tokens = document_tokens[:remaining]
    token_ids = (
        _structural("[CLS]"), _structural("[QUERY]"), *query_tokens,
        _structural("[SEP]"), _structural("[DOCUMENT]"), *document_tokens,
        _structural("[SEP]"),
    )
    return PairInput(tuple(token_ids), tuple(query_tokens), tuple(document_tokens), len(query_tokens) + len(document_tokens))


@dataclasses.dataclass(frozen=True)
class DecisionInput:
    token_ids: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    content_tokens: int


def format_decision(
    query_serialized: bytes,
    candidates: Sequence[RegistryRecord],
    tokenizer: RawByteBPE,
) -> DecisionInput:
    if not 1 <= len(candidates) <= 2:
        raise invalid_input("decision classifier accepts one or two candidates")
    try:
        request_value = json.loads(query_serialized.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise invalid_input("request bytes are not valid UTF-8 JSON", reason=str(exc)) from exc
    if not isinstance(request_value, dict) or tuple(request_value.keys()) != ("history", "user"):
        raise invalid_input("decision request must be the validated compact request object")

    def serialize(records: Sequence[RegistryRecord]) -> bytes:
        return canonical_json_bytes({"request": request_value, "capabilities": [record.as_mapping() for record in records]})

    selected = list(candidates)
    serialized = serialize(selected)
    content_tokens = tokenizer.encode(serialized.decode("utf-8"))
    if len(content_tokens) > MAX_ORDINARY_DECISION_CONTENT_TOKENS and len(selected) == 2:
        selected = selected[:1]
        serialized = serialize(selected)
        content_tokens = tokenizer.encode(serialized.decode("utf-8"))
    if len(content_tokens) > MAX_ORDINARY_DECISION_CONTENT_TOKENS:
        raise input_too_long("decision context exceeds the ordinary-token budget", content_tokens=len(content_tokens))
    token_ids = (_structural("[CLS]"), _structural("[QUERY]"), *content_tokens, _structural("[SEP]"))
    return DecisionInput(tuple(token_ids), tuple(record.id for record in selected), len(content_tokens))


def validate_decision(decision: Mapping[str, Any], allowed_ids: set[str]) -> None:
    if tuple(decision.keys()) != ("decision", "capability_id"):
        raise invalid_input("decision must contain decision and capability_id in frozen order")
    if decision["decision"] not in CLASS_NAMES:
        raise invalid_input("unknown decision class")
    capability_id = decision["capability_id"]
    if decision["decision"] == "CALL":
        if not isinstance(capability_id, str) or capability_id not in allowed_ids:
            raise invalid_input("CALL capability_id is not present in the current registry")
    elif capability_id is not None:
        raise invalid_input("non-CALL decisions must have a null capability_id")
