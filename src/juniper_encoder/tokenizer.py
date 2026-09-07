"""Raw UTF-8 byte BPE with a deterministic, lossless artifact format."""

from __future__ import annotations

import base64
import collections
import dataclasses
import re
from pathlib import Path
from typing import Iterable, Sequence

from .constants import (
    BYTE_OFFSET,
    FIRST_MERGE_ID,
    LAST_TOKEN_ID,
    MERGE_COUNT,
    STRUCTURAL_IDS,
    VOCAB_SIZE,
)
from .errors import EncoderError, input_too_long, invalid_input
from .utils import canonical_json_bytes, sha256_bytes, sha256_file, write_json


STRUCTURAL_TEXT = frozenset(STRUCTURAL_IDS)
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _strict_utf8(text: str) -> bytes:
    if not isinstance(text, str):
        raise invalid_input("content must be a Unicode string")
    if _SURROGATE_RE.search(text):
        raise invalid_input("lone UTF-16 surrogate is not a Unicode scalar sequence")
    try:
        return text.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise invalid_input("content is not valid UTF-8", reason=str(exc)) from exc


def _token_bytes(token_id: int) -> bytes:
    if not BYTE_OFFSET <= token_id <= BYTE_OFFSET + 255:
        raise ValueError(f"not a byte token: {token_id}")
    return bytes((token_id - BYTE_OFFSET,))


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    merge_count: int = MERGE_COUNT
    vocab_size: int = VOCAB_SIZE
    implementation_version: str = "raw-byte-bpe-v1"
    unicode_database: str = "python-unicode-database"
    tie_break: str = "numeric-left-right"
    overlap_rule: str = "left-to-right-non-overlapping"

    def validate(self) -> None:
        if self.vocab_size != VOCAB_SIZE or self.merge_count != MERGE_COUNT:
            raise ValueError("deployment tokenizer must use the frozen 16,384 vocabulary and 16,121 merges")


@dataclasses.dataclass(frozen=True)
class Merge:
    left: int
    right: int
    new_id: int

    def as_list(self) -> list[int]:
        return [self.left, self.right, self.new_id]


class RawByteBPE:
    """A tokenizer that never normalizes, pretokenizes, or replaces bytes."""

    def __init__(self, merges: Sequence[Merge], *, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig(merge_count=len(merges), vocab_size=FIRST_MERGE_ID + len(merges))
        if self.config.merge_count != len(merges):
            raise ValueError("config merge_count does not match merges")
        self.merges = tuple(merges)
        self._pair_to_id: dict[tuple[int, int], int] = {}
        self._bytes_by_id: dict[int, bytes] = {i: _token_bytes(i) for i in range(BYTE_OFFSET, BYTE_OFFSET + 256)}
        for merge in self.merges:
            if merge.new_id in self._bytes_by_id:
                raise ValueError(f"duplicate token id {merge.new_id}")
            if merge.new_id != FIRST_MERGE_ID + len(self._pair_to_id):
                raise ValueError("merge IDs must be contiguous in learned order")
            if (merge.left, merge.right) in self._pair_to_id:
                raise ValueError("duplicate merge pair")
            if merge.left not in self._bytes_by_id or merge.right not in self._bytes_by_id:
                raise ValueError("merge history references a token not learned yet")
            merged = self._bytes_by_id[merge.left] + self._bytes_by_id[merge.right]
            if merged in self._bytes_by_id.values():
                raise ValueError("learned token has a duplicate byte representation")
            self._pair_to_id[(merge.left, merge.right)] = merge.new_id
            self._bytes_by_id[merge.new_id] = merged
        if len(self._bytes_by_id) != 256 + len(self.merges):
            raise ValueError("token byte map is not unique")

    @classmethod
    def from_merges(cls, pairs: Sequence[tuple[int, int]], *, toy: bool = False) -> "RawByteBPE":
        merges = [Merge(left, right, FIRST_MERGE_ID + i) for i, (left, right) in enumerate(pairs)]
        if toy:
            config = TokenizerConfig(merge_count=len(merges), vocab_size=FIRST_MERGE_ID + len(merges))
        else:
            config = TokenizerConfig()
        return cls(merges, config=config)

    @classmethod
    def train(
        cls,
        documents: Iterable[str],
        *,
        config: TokenizerConfig | None = None,
        require_full: bool = True,
    ) -> "RawByteBPE":
        config = config or TokenizerConfig()
        documents_bytes = [_strict_utf8(document) for document in documents]
        sequences = [[BYTE_OFFSET + value for value in data] for data in documents_bytes if data]
        byte_map = {i: _token_bytes(i) for i in range(BYTE_OFFSET, BYTE_OFFSET + 256)}
        merges: list[Merge] = []
        next_id = FIRST_MERGE_ID

        while len(merges) < config.merge_count:
            frequencies: collections.Counter[tuple[int, int]] = collections.Counter()
            for sequence in sequences:
                frequencies.update(zip(sequence, sequence[1:]))
            if not frequencies:
                break
            # max frequency, then lexicographically smallest numeric pair.
            pair = min(frequencies, key=lambda candidate: (-frequencies[candidate], candidate[0], candidate[1]))
            left, right = pair
            merged_bytes = byte_map[left] + byte_map[right]
            if merged_bytes in byte_map.values():
                raise ValueError("training would create a duplicate byte representation")
            merge = Merge(left, right, next_id)
            merges.append(merge)
            byte_map[next_id] = merged_bytes
            for index, sequence in enumerate(sequences):
                replaced: list[int] = []
                cursor = 0
                while cursor < len(sequence):
                    if cursor + 1 < len(sequence) and sequence[cursor] == left and sequence[cursor + 1] == right:
                        replaced.append(next_id)
                        cursor += 2
                    else:
                        replaced.append(sequence[cursor])
                        cursor += 1
                sequences[index] = replaced
            next_id += 1

        if require_full and len(merges) != config.merge_count:
            raise EncoderError(
                "BLOCKED_TOKENIZER_CORPUS",
                "approved corpus did not yield the complete frozen merge vocabulary",
                {"learned_merges": len(merges), "required_merges": config.merge_count},
            )
        return cls(merges, config=config)

    def encode(self, text: str) -> list[int]:
        data = _strict_utf8(text)
        if not data:
            return []
        sequence = [BYTE_OFFSET + value for value in data]
        for merge in self.merges:
            replaced: list[int] = []
            cursor = 0
            while cursor < len(sequence):
                if cursor + 1 < len(sequence) and sequence[cursor] == merge.left and sequence[cursor + 1] == merge.right:
                    replaced.append(merge.new_id)
                    cursor += 2
                else:
                    replaced.append(sequence[cursor])
                    cursor += 1
            sequence = replaced
        return sequence

    def decode(self, token_ids: Sequence[int]) -> str:
        try:
            data = b"".join(self._bytes_by_id[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise invalid_input("unknown tokenizer token ID", token_id=exc.args[0]) from exc
        try:
            return data.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise invalid_input("token sequence is not valid UTF-8", reason=str(exc)) from exc

    def token_bytes(self, token_id: int) -> bytes:
        try:
            return self._bytes_by_id[token_id]
        except KeyError as exc:
            raise invalid_input("unknown tokenizer token ID", token_id=token_id) from exc

    def payload(self) -> dict:
        return {
            "config": dataclasses.asdict(self.config),
            "structural_ids": STRUCTURAL_IDS,
            "vocab": {
                str(token_id): base64.b64encode(value).decode("ascii")
                for token_id, value in sorted(self._bytes_by_id.items())
            },
            "merges": [merge.as_list() for merge in self.merges],
        }

    def identity(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))

    def save(self, directory: str | Path, *, source_sha: str | None = None, corpus_hash: str | None = None) -> dict:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        payload = self.payload()
        write_json(destination / "tokenizer.json", payload)
        metadata = {
            "artifact_type": "juniper-tokenizer",
            "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
            "source_sha": source_sha,
            "corpus_hash": corpus_hash,
            "implementation_version": self.config.implementation_version,
        }
        write_json(destination / "manifest.json", metadata)
        return metadata

    @classmethod
    def load(cls, directory: str | Path) -> "RawByteBPE":
        import json

        path = Path(directory) / "tokenizer.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = TokenizerConfig(**payload["config"])
        merges = [Merge(*item) for item in payload["merges"]]
        tokenizer = cls(merges, config=config)
        if tokenizer.payload() != payload:
            raise ValueError("tokenizer payload is not canonical or does not round-trip")
        return tokenizer

    def compare_payload(self, other: "RawByteBPE") -> bool:
        return self.payload() == other.payload()


def ordinary_token_count(tokenizer: RawByteBPE, text: str) -> int:
    return len(tokenizer.encode(text))


def validate_round_trip(tokenizer: RawByteBPE, strings: Iterable[str]) -> None:
    for text in strings:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        if decoded != text:
            raise AssertionError("tokenizer full-string round trip failed")
        if any(token_id in STRUCTURAL_IDS.values() for token_id in token_ids):
            raise AssertionError("ordinary content emitted a structural ID")
