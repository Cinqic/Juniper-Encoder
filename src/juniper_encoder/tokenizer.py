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
        byte_map = {i: _token_bytes(i) for i in range(BYTE_OFFSET, BYTE_OFFSET + 256)}
        byte_representations = set(byte_map.values())
        merges: list[Merge] = []
        next_id = FIRST_MERGE_ID

        # Maintain each document as a linked list of token nodes. Pair counts
        # and occurrence sets are updated only around a replacement, while a
        # heap supplies the same ``max frequency, numeric pair`` ordering as
        # the straightforward reference implementation. ``position`` never
        # changes, so sorting occurrences preserves the required left-to-right
        # non-overlapping replacement order even after earlier merges remove
        # nodes from the list.
        token: list[int] = []
        next_node: list[int] = []
        previous_node: list[int] = []
        active: list[bool] = []
        position: list[int] = []
        document_heads: list[int] = []
        document_tails: list[int] = []
        for data in documents_bytes:
            if not data:
                continue
            head = len(token)
            for index, value in enumerate(data):
                token.append(BYTE_OFFSET + value)
                next_node.append(head + index + 1 if index + 1 < len(data) else -1)
                previous_node.append(head + index - 1 if index else -1)
                active.append(True)
                position.append(index)
            document_heads.append(head)
            document_tails.append(head + len(data) - 1)

        frequencies: collections.Counter[tuple[int, int]] = collections.Counter()
        occurrences: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
        for head in document_heads:
            current = head
            while current != -1:
                right = next_node[current]
                if right != -1:
                    pair = (token[current], token[right])
                    frequencies[pair] += 1
                    occurrences[pair].add(current)
                current = right

        import heapq

        heap: list[tuple[int, int, int]] = [(-count, left, right) for (left, right), count in frequencies.items()]
        heapq.heapify(heap)

        def remove_edge(left_node: int, right_node: int) -> None:
            if left_node == -1 or right_node == -1:
                return
            pair = (token[left_node], token[right_node])
            frequencies[pair] -= 1
            occurrences[pair].discard(left_node)
            heapq.heappush(heap, (-frequencies[pair], pair[0], pair[1]))

        def add_edge(left_node: int, right_node: int) -> None:
            if left_node == -1 or right_node == -1:
                return
            pair = (token[left_node], token[right_node])
            frequencies[pair] += 1
            occurrences[pair].add(left_node)
            heapq.heappush(heap, (-frequencies[pair], pair[0], pair[1]))

        while len(merges) < config.merge_count:
            pair: tuple[int, int] | None = None
            while heap:
                neg_count, left, right = heapq.heappop(heap)
                current_count = frequencies[(left, right)]
                if -neg_count == current_count and current_count > 0:
                    pair = (left, right)
                    break
            if pair is None:
                break
            left, right = pair
            merged_bytes = byte_map[left] + byte_map[right]
            if merged_bytes in byte_representations:
                raise ValueError("training would create a duplicate byte representation")
            merge = Merge(left, right, next_id)
            merges.append(merge)
            byte_map[next_id] = merged_bytes
            byte_representations.add(merged_bytes)

            selected_nodes: list[int] = []
            for node in sorted(occurrences[pair], key=lambda value: (position[value], value)):
                right_node = next_node[node]
                if active[node] and right_node != -1 and active[right_node] and token[node] == left and token[right_node] == right:
                    selected_nodes.append(node)
            for node in selected_nodes:
                right_node = next_node[node]
                if not active[node] or right_node == -1 or not active[right_node] or token[node] != left or token[right_node] != right:
                    continue
                before = previous_node[node]
                after = next_node[right_node]
                remove_edge(before, node)
                remove_edge(node, right_node)
                remove_edge(right_node, after)
                token[node] = next_id
                next_node[node] = after
                if after != -1:
                    previous_node[after] = node
                if before != -1:
                    next_node[before] = node
                active[right_node] = False
                previous_node[right_node] = -1
                next_node[right_node] = -1
                add_edge(before, node)
                add_edge(node, after)
            next_id += 1

        if require_full and len(merges) != config.merge_count:
            raise EncoderError(
                "BLOCKED_TOKENIZER_CORPUS",
                "approved corpus did not yield the complete frozen merge vocabulary",
                {"learned_merges": len(merges), "required_merges": config.merge_count},
            )
        if len(merges) != config.merge_count:
            # A caller may explicitly request a diagnostic partial build. Do
            # not let that artifact claim the frozen deployment vocabulary;
            # its config must describe the actual payload it contains.
            config = dataclasses.replace(config, merge_count=len(merges), vocab_size=FIRST_MERGE_ID + len(merges))
        return cls(merges, config=config)

    def encode(self, text: str) -> list[int]:
        data = _strict_utf8(text)
        if not data:
            return []
        token: list[int] = [BYTE_OFFSET + value for value in data]
        next_node = [index + 1 if index + 1 < len(token) else -1 for index in range(len(token))]
        previous_node = [index - 1 if index else -1 for index in range(len(token))]
        active = [True] * len(token)
        position = list(range(len(token)))
        pair_to_rank = {(merge.left, merge.right): rank for rank, merge in enumerate(self.merges)}
        occurrences: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
        for node in range(len(token) - 1):
            pair = (token[node], token[node + 1])
            if pair in pair_to_rank:
                occurrences[pair].add(node)

        def remove_edge(left_node: int, right_node: int) -> None:
            if left_node == -1 or right_node == -1:
                return
            pair = (token[left_node], token[right_node])
            if pair in pair_to_rank:
                occurrences[pair].discard(left_node)

        def add_edge(left_node: int, right_node: int) -> None:
            if left_node == -1 or right_node == -1:
                return
            pair = (token[left_node], token[right_node])
            if pair in pair_to_rank:
                occurrences[pair].add(left_node)

        # Proposal B applies the ordered merge history one rank at a time.
        # Newly created pairs whose rank has already passed are intentionally
        # left untouched; this is why a generic "merge lowest current pair"
        # loop would not be equivalent.
        for merge in self.merges:
            pair = (merge.left, merge.right)
            selected_nodes: list[int] = []
            for node in sorted(occurrences[pair], key=lambda value: (position[value], value)):
                right_node = next_node[node]
                if active[node] and right_node != -1 and active[right_node] and token[node] == merge.left and token[right_node] == merge.right:
                    selected_nodes.append(node)
            for node in selected_nodes:
                right_node = next_node[node]
                if not active[node] or right_node == -1 or not active[right_node] or token[node] != merge.left or token[right_node] != merge.right:
                    continue
                before = previous_node[node]
                after = next_node[right_node]
                remove_edge(before, node)
                remove_edge(node, right_node)
                remove_edge(right_node, after)
                token[node] = merge.new_id
                next_node[node] = after
                if after != -1:
                    previous_node[after] = node
                if before != -1:
                    next_node[before] = node
                active[right_node] = False
                previous_node[right_node] = -1
                next_node[right_node] = -1
                add_edge(before, node)
                add_edge(node, after)
        sequence: list[int] = []
        node = 0
        while node != -1:
            if active[node]:
                sequence.append(token[node])
            node = next_node[node]
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
            "tokenizer_identity": self.identity(),
            "vocab_size": len(self._bytes_by_id) + len(STRUCTURAL_IDS),
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
        manifest_path = Path(directory) / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = sha256_bytes(canonical_json_bytes(payload))
            if manifest.get("payload_sha256") != expected or manifest.get("tokenizer_identity") != tokenizer.identity():
                raise ValueError("tokenizer manifest identity does not match tokenizer payload")
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
