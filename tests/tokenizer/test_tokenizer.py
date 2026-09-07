import pytest

from juniper_encoder.errors import EncoderError
from juniper_encoder.tokenizer import RawByteBPE, TokenizerConfig


def test_all_byte_ids_are_lossless_tokens():
    tokenizer = RawByteBPE.from_merges([], toy=True)
    assert [tokenizer.token_bytes(7 + value) for value in range(256)] == [bytes([value]) for value in range(256)]
    assert tokenizer.encode("[PAD]") == [7 + value for value in b"[PAD]"]


def test_unicode_and_literal_structural_strings_round_trip():
    tokenizer = RawByteBPE.from_merges([], toy=True)
    for text in ["hello", "e\u0301", "🙂", "\x00", "[PAD]", "\u2028", "\u05e9\u05dc\u05d5\u05dd"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_surrogates_are_rejected_without_replacement():
    tokenizer = RawByteBPE.from_merges([], toy=True)
    with pytest.raises(EncoderError) as error:
        tokenizer.encode("\ud800")
    assert error.value.code == "INVALID_INPUT"


def test_deterministic_tie_break_and_repeat_build():
    config = TokenizerConfig(merge_count=2, vocab_size=265)
    first = RawByteBPE.train(["abab", "cdcd"], config=config, require_full=False)
    second = RawByteBPE.train(["abab", "cdcd"], config=config, require_full=False)
    assert first.payload() == second.payload()
    assert first.merges[0].left < first.merges[0].right or first.merges[0].left == first.merges[0].right


def test_incomplete_deployment_vocabulary_is_a_blocker():
    config = TokenizerConfig()
    with pytest.raises(EncoderError) as error:
        RawByteBPE.train(["tiny"], config=config, require_full=True)
    assert error.value.code == "BLOCKED_TOKENIZER_CORPUS"
