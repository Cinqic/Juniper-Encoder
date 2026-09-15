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


def test_partial_build_describes_its_actual_vocabulary():
    tokenizer = RawByteBPE.train(["tiny"], config=TokenizerConfig(), require_full=False)
    assert tokenizer.config.merge_count == len(tokenizer.merges)
    assert tokenizer.config.vocab_size == 263 + len(tokenizer.merges)


def test_bpe_overlap_replacement_is_left_to_right_and_non_overlapping():
    config = TokenizerConfig(merge_count=1, vocab_size=264)
    tokenizer = RawByteBPE.train(["aaa"], config=config, require_full=False)
    assert tokenizer.merges[0].left == 7 + ord("a")
    assert tokenizer.merges[0].right == 7 + ord("a")
    assert tokenizer.encode("aaa") == [263, 7 + ord("a")]


def test_incomplete_deployment_vocabulary_is_a_blocker():
    config = TokenizerConfig()
    with pytest.raises(EncoderError) as error:
        RawByteBPE.train(["tiny"], config=config, require_full=True)
    assert error.value.code == "BLOCKED_TOKENIZER_CORPUS"
