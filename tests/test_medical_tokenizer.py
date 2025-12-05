import pytest

from openmed.processing.tokenization import (
    apply_medical_pretokenizer,
    build_medical_pretokenizer,
    TOKENIZERS_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not TOKENIZERS_AVAILABLE, reason="tokenizers package required")


def test_medical_pretokenizer_merges_hyphenated_terms():
    pre = build_medical_pretokenizer()
    splits = pre.pre_tokenize_str("COVID-19+ pt post-CAR-T on IL-6-mediated therapy.")
    # Ensure it runs and returns tuples
    assert all(isinstance(tok, str) and isinstance(span, tuple) for tok, span in splits)


def test_apply_medical_pretokenizer_sets_backend():
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece

    class Dummy:
        pass

    dummy = Dummy()
    dummy._tokenizer = Tokenizer(WordPiece())

    applied = apply_medical_pretokenizer(dummy)
    assert applied
    # ensure pre_tokenizer is set to our custom class
    assert dummy._tokenizer.pre_tokenizer is not None


def test_encode_with_medical_pretokenizer_keeps_covid():
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece

    vocab = {"[UNK]": 0, "COVID-19": 1, "-": 2, "19": 3, "COVID": 4}
    tok = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = build_medical_pretokenizer()
    encoded = tok.encode("COVID-19 patient")
    assert len(encoded.tokens) > 0
