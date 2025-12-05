import pytest

from openmed.processing.tokenization import (
    MedicalPreTokenizer,
    apply_medical_pretokenizer,
    build_medical_pretokenizer,
    TOKENIZERS_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not TOKENIZERS_AVAILABLE, reason="tokenizers package required")


def test_medical_pretokenizer_merges_hyphenated_terms():
    pre = MedicalPreTokenizer()
    text = "COVID-19+ pt post-CAR-T on IL-6-mediated therapy."
    tokens = pre.pre_tokenize_str(text)
    surface = [t for t, _ in tokens]
    assert "COVID-19" in surface
    assert "post-CAR-T" in surface
    assert "IL-6-mediated" in surface


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
