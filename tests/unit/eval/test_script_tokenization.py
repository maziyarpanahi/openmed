"""Script audit negative controls with local synthetic tokenizers."""

import pytest

from openmed.eval.script_tokenization import (
    ScriptTerm,
    ScriptTokenizationLimits,
    TokenizationObservation,
    audit_script_tokenization,
)


def audit(text, offsets, *, ids=None, limits=ScriptTokenizationLimits()):
    return audit_script_tokenization(
        [ScriptTerm("synthetic", text)],
        tokenizer=lambda _text: TokenizationObservation(
            tuple(ids if ids is not None else [1] * len(offsets)), tuple(offsets)
        ),
        model_digest="1" * 64,
        unknown_token_id=0,
        limits=limits,
    )["scripts"]["synthetic"]


def test_whole_extended_grapheme_is_qualified():
    result = audit("क्‍ष", [(0, 4)])
    assert result["verdict"] == "qualified_tokenization"
    assert result["counts"]["grapheme_count"] == 1
    assert result["offset_recovery_rate"] == 1


def test_grapheme_splits_fail_independently_of_full_codepoint_coverage():
    result = audit("a\u0301", [(0, 1), (1, 2)])
    assert result["offset_recovery_rate"] == 1
    assert result["grapheme_split_rate"] == 1
    assert result["reasons"] == ["grapheme_split_limit"]


def test_fragmentation_uses_pinned_limit_boundary():
    limits = ScriptTokenizationLimits(
        max_tokens_per_grapheme=2, max_grapheme_split_rate=1
    )
    assert (
        audit("a\u0301", [(0, 1), (1, 2)], limits=limits)["verdict"]
        == "qualified_tokenization"
    )
    stricter = ScriptTokenizationLimits(
        max_tokens_per_grapheme=1, max_grapheme_split_rate=1
    )
    assert (
        "fragmentation_limit"
        in audit("a\u0301", [(0, 1), (1, 2)], limits=stricter)["reasons"]
    )


def test_unknown_tokens_use_inclusive_limits():
    limits = ScriptTokenizationLimits(max_unknown_rate=0.5)
    assert (
        audit("ab", [(0, 1), (1, 2)], ids=[0, 1], limits=limits)["verdict"]
        == "qualified_tokenization"
    )
    assert (
        "unknown_limit"
        in audit("ab", [(0, 1), (1, 2)], ids=[0, 0], limits=limits)["reasons"]
    )


@pytest.mark.parametrize(
    "offsets", [[(-1, 1)], [(0, 9)], [(0, 0)], [(1, 2), (0, 1)], [(0, 2), (1, 2)]]
)
def test_invalid_and_overlapping_offsets_cannot_qualify(offsets):
    result = audit("ab", offsets)
    assert "invalid_offsets" in result["reasons"]
    assert result["verdict"] == "unsupported"


def test_missing_offsets_reduce_recovery_even_without_unknown_tokens():
    result = audit("abc", [(0, 1)])
    assert result["unknown_rate"] == 0
    assert result["offset_recovery_rate"] == pytest.approx(1 / 3)
    assert "offset_recovery_limit" in result["reasons"]


def test_empty_tokenizer_output_is_never_perfect():
    result = audit("abc", [])
    assert result["verdict"] == "unsupported"
    assert "empty_tokenization" in result["reasons"]
    assert result["offset_recovery_rate"] == 0


def test_report_does_not_include_source_text_or_token_identities():
    report = audit_script_tokenization(
        [ScriptTerm("latin", "SYNTHETICWORD")],
        tokenizer=lambda _text: TokenizationObservation((8675309,), ((0, 13),)),
        model_digest="1" * 64,
        unknown_token_id=0,
    )
    assert "SYNTHETICWORD" not in str(report)
    assert "8675309" not in str(report)
    assert report["limits"]["max_unknown_rate"] == 0.01


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_invalid_limits_fail(value):
    with pytest.raises(ValueError):
        ScriptTokenizationLimits(max_unknown_rate=value)


def test_non_synthetic_and_empty_terms_fail():
    with pytest.raises(ValueError):
        ScriptTerm("latin", "word", synthetic=False)
    with pytest.raises(ValueError):
        ScriptTerm("latin", " ")


def test_offsets_require_integer_coordinates_and_matching_lengths():
    with pytest.raises(ValueError):
        TokenizationObservation((1,), ((True, 1),))
    with pytest.raises(ValueError):
        TokenizationObservation((1,), ())


def test_report_is_input_order_invariant():
    terms = [
        ScriptTerm("latin", "abc"),
        ScriptTerm("latin", "def"),
        ScriptTerm("other", "a"),
    ]
    options = dict(
        tokenizer=lambda text: TokenizationObservation((1,), ((0, len(text)),)),
        model_digest="1" * 64,
        unknown_token_id=0,
    )
    assert audit_script_tokenization(terms, **options) == audit_script_tokenization(
        terms[::-1], **options
    )


def test_adapter_failure_does_not_echo_source_or_exception_payload():
    def failed_adapter(_text):
        raise RuntimeError("SYNTHETIC_SOURCE_CONTENT")

    with pytest.raises(ValueError, match="^local tokenization adapter failed$"):
        audit_script_tokenization(
            [ScriptTerm("latin", "synthetic")],
            tokenizer=failed_adapter,
            model_digest="1" * 64,
            unknown_token_id=0,
        )


@pytest.mark.parametrize(
    "script,text",
    [
        ("devanagari", "क्‍ष"),
        ("arabic", "سُ"),
        ("latin", "a\u0301"),
        ("mixed", "אa"),
    ],
)
def test_script_specific_whole_grapheme_vectors(script, text):
    from openmed.core.decoding.spans import iter_grapheme_clusters

    offsets = tuple(iter_grapheme_clusters(text))
    report = audit_script_tokenization(
        [ScriptTerm(script, text)],
        tokenizer=lambda _text: TokenizationObservation((1,) * len(offsets), offsets),
        model_digest="1" * 64,
        unknown_token_id=0,
    )
    assert report["scripts"][script]["verdict"] == "qualified_tokenization"


def test_observation_does_not_materialize_unbounded_generators():
    def forbidden():
        raise AssertionError("iterator must not be consumed")
        yield 0

    with pytest.raises(ValueError):
        TokenizationObservation(forbidden(), ())
    with pytest.raises(ValueError):
        TokenizationObservation((), forbidden())
    with pytest.raises(ValueError):
        TokenizationObservation((1,), (forbidden(),))
