from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from openmed.clinical.nli_premise_window import (
    EvidenceSpan,
    PremiseWindowError,
    PremiseWindowRefusal,
    PremiseWindowStatus,
    select_minimal_premise_window,
    select_premise_window,
)


@dataclass(frozen=True)
class _CitationLike:
    source_start: int
    source_end: int


def test_selects_exact_smallest_window_covering_all_spans() -> None:
    text = "omitted|finding-alpha|gap|finding-beta|omitted"
    alpha_start = text.index("finding-alpha")
    alpha_end = alpha_start + len("finding-alpha")
    beta_start = text.index("finding-beta")
    beta_end = beta_start + len("finding-beta")

    result = select_minimal_premise_window(
        text,
        [
            _CitationLike(beta_start, beta_end),
            EvidenceSpan(alpha_start, alpha_end),
        ],
        max_characters=64,
        deidentified=True,
    )

    assert result.status is PremiseWindowStatus.READY
    assert result.verification_allowed is True
    assert result.window_start == alpha_start
    assert result.window_end == beta_end
    assert result.premise == text[alpha_start:beta_end]
    assert result.character_count == beta_end - alpha_start
    assert result.source_spans == tuple(sorted(result.source_spans))


def test_exact_ceiling_is_allowed_and_alias_matches() -> None:
    kwargs = {
        "text": "prefix|bounded-evidence|suffix",
        "source_spans": [(7, 23)],
        "max_characters": 16,
        "deidentified": True,
    }

    assert select_premise_window(**kwargs) == select_minimal_premise_window(**kwargs)
    assert select_premise_window(**kwargs).premise == "bounded-evidence"


def test_required_window_over_ceiling_refuses_verification_without_text() -> None:
    text = "alpha|unrelated-context|beta"

    result = select_minimal_premise_window(
        text,
        [(0, 5), (24, 28)],
        max_characters=27,
        deidentified=True,
    )

    assert result.status is PremiseWindowStatus.REFUSED
    assert result.refusal_reason is PremiseWindowRefusal.WINDOW_LIMIT_EXCEEDED
    assert result.verification_allowed is False
    assert result.premise is None
    assert result.window_start == 0
    assert result.window_end == 28


def test_non_deidentified_input_is_refused_before_verification() -> None:
    result = select_minimal_premise_window(
        "SENSITIVE_SENTINEL",
        [(0, 9)],
        max_characters=20,
        deidentified=False,
    )

    assert result.refusal_reason is PremiseWindowRefusal.INPUT_NOT_DEIDENTIFIED
    assert result.premise is None
    assert result.window_start is None
    assert result.window_end is None


def test_missing_spans_refuses_verification() -> None:
    result = select_minimal_premise_window(
        "deidentified synthetic note",
        [],
        max_characters=100,
        deidentified=True,
    )

    assert result.refusal_reason is PremiseWindowRefusal.MISSING_SOURCE_SPANS
    assert result.verification_allowed is False


def test_span_shapes_are_canonicalized_and_deduplicated() -> None:
    result = select_minimal_premise_window(
        "0123456789",
        [
            {"start": 4, "end": 6},
            {"source_start": 1, "source_end": 3},
            (4, 6),
        ],
        max_characters=8,
        deidentified=True,
    )

    assert result.source_spans == (EvidenceSpan(1, 3), EvidenceSpan(4, 6))
    assert result.premise == "12345"


@pytest.mark.parametrize(
    "spans",
    [
        [(2, 2)],
        [(-1, 2)],
        [(True, 2)],
        [{"start": 1}],
        [(0, 99)],
        "not-a-span-collection",
    ],
)
def test_invalid_spans_fail_without_echoing_input(spans: object) -> None:
    sentinel = "SENSITIVE_SENTINEL"

    with pytest.raises(PremiseWindowError) as error:
        select_minimal_premise_window(
            sentinel,
            spans,  # type: ignore[arg-type]
            max_characters=32,
            deidentified=True,
        )
    assert sentinel not in str(error.value)


@pytest.mark.parametrize(
    ("max_characters", "deidentified"),
    [(0, True), (-1, True), (True, True), (12, 1)],
)
def test_invalid_configuration_fails_closed(
    max_characters: object,
    deidentified: object,
) -> None:
    with pytest.raises(PremiseWindowError):
        select_minimal_premise_window(
            "synthetic",
            [(0, 3)],
            max_characters=max_characters,  # type: ignore[arg-type]
            deidentified=deidentified,  # type: ignore[arg-type]
        )


def test_reports_and_repr_never_include_premise_text() -> None:
    sentinel = "SENSITIVE_SENTINEL"
    result = select_minimal_premise_window(
        sentinel,
        [(0, len(sentinel))],
        max_characters=64,
        deidentified=True,
    )

    assert result.premise == sentinel
    assert sentinel not in repr(result)
    assert sentinel not in str(result.to_dict())
    assert "premise" not in result.to_dict()


def test_iterable_failures_do_not_echo_upstream_content() -> None:
    sentinel = "SENSITIVE_SENTINEL"

    def broken_spans():
        raise RuntimeError(sentinel)
        yield (0, 1)

    with pytest.raises(
        PremiseWindowError,
        match="invalid evidence span collection",
    ) as error:
        select_minimal_premise_window(
            "synthetic",
            broken_spans(),
            max_characters=10,
            deidentified=True,
        )
    assert sentinel not in str(error.value)


def test_results_and_spans_are_immutable() -> None:
    span = EvidenceSpan(0, 3)
    result = select_minimal_premise_window(
        "abc",
        [span],
        max_characters=3,
        deidentified=True,
    )

    with pytest.raises(FrozenInstanceError):
        span.start = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.status = PremiseWindowStatus.REFUSED  # type: ignore[misc]
