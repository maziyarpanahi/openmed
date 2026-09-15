"""Focused tests for deterministic, value-free citation minimality checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.citation_minimality import (
    CITATION_MINIMALITY_DISCLAIMER,
    CITATION_MINIMALITY_SCHEMA_VERSION,
    AtomicClaim,
    CitationMinimalityError,
    CitationMinimalityReport,
    CitationMinimalityStatus,
    CitationSpan,
    ClaimCitation,
    check_citation_minimality,
    export_citation_minimality,
)


def _reference(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _span(source: str, value: str) -> CitationSpan:
    start = source.index(value)
    return CitationSpan(start, start + len(value))


def _claim(source: str, value: str = "beta") -> AtomicClaim:
    return AtomicClaim(_reference("claim-1"), _span(source, value))


def test_exact_citation_is_minimal_and_value_free() -> None:
    source = "alpha beta gamma delta"
    claim = _claim(source)
    citation = ClaimCitation(
        claim_id=claim.claim_id,
        source_span=claim.required_span,
        citation_id=_reference("citation-1"),
    )

    report = check_citation_minimality(source, [claim], [citation])
    record = report.records[0]

    assert report.claim_count == 1
    assert report.citation_count == 1
    assert report.flagged_count == 0
    assert record.status is CitationMinimalityStatus.MINIMAL
    assert record.is_minimal is True
    assert record.flagged is False
    assert record.citation_token_count == 1
    assert record.required_token_count == 1
    assert record.excess_token_count == 0
    assert record.excess_context_spans == ()
    assert record.to_dict()["citation_offset"] == {
        "start": claim.required_span.start,
        "end": claim.required_span.end,
    }
    assert CITATION_MINIMALITY_DISCLAIMER in report.to_json()
    assert source not in report.to_json()


def test_extra_context_is_flagged_with_offsets_and_token_counts() -> None:
    source = "alpha beta gamma delta"
    claim = _claim(source)
    citation_span = CitationSpan(source.index("beta"), source.index("delta"))
    citation = ClaimCitation(claim.claim_id, citation_span, _reference("citation-2"))

    report = check_citation_minimality(source, [claim], [citation])
    record = report.records[0]

    assert record.status is CitationMinimalityStatus.EXCESS_CONTEXT
    assert record.review_required is True
    assert record.citation_token_count == 2
    assert record.required_token_count == 1
    assert record.excess_token_count == 1
    assert [span.source_offset for span in record.excess_context_spans] == [
        (claim.required_span.end, citation_span.end)
    ]
    assert record.to_dict()["excess_context_offsets"] == [
        {"start": claim.required_span.end, "end": citation_span.end}
    ]


def test_context_budget_is_explicit_and_deterministic() -> None:
    source = "alpha beta gamma delta"
    claim = _claim(source)
    citation = ClaimCitation(
        claim.claim_id,
        CitationSpan(source.index("beta"), source.index("delta")),
        _reference("citation-3"),
    )

    within_budget = check_citation_minimality(
        source,
        (claim,),
        (citation,),
        max_excess_tokens=1,
    )
    over_budget = check_citation_minimality(source, [claim], [citation])

    assert within_budget.records[0].status is CitationMinimalityStatus.MINIMAL
    assert within_budget.records[0].excess_token_count == 1
    assert within_budget.flagged_count == 0
    assert over_budget.records[0].status is CitationMinimalityStatus.EXCESS_CONTEXT
    assert within_budget.max_excess_tokens == 1


def test_citation_missing_required_span_is_flagged_fail_closed() -> None:
    source = "alpha beta gamma"
    claim = _claim(source)
    citation = ClaimCitation(
        claim.claim_id,
        _span(source, "alpha"),
        _reference("citation-4"),
    )

    record = check_citation_minimality(source, [claim], [citation]).records[0]

    assert record.status is CitationMinimalityStatus.MISSING_REQUIRED_SPAN
    assert record.review_required is True
    assert record.excess_token_count == 0
    assert record.excess_context_spans == ()


def test_mapping_inputs_and_derived_citation_reference_are_supported() -> None:
    source = "alpha beta gamma"
    claim_id = _reference("claim-mapping")
    claim_start = source.index("beta")
    claim_end = claim_start + len("beta")
    citation_end = source.index("gamma") + len("gamma")

    report = check_citation_minimality(
        source,
        [
            {
                "claim_id": claim_id,
                "required_offset": {"start": claim_start, "end": claim_end},
            }
        ],
        [
            {
                "claim_id": claim_id,
                "citation_offset": {"start": claim_start, "end": citation_end},
            }
        ],
    )

    record = report.records[0]
    assert record.citation_id.startswith("sha256:")
    assert record.citation_id != claim_id
    assert record.status is CitationMinimalityStatus.EXCESS_CONTEXT


def test_input_order_does_not_change_json_or_record_order() -> None:
    source = "alpha beta gamma delta"
    first_claim = AtomicClaim(_reference("claim-a"), _span(source, "beta"))
    second_claim = AtomicClaim(_reference("claim-b"), _span(source, "gamma"))
    citations = [
        ClaimCitation(
            second_claim.claim_id,
            CitationSpan(source.index("gamma"), source.index("delta")),
            _reference("citation-b"),
        ),
        ClaimCitation(
            first_claim.claim_id,
            first_claim.required_span,
            _reference("citation-a"),
        ),
    ]

    first = check_citation_minimality(
        source,
        [first_claim, second_claim],
        citations,
    )
    second = check_citation_minimality(
        source,
        reversed([second_claim, first_claim]),
        reversed(citations),
    )

    assert first.to_json() == second.to_json()
    assert first.to_json() == first.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_export_is_json_ready_and_schema_is_stable() -> None:
    source = "alpha beta"
    claim = _claim(source)
    exported = export_citation_minimality(
        source,
        [claim],
        [ClaimCitation(claim.claim_id, claim.required_span)],
    )

    assert exported["schema_version"] == CITATION_MINIMALITY_SCHEMA_VERSION
    assert set(exported) == {
        "schema_version",
        "claim_count",
        "citation_count",
        "flagged_count",
        "max_excess_tokens",
        "records",
        "disclaimer",
    }
    assert exported["records"][0]["citation_id"].startswith("sha256:")


def test_checker_is_available_from_the_public_clinical_namespace() -> None:
    from openmed.clinical import check_citation_minimality as public_checker

    source = "alpha beta"
    claim = _claim(source)

    report = public_checker(
        source,
        [claim],
        [ClaimCitation(claim.claim_id, claim.required_span)],
    )

    assert report.flagged_count == 0


def test_source_text_is_not_retained_or_serialized() -> None:
    source = "SYNTHETIC_PRIVATE_VALUE alpha beta"
    claim = _claim(source, "beta")
    report = check_citation_minimality(
        source,
        [claim],
        [ClaimCitation(claim.claim_id, claim.required_span)],
    )

    rendered = report.to_json() + report.to_markdown() + repr(report)
    assert "SYNTHETIC_PRIVATE_VALUE" not in rendered
    assert not hasattr(report, "source_text")


def test_invalid_inputs_do_not_echo_sensitive_values() -> None:
    sentinel = "SYNTHETIC_PRIVATE_IDENTIFIER"
    with pytest.raises(CitationMinimalityError) as error:
        AtomicClaim(sentinel, CitationSpan(0, 1))
    assert sentinel not in str(error.value)

    source = "alpha beta"
    claim = _claim(source)
    with pytest.raises(CitationMinimalityError) as error:
        check_citation_minimality(
            source,
            [claim],
            [ClaimCitation(claim.claim_id, CitationSpan(0, len(source) + 1))],
        )
    assert source not in str(error.value)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 2), (2, 2), (4, 3), (True, 2), (1, False)],
)
def test_invalid_span_offsets_fail_without_echoing_values(
    start: object,
    end: object,
) -> None:
    with pytest.raises(CitationMinimalityError, match="citation"):
        CitationSpan(start, end)  # type: ignore[arg-type]


def test_invalid_policy_and_schema_values_fail_closed() -> None:
    source = "alpha beta"
    claim = _claim(source)
    citation = ClaimCitation(claim.claim_id, claim.required_span)

    with pytest.raises(CitationMinimalityError, match="max_excess_tokens"):
        check_citation_minimality(
            source,
            [claim],
            [citation],
            max_excess_tokens=-1,
        )
    with pytest.raises(CitationMinimalityError, match="schema version"):
        CitationMinimalityReport((), 0, schema_version=2)


def test_records_are_immutable() -> None:
    source = "alpha beta"
    claim = _claim(source)
    report = check_citation_minimality(
        source,
        [claim],
        [ClaimCitation(claim.claim_id, claim.required_span)],
    )

    with pytest.raises(FrozenInstanceError):
        report.records[0].review_required = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.records = ()  # type: ignore[misc]
