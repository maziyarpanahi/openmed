from __future__ import annotations

import builtins
import math
import traceback
import urllib.request
from typing import Any

import pytest

from openmed.agent.workflows import (
    ABSTRACTION_EVIDENCE_SCHEMA,
    AbstractionEvidenceChain,
    AbstractionEvidenceError,
    AbstractionEvidenceIssue,
    AbstractionEvidenceReport,
    ChartAbstractionEvidence,
    FinalizedAbstractionEvidence,
    ReviewerState,
    SourceKind,
    SourceLocation,
    TransformationKind,
)

SOURCE_A = "sha256:" + "a" * 64
SOURCE_B = "sha256:" + "b" * 64
FACT_A = "sha256:" + "c" * 64
FACT_B = "sha256:" + "d" * 64
TRANSFORM_A = "sha256:" + "e" * 64
TRANSFORM_B = "sha256:" + "f" * 64


def _source(
    digest: str = SOURCE_A,
    *,
    start: int = 10,
    end: int = 20,
    kind: SourceKind = SourceKind.CLINICAL_RECORD,
) -> SourceLocation:
    return SourceLocation(digest, start, end, kind)


def _chain(
    field_id: str = "registry.primary_diagnosis",
    *,
    sources: tuple[SourceLocation, ...] | None = None,
    fact_digest: str = FACT_A,
    transformation_kind: TransformationKind = TransformationKind.RULE,
    transformation_digest: str = TRANSFORM_A,
    uncertainty: float = 0.1,
    reviewer_state: ReviewerState = ReviewerState.APPROVED,
) -> AbstractionEvidenceChain:
    return AbstractionEvidenceChain(
        field_id,
        (_source(),) if sources is None else sources,
        fact_digest,
        transformation_kind,
        transformation_digest,
        uncertainty,
        reviewer_state,
    )


def test_chain_records_complete_value_free_provenance() -> None:
    chain = _chain()

    assert chain.has_clinical_source
    assert chain.to_dict() == {
        "field_id": "registry.primary_diagnosis",
        "normalized_fact_digest": FACT_A,
        "reviewer_state": "approved",
        "source_locations": [
            {
                "end_offset": 20,
                "kind": "clinical_record",
                "source_digest": SOURCE_A,
                "start_offset": 10,
            }
        ],
        "transformation_digest": TRANSFORM_A,
        "transformation_kind": "rule",
        "uncertainty": 0.1,
    }
    assert chain.chain_digest.startswith("sha256:")


def test_finalization_is_order_independent_and_deterministic() -> None:
    first = _chain()
    second = _chain(
        "registry.onset_date",
        sources=(_source(SOURCE_B, start=30, end=40),),
        fact_digest=FACT_B,
        transformation_kind=TransformationKind.MODEL,
        transformation_digest=TRANSFORM_B,
        uncertainty=0.25,
    )

    evidence = ChartAbstractionEvidence((first, second))
    reordered = ChartAbstractionEvidence((second, first))
    receipt = evidence.finalize(("registry.primary_diagnosis", "registry.onset_date"))
    other_receipt = reordered.finalize(
        ("registry.onset_date", "registry.primary_diagnosis")
    )

    assert receipt == other_receipt
    assert receipt.chain_count == 2
    assert receipt.schema == ABSTRACTION_EVIDENCE_SCHEMA
    assert receipt.to_json() == other_receipt.to_json()


def test_missing_required_field_rejects_finalization() -> None:
    evidence = ChartAbstractionEvidence((_chain(),))

    with pytest.raises(AbstractionEvidenceError) as caught:
        evidence.finalize(("registry.primary_diagnosis", "registry.onset_date"))

    assert caught.value.code == "evidence_not_finalizable"
    assert caught.value.report is not None
    assert caught.value.report.is_finalizable is False
    assert [issue.to_dict() for issue in caught.value.report.issues] == [
        {"code": "missing_field_evidence", "field_id": "registry.onset_date"}
    ]


def test_missing_source_evidence_rejects_finalization() -> None:
    evidence = ChartAbstractionEvidence((_chain(sources=()),))

    report = evidence.evaluate(("registry.primary_diagnosis",))

    assert not report.is_finalizable
    assert [issue.code for issue in report.issues] == ["missing_source_evidence"]
    with pytest.raises(AbstractionEvidenceError, match="evidence_not_finalizable"):
        evidence.finalize(("registry.primary_diagnosis",))


def test_generated_only_evidence_rejects_finalization() -> None:
    generated = _source(kind=SourceKind.GENERATED_TEXT)
    evidence = ChartAbstractionEvidence((_chain(sources=(generated,)),))

    report = evidence.evaluate(("registry.primary_diagnosis",))

    assert [issue.code for issue in report.issues] == ["generated_only_evidence"]
    assert report.clinical_source_count == 0


def test_generated_span_may_supplement_but_not_replace_clinical_source() -> None:
    sources = (
        _source(kind=SourceKind.GENERATED_TEXT),
        _source(SOURCE_B, start=21, end=31),
    )
    evidence = ChartAbstractionEvidence((_chain(sources=sources),))

    report = evidence.evaluate(("registry.primary_diagnosis",))

    assert report.is_finalizable
    assert report.clinical_source_count == 1


@pytest.mark.parametrize("state", [ReviewerState.PENDING, ReviewerState.REJECTED])
def test_nonapproved_review_state_rejects_finalization(state: ReviewerState) -> None:
    evidence = ChartAbstractionEvidence((_chain(reviewer_state=state),))

    report = evidence.evaluate(("registry.primary_diagnosis",))

    assert [issue.code for issue in report.issues] == ["review_not_approved"]
    assert report.approved_field_count == 0


def test_report_contains_only_digests_offsets_counts_and_closed_metadata() -> None:
    sentinel = "Synthetic Person has private diagnosis Z99.999"
    evidence = ChartAbstractionEvidence((_chain(),))

    report = evidence.evaluate(("registry.primary_diagnosis",))
    rendered = report.to_json()

    assert sentinel not in rendered
    assert SOURCE_A not in rendered
    assert FACT_A not in rendered
    assert TRANSFORM_A not in rendered
    assert report.report_digest.startswith("sha256:")


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"source_digest": "not-a-digest"}, "invalid_digest"),
        ({"start_offset": -1}, "invalid_offset"),
        ({"start_offset": 20, "end_offset": 20}, "invalid_span"),
        ({"kind": "clinical_record"}, "invalid_source_kind"),
    ],
)
def test_source_location_fails_closed(kwargs: dict[str, Any], code: str) -> None:
    values: dict[str, Any] = {
        "source_digest": SOURCE_A,
        "start_offset": 10,
        "end_offset": 20,
        "kind": SourceKind.CLINICAL_RECORD,
    }
    values.update(kwargs)

    with pytest.raises(AbstractionEvidenceError) as caught:
        SourceLocation(**values)

    assert caught.value.code == code
    assert all(str(value) not in str(caught.value) for value in kwargs.values())


@pytest.mark.parametrize("uncertainty", [-0.1, 1.1, math.inf, math.nan, True])
def test_uncertainty_is_bounded_and_finite(uncertainty: Any) -> None:
    with pytest.raises(AbstractionEvidenceError, match="invalid_uncertainty"):
        _chain(uncertainty=uncertainty)


def test_duplicate_fields_and_source_spans_fail_closed() -> None:
    chain = _chain()
    with pytest.raises(AbstractionEvidenceError, match="duplicate_field"):
        ChartAbstractionEvidence((chain, chain))
    with pytest.raises(AbstractionEvidenceError, match="duplicate_source"):
        _chain(sources=(_source(), _source()))


def test_empty_required_field_inventory_fails_closed() -> None:
    with pytest.raises(AbstractionEvidenceError, match="empty_collection"):
        ChartAbstractionEvidence((_chain(),)).finalize(())


def test_report_types_cannot_be_constructed_with_unbounded_text() -> None:
    sentinel = "Synthetic Person: Z99.999 /tmp/chart.txt"

    with pytest.raises(AbstractionEvidenceError) as caught:
        AbstractionEvidenceIssue(sentinel, "registry.primary_diagnosis")
    assert sentinel not in str(caught.value)

    with pytest.raises(AbstractionEvidenceError, match="invalid_issues"):
        AbstractionEvidenceReport(
            FACT_A,
            FACT_B,
            TRANSFORM_A,
            (sentinel,),  # type: ignore[arg-type]
            1,
            1,
            1,
            1,
        )

    with pytest.raises(AbstractionEvidenceError, match="invalid_schema"):
        FinalizedAbstractionEvidence(
            FACT_A,
            FACT_B,
            TRANSFORM_A,
            1,
            schema=sentinel,
        )


def test_invalid_identifiers_and_values_do_not_leak_in_errors() -> None:
    sentinel = "Synthetic Person: Z99.999 /tmp/chart.txt"

    with pytest.raises(AbstractionEvidenceError) as caught:
        _chain(field_id=sentinel)

    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert caught.value.code == "invalid_field_id"
    assert sentinel not in rendered


def test_validation_performs_no_file_or_network_io(monkeypatch) -> None:
    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("evidence validation must remain local and in-memory")

    monkeypatch.setattr(builtins, "open", unexpected_io)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_io)

    receipt = ChartAbstractionEvidence((_chain(),)).finalize(
        ("registry.primary_diagnosis",)
    )
    assert receipt.chain_count == 1
