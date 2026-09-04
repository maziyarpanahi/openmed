from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.citation_ordering import (
    CITATION_ORDERING_SCHEMA_VERSION,
    Citation,
    CitationOrdering,
    CitationOrderingError,
    order_citations,
)


def _opaque(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _citation(
    evidence_id: str,
    *,
    document_id: str = "doc-01",
    section: str = "assessment",
    start: int = 10,
    end: int = 20,
    primary: bool = False,
) -> Citation:
    return Citation(
        document_id=_opaque(document_id),
        section=_opaque(section),
        source_start=start,
        source_end=end,
        evidence_id=_opaque(evidence_id),
        primary=primary,
    )


def test_ordering_uses_every_declared_key_in_sequence() -> None:
    citations = [
        _citation("evidence-b"),
        _citation("evidence-a"),
        _citation("evidence-c", end=19),
        _citation("evidence-d", start=5, end=9),
        _citation("evidence-e", section="history"),
        _citation("evidence-f", document_id="doc-00", section="z-section"),
    ]

    ordered = order_citations(reversed(citations))

    assert ordered == tuple(
        sorted(
            citations,
            key=lambda citation: (
                citation.document_id,
                citation.section,
                citation.source_start,
                citation.source_end,
                citation.evidence_id,
            ),
        )
    )


def test_empty_and_generator_inputs_are_supported() -> None:
    assert order_citations(()) == ()

    citations = (_citation(f"evidence-{index}") for index in (3, 1, 2))
    ordered = order_citations(citations)
    assert [citation.evidence_id for citation in ordered] == sorted(
        _opaque(f"evidence-{index}") for index in (3, 1, 2)
    )


def test_primary_marker_is_preserved_without_overriding_metadata_order() -> None:
    primary = _citation("evidence-z", primary=True)
    secondary = _citation("evidence-a")

    ordered = order_citations([primary, secondary])

    assert ordered == tuple(
        sorted(
            (primary, secondary),
            key=lambda citation: (
                citation.document_id,
                citation.section,
                citation.source_start,
                citation.source_end,
                citation.evidence_id,
            ),
        )
    )
    assert sum(citation.is_primary for citation in ordered) == 1
    assert primary.is_primary is True
    assert secondary.is_primary is False


def test_multiple_primary_citations_are_rejected_without_echoing_ids() -> None:
    sentinel = "SENSITIVE_SENTINEL"

    with pytest.raises(
        CitationOrderingError,
        match="conflicting citation primary markers",
    ) as error:
        order_citations(
            [
                _citation("evidence-a", document_id=sentinel, primary=True),
                _citation("evidence-b", primary=True),
            ]
        )
    assert sentinel not in str(error.value)


def test_duplicate_key_cannot_disagree_about_primary_marker() -> None:
    with pytest.raises(
        CitationOrderingError,
        match="conflicting citation primary markers",
    ):
        order_citations(
            [
                _citation("evidence-a"),
                _citation("evidence-a", primary=True),
            ]
        )


def test_exact_secondary_duplicates_remain_stably_ordered() -> None:
    citation = _citation("evidence-a")

    assert order_citations([citation, citation]) == (citation, citation)


def test_artifact_is_versioned_byte_stable_and_input_order_independent() -> None:
    first = CitationOrdering(
        (
            _citation("evidence-b", start=21, end=30),
            _citation("evidence-a", primary=True),
        )
    )
    second = CitationOrdering(tuple(reversed(first.citations)))

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.to_json() == (
        json.dumps(
            first.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    assert json.loads(first.to_json()) == first.to_dict()


def test_empty_artifact_has_a_stable_shape() -> None:
    artifact = CitationOrdering(())

    assert artifact.to_dict() == {
        "schema_version": CITATION_ORDERING_SCHEMA_VERSION,
        "citations": [],
    }
    assert artifact.to_json() == '{"citations":[],"schema_version":1}\n'


@pytest.mark.parametrize(
    "kwargs",
    [
        {"document_id": ""},
        {"document_id": "/srv/charts/123"},
        {"section": "assessment and plan"},
        {"evidence_id": "https://internal.invalid/evidence"},
        {"source_start": -1},
        {"source_start": True},
        {"source_end": 10},
        {"source_end": 9},
        {"primary": 1},
    ],
)
def test_invalid_metadata_fails_without_echoing_values(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "document_id": _opaque("doc-01"),
        "section": _opaque("assessment"),
        "source_start": 10,
        "source_end": 20,
        "evidence_id": _opaque("evidence-01"),
        "primary": False,
    }
    values.update(kwargs)

    with pytest.raises(CitationOrderingError) as error:
        Citation(**values)  # type: ignore[arg-type]
    assert all(
        not isinstance(value, str) or not value or value not in str(error.value)
        for value in kwargs.values()
    )


def test_artifact_schema_cannot_carry_sensitive_values() -> None:
    citation = _citation("evidence-a")
    payload = CitationOrdering((citation,)).to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert set(payload) == {"schema_version", "citations"}
    assert set(payload["citations"][0]) == {
        "document_id",
        "section",
        "source_offset",
        "evidence_id",
        "primary",
    }
    for forbidden in (
        "prompt",
        "argument",
        "output",
        "evidence_text",
        "bearer",
        "path",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("field", "sentinel"),
    [
        ("document_id", "Patient_Jane_Doe"),
        ("section", "diagnosis_HIV"),
        ("evidence_id", "MRN_123456"),
    ],
)
def test_token_shaped_sensitive_identifiers_are_rejected(
    field: str,
    sentinel: str,
) -> None:
    values = {
        "document_id": _opaque("doc-01"),
        "section": _opaque("assessment"),
        "source_start": 10,
        "source_end": 20,
        "evidence_id": _opaque("evidence-01"),
    }
    values[field] = sentinel

    with pytest.raises(CitationOrderingError) as error:
        Citation(**values)  # type: ignore[arg-type]
    assert sentinel not in str(error.value)


def test_invalid_collections_and_schema_versions_fail_closed() -> None:
    with pytest.raises(CitationOrderingError, match="invalid citation collection"):
        order_citations("not-a-collection")  # type: ignore[arg-type]
    with pytest.raises(CitationOrderingError, match="invalid citation collection"):
        order_citations([object()])  # type: ignore[list-item]
    with pytest.raises(
        CitationOrderingError,
        match="unsupported citation ordering schema",
    ):
        CitationOrdering((), schema_version=2)


def test_iterable_failures_do_not_echo_upstream_content() -> None:
    sentinel = "SYNTHETIC_PATIENT_TEXT_FROM_GENERATOR"

    def broken_citations():
        raise RuntimeError(sentinel)
        yield _citation("unreachable")

    with pytest.raises(
        CitationOrderingError,
        match="invalid citation collection",
    ) as error:
        order_citations(broken_citations())
    assert sentinel not in str(error.value)


def test_citations_and_artifacts_are_immutable() -> None:
    citation = _citation("evidence-a")
    artifact = CitationOrdering((citation,))

    with pytest.raises(FrozenInstanceError):
        citation.primary = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        artifact.citations = ()  # type: ignore[misc]
