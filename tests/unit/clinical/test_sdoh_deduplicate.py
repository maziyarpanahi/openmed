"""Synthetic offline tests for SDOH evidence deduplication."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.clinical.sdoh_deduplicate import (
    SDOHEvidenceObservation,
    SDOHSourceReference,
    deduplicate_sdoh_evidence,
)

SYNTHETIC_MARKER = "SYNTHETIC-SENSITIVE-SDOH-VALUE"


def _source(version: str, start: int = 4, end: int = 17) -> SDOHSourceReference:
    return SDOHSourceReference(
        source_id="document-local-1",
        version_id=version,
        start=start,
        end=end,
    )


def _observation(
    observation_id: str,
    version: str,
    text: str,
    *,
    status: str = "current",
) -> SDOHEvidenceObservation:
    return SDOHEvidenceObservation(
        observation_id=observation_id,
        category="housing",
        status=status,
        temporality="recent",
        source=_source(version),
        protected_text=text,
    )


def test_exact_duplicates_retain_every_source_reference() -> None:
    result = deduplicate_sdoh_evidence(
        (
            _observation("observation-1", "v1", "synthetic housing concern"),
            _observation("observation-2", "v2", "synthetic housing concern"),
        )
    )

    assert result.evidence_count == 2
    assert result.independent_evidence_count == 1
    cluster = result.clusters[0]
    assert cluster.duplicate_kind == "exact"
    assert cluster.observation_ids == ("observation-1", "observation-2")
    assert [source.version_id for source in cluster.source_references] == ["v1", "v2"]


def test_normalized_duplicates_cluster_across_document_versions() -> None:
    result = deduplicate_sdoh_evidence(
        (
            _observation("observation-1", "v1", "Housing insecurity"),
            _observation("observation-2", "v2", "  HOUSING---insecurity!  "),
        )
    )

    assert len(result.clusters) == 1
    assert result.clusters[0].duplicate_kind == "normalized"
    assert len(result.clusters[0].exact_fingerprints) == 2


def test_conflicting_status_is_not_collapsed() -> None:
    result = deduplicate_sdoh_evidence(
        (
            _observation("observation-1", "v1", "synthetic support", status="current"),
            _observation("observation-2", "v2", "synthetic support", status="none"),
        )
    )

    assert result.independent_evidence_count == 2
    assert {cluster.status for cluster in result.clusters} == {"current", "none"}


def test_deduplication_is_input_order_independent() -> None:
    observations = (
        _observation("observation-2", "v2", "synthetic support"),
        _observation("observation-1", "v1", "SYNTHETIC support!"),
    )

    first = deduplicate_sdoh_evidence(observations).to_dict()
    second = deduplicate_sdoh_evidence(reversed(observations)).to_dict()

    assert first == second


def test_protected_text_is_absent_from_representations_and_report() -> None:
    observation = _observation("observation-1", "v1", SYNTHETIC_MARKER)
    result = deduplicate_sdoh_evidence((observation,))
    rendered = json.dumps(result.to_dict(), sort_keys=True)

    assert SYNTHETIC_MARKER not in repr(observation)
    assert SYNTHETIC_MARKER not in repr(result)
    assert SYNTHETIC_MARKER not in rendered
    assert '"normalized_fingerprint": "sha256:' in rendered


def test_invalid_protected_text_error_does_not_echo_value() -> None:
    with pytest.raises(ValueError) as error:
        _observation("observation-1", "v1", "   ")

    assert SYNTHETIC_MARKER not in str(error.value)


def test_deduplication_performs_no_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    result = deduplicate_sdoh_evidence(
        (_observation("observation-1", "v1", "synthetic support"),)
    )

    assert result.independent_evidence_count == 1
