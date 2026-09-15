"""Focused tests for value-free DICOM-SR provenance mapping."""

from __future__ import annotations

import json

import pytest

from openmed.multimodal import ExtractedDocument, SourceSpan
from openmed.multimodal.dicom_sr_provenance import (
    DICOM_SR_PROVENANCE_ADVISORY,
    AmbiguousDicomSrItemPathError,
    DicomSrProvenanceError,
    build_dicom_sr_provenance,
    render_dicom_sr_provenance,
    serialize_dicom_sr_provenance,
)

_CONTENT_ITEMS = [
    {
        "node_path": "1",
        "template_id": "1500",
        "value": "Synthetic report title",
    },
    {
        "node_path": "1.3",
        "template_id": "1501",
        "value": "Synthetic measurement container",
    },
    {
        "node_path": "1.3.1",
        "template_id": "1502",
        "value": "Synthetic measurement group",
    },
    {
        "node_path": "1.3.1.3",
        "value": "12.5 mm",
    },
    {
        "node_path": "1.4.1",
        "value": "Synthetic protected finding text",
    },
]

_SPANS = (
    SourceSpan(
        start=0,
        end=20,
        metadata={"node_path": "1"},
    ),
    SourceSpan(
        start=21,
        end=48,
        metadata={"node_path": "1.3"},
    ),
    SourceSpan(
        start=49,
        end=78,
        metadata={"node_path": "1.3.1"},
    ),
    SourceSpan(
        start=79,
        end=104,
        metadata={"node_path": "1.3.1.3"},
    ),
    SourceSpan(
        start=105,
        end=145,
        metadata={"node_path": "1.4.1"},
    ),
)


def _document() -> ExtractedDocument:
    return ExtractedDocument(
        text="x" * 145,
        spans=_SPANS,
        metadata={"content_items": _CONTENT_ITEMS},
    )


def test_maps_finding_ids_to_paths_templates_and_available_offsets():
    document = _document()
    findings = [
        {
            "finding_id": "finding-b",
            "item_path": "1.4.1",
            "value": "Synthetic protected finding text",
        },
        {
            "finding_id": "finding-a",
            "node_path": "1.3.1.3",
            "value": "12.5 mm",
        },
    ]

    records = build_dicom_sr_provenance(findings, document=document)

    assert [record["finding_id"] for record in records] == [
        "finding-a",
        "finding-b",
    ]
    assert records[0].to_dict() == {
        "finding_id": "finding-a",
        "item_path": "1.3.1.3",
        "template_id": "1502",
        "source_start": 79,
        "source_end": 104,
    }
    # The nearest declared template is used for a leaf item without its own
    # ContentTemplateSequence.
    assert records[1]["template_id"] == "1500"
    assert records[1].source_offsets == (105, 145)


def test_resolves_item_path_from_finding_offsets():
    span = _SPANS[3]
    records = build_dicom_sr_provenance(
        [{"finding_id": "finding-a", "source_offsets": (80, 90)}],
        _CONTENT_ITEMS,
        _SPANS,
    )

    assert records[0]["item_path"] == "1.3.1.3"
    assert records[0]["source_start"] == 80
    assert records[0]["source_end"] == 90
    assert span.start <= records[0]["source_start"] < records[0]["source_end"]


def test_direct_id_to_path_mapping_is_sorted_and_value_free():
    records = build_dicom_sr_provenance(
        {
            "finding-z": {"item_path": "1.3.1.3", "value": "12.5 mm"},
            "finding-a": "1.4.1",
        },
        _CONTENT_ITEMS,
    )
    rendered = render_dicom_sr_provenance(records)
    serialized = serialize_dicom_sr_provenance(records)

    assert [row["finding_id"] for row in rendered] == [
        "finding-a",
        "finding-z",
    ]
    assert all(
        set(row)
        == {
            "finding_id",
            "item_path",
            "template_id",
            "source_start",
            "source_end",
        }
        for row in rendered
    )
    assert "Synthetic protected finding text" not in serialized
    assert "12.5 mm" not in serialized
    assert json.loads(serialized) == rendered


def test_duplicate_content_item_paths_are_rejected():
    with pytest.raises(AmbiguousDicomSrItemPathError, match="duplicate item path"):
        build_dicom_sr_provenance(
            [{"finding_id": "finding-a", "item_path": "1.1"}],
            [{"node_path": "1.1"}, {"node_path": "1.1"}],
        )


def test_duplicate_source_span_paths_are_rejected():
    spans = (
        SourceSpan(0, 10, metadata={"node_path": "1.1"}),
        SourceSpan(10, 20, metadata={"node_path": "1.1"}),
    )
    with pytest.raises(AmbiguousDicomSrItemPathError, match="duplicate item path"):
        build_dicom_sr_provenance(
            [{"finding_id": "finding-a", "source_offsets": (1, 2)}],
            (),
            spans,
        )


def test_offsets_matching_overlapping_spans_are_rejected():
    spans = (
        SourceSpan(0, 10, metadata={"node_path": "1.1"}),
        SourceSpan(0, 10, metadata={"node_path": "1.2"}),
    )
    with pytest.raises(AmbiguousDicomSrItemPathError, match="multiple item paths"):
        build_dicom_sr_provenance(
            [{"finding_id": "finding-a", "source_offsets": (2, 4)}],
            (),
            spans,
        )


def test_conflicting_path_aliases_fail_closed_without_echoing_finding_value():
    protected = "Synthetic protected finding text"
    with pytest.raises(AmbiguousDicomSrItemPathError) as error:
        build_dicom_sr_provenance(
            [
                {
                    "finding_id": "finding-a",
                    "item_path": "1.1",
                    "node_path": "1.2",
                    "value": protected,
                }
            ],
            _CONTENT_ITEMS,
        )

    assert protected not in str(error.value)


def test_malformed_and_unresolved_references_are_rejected():
    with pytest.raises(DicomSrProvenanceError, match="1-based dotted"):
        build_dicom_sr_provenance(
            [{"finding_id": "finding-a", "item_path": "1.0"}],
            _CONTENT_ITEMS,
        )

    with pytest.raises(DicomSrProvenanceError, match="must supply an item path"):
        build_dicom_sr_provenance(
            [{"finding_id": "finding-a"}],
            _CONTENT_ITEMS,
        )


def test_provenance_is_deterministic_and_advisory_is_value_free():
    first = serialize_dicom_sr_provenance(
        build_dicom_sr_provenance(
            [{"finding_id": "finding-b", "item_path": "1.4.1"}],
            _CONTENT_ITEMS,
        )
    )
    second = serialize_dicom_sr_provenance(
        build_dicom_sr_provenance(
            [{"finding_id": "finding-b", "item_path": "1.4.1"}],
            _CONTENT_ITEMS,
        )
    )

    assert first == second
    assert "report values" in DICOM_SR_PROVENANCE_ADVISORY
