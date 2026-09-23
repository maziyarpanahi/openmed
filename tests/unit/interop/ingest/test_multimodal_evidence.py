"""Synthetic PDF, OCR, image, and DICOM evidence-coordinate tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.interop.ingest import (
    CoordinateTransformChain,
    CoordinateTransformStep,
    DICOMElementReference,
    DICOMEvidenceAdapter,
    DICOMEvidenceInput,
    EvidenceAdapterContext,
    MultimodalCoordinateError,
    StructuredEvidenceQuarantine,
    StructuredEvidenceResult,
    VisualEvidenceAdapter,
    VisualEvidenceFrame,
    VisualEvidenceInput,
    VisualEvidenceRegion,
    load_evidence_adapter_schema,
    load_multimodal_coordinate_schema,
)
from openmed.multimodal.base import ExtractedDocument, SourceSpan
from openmed.multimodal.ocr import OcrResult, OcrWord
from openmed.structured.store import StoreState

T0 = "2026-01-02T03:04:05Z"
FIXTURE = "tests/fixtures/multimodal/evidence_coordinates_golden.json"


@pytest.fixture
def context() -> EvidenceAdapterContext:
    return EvidenceAdapterContext(
        source_id="source_0123456789abcdef",
        subject_id="subject_0123456789abcdef",
        encounter_id="encounter_0123456789abcdef",
        recorded_at=T0,
    )


def _result(value: object) -> StructuredEvidenceResult:
    assert hasattr(value, "ok") and value.ok  # type: ignore[attr-defined]
    result = value.value  # type: ignore[attr-defined]
    assert isinstance(result, StructuredEvidenceResult)
    return result


def _step(payload: dict[str, object]) -> CoordinateTransformStep:
    input_dimensions = payload["input_dimensions"]
    output_dimensions = payload["output_dimensions"]
    assert isinstance(input_dimensions, list)
    assert isinstance(output_dimensions, list)
    return CoordinateTransformStep(
        operation=str(payload["operation"]),
        input_width=float(input_dimensions[0]),
        input_height=float(input_dimensions[1]),
        output_width=float(output_dimensions[0]),
        output_height=float(output_dimensions[1]),
        parameters=payload["parameters"],  # type: ignore[arg-type]
    )


def _input_from_case(case: dict[str, object]) -> VisualEvidenceInput:
    dimensions = case["source_dimensions"]
    assert isinstance(dimensions, list)
    steps = case["steps"]
    assert isinstance(steps, list)
    name = str(case["name"])
    chain = CoordinateTransformChain(
        source_width=float(dimensions[0]),
        source_height=float(dimensions[1]),
        steps=tuple(_step(item) for item in steps),
    )
    return VisualEvidenceInput(
        source_bytes=f"synthetic-{name}".encode(),
        source_format=str(case["source_format"]),
        source_version=str(case["source_version"]),
        media_type=str(case["media_type"]),
        frames=(
            VisualEvidenceFrame(
                frame_id="frame_0001",
                page=1,
                coordinate_space=str(case["coordinate_space"]),
                transform=chain,
                tolerance=float(case["tolerance"]),
            ),
        ),
        regions=(
            VisualEvidenceRegion(
                frame_id="frame_0001",
                box=tuple(case["observed_box"]),  # type: ignore[arg-type]
                region_kind=f"{case['source_format']}_region",
            ),
        ),
    )


def test_visual_coordinate_golden_round_trip(
    context: EvidenceAdapterContext,
) -> None:
    fixture = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))
    coordinate_schema = load_multimodal_coordinate_schema()
    result_schema = load_evidence_adapter_schema("result")
    validator_for(coordinate_schema).check_schema(coordinate_schema)
    validator_for(result_schema).check_schema(result_schema)
    coordinate_validator = validator_for(coordinate_schema)(coordinate_schema)
    result_validator = validator_for(result_schema)(result_schema)

    assert fixture["schema_version"] == 1
    for case in fixture["cases"]:
        source = _input_from_case(case)
        result = _result(VisualEvidenceAdapter().adapt(source, context))
        locator = result.locators[0]
        coordinate_record = locator.transform["coordinate_record"]
        serialized = json.loads(result.to_json())
        serialized_coordinate = serialized["locators"][0]["transform"][
            "coordinate_record"
        ]

        assert locator.location["box"] == pytest.approx(
            case["expected_source_box"], abs=case["tolerance"]
        )
        assert coordinate_record["frame_id"] == "frame_0001"
        assert not tuple(coordinate_validator.iter_errors(serialized_coordinate))
        assert not tuple(result_validator.iter_errors(serialized))
        resolved = VisualEvidenceAdapter().resolve(source, locator)
        assert resolved.ok and resolved.value["box"] == pytest.approx(
            case["expected_source_box"], abs=case["tolerance"]
        )


@given(
    x0=st.integers(min_value=40, max_value=350),
    y0=st.integers(min_value=20, max_value=170),
    x1=st.integers(min_value=50, max_value=360),
    y1=st.integers(min_value=30, max_value=180),
)
def test_crop_and_ocr_rescale_round_trip_property(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> None:
    assume(x1 > x0 and y1 > y0)
    chain = CoordinateTransformChain(
        source_width=400,
        source_height=200,
        steps=(
            CoordinateTransformStep(
                operation="crop",
                input_width=400,
                input_height=200,
                output_width=320,
                output_height=160,
                parameters={"box": [40, 20, 360, 180]},
            ),
            CoordinateTransformStep(
                operation="ocr_scale",
                input_width=320,
                input_height=160,
                output_width=640,
                output_height=320,
            ),
        ),
    )
    source_box = (float(x0), float(y0), float(x1), float(y1))

    observed = chain.forward_box(source_box)
    restored = chain.inverse_box(observed)

    assert restored == pytest.approx(source_box, abs=1e-9)


@pytest.mark.parametrize(
    ("degrees", "output_dimensions"),
    ((90, (200, 100)), (180, (100, 200)), (270, (200, 100))),
)
def test_all_supported_page_rotations_round_trip(
    degrees: int,
    output_dimensions: tuple[int, int],
) -> None:
    step = CoordinateTransformStep(
        operation="rotate_clockwise",
        input_width=100,
        input_height=200,
        output_width=output_dimensions[0],
        output_height=output_dimensions[1],
        parameters={"degrees": degrees},
    )
    source_box = (10.0, 20.0, 40.0, 80.0)

    observed = step.forward_box(source_box)

    assert step.inverse_box(observed) == pytest.approx(source_box, abs=1e-9)


def test_pdf_document_bridge_preserves_text_span_and_page_geometry(
    context: EvidenceAdapterContext,
) -> None:
    document = ExtractedDocument(
        text="synthetic",
        spans=(
            SourceSpan(
                start=0,
                end=9,
                page=0,
                bbox=(10.0, 20.0, 40.0, 50.0),
            ),
        ),
        metadata={"format": "pdf"},
    )
    source = VisualEvidenceInput.from_document(
        document,
        source_bytes=b"synthetic-pdf-bytes",
        source_format="pdf",
        source_version="1.7",
        media_type="application/pdf",
        frames=(
            VisualEvidenceFrame(
                frame_id="page_0001",
                page=1,
                coordinate_space="points",
                transform=CoordinateTransformChain(612, 792),
            ),
        ),
    )

    result = _result(VisualEvidenceAdapter().adapt(source, context))
    locator = result.locators[0]

    assert locator.location == {
        "box": (10.0, 20.0, 40.0, 50.0),
        "coordinate_space": "points",
        "page": 1,
        "page_height": 792.0,
        "page_width": 612.0,
    }
    assert locator.transform["normalized_start"] == 0
    assert locator.transform["normalized_end"] == 9


def test_ocr_bridge_preserves_boxes_after_preprocessing(
    context: EvidenceAdapterContext,
) -> None:
    ocr = OcrResult(
        words=(OcrWord("synthetic", (20.0, 10.0, 80.0, 30.0), 0.99),),
        metadata={"engine": "synthetic"},
    )
    source = VisualEvidenceInput.from_ocr(
        ocr,
        source_bytes=b"synthetic-image-bytes",
        source_version="synthetic",
        media_type="image/png",
        frames=(
            VisualEvidenceFrame(
                frame_id="frame_0001",
                page=1,
                coordinate_space="pixels",
                transform=CoordinateTransformChain(
                    source_width=200,
                    source_height=100,
                    steps=(
                        CoordinateTransformStep(
                            operation="ocr_scale",
                            input_width=200,
                            input_height=100,
                            output_width=100,
                            output_height=50,
                        ),
                    ),
                ),
            ),
        ),
    )

    result = _result(VisualEvidenceAdapter().adapt(source, context))

    assert result.locators[0].location["box"] == pytest.approx(
        (40.0, 20.0, 160.0, 60.0)
    )
    assert result.locators[0].transform["region_kind"] == "ocr_word"


def test_incomplete_or_out_of_bounds_visual_coordinates_quarantine(
    context: EvidenceAdapterContext,
) -> None:
    document = ExtractedDocument(
        text="synthetic",
        spans=(SourceSpan(start=0, end=9),),
    )
    incomplete = VisualEvidenceInput.from_document(
        document,
        source_bytes=b"synthetic",
        source_format="pdf",
        source_version="1.7",
        media_type="application/pdf",
        frames=(
            VisualEvidenceFrame(
                frame_id="page_0001",
                page=1,
                coordinate_space="points",
                transform=CoordinateTransformChain(100, 100),
            ),
        ),
    )
    outside = replace(
        incomplete,
        unmapped_count=0,
        regions=(
            VisualEvidenceRegion(
                frame_id="page_0001",
                box=(90.0, 90.0, 110.0, 110.0),
                region_kind="pdf_text",
            ),
        ),
    )

    partial = VisualEvidenceAdapter().adapt(incomplete, context)
    conflicted = VisualEvidenceAdapter().adapt(outside, context)
    unsupported = VisualEvidenceAdapter().adapt(b"synthetic-canary", context)
    unknown = VisualEvidenceAdapter().adapt(
        replace(incomplete, unmapped_count=0, regions=()),
        context,
    )
    denied = VisualEvidenceAdapter(max_source_bytes=1).adapt(outside, context)

    assert partial.state is StoreState.PARTIAL
    assert partial.code == "visual_coordinates_incomplete"
    assert conflicted.state is StoreState.CONFLICT
    assert conflicted.code == "coordinate_projection_invalid"
    assert unsupported.state is StoreState.UNSUPPORTED
    assert unknown.state is StoreState.UNKNOWN
    assert unknown.code == "visual_evidence_missing"
    assert denied.state is StoreState.DENIED
    assert denied.code == "source_limit_exceeded"
    assert isinstance(unsupported.value, StructuredEvidenceQuarantine)
    assert "synthetic-canary" not in unsupported.value.to_json()


def test_visual_adaptation_is_idempotent_and_refuses_different_bytes(
    context: EvidenceAdapterContext,
) -> None:
    fixture = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))
    source = _input_from_case(fixture["cases"][0])
    adapter = VisualEvidenceAdapter()

    first = _result(adapter.adapt(source, context))
    second = _result(adapter.adapt(source, context))
    altered = replace(source, source_bytes=b"different-synthetic-bytes")
    mismatch = adapter.resolve(altered, first.locators[0])

    assert first == second
    assert mismatch.state is StoreState.CONFLICT
    assert mismatch.code == "source_digest_mismatch"


def test_dicom_sr_bridge_preserves_explicit_tag_and_node_provenance(
    context: EvidenceAdapterContext,
) -> None:
    golden = json.loads(Path(FIXTURE).read_text(encoding="utf-8"))["dicom_sr"]
    document = ExtractedDocument(
        text="synthetic root\nsynthetic child",
        spans=(
            SourceSpan(0, 14, metadata={"node_path": "1"}),
            SourceSpan(15, 30, metadata={"node_path": "1.1"}),
        ),
        metadata={"format": "dicom_sr"},
    )
    source = DICOMEvidenceInput.from_sr_document(
        document,
        source_bytes=b"synthetic-dicom-sr",
        study_uid=golden["study_uid"],
        series_uid=golden["series_uid"],
        instance_uid=golden["instance_uid"],
    )
    adapter = DICOMEvidenceAdapter()

    result = _result(adapter.adapt(source, context))

    assert len(result.locators) == 2
    assert {item.location["tag"] for item in result.locators} == {golden["tag"]}
    assert {
        item.transform["coordinate_record"]["node_path"] for item in result.locators
    } == set(golden["node_paths"])
    schema = load_multimodal_coordinate_schema()
    validator = validator_for(schema)(schema)
    serialized_records = [
        item["transform"]["coordinate_record"]
        for item in json.loads(result.to_json())["locators"]
    ]
    assert all(
        not tuple(validator.iter_errors(record)) for record in serialized_records
    )
    assert adapter.resolve(source, result.locators[0]).ok
    for uid in (source.study_uid, source.series_uid, source.instance_uid):
        assert uid not in repr(source)
        assert uid not in repr(result)


def test_invalid_dicom_tag_is_quarantined_without_a_locator(
    context: EvidenceAdapterContext,
) -> None:
    source = DICOMEvidenceInput(
        source_bytes=b"synthetic-dicom",
        study_uid="1.2.3",
        series_uid="1.2.3.4",
        instance_uid="1.2.3.4.5",
        elements=(DICOMElementReference(tag="not-a-tag"),),
    )

    result = DICOMEvidenceAdapter().adapt(source, context)

    assert result.state is StoreState.FAILURE
    assert result.code == "dicom_coordinates_invalid"
    assert isinstance(result.value, StructuredEvidenceQuarantine)


def test_transform_contracts_reject_malformed_or_discontinuous_geometry() -> None:
    with pytest.raises(MultimodalCoordinateError, match="rotation"):
        CoordinateTransformStep(
            operation="rotate_clockwise",
            input_width=100,
            input_height=200,
            output_width=100,
            output_height=200,
            parameters={"degrees": 90},
        )
    with pytest.raises(MultimodalCoordinateError, match="discontinuous"):
        CoordinateTransformChain(
            source_width=100,
            source_height=100,
            steps=(
                CoordinateTransformStep(
                    operation="scale",
                    input_width=50,
                    input_height=50,
                    output_width=100,
                    output_height=100,
                ),
            ),
        )
