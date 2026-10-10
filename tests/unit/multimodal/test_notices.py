"""Offline mandatory notice, frozen wording and public result inventory gates."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from openmed.mlx.vlm import VisionLanguageGeneration
from openmed.multimodal.notices import (
    NOTICE_CATALOG,
    NOTICE_RESULT_TYPES,
    MeasurementReviewResult,
    MultimodalNotice,
    MultimodalNoticeError,
    NoticeBoundOutput,
    NoticeKind,
    validate_notice_registry,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = json.loads((ROOT / "tests/fixtures/multimodal/notices_v1.json").read_text())
RESULT_TYPES = {**NOTICE_RESULT_TYPES, "vision_generation": VisionLanguageGeneration}

# Immutable published v1 commitments. Append a new ID for wording changes;
# never replace a published commitment to bless a text edit under the same ID.
PUBLISHED_NOTICE_DIGESTS = {
    "openmed.multimodal.measurement_for_review.v1": "2dd5b668040d65ddc5484f31b139b985dae75f01da1d0a3f22eac14049da7376",
    "openmed.multimodal.visual_description.v1": "7c725f619f5aa25d92c0c85328c5a71a67e691895ba01a22e87549d892d032bf",
    "openmed.multimodal.draft_for_review.v1": "e7ee2fafb0e2965272535403f1481add36578a6f1e3fd715477105d0f7040ed6",
}


def test_notice_text_changes_require_new_versioned_identifier():
    for notice in NOTICE_CATALOG.values():
        assert (
            hashlib.sha256(notice.text.encode()).hexdigest()
            == PUBLISHED_NOTICE_DIGESTS[notice.identifier]
        )
    for case in FIXTURES:
        notice = case["result"]["notice"]
        assert (
            hashlib.sha256(notice["text"].encode()).hexdigest()
            == PUBLISHED_NOTICE_DIGESTS[notice["identifier"]]
        )


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case["kind"])
def test_shared_python_swift_wire_round_trip(case):
    cls = RESULT_TYPES[case["kind"]]
    result = cls.from_dict(case["result"])
    assert result.to_dict() == case["result"]
    assert cls.from_json(result.to_json()) == result
    assert case["result"]["notice"]["identifier"] in str(result)
    assert case["result"]["notice"]["text"] in str(result)
    assert result.requires_reviewer_confirmation is True
    assert result.is_diagnostic is False
    with pytest.raises(MultimodalNoticeError, match="confirmation"):
        result.require_reviewer_confirmation(reviewer_confirmed=False)
    with pytest.raises(MultimodalNoticeError):
        result.require_reviewer_confirmation(reviewer_confirmed=1)
    result.require_reviewer_confirmation(reviewer_confirmed=True)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case["kind"])
def test_construction_requires_matching_notice(case):
    cls = RESULT_TYPES[case["kind"]]
    kwargs = {
        k: v
        for k, v in case["result"].items()
        if k
        not in {
            "notice",
            "schema_version",
            "requires_reviewer_confirmation",
            "is_diagnostic",
        }
    }
    with pytest.raises(TypeError):
        cls(**kwargs)
    for invalid in (
        None,
        "",
        case["result"]["notice"],
        NOTICE_CATALOG[
            NoticeKind.DRAFT
            if cls.NOTICE_KIND != NoticeKind.DRAFT
            else NoticeKind.MEASUREMENT
        ],
    ):
        with pytest.raises(MultimodalNoticeError):
            cls(**kwargs, notice=invalid)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case["kind"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_notice",
        "missing_identifier",
        "missing_text",
        "wrong_identifier",
        "altered_text",
        "wrong_kind",
        "review_disabled",
        "diagnostic",
        "missing_review",
        "extra_field",
    ],
)
def test_deserialization_never_repairs_notice_or_safety_contract(case, mutation):
    payload = json.loads(json.dumps(case["result"]))
    if mutation == "missing_notice":
        del payload["notice"]
    elif mutation.startswith("missing_") and mutation != "missing_review":
        del payload["notice"][mutation.removeprefix("missing_")]
    elif mutation == "wrong_identifier":
        payload["notice"]["identifier"] = "synthetic-mrn-98765"
    elif mutation == "altered_text":
        payload["notice"]["text"] += " Synthetic Ada at 72 bpm."
    elif mutation == "wrong_kind":
        payload["notice"] = NOTICE_CATALOG[
            NoticeKind.DRAFT
            if RESULT_TYPES[case["kind"]].NOTICE_KIND != NoticeKind.DRAFT
            else NoticeKind.MEASUREMENT
        ].to_dict()
    elif mutation == "review_disabled":
        payload["requires_reviewer_confirmation"] = False
    elif mutation == "diagnostic":
        payload["is_diagnostic"] = True
    elif mutation == "missing_review":
        del payload["requires_reviewer_confirmation"]
    else:
        payload["synthetic-private-value"] = "Synthetic Ada at 72 bpm."
    with pytest.raises(MultimodalNoticeError) as caught:
        RESULT_TYPES[case["kind"]].from_json(json.dumps(payload))
    assert "Ada" not in str(caught.value)
    assert "98765" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "payload", [b"\xff", "{", "null", "[]", "x" * 65537, '{"notice":{},"notice":{}}']
)
def test_malformed_bounded_json_fails_without_payload(payload):
    with pytest.raises(MultimodalNoticeError) as caught:
        MeasurementReviewResult.from_json(payload)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_serialization_revalidates_even_bypassed_frozen_notice():
    result = MeasurementReviewResult(
        output_digest="a" * 64, notice=NOTICE_CATALOG[NoticeKind.MEASUREMENT]
    )
    object.__setattr__(result, "notice", None)
    with pytest.raises(MultimodalNoticeError):
        result.to_json()


def test_catalog_and_diagnostics_never_interpolate_patient_values():
    for payload in (
        "Synthetic Ada",
        "72 bpm",
        "synthetic-mrn-98765",
        "/private/synthetic-record",
        "秘密-患者",
    ):
        with pytest.raises(MultimodalNoticeError) as caught:
            MultimodalNotice(identifier=payload, text=payload)
        assert payload not in str(caught.value)
        assert all(payload not in n.text for n in NOTICE_CATALOG.values())
        with pytest.raises(MultimodalNoticeError) as caught:
            MeasurementReviewResult(
                output_digest=payload, notice=NOTICE_CATALOG[NoticeKind.MEASUREMENT]
            )
        assert payload not in str(caught.value)
    with pytest.raises(TypeError):
        NOTICE_CATALOG[NoticeKind.MEASUREMENT] = None


def test_vision_repr_excludes_protected_text_and_tokens():
    generation = VisionLanguageGeneration.from_dict(FIXTURES[-1]["result"])
    generation = replace(
        generation, text="Synthetic Ada synthetic-mrn-98765", token_ids=(98765,)
    )
    assert "Ada" not in repr(generation)
    assert "98765" not in repr(generation)
    assert generation.notice == NOTICE_CATALOG[NoticeKind.VISUAL_DESCRIPTION]


def test_registry_rejects_new_covered_result_without_declared_notice():
    @dataclass(frozen=True)
    class UnnoticedMeasurement:
        value: float

    @dataclass(frozen=True)
    class DefaultedNotice(NoticeBoundOutput):
        NOTICE_KIND = NoticeKind.MEASUREMENT
        notice: MultimodalNotice = NOTICE_CATALOG[NoticeKind.MEASUREMENT]

    validate_notice_registry(RESULT_TYPES.values())
    for cls in (UnnoticedMeasurement, DefaultedNotice):
        with pytest.raises(MultimodalNoticeError):
            validate_notice_registry([*RESULT_TYPES.values(), cls])


def _public_data_classes(source: str, module: str) -> set[str]:
    return {
        f"{module}.{node.name}"
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        and not node.name.startswith("_")
        and (
            any(isinstance(field, ast.AnnAssign) for field in node.body)
            or any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "dataclass"
                )
                for d in node.decorator_list
            )
            or (
                node.name.endswith(
                    ("Result", "Generation", "Measurement", "Draft", "Aggregate")
                )
                and not node.name.endswith("Error")
            )
        )
    }


def _assert_classified(discovered: set[str]) -> None:
    covered = {f"notices.{cls.__name__}" for cls in NOTICE_RESULT_TYPES.values()} | {
        "vlm.VisionLanguageGeneration"
    }
    assert not discovered - covered - NON_CLINICAL_TYPES, (
        "New public result requires notice classification"
    )
    assert covered <= discovered


def test_every_public_multimodal_data_result_declares_notice_or_exemption():
    discovered = set()
    for path in (ROOT / "openmed/multimodal").rglob("*.py"):
        module = str(
            path.relative_to(ROOT / "openmed/multimodal").with_suffix("")
        ).replace("/", ".")
        discovered |= _public_data_classes(path.read_text(), module)
    discovered |= _public_data_classes((ROOT / "openmed/mlx/vlm.py").read_text(), "vlm")
    _assert_classified(discovered)
    validate_notice_registry(RESULT_TYPES.values())


def test_inventory_detects_undeclared_new_measurement_result():
    discovered = _public_data_classes(
        "@dataclass\nclass NewIntervalResult:\n    milliseconds: float\n", "new_module"
    )
    with pytest.raises(AssertionError, match="notice classification"):
        _assert_classified(discovered)


# Audited operational/ingestion types: geometry, transport metadata, provenance,
# redaction and resource/preflight reports do not infer clinical measurements.
# This explicit inventory makes new public data types fail until reviewed.
NON_CLINICAL_TYPES = set(
    """
abstention.AbstentionRecord
asr_audio_profile.AsrAudioProfile
asr_audio_profile.AsrCompatibilityReport
asset_batch.BatchFinding
asset_batch.AssetBatch
asset_limits.LimitFinding
asset_limits.LimitProfile
asset_manifest.AssetManifest
audio_format_summary.AudioFormatRecord
audio_format_summary.CategoryCount
audio_format_summary.AudioFormatSummary
audio_resample_plan.AudioResamplePlan
audio_windows.AudioWindow
audio_windows.AudioWindowPlan
base.SourceSpan
base.ExtractedDocument
batch_memory.MemoryEstimationPolicy
batch_memory.AssetMemoryEstimate
batch_memory.PlannedBatch
batch_memory.BatchMemoryPlan
bmp_dimensions.BmpDimensions
box_normalization.PageSize
box_normalization.NormalizedBox
chatlog_jsonl.ChatLogRedactionSummary
chatlog_jsonl.RedactedChatLog
chatlog_jsonl.TurnRecordAdapter
chatlog_jsonl.MessagesListAdapter
chatlog_jsonl.ChatSchemaAdapter
chw_forms.ChwFieldDecision
chw_forms.RedactedChwForm
dicom.DicomHeaderDeidPolicy
dicom.DicomHeaderAction
dicom.DicomHeaderDeidResult
dicom.DicomPixelRedactionPolicy
dicom.DicomPixelFinding
dicom.DicomResidualTextReport
dicom.DicomPixelRedactionResult
dicom_sr_provenance.DicomSrProvenanceRecord
dicom_sr.SrContentItem
digest.AssetDigest
document_graph.SourceRegion
document_graph.DocumentNode
document_graph.DocumentTableCell
document_graph.DocumentTable
document_graph.DocumentColumn
document_graph.DocumentFormField
document_graph.DocumentPage
document_graph.DocumentGraph
documents_docx.DocxRunRange
documents_docx.DocxRedaction
documents_pdf.ProjectedRectangle
documents_pdf_layout.PdfColumn
documents_pdf_layout.PdfPageLayout
documents_pdf_tables.TableCell
documents_pdf_tables.TableRegion
documents_pdf_tables.CaptionRegion
documents_pdf_tables.PdfRegions
email.EmailAttachmentReport
email.RedactedEmail
frame_sampling_manifest.FrameSamplingManifest
gif_dimensions.GifDimensions
image.ImageMetadataReport
image.ResidualPhi
image.ResidualPhiReport
image.RedactedImage
image_geometry.ImageGeometry
image_header.ImageHeader
layout.LayoutSpan
layout.LayoutBlock
layout.LayoutColumn
layout.LayoutBand
layout.LayoutTableCell
layout.LayoutTable
layout.LayoutQuality
layout.LayoutDocument
manifest_profiles.ValidationFinding
manifest_profiles.ManifestProfile
metadata_scrub.MetadataFinding
metadata_scrub.ResidualMetadataReport
metadata_scrub.MetadataScrubResult
notices.MultimodalNotice
notices.NoticeBoundOutput
ocr.OcrWord
ocr.OcrResult
ocr.OcrEngine
orientation_preflight.ImageTransform
orientation_preflight.OrientationReport
page_rotation.PageSize
page_rotation.PageTransform
page_windows.PageBatch
page_windows.PageBatchPlan
pdf_geometry.PdfPageGeometry
pdf_geometry.PdfGeometryReport
pptx.PptxRunRange
pptx.PptxRedaction
preflight.PreflightFinding
preflight.PreflightReport
processing_diff.MediaTypeDelta
processing_diff.OutcomeDelta
processing_diff.AbstentionDelta
processing_diff.DigestChange
processing_diff.ProcessingDiff
processing_summary.AssetProcessingResult
processing_summary.MediaTypeTotals
processing_summary.OutcomeCount
processing_summary.AbstentionCount
processing_summary.AssetDigestEntry
processing_summary.ProcessingSummary
provider_result.ProviderResultEnvelope
render_pdf.PdfRedactionRegion
render_pdf.PdfPageFidelity
render_pdf.PdfLayoutFidelityReport
render_pdf.PdfRedactionResult
render_raster.RasterRedactionPage
render_raster.RasterExportResult
sms_messages.ShortTextPreset
sms_messages.SMSRedactionSummary
sms_messages.RedactedSMSExport
tabular_csv.ColumnDecision
tabular_csv.TableView
tabular_csv.RedactedTable
tiff_metadata.TiffMetadata
trace_recovery.TraceRecoveryJournal
trace_recovery.TraceRedactionResult
verify_pdf.RegionFidelity
verify_pdf.PdfFidelityReport
verify_pdf.TextRemovalRegion
verify_pdf.PdfTextRemovalReport
wav_metadata.WavMetadata
webp_dimensions.WebpDimensions
xlsx.XlsxCellRedaction
xlsx.XlsxRedactionResult
vlm.CompassImageBatch
vlm.CompassPreparedInput
""".split()
)
