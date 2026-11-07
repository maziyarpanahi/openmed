"""Offline mandatory notice, frozen wording and public result inventory gates."""

from __future__ import annotations

import ast
import hashlib
import importlib
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
    if cls is VisionLanguageGeneration:
        assert cls(**kwargs).notice is NOTICE_CATALOG[NoticeKind.VISUAL_DESCRIPTION]
        with pytest.raises(TypeError):
            cls(**kwargs, notice=None)
        return
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
    assert "Ada" not in str(generation)
    assert "Ada" in generation.render_text()
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
    covered.update(DECLARED_NOTICE_TYPES)
    unclassified = discovered - covered - NON_CLINICAL_TYPES
    assert not unclassified, (
        f"New public result requires notice classification: {sorted(unclassified)}"
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
    for name, kind in DECLARED_NOTICE_TYPES.items():
        module, class_name = name.rsplit(".", 1)
        result_type = getattr(
            importlib.import_module(f"openmed.multimodal.{module}"), class_name
        )
        assert result_type.review_notice is NOTICE_CATALOG[kind]


def test_inventory_detects_undeclared_new_measurement_result():
    discovered = _public_data_classes(
        "@dataclass\nclass NewIntervalResult:\n    milliseconds: float\n", "new_module"
    )
    with pytest.raises(AssertionError, match="notice classification"):
        _assert_classified(discovered)


DECLARED_NOTICE_TYPES = {
    "ecg.interval_candidates.IntervalCandidates": NoticeKind.MEASUREMENT,
    "ecg.interval_candidates.TimingCandidate": NoticeKind.MEASUREMENT,
    "ecg.signal_quality.GatedMeasurement": NoticeKind.MEASUREMENT,
    "streaming.ambient.note_evidence.AmbientNoteDraft": NoticeKind.DRAFT,
    "streaming.ambient.note_evidence.ReviewedAmbientNote": NoticeKind.DRAFT,
    "vision.dicom.series_sampler.SampledSeriesInference": NoticeKind.VISUAL_DESCRIPTION,
    "vision.pathology.tile_provenance.SlideAggregation": NoticeKind.MEASUREMENT,
    "vision.providers.base.VLMResult": NoticeKind.VISUAL_DESCRIPTION,
    "vision.safety.quality_gate.QualityReport": NoticeKind.VISUAL_DESCRIPTION,
}


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
aecg.AecgLead
aecg.AecgLimits
aecg.AecgRecord
aecg.AecgSeries
apple_contract.LocalMultimodalAdapter
apple_contract.MultimodalPacket
arbitration.ocr_vlm_router.AgreedValues
arbitration.ocr_vlm_router.AgreementPolicy
arbitration.ocr_vlm_router.ReadingCandidate
arbitration.ocr_vlm_router.ReadingComparison
arbitration.ocr_vlm_router.ReadingPair
arbitration.ocr_vlm_router.RoutingReport
asr_hallucination.AsrHypothesisSegment
asr_hallucination.AsrSegmentStatistics
asr_hallucination.HallucinationDecision
asr_hallucination.HallucinationReport
asr_hallucination.HallucinationThresholds
asr_hallucination.VoiceActivityInterval
asr_hallucination.VoiceActivityTimeline
audio_sample_conversion.ConversionBudget
audio_sample_conversion.ConversionReport
audio_sample_conversion.ConvertedAudio
batch_preflight.BatchAssetResult
batch_preflight.BatchPreflightAsset
batch_preflight.BatchPreflightFinding
batch_preflight.BatchPreflightLimits
batch_preflight.BatchPreflightReport
bigtiff_metadata.BigTiffLevel
bigtiff_metadata.BigTiffLimits
bigtiff_metadata.BigTiffPyramid
capability_matrix.CompatibilityEntry
capability_matrix.ModalityCapability
capability_matrix.ModalityProfile
capability_matrix.ProviderCapabilityEnvelope
decoder_guard.DecodeReport
decoder_guard.DecoderLimits
dicom_transfer_syntax.TransferSyntaxCapability
dicom_transfer_syntax.TransferSyntaxReport
ecg.annotation_privacy.AnnotationField
ecg.annotation_privacy.AnnotationPage
ecg.annotation_privacy.AnnotationPlan
ecg.annotation_privacy.AnnotationPolicy
ecg.annotation_privacy.AnnotationReview
ecg.annotation_privacy.EcgAnnotationSource
ecg.annotation_privacy.FieldRedaction
ecg.annotation_privacy.LocalAnnotationDetector
ecg.annotation_privacy.LocalAnnotationOcr
ecg.annotation_privacy.PageRegion
ecg.annotation_privacy.PixelRedaction
ecg.annotation_privacy.RedactedEcgAnnotations
ecg.evidence_windows.EcgEvidenceWindows
ecg.evidence_windows.MeasurementWindow
ecg.evidence_windows.WindowFiducial
ecg.evidence_windows.WindowPolicy
ecg.evidence_windows.WindowSamples
ecg.interval_candidates.BeatEvidence
ecg.interval_candidates.IntervalPolicy
ecg.interval_candidates.LeadIntervals
ecg.interval_candidates.SampleBound
ecg.interval_candidates.SampleRange
ecg.interval_candidates.WaveBoundary
ecg.lead_layout.LeadFinding
ecg.lead_layout.LeadLayout
ecg.lead_layout.LeadLayoutReport
ecg.lead_layout.LeadObservation
ecg.lead_layout.LeadSlot
ecg.signal_quality.AcquisitionEvidence
ecg.signal_quality.CalibrationSegment
ecg.signal_quality.LeadCalibration
ecg.signal_quality.LeadQuality
ecg.signal_quality.QualityFinding
ecg.signal_quality.QualityLimits
ecg.signal_quality.SignalQualityPolicy
ecg.signal_quality.SignalQualityReport
ecg.waveform.EcgLead
ecg.waveform.EcgProvenance
ecg.waveform.EcgWaveform
edf.EdfAnnotation
edf.EdfGap
edf.EdfIdentityStatus
edf.EdfLimits
edf.EdfRecordWindow
edf.EdfRecording
edf.EdfSignal
edf.EdfSignalWindow
evidence.fused_graph.ClaimAssessment
evidence.fused_graph.DerivationEdge
evidence.fused_graph.EvidenceNode
evidence.fused_graph.Finding
evidence.fused_graph.FusedEvidenceGraph
evidence.fused_graph.PageGeometry
generated_text_guard.AssetIdentifierDigest
generated_text_guard.GuardPolicy
generated_text_guard.GuardedTextResult
generated_text_guard.PHISpan
preprocessing.PreprocessingEvidence
preprocessing.PreprocessingOutput
preprocessing.PreprocessingProfile
preprocessing.PreprocessingQualification
privacy.mask_lineage.Affine
privacy.mask_lineage.CoordinateSpace
privacy.mask_lineage.MaskLineageResult
privacy.mask_lineage.MaskProjection
privacy.mask_lineage.Rectangle
privacy.mask_lineage.TransformStep
privacy.pixel_coverage.LocalPixelDetector
privacy.pixel_coverage.PixelCoverageLimits
privacy.pixel_coverage.PixelCoverageResult
privacy.pixel_coverage.PixelFrame
privacy.pixel_coverage.PixelFramePlan
privacy.pixel_coverage.PixelInspection
provenance.clock_alignment.AlignmentPolicy
provenance.clock_alignment.AlignmentResult
provenance.clock_alignment.AudioChunkEvidence
provenance.clock_alignment.AudioPlanEvidence
provenance.clock_alignment.ClockAnchor
provenance.clock_alignment.ClockBinding
provenance.clock_alignment.ClockBreak
provenance.clock_alignment.ClockCalibration
provenance.clock_alignment.ClockEpoch
provenance.clock_alignment.ClockFit
provenance.clock_alignment.EcgEvidence
provenance.clock_alignment.ImageEvidence
provenance.clock_alignment.NanosecondBound
provenance.clock_alignment.TranscriptEvidence
reference_workflow.ReferenceInputs
runtime.thermal_gate.EdgeBudget
runtime.thermal_gate.LocalMode
runtime.thermal_gate.PlatformObservation
runtime.thermal_gate.ThermalOutcome
runtime_session.AssetHandle
runtime_session.ProviderLease
runtime_session.SessionEvent
runtime_session.SessionLimits
runtime_session.SessionResult
speech_endpoints.EndpointEvent
speech_endpoints.LocalActivityDetector
streaming.ambient.note_evidence.NoteCitation
streaming.ambient.note_evidence.NoteExtractionRequest
streaming.ambient.note_evidence.NoteExtractionStep
streaming.ambient.note_evidence.NoteOffsetMapping
streaming.ambient.note_evidence.NotePolicy
streaming.ambient.note_evidence.NoteProposal
streaming.ambient.note_evidence.NoteProviderDeclaration
streaming.ambient.note_evidence.NoteSection
streaming.ambient.note_evidence.NoteStatement
streaming.ambient.note_evidence.NoteVerificationRequest
streaming.ambient.note_evidence.NoteVerificationResult
streaming.ambient.note_evidence.ScreenedRoleEvidence
streaming.ambient.note_evidence.StatementSupport
streaming.ambient.note_evidence.StatementVerification
streaming.ambient.speaker_roles.ReviewedRoleSpan
streaming.ambient.speaker_roles.RoleAssessment
streaming.ambient.speaker_roles.RoleAttributionReview
streaming.ambient.speaker_roles.RoleBoundDraft
streaming.ambient.speaker_roles.RoleCheckedNote
streaming.ambient.speaker_roles.RoleHypothesis
streaming.ambient.speaker_roles.RoleObservation
streaming.ambient.speaker_roles.RoleReport
streaming.ambient.speaker_roles.TranscriptSource
streaming.privacy.audio_muting.MutedAudioReport
streaming.privacy.audio_muting.RetainedClipAlignment
streaming.privacy.audio_retention.AudioRetentionAudit
streaming.privacy.audio_retention.AudioRetentionPolicy
streaming.privacy.audio_retention.DestructionReceipt
streaming.privacy.audio_retention.ExpectedAudioArtifact
streaming.privacy.withholding_buffer.PhiRedactionSpan
streaming.privacy.withholding_buffer.PhiScreenDeclaration
streaming.privacy.withholding_buffer.PhiScreenRequest
streaming.privacy.withholding_buffer.PhiScreenResult
streaming.privacy.withholding_buffer.ProtectedSpeechSegment
streaming.privacy.withholding_buffer.RedactionMapping
streaming.privacy.withholding_buffer.ReleasedSpeechSegment
streaming.privacy.withholding_buffer.SegmentPhiDecision
streaming.privacy.withholding_buffer.WithholdingPolicy
streaming.providers.AsrUpdate
streaming.providers.OfflineAsrPolicy
streaming.providers.ScreenedAsrSegment
streaming.session.AmbientReference
streaming.session.AmbientReviewCopy
streaming.session.AmbientSessionPolicy
streaming.speech.diarization.AcousticTurn
streaming.speech.diarization.DiarizationBatch
streaming.speech.diarization.DiarizationRequest
streaming.speech.providers.AudioChunk
streaming.speech.providers.FinalizedSegment
streaming.speech.providers.LanguageConfidence
streaming.speech.providers.PartialHypothesis
streaming.speech.providers.ProviderProvenance
streaming.speech.providers.RegisteredSpeechProvider
streaming.speech.providers.SpeechProviderDeclaration
streaming.speech.providers.SpeechStreamLimits
streaming.speech.providers.StreamingAsrSessionReport
streaming.speech.providers.TokenTime
streaming.speech.speaker_aliases.LineageEvent
streaming.speech.speaker_aliases.RoleReview
streaming.transcript.revision_ledger.MappedSpan
streaming.transcript.revision_ledger.SpanReference
streaming.transcript.revision_ledger.TranscriptToken
streaming.transcript.window_stitching.AlignedWindow
streaming.transcript.window_stitching.ReconciledWindows
streaming.transcript.window_stitching.TranscriptEdit
transcript_language.TranscriptLanguageDecision
transcript_language.TranscriptLanguageRun
transcript_language.TranscriptPHISpan
vision.dicom.pixel_provenance.PixelHeader
vision.dicom.pixel_provenance.PixelProvenance
vision.dicom.pixel_provenance.PixelProvenanceLimits
vision.dicom.pixel_provenance.PixelRenderSpec
vision.dicom.pixel_provenance.PixelStage
vision.dicom.pixel_provenance.ProvenancedFrame
vision.dicom.pixel_provenance.ProvenancedInference
vision.dicom.pixel_provenance.RegionProvenance
vision.dicom.series_sampler.DisplayWindow
vision.dicom.series_sampler.SampledFrameInference
vision.dicom.series_sampler.SamplingPolicy
vision.dicom.series_sampler.SeriesSamplingPlan
vision.dicom.series_sampler.SourceFrame
vision.documents.page_consistency.ConsistencyPolicy
vision.documents.page_consistency.IdentityObservation
vision.documents.page_consistency.PacketConsistency
vision.documents.page_consistency.PageFinding
vision.documents.page_consistency.PageGroup
vision.documents.page_consistency.PageNumber
vision.documents.page_consistency.PageObservation
vision.documents.page_consistency.TextDigest
vision.forms.field_grounding.FieldAssociation
vision.forms.field_grounding.FormAssociationReview
vision.forms.field_grounding.FormGroundingReport
vision.forms.field_grounding.FormLabel
vision.forms.field_grounding.FormSection
vision.forms.field_grounding.FormValue
vision.forms.field_grounding.GroundedFormExport
vision.forms.field_grounding.GroundingPolicy
vision.forms.field_grounding.ValueGrounding
vision.forms.field_grounding.VisualForm
vision.handwriting.uncertainty.HandwritingCandidate
vision.handwriting.uncertainty.HandwritingExport
vision.handwriting.uncertainty.HandwritingReview
vision.handwriting.uncertainty.ReviewedHandwriting
vision.handwriting.uncertainty.TranscriptionAlternative
vision.pathology.slide_tiler.PyramidLevel
vision.pathology.slide_tiler.SlideTile
vision.pathology.slide_tiler.SlideTilePlan
vision.pathology.slide_tiler.TilePolicy
vision.pathology.slide_tiler.TissueMask
vision.pathology.tile_provenance.TileFeatures
vision.pathology.tile_provenance.TileTransform
vision.providers.base.Confidence
vision.providers.base.ImageInput
vision.providers.base.ImageRegion
vision.providers.base.LocalProviderMetadata
vision.providers.base.Prompt
vision.providers.base.RuntimeMetadata
vision.providers.base.VLMRequest
vision.providers.base.VisualEvidence
vision.providers.local_assets.LocalVLMAssets
vision.runtime.resource_planner.DeviceTier
vision.runtime.resource_planner.PreprocessingPlan
vision.runtime.resource_planner.TierCandidate
vision.runtime.resource_planner.VLMExecutionPlan
vision.runtime.resource_planner.VLMExecutionPolicy
vision.runtime.resource_planner.VLMResourceEstimate
vision.safety.instruction_injection.DocumentRegion
vision.safety.instruction_injection.DocumentScreeningReport
vision.safety.instruction_injection.InstructionCue
vision.safety.instruction_injection.InstructionPolicy
vision.safety.instruction_injection.PreparedDocumentPrompt
vision.safety.instruction_injection.RegionAssessment
vision.safety.instruction_injection.RegionRelease
vision.safety.quality_gate.QualityMetrics
vision.safety.quality_gate.QualityPolicy
voice_embeddings.EmbeddingDestructionReceipt
voice_embeddings.VoiceSimilarity
wfdb.WfdbAnnotations
wfdb.WfdbLimits
wfdb.WfdbRecord
wfdb.WfdbSignal
""".split()
)
