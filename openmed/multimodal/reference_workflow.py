"""Offline reference orchestration with injected, explicitly synthetic providers.

This source-checkout example is an integration harness, not a clinical recorder.
Native model qualification is separate. It never accepts reviewer approval,
exports protected content, or certifies erasure of immutable Python objects.
"""

from __future__ import annotations

import hashlib
import io
import json
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openmed.multimodal.arbitration.ocr_vlm_router import (
    AgreementPolicy,
    ReadingCandidate,
    ReadingPair,
    ReadingSource,
    SemanticType,
    TextNormalization,
    route_ocr_vlm,
)
from openmed.multimodal.capability_matrix import (
    ModalityCapability,
    ModalityProfile,
    ProviderCapabilityEnvelope,
)
from openmed.multimodal.evidence import (
    DerivationEdge,
    EdgeKind,
    EvidenceNode,
    FusedEvidenceGraph,
    NodeKind,
    PageGeometry,
)
from openmed.multimodal.preflight import PreflightStatus, preflight_asset
from openmed.multimodal.privacy.pixel_coverage import (
    LocalPixelDetector,
    PixelFramePlan,
    audit_pixel_redaction,
    inventory_pixel_frames,
)
from openmed.multimodal.provenance.clock_alignment import (
    AlignmentPolicy,
    AlignmentState,
    AudioChunkEvidence,
    ClockAnchor,
    ClockBinding,
    ClockEpoch,
    ImageEvidence,
    TranscriptEvidence,
    align_evidence,
    audio_chunk_digest,
    transcript_source_digest,
)
from openmed.multimodal.runtime.thermal_gate import (
    EdgeBudget,
    GateReason,
    LocalMode,
    MemoryPressure,
    PlatformObservation,
    ThermalGate,
    ThermalState,
    vlm_adapter_digest,
)
from openmed.multimodal.streaming.ambient.speaker_roles import (
    RoleObservation,
    TranscriptSource,
    assess_speaker_roles,
)
from openmed.multimodal.streaming.privacy.audio_retention import (
    ArtifactKind,
    AudioRetentionSession,
    ExpectedAudioArtifact,
    SessionEnd,
)
from openmed.multimodal.streaming.privacy.withholding_buffer import (
    LocalPhiScreener,
    WithholdingBuffer,
    WithholdingPolicy,
)
from openmed.multimodal.streaming.session import (
    AmbientArtifactKind,
    AmbientSession,
    AmbientSessionPolicy,
)
from openmed.multimodal.streaming.speech.providers import (
    AudioChunk,
    FinalizedSegment,
    SpeechProviderDeclaration,
    StreamingAsrSession,
    StreamState,
)
from openmed.multimodal.streaming.speech.speaker_aliases import SpeakerAliases
from openmed.multimodal.streaming.transcript import RevisionLedger, TranscriptToken
from openmed.multimodal.vision.providers.base import (
    Confidence,
    EdgeVLMProvider,
    ImageInput,
    Prompt,
    RuntimeBackend,
    VLMRequest,
)

_NS = 1_000_000_000


def _ref(number: int) -> str:
    return f"{number:032x}"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceInputs:
    """Synthetic fixture and trusted local callbacks for the bounded harness.

    Args:
        image: One synthetic 64 by 48 PNG, at most 64 KiB.
        audio: One synthetic mono PCM16 WAV, 16 kHz and one second.
        detector: Fixture-only pixel marker detector, also used on redacted output.
        vlm: Synthetic backend consuming the actual redacted PNG.
        speech: Explicit fixture-replay speech declaration; never a native adapter.
        transcribe: Local fixture callback consuming the owned PCM view.
        screener: Fixture-only local transcript identifier screen.
        clock: Host monotonic clock; deterministic fixture clocks are permitted.
        observe: Fresh host resource observations for the actual VLM gate.
        checkpoint: Local event hook; return True to cancel before that stage.
        ocr_text: Fixture OCR reading for the original lower-page region.
        masks: Original pixel redaction rectangles; no transforms are applied.

    Callbacks are trusted code, not sandboxed. The harness forbids native provider
    declarations because its thresholds and capture clocks are synthetic controls.
    All supplied objects and immutable decoder/provider copies remain outside the
    managed-buffer destruction guarantee. Do not supply patient data.
    """

    image: bytes = field(repr=False)
    audio: bytes = field(repr=False)
    detector: LocalPixelDetector = field(repr=False)
    vlm: EdgeVLMProvider = field(repr=False)
    speech: SpeechProviderDeclaration = field(repr=False)
    transcribe: Callable[[memoryview], FinalizedSegment] = field(repr=False)
    screener: LocalPhiScreener = field(repr=False)
    clock: Callable[[], float] = field(repr=False)
    observe: Callable[[], PlatformObservation] = field(repr=False)
    checkpoint: Callable[[str], bool] = field(repr=False)
    ocr_text: str = field(repr=False)
    masks: tuple[tuple[int, int, int, int], ...] = ((4, 4, 20, 12),)


class _Hold(Exception):
    """Internal fixed reason code, never a provider exception."""


def _manifest(data: bytes, media_type: str, **metadata: int) -> dict[str, Any]:
    return {
        "asset_id": "synthetic-reference",
        "media_type": media_type,
        "sha256": _sha(data),
        "byte_size": len(data),
        **metadata,
    }


def _binding(epoch: ClockEpoch, digest: str) -> ClockBinding:
    return ClockBinding(
        source_digest=digest,
        clock_ref=epoch.clock_ref,
        epoch_ref=epoch.epoch_ref,
        evidence_digest=_sha(b"synthetic capture manifest v1"),
        local_start_ns=0,
        local_end_ns=_NS,
        monotonic_start_ns=0,
        timing_uncertainty_ns=0,
    )


def run_reference_workflow(inputs: ReferenceInputs) -> dict[str, Any]:
    """Run synthetic components to a content-free, unapproved review handoff.

    Args:
        inputs: Synthetic fixtures and local callbacks described by ReferenceInputs.

    Returns:
        Stage receipts, source coordinates, clock lineage and scoped cleanup
        accounting. ``complete`` and ``clinical_write_allowed`` are always false.
        ``reached_review`` means all processing stages ran, not clinical approval.
        Failures discard review output and contain only fixed reason codes.
    """
    report: dict[str, Any] = {
        "schema": "openmed.synthetic_reference.v1",
        "provider_mode": "synthetic_fixture_replay",
        "complete": False,
        "clinical_write_allowed": False,
        "requires_review": True,
        "reached_review": False,
        "status": "held",
        "stages": [],
        "review": None,
    }
    session = AmbientSession(
        _ref(1),
        (_ref(2),),
        policy=AmbientSessionPolicy(revision=_sha(b"synthetic reference policy v1")),
        clock=inputs.clock,
        automatic_expiry=False,
    )
    retention: AudioRetentionSession | None = None
    stream: StreamingAsrSession | None = None
    withholding: WithholdingBuffer | None = None
    pcm = bytearray()

    def stage(name: str) -> None:
        report["stages"].append(name)
        if inputs.checkpoint(name):
            session.transition("interrupt")
            raise _Hold("cancelled")
        # The synchronous host checks the original deadline at every boundary.
        if session.snapshot()["state"] != "capturing":
            raise _Hold("session_not_capturing")

    try:
        if (
            inputs.vlm.metadata.backend is not RuntimeBackend.SYNTHETIC
            or inputs.speech.provenance.provider_id != "synthetic-replay"
            or len(inputs.image) > 65_536
            or len(inputs.audio) > 65_536
        ):
            raise _Hold("unsupported_reference_input")
        session.consent(_ref(2))
        session.transition("start")
        # Input fixtures and immutable decoder/provider outputs are external.
        # A successful overwrite of our copies must never certify all-data erasure.
        session.declare_external_copy()
        stage("preflight")
        preflights = (
            preflight_asset(
                _manifest(inputs.image, "image/png", width=64, height=48), inputs.image
            ),
            preflight_asset(
                _manifest(inputs.audio, "audio/wav", duration_seconds=1), inputs.audio
            ),
        )
        report["preflight"] = [item.to_dict() for item in preflights]
        if any(item.status is not PreflightStatus.ACCEPT for item in preflights):
            raise _Hold("preflight_refused")

        stage("decode")
        from PIL import Image

        with Image.open(io.BytesIO(inputs.image)) as decoded:
            if decoded.size != (64, 48) or decoded.format != "PNG":
                raise _Hold("image_geometry_refused")
        frames = inventory_pixel_frames(inputs.image, source_kind="raster")
        if len(frames) != 1:
            raise _Hold("image_geometry_refused")
        with wave.open(io.BytesIO(inputs.audio), "rb") as audio:
            if (
                audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16000
                or audio.getnframes() != 16000
                or audio.getcomptype() != "NONE"
            ):
                raise _Hold("audio_format_refused")
            pcm.extend(audio.readframes(16000))
        if len(pcm) != 32000:
            raise _Hold("audio_truncated")
        retention = AudioRetentionSession(
            _ref(1),
            (
                ExpectedAudioArtifact(_ref(3), ArtifactKind.BUFFER),
                ExpectedAudioArtifact(_ref(4), ArtifactKind.EXTERNAL),
            ),
            clock=inputs.clock,
            automatic_expiry=False,
        )
        retention.capture(_ref(3), pcm)
        pcm[:] = bytes(len(pcm))

        stage("redaction")
        pixels = audit_pixel_redaction(
            inputs.image,
            (PixelFramePlan(reference=frames[0].reference, masks=inputs.masks),),
            detector=inputs.detector,
            residual_detector=inputs.detector,
        )
        report["pixel_redaction"] = json.loads(pixels.to_json())
        if not pixels.accepted:
            raise _Hold("pixel_redaction_refused")
        image = ImageInput(
            data=pixels.outputs[0], media_type="image/png", width=64, height=48
        )
        session.retain(_ref(5), AmbientArtifactKind.CACHE, image.data)

        stage("visual_inference")
        gate = ThermalGate(
            ModalityProfile("synthetic-reference", "image"),
            (
                LocalMode(
                    adapter_digest=vlm_adapter_digest(inputs.vlm.metadata),
                    qualification_digest=_sha(b"synthetic controls, not qualification"),
                    capabilities=ProviderCapabilityEnvelope(
                        provider_id="synthetic",
                        capability_version="1.0.0",
                        capabilities=(ModalityCapability("image"),),
                    ),
                    required_available_bytes=1_000_000,
                    max_thermal=ThermalState.NOMINAL,
                    max_memory_pressure=MemoryPressure.NORMAL,
                ),
            ),
            EdgeBudget(max_latency_ms=1000, max_queue_ms=100, max_output_age_ms=1200),
            inputs.observe,
            clock=inputs.clock,
        )
        outcome = gate.run_vlm(
            (inputs.vlm,),
            VLMRequest(image=image, prompt=Prompt(text="Read the synthetic control.")),
            queued_at=inputs.clock(),
        )
        report["resources"] = json.loads(outcome.report_json)
        if outcome.reason is not GateReason.READY or outcome.value is None:
            raise _Hold("resource_or_provider_refused")
        visual = outcome.value
        if len(visual.evidence) != 1 or visual.abstention is not None:
            raise _Hold("visual_evidence_missing")
        citation = visual.evidence[0]
        region = citation.region
        if citation.start != 0 or citation.end != len(visual.text):
            raise _Hold("visual_evidence_incomplete")
        geometry = PageGeometry(page=0, width=64, height=48)
        # Redaction keeps dimensions and coordinates unchanged. Bind to the
        # original digest, retaining the actual original-to-output mask receipt.
        visual_node = EvidenceNode(
            node_id=_ref(10),
            kind=NodeKind.PIXELS,
            source_digest=_sha(inputs.image),
            geometry=geometry,
            region=(region.x0, region.y0, region.x1, region.y1),
        )
        ocr_node = EvidenceNode(
            node_id=_ref(11),
            kind=NodeKind.OCR,
            source_digest=_sha(inputs.image),
            geometry=geometry,
            region=(4, 24, 60, 40),
            span=(0, len(inputs.ocr_text)),
        )
        routing = route_ocr_vlm(
            (
                ReadingPair(
                    pair_id=_ref(12),
                    ocr=ReadingCandidate(
                        candidate_id=_ref(13),
                        field_id=_ref(15),
                        source=ReadingSource.OCR,
                        text=inputs.ocr_text,
                        semantic_type=SemanticType.TEXT,
                        confidence=Confidence(score=1),
                        evidence=ocr_node,
                    ),
                    vlm=ReadingCandidate(
                        candidate_id=_ref(14),
                        field_id=_ref(15),
                        source=ReadingSource.VLM,
                        text=visual.text,
                        semantic_type=SemanticType.TEXT,
                        confidence=visual.confidence,
                        evidence=visual_node,
                    ),
                ),
            ),
            AgreementPolicy(
                version="synthetic-v1",
                allowed_types=(SemanticType.TEXT,),
                normalization=TextNormalization.EXACT,
                minimum_geometry_iou=1,
                minimum_confidence=1,
                preferred_source=ReadingSource.OCR,
                maximum_review_items=1,
            ),
        )
        graph = FusedEvidenceGraph(
            nodes=(
                visual_node,
                ocr_node,
                EvidenceNode(node_id=_ref(16), kind=NodeKind.FACT),
                EvidenceNode(node_id=_ref(17), kind=NodeKind.CLAIM),
            ),
            edges=(
                DerivationEdge(
                    source=_ref(10), target=_ref(16), kind=EdgeKind.INTERPRETS
                ),
                DerivationEdge(
                    source=_ref(11), target=_ref(16), kind=EdgeKind.EXTRACTS
                ),
                DerivationEdge(
                    source=_ref(16), target=_ref(17), kind=EdgeKind.SUPPORTS
                ),
            )
            + (
                (
                    DerivationEdge(
                        source=_ref(10), target=_ref(11), kind=EdgeKind.CONFLICTS
                    ),
                )
                if routing.review_queue
                else ()
            ),
        )

        stage("audio_inference")

        def audio_resources() -> None:
            sample = inputs.observe()
            now = inputs.clock()
            if (
                not 0 <= now - sample.observed_at <= 0.1
                or sample.thermal is not ThermalState.NOMINAL
                or sample.memory_pressure is not MemoryPressure.NORMAL
                or sample.available_bytes is None
                or sample.available_bytes < 1_000_000
            ):
                raise _Hold("audio_resources_refused")

        audio_resources()
        asr_started = inputs.clock()
        stream = StreamingAsrSession(inputs.speech)
        chunk = retention.infer(
            _ref(3),
            lambda view: AudioChunk(
                sequence=0,
                payload=bytes(view),
                sample_rate_hz=16000,
                channel_count=1,
                duration_ms=1000,
            ),
        )
        stream.push_chunk(chunk)
        segment = retention.infer(_ref(3), inputs.transcribe)
        audio_resources()
        if not 0 <= inputs.clock() - asr_started <= 1:
            raise _Hold("audio_latency_refused")
        stream.submit_segment(segment)
        stream.drain()
        stream.finalize()
        if (segment.start_ms, segment.end_ms) != (0, 1000) or not segment.text:
            raise _Hold("speech_timing_refused")
        if (
            len(segment.token_times) != 1
            or segment.token_times[0].token != segment.text
            or segment.token_times[0].start_ms != segment.start_ms
            or segment.token_times[0].end_ms != segment.end_ms
        ):
            raise _Hold("speech_token_lineage_refused")
        source_token = segment.token_times[0]
        token = TranscriptToken(
            token_ref=_ref(21),
            start=0,
            end=len(segment.text),
            start_ms=source_token.start_ms,
            end_ms=source_token.end_ms,
        )
        ledger = RevisionLedger(_ref(1), _ref(20), len(segment.text), (token,))
        span = ledger.capture_span(_ref(22), 0, len(segment.text))
        ledger.finalize(expected_revision=0)

        stage("transcript_screen")
        withholding = WithholdingBuffer(
            _ref(1),
            inputs.screener,
            clock=inputs.clock,
            policy=WithholdingPolicy(trailing_segments=0),
        )
        if not withholding.append(
            _ref(20), segment.text, (token,)
        ) or not withholding.finalize(_ref(20), expected_revision=0):
            raise _Hold("transcript_refused")
        speech_gate = ThermalGate(
            ModalityProfile("synthetic-reference", "audio"),
            (
                LocalMode(
                    adapter_digest=withholding.screen_artifact_digest,
                    qualification_digest=_sha(b"synthetic screen controls only"),
                    capabilities=ProviderCapabilityEnvelope(
                        provider_id="synthetic",
                        capability_version="1.0.0",
                        capabilities=(ModalityCapability("audio"),),
                    ),
                    required_available_bytes=1_000_000,
                    max_thermal=ThermalState.NOMINAL,
                    max_memory_pressure=MemoryPressure.NORMAL,
                ),
            ),
            EdgeBudget(max_latency_ms=1000, max_queue_ms=100, max_output_age_ms=1200),
            inputs.observe,
            clock=inputs.clock,
        )
        speech_outcome = speech_gate.release_speech(
            withholding, queued_at=inputs.clock()
        )
        report["speech_resources"] = json.loads(speech_outcome.report_json)
        if (
            speech_outcome.reason is not GateReason.READY
            or speech_outcome.value is None
        ):
            raise _Hold("speech_release_refused")
        released = speech_outcome.value
        if len(released) != 1:
            raise _Hold("transcript_phi_unresolved")
        session.retain(
            _ref(23), AmbientArtifactKind.TRANSCRIPT, released[0].text.encode()
        )

        stage("evidence_fusion")
        epoch = ClockEpoch(
            clock_ref=_ref(30),
            epoch_ref=_ref(31),
            wall_domain_digest="a" * 64,
            calibration_digest="b" * 64,
            anchors=(ClockAnchor(0, 0, 0), ClockAnchor(_NS, _NS, 0)),
            maximum_rate_error_ppb=0,
            nonlinearity_ns=0,
            breaks=(),
        )
        image_evidence = ImageEvidence(
            visual_node,
            0,
            _NS,
            _sha(b"synthetic exposure v1"),
            _binding(epoch, _sha(inputs.image)),
        )
        alignment_policy = AlignmentPolicy(
            version="synthetic-v1",
            evidence_digest="c" * 64,
            maximum_uncertainty_ns=0,
            maximum_anchor_gap_ns=_NS,
        )
        aligned = tuple(
            align_evidence(source, image_evidence, epoch, epoch, alignment_policy)
            for source in (
                AudioChunkEvidence(
                    chunk, 0, _binding(epoch, audio_chunk_digest(chunk))
                ),
                TranscriptEvidence(
                    ledger, span, _binding(epoch, transcript_source_digest(ledger))
                ),
            )
        )
        if any(item.state is not AlignmentState.ALIGNED for item in aligned):
            raise _Hold("clock_lineage_refused")
        aliases = SpeakerAliases(_ref(1))
        aliases.apply_revision(0, (_ref(40),))
        roles = assess_speaker_roles(
            aliases,
            (TranscriptSource(ledger=ledger, text=segment.text),),
            (
                RoleObservation(
                    observation_id=_ref(41),
                    cluster_ref=_ref(40),
                    diarization_revision=0,
                    citation=span,
                    hypotheses=(),
                ),
            ),
        )
        stage("review")
        session.transition("stop")
        report["review"] = {
            "visual_graph": graph.to_dict(),
            "visual_routing": json.loads(routing.to_json()),
            "clock_alignment": [json.loads(item.to_report_json()) for item in aligned],
            "transcript": released[0].to_dict(),
            "transcript_lineage": ledger.to_dict(),
            "speaker_roles": json.loads(roles.to_json()),
            "note_created": False,
            "clinical_approval": False,
        }
        report["status"] = "review_required"
        report["reached_review"] = True
    except _Hold as error:
        report["reason"] = str(error)
    except Exception:
        # No exception text, original traceback, protected provider value or path.
        report["reason"] = "component_failure"
    finally:
        pcm[:] = bytes(len(pcm))
        cleanup_failed = False
        if withholding is not None:
            try:
                withholding.close()
            except Exception:
                cleanup_failed = True
        if stream is not None:
            try:
                if stream.state is StreamState.OPEN:
                    stream.cancel()
                report["speech"] = stream.report().to_dict()
            except Exception:
                cleanup_failed = True
        if retention is not None:
            try:
                retention.finish(
                    SessionEnd.FINALIZED
                    if report["reached_review"]
                    else SessionEnd.CANCELLED
                )
                audit = retention.audit()
                report["audio_retention"] = json.loads(audit.to_json())
                if not any(
                    item.artifact_ref == _ref(3)
                    and item.action.value == "owned_buffer_overwritten"
                    for item in audit.receipts
                ):
                    cleanup_failed = True
            except Exception:
                cleanup_failed = True
            finally:
                try:
                    retention.close()
                except Exception:
                    cleanup_failed = True
        try:
            session.delete()
            report["session"] = session.snapshot()
            if report["session"]["owned_bytes"]:
                cleanup_failed = True
        except Exception:
            cleanup_failed = True
        if cleanup_failed:
            report.update(
                status="held",
                reached_review=False,
                review=None,
                reason="cleanup_failed",
            )
        report["cleanup_failed"] = cleanup_failed
        report["all_data_erased"] = False
        report["cleanup_scope"] = (
            "owned mutable copies only; external fixture and immutable copies excluded"
        )
    return report
