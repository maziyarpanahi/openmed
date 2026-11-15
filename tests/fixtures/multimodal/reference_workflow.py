"""Generated synthetic PNG/WAV and digest-bound replay adapters; no model weights."""

import hashlib
import io
import math
import struct
import wave
from dataclasses import replace

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from openmed.multimodal.privacy.pixel_coverage import (
    LocalPixelDetector,
    PixelInspection,
)
from openmed.multimodal.reference_workflow import ReferenceInputs
from openmed.multimodal.runtime.thermal_gate import (
    MemoryPressure,
    PlatformObservation,
    ThermalState,
)
from openmed.multimodal.streaming.privacy.withholding_buffer import (
    PhiRedactionSpan,
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
)
from openmed.multimodal.streaming.speech.providers import (
    FinalizedSegment,
    LanguageConfidence,
    ProviderProvenance,
    SpeechProviderDeclaration,
    SpeechStreamLimits,
    TokenTime,
)
from openmed.multimodal.vision.providers.base import (
    Confidence,
    ImageRegion,
    LocalProviderMetadata,
    RuntimeBackend,
    RuntimeMetadata,
    VisualEvidence,
    VLMResult,
)

PRIVATE = "SYNTHETIC-PRIVATE-REFERENCE"
CONTROL = "SYNTHETIC CONTROL"
TRANSCRIPT = f"{PRIVATE}. Synthetic speaker requests review."
MARK = (4, 4, 20, 12)


def image_bytes(*, redacted=False, size=(64, 48)):
    image = Image.new("RGB", size, (30, 30, 30))
    image.paste((0, 0, 0) if redacted else (255, 255, 255), MARK)
    image.paste((0, 128, 0), (4, 24, 60, 40))
    info = PngInfo()
    info.add_text("Comment", PRIVATE)
    stream = io.BytesIO()
    image.save(stream, format="PNG", **({} if redacted else {"pnginfo": info}))
    image.close()
    return stream.getvalue()


def audio_bytes():
    # Tone fixture: this is NOT speech and the replay is NOT speech recognition.
    pcm = b"".join(
        struct.pack("<h", round(2000 * math.sin(2 * math.pi * 440 * index / 16000)))
        for index in range(16000)
    )
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm)
    return stream.getvalue(), pcm


class ReferenceControls:
    """Fault injection shared by the real component calls in this replay."""

    def __init__(self, fault=None, observe_buffers=False):
        self.fault = fault
        self.observe_buffers = observe_buffers
        self.now = 0.0
        self.stages = []
        self.vlm_calls = self.asr_calls = self.screen_calls = 0
        self.borrowed_audio = []

    def clock(self):
        return self.now

    def checkpoint(self, stage):
        self.stages.append(stage)
        if self.fault == "expiry" and stage == "transcript_screen":
            self.now = 301.0
        if self.fault == "audio_memory" and stage == "audio_inference":
            self.fault = "memory"
        if self.fault == "screen_thermal" and stage == "transcript_screen":
            self.fault = "thermal"
        return self.fault == "cancel:" + stage

    def observe(self):
        return PlatformObservation(
            observed_at=self.now,
            thermal=ThermalState.SERIOUS
            if self.fault == "thermal"
            else ThermalState.NOMINAL,
            memory_pressure=MemoryPressure.CRITICAL
            if self.fault == "memory"
            else MemoryPressure.NORMAL,
            available_bytes=10_000_000,
        )

    def detect(self, image):
        # Inspect real pixels, including the actual rewritten output.
        pixels = image.convert("RGB")
        try:
            points = [
                (x, y)
                for y in range(image.height)
                for x in range(image.width)
                if min(pixels.getpixel((x, y))) > 250
            ]
        finally:
            pixels.close()
        boxes = (
            ()
            if not points
            else (
                (
                    min(x for x, _ in points),
                    min(y for _, y in points),
                    max(x for x, _ in points) + 1,
                    max(y for _, y in points) + 1,
                ),
            )
        )
        return PixelInspection(boxes, True)


class ReplayVLM:
    """Read only the exact fixture's redacted pixels; return a known control."""

    metadata = LocalProviderMetadata(
        backend=RuntimeBackend.SYNTHETIC,
        model_digest=hashlib.sha256(b"reference pixel replay v1").hexdigest(),
        preprocessing_digest=hashlib.sha256(b"identity geometry v1").hexdigest(),
    )

    def __init__(self, controls):
        self.controls = controls

    def infer(self, request):
        self.controls.vlm_calls += 1
        with Image.open(io.BytesIO(request.image.data)) as actual:
            with Image.open(io.BytesIO(image_bytes(redacted=True))) as expected:
                if actual.tobytes() != expected.tobytes() or actual.info:
                    raise ValueError(PRIVATE)
        if self.controls.fault == "provider_exception":
            raise ValueError(PRIVATE)
        self.controls.now += 0.01
        if self.controls.fault == "post_thermal":
            self.controls.fault = "thermal"
        text = "CONFLICTING CONTROL" if self.controls.fault == "conflict" else CONTROL
        return VLMResult(
            request_digest=request.digest,
            runtime=RuntimeMetadata(
                provider=self.metadata, duration_ms=10, output_tokens=3
            ),
            text=text,
            confidence=Confidence(score=1),
            evidence=(
                VisualEvidence(
                    region=ImageRegion(
                        image_digest=request.image.digest, x0=4, y0=24, x1=60, y1=40
                    ),
                    start=0,
                    end=len(text),
                    confidence=Confidence(score=1),
                ),
            ),
        )


class ReplayScreen:
    declaration = PhiScreenDeclaration(
        artifact_digest=hashlib.sha256(b"known synthetic marker screen v1").hexdigest()
    )

    def __init__(self, controls):
        self.controls = controls

    def screen(self, request):
        self.controls.screen_calls += 1
        return PhiScreenResult(
            request_digest=request.request_digest,
            decisions=tuple(
                SegmentPhiDecision(
                    segment_ref=segment.segment_ref,
                    revision=segment.revision,
                    resolution=PhiResolution.UNRESOLVED
                    if self.controls.fault == "unresolved_phi"
                    else PhiResolution.RESOLVED,
                    redactions=(
                        PhiRedactionSpan(
                            segment.text.index(PRIVATE),
                            segment.text.index(PRIVATE) + len(PRIVATE),
                        ),
                    ),
                )
                for segment in request.segments
            ),
        )


def reference_fixture(fault=None, *, observe_buffers=False):
    """Return a fresh synthetic input and observable local control hooks."""
    controls = ReferenceControls(fault, observe_buffers)
    audio, pcm = audio_bytes()
    pcm_digest = hashlib.sha256(pcm).hexdigest()

    def transcribe(view):
        controls.asr_calls += 1
        # Retain a view only in the test instrumentation to observe real overwrite.
        if controls.observe_buffers:
            controls.borrowed_audio.append(memoryview(view))
        if hashlib.sha256(view).hexdigest() != pcm_digest:
            raise ValueError(PRIVATE)
        if controls.fault == "slow_asr":
            controls.now += 2
        if controls.fault == "asr_memory":
            controls.fault = "memory"
        if controls.fault == "asr_exception":
            raise ValueError(PRIVATE)
        return FinalizedSegment(
            segment_index=0,
            text=TRANSCRIPT,
            start_ms=0,
            end_ms=1000,
            token_times=(
                TokenTime(
                    token_index=0,
                    start_ms=0,
                    end_ms=1000,
                    token=TRANSCRIPT,
                    confidence_ppm=1000000,
                ),
            ),
            language=LanguageConfidence("en", 1000000),
        )

    inputs = ReferenceInputs(
        image=image_bytes(),
        audio=audio,
        detector=LocalPixelDetector(
            digest=hashlib.sha256(b"bright marker v1").hexdigest(),
            inspect=controls.detect,
            local_only=True,
        ),
        vlm=ReplayVLM(controls),
        speech=SpeechProviderDeclaration(
            provenance=ProviderProvenance(
                provider_id="synthetic-replay",
                provider_revision="1.0.0",
                model_fingerprint="sha256:" + pcm_digest,
            ),
            languages=("en",),
            sample_rates_hz=(16000,),
            limits=SpeechStreamLimits(
                max_buffered_chunks=2,
                max_buffered_byte_count=32000,
                max_buffered_duration_ms=1000,
                max_stream_duration_ms=1000,
                max_partial_hypotheses=2,
                max_segments=1,
            ),
        ),
        transcribe=transcribe,
        screener=ReplayScreen(controls),
        clock=controls.clock,
        observe=controls.observe,
        checkpoint=controls.checkpoint,
        ocr_text=CONTROL,
    )
    if fault == "malformed":
        inputs = replace(inputs, image=b"not a PNG " + PRIVATE.encode())
    if fault == "wrong_geometry":
        inputs = replace(inputs, image=image_bytes(size=(65, 48)))
    if fault == "truncated_audio":
        inputs = replace(inputs, audio=audio[:-2])
    if fault == "missing_mask":
        inputs = replace(inputs, masks=())
    return inputs, controls
