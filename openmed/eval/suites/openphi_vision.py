"""Executed synthetic vision privacy controls with independent release axes.

The public corpus measures software wiring and known-marker leakage, not
clinical detector recall or resistance to training contamination. No weights,
restricted datasets, telemetry, or model-license allowlist are supplied.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

from openmed.eval.synthetic_fidelity import FidelityCounts, wilson_interval

_SCHEMA = "openmed.openphi_vision.v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SIZE = (256, 96)
_PUBLIC_BOX = (8, 64, 248, 88)


class VisionGateError(ValueError):
    """Fixed input failure without source, model paths or patient values."""


def _require(condition: bool) -> None:
    if not condition:
        raise VisionGateError("invalid_multimodal_gate_input")


def _digest(value: str) -> None:
    _require(type(value) is str and _HEX.fullmatch(value) is not None)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return _sha(_json(value).encode())


class VisionTrack(str, Enum):
    """Separate required synthetic tracks; none can compensate for another."""

    DOCUMENT = "document"
    IMAGE = "image"
    DICOM = "dicom_preview"
    WSI = "wsi_label_and_overview"
    OCR_CONFLICT = "ocr_conflict"
    MIXED_LANGUAGE = "mixed_language"
    CROSS_DOCUMENT = "cross_document"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseProvenance:
    """Immutable artifact and license snapshots, not license eligibility rules.

    Args:
        model_digest: Exact model or deterministic control artifact manifest hash.
        preprocessing_digest: Exact input/output processing declaration hash.
        hardware_digest: Host hardware declaration hash, not remote attestation.
        runtime_digest: Runtime versions and execution configuration hash.
        model_license_digest: Terms snapshot for these exact model assets.
        dataset_license_digest: Terms snapshot for this exact dataset revision.

    Harness and actual dataset digests are computed by the suite. A changed
    checkpoint or license snapshot creates different provenance; terms are not
    fetched, interpreted, accepted or restricted at inference time.
    """

    model_digest: str
    preprocessing_digest: str
    hardware_digest: str
    runtime_digest: str
    model_license_digest: str
    dataset_license_digest: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            _digest(value)

    @property
    def digest(self) -> str:
        """Bind every immutable host declaration and license snapshot."""
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True, kw_only=True)
class VisionAsset:
    """One bounded protected synthetic raster or native DICOM input."""

    payload: bytes = field(repr=False)
    kind: Literal["raster", "dicom"] = "raster"

    def __post_init__(self) -> None:
        _require(type(self.payload) is bytes and 0 < len(self.payload) <= 1_048_576)
        _require(self.kind in ("raster", "dicom"))

    @property
    def digest(self) -> str:
        """Digest the actual original encoded asset."""
        return _sha(self.payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class VisionInput:
    """Protected data sent to a trusted local candidate, without gold answers."""

    assets: tuple[VisionAsset, ...] = field(repr=False)
    track: VisionTrack
    languages: tuple[str, ...]
    observations: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        _require(type(self.assets) is tuple and 1 <= len(self.assets) <= 2)
        _require(all(type(asset) is VisionAsset for asset in self.assets))
        _require(type(self.track) is VisionTrack)
        _require(type(self.languages) is tuple and 1 <= len(self.languages) <= 3)
        _require(tuple(sorted(set(self.languages))) == self.languages)
        _require(set(self.languages) <= {"en", "es", "fr"})
        _require(type(self.observations) is tuple and len(self.observations) <= 4)
        _require(all(type(s) is str and 0 < len(s) <= 256 for s in self.observations))


@dataclass(frozen=True, slots=True, kw_only=True)
class VisionCase:
    """Synthetic input and private oracle kept by the evaluator."""

    request: VisionInput = field(repr=False)
    expected_hold: bool
    private_markers: tuple[str, ...] = field(repr=False)
    public_rgb: tuple[bytes, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.request) is VisionInput and type(self.expected_hold) is bool)
        _require(
            type(self.private_markers) is tuple and 1 <= len(self.private_markers) <= 4
        )
        _require(
            all(type(s) is str and 0 < len(s) <= 256 for s in self.private_markers)
        )
        _require(
            type(self.public_rgb) is tuple
            and len(self.public_rgb) == len(self.request.assets)
        )
        expected = (
            (_PUBLIC_BOX[2] - _PUBLIC_BOX[0]) * (_PUBLIC_BOX[3] - _PUBLIC_BOX[1]) * 3
        )
        _require(
            all(
                type(pixels) is bytes and len(pixels) == expected
                for pixels in self.public_rgb
            )
        )

    @property
    def digest(self) -> str:
        """Bind encoded assets, oracle, observations and declared slices."""
        return _hash(
            {
                "assets": [(asset.kind, asset.digest) for asset in self.request.assets],
                "track": self.request.track,
                "languages": self.request.languages,
                "observations": [
                    _sha(value.encode()) for value in self.request.observations
                ],
                "expected_hold": self.expected_hold,
                "private_markers": [
                    _sha(value.encode()) for value in self.private_markers
                ],
                "public_rgb": [_sha(value) for value in self.public_rgb],
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VisionOutput:
    """Candidate PNG exports or an explicit all-or-nothing abstention."""

    pages: tuple[bytes, ...] = field(default=(), repr=False)
    abstained: bool = False
    structural_hold: bool = False

    def __post_init__(self) -> None:
        _require(type(self.abstained) is bool and type(self.structural_hold) is bool)
        _require(type(self.pages) is tuple and len(self.pages) <= 2)
        _require(
            all(
                type(page) is bytes and 0 < len(page) <= 1_048_576
                for page in self.pages
            )
        )
        _require(not self.pages if self.abstained else bool(self.pages))


class VisionCandidate(Protocol):
    """Trusted local code; the evaluator is not a provider sandbox."""

    @property
    def provenance(self) -> ReleaseProvenance:
        """Return fixed artifact, runtime and license lineage."""
        ...

    @property
    def local_only(self) -> bool:
        """Declare that no download, telemetry or remote fallback occurs."""
        ...

    def process(self, request: VisionInput) -> VisionOutput:
        """Process actual inputs; do not persist, log or retain protected data."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class VisionGatePolicy:
    """Explicit per-track thresholds; critical leakage always fails independently."""

    minimum_cases_per_track: int
    minimum_utility_rate: float
    minimum_quality_rate: float
    maximum_abstention_rate: float

    def __post_init__(self) -> None:
        import math

        _require(
            type(self.minimum_cases_per_track) is int
            and 1 <= self.minimum_cases_per_track <= 128
        )
        for value in (
            self.minimum_utility_rate,
            self.minimum_quality_rate,
            self.maximum_abstention_rate,
        ):
            _require(
                type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1
            )


def _rate(successes: int, count: int) -> dict[str, object]:
    return {
        "events": successes,
        "n": count,
        "rate": successes / count if count else None,
        "confidence_interval_95": wilson_interval(FidelityCounts(successes, count)),
        "interval_method": "wilson_descriptive",
    }


def _source_digest() -> str:
    return _sha(Path(__file__).read_bytes())


def _inspect_output(case: VisionCase, output: VisionOutput) -> tuple[bool, bool, bool]:
    """Inspect actual whole exports, metadata and public pixels with the private oracle."""
    from PIL import Image

    _require(len(output.pages) == len(case.request.assets))
    critical = any_phi = False
    utility = True
    for index, payload in enumerate(output.pages):
        with Image.open(io.BytesIO(payload)) as image:
            _require(
                image.format == "PNG" and image.size == _SIZE and image.mode == "RGB"
            )
            _require(getattr(image, "n_frames", 1) == 1)
            # The synthetic private carrier is white. Scan the ENTIRE output,
            # including moved/copied markers, rather than only original boxes.
            pixels = image.tobytes()
            white = any(
                min(pixels[offset : offset + 3]) > 250
                for offset in range(0, len(pixels), 3)
            )
            critical |= white
            # Fresh exports must not carry private metadata or copied source strings.
            metadata_leak = any(
                marker in repr(image.info) for marker in case.private_markers
            )
            # Unknown metadata is an unassessed export, not a measured PHI hit.
            _require(not image.info or metadata_leak)
            any_phi |= (
                white
                or metadata_leak
                or any(marker.encode() in payload for marker in case.private_markers)
            )
            with image.crop(_PUBLIC_BOX) as public:
                utility &= public.tobytes() == case.public_rgb[index]
    return critical, any_phi, utility


def run_openphi_vision(
    candidate: VisionCandidate,
    cases: tuple[VisionCase, ...],
    policy: VisionGatePolicy,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Execute every synthetic case and return separate bound release axes.

    Critical burned-in and document-any-PHI leakage are non-compensable failures.
    Missing tracks, failed operations and empty emitted evidence cannot pass.
    Confidence intervals describe these public controls, not deployment risk.
    Candidate exceptions produce fixed error counts; their text is never returned.
    """
    _require(type(cases) is tuple and 0 < len(cases) <= 896)
    _require(all(type(case) is VisionCase for case in cases))
    _require(
        sum(len(asset.payload) for case in cases for asset in case.request.assets)
        <= 64 * 1024 * 1024
    )
    _require(type(policy) is VisionGatePolicy and callable(cancelled))
    _require(len({case.digest for case in cases}) == len(cases))
    provenance = None
    try:
        provenance = candidate.provenance
        _require(type(provenance) is ReleaseProvenance and candidate.local_only is True)
    except Exception:
        provenance = None
    _require(type(provenance) is ReleaseProvenance)
    assert provenance is not None
    rows: list[dict[str, Any]] = []
    interrupted = False
    for case in cases:
        try:
            stop = cancelled()
            _require(type(stop) is bool)
        except Exception:
            stop = True
        if stop:
            interrupted = True
            break
        row = {
            "case_digest": case.digest,
            "track": case.request.track.value,
            "languages": case.request.languages,
            "failed": False,
            "abstained": False,
            "critical": False,
            "any_phi": False,
            "utility": False,
            "quality": False,
            "emitted": False,
            "assessed": False,
            "output_digest": None,
        }
        try:
            _require(
                candidate.local_only is True and candidate.provenance == provenance
            )
            output = candidate.process(case.request)
            _require(type(output) is VisionOutput)
            _require(
                candidate.local_only is True and candidate.provenance == provenance
            )
            row["output_digest"] = _hash(
                {
                    "pages": [_sha(page) for page in output.pages],
                    "abstained": output.abstained,
                    "structural_hold": output.structural_hold,
                }
            )
            row["abstained"] = output.abstained
            if not output.abstained:
                row["emitted"] = True
                row["critical"], row["any_phi"], row["utility"] = _inspect_output(
                    case, output
                )
                row["quality"] = output.structural_hold == case.expected_hold
                row["assessed"] = True
        except Exception:
            row["failed"] = True
        rows.append(row)
    tracks = {}
    for track in VisionTrack:
        selected = [row for row in rows if row["track"] == track.value]
        count = len(selected)
        emitted = sum(row["emitted"] for row in selected)
        assessed = sum(row["assessed"] for row in selected)
        failed = sum(row["failed"] for row in selected)
        critical = sum(row["critical"] for row in selected)
        any_phi = sum(row["any_phi"] for row in selected)
        abstained = sum(row["abstained"] for row in selected)
        utility = sum(row["utility"] for row in selected)
        quality = sum(row["quality"] for row in selected)
        enough = count >= policy.minimum_cases_per_track
        complete = enough and not failed and bool(assessed) and not interrupted
        tracks[track.value] = {
            "n": count,
            "emitted": emitted,
            "assessed": assessed,
            "failed": failed,
            "critical_burned_in": _rate(critical, assessed),
            "document_any_phi": _rate(any_phi, assessed),
            "utility": _rate(utility, count),
            "quality": _rate(quality, count),
            "abstention": _rate(abstained, count),
            "privacy_passed": complete and critical == 0 and any_phi == 0,
            "utility_passed": complete
            and utility / count >= policy.minimum_utility_rate,
            "quality_passed": complete
            and quality / count >= policy.minimum_quality_rate,
            "abstention_passed": complete
            and abstained / count <= policy.maximum_abstention_rate,
        }
    privacy_passed = all(track["privacy_passed"] for track in tracks.values())
    passed = all(
        all(
            track[key]
            for key in (
                "privacy_passed",
                "utility_passed",
                "quality_passed",
                "abstention_passed",
            )
        )
        for track in tracks.values()
    )
    dataset_digest = _hash([case.digest for case in cases])
    binding = {
        **asdict(provenance),
        "harness_digest": _source_digest(),
        "dataset_digest": dataset_digest,
        "policy_digest": _hash(asdict(policy)),
    }
    report = {
        "schema": _SCHEMA,
        "provenance": binding,
        "binding_digest": _hash(binding),
        "dataset_kind": "caller_declared_synthetic_development_controls",
        "metric_definitions": {
            "critical_burned_in": "case with any residual known synthetic white carrier in any output",
            "document_any_phi": "case with residual known carrier or declared private metadata in any output",
            "quality": "expected structural routing hold matched",
            "utility": "all original public control pixels preserved exactly",
            "abstention": "explicit no-export result",
        },
        "independent_validation": False,
        "contamination_resistant": False,
        "clinical_readiness": False,
        "requires_review": True,
        "interrupted": interrupted,
        "expected_cases": len(cases),
        "executed_cases": len(rows),
        "output_digest": _hash(
            [(row["case_digest"], row["output_digest"], row["failed"]) for row in rows]
        ),
        "privacy_passed": privacy_passed,
        "passed": passed,
        "tracks": tracks,
        "language_slices": {
            language: {
                "n": len(selected := [r for r in rows if language in r["languages"]]),
                "critical_burned_in": _rate(
                    sum(r["critical"] for r in selected),
                    sum(r["assessed"] for r in selected),
                ),
                "document_any_phi": _rate(
                    sum(r["any_phi"] for r in selected),
                    sum(r["assessed"] for r in selected),
                ),
                "utility": _rate(sum(r["utility"] for r in selected), len(selected)),
                "quality": _rate(sum(r["quality"] for r in selected), len(selected)),
                "abstention": _rate(
                    sum(r["abstained"] for r in selected), len(selected)
                ),
            }
            for language in ("en", "es", "fr")
        },
        "notice": "Known synthetic carrier and routing controls only; no model accuracy, clinical qualification or all-data destruction claim.",
    }
    report["report_digest"] = _hash(report)
    return report


def _make_asset(
    marker: str, index: int, *, dicom: bool = False
) -> tuple[VisionAsset, bytes]:
    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    image = Image.new("RGB", _SIZE, (20 + index % 30, 30, 40))
    try:
        draw = ImageDraw.Draw(image)
        shift = index % 8
        draw.rectangle((8 + shift, 8, 239, 39), fill="white")
        draw.text(
            (12 + shift, 16), marker, font=ImageFont.load_default(size=11), fill="black"
        )
        draw.rectangle(_PUBLIC_BOX, fill=(30, 100 + index % 60, 50))
        draw.text(
            (16, 68),
            "Synthetic public control",
            font=ImageFont.load_default(size=11),
            fill="black",
        )
        with image.crop(_PUBLIC_BOX) as public:
            public_rgb = public.tobytes()
        stream = io.BytesIO()
        if dicom:
            from pydicom import dcmwrite
            from pydicom.dataset import Dataset, FileMetaDataset
            from pydicom.uid import (
                UID,
                ExplicitVRLittleEndian,
                SecondaryCaptureImageStorage,
            )

            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
            meta.MediaStorageSOPInstanceUID = UID(
                f"1.2.826.0.1.3680043.10.777.{index + 1}"
            )
            dataset = Dataset()
            dataset.file_meta = meta
            dataset.SOPClassUID = meta.MediaStorageSOPClassUID
            dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            dataset.SpecificCharacterSet = "ISO_IR 192"
            dataset.PatientName = marker
            dataset.PatientID = marker
            dataset.BurnedInAnnotation = "YES"
            dataset.Rows, dataset.Columns = _SIZE[1], _SIZE[0]
            dataset.SamplesPerPixel = 3
            dataset.PhotometricInterpretation = "RGB"
            dataset.PlanarConfiguration = 0
            dataset.BitsAllocated = dataset.BitsStored = 8
            dataset.HighBit = 7
            dataset.PixelRepresentation = 0
            dataset.PixelData = image.tobytes()
            dcmwrite(stream, dataset, enforce_file_format=True)
        else:
            info = PngInfo()
            info.add_text("SyntheticIdentifier", marker)
            image.save(stream, format="PNG", pnginfo=info)
        return VisionAsset(
            payload=stream.getvalue(), kind="dicom" if dicom else "raster"
        ), public_rgb
    finally:
        image.close()


def synthetic_vision_cases(*, cases_per_track: int = 3) -> tuple[VisionCase, ...]:
    """Generate deterministic public development controls in memory.

    Each document is a rendered PNG page, not a PDF-parser benchmark. DICOM
    exercises native RGB source headers/pixels to fresh PNG preview export.
    WSI covers associated label and overview rasters, not whole-slide decoding.
    Mixed-language controls use actual synthetic English, French and Spanish
    text. Candidate outputs are scanned across all pages for the known white
    carrier and private metadata; this oracle is not a clinical OCR detector.
    """
    _require(type(cases_per_track) is int and 1 <= cases_per_track <= 32)
    cases = []
    for track_index, track in enumerate(VisionTrack):
        for index in range(cases_per_track):
            number = track_index * 100 + index
            languages = (
                ("en", "es", "fr") if track is VisionTrack.MIXED_LANGUAGE else ("en",)
            )
            count = (
                2
                if track
                in (
                    VisionTrack.DOCUMENT,
                    VisionTrack.WSI,
                    VisionTrack.CROSS_DOCUMENT,
                    VisionTrack.MIXED_LANGUAGE,
                )
                else 1
            )
            markers = tuple(
                f"SYNTHETIC-{number}-{page}"
                + (" Élodie Álvarez" if track is VisionTrack.MIXED_LANGUAGE else "")
                for page in range(count)
            )
            built = tuple(
                _make_asset(marker, number * 2 + page, dicom=track is VisionTrack.DICOM)
                for page, marker in enumerate(markers)
            )
            observations = (
                ("1 mg", "10 mg")
                if track is VisionTrack.OCR_CONFLICT
                else (
                    (markers[0], markers[1])
                    if track is VisionTrack.CROSS_DOCUMENT
                    else ()
                )
            )
            cases.append(
                VisionCase(
                    request=VisionInput(
                        assets=tuple(item[0] for item in built),
                        track=track,
                        languages=languages,
                        observations=observations,
                    ),
                    expected_hold=track
                    in (VisionTrack.OCR_CONFLICT, VisionTrack.CROSS_DOCUMENT),
                    private_markers=markers,
                    public_rgb=tuple(item[1] for item in built),
                )
            )
    return tuple(cases)


def _structural_hold(request: VisionInput) -> bool:
    """Run actual local routing/packet components on synthetic OCR observations."""
    from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry

    if request.track is VisionTrack.OCR_CONFLICT:
        from openmed.multimodal.arbitration.ocr_vlm_router import (
            AgreementPolicy,
            ReadingCandidate,
            ReadingPair,
            ReadingSource,
            SemanticType,
            TextNormalization,
            route_ocr_vlm,
        )
        from openmed.multimodal.vision.providers.base import Confidence

        _require(len(request.observations) == 2)
        candidates = []
        for index, source in enumerate((ReadingSource.OCR, ReadingSource.VLM)):
            value = request.observations[index]
            candidates.append(
                ReadingCandidate(
                    candidate_id=f"{index + 1:032x}",
                    field_id="a" * 32,
                    source=source,
                    text=value,
                    semantic_type=SemanticType.TEXT,
                    confidence=Confidence(score=1),
                    evidence=EvidenceNode(
                        node_id=f"{index + 3:032x}",
                        kind=NodeKind.OCR if index == 0 else NodeKind.PIXELS,
                        source_digest=request.assets[0].digest,
                        geometry=PageGeometry(page=0, width=256, height=96),
                        region=(8, 64, 248, 88),
                        span=(0, len(value)) if index == 0 else None,
                    ),
                )
            )
        report = route_ocr_vlm(
            (ReadingPair(pair_id="b" * 32, ocr=candidates[0], vlm=candidates[1]),),
            AgreementPolicy(
                version="synthetic-gate-v1",
                allowed_types=(SemanticType.TEXT,),
                normalization=TextNormalization.EXACT,
                minimum_geometry_iou=1,
                minimum_confidence=1,
                preferred_source=ReadingSource.OCR,
                maximum_review_items=1,
            ),
        )
        return bool(report.review_queue)
    if request.track is VisionTrack.CROSS_DOCUMENT:
        from PIL import Image

        from openmed.multimodal.vision.documents.page_consistency import (
            ConsistencyPolicy,
            IdentityKind,
            anchor_from_span,
            assess_page_consistency,
            capture_page,
            identity_from_span,
            numbering_from_span,
        )

        _require(len(request.observations) == len(request.assets))
        pages = []
        key = b"public-synthetic-packet-control-key-3205"
        for index, asset in enumerate(request.assets):
            page_source = EvidenceNode(
                node_id=f"{index * 10 + 1:032x}",
                kind=NodeKind.PAGE,
                source_digest=asset.digest,
                geometry=PageGeometry(page=0, width=256, height=96),
            )
            values = (request.observations[index], "Synthetic form", f"{index + 1}/2")
            text = "|".join(values)
            spans = []
            offset = 0
            for slot, value in enumerate(values):
                spans.append(
                    EvidenceNode(
                        node_id=f"{index * 10 + slot + 2:032x}",
                        kind=NodeKind.OCR,
                        source_digest=asset.digest,
                        geometry=page_source.geometry,
                        region=(8, 8, 240, 40),
                        span=(offset, offset + len(value)),
                    )
                )
                offset += len(value) + 1
            identity = identity_from_span(
                text,
                spans[0],
                key=key,
                kind=IdentityKind.PATIENT_ID,
                namespace_ref="a" * 32,
            )
            with Image.open(io.BytesIO(asset.payload)) as image:
                pages.append(
                    capture_page(
                        page_source,
                        image.tobytes(),
                        _SIZE,
                        identities=(identity,),
                        anchors=(anchor_from_span(text, spans[1], key=key),),
                        numbering=numbering_from_span(text, spans[2]),
                    )
                )
        return not assess_page_consistency(
            tuple(pages), policy=ConsistencyPolicy(identity_namespace="a" * 32)
        ).whole_packet_consistent
    if request.track is VisionTrack.WSI:
        from openmed.multimodal.vision.pathology import (
            PyramidLevel,
            TilePolicy,
            plan_slide_tiles,
        )

        plan = plan_slide_tiles(
            request.assets[1].digest,
            256,
            96,
            (PyramidLevel(index=0, width=256, height=96, scale_numerator=1),),
            TilePolicy(
                width=128,
                height=48,
                overlap_x=0,
                overlap_y=0,
                working_bytes_per_pixel=3,
                max_inflight_tiles=2,
                budget_bytes=1_000_000,
                reserved_bytes=0,
                min_tissue_percent=0,
            ),
        )
        _require(len(plan.selected_tiles) == 4)
    return False


class SyntheticVisionControl:
    """Known-layout positive control running the actual pixel privacy gate.

    It masks the fixture's upper region and detects its white carrier. This
    intentionally privileged control measures wiring, not model recognition.
    """

    local_only = True

    def __init__(self) -> None:
        from importlib.metadata import distribution, version

        license_text = distribution("openmed").read_text("licenses/LICENSE")
        _require(type(license_text) is str and bool(license_text))
        assert license_text is not None
        license_digest = _sha(license_text.encode())
        self._provenance = ReleaseProvenance(
            model_digest=_source_digest(),
            preprocessing_digest=_sha(
                b"native RGB to verified fresh PNG; identity geometry v1"
            ),
            hardware_digest=_hash(
                {"machine": platform.machine(), "system": platform.system()}
            ),
            runtime_digest=_hash(
                {
                    "python": platform.python_version(),
                    "pillow": version("Pillow"),
                    "pydicom": version("pydicom"),
                }
            ),
            model_license_digest=license_digest,
            dataset_license_digest=license_digest,
        )

    @property
    def provenance(self) -> ReleaseProvenance:
        """Return the control's declared local artifact and runtime fingerprints."""
        return self._provenance

    def process(self, request: VisionInput) -> VisionOutput:
        """Execute actual regional inspection, masking and residual verification."""
        from openmed.multimodal.privacy.pixel_coverage import (
            LocalPixelDetector,
            PixelFramePlan,
            PixelInspection,
            audit_pixel_redaction,
            inventory_pixel_frames,
        )
        from openmed.multimodal.vision.dicom.series_sampler import DisplayWindow

        def inspect(image: Any) -> Any:
            points = [
                (x, y)
                for y in range(image.height)
                for x in range(image.width)
                if min(image.getpixel((x, y))) > 250
            ]
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

        detector = LocalPixelDetector(
            digest=self.provenance.model_digest, inspect=inspect, local_only=True
        )
        outputs: list[bytes] = []
        for asset in request.assets:
            window = (
                DisplayWindow(center=127.5, width=255)
                if asset.kind == "dicom"
                else None
            )
            frames = inventory_pixel_frames(
                asset.payload, source_kind=asset.kind, window=window
            )
            result = audit_pixel_redaction(
                asset.payload,
                tuple(
                    PixelFramePlan(reference=frame.reference, masks=((0, 0, 256, 56),))
                    for frame in frames
                ),
                detector=detector,
                residual_detector=detector,
                source_kind=asset.kind,
                window=window,
            )
            if not result.accepted or len(result.outputs) != 1:
                return VisionOutput(abstained=True)
            outputs.extend(result.outputs)
        return VisionOutput(
            pages=tuple(outputs), structural_hold=_structural_hold(request)
        )
