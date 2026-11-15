"""Generated DICOM fixtures with paired metadata and actual burned-in masks.

Only internally generated pixels and facts enter this fixture builder. DICOM,
image and text payloads are explicit synthetic test data, excluded from reports.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar

from openmed.core.labels import DATE, ID_NUM, PERSON
from openmed.multimodal.evidence.fused_graph import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.privacy.mask_lineage import (
    CoordinateSpace,
    MaskLineageResult,
    propagate_masks,
)

from .burned_in import (
    COMPLEX_LAYOUT_SCRIPTS,
    PERMISSIVE_FONT_LICENSES,
    PILLOW_DEFAULT_FONT_LICENSE,
    BurnedInFont,
    BurnedInGenerator,
    _check_glyph_coverage,
    _load_face,
    _span_scripts,
    font_supports_complex_layout,
)
from .note_grammar import synthetic_note_facts

SCHEMA_VERSION = "openmed.synthetic.dicom-overlays.v1"
__all__ = [
    "DicomCanvas",
    "DicomOverlayError",
    "DicomOverlayFixture",
    "DicomOverlayLimits",
    "DicomOverlaySpec",
    "DicomPrivacyField",
    "OverlayAnnotation",
    "OverlayRole",
    "RulerAnnotation",
    "SyntheticDicomField",
    "generate_dicom_overlays",
    "reproduce_dicom_overlays",
]
T = TypeVar("T")
Box = tuple[int, int, int, int]


class DicomOverlayError(ValueError):
    """Fixed-message rejection without source values or backend exception context."""


class DicomCanvas(str, Enum):
    """Generated grayscale patterns; none is an anatomical or diagnostic image."""

    RINGS = "rings"
    GRADIENT = "gradient"
    NOISE = "noise"


class DicomPrivacyField(str, Enum):
    """Closed metadata fields containing generated privacy-test identifiers."""

    PATIENT_NAME = "PatientName"
    PATIENT_ID = "PatientID"
    BIRTH_DATE = "PatientBirthDate"
    STUDY_DATE = "StudyDate"
    ACCESSION = "AccessionNumber"
    STUDY_UID = "StudyInstanceUID"
    SERIES_UID = "SeriesInstanceUID"
    INSTANCE_UID = "SOPInstanceUID"


class OverlayRole(str, Enum):
    """Actual painted identity, label, orientation or acquisition annotation."""

    IDENTIFIER = "identifier"
    FIELD_LABEL = "field_label"
    ORIENTATION = "orientation"
    ACQUISITION = "acquisition"
    DISCLOSURE = "disclosure"


def _require(condition: bool) -> None:
    if not condition:
        raise DicomOverlayError("synthetic_dicom_overlay_rejected")


def _safe(call: Callable[[], T]) -> T:
    failed = False
    result = None
    try:
        result = call()
    except Exception:
        failed = True
    if failed:
        raise DicomOverlayError("synthetic_dicom_overlay_rejected")
    return result  # type: ignore[return-value]


def _integer(value, low, high):
    _require(type(value) is int and low <= value <= high)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _digest(value):
    _require(type(value) is bytes)
    return hashlib.sha256(value).hexdigest()


def _hex(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _box(value, size):
    _require(type(value) is tuple and len(value) == 4)
    for coordinate in value:
        _integer(coordinate, 0, max(size))
    x0, y0, x1, y1 = value
    _require(0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1])


@dataclass(frozen=True, slots=True, kw_only=True)
class DicomOverlaySpec:
    """Deterministic generated fixture choices with no arbitrary source-data input.

    Args:
        seed: Seed for generated identities, canvas patterns and acquisition labels.
        size: Pixel (width, height), each between 384 and 1024.
        canvas: Generated background pattern, never a patient-derived image.
        font_size: Maximum font size; long generated fields may shrink to 10 pixels.
    """

    seed: int
    size: tuple[int, int] = (640, 512)
    canvas: DicomCanvas = DicomCanvas.RINGS
    font_size: int = 18

    def __post_init__(self) -> None:
        _integer(self.seed, 0, 2**63 - 1)
        _require(type(self.size) is tuple and len(self.size) == 2)
        for dimension in self.size:
            _integer(dimension, 384, 1024)
        _require(type(self.canvas) is DicomCanvas)
        _integer(self.font_size, 12, 24)


@dataclass(frozen=True, slots=True, kw_only=True)
class DicomOverlayLimits:
    """Cooperative pixel, mask-work and payload ceilings for trusted local assets."""

    max_pixels: int = 1_048_576
    max_mask_work_pixels: int = 32_000_000
    max_payload_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        _integer(self.max_pixels, 1, 1_048_576)
        _integer(self.max_mask_work_pixels, 1, 64_000_000)
        _integer(self.max_payload_bytes, 1, 16 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class SyntheticDicomField:
    """A generated metadata identifier, protected separately from its label/hash."""

    keyword: DicomPrivacyField
    label: str
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.keyword) is DicomPrivacyField)
        expected = (
            PERSON
            if self.keyword is DicomPrivacyField.PATIENT_NAME
            else DATE
            if self.keyword
            in (DicomPrivacyField.BIRTH_DATE, DicomPrivacyField.STUDY_DATE)
            else ID_NUM
        )
        _require(type(self.label) is str and self.label == expected)
        _require(type(self.value) is str and 0 < len(self.value) <= 64)
        _require(all(32 <= ord(char) <= 126 and char != "\\" for char in self.value))


@dataclass(frozen=True, slots=True)
class OverlayAnnotation:
    """One actually painted text segment and its tight grayscale glyph mask."""

    annotation_id: str
    span: tuple[int, int]
    box: Box
    role: OverlayRole
    field_code: DicomPrivacyField | None
    label: str | None
    font_size: int
    mask_png: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class RulerAnnotation:
    """Actual pixel ruler geometry, with no millimetre/anatomical calibration."""

    box: Box
    tick_spacing: int
    ticks: tuple[int, ...]
    mask_png: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class DicomOverlayFixture:
    """Protected synthetic Part 10 DICOM and PNG payloads with privacy-test truth."""

    spec: DicomOverlaySpec
    font_digest: str
    font_license: str
    runtime_digest: str
    annotations: tuple[OverlayAnnotation, ...]
    ruler: RulerAnnotation
    lineage: MaskLineageResult
    metadata: tuple[SyntheticDicomField, ...] = field(repr=False)
    text: str = field(repr=False)
    background_png: bytes = field(repr=False)
    image_png: bytes = field(repr=False)
    phi_mask_png: bytes = field(repr=False)
    dicom_bytes: bytes = field(repr=False)

    def to_json(self) -> str:
        """Return validated value-free hashes, coordinates and controlled labels."""

        def report():
            _validate(self)
            return _json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "synthetic": True,
                    "clinical_validation": False,
                    "spec": asdict(self.spec),
                    "font_digest": self.font_digest,
                    "font_license": self.font_license,
                    "runtime_digest": self.runtime_digest,
                    "text_digest": _digest(self.text.encode()),
                    "metadata": [
                        {
                            "keyword": item.keyword.value,
                            "label": item.label,
                            "value_digest": _digest(item.value.encode()),
                        }
                        for item in self.metadata
                    ],
                    "annotations": [
                        {
                            "annotation_id": item.annotation_id,
                            "span": item.span,
                            "box": item.box,
                            "role": item.role.value,
                            "field_code": item.field_code.value
                            if item.field_code
                            else None,
                            "label": item.label,
                            "font_size": item.font_size,
                            "mask_digest": _digest(item.mask_png),
                        }
                        for item in self.annotations
                    ],
                    "ruler": {
                        "box": self.ruler.box,
                        "tick_spacing": self.ruler.tick_spacing,
                        "ticks": self.ruler.ticks,
                        "units": "pixels",
                        "mask_digest": _digest(self.ruler.mask_png),
                    },
                    "background_digest": _digest(self.background_png),
                    "image_digest": _digest(self.image_png),
                    "phi_mask_digest": _digest(self.phi_mask_png),
                    "dicom_digest": _digest(self.dicom_bytes),
                    "lineage": json.loads(self.lineage.to_json()),
                }
            )

        return _safe(report)

    @property
    def digest(self) -> str:
        """Bind actual DICOM/pixel payloads, generated metadata and mask provenance."""
        return _digest(self.to_json().encode())


class _Output(BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, value):
        _require(self.tell() + len(value) <= self.limit)
        return super().write(value)


def _png(image, limits):
    with _Output(limits.max_payload_bytes) as stream:
        image.save(stream, format="PNG", optimize=False)
        return stream.getvalue()


def _font(requested, spec):
    from PIL import ImageFont

    helper = BurnedInGenerator(seed=spec.seed)
    if requested is None:
        data = ImageFont.load_default(size=spec.font_size).font_bytes
        license_id, index = PILLOW_DEFAULT_FONT_LICENSE, 0
    else:
        _require(type(requested) is BurnedInFont)
        requested.__post_init__()
        path = Path(requested.path)
        _require(path.is_file() and path.stat().st_size <= 16 * 1024 * 1024)
        with path.open("rb") as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        license_id, index = requested.license, requested.index
    _require(type(data) is bytes and 0 < len(data) <= 16 * 1024 * 1024)
    return helper._describe_font(
        data,
        index=index,
        display_name="FixtureFont",
        license_id=license_id,
        fallback=False,
    )


def _runtime():
    import faker
    import PIL
    import pydicom
    from PIL import features

    return _hash(
        [
            PIL.__version__,
            pydicom.__version__,
            faker.VERSION,
            sys.version_info[:2],
            features.version_module("freetype2"),
            features.version_codec("zlib"),
            features.version_feature("raqm"),
        ]
    )


def _background(spec):
    from PIL import Image, ImageDraw

    rng = random.Random(_hash([spec.seed, spec.canvas.value]))
    width, height = spec.size
    if spec.canvas is DicomCanvas.NOISE:
        return Image.frombytes(
            "L", spec.size, bytes(rng.randrange(16, 80) for _ in range(width * height))
        )
    image = Image.new("L", spec.size, 12)
    draw = ImageDraw.Draw(image)
    if spec.canvas is DicomCanvas.GRADIENT:
        start, end = rng.randrange(8, 25), rng.randrange(65, 100)
        for y in range(height):
            draw.line(
                (0, y, width - 1, y),
                fill=round(start + (end - start) * y / (height - 1)),
            )
    else:
        for radius in range(min(spec.size) // 3, 16, -16):
            draw.ellipse(
                (
                    width // 2 - radius,
                    height // 2 - radius,
                    width // 2 + radius,
                    height // 2 + radius,
                ),
                outline=rng.randrange(35, 100),
                width=5,
            )
    return image


def _mask(value, font):
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = font.getbbox(value)
    _require(0 < x1 - x0 <= 2048 and 0 < y1 - y0 <= 256)
    with Image.new("L", (x1 - x0 + 4, y1 - y0 + 4)) as image:
        ImageDraw.Draw(image).text((2 - x0, 2 - y0), value, font=font, fill=255)
        box = image.getbbox()
        _require(box is not None)
        return image.crop(box)


def _uid(value):
    return "2.25." + str(int(_hash(value)[:32], 16))


def _dicom(spec, pixels, metadata, limits):
    from pydicom import dcmwrite
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    fields = {item.keyword.value: item.value for item in metadata}
    meta = FileMetaDataset()
    meta.FileMetaInformationVersion = b"\x00\x01"
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = fields["SOPInstanceUID"]
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = _uid([SCHEMA_VERSION, "implementation"])
    meta.ImplementationVersionName = "OPENMED_SYNTH1"
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    for keyword, value in fields.items():
        setattr(dataset, keyword, value)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SpecificCharacterSet = "ISO_IR 100"
    dataset.Modality = "OT"
    dataset.ImageType = ["DERIVED", "SECONDARY"]
    dataset.ConversionType = "WSD"
    dataset.Manufacturer = "SYNTHETIC FIXTURE"
    dataset.SeriesDescription = "SYNTHETIC PRIVACY TEST"
    dataset.PatientSex = ""
    dataset.StudyID = ""
    dataset.StudyTime = "120000"
    dataset.ContentDate = fields["StudyDate"]
    dataset.ContentTime = "120000"
    dataset.SeriesNumber = 1
    dataset.InstanceNumber = 1
    dataset.PatientOrientation = ""
    dataset.BurnedInAnnotation = "YES"
    dataset.PatientIdentityRemoved = "NO"
    dataset.RecognizableVisualFeatures = "NO"
    dataset.Rows, dataset.Columns = spec.size[1], spec.size[0]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels
    dataset[0x7FE00010].VR = "OB"
    with _Output(limits.max_payload_bytes) as stream:
        dcmwrite(stream, dataset, enforce_file_format=True)
        return stream.getvalue()


def _lineage(spec, dicom_bytes, annotations):
    digest = _digest(dicom_bytes)
    geometry = PageGeometry(page=0, width=spec.size[0], height=spec.size[1])
    space = CoordinateSpace(
        space_ref=_hash([SCHEMA_VERSION, digest, "pixels"])[:32],
        source_digest=digest,
        geometry=geometry,
    )
    nodes = tuple(
        EvidenceNode(
            node_id=item.annotation_id[:32],
            kind=NodeKind.PIXELS,
            source_digest=digest,
            geometry=geometry,
            region=item.box,
        )
        for item in annotations
        if item.role is OverlayRole.IDENTIFIER
    )
    result = propagate_masks(space, nodes, ())
    _require(result.accepted)
    return result


def _generate(spec, limits, requested_font):
    from PIL import Image, ImageChops, ImageDraw, ImageFont

    _require(type(spec) is DicomOverlaySpec and type(limits) is DicomOverlayLimits)
    spec.__post_init__()
    limits.__post_init__()
    width, height = spec.size
    _require(width * height <= limits.max_pixels)
    _require(width * height * 18 <= limits.max_mask_work_pixels)
    resolved = _font(requested_font, spec)
    runtime = _runtime()
    facts = {fact.key: fact for fact in synthetic_note_facts(seed=spec.seed)}
    metadata = [
        SyntheticDicomField(
            DicomPrivacyField.PATIENT_NAME, PERSON, facts["person"].value
        ),
        SyntheticDicomField(
            DicomPrivacyField.PATIENT_ID, ID_NUM, facts["identifier"].value
        ),
        SyntheticDicomField(DicomPrivacyField.BIRTH_DATE, DATE, "20000115"),
        SyntheticDicomField(DicomPrivacyField.STUDY_DATE, DATE, "20200115"),
        SyntheticDicomField(
            DicomPrivacyField.ACCESSION,
            ID_NUM,
            "SYN" + _hash([spec.seed, "accession"])[:12].upper(),
        ),
    ]
    full_text = (
        " ".join(item.value for item in metadata)
        + " Name: ID: Born: Acquired: Accession: L R SYNTHETIC PIXELS OT SERIES 01"
    )
    _check_glyph_coverage(resolved, full_text, label=PERSON)
    base_font = _load_face(ImageFont, resolved, spec.font_size)
    if any(
        script in COMPLEX_LAYOUT_SCRIPTS or script == "Unknown"
        for script in _span_scripts(full_text)
    ):
        _require(font_supports_complex_layout(base_font))
    background = _background(spec)
    image = background.copy()
    phi_mask = Image.new("L", spec.size)
    annotations = []
    text = ""

    def place(value, position, role, font_size, field_code=None, label=None):
        nonlocal text, phi_mask
        face = _load_face(ImageFont, resolved, font_size)
        mask = _mask(value, face)
        try:
            x, y = position
            box = (x, y, x + mask.width, y + mask.height)
            _box(box, spec.size)
            # Prevent unreported occlusion between generated text segments.
            for prior in annotations:
                px0, py0, px1, py1 = prior.box
                _require(box[2] <= px0 or px1 <= x or box[3] <= py0 or py1 <= y)
            start = len(text)
            text += value + "\n"
            image.paste(255, box, mask)
            annotation = OverlayAnnotation(
                _hash([SCHEMA_VERSION, start, value, role.value]),
                (start, start + len(value)),
                box,
                role,
                field_code,
                label,
                font_size,
                _png(mask, limits),
            )
            annotations.append(annotation)
            if role is OverlayRole.IDENTIFIER:
                with Image.new("L", spec.size) as full:
                    full.paste(mask, box)
                    combined = ImageChops.lighter(phi_mask, full)
                phi_mask.close()
                phi_mask = combined
            return mask.width
        finally:
            mask.close()

    try:
        line_height = sum(base_font.getmetrics()) + 8
        y = 18
        for prefix, item in zip(
            ("Name:", "ID:", "Born:", "Acquired:", "Accession:"), metadata, strict=True
        ):
            size = spec.font_size
            while size >= 10:
                face = _load_face(ImageFont, resolved, size)
                with _mask(prefix, face) as left, _mask(item.value, face) as right:
                    fits = 20 + left.width + 10 + right.width <= width - 45
                if fits:
                    break
                size -= 1
            _require(size >= 10)
            prefix_width = place(
                prefix, (20, y), OverlayRole.FIELD_LABEL, size, item.keyword
            )
            place(
                item.value,
                (30 + prefix_width, y),
                OverlayRole.IDENTIFIER,
                size,
                item.keyword,
                item.label,
            )
            y += line_height
        place("L", (20, height // 2), OverlayRole.ORIENTATION, spec.font_size)
        place("R", (width - 78, height // 2), OverlayRole.ORIENTATION, spec.font_size)
        place(
            "OT SERIES 01", (20, height - 70), OverlayRole.ACQUISITION, spec.font_size
        )
        place(
            "SYNTHETIC PIXELS",
            (20, height - 38),
            OverlayRole.DISCLOSURE,
            spec.font_size,
        )
        with Image.new("L", spec.size) as ruler_mask:
            draw = ImageDraw.Draw(ruler_mask)
            top, bottom, x = height // 3, height - 100, width - 25
            spacing = 16
            ticks = tuple(range(top, bottom + 1, spacing))
            draw.line((x, top, x, ticks[-1]), fill=255)
            for index, tick in enumerate(ticks):
                length = 12 if index % 4 == 0 else 7
                draw.line((x - length, tick, x, tick), fill=255)
            box = ruler_mask.getbbox()
            _require(box is not None)
            for item in annotations:
                x0, y0, x1, y1 = item.box
                _require(x1 <= box[0] or box[2] <= x0 or y1 <= box[1] or box[3] <= y0)
            image.paste(255, (0, 0, width, height), ruler_mask)
            with ruler_mask.crop(box) as tight:
                ruler = RulerAnnotation(box, spacing, ticks, _png(tight, limits))
        pixels = image.tobytes()
        for keyword in (
            DicomPrivacyField.STUDY_UID,
            DicomPrivacyField.SERIES_UID,
            DicomPrivacyField.INSTANCE_UID,
        ):
            metadata.append(
                SyntheticDicomField(
                    keyword,
                    ID_NUM,
                    _uid(
                        [SCHEMA_VERSION, asdict(spec), _digest(pixels), keyword.value]
                    ),
                )
            )
        dicom = _dicom(spec, pixels, metadata, limits)
        fixture = DicomOverlayFixture(
            spec,
            resolved.sha256,
            resolved.license,
            runtime,
            tuple(annotations),
            ruler,
            _lineage(spec, dicom, annotations),
            tuple(metadata),
            text,
            _png(background, limits),
            _png(image, limits),
            _png(phi_mask, limits),
            dicom,
        )
        _validate(fixture)
        return fixture
    finally:
        background.close()
        image.close()
        phi_mask.close()


def _image(payload, size, *, tight=False):
    from PIL import Image

    _require(type(payload) is bytes and 0 < len(payload) <= 16 * 1024 * 1024)
    with Image.open(BytesIO(payload)) as image:
        _require(
            image.format == "PNG"
            and image.mode == "L"
            and image.size == size
            and not image.info
        )
        image.load()
        if tight:
            _require(image.getbbox() == (0, 0, *size))
        return image.copy()


def _validate(fixture):
    from PIL import Image, ImageChops, ImageDraw
    from pydicom import dcmread
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    _require(
        type(fixture) is DicomOverlayFixture and type(fixture.spec) is DicomOverlaySpec
    )
    fixture.spec.__post_init__()
    for value in (fixture.font_digest, fixture.runtime_digest):
        _hex(value)
    _require(
        type(fixture.font_license) is str
        and fixture.font_license in PERMISSIVE_FONT_LICENSES
    )
    _require(
        type(fixture.metadata) is tuple
        and len(fixture.metadata) == len(DicomPrivacyField)
    )
    for item in fixture.metadata:
        _require(type(item) is SyntheticDicomField)
        item.__post_init__()
    _require(
        tuple(item.keyword for item in fixture.metadata) == tuple(DicomPrivacyField)
    )
    values = {item.keyword: item for item in fixture.metadata}
    _require(type(fixture.text) is str and 0 < len(fixture.text) <= 4096)
    _require(type(fixture.annotations) is tuple and len(fixture.annotations) == 14)
    _require(
        type(fixture.dicom_bytes) is bytes
        and 0 < len(fixture.dicom_bytes) <= 16 * 1024 * 1024
    )
    dataset = dcmread(BytesIO(fixture.dicom_bytes), force=False)
    _require(dataset.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian)
    _require(
        dataset.SOPClassUID
        == dataset.file_meta.MediaStorageSOPClassUID
        == SecondaryCaptureImageStorage
    )
    _require(dataset.SOPInstanceUID == dataset.file_meta.MediaStorageSOPInstanceUID)
    _require(
        dataset.Rows == fixture.spec.size[1] and dataset.Columns == fixture.spec.size[0]
    )
    _require(dataset.BitsAllocated == dataset.BitsStored == 8 and dataset.HighBit == 7)
    _require(
        dataset.SamplesPerPixel == 1
        and dataset.PixelRepresentation == 0
        and dataset.PhotometricInterpretation == "MONOCHROME2"
    )
    _require(
        dataset.BurnedInAnnotation == "YES" and dataset.PatientIdentityRemoved == "NO"
    )
    for item in fixture.metadata:
        _require(str(getattr(dataset, item.keyword.value)) == item.value)
    with (
        _image(fixture.background_png, fixture.spec.size) as background,
        _image(fixture.image_png, fixture.spec.size) as image,
        _image(fixture.phi_mask_png, fixture.spec.size) as phi,
        Image.new("L", fixture.spec.size) as union,
    ):
        pixels = image.tobytes()
        _require(dataset.PixelData == pixels + (b"\0" if len(pixels) % 2 else b""))
        painted = background.copy()
        cursor = 0
        identifiers = []
        try:
            for item in fixture.annotations:
                _require(type(item) is OverlayAnnotation)
                _hex(item.annotation_id)
                _require(type(item.span) is tuple and len(item.span) == 2)
                start, end = item.span
                _integer(start, 0, len(fixture.text) - 1)
                _integer(end, start + 1, len(fixture.text))
                _require(start == cursor and fixture.text[end : end + 1] == "\n")
                cursor = end + 1
                _require(type(item.role) is OverlayRole)
                _require(
                    item.field_code is None
                    or type(item.field_code) is DicomPrivacyField
                )
                _require(
                    item.annotation_id
                    == _hash(
                        [
                            SCHEMA_VERSION,
                            start,
                            fixture.text[start:end],
                            item.role.value,
                        ]
                    )
                )
                _integer(item.font_size, 10, 24)
                _box(item.box, fixture.spec.size)
                x0, y0, x1, y1 = item.box
                with _image(item.mask_png, (x1 - x0, y1 - y0), tight=True) as mask:
                    painted.paste(255, item.box, mask)
                    if item.role is OverlayRole.IDENTIFIER:
                        _require(item.field_code in values)
                        truth = values[item.field_code]
                        _require(
                            item.label == truth.label
                            and fixture.text[start:end] == truth.value
                        )
                        union.paste(mask, item.box)
                        identifiers.append(item.field_code)
                    else:
                        _require(item.label is None)
            _require(cursor == len(fixture.text))
            _require(tuple(identifiers) == tuple(DicomPrivacyField)[:5])
            _require(ImageChops.difference(union, phi).getbbox() is None)
            ruler = fixture.ruler
            _require(type(ruler) is RulerAnnotation)
            _box(ruler.box, fixture.spec.size)
            _integer(ruler.tick_spacing, 1, 64)
            _require(type(ruler.ticks) is tuple and len(ruler.ticks) > 1)
            for tick in ruler.ticks:
                _integer(tick, 0, fixture.spec.size[1] - 1)
            _require(
                all(
                    b - a == ruler.tick_spacing
                    for a, b in zip(ruler.ticks, ruler.ticks[1:])
                )
            )
            width, height = fixture.spec.size
            _require(ruler.tick_spacing == 16)
            _require(ruler.ticks == tuple(range(height // 3, height - 99, 16)))
            _require(
                ruler.box
                == (width - 37, ruler.ticks[0], width - 24, ruler.ticks[-1] + 1)
            )
            x0, y0, x1, y1 = ruler.box
            with _image(ruler.mask_png, (x1 - x0, y1 - y0), tight=True) as mask:
                with Image.new("L", mask.size) as expected:
                    draw = ImageDraw.Draw(expected)
                    draw.line((12, 0, 12, mask.height - 1), fill=255)
                    for index, tick in enumerate(ruler.ticks):
                        length = 12 if index % 4 == 0 else 7
                        draw.line((12 - length, tick - y0, 12, tick - y0), fill=255)
                    _require(ImageChops.difference(mask, expected).getbbox() is None)
                painted.paste(255, ruler.box, mask)
            _require(ImageChops.difference(painted, image).getbbox() is None)
        finally:
            painted.close()
    _require(type(fixture.lineage) is MaskLineageResult)
    _require(
        fixture.lineage
        == _lineage(fixture.spec, fixture.dicom_bytes, fixture.annotations)
    )


def generate_dicom_overlays(
    *,
    spec: DicomOverlaySpec,
    limits: DicomOverlayLimits = DicomOverlayLimits(),
    font: BurnedInFont | None = None,
) -> DicomOverlayFixture:
    """Generate actual synthetic DICOM, matching pixels, metadata and privacy masks.

    Args:
        spec: Seed, generated canvas, dimensions and font-size controls.
        limits: Cooperative allocation/work ceilings; overflow rejects all output.
        font: Optional existing local permissively licensed font. The default is
            Pillow's explicit embedded face, rendered from its digested bytes.

    Returns:
        Explicit synthetic Part 10 Secondary Capture bytes and PNGs, generated
        metadata, tight annotation masks and source-bound mask lineage. Reports
        omit raw values. Nothing is written or downloaded. Rulers measure pixels
        only; fixtures do not represent real anatomy or clinical qualification.
    """
    return _safe(lambda: _generate(spec, limits, font))


def reproduce_dicom_overlays(
    fixture: DicomOverlayFixture,
    *,
    limits: DicomOverlayLimits = DicomOverlayLimits(),
    font: BurnedInFont | None = None,
) -> DicomOverlayFixture:
    """Rerender and serialize a prior fixture, rejecting changed payloads or truth.

    Args:
        fixture: Prior result, including its protected DICOM/pixel payloads.
        limits: Current resource limits; prior receipts cannot override them.
        font: Same local font asset/face if the prior result used a custom font.
    """

    def reproduce():
        _validate(fixture)
        result = _generate(fixture.spec, limits, font)
        _require(result.to_json() == fixture.to_json())
        return result

    return _safe(reproduce)
