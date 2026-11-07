"""Reproducible DICOM display pixels and source-bound local visual evidence.

This is a bounded display pipeline, not a diagnostic renderer, de-identification
step or assertion that a model's clinical interpretation is correct.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any, TypeVar

from ...dicom_transfer_syntax import TransferSyntaxOutcome, read_dicom_transfer_syntax
from ...page_rotation import PageRotation
from ..providers.base import (
    EdgeVLMProvider,
    ImageInput,
    ImageRegion,
    Prompt,
    VisualEvidence,
    VLMRequest,
    VLMResult,
    run_local_vlm,
)
from .series_sampler import (
    DisplayWindow,
    _Header,
    _header_integer,
    _quiet_parser,
    _QuietParser,
)

_SCHEMA = "openmed.dicom.pixel-provenance.v1"
_OPERATIONS = (
    "transfer_syntax",
    "frame",
    "decode",
    "rescale",
    "window",
    "color",
    "orientation",
    "resize",
    "encode",
)
_NATIVE = ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1")
_RLE = "1.2.840.10008.1.2.5"
_JPEG = "1.2.840.10008.1.2.4.50"
_PHOTOS = ("MONOCHROME1", "MONOCHROME2", "RGB", "YBR_FULL", "YBR_FULL_422")
_TAGS = [
    "Rows",
    "Columns",
    "NumberOfFrames",
    "SamplesPerPixel",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "PlanarConfiguration",
    "RescaleSlope",
    "RescaleIntercept",
    "ModalityLUTSequence",
    "PresentationLUTShape",
    "PresentationLUTSequence",
    "SharedFunctionalGroupsSequence",
    "PerFrameFunctionalGroupsSequence",
]
T = TypeVar("T")
Box = tuple[int, int, int, int]

__all__ = [
    "PixelDecoder",
    "PixelHeader",
    "PixelProvenance",
    "PixelProvenanceError",
    "PixelProvenanceLimits",
    "PixelRenderSpec",
    "PixelStage",
    "ProvenancedFrame",
    "ProvenancedInference",
    "RegionProvenance",
    "ResizeFilter",
    "bind_region_provenance",
    "render_dicom_frame",
    "reproduce_dicom_frame",
    "run_provenanced_dicom_vlm",
    "verify_region_provenance",
]


class PixelProvenanceError(ValueError):
    """Fixed-message rejection with no source, decoder or provider exception context."""


def _require(condition: bool) -> None:
    if not condition:
        raise PixelProvenanceError("dicom_pixel_provenance_rejected")


def _safe(call: Callable[[], T]) -> T:
    result = None
    failed = False
    try:
        result = call()
    except Exception:
        failed = True
    if failed:
        raise PixelProvenanceError("dicom_pixel_provenance_rejected")
    return result  # type: ignore[return-value]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _digest(value: str) -> None:
    _require(
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _int(value: int, low: int, high: int) -> None:
    _require(type(value) is int and low <= value <= high)


def _finite(value: float, limit: float) -> None:
    _require(
        type(value) in (float, int) and math.isfinite(value) and abs(value) <= limit
    )


class PixelDecoder(str, Enum):
    """Explicit locally installed decoder; never automatic plugin fallback."""

    NATIVE = "pydicom_native"
    RLE = "pydicom_rle"
    JPEG = "pillow_grayscale_jpeg_baseline"


class ResizeFilter(str, Enum):
    """Executed Pillow resize filter recorded in the digest chain."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"


@dataclass(frozen=True, slots=True, kw_only=True)
class PixelRenderSpec:
    """Explicit frame, display window, stored-pixel rotation and resize request.

    Args:
        frame_index: Zero-based source frame; no implicit representative frame.
        window: Required for monochrome; explicit linear clipping after rescale.
            Must be None for RGB/YBR input. DICOM VOI presets are not selected.
        decoder: Exact native/RLE/grayscale-JPEG implementation to execute.
        rotation: Clockwise quarter turn; never inferred patient orientation.
        output_size: Optional (width, height) after rotation; None preserves size.
        resize_filter: Explicit filter, retained even for an identity resize.
    """

    frame_index: int
    window: DisplayWindow | None = None
    decoder: PixelDecoder = PixelDecoder.NATIVE
    rotation: PageRotation = PageRotation.ROTATE_0
    output_size: tuple[int, int] | None = None
    resize_filter: ResizeFilter = ResizeFilter.NEAREST

    def __post_init__(self) -> None:
        _int(self.frame_index, 0, 4095)
        _require(self.window is None or type(self.window) is DisplayWindow)
        if self.window is not None:
            self.window.__post_init__()
        _require(
            type(self.decoder) is PixelDecoder and type(self.rotation) is PageRotation
        )
        _require(type(self.resize_filter) is ResizeFilter)
        if self.output_size is not None:
            _require(type(self.output_size) is tuple and len(self.output_size) == 2)
            for value in self.output_size:
                _int(value, 1, 65535)


@dataclass(frozen=True, slots=True, kw_only=True)
class PixelProvenanceLimits:
    """Cooperative budgets, not a native codec process sandbox.

    Args:
        max_source_bytes: Maximum complete encoded DICOM input size.
        max_header_bytes: Maximum parser position before pixel data.
        max_frame_pixels: Maximum source or resized output frame area.
        max_frames: Maximum declared source frame count.
        max_output_bytes: Maximum fresh encoded PNG size.
    """

    max_source_bytes: int = 64 * 1024 * 1024
    max_header_bytes: int = 1024 * 1024
    max_frame_pixels: int = 4_000_000
    max_frames: int = 512
    max_output_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for value, low, high in (
            (self.max_source_bytes, 132, 256 * 1024 * 1024),
            (self.max_header_bytes, 132, 8 * 1024 * 1024),
            (self.max_frame_pixels, 1, 16_000_000),
            (self.max_frames, 1, 4096),
            (self.max_output_bytes, 1, 32 * 1024 * 1024),
        ):
            _int(value, low, high)


@dataclass(frozen=True, slots=True, kw_only=True)
class PixelHeader:
    """Whitelisted numeric/enum source properties actually used by the renderer."""

    transfer_syntax: str
    rows: int
    columns: int
    frames: int
    samples: int
    bits_allocated: int
    bits_stored: int
    signed: int
    photometric: str
    planar_configuration: int
    slope: float
    intercept: float

    def __post_init__(self) -> None:
        _require(
            type(self.transfer_syntax) is str
            and self.transfer_syntax in (*_NATIVE, _RLE, _JPEG)
        )
        for value in (self.rows, self.columns):
            _int(value, 1, 65535)
        _int(self.frames, 1, 4096)
        _require(self.samples in (1, 3) and type(self.samples) is int)
        _require(self.bits_allocated in (8, 16) and type(self.bits_allocated) is int)
        _int(self.bits_stored, 1, self.bits_allocated)
        _int(self.signed, 0, 1)
        _int(self.planar_configuration, 0, 1)
        _require(type(self.photometric) is str and self.photometric in _PHOTOS)
        _finite(self.slope, 1e6)
        _finite(self.intercept, 1e9)
        _require(self.slope != 0)


@dataclass(frozen=True, slots=True)
class PixelStage:
    """One executed operation, with parameter/output hashes and predecessor link."""

    operation: str
    parameters_digest: str
    output_digest: str
    previous_digest: str

    def __post_init__(self) -> None:
        _require(type(self.operation) is str and self.operation in _OPERATIONS)
        for value in (self.parameters_digest, self.output_digest, self.previous_digest):
            _digest(value)

    @property
    def digest(self) -> str:
        """Bind operation, parameters, actual output and preceding stage."""
        self.__post_init__()
        return _hash([_SCHEMA, *asdict(self).values()])


@dataclass(frozen=True, slots=True, kw_only=True)
class PixelProvenance:
    """Canonical receipt for the actual source, runtime, operations and PNG.

    This is reproducibility evidence, not a signature or clinical certification.
    Reuse requires reproduce_dicom_frame or verify_region_provenance, which
    reconstruct the pixels instead of trusting this receipt's declarations.
    """

    source_digest: str
    frame_reference: str
    spec: PixelRenderSpec
    header: PixelHeader
    runtime_digest: str
    stages: tuple[PixelStage, ...]
    size: tuple[int, int]
    pixels_digest: str
    image_digest: str

    def __post_init__(self) -> None:
        for value in (
            self.source_digest,
            self.frame_reference,
            self.runtime_digest,
            self.pixels_digest,
            self.image_digest,
        ):
            _digest(value)
        _require(
            type(self.spec) is PixelRenderSpec and type(self.header) is PixelHeader
        )
        self.spec.__post_init__()
        self.header.__post_init__()
        _require(
            self.frame_reference == _hash([self.source_digest, self.spec.frame_index])
        )
        _require(self.spec.frame_index < self.header.frames)
        _require(type(self.stages) is tuple and len(self.stages) == len(_OPERATIONS))
        previous = self.source_digest
        for stage, operation in zip(self.stages, _OPERATIONS, strict=True):
            _require(type(stage) is PixelStage)
            stage.__post_init__()
            _require(stage.operation == operation and stage.previous_digest == previous)
            previous = stage.digest
        _require(self.stages[-1].output_digest == self.image_digest)
        _require(self.stages[-2].output_digest == self.pixels_digest)
        _require(type(self.size) is tuple and len(self.size) == 2)
        for coordinate in self.size:
            _int(coordinate, 1, 65535)

    @property
    def digest(self) -> str:
        """Bind the complete receipt, including explicit no-op operations."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        """Serialize only whitelisted numeric/enum metadata and hashes."""
        self.__post_init__()
        return _json({"schema_version": _SCHEMA, **asdict(self)})


@dataclass(frozen=True, slots=True)
class ProvenancedFrame:
    """Protected PNG and its executed pixel history; no source tags are retained."""

    provenance: PixelProvenance
    image: ImageInput = field(repr=False)


@dataclass(frozen=True, slots=True)
class RegionProvenance:
    """Existing image-space region bound to the full chain and actual crop pixels."""

    provenance_digest: str
    region: ImageRegion
    pixels_digest: str

    def __post_init__(self) -> None:
        _digest(self.provenance_digest)
        _digest(self.pixels_digest)
        _require(type(self.region) is ImageRegion)
        self.region.__post_init__()


@dataclass(frozen=True, slots=True)
class ProvenancedInference:
    """Actual local provider result and region bindings, with guarded evidence reuse."""

    provenance: PixelProvenance
    regions: tuple[RegionProvenance, ...]
    result: VLMResult = field(repr=False)

    def require_evidence(
        self, source: bytes, *, limits: PixelProvenanceLimits = PixelProvenanceLimits()
    ) -> tuple[VisualEvidence, ...]:
        """Reproduce source pixels and bindings before exposing non-abstained evidence.

        Args:
            source: Exact original protected DICOM bytes.
            limits: Current execution limits; no receipt can override them.
        """

        def verify():
            _require(type(self.result) is VLMResult)
            self.result.__post_init__()
            _require(self.result.abstention is None)
            verified = verify_region_provenance(
                source, self.provenance, self.regions, limits=limits
            )
            _require(verified == tuple(item.region for item in self.result.evidence))
            return self.result.evidence

        return _safe(verify)

    def to_json(self) -> str:
        """Return chain and region evidence without pixels, prompts or generated text."""
        self.provenance.__post_init__()
        _require(type(self.regions) is tuple and len(self.regions) <= 256)
        for region in self.regions:
            _require(type(region) is RegionProvenance)
            region.__post_init__()
        return _json(
            {
                "schema_version": _SCHEMA,
                "provenance": json.loads(self.provenance.to_json()),
                "regions": [asdict(region) for region in self.regions],
                "result": json.loads(self.result.to_json()),
                "notice": self.result.notice,
            }
        )


@contextmanager
def _quiet_decoders():
    from PIL import Image

    Image.init()
    guard = _QuietParser()
    loggers = [
        logger
        for name, logger in logging.Logger.manager.loggerDict.copy().items()
        if (name == "PIL" or name.startswith("PIL."))
        and isinstance(logger, logging.Logger)
    ]
    with _quiet_parser():
        for logger in loggers:
            logger.addFilter(guard)
        try:
            yield
            _require(not guard.warned)
        finally:
            for logger in loggers:
                logger.removeFilter(guard)


def _array_digest(array):
    import numpy as np

    _require(array.dtype.kind in ("i", "u", "f") and array.dtype.itemsize in (1, 2, 8))
    canonical = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C")
    digest = hashlib.sha256(
        _json(["openmed.pixel-array.v1", canonical.dtype.str, canonical.shape]).encode()
    )
    digest.update(b"\x00")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _runtime_digest():
    import zlib

    import numpy
    import PIL
    import pydicom
    from PIL import features

    return _hash(
        [
            pydicom.__version__,
            numpy.__version__,
            PIL.__version__,
            zlib.ZLIB_RUNTIME_VERSION,
            features.version_codec("jpg"),
        ]
    )


def _rescale(ds, count, index):
    from pydicom.dataset import Dataset

    shared = getattr(ds, "SharedFunctionalGroupsSequence", ())
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", ())
    _require(len(shared) <= 1 and len(per_frame) in (0, count))
    for owner in (
        per_frame[index] if per_frame else Dataset(),
        shared[0] if shared else Dataset(),
        ds,
    ):
        _require(not hasattr(owner, "ModalityLUTSequence"))
        sequence = getattr(owner, "PixelValueTransformationSequence", None)
        if sequence is not None:
            _require(len(sequence) == 1)
            owner = sequence[0]
            _require(not hasattr(owner, "ModalityLUTSequence"))
        present = (hasattr(owner, "RescaleSlope"), hasattr(owner, "RescaleIntercept"))
        if any(present):
            _require(all(present))
            slope, intercept = float(owner.RescaleSlope), float(owner.RescaleIntercept)
            _finite(slope, 1e6)
            _finite(intercept, 1e9)
            _require(slope != 0)
            return slope, intercept
        _require(sequence is None)
    return 1.0, 0.0


def _header(source, spec, limits):
    import pydicom

    preflight = read_dicom_transfer_syntax(source, max_bytes=limits.max_header_bytes)
    syntax = preflight.transfer_syntax_uid
    _require(
        preflight.outcome
        is (
            TransferSyntaxOutcome.NATIVE
            if spec.decoder is PixelDecoder.NATIVE
            else TransferSyntaxOutcome.OPTIONAL_DECODER
        )
    )
    expected = {
        PixelDecoder.NATIVE: _NATIVE,
        PixelDecoder.RLE: (_RLE,),
        PixelDecoder.JPEG: (_JPEG,),
    }
    _require(syntax in expected[spec.decoder])
    ds = pydicom.dcmread(
        _Header(source, limits.max_header_bytes),
        stop_before_pixels=True,
        specific_tags=_TAGS,
    )
    _require(str(ds.file_meta.TransferSyntaxUID) == syntax)
    rows, cols = (
        _header_integer(getattr(ds, key), 1, 65535) for key in ("Rows", "Columns")
    )
    _require(rows * cols <= limits.max_frame_pixels)
    count = _header_integer(getattr(ds, "NumberOfFrames", 1), 1, limits.max_frames)
    _require(spec.frame_index < count)
    bits = _header_integer(ds.BitsAllocated, 8, 16)
    stored = _header_integer(ds.BitsStored, 1, bits)
    _require(bits in (8, 16) and _header_integer(ds.HighBit, 0, bits - 1) == stored - 1)
    samples = _header_integer(ds.SamplesPerPixel, 1, 3)
    photo = str(ds.PhotometricInterpretation)
    signed = _header_integer(ds.PixelRepresentation, 0, 1)
    planar = _header_integer(getattr(ds, "PlanarConfiguration", 0), 0, 1)
    _require(
        not hasattr(ds, "ModalityLUTSequence")
        and not hasattr(ds, "PresentationLUTSequence")
    )
    _require(str(getattr(ds, "PresentationLUTShape", "IDENTITY")) == "IDENTITY")
    slope, intercept = _rescale(ds, count, spec.frame_index)
    if samples == 1:
        _require(photo in ("MONOCHROME1", "MONOCHROME2") and spec.window is not None)
    else:
        _require(samples == 3 and photo in ("RGB", "YBR_FULL", "YBR_FULL_422"))
        _require(bits == stored == 8 and signed == 0 and spec.window is None)
        _require(slope == 1 and intercept == 0 and hasattr(ds, "PlanarConfiguration"))
        if photo == "YBR_FULL_422":
            _require(
                spec.decoder is PixelDecoder.NATIVE and cols % 2 == 0 and planar == 0
            )
    if spec.decoder is PixelDecoder.JPEG:
        _require(samples == 1 and bits == stored == 8 and signed == 0)
    return PixelHeader(
        transfer_syntax=syntax,
        rows=rows,
        columns=cols,
        frames=count,
        samples=samples,
        bits_allocated=bits,
        bits_stored=stored,
        signed=signed,
        photometric=photo,
        planar_configuration=planar,
        slope=slope,
        intercept=intercept,
    )


def _validate_rle(payload, header):
    # Bound PackBits expansion before pydicom allocates a decoded segment.
    _require(len(payload) >= 64)
    segments = int.from_bytes(payload[:4], "little")
    _require(segments == header.samples * (header.bits_allocated // 8))
    offsets = [
        int.from_bytes(payload[4 + i * 4 : 8 + i * 4], "little")
        for i in range(segments)
    ]
    _require(offsets[0] == 64 and all(a < b for a, b in zip(offsets, offsets[1:])))
    offsets.append(len(payload))
    expected = header.rows * header.columns
    for start, stop in zip(offsets, offsets[1:]):
        _require(64 <= start < stop <= len(payload))
        position, expanded = start, 0
        while position < stop:
            if expanded == expected and position == stop - 1 and payload[position] == 0:
                position += 1
                break
            control = payload[position]
            position += 1
            if control < 128:
                position += control + 1
                expanded += control + 1
            elif control > 128:
                position += 1
                expanded += 257 - control
            _require(position <= stop and expanded <= expected)
        _require(expanded == expected)


def _frame_payload(source, header, spec):
    import pydicom
    from PIL import Image
    from pydicom.encaps import get_frame

    ds = pydicom.dcmread(
        BytesIO(source),
        specific_tags=[
            "PixelData",
            "ExtendedOffsetTable",
            "ExtendedOffsetTableLengths",
        ],
    )
    data = ds.PixelData
    _require(type(data) is bytes)
    if spec.decoder is PixelDecoder.NATIVE:
        samples = 2 if header.photometric == "YBR_FULL_422" else header.samples
        length = header.rows * header.columns * samples * (header.bits_allocated // 8)
        required = length * header.frames
        _require(len(data) == required + required % 2)
        return data[spec.frame_index * length : (spec.frame_index + 1) * length]
    offsets = None
    if hasattr(ds, "ExtendedOffsetTable") or hasattr(ds, "ExtendedOffsetTableLengths"):
        _require(
            hasattr(ds, "ExtendedOffsetTable")
            and hasattr(ds, "ExtendedOffsetTableLengths")
        )
        offsets = (ds.ExtendedOffsetTable, ds.ExtendedOffsetTableLengths)
        _require(
            all(
                type(value) is bytes and len(value) == header.frames * 8
                for value in offsets
            )
        )
    payload = get_frame(
        data, spec.frame_index, number_of_frames=header.frames, extended_offsets=offsets
    )
    if spec.decoder is PixelDecoder.RLE:
        _validate_rle(payload, header)
    else:
        with Image.open(BytesIO(payload)) as image:
            _require(
                image.format == "JPEG"
                and image.mode == "L"
                and image.size == (header.columns, header.rows)
            )
    return payload


class _Output(BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        _require(self.tell() + len(data) <= self.limit)
        return super().write(data)


def _render(source, spec, limits, cancel_check=None):
    import numpy as np
    from PIL import Image
    from pydicom.pixels import convert_color_space, pixel_array

    def cancel():
        _require(cancel_check is None or not cancel_check())

    _require(
        type(source) is bytes
        and type(spec) is PixelRenderSpec
        and type(limits) is PixelProvenanceLimits
    )
    spec.__post_init__()
    limits.__post_init__()
    _require(0 < len(source) <= limits.max_source_bytes)
    if spec.output_size is not None:
        _require(math.prod(spec.output_size) <= limits.max_frame_pixels)
    cancel()
    stages = []
    source_digest = hashlib.sha256(source).hexdigest()
    frame_reference = _hash([source_digest, spec.frame_index])
    runtime = _runtime_digest()

    def stage(operation, parameters, output_digest):
        cancel()
        stages.append(
            PixelStage(
                operation,
                _hash(parameters),
                output_digest,
                stages[-1].digest if stages else source_digest,
            )
        )

    with _quiet_decoders():
        header = _header(source, spec, limits)
        stage(
            "transfer_syntax",
            [header.transfer_syntax, spec.decoder.value, runtime],
            source_digest,
        )
        payload = _frame_payload(source, header, spec)
        stage(
            "frame",
            [spec.frame_index, header.frames, frame_reference],
            hashlib.sha256(payload).hexdigest(),
        )
        plugin = {
            PixelDecoder.NATIVE: "",
            PixelDecoder.RLE: "pydicom",
            PixelDecoder.JPEG: "pillow",
        }[spec.decoder]
        array = pixel_array(
            BytesIO(source), index=spec.frame_index, raw=True, decoding_plugin=plugin
        )
        expected = (header.rows, header.columns) + ((3,) if header.samples == 3 else ())
        _require(
            array.shape == expected
            and array.dtype.kind in ("i", "u")
            and array.dtype.itemsize == header.bits_allocated // 8
        )
        _require((array.dtype.kind == "i") == bool(header.signed))
        stage(
            "decode",
            [asdict(header), plugin, "raw_unpack_reshape_subsampling_signedness"],
            _array_digest(array),
        )
        if header.samples == 1:
            array = array.astype(np.float64) * header.slope + header.intercept
        stage(
            "rescale",
            [header.slope, header.intercept, header.samples == 1],
            _array_digest(array),
        )
        if spec.window is not None:
            window = spec.window
            array = np.floor(
                np.clip(
                    (array - (window.center - window.width / 2)) / window.width, 0, 1
                )
                * 255
            ).astype("uint8")
        stage(
            "window",
            [asdict(spec.window) if spec.window else None, "linear_clip_floor_uint8"],
            _array_digest(array),
        )
        if header.samples == 1:
            if header.photometric == "MONOCHROME1":
                array = 255 - array
            array = np.repeat(array[:, :, None], 3, axis=2)
        elif header.photometric in ("YBR_FULL", "YBR_FULL_422"):
            array = convert_color_space(array, "YBR_FULL", "RGB")
        stage("color", [header.photometric, "RGB"], _array_digest(array))
        array = np.rot90(array, k=-(int(spec.rotation) // 90))
        stage(
            "orientation",
            [int(spec.rotation), "stored_pixels_clockwise"],
            _array_digest(array),
        )
        with Image.fromarray(array) as image:
            size = spec.output_size or image.size
            mode = (
                Image.Resampling.NEAREST
                if spec.resize_filter is ResizeFilter.NEAREST
                else Image.Resampling.BILINEAR
            )
            with image.resize(size, mode) as resized:
                pixels_digest = _array_digest(np.asarray(resized))
                stage("resize", [size, spec.resize_filter.value], pixels_digest)
                with _Output(limits.max_output_bytes) as stream:
                    resized.save(stream, format="PNG", optimize=False)
                    content = stream.getvalue()
        with Image.open(BytesIO(content)) as decoded:
            _require(
                decoded.mode == "RGB" and decoded.size == size and not decoded.info
            )
            _require(_array_digest(np.asarray(decoded)) == pixels_digest)
        image = ImageInput(
            data=content, media_type="image/png", width=size[0], height=size[1]
        )
        stage("encode", ["PNG", "RGB", "fresh_no_metadata"], image.digest)
    provenance = PixelProvenance(
        source_digest=source_digest,
        frame_reference=frame_reference,
        spec=spec,
        header=header,
        runtime_digest=runtime,
        stages=tuple(stages),
        size=size,
        pixels_digest=pixels_digest,
        image_digest=image.digest,
    )
    cancel()
    return ProvenancedFrame(provenance, image)


def render_dicom_frame(
    source: bytes,
    *,
    spec: PixelRenderSpec,
    limits: PixelProvenanceLimits = PixelProvenanceLimits(),
    cancel_check: Callable[[], bool] | None = None,
) -> ProvenancedFrame:
    """Execute the declared local pipeline and record actual intermediate hashes.

    Args:
        source: Complete protected DICOM bytes; never a filename or URL.
        spec: Explicit source frame, decoder and display transformations.
        limits: Resource ceilings checked before allocation and output encoding.
        cancel_check: Cooperative cancellation between stages; not codec preemption.

    Returns:
        Fresh protected RGB PNG with a canonical, value-free digest chain.

    Raises:
        PixelProvenanceError: Unsupported encoding, ambiguous transforms, invalid
            geometry, resource/cancellation failure or decoder/encoder error.
    """
    return _safe(lambda: _render(source, spec, limits, cancel_check))


def reproduce_dicom_frame(
    source: bytes,
    provenance: PixelProvenance,
    *,
    limits: PixelProvenanceLimits = PixelProvenanceLimits(),
) -> ProvenancedFrame:
    """Re-execute every stage and reject any difference from the recorded chain.

    Args:
        source: Exact original source bytes, supplied by the caller's protected store.
        provenance: Previously returned chain, never treated as self-authenticating.
        limits: Current resource ceilings; insufficient budgets reject reproduction.
    """

    def reproduce():
        _require(type(provenance) is PixelProvenance)
        provenance.__post_init__()
        actual = _render(source, provenance.spec, limits)
        _require(actual.provenance == provenance)
        return actual

    return _safe(reproduce)


def _bind(frame, regions):
    import numpy as np
    from PIL import Image

    _require(type(regions) is tuple and len(regions) <= 256)
    bindings = []
    with Image.open(BytesIO(frame.image.data)) as image:
        for region in regions:
            _require(type(region) is ImageRegion)
            region.__post_init__()
            _require(
                region.image_digest == frame.image.digest
                and region.x1 <= image.width
                and region.y1 <= image.height
            )
            with image.crop((region.x0, region.y0, region.x1, region.y1)) as crop:
                bindings.append(
                    RegionProvenance(
                        frame.provenance.digest, region, _array_digest(np.asarray(crop))
                    )
                )
    return tuple(bindings)


def bind_region_provenance(
    source: bytes,
    provenance: PixelProvenance,
    regions: tuple[ImageRegion, ...],
    *,
    limits: PixelProvenanceLimits = PixelProvenanceLimits(),
) -> tuple[RegionProvenance, ...]:
    """Reproduce actual pixels before attaching the chain to downstream regions.

    Args:
        source: Protected original source bytes.
        provenance: Receipt from render_dicom_frame.
        regions: Existing encoded-image-bound regions in the rendered pixel space.
        limits: Current reproduction resource ceilings.
    """
    return _safe(
        lambda: _bind(reproduce_dicom_frame(source, provenance, limits=limits), regions)
    )


def verify_region_provenance(
    source: bytes,
    provenance: PixelProvenance,
    regions: tuple[RegionProvenance, ...],
    *,
    limits: PixelProvenanceLimits = PixelProvenanceLimits(),
) -> tuple[ImageRegion, ...]:
    """Reproduce and compare every chain/image/region/crop binding before reuse.

    Args:
        source: Protected original source bytes.
        provenance: Complete expected render receipt.
        regions: Previously bound downstream regions, in their original order.
        limits: Current resource ceilings for actual reproduction.
    """

    def verify():
        _require(type(regions) is tuple and len(regions) <= 256)
        for binding in regions:
            _require(type(binding) is RegionProvenance)
            binding.__post_init__()
        frame = reproduce_dicom_frame(source, provenance, limits=limits)
        plain = tuple(binding.region for binding in regions)
        _require(_bind(frame, plain) == regions)
        return plain

    return _safe(verify)


def run_provenanced_dicom_vlm(
    source: bytes,
    provider: EdgeVLMProvider,
    prompt: Prompt,
    *,
    spec: PixelRenderSpec,
    regions: tuple[Box, ...] = (),
    limits: PixelProvenanceLimits = PixelProvenanceLimits(),
) -> ProvenancedInference:
    """Infer on actual rendered pixels and attach reproduction-checked evidence.

    Args:
        source: Protected source bytes for the explicitly selected frame.
        provider: Qualified, already loaded local VLM adapter.
        prompt: Protected prompt and generation limits from the existing contract.
        spec: Actual decode/display operations to execute.
        regions: Optional half-open rendered-pixel boxes restricting model evidence.
        limits: Current rendering/reproduction resource ceilings.

    Returns:
        Protected result plus source/chain/crop bindings for its VisualEvidence.
        require_evidence revalidates source pixels and rejects model abstentions.
        Frame references join the sampler's SourceFrame records; this call alone
        does not claim series sampling coverage or de-identification.
    """

    def run():
        frame = _render(source, spec, limits)
        _require(type(regions) is tuple and len(regions) <= 256)
        selected = []
        for box in regions:
            _require(type(box) is tuple and len(box) == 4)
            selected.append(
                ImageRegion(
                    image_digest=frame.image.digest,
                    x0=box[0],
                    y0=box[1],
                    x1=box[2],
                    y1=box[3],
                )
            )
        request = VLMRequest(image=frame.image, prompt=prompt, regions=tuple(selected))
        result = run_local_vlm(provider, request)
        bindings = bind_region_provenance(
            source,
            frame.provenance,
            tuple(item.region for item in result.evidence),
            limits=limits,
        )
        return ProvenancedInference(frame.provenance, bindings, result)

    return _safe(run)
