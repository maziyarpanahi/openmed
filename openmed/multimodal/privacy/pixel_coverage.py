"""Execute bounded regional privacy checks before releasing fresh PNG rasters.

Only the returned PNG bundle is gated. The input asset, DICOM tags, source
containers and unrendered representations are never certified or returned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any, Literal

from ..evidence.fused_graph import EvidenceNode, NodeKind, PageGeometry
from ..page_rotation import PageRotation
from ..render_raster import RasterRedactionPage, render_redacted_raster_pages
from ..vision.dicom.series_sampler import (
    DisplayWindow,
    SeriesSamplingPlan,
    _header_integer,
    _quiet_parser,
)
from .mask_lineage import CoordinateSpace, Rectangle, TransformStep, propagate_masks

Box = tuple[int, int, int, int]
_SCHEMA = "openmed.pixel-coverage.v1"


class PixelCoverageError(ValueError):
    """Fixed-code rejection without source content or decoder exceptions."""


def _require(condition: bool) -> None:
    if not condition:
        raise PixelCoverageError("pixel_coverage_invalid_input")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _digest(value: str) -> None:
    _require(type(value) is str and len(value) == 64)
    _require(all(c in "0123456789abcdef" for c in value))


def _box(value: Box, width: int, height: int) -> None:
    _require(type(value) is tuple and len(value) == 4)
    _require(all(type(v) is int for v in value))
    x0, y0, x1, y1 = value
    _require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)


@dataclass(frozen=True, slots=True, kw_only=True)
class PixelCoverageLimits:
    """Cooperative memory/work ceilings; these do not sandbox native decoders."""

    max_source_bytes: int = 32 * 1024 * 1024
    max_frames: int = 128
    max_frame_pixels: int = 4_000_000
    max_total_pixels: int = 16_000_000
    max_regions: int = 64
    max_output_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for value, ceiling in (
            (self.max_source_bytes, 128 * 1024 * 1024),
            (self.max_frames, 1024),
            (self.max_frame_pixels, 16_000_000),
            (self.max_total_pixels, 64_000_000),
            (self.max_regions, 64),
            (self.max_output_bytes, 256 * 1024 * 1024),
        ):
            _require(type(value) is int and 1 <= value <= ceiling)


@dataclass(frozen=True, slots=True)
class PixelInspection:
    """A detector's completed crop inspection and critical boxes in crop pixels.

    All returned boxes require masking; uncertain or truncated inference must
    set complete=False. Text, labels and confidence prose have no report field.
    """

    critical_regions: tuple[Box, ...] = ()
    complete: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalPixelDetector:
    """Caller-qualified local detector, identified by model/configuration digest.

    Args:
        digest: SHA-256 covering the chosen model, revision and preprocessing.
        inspect: Inspect a fresh metadata-free RGB Pillow crop in memory.
            Return PixelInspection in that crop's coordinate system.
        local_only: Explicit confirmation of no network, logging or disk writes.

    A declaration is not a process sandbox or recall certification. The caller
    must qualify the implementation, including empty results and crop context.
    """

    digest: str
    inspect: Callable[[Any], PixelInspection] = field(repr=False, compare=False)
    local_only: bool

    def __post_init__(self) -> None:
        _digest(self.digest)
        _require(callable(self.inspect) and self.local_only is True)


@dataclass(frozen=True, slots=True, kw_only=True)
class PixelFrame:
    """Actual decoded carrier geometry in stored, top-left pixel coordinates.

    Overlay planes are independent rasters, including pixels outside the main
    image. They are not composited or interpreted as a presentation state.
    """

    reference: str
    source_digest: str
    carrier: Literal["image", "icon", "overlay"]
    carrier_index: int
    frame_index: int
    width: int
    height: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PixelFramePlan:
    """Regions to inspect and mask, plus the actual crop/rotation/resize request.

    Args:
        reference: Exact frame reference from inventory_pixel_frames.
        inspected_regions: None scans the entire frame; an explicit tuple must
            cover the entire frame without holes. Empty means uninspected.
        ignored_regions: Explicit uninspected areas; any such area holds export.
        masks: Source-space black redaction rectangles.
        critical_regions: Additional known identifier regions, even if OCR misses.
        crop: Optional source-space crop. Losing protected support is rejected.
        rotation: Clockwise lossless quarter turn after cropping.
        output_size: Optional nearest-neighbor resize after rotation.
    """

    reference: str
    inspected_regions: tuple[Box, ...] | None = None
    ignored_regions: tuple[Box, ...] = ()
    masks: tuple[Box, ...] = ()
    critical_regions: tuple[Box, ...] = ()
    crop: Box | None = None
    rotation: PageRotation = PageRotation.ROTATE_0
    output_size: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class PixelCoverageResult:
    """Value-free regional evidence, with no output bytes on any failed gate."""

    accepted: bool
    failure: str | None
    report_json: str
    outputs: tuple[bytes, ...] = field(default=(), repr=False)

    def to_json(self) -> str:
        """Return deterministic geometry/hash evidence; never source or output data."""
        return self.report_json


@dataclass
class _Decoded:
    frame: PixelFrame
    image: Any = field(repr=False)


class _GateFailure(Exception):
    def __init__(self, code: str) -> None:
        self.code = (
            code
            if type(code) is str
            and code
            in {
                "cancelled",
                "unsupported_pixel_carrier",
                "detector_incomplete",
                "mask_lineage_rejected",
                "frame_coverage_incomplete",
                "regional_coverage_incomplete",
                "critical_region_unresolved",
                "residual_identifier",
            }
            else "pixel_audit_failed"
        )


def _cancel(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise _GateFailure("cancelled")


def _size(width: int, height: int, limits: PixelCoverageLimits) -> None:
    _require(type(width) is int and type(height) is int)
    _require(0 < width and 0 < height and width * height <= limits.max_frame_pixels)


class _QuietMedia(logging.Filter):
    def __init__(self):
        super().__init__()
        self.thread = threading.get_ident()
        self.warned = False

    def filter(self, record):
        if record.thread != self.thread:
            return True
        self.warned |= record.levelno >= logging.WARNING
        return False


@contextmanager
def _quiet_media():
    from PIL import Image

    Image.init()
    guard = _QuietMedia()
    loggers = [logging.getLogger("PIL")]
    loggers.extend(
        logger
        for name, logger in logging.Logger.manager.loggerDict.copy().items()
        if name.startswith("PIL.") and isinstance(logger, logging.Logger)
    )
    # Decoder diagnostics can contain metadata even at DEBUG. Suppress only
    # this thread, restore every logger and reject decoder warnings.
    with _quiet_parser():
        for logger in loggers:
            logger.addFilter(guard)
        try:
            yield
            _require(not guard.warned)
        finally:
            for logger in loggers:
                logger.removeFilter(guard)


def _decode(source, kind, window, limits, frames, cancel):
    from PIL import Image

    _require(type(source) is bytes and 0 < len(source) <= limits.max_source_bytes)
    _require(kind in ("raster", "dicom"))
    digest = hashlib.sha256(source).hexdigest()
    total = 0

    def append(image, carrier, carrier_index, index):
        nonlocal total
        _cancel(cancel)
        width, height = image.size
        _size(width, height, limits)
        total += width * height
        _require(total <= limits.max_total_pixels and len(frames) < limits.max_frames)
        # Only flattened display pixels survive. Metadata, alpha-channel payloads
        # and thumbnails in the original container never reach the PNG encoder.
        with image.convert("RGBA") as rgba:
            with Image.new("RGBA", image.size, "white") as white:
                with Image.alpha_composite(white, rgba) as flat:
                    with flat.convert("RGB") as rgb:
                        pixels = Image.frombytes("RGB", image.size, rgb.tobytes())
        frame = PixelFrame(
            reference=_hash([digest, carrier, carrier_index, index]),
            source_digest=digest,
            carrier=carrier,
            carrier_index=carrier_index,
            frame_index=index,
            width=width,
            height=height,
        )
        frames.append(_Decoded(frame, pixels))

    if kind == "raster":
        _require(window is None)
        with Image.open(BytesIO(source)) as image:
            _require(image.format in ("PNG", "JPEG", "TIFF", "GIF", "BMP", "WEBP"))
            count = getattr(image, "n_frames", 1)
            _require(0 < count <= limits.max_frames)
            for index in range(count):
                _cancel(cancel)
                image.seek(index)
                _size(*image.size, limits)
                append(image, "image", 0, index)
    else:
        _require(type(window) is DisplayWindow)
        _decode_dicom(source, window, limits, append, cancel)
    _require(bool(frames))


def _decode_dicom(source, window, limits, append, cancel):
    import numpy as np
    import pydicom
    from PIL import Image

    from ..dicom_transfer_syntax import read_dicom_transfer_syntax

    syntax = read_dicom_transfer_syntax(source)
    _require(
        syntax.ok
        and syntax.transfer_syntax_uid in ("1.2.840.10008.1.2", "1.2.840.10008.1.2.1")
    )
    with _quiet_parser():
        ds = pydicom.dcmread(BytesIO(source), force=False)
        _require(str(ds.file_meta.TransferSyntaxUID) == syntax.transfer_syntax_uid)
        seen_elements = 0

        def check_nested(dataset, depth=0, icon=False):
            nonlocal seen_elements
            _require(depth <= 8)
            for element in dataset:
                seen_elements += 1
                _require(seen_elements <= 10_000)
                if depth and 0x6000 <= element.tag.group <= 0x60FF:
                    raise _GateFailure("unsupported_pixel_carrier")
                if element.tag in (0x7FE00008, 0x7FE00009):
                    raise _GateFailure("unsupported_pixel_carrier")
                if element.tag == 0x7FE00010 and depth and not icon:
                    raise _GateFailure("unsupported_pixel_carrier")
                if element.VR == "SQ":
                    _require(len(element.value) <= limits.max_frames)
                    for item in element.value:
                        check_nested(
                            item, depth + 1, depth == 0 and element.tag == 0x00880200
                        )

        check_nested(ds)

        def pixels(dataset, carrier, ordinal):
            _cancel(cancel)
            rows = _header_integer(dataset.Rows, 1, limits.max_frame_pixels)
            cols = _header_integer(dataset.Columns, 1, limits.max_frame_pixels)
            _size(cols, rows, limits)
            count = _header_integer(
                getattr(dataset, "NumberOfFrames", 1), 1, limits.max_frames
            )
            samples = _header_integer(dataset.SamplesPerPixel, 1, 3)
            bits = _header_integer(dataset.BitsAllocated, 8, 16)
            stored = _header_integer(dataset.BitsStored, 1, bits)
            _require(bits in (8, 16) and int(dataset.HighBit) == stored - 1)
            signed = _header_integer(dataset.PixelRepresentation, 0, 1)
            photo = dataset.PhotometricInterpretation
            _require(
                (samples == 1 and photo in ("MONOCHROME1", "MONOCHROME2"))
                or (
                    samples == 3
                    and photo == "RGB"
                    and bits == stored == 8
                    and signed == 0
                )
            )
            if samples == 3:
                _require(dataset.PlanarConfiguration == 0)
            needed = rows * cols * count * samples * (bits // 8)
            _require(len(dataset.PixelData) == needed + needed % 2)
            _require(rows * cols * count <= limits.max_total_pixels)
            # Do not dispatch arbitrary compressed-pixel plugins. Native storage
            # is normalized explicitly, including stored-bit sign extension.
            raw = np.frombuffer(dataset.PixelData[:needed], dtype=f"<u{bits // 8}")
            values = raw.astype(np.int32) & ((1 << stored) - 1)
            if signed:
                values = (values ^ (1 << (stored - 1))) - (1 << (stored - 1))
            shape = (count, rows, cols, 3) if samples == 3 else (count, rows, cols)
            values = values.reshape(shape)
            slope = float(getattr(dataset, "RescaleSlope", 1))
            intercept = float(getattr(dataset, "RescaleIntercept", 0))
            _require(math.isfinite(slope) and math.isfinite(intercept) and slope != 0)
            # Enhanced per-frame LUT/windowing is deliberately not guessed.
            for name in (
                "SharedFunctionalGroupsSequence",
                "PerFrameFunctionalGroupsSequence",
            ):
                for group in getattr(dataset, name, ()):
                    _require(not hasattr(group, "PixelValueTransformationSequence"))
            for index, value in enumerate(values):
                _cancel(cancel)
                if samples == 1:
                    value = np.clip(
                        (value * slope + intercept - window.center + window.width / 2)
                        * (255 / window.width),
                        0,
                        255,
                    ).astype("uint8")
                    if photo == "MONOCHROME1":
                        value = 255 - value
                else:
                    value = value.astype("uint8")
                with Image.fromarray(value) as image:
                    append(image, carrier, ordinal, index)

        pixels(ds, "image", 0)
        for index, icon in enumerate(getattr(ds, "IconImageSequence", ())):
            pixels(icon, "icon", index)
        for group in range(0x6000, 0x6020, 2):
            if not any(tag.group == group for tag in ds.keys()):
                continue
            # Retired embedded overlays require separate handling; never silently
            # strip their high bits and then report complete source coverage.
            _require((group, 0x3000) in ds)
            rows = _header_integer(
                ds[(group, 0x0010)].value, 1, limits.max_frame_pixels
            )
            cols = _header_integer(
                ds[(group, 0x0011)].value, 1, limits.max_frame_pixels
            )
            _size(cols, rows, limits)
            count = (
                _header_integer(ds[(group, 0x0015)].value, 1, limits.max_frames)
                if (group, 0x0015) in ds
                else 1
            )
            _require(ds[(group, 0x0100)].value == 1 and ds[(group, 0x0102)].value == 0)
            needed = (rows * cols * count + 7) // 8
            data = ds[(group, 0x3000)].value
            _require(len(data) == needed + needed % 2)
            _require(rows * cols * count <= limits.max_total_pixels)
            bits = np.unpackbits(np.frombuffer(data, dtype="uint8"), bitorder="little")
            masks = bits[: rows * cols * count].reshape(count, rows, cols)
            for index, mask in enumerate(masks):
                with Image.fromarray(mask * 255) as image:
                    append(image, "overlay", (group - 0x6000) // 2, index)
        _require(not any(0x6020 <= tag.group <= 0x60FF for tag in ds.keys()))


def inventory_pixel_frames(
    source: bytes,
    *,
    source_kind: Literal["raster", "dicom"] = "raster",
    window: DisplayWindow | None = None,
    limits: PixelCoverageLimits = PixelCoverageLimits(),
) -> tuple[PixelFrame, ...]:
    """Decode bounded actual frame/carrier geometry without returning content.

    DICOM requires an explicit display window and native little-endian pixels.
    Fresh PNG exports exclude original metadata/embedded thumbnails. Native
    DICOM icon and overlay planes are inventoried as separate output rasters.
    """
    frames: list[_Decoded] = []
    result = None
    try:
        _require(type(limits) is PixelCoverageLimits)
        with _quiet_media():
            _decode(source, source_kind, window, limits, frames, None)
        result = tuple(item.frame for item in frames)
    except Exception:
        pass
    finally:
        for item in frames:
            item.image.close()
    if result is None:
        raise PixelCoverageError("pixel_inventory_rejected")
    return result


def _covered(region, boxes):
    # Disjoint vertical strips and merged intervals compute the union, so
    # overlapping boxes never hide a gap or count the same area twice.
    left, top, right, bottom = region
    edges = sorted(
        {left, right} | {max(left, min(right, b[i])) for b in boxes for i in (0, 2)}
    )
    for x0, x1 in zip(edges, edges[1:]):
        end = top
        intervals = sorted(
            (max(top, b[1]), min(bottom, b[3]))
            for b in boxes
            if b[0] <= x0 and b[2] >= x1
        )
        for y0, y1 in intervals:
            if y0 > end:
                break
            end = max(end, y1)
        if end < bottom:
            return False
    return True


def _regions(values, width, height, limits):
    _require(type(values) is tuple and len(values) <= limits.max_regions)
    for value in values:
        _box(value, width, height)


def _inspect(detector, image, regions, limits, cancel):
    findings = set()
    for region in regions:
        _cancel(cancel)
        with image.crop(region) as crop:
            inspection = detector.inspect(crop)
            _require(type(inspection) is PixelInspection)
            _require(type(inspection.complete) is bool)
            _regions(inspection.critical_regions, *crop.size, limits)
            if not inspection.complete:
                raise _GateFailure("detector_incomplete")
            for x0, y0, x1, y1 in inspection.critical_regions:
                findings.add(
                    (x0 + region[0], y0 + region[1], x1 + region[0], y1 + region[1])
                )
                _require(len(findings) <= limits.max_regions)
        _cancel(cancel)
    return tuple(sorted(findings))


def _transform(item, plan, limits, ordinal):
    from PIL import Image

    frame = item.frame
    space = CoordinateSpace(
        space_ref=_hash([frame.reference, 0])[:32],
        source_digest=frame.source_digest,
        geometry=PageGeometry(page=ordinal, width=frame.width, height=frame.height),
    )
    origin = space
    steps = []
    image = item.image.copy()

    def target(width, height):
        _size(width, height, limits)
        return CoordinateSpace(
            space_ref=_hash([frame.reference, len(steps) + 1])[:32],
            source_digest=frame.source_digest,
            geometry=PageGeometry(page=ordinal, width=width, height=height),
        )

    try:
        if plan.crop is not None:
            _box(plan.crop, *image.size)
            x0, y0, x1, y1 = plan.crop
            next_space = target(x1 - x0, y1 - y0)
            steps.append(TransformStep.crop(space, next_space, Rectangle(*plan.crop)))
            new_image = image.crop(plan.crop)
            image.close()
            image, space = new_image, next_space
        _require(type(plan.rotation) is PageRotation)
        if plan.rotation:
            width, height = image.size
            if plan.rotation in (PageRotation.ROTATE_90, PageRotation.ROTATE_270):
                width, height = height, width
            next_space = target(width, height)
            steps.append(TransformStep.rotate(space, next_space, plan.rotation))
            method = {
                90: Image.Transpose.ROTATE_270,
                180: Image.Transpose.ROTATE_180,
                270: Image.Transpose.ROTATE_90,
            }[plan.rotation]
            new_image = image.transpose(method)
            image.close()
            image, space = new_image, next_space
        if plan.output_size is not None:
            _require(type(plan.output_size) is tuple and len(plan.output_size) == 2)
            next_space = target(*plan.output_size)
            steps.append(TransformStep.scale(space, next_space, resampling_margin=0))
            new_image = image.resize(plan.output_size, Image.Resampling.NEAREST)
            image.close()
            image, space = new_image, next_space
        masks = plan.masks
        if masks:
            nodes = tuple(
                EvidenceNode(
                    node_id=_hash([frame.reference, "mask", index])[:32],
                    kind=NodeKind.PIXELS,
                    source_digest=frame.source_digest,
                    geometry=origin.geometry,
                    region=box,
                )
                for index, box in enumerate(masks)
            )
            lineage = propagate_masks(origin, nodes, tuple(steps))
            if not lineage.accepted:
                raise _GateFailure("mask_lineage_rejected")
            masks = tuple(mask.raster_box for mask in lineage.masks)
        return image, masks, tuple(step.to_dict() for step in steps)
    except Exception:
        image.close()
        raise


def audit_pixel_redaction(
    source: bytes,
    plans: tuple[PixelFramePlan, ...],
    *,
    detector: LocalPixelDetector,
    residual_detector: LocalPixelDetector,
    source_kind: Literal["raster", "dicom"] = "raster",
    window: DisplayWindow | None = None,
    sampling_plan: SeriesSamplingPlan | None = None,
    limits: PixelCoverageLimits = PixelCoverageLimits(),
    cancel_check: Callable[[], bool] | None = None,
) -> PixelCoverageResult:
    """Inspect every carrier, validate masks, render and gate fresh PNG outputs.

    Args:
        source: Protected encoded image or native DICOM bytes, read in memory.
        plans: Exactly one source-bound plan for every inventoried frame/carrier.
        detector: Qualified local detector called on every requested source crop.
        residual_detector: Explicit qualified detector called on each complete
            decoded output. May be the same engine, but never defaults to absent.
        source_kind: Raster or DICOM; unknown formats fail closed.
        window: Required DICOM display window; no inference from patient tags.
        sampling_plan: Optional DICOM VLM manifest for this single source. Frame
            mapping is checked against actual pixels; selection never grants
            privacy coverage or skips a scan. Model geometry is not requalified.
        limits: Bounded input, frame, region, pixel and encoded-output budgets.
        cancel_check: Cooperative cancellation between decode/inspect/export work.

    Returns:
        All PNGs only when every gate passes. Failure returns no partial output.
        Evidence tracks actual crops, ignored regions, critical boxes, source and
        projected masks, executed transforms, sampling, detector and output hashes.
        Acceptance applies to these PNG pixels, not the source file, detector
        recall, compliance, clinical correctness or a DICOM de-identification claim.
    """
    frames: list[_Decoded] = []
    records: list[dict[str, Any]] = []
    outputs: list[bytes] = []
    failure = None
    report: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "records": records,
        "source_container_released": False,
        "source_metadata": "excluded_from_fresh_png",
        "coordinates": "stored_top_left_pixels",
        "pixel_export": "flattened_rgb_png_per_carrier_frame",
    }
    try:
        _require(type(limits) is PixelCoverageLimits)
        _require(
            type(detector) is LocalPixelDetector
            and type(residual_detector) is LocalPixelDetector
        )
        detector.__post_init__()
        residual_detector.__post_init__()
        report.update(
            detector_digest=detector.digest,
            residual_detector_digest=residual_detector.digest,
            limits=asdict(limits),
        )
        with _quiet_media():
            _decode(source, source_kind, window, limits, frames, cancel_check)
        report.update(
            source_digest=frames[0].frame.source_digest,
            frame_count=len(frames),
            window=asdict(window) if window else None,
        )
        _require(type(plans) is tuple and len(plans) <= limits.max_frames)
        _require(all(type(plan) is PixelFramePlan for plan in plans))
        for plan in plans:
            _digest(plan.reference)
        by_ref = {plan.reference: plan for plan in plans}
        if len(by_ref) != len(plans) or set(by_ref) != {
            item.frame.reference for item in frames
        }:
            raise _GateFailure("frame_coverage_incomplete")
        selected = None
        if sampling_plan is not None:
            _require(
                source_kind == "dicom" and type(sampling_plan) is SeriesSamplingPlan
            )
            sampling_plan.__post_init__()
            actual = {
                (f.frame.source_digest, f.frame.frame_index)
                for f in frames
                if f.frame.carrier == "image"
            }
            _require(
                {(f.source_digest, f.frame_index) for f in sampling_plan.frames}
                == actual
            )
            selected = {
                (f.source_digest, f.frame_index)
                for f in sampling_plan.frames
                if f.reference in sampling_plan.selected
            }
            report["sampling_plan_digest"] = sampling_plan.digest
            report["sampling_is_privacy_coverage"] = False
        total_pixels = total_bytes = 0
        for ordinal, item in enumerate(frames):
            _cancel(cancel_check)
            frame = item.frame
            plan = by_ref[frame.reference]
            full = (0, 0, frame.width, frame.height)
            regions = (
                (full,) if plan.inspected_regions is None else plan.inspected_regions
            )
            for values in (
                regions,
                plan.ignored_regions,
                plan.masks,
                plan.critical_regions,
            ):
                _regions(values, frame.width, frame.height, limits)
            record: dict[str, Any] = {
                "frame": asdict(frame),
                "inspected_regions": regions,
                "ignored_regions": plan.ignored_regions,
                "source_masks": plan.masks,
                "known_critical_regions": plan.critical_regions,
                "vlm_selected": (frame.source_digest, frame.frame_index) in selected
                if selected is not None and frame.carrier == "image"
                else None,
                "complete": False,
            }
            records.append(record)
            if plan.ignored_regions or not _covered(full, regions):
                raise _GateFailure("regional_coverage_incomplete")
            critical = _inspect(detector, item.image, regions, limits, cancel_check)
            record["detected_critical_regions"] = critical
            unresolved = tuple(
                box
                for box in (*critical, *plan.critical_regions)
                if not _covered(box, plan.masks)
            )
            record["unresolved_regions"] = unresolved
            if unresolved:
                raise _GateFailure("critical_region_unresolved")
            transformed, masks, steps = _transform(item, plan, limits, ordinal)
            with transformed:
                total_pixels += transformed.width * transformed.height
                _require(total_pixels <= limits.max_total_pixels)
                record.update(
                    transforms=steps, output_masks=masks, output_size=transformed.size
                )
                exported = render_redacted_raster_pages(
                    (RasterRedactionPage(transformed, masks),),
                    output_format="png",
                    padding_pixels=0,
                    max_page_pixels=limits.max_frame_pixels,
                    max_total_pixels=limits.max_total_pixels,
                    max_regions=limits.max_regions,
                    max_output_bytes=limits.max_output_bytes - total_bytes,
                    cancel_check=cancel_check,
                )
            from PIL import Image

            with Image.open(BytesIO(exported.content)) as decoded:
                residual = _inspect(
                    residual_detector,
                    decoded,
                    ((0, 0, *decoded.size),),
                    limits,
                    cancel_check,
                )
            record["residual_regions"] = residual
            if residual:
                raise _GateFailure("residual_identifier")
            record.update(
                complete=True,
                output_sha256=hashlib.sha256(exported.content).hexdigest(),
                pixel_verification="exact_decoded_raster",
            )
            total_bytes += len(exported.content)
            outputs.append(exported.content)
        _cancel(cancel_check)
    except _GateFailure as error:
        failure = error.code
    except Exception:
        failure = "pixel_audit_failed"
    finally:
        for item in frames:
            item.image.close()
    report.update(accepted=failure is None, failure=failure)
    return PixelCoverageResult(
        failure is None,
        failure,
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False),
        tuple(outputs) if failure is None else (),
    )
