"""Deterministic DICOM frame selection with explicit unexamined coverage.

Native pixel previews are selection signals, not clinical quality assessments.
Encoded source objects and model outputs are protected in-memory payloads.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
import threading
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any, TypeVar, cast

from openmed.multimodal.dicom_transfer_syntax import read_dicom_transfer_syntax
from openmed.multimodal.vision.providers.base import (
    EdgeVLMProvider,
    ImageInput,
    Prompt,
    VLMRequest,
    VLMResult,
    run_local_vlm,
)

T = TypeVar("T")
_PARSER_LOCK = threading.RLock()
_SCHEMA = "openmed.dicom.series_sampling.v1"
_NOTICE = (
    "Evidence covers only the successfully inferred selected frames. Unexamined "
    "frames may contain relevant findings. Non-diagnostic; human review required."
)
_TAGS = [
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "FrameOfReferenceUID",
    "Modality",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "SamplesPerPixel",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "AcquisitionNumber",
    "TemporalPositionIdentifier",
    "EchoNumbers",
    "RescaleSlope",
    "RescaleIntercept",
    "SharedFunctionalGroupsSequence",
    "PerFrameFunctionalGroupsSequence",
]


class SeriesSamplingError(ValueError):
    """Fixed-code rejection without source tags, filenames or backend context."""


def _require(value: bool) -> None:
    if not value:
        raise SeriesSamplingError("dicom_series_sampling_rejected")


def _integer(value: object, minimum: int, maximum: int) -> None:
    _require(type(value) is int and minimum <= value <= maximum)


def _header_integer(value: object, minimum: int, maximum: int) -> int:
    # pydicom can expose an ISfloat when application validation is disabled.
    # Never silently truncate fractional acquisition or frame-count metadata.
    encoded = str(value).strip()
    _require(len(encoded) <= 16 and re.fullmatch(r"[+-]?[0-9]+", encoded) is not None)
    number = int(encoded)
    _integer(number, minimum, maximum)
    return number


def _uid(value: str) -> None:
    _require(
        0 < len(value) <= 64
        and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))+", value)
        is not None
    )


def _finite(value: object, minimum: float, maximum: float) -> None:
    _require(type(value) in (int, float))
    number = cast("int | float", value)
    _require(minimum <= number <= maximum and math.isfinite(number))


def _digest(value: object) -> None:
    _require(
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _protected(call: Callable[[], T]) -> T:
    failed = False
    result: T | None = None
    try:
        result = call()
    except Exception:
        failed = True
    if failed:
        raise SeriesSamplingError("dicom_series_sampling_rejected")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingPolicy:
    """Bound work and require spatial/acquisition coverage before diversity.

    Args:
        max_instances: Maximum caller-supplied DICOM objects.
        max_frames: Maximum total source frames inspected.
        max_source_bytes: Per-object encoded byte limit.
        max_total_bytes: Total encoded source limit.
        max_header_bytes: Maximum header position during metadata parsing.
        max_frame_pixels: Per-frame allocation ceiling before native decoding.
        max_selected: Hard VLM frame count ceiling.
        max_gap_mm: Maximum distance to a selected frame in the same acquisition.
        min_signal_range: Minimum 8-bit preview range, not clinical quality.
        min_diversity_distance: Minimum mean 8-bit preview difference for extra slots.
        require_acquisition: Require an explicit numeric acquisition discriminator.
            If false, each frame without one forms its own coverage group.
    """

    max_instances: int = 128
    max_frames: int = 512
    max_source_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_header_bytes: int = 1024 * 1024
    max_frame_pixels: int = 4_000_000
    max_selected: int = 16
    max_gap_mm: float = 10
    min_signal_range: int = 4
    min_diversity_distance: float = 8
    require_acquisition: bool = True

    def __post_init__(self) -> None:
        for value, ceiling in (
            (self.max_instances, 1024),
            (self.max_frames, 4096),
            (self.max_source_bytes, 256 * 1024 * 1024),
            (self.max_total_bytes, 1024 * 1024 * 1024),
            (self.max_header_bytes, 8 * 1024 * 1024),
            (self.max_frame_pixels, 16_000_000),
            (self.max_selected, 128),
        ):
            _integer(value, 1, ceiling)
        _require(self.max_header_bytes >= 132)
        _require(self.max_selected <= self.max_frames)
        _finite(self.max_gap_mm, 0, 100_000)
        _integer(self.min_signal_range, 0, 255)
        _finite(self.min_diversity_distance, 1e-6, 255)
        _require(type(self.require_acquisition) is bool)


@dataclass(frozen=True, slots=True, kw_only=True)
class DisplayWindow:
    """Explicit physical-value window shared by previews and selected VLM images.

    Args:
        center: Center after the declared rescale slope/intercept.
        width: Positive width of the linear clipped display interval.

    This is a caller-qualified display transform, not automatic lesion-aware
    windowing or a full DICOM VOI presentation-state implementation.
    """

    center: float
    width: float

    def __post_init__(self) -> None:
        _finite(self.center, -1e9, 1e9)
        _finite(self.width, 1e-6, 1e9)


@dataclass(frozen=True, slots=True)
class _Frame:
    source_digest: str
    index: int
    reference: str
    group: str
    unknown_acquisition: bool
    position: tuple[float, ...] = field(repr=False)
    projection: float = field(repr=False)
    orientation: tuple[float, ...] = field(repr=False)
    spacing: tuple[float, ...] = field(repr=False)
    rows: int
    columns: int
    slope: float = field(repr=False)
    intercept: float = field(repr=False)
    monochrome_one: bool
    signal_range: int
    preview: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class SourceFrame:
    """Resolve an opaque frame reference against protected caller-owned sources.

    Args:
        reference: Stable digest derived from source bytes and frame index.
        source_digest: SHA-256 of the complete encoded source object.
        frame_index: Zero-based frame position within that object.
        group: Opaque acquisition/geometry coverage group.
    """

    reference: str
    source_digest: str
    frame_index: int
    group: str

    def __post_init__(self) -> None:
        for value in (self.reference, self.source_digest, self.group):
            _digest(value)
        _integer(self.frame_index, 0, 4095)
        _require(self.reference == _hash([self.source_digest, self.frame_index]))


@dataclass(frozen=True, slots=True)
class SeriesSamplingPlan:
    """Source-bound selection manifest; serialize only through ``to_json``.

    Args:
        source_digest: Digest of every complete supplied source, in sorted order.
        policy_digest: Digest of the full policy and display window.
        frames_digest: Digest of inspected geometry and computed preview signals.
        frames: Source digest, frame index and coverage group for every reference.
        selected: Deterministically ordered selected frame references.
        unexamined: All other supplied frame references; never silently omitted.
        low_signal: Frames rejected by the configured preview signal threshold.
        unknown_acquisition: Frames lacking an explicit acquisition discriminator.
        groups: Acquisition-group references and observed frame counts.
        maximum_gap_mm: Maximum achieved within-group coverage distance.

    A manifest is reproducible evidence, not authentication or clinical coverage.
    """

    source_digest: str
    policy_digest: str
    frames_digest: str
    frames: tuple[SourceFrame, ...]
    selected: tuple[str, ...]
    unexamined: tuple[str, ...]
    low_signal: tuple[str, ...]
    unknown_acquisition: tuple[str, ...]
    groups: tuple[tuple[str, int], ...]
    maximum_gap_mm: float

    def __post_init__(self) -> None:
        for value in (self.source_digest, self.policy_digest, self.frames_digest):
            _digest(value)
        for values in (
            self.selected,
            self.unexamined,
            self.low_signal,
            self.unknown_acquisition,
        ):
            _require(type(values) is tuple and len(values) <= 4096)
            for value in values:
                _digest(value)
            _require(tuple(sorted(set(values))) == values)
        _require(0 < len(self.selected) <= 128)
        _require(not set(self.selected) & set(self.unexamined))
        _require(type(self.frames) is tuple and len(self.frames) <= 4096)
        counts: dict[str, int] = {}
        for frame in self.frames:
            _require(type(frame) is SourceFrame)
            frame.__post_init__()
            counts[frame.group] = counts.get(frame.group, 0) + 1
        _require(
            tuple(f.reference for f in self.frames)
            == tuple(sorted((*self.selected, *self.unexamined)))
        )
        _require(tuple(sorted(counts.items())) == self.groups)
        _require(set(self.low_signal) <= set(self.unexamined))
        _require(set(self.unknown_acquisition) <= set(self.selected))
        _require(type(self.groups) is tuple and 0 < len(self.groups) <= 128)
        for group, count in self.groups:
            _digest(group)
            _integer(count, 1, 4096)
        _require(tuple(sorted(self.groups)) == self.groups)
        _require(len({group for group, _ in self.groups}) == len(self.groups))
        _require(
            sum(n for _, n in self.groups) == len(self.selected) + len(self.unexamined)
        )
        _finite(self.maximum_gap_mm, 0, 1e6)

    @property
    def digest(self) -> str:
        """Bind all selected and explicitly unexamined source frames."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        """Return only digests, frame references, counts and relative coverage."""
        return json.dumps(
            {"schema_version": _SCHEMA, **asdict(self), "notice": _NOTICE},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class _QuietParser(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.thread = threading.get_ident()
        self.warned = False

    def filter(self, record: logging.LogRecord) -> bool:
        if record.thread != self.thread:
            return True
        self.warned |= record.levelno >= logging.WARNING
        return False


@contextmanager
def _quiet_parser():
    # pydicom's config and warning helper both use this logger. Suppress this
    # thread's diagnostics before they can reach any application log handler.
    with _PARSER_LOCK:
        logger = logging.getLogger("pydicom")
        guard = _QuietParser()
        logger.addFilter(guard)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                yield
                _require(not caught and not guard.warned)
        finally:
            logger.removeFilter(guard)


class _Header(io.BytesIO):
    def __init__(self, data: bytes, limit: int) -> None:
        super().__init__(data)
        self.limit = limit

    def read(self, size: int | None = -1) -> bytes:
        _require(type(size) is int)
        assert size is not None
        _require(size >= 0 and self.tell() + size <= self.limit)
        return super().read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        position = offset + (self.tell() if whence == 1 else 0)
        _require(whence in (0, 1) and 0 <= position <= self.limit)
        return super().seek(offset, whence)


def _vector(value: Any, count: int, *, positive: bool = False) -> tuple[float, ...]:
    _require(value is not None and len(value) == count)
    result = tuple(float(v) for v in value)
    for number in result:
        _finite(number, 1e-9 if positive else -1e6, 1e6)
    return result


def _group_value(frame: Any, shared: Any, root: Any, sequence: str, key: str) -> Any:
    for owner in (frame, shared):
        items = getattr(owner, sequence, None)
        if items is not None:
            _require(len(items) == 1)
            value = getattr(items[0], key, None)
            if value is not None:
                return value
    return getattr(root, key, None)


def _acquisition(frame: Any, root: Any, required: bool) -> tuple[int | None, ...]:
    contents = getattr(frame, "FrameContentSequence", None)
    if contents is not None:
        _require(len(contents) == 1)
        item = contents[0]
    else:
        item = frame
    values = (
        getattr(
            item, "FrameAcquisitionNumber", getattr(root, "AcquisitionNumber", None)
        ),
        getattr(
            item,
            "TemporalPositionIndex",
            getattr(root, "TemporalPositionIdentifier", None),
        ),
        getattr(root, "EchoNumbers", None),
    )
    result = tuple(
        None if v is None else _header_integer(v, 0, (1 << 31) - 1) for v in values
    )
    _require(not required or any(v is not None for v in result))
    return result


def _display(data: bytes, frame: _Frame, window: DisplayWindow) -> Any:
    import numpy as np
    from pydicom.pixels import pixel_array

    array = pixel_array(io.BytesIO(data), index=frame.index, raw=True)
    _require(array.shape == (frame.rows, frame.columns))
    _require(array.dtype.kind in ("i", "u") and array.dtype.itemsize <= 2)
    physical = array.astype(np.float64) * frame.slope + frame.intercept
    display = np.floor(
        np.clip((physical - (window.center - window.width / 2)) / window.width, 0, 1)
        * 255
    ).astype(np.uint8)
    if frame.monochrome_one:
        display = 255 - display
    return display


def _inspect(
    sources: tuple[bytes, ...], window: DisplayWindow, policy: SamplingPolicy
) -> tuple[dict[str, bytes], tuple[_Frame, ...]]:
    import numpy as np
    from pydicom import dcmread
    from pydicom.dataset import Dataset

    _require(type(sources) is tuple and 0 < len(sources) <= policy.max_instances)
    _require(
        all(
            type(data) is bytes and 0 < len(data) <= policy.max_source_bytes
            for data in sources
        )
    )
    _require(sum(map(len, sources)) <= policy.max_total_bytes)
    by_digest = {hashlib.sha256(data).hexdigest(): data for data in sources}
    _require(len(by_digest) == len(sources))
    result: list[_Frame] = []
    series: set[str] = set()
    instances: set[str] = set()
    with _quiet_parser():
        for source_digest, data in sorted(by_digest.items()):
            preflight = read_dicom_transfer_syntax(
                data, max_bytes=policy.max_header_bytes
            )
            _require(preflight.ok)
            ds = dcmread(
                _Header(data, policy.max_header_bytes),
                stop_before_pixels=True,
                specific_tags=_TAGS,
            )
            _require(
                str(ds.file_meta.TransferSyntaxUID) == preflight.transfer_syntax_uid
            )
            uid, series_uid, coordinate_uid = (
                str(getattr(ds, key, ""))
                for key in (
                    "SOPInstanceUID",
                    "SeriesInstanceUID",
                    "FrameOfReferenceUID",
                )
            )
            for value in (uid, series_uid, coordinate_uid):
                _uid(value)
            _require(uid not in instances)
            instances.add(uid)
            series.add(series_uid)
            _require(len(series) == 1)
            rows = _header_integer(ds.Rows, 1, 65535)
            columns = _header_integer(ds.Columns, 1, 65535)
            _require(0 < rows * columns <= policy.max_frame_pixels)
            count = _header_integer(
                getattr(ds, "NumberOfFrames", 1), 1, policy.max_frames
            )
            _require(len(result) + count <= policy.max_frames)
            bits = _header_integer(ds.BitsAllocated, 8, 16)
            _require(_header_integer(ds.SamplesPerPixel, 1, 1) == 1 and bits in (8, 16))
            stored = _header_integer(ds.BitsStored, 1, bits)
            _require(
                _header_integer(ds.HighBit, 0, bits - 1) == stored - 1
                and _header_integer(ds.PixelRepresentation, 0, 1) in (0, 1)
            )
            photometric = str(ds.PhotometricInterpretation)
            _require(photometric in ("MONOCHROME1", "MONOCHROME2"))
            shared_items = getattr(ds, "SharedFunctionalGroupsSequence", ())
            _require(len(shared_items) <= 1)
            shared = shared_items[0] if shared_items else Dataset()
            per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", ())
            _require(len(per_frame) in (0, count))
            # Repeated top-level positions do not describe a multi-frame stack.
            _require(count == 1 or len(per_frame) == count)
            for index in range(count):
                fg = per_frame[index] if per_frame else Dataset()
                position = _vector(
                    _group_value(
                        fg,
                        shared,
                        ds if count == 1 else Dataset(),
                        "PlanePositionSequence",
                        "ImagePositionPatient",
                    ),
                    3,
                )
                orientation = _vector(
                    _group_value(
                        fg,
                        shared,
                        ds,
                        "PlaneOrientationSequence",
                        "ImageOrientationPatient",
                    ),
                    6,
                )
                spacing = _vector(
                    _group_value(
                        fg, shared, ds, "PixelMeasuresSequence", "PixelSpacing"
                    ),
                    2,
                    positive=True,
                )
                row, col = np.array(orientation[:3]), np.array(orientation[3:])
                _require(
                    abs(float(row @ row) - 1) <= 1e-6
                    and abs(float(col @ col) - 1) <= 1e-6
                    and abs(float(row @ col)) <= 1e-6
                )
                normal = np.cross(row, col)
                slope = _group_value(
                    fg, shared, ds, "PixelValueTransformationSequence", "RescaleSlope"
                )
                intercept = _group_value(
                    fg,
                    shared,
                    ds,
                    "PixelValueTransformationSequence",
                    "RescaleIntercept",
                )
                slope, intercept = (
                    float(1 if slope is None else slope),
                    float(0 if intercept is None else intercept),
                )
                _finite(slope, -1e6, 1e6)
                _require(slope != 0)
                _finite(intercept, -1e9, 1e9)
                acquisition = _acquisition(fg, ds, policy.require_acquisition)
                unknown_acquisition = all(v is None for v in acquisition)
                reference = _hash([source_digest, index])
                group = _hash(
                    [
                        coordinate_uid,
                        str(getattr(ds, "Modality", "")),
                        acquisition,
                        reference if unknown_acquisition else None,
                        rows,
                        columns,
                        orientation,
                        spacing,
                    ]
                )
                frame = _Frame(
                    source_digest,
                    index,
                    reference,
                    group,
                    unknown_acquisition,
                    position,
                    float(np.array(position) @ normal),
                    orientation,
                    spacing,
                    rows,
                    columns,
                    slope,
                    intercept,
                    photometric == "MONOCHROME1",
                    0,
                    b"",
                )
                display = _display(data, frame, window)
                ys = np.linspace(0, rows - 1, 16, dtype=np.int64)
                xs = np.linspace(0, columns - 1, 16, dtype=np.int64)
                preview = display[np.ix_(ys, xs)].tobytes()
                signal = int(display.max()) - int(display.min())
                result.append(replace(frame, signal_range=signal, preview=preview))
    return by_digest, tuple(sorted(result, key=lambda frame: frame.reference))


def _distance(left: _Frame, right: _Frame) -> float:
    return math.dist(left.position, right.position)


def _appearance(left: _Frame, right: _Frame) -> float:
    return (
        sum(abs(a - b) for a, b in zip(left.preview, right.preview, strict=True)) / 256
    )


def _select(
    frames: tuple[_Frame, ...], policy: SamplingPolicy
) -> tuple[tuple[str, ...], float]:
    groups: dict[str, list[_Frame]] = {}
    for frame in frames:
        groups.setdefault(frame.group, []).append(frame)
    _require(len(groups) <= policy.max_selected)
    chosen: dict[str, _Frame] = {}
    eligible = [
        frame for frame in frames if frame.signal_range >= policy.min_signal_range
    ]
    for group, members in sorted(groups.items()):
        candidates = [f for f in eligible if f.group == group]
        _require(bool(candidates))
        first = min(candidates, key=lambda f: (-f.signal_range, f.reference))
        local = [first]
        chosen[first.reference] = first
        _require(len(chosen) <= policy.max_selected)
        while True:
            target = max(
                members,
                key=lambda f: (min(_distance(f, s) for s in local), f.reference),
            )
            gap = min(_distance(target, s) for s in local)
            if gap <= policy.max_gap_mm:
                break
            available = [f for f in candidates if f.reference not in chosen]
            _require(bool(available) and len(chosen) < policy.max_selected)
            candidate = min(
                available,
                key=lambda f: (_distance(f, target), -f.signal_range, f.reference),
            )
            _require(_distance(candidate, target) < gap)
            local.append(candidate)
            chosen[candidate.reference] = candidate
    # Coverage always wins. Remaining slots add actual preview diversity only.
    while len(chosen) < policy.max_selected:
        available = [f for f in eligible if f.reference not in chosen]
        if not available:
            break

        def novelty(frame: _Frame) -> float:
            return min(
                _appearance(frame, s) for s in chosen.values() if s.group == frame.group
            )

        candidate = min(
            available, key=lambda f: (-novelty(f), -f.signal_range, f.reference)
        )
        if novelty(candidate) < policy.min_diversity_distance:
            break
        chosen[candidate.reference] = candidate
    maximum = max(
        min(_distance(f, s) for s in chosen.values() if s.group == f.group)
        for f in frames
    )
    _require(maximum <= policy.max_gap_mm)
    return tuple(sorted(chosen)), maximum


def _plan(
    sources: tuple[bytes, ...], window: DisplayWindow, policy: SamplingPolicy
) -> tuple[SeriesSamplingPlan, dict[str, bytes], tuple[_Frame, ...]]:
    _require(type(window) is DisplayWindow and type(policy) is SamplingPolicy)
    window.__post_init__()
    policy.__post_init__()
    source_map, frames = _inspect(sources, window, policy)
    selected, maximum_gap = _select(frames, policy)
    group_counts: dict[str, int] = {}
    for frame in frames:
        group_counts[frame.group] = group_counts.get(frame.group, 0) + 1
    records = []
    for frame in frames:
        record = asdict(frame)
        record["preview"] = hashlib.sha256(frame.preview).hexdigest()
        records.append(record)
    plan = SeriesSamplingPlan(
        _hash(sorted(source_map)),
        _hash([asdict(policy), asdict(window)]),
        _hash(records),
        tuple(
            SourceFrame(f.reference, f.source_digest, f.index, f.group) for f in frames
        ),
        selected,
        tuple(f.reference for f in frames if f.reference not in selected),
        tuple(f.reference for f in frames if f.signal_range < policy.min_signal_range),
        tuple(f.reference for f in frames if f.unknown_acquisition),
        tuple(sorted(group_counts.items())),
        maximum_gap,
    )
    return plan, source_map, frames


def sample_dicom_series(
    sources: tuple[bytes, ...],
    *,
    window: DisplayWindow,
    policy: SamplingPolicy | None = None,
) -> SeriesSamplingPlan:
    """Inspect actual native DICOM frames and produce a deterministic selection.

    Args:
        sources: Complete, bounded in-memory objects from one supplied series.
        window: Explicit display transform for all preview/selected frame pixels.
        policy: Resource, coverage and signal bounds; defaults are conservative.

    Returns:
        Manifest explicitly distinguishing selected and unexamined supplied frames.

    Raises:
        SeriesSamplingError: Missing/ambiguous metadata, unsupported native layout,
            resource excess, insufficient eligible coverage or a parser failure.

    Supports native 8/16-bit monochrome classic and enhanced multi-frame objects.
    Compression, color, missing positions and ambiguous acquisition information
    fail closed. All frames receive bounded preview inspection; only selected
    frames are sent for VLM inference. This does not establish series completeness
    beyond the supplied objects or clinically sufficient sampling.
    """
    active = policy if policy is not None else SamplingPolicy()
    return _protected(lambda: _plan(sources, window, active)[0])


@dataclass(frozen=True, slots=True)
class SampledFrameInference:
    """One original source-frame reference and its protected provider response."""

    frame_reference: str
    result: VLMResult = field(repr=False)

    def __post_init__(self) -> None:
        _digest(self.frame_reference)
        _require(type(self.result) is VLMResult)


@dataclass(frozen=True, slots=True)
class SampledSeriesInference:
    """Actual provider results whose evidence scope cannot include unseen frames."""

    plan: SeriesSamplingPlan
    frames: tuple[SampledFrameInference, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.plan) is SeriesSamplingPlan and type(self.frames) is tuple)
        self.plan.__post_init__()
        _require(all(type(f) is SampledFrameInference for f in self.frames))
        _require(tuple(f.frame_reference for f in self.frames) == self.plan.selected)

    @property
    def notice(self) -> str:
        """Mandatory sampled-only, non-diagnostic scope for downstream displays."""
        return _NOTICE

    def require_evidence(self, frame_reference: str) -> VLMResult:
        """Return evidence only for a successfully inferred selected frame.

        Args:
            frame_reference: Exact source-bound frame reference to cite.
        """
        for frame in self.frames:
            if (
                frame.frame_reference == frame_reference
                and frame.result.abstention is None
            ):
                return frame.result
        raise SeriesSamplingError("dicom_frame_not_examined")

    def to_json(self) -> str:
        """Report sampling scope without generated text, pixel data or DICOM tags."""
        return json.dumps(
            {
                "schema_version": _SCHEMA,
                "manifest_digest": self.plan.digest,
                "evidence_scope": "selected_frames_only",
                "examined": [
                    f.frame_reference
                    for f in self.frames
                    if f.result.abstention is None
                ],
                "unexamined": sorted(
                    (
                        *self.plan.unexamined,
                        *(
                            f.frame_reference
                            for f in self.frames
                            if f.result.abstention is not None
                        ),
                    )
                ),
                "notice": self.notice,
                "review_required": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def run_sampled_dicom_vlm(
    sources: tuple[bytes, ...],
    plan: SeriesSamplingPlan,
    provider: EdgeVLMProvider,
    prompt: Prompt,
    *,
    window: DisplayWindow,
    policy: SamplingPolicy | None = None,
) -> SampledSeriesInference:
    """Revalidate the entire sampling manifest and infer only its selected frames.

    Args:
        sources: The exact supplied DICOM objects, in any order.
        plan: Previously returned source/policy-bound sampling manifest.
        provider: Already loaded and qualified local VLM adapter.
        prompt: Protected frame-scoped prompt and original generation limits.
        window: Same explicit display window as the sampling operation.
        policy: Same policy as the sampling operation.

    Returns:
        Protected per-frame drafts with a mandatory sampled-only scope. Provider
        abstentions remain unexamined for evidence access and reporting.

    Raises:
        SeriesSamplingError: Any binding, decode or provider contract failure.
            Partial result bundles are never returned on execution failure.
    """

    def run() -> SampledSeriesInference:
        from PIL import Image

        _require(type(plan) is SeriesSamplingPlan and type(prompt) is Prompt)
        plan.__post_init__()
        prompt.__post_init__()
        active = policy if policy is not None else SamplingPolicy()
        expected, source_map, frames = _plan(sources, window, active)
        _require(expected == plan)
        selected = {f.reference: f for f in frames}
        results = []
        for reference in plan.selected:
            frame = selected[reference]
            with _quiet_parser():
                pixels = _display(source_map[frame.source_digest], frame, window)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            image = ImageInput(
                data=buffer.getvalue(),
                media_type="image/png",
                width=frame.columns,
                height=frame.rows,
            )
            request = VLMRequest(image=image, prompt=prompt)
            result = run_local_vlm(provider, request)
            results.append(SampledFrameInference(reference, result))
        return SampledSeriesInference(plan, tuple(results))

    return _protected(run)
