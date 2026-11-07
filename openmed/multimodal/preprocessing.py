"""Bounded, executable VLM preprocessing conformance on decoded synthetic arrays.

This is an operator contract, not model or diagnostic-image qualification.
No decoder, model selection, network transport, or clinical decision is included.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import NoReturn, Protocol

from .orientation_preflight import ImageTransform

NOTICE = "synthetic_preprocessing_only_non_diagnostic"
MAX_PIXELS = 4096


class PreprocessingError(ValueError):
    """A controlled, value-free preprocessing failure."""


def _fail(code: str) -> NoReturn:
    raise PreprocessingError(code)


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and type(value) in (int, float)
        and abs(value) <= 1e300
        and math.isfinite(value)
    )


def _digest(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class PreprocessingProfile:
    """Version-one operator plan, in explicit execution order.

    Args:
        source_size: Decoded width and height; RGB/RGBA byte samples in sRGB.
        crop: Half-open integer rectangle in the oriented image.
        target_size: Output width and height, with half-pixel sampling.
        orientation: EXIF orientation, applied before crop.
        interpolation: ``nearest`` or ``bilinear``; no antialiasing or gamma step.
        color_order: Output RGB or BGR channels.
        background: RGB bytes for straight-alpha composition before orientation.
        scale: Applied after resize, before channel-wise mean and std.
        mean: Per-output-channel subtraction constants.
        std: Per-output-channel positive division constants.
        layout: HWC or CHW flat tensor order.
        dtype: Final little-endian float32 or float64 rounding only.
        atol: Maximum absolute tensor difference; geometry/masks have no tolerance.
        version: Exact supported executable schema version.
    """

    source_size: tuple[int, int]
    crop: tuple[int, int, int, int]
    target_size: tuple[int, int]
    orientation: int = 1
    interpolation: str = "nearest"
    color_order: str = "RGB"
    background: tuple[int, int, int] = (0, 0, 0)
    scale: float = 1 / 255
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    layout: str = "HWC"
    dtype: str = "float32"
    atol: float = 1e-6
    version: int = 1

    def __post_init__(self) -> None:
        for size in (self.source_size, self.target_size):
            if (
                type(size) is not tuple
                or len(size) != 2
                or not all(_integer(v, 1, 64) for v in size)
                or math.prod(size) > MAX_PIXELS
            ):
                _fail("dimensions_invalid")
        if not _integer(self.version, 1, 1):
            _fail("version_unsupported")
        if not _integer(self.orientation, 1, 8):
            _fail("orientation_unsupported")
        if (
            any(
                type(v) is not str
                for v in (self.interpolation, self.color_order, self.layout, self.dtype)
            )
            or self.interpolation not in ("nearest", "bilinear")
            or self.color_order not in ("RGB", "BGR")
            or self.layout not in ("HWC", "CHW")
            or self.dtype not in ("float32", "float64")
        ):
            _fail("operator_unsupported")
        w, h = self.oriented_size
        if (
            type(self.crop) is not tuple
            or len(self.crop) != 4
            or not all(_integer(v, 0, 64) for v in self.crop)
        ):
            _fail("crop_invalid")
        x0, y0, x1, y1 = self.crop
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            _fail("crop_invalid")
        if (
            type(self.background) is not tuple
            or len(self.background) != 3
            or not all(_integer(v, 0, 255) for v in self.background)
        ):
            _fail("background_invalid")
        for vector in (self.mean, self.std):
            if (
                type(vector) is not tuple
                or len(vector) != 3
                or not all(_finite(v) and abs(v) <= 1e6 for v in vector)
            ):
                _fail("normalization_invalid")
        if (
            not _finite(self.scale)
            or not 0 < self.scale <= 1e6
            or any(v < 1e-6 for v in self.std)
            or not _finite(self.atol)
            or not 0 <= self.atol <= 1e-5
        ):
            _fail("numeric_behavior_unsupported")

    @property
    def oriented_size(self) -> tuple[int, int]:
        """Return the exact oriented width and height."""
        w, h = self.source_size
        return (h, w) if self.orientation in (5, 6, 7, 8) else (w, h)

    @property
    def digest(self) -> str:
        """Hash the versioned binary operator identity, shared with OpenMedKit."""
        integers = (
            self.version,
            *self.source_size,
            *self.crop,
            *self.target_size,
            self.orientation,
            *self.background,
        )
        data = b"openmed.preprocessing.v1\0" + struct.pack("<13I", *integers)
        data += (
            "\0".join(
                (self.interpolation, self.color_order, self.layout, self.dtype)
            ).encode("ascii")
            + b"\0"
        )
        data += struct.pack("<8d", self.scale, *self.mean, *self.std, self.atol)
        return _digest(data)


@dataclass(frozen=True)
class PreprocessingOutput:
    """Transient tensor and exact lineage; payloads are excluded from repr.

    Geometry is a six-coefficient source-edge to output-edge affine transform,
    stored as reduced rational strings. Protected masks conservatively include
    every source pixel with nonzero sampling weight, not just nearest centers.
    """

    profile_digest: str
    shape: tuple[int, int, int]
    tensor: tuple[float, ...] = field(repr=False)
    geometry: tuple[str, ...]
    protected_mask: tuple[bool, ...] = field(repr=False)
    dtype: str

    def __post_init__(self) -> None:
        if (
            not _sha(self.profile_digest)
            or type(self.shape) is not tuple
            or len(self.shape) != 3
            or not all(_integer(v, 1, 64) for v in self.shape)
            or type(self.tensor) is not tuple
            or len(self.tensor) != math.prod(self.shape)
            or len(self.tensor) > MAX_PIXELS * 3
            or any(not _finite(v) for v in self.tensor)
            or self.dtype not in ("float32", "float64")
            or type(self.protected_mask) is not tuple
            or len(self.protected_mask) * 3 != len(self.tensor)
            or any(type(v) is not bool for v in self.protected_mask)
            or type(self.geometry) is not tuple
            or len(self.geometry) != 6
        ):
            _fail("output_invalid")
        for v in self.geometry:
            if (
                type(v) is not str
                or not 1 <= len(v) <= 32
                or any(c not in "0123456789-/" for c in v)
            ):
                _fail("coordinate_invalid")
            try:
                canonical = str(Fraction(v))
            except (ValueError, ZeroDivisionError):
                raise PreprocessingError("coordinate_invalid") from None
            if canonical != v:
                _fail("coordinate_invalid")
        if self.dtype == "float32":
            try:
                rounded = struct.unpack(
                    f"<{len(self.tensor)}f",
                    struct.pack(f"<{len(self.tensor)}f", *self.tensor),
                )
            except (OverflowError, struct.error):
                raise PreprocessingError("tensor_invalid") from None
            if rounded != self.tensor:
                _fail("numeric_behavior_unsupported")

    @property
    def tensor_digest(self) -> str:
        """Hash actual rounded tensor bytes, shape, dtype, and profile."""
        code = "f" if self.dtype == "float32" else "d"
        try:
            data = bytes.fromhex(self.profile_digest)
            data += struct.pack("<3I", *self.shape) + self.dtype.encode("ascii") + b"\0"
            data += struct.pack(f"<{len(self.tensor)}{code}", *self.tensor)
        except (ValueError, TypeError, struct.error, OverflowError):
            _fail("tensor_invalid")
        return _digest(data)

    @property
    def coordinate_digest(self) -> str:
        """Bind exact geometry and protected coverage to the profile."""
        data = (self.profile_digest + "|" + "|".join(self.geometry)).encode("ascii")
        return _digest(data + b"\0" + bytes(self.protected_mask))


def _position(x: int, y: int, profile: PreprocessingProfile) -> tuple[int, int]:
    w, h = profile.source_size
    transform = ImageTransform.from_exif_orientation(profile.orientation)
    if transform.mirrored:
        x = w - 1 - x
    if transform.rotation_degrees == 90:
        return h - 1 - y, x
    if transform.rotation_degrees == 180:
        return w - 1 - x, h - 1 - y
    if transform.rotation_degrees == 270:
        return y, w - 1 - x
    return x, y


def _geometry(profile: PreprocessingProfile) -> tuple[str, ...]:
    w, h = profile.source_size
    # Edge transforms, unlike center indexes, use full width/height offsets.
    matrices = {
        1: (1, 0, 0, 0, 1, 0),
        2: (-1, 0, w, 0, 1, 0),
        3: (-1, 0, w, 0, -1, h),
        4: (1, 0, 0, 0, -1, h),
        5: (0, 1, 0, 1, 0, 0),
        6: (0, -1, h, 1, 0, 0),
        7: (0, -1, h, -1, 0, w),
        8: (0, 1, 0, -1, 0, w),
    }
    a, b, c, d, e, f = matrices[profile.orientation]
    x0, y0, x1, y1 = profile.crop
    tw, th = profile.target_size
    sx, sy = Fraction(tw, x1 - x0), Fraction(th, y1 - y0)
    return tuple(
        str(v) for v in (a * sx, b * sx, (c - x0) * sx, d * sy, e * sy, (f - y0) * sy)
    )


def _samples(
    index: int, source: int, target: int, mode: str
) -> tuple[tuple[int, float], ...]:
    point = max(
        Fraction(0),
        min(
            Fraction(source - 1),
            Fraction((2 * index + 1) * source, 2 * target) - Fraction(1, 2),
        ),
    )
    if mode == "nearest":
        return ((math.floor(point + Fraction(1, 2)), 1.0),)
    low = math.floor(point)
    weight = point - low
    return tuple(
        (i, float(v))
        for i, v in ((low, 1 - weight), (min(low + 1, source - 1), weight))
        if v
    )


def execute_profile(
    profile: PreprocessingProfile,
    pixels: tuple[tuple[int, ...], ...],
    protected_mask: tuple[bool, ...],
) -> PreprocessingOutput:
    """Execute on bounded row-major RGB/RGBA bytes without I/O.

    Args:
        profile: Validated executable operator plan.
        pixels: Flattened source pixels, all RGB or all straight RGBA.
        protected_mask: Flattened source mask with exact boolean entries.

    Returns:
        Transient tensor and profile-bound coordinate evidence.

    Raises:
        PreprocessingError: Controlled code for unsupported input behavior.
    """
    count = math.prod(profile.source_size)
    if (
        type(pixels) is not tuple
        or len(pixels) != count
        or type(protected_mask) is not tuple
        or len(protected_mask) != count
        or any(type(v) is not bool for v in protected_mask)
    ):
        _fail("source_invalid")
    channels = len(pixels[0]) if type(pixels[0]) is tuple else 0
    if channels not in (3, 4) or any(
        type(p) is not tuple
        or len(p) != channels
        or not all(_integer(v, 0, 255) for v in p)
        for p in pixels
    ):
        _fail("pixels_invalid")
    w, _ = profile.source_size
    ow, oh = profile.oriented_size
    oriented: list[tuple[float, ...]] = [(0.0, 0.0, 0.0)] * (ow * oh)
    masks = [False] * (ow * oh)
    for i, pixel in enumerate(pixels):
        x, y = _position(i % w, i // w, profile)
        alpha = pixel[3] / 255 if channels == 4 else 1.0
        oriented[y * ow + x] = tuple(
            pixel[c] * alpha + profile.background[c] * (1 - alpha) for c in range(3)
        )
        masks[y * ow + x] = protected_mask[i]
    x0, y0, x1, y1 = profile.crop
    tw, th = profile.target_size
    tensor: list[float] = []
    coverage: list[bool] = []
    order = (0, 1, 2) if profile.color_order == "RGB" else (2, 1, 0)
    for y in range(th):
        for x in range(tw):
            values = [0.0] * 3
            protected = False
            for iy, wy in _samples(y, y1 - y0, th, profile.interpolation):
                for ix, wx in _samples(x, x1 - x0, tw, profile.interpolation):
                    index = (iy + y0) * ow + ix + x0
                    for c in range(3):
                        values[c] += oriented[index][c] * wx * wy
                    protected |= masks[index]
            tensor.extend(
                (values[c] * profile.scale - profile.mean[k]) / profile.std[k]
                for k, c in enumerate(order)
            )
            coverage.append(protected)
    if profile.layout == "CHW":
        tensor = [tensor[i * 3 + c] for c in range(3) for i in range(tw * th)]
    code = "f" if profile.dtype == "float32" else "d"
    rounded = struct.unpack(
        f"<{len(tensor)}{code}", struct.pack(f"<{len(tensor)}{code}", *tensor)
    )
    shape = (th, tw, 3) if profile.layout == "HWC" else (3, th, tw)
    return PreprocessingOutput(
        profile.digest,
        shape,
        rounded,
        _geometry(profile),
        tuple(coverage),
        profile.dtype,
    )


class PreprocessingAdapter(Protocol):
    """Optional local runtime seam; implementations must never persist inputs."""

    def __call__(
        self,
        profile: PreprocessingProfile,
        pixels: tuple[tuple[int, ...], ...],
        protected_mask: tuple[bool, ...],
    ) -> PreprocessingOutput:
        """Return actual tensor, geometry and mask from the runtime under test."""
        ...


def _compare(
    profile: PreprocessingProfile,
    actual: PreprocessingOutput,
    expected: PreprocessingOutput,
) -> None:
    if type(actual) is not PreprocessingOutput:
        _fail("output_invalid")
    if actual.profile_digest != profile.digest or actual.dtype != profile.dtype:
        _fail("profile_mismatch")
    if (
        actual.shape != expected.shape
        or actual.geometry != expected.geometry
        or actual.protected_mask != expected.protected_mask
    ):
        _fail("coordinate_mismatch")
    if len(actual.tensor) != len(expected.tensor) or any(
        not _finite(a) or abs(a - b) > profile.atol
        for a, b in zip(actual.tensor, expected.tensor)
    ):
        _fail("tensor_mismatch")


@dataclass(frozen=True)
class PreprocessingQualification:
    """Synthetic conformance binding, not a provider or clinical attestation."""

    profile_digest: str
    runtime_digest: str
    vectors_digest: str
    notice: str = field(default=NOTICE, init=False)
    reviewer_confirmation_required: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not all(
            _sha(v)
            for v in (self.profile_digest, self.runtime_digest, self.vectors_digest)
        ):
            _fail("qualification_invalid")

    @property
    def digest(self) -> str:
        """Return the profile/runtime/vector qualification identity."""
        return _digest(
            (self.profile_digest + self.runtime_digest + self.vectors_digest).encode(
                "ascii"
            )
        )


def qualify_profile(
    profile: PreprocessingProfile, adapter: PreprocessingAdapter, *, runtime_digest: str
) -> PreprocessingQualification:
    """Compare a local adapter against two deterministic synthetic operator grids.

    The runtime digest must identify caller-controlled code/build configuration.
    Re-run after changing the runtime. Exceptions are sanitized at this boundary.
    Geometry/masks compare exactly; only tensors use the profile's absolute bound.
    """
    if not _sha(runtime_digest):
        _fail("runtime_digest_invalid")
    count = math.prod(profile.source_size)
    identity = bytearray(b"openmed.preprocessing.vectors.v1\0")
    for channels in (3, 4):
        pixels = tuple(
            tuple((i * 37 + c * 71) % 256 for c in range(3))
            + (((0, 128, 255)[i % 3],) if channels == 4 else ())
            for i in range(count)
        )
        mask = tuple(i % 3 == 0 for i in range(count))
        expected = execute_profile(profile, pixels, mask)
        try:
            actual = adapter(profile, pixels, mask)
        except Exception:
            raise PreprocessingError("adapter_failed") from None
        _compare(profile, actual, expected)
        identity.extend(bytes(v for p in pixels for v in p))
        identity.extend(bytes(mask))
        identity.extend(bytes.fromhex(actual.tensor_digest))
        identity.extend(bytes.fromhex(actual.coordinate_digest))
    return PreprocessingQualification(profile.digest, runtime_digest, _digest(identity))


@dataclass(frozen=True)
class PreprocessingEvidence:
    """Content-free downstream binding with mandatory non-diagnostic review flags."""

    qualification_digest: str
    tensor_digest: str
    coordinate_digest: str
    notice: str = field(default=NOTICE, init=False)
    reviewer_confirmation_required: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not all(
            _sha(v)
            for v in (
                self.qualification_digest,
                self.tensor_digest,
                self.coordinate_digest,
            )
        ):
            _fail("evidence_invalid")

    def require_current(
        self,
        profile: PreprocessingProfile,
        output: PreprocessingOutput,
        qualification: PreprocessingQualification,
        *,
        runtime_digest: str,
    ) -> None:
        """Reject stale profile, runtime, tensor or coordinate evidence."""
        _binding(profile, qualification, runtime_digest)
        if (
            self.qualification_digest != qualification.digest
            or self.tensor_digest != output.tensor_digest
            or self.coordinate_digest != output.coordinate_digest
            or output.profile_digest != profile.digest
        ):
            _fail("evidence_stale")


def _binding(
    profile: PreprocessingProfile,
    qualification: PreprocessingQualification,
    runtime_digest: str,
) -> None:
    if (
        not _sha(runtime_digest)
        or qualification.profile_digest != profile.digest
        or qualification.runtime_digest != runtime_digest
    ):
        _fail("qualification_stale")


def verify_preprocessing(
    profile: PreprocessingProfile,
    pixels: tuple[tuple[int, ...], ...],
    protected_mask: tuple[bool, ...],
    output: PreprocessingOutput,
    qualification: PreprocessingQualification,
    *,
    runtime_digest: str,
) -> PreprocessingEvidence:
    """Verify one actual runtime output before emitting downstream evidence.

    This does not authorize a clinical action. Consumers must separately obtain
    explicit reviewer confirmation for consequential use. No payload is retained
    in the returned evidence.
    """
    _binding(profile, qualification, runtime_digest)
    _compare(profile, output, execute_profile(profile, pixels, protected_mask))
    return PreprocessingEvidence(
        qualification.digest, output.tensor_digest, output.coordinate_digest
    )
