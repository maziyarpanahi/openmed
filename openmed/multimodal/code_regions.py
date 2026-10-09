"""Local identifier-bearing code detection with a payload-free audit boundary."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Literal, Protocol

CodeDetectionPolicy = Literal["fail_closed", "coverage_incomplete"]


class CodeDetectionError(ValueError):
    """A code scan failed; messages never include backend errors or payloads."""


@dataclass(frozen=True)
class CodeRegion:
    """A code class and top-left, right/bottom-exclusive pixel bounding box.

    Args:
        symbology: Controlled class: linear, qr, datamatrix, or other.
        bbox: Pixel coordinates covering the complete symbol.
        page: Zero-based frame/page index, assigned by the caller.
        payload_sha256: Optional opaque digest, excluded from coverage reports.
    """

    symbology: str
    bbox: tuple[int, int, int, int]
    page: int = 0
    payload_sha256: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return code geometry only, without decoded content or its digest."""
        return {"symbology": self.symbology, "bbox": list(self.bbox), "page": self.page}


@dataclass(frozen=True)
class CodeCoverageReport:
    """Counts and boxes from a scan, with explicit unavailable coverage."""

    regions: tuple[CodeRegion, ...] = ()
    coverage_incomplete: bool = False

    @property
    def clean(self) -> bool:
        """Whether an available detector found no remaining symbols."""
        return not self.regions and not self.coverage_incomplete

    def to_dict(self) -> dict[str, Any]:
        """Return payload-free coverage evidence."""
        return {
            "count": len(self.regions),
            "boxes": [region.to_dict() for region in self.regions],
            "coverage_incomplete": self.coverage_incomplete,
        }


class CodeRegionDetector(Protocol):
    """Caller-supplied local detector accepting a Pillow image or uint8 array."""

    def detect(self, image: Any) -> Iterable[CodeRegion | Mapping[str, Any]]:
        """Return all located symbols, including unreadable identifier regions."""
        ...


class FakeCodeRegionDetector:
    """Return fixed geometry for deterministic tests; never decode pixels.

    Args:
        regions: Synthetic regions returned on every scan. Use an empty tuple
            for a negative control. Residual tests should inject a pixel-aware
            fake or separate verification detector.
    """

    def __init__(self, regions: Iterable[CodeRegion] = ()) -> None:
        self._regions = tuple(regions)

    def detect(self, image: Any) -> tuple[CodeRegion, ...]:
        """Return the configured synthetic regions."""
        return self._regions


class ZXingCodeRegionDetector:
    """Optional Apache-2.0 zxing-cpp backend (caller-installed, no downloads).

    ZXing may internally decode symbols. This adapter never reads decoded text,
    bytes, or decoder error messages. ``return_errors`` includes located symbols
    that failed decoding. Detection recall is not clinically qualified.
    """

    def detect(self, image: Any) -> tuple[CodeRegion, ...]:
        """Locate code regions locally, discarding decoded payloads."""
        try:
            return self._detect(image)
        except Exception:
            raise CodeDetectionError("code_detection_unavailable") from None

    def _detect(self, image: Any) -> tuple[CodeRegion, ...]:
        backend = importlib.import_module("zxingcpp")
        regions = []
        for code in backend.read_barcodes(image, return_errors=True):
            position = code.position
            points = (
                position.top_left,
                position.top_right,
                position.bottom_left,
                position.bottom_right,
            )
            name = code.format.name
            if name in {"QRCode", "MicroQRCode", "RMQRCode"}:
                symbology = "qr"
            elif name == "DataMatrix":
                symbology = "datamatrix"
            elif name in {
                "Codabar",
                "Code39",
                "Code93",
                "Code128",
                "EAN8",
                "EAN13",
                "ITF",
                "DataBar",
                "DataBarExpanded",
                "DataBarLimited",
                "UPCA",
                "UPCE",
                "DXFilmEdge",
            }:
                symbology = "linear"
            else:
                symbology = "other"
            regions.append(
                CodeRegion(
                    symbology,
                    (
                        min(point.x for point in points),
                        min(point.y for point in points),
                        max(point.x for point in points) + 1,
                        max(point.y for point in points) + 1,
                    ),
                )
            )
        return tuple(regions)


def detect_code_regions(
    image: Any,
    *,
    detector: CodeRegionDetector | None = None,
    policy: CodeDetectionPolicy = "coverage_incomplete",
    page: int = 0,
    max_regions: int = 25_000,
) -> CodeCoverageReport:
    """Scan an image and reduce untrusted backend results to safe geometry.

    Args:
        image: Pillow image or uint8 NumPy array, never a source path.
        detector: Local provider; defaults to the optional ZXing adapter.
        policy: Fail closed, or explicitly report unavailable/failed coverage.
        page: Frame/page index assigned to every returned region.
        max_regions: Maximum number of provider regions accepted.

    Returns:
        Counts/boxes only. Mapping providers may return ``payload``, ``text``
        or ``bytes``; that value is hashed and never retained as plaintext.

    Raises:
        CodeDetectionError: Failed/unavailable scan under fail-closed policy,
            invalid geometry, or exhausted limits. Messages contain fixed codes.
    """
    if policy not in {"fail_closed", "coverage_incomplete"}:
        raise CodeDetectionError("code_detection_invalid_policy")
    if (
        type(page) is not int
        or page < 0
        or type(max_regions) is not int
        or max_regions < 1
    ):
        raise CodeDetectionError("code_detection_invalid_options")
    try:
        shape = getattr(image, "shape", None)
        width, height = (shape[1], shape[0]) if shape is not None else image.size
        try:
            # Materialize a bounded iterator inside the untrusted-provider error
            # boundary: even provider-raised CodeDetectionError text is private.
            provider = detector if detector is not None else ZXingCodeRegionDetector()
            raw = list(islice(provider.detect(image), max_regions + 1))
        except Exception:
            if policy == "fail_closed":
                raise CodeDetectionError("code_detection_unavailable") from None
            return CodeCoverageReport(coverage_incomplete=True)
        regions = []
        for index, item in enumerate(raw):
            if index >= max_regions:
                raise CodeDetectionError("code_detection_region_limit")
            if isinstance(item, CodeRegion):
                symbology, bbox, digest = item.symbology, item.bbox, item.payload_sha256
            elif isinstance(item, Mapping):
                symbology, bbox = item.get("symbology"), item.get("bbox")
                payload = item.get("payload", item.get("text", item.get("bytes")))
                if payload is not None and not isinstance(payload, (str, bytes)):
                    raise CodeDetectionError("code_detection_invalid_result")
                digest = (
                    hashlib.sha256(
                        payload.encode("utf-8") if isinstance(payload, str) else payload
                    ).hexdigest()
                    if payload is not None
                    else None
                )
            else:
                raise CodeDetectionError("code_detection_invalid_result")
            if not isinstance(symbology, str) or symbology not in {
                "linear",
                "qr",
                "datamatrix",
                "other",
            }:
                symbology = "other"
            if (
                not isinstance(bbox, (tuple, list))
                or len(bbox) != 4
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in bbox
                )
            ):
                raise CodeDetectionError("code_detection_invalid_geometry")
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise CodeDetectionError("code_detection_invalid_geometry")
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise CodeDetectionError("code_detection_invalid_result")
            regions.append(
                CodeRegion(
                    symbology,
                    (
                        math.floor(x0),
                        math.floor(y0),
                        math.ceil(x1),
                        math.ceil(y1),
                    ),
                    page,
                    digest,
                )
            )
        return CodeCoverageReport(tuple(regions))
    except CodeDetectionError as exc:
        # Custom mapping accessors are provider code too. Do not echo their
        # exception text, even if they use our public exception type.
        if str(exc) not in {
            "code_detection_unavailable",
            "code_detection_region_limit",
            "code_detection_invalid_result",
            "code_detection_invalid_geometry",
        }:
            raise CodeDetectionError("code_detection_invalid_result") from None
        raise CodeDetectionError(str(exc)) from None
    except Exception:
        if policy == "fail_closed":
            raise CodeDetectionError("code_detection_unavailable") from None
        return CodeCoverageReport(coverage_incomplete=True)


def merge_code_coverage(reports: Iterable[CodeCoverageReport]) -> CodeCoverageReport:
    """Combine frame/page geometry without exposing provider metadata."""
    reports = tuple(reports)
    return CodeCoverageReport(
        tuple(region for report in reports for region in report.regions),
        any(report.coverage_incomplete for report in reports),
    )


def assert_no_residual_codes(report: CodeCoverageReport) -> None:
    """Reject any symbol that remains detectable after redaction."""
    if report.regions:
        raise CodeDetectionError("code_redaction_residual_regions")
