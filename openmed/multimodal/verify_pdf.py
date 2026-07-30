"""Redacted-PDF text-layer leakage and visual fidelity verification.

Drawing a black box over PHI hides it visually, but a copy-pasteable text layer
underneath frequently survives — a classic real-world redaction failure where
the selectable text still leaks the very names that look blacked out. This
module verifies a *redacted* PDF against the regions that were supposed to be
scrubbed and **fails closed** when either:

* residual selectable text remains under a redaction box (text-layer leak), or
* a redaction region shows no visible change — no opaque box was actually drawn
  (an unchanged region where the redaction silently did not apply).

The verifier is deterministic and works from the PDF's own text and vector
content (via ``pdfplumber``); it optionally corroborates the visual check by
rendering each region to pixels when a raster backend is available (or when a
``rasterizer`` callable is supplied). The structured, PHI-safe report is
suitable for embedding into a de-identification audit report — it records
offsets, bounding boxes, hashes, and counts, never plaintext identifiers.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .documents_pdf import extract_pdf, project_text_spans
from .exceptions import MissingDependencyError

_PDF_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'
_PDF_WORD_FIELDS = ("x0", "top", "x1", "bottom")

# A word counts as "under the box" when most of its area falls inside the region.
_WORD_OVERLAP_FRACTION = 0.5
# A rectangle counts as a redaction box when it opaquely covers most of the region.
_BOX_COVER_FRACTION = 0.6

Rasterizer = Callable[[str | Path, int, "tuple[float, float, float, float]"], bytes]


class RedactionFidelityError(RuntimeError):
    """Raised when a redacted PDF fails the leakage / visual-fidelity check."""

    def __init__(self, report: "PdfFidelityReport") -> None:
        self.report = report
        super().__init__(report.summary())


@dataclass(frozen=True)
class RegionFidelity:
    """The verification result for one redaction region (PHI-safe)."""

    page: int
    bbox: tuple[float, float, float, float]
    residual_text_found: bool
    residual_word_count: int
    residual_sha256: tuple[str, ...]
    redaction_box_present: bool
    pixels_changed: bool | None
    visual_method: str
    label: str | None = None

    @property
    def visual_ok(self) -> bool:
        """True when the region visibly changed (a box was drawn over it)."""
        if not self.redaction_box_present:
            return False
        # If pixels were rendered, they must confirm a change as well.
        return self.pixels_changed is not False

    @property
    def passed(self) -> bool:
        """A region passes only when text is scrubbed AND it visibly changed."""
        return (not self.residual_text_found) and self.visual_ok

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-safe representation (hashes/offsets only)."""
        payload: dict[str, Any] = {
            "page": self.page,
            "bbox": list(self.bbox),
            "passed": self.passed,
            "residual_text_found": self.residual_text_found,
            "residual_word_count": self.residual_word_count,
            "residual_sha256": list(self.residual_sha256),
            "redaction_box_present": self.redaction_box_present,
            "visual_method": self.visual_method,
        }
        if self.pixels_changed is not None:
            payload["pixels_changed"] = self.pixels_changed
        if self.label is not None:
            payload["label"] = self.label
        return payload


@dataclass(frozen=True)
class PdfFidelityReport:
    """Structured, per-region fidelity report for a redacted PDF."""

    regions: tuple[RegionFidelity, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True only when every region passed (fails closed on empty input)."""
        return bool(self.regions) and all(region.passed for region in self.regions)

    @property
    def failing_regions(self) -> tuple[RegionFidelity, ...]:
        return tuple(region for region in self.regions if not region.passed)

    @property
    def residual_text_regions(self) -> tuple[RegionFidelity, ...]:
        return tuple(region for region in self.regions if region.residual_text_found)

    def to_dict(self) -> dict[str, Any]:
        """PHI-safe structured report suitable for the audit report."""
        return {
            "check": "redacted_pdf_fidelity",
            "passed": self.passed,
            "region_count": len(self.regions),
            "failing_region_count": len(self.failing_regions),
            "residual_text_region_count": len(self.residual_text_regions),
            "regions": [region.to_dict() for region in self.regions],
            "metadata": dict(self.metadata),
        }

    def summary(self) -> str:
        if not self.regions:
            return "PDF fidelity check FAILED: no redaction regions were provided."
        if self.passed:
            return f"PDF fidelity check PASSED for all {len(self.regions)} region(s)."
        leaks = len(self.residual_text_regions)
        # Count "no visible box" only for regions that did not also leak text, so
        # the two categories are disjoint and never sum past failing_regions.
        no_box = sum(
            1
            for region in self.failing_regions
            if not region.residual_text_found and not region.visual_ok
        )
        return (
            f"PDF fidelity check FAILED: {len(self.failing_regions)} of "
            f"{len(self.regions)} region(s) failed "
            f"({leaks} with residual selectable text, {no_box} with no visible box)."
        )

    def raise_for_leakage(self) -> "PdfFidelityReport":
        """Raise :class:`RedactionFidelityError` unless every region passed."""
        if not self.passed:
            raise RedactionFidelityError(self)
        return self


def verify_redacted_pdf(
    original: str | Path,
    redacted: str | Path,
    spans: Iterable[Any],
    *,
    rasterizer: Rasterizer | None = None,
    strict: bool = False,
) -> PdfFidelityReport:
    """Verify that ``redacted`` scrubbed the PHI ``spans`` present in ``original``.

    ``spans`` describes what was redacted. Each item may be:

    * a ``(page, bbox)`` region (a mapping/object with ``page`` and ``bbox``, or
      a :class:`~openmed.multimodal.documents_pdf.ProjectedRectangle`), or
    * a character span into ``original``'s extracted text (a ``(start, end)``
      tuple, or a mapping/object with ``start``/``end``) that is projected to a
      page rectangle using ``original``.

    For each region the verifier asserts (a) no selectable word in ``redacted``
    remains under the region and (b) an opaque redaction box covers the region.
    When a raster backend is available (or ``rasterizer`` is supplied) the
    region is also rendered to pixels in both documents and required to differ.

    Returns a :class:`PdfFidelityReport`. Pass ``strict=True`` to raise
    :class:`RedactionFidelityError` on any residual leakage instead.
    """
    regions = _resolve_regions(original, spans)
    layout = _read_pdf_layout(redacted)

    results: list[RegionFidelity] = []
    for page, bbox, label in regions:
        words, rects = layout.get(page, ((), ()))
        residual = _residual_words(words, bbox)
        box_present = _box_covers(rects, bbox)
        pixels_changed, method = _visual_change(
            original, redacted, page, bbox, rasterizer=rasterizer
        )
        results.append(
            RegionFidelity(
                page=page,
                bbox=bbox,
                residual_text_found=bool(residual),
                residual_word_count=len(residual),
                residual_sha256=tuple(_sha256(text) for text in residual),
                redaction_box_present=box_present,
                pixels_changed=pixels_changed,
                visual_method=method,
                label=label,
            )
        )

    report = PdfFidelityReport(
        regions=tuple(results),
        # Do not echo raw file paths — they can themselves contain identifiers
        # (e.g. /exports/John_Doe_MRN12345.pdf). The report stays PHI-safe.
        metadata={"region_count": len(results)},
    )
    if strict:
        report.raise_for_leakage()
    return report


# ---------------------------------------------------------------------------
# Region resolution
# ---------------------------------------------------------------------------


def _resolve_regions(
    original: str | Path, spans: Iterable[Any]
) -> list[tuple[int, tuple[float, float, float, float], str | None]]:
    direct: list[tuple[int, tuple[float, float, float, float], str | None]] = []
    char_spans: list[Any] = []
    for span in spans:
        region = _as_region(span)
        if region is not None:
            direct.append(region)
        else:
            char_spans.append(span)

    if char_spans:
        document = extract_pdf(original)
        for rectangle in project_text_spans(document, char_spans):
            direct.append((rectangle.page, tuple(rectangle.bbox), rectangle.label))
    return direct


def _as_region(
    span: Any,
) -> tuple[int, tuple[float, float, float, float], str | None] | None:
    if isinstance(span, Mapping):
        page = span.get("page")
        bbox = span.get("bbox")
        label = span.get("label", span.get("entity_type"))
    elif isinstance(span, Sequence) and not isinstance(span, (str, bytes, bytearray)):
        # A (page, bbox) region tuple has a 4-length bbox sequence as its second
        # element; a bare (start, end) character span (two scalars) does not and
        # is resolved against the original instead.
        if (
            len(span) >= 2
            and isinstance(span[1], Sequence)
            and not isinstance(span[1], (str, bytes, bytearray))
            and len(span[1]) >= 4
        ):
            page, bbox, label = span[0], span[1], None
        else:
            return None
    else:
        page = getattr(span, "page", None)
        bbox = getattr(span, "bbox", None)
        label = getattr(span, "label", getattr(span, "entity_type", None))
    if page is None or bbox is None:
        return None
    coerced = _coerce_bbox(bbox)
    if coerced is None:
        return None
    return int(page), coerced, (str(label) if label is not None else None)


def _coerce_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if isinstance(bbox, Mapping):
        try:
            values = [float(bbox[key]) for key in _PDF_WORD_FIELDS]
        except (KeyError, TypeError, ValueError):
            return None
        return tuple(values)  # type: ignore[return-value]
    if isinstance(bbox, Sequence) and len(bbox) >= 4:
        try:
            return tuple(float(value) for value in tuple(bbox)[:4])  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# PDF layout reading (words + rectangles), lazy pdfplumber
# ---------------------------------------------------------------------------


def _import_pdfplumber() -> Any:
    try:
        return importlib.import_module("pdfplumber")
    except ImportError as exc:  # pragma: no cover - exercised without the extra.
        raise MissingDependencyError(
            dependency="pdfplumber", instruction=_PDF_INSTALL_HINT
        ) from exc


def _read_pdf_layout(
    path: str | Path,
) -> dict[int, tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]]:
    pdfplumber = _import_pdfplumber()
    layout: dict[int, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(getattr(pdf, "pages", ())):
            words = tuple(
                word
                for word in page.extract_words(
                    x_tolerance=1,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=True,
                )
                if str(word.get("text", "")).strip()
            )
            rects = tuple(getattr(page, "rects", ()) or ())
            layout[page_index] = (words, rects)
    return layout


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _residual_words(
    words: Iterable[Mapping[str, Any]],
    region: tuple[float, float, float, float],
) -> list[str]:
    residual: list[str] = []
    region_area = _area(region)
    for word in words:
        bbox = _word_bbox(word)
        if bbox is None:
            continue
        word_area = _area(bbox)
        if word_area <= 0 and region_area <= 0:
            continue
        overlap = _intersection_area(bbox, region)
        if overlap <= 0:
            continue
        if overlap / max(word_area, 1e-6) >= _WORD_OVERLAP_FRACTION:
            residual.append(str(word.get("text", "")).strip())
    return [text for text in residual if text]


def _box_covers(
    rects: Iterable[Mapping[str, Any]],
    region: tuple[float, float, float, float],
) -> bool:
    region_area = _area(region)
    if region_area <= 0:
        return False
    for rect in rects:
        if not _is_opaque(rect):
            continue
        bbox = _rect_bbox(rect)
        if bbox is None:
            continue
        if _intersection_area(bbox, region) / region_area >= _BOX_COVER_FRACTION:
            return True
    return False


def _visual_change(
    original: str | Path,
    redacted: str | Path,
    page: int,
    region: tuple[float, float, float, float],
    *,
    rasterizer: Rasterizer | None,
) -> tuple[bool | None, str]:
    render = rasterizer or _default_rasterizer()
    if render is None:
        return None, "vector"
    try:
        before = render(original, page, region)
        after = render(redacted, page, region)
    except (FileNotFoundError, OSError) as exc:
        # A raster backend was selected but a PDF could not be read. Do NOT
        # silently drop the visual check and let a clean-looking box report a
        # pass — that would fail open on a missing/unreadable original. Surface
        # it so the caller (and the CLI) treat it as a verification error.
        raise FileNotFoundError(
            f"Could not read a PDF for visual verification of {original!r} / "
            f"{redacted!r}: {exc}"
        ) from exc
    except Exception:  # pragma: no cover - non-IO backend hiccup -> vector fallback.
        return None, "vector"
    return (before != after), "pixel"


def _default_rasterizer() -> Rasterizer | None:
    """Return a pdfplumber-backed region rasterizer if a backend is available."""
    try:
        importlib.import_module("pdfplumber")
    except ImportError:  # pragma: no cover - no PDF stack installed.
        return None
    # A raster backend (pypdfium2 or ImageMagick/Wand) is required for to_image().
    if (
        importlib.util.find_spec("pypdfium2") is None
        and importlib.util.find_spec("wand") is None
    ):
        return None
    return _pdfplumber_region_bytes  # pragma: no cover - exercised only with a backend.


def _pdfplumber_region_bytes(  # pragma: no cover - requires a raster backend.
    path: str | Path,
    page: int,
    region: tuple[float, float, float, float],
    resolution: int = 150,
) -> bytes:
    pdfplumber = _import_pdfplumber()
    scale = resolution / 72.0
    with pdfplumber.open(path) as pdf:
        page_obj = pdf.pages[page]
        image = page_obj.to_image(resolution=resolution)
        crop = image.original.crop(
            (
                int(region[0] * scale),
                int(region[1] * scale),
                int(region[2] * scale),
                int(region[3] * scale),
            )
        )
        return crop.convert("L").tobytes()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _word_bbox(word: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(word[field]) for field in _PDF_WORD_FIELDS)  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError):
        return None


def _rect_bbox(rect: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    return _word_bbox(rect)


def _is_opaque(rect: Mapping[str, Any]) -> bool:
    # A redaction box is a *filled* rectangle. Stroke-only rects (table borders,
    # cell rules, underlines) are not redactions even though pdfplumber still
    # reports a non_stroking_color for them, so require an actual fill first,
    # then reject white / near-white fills (which hide nothing).
    if not rect.get("fill"):
        return False
    color = rect.get("non_stroking_color")
    if color is None:
        return True  # filled with the graphics-state default (typically black)
    if isinstance(color, (int, float)):
        return float(color) < 0.95
    if isinstance(color, Sequence) and not isinstance(color, (str, bytes)):
        try:
            channels = [float(value) for value in color]
        except (TypeError, ValueError):
            return True
        return any(channel < 0.95 for channel in channels)
    return True


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "PdfFidelityReport",
    "RedactionFidelityError",
    "RegionFidelity",
    "verify_redacted_pdf",
]
