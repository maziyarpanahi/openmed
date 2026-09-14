"""Deterministic, layout-preserving redacted-PDF rendering and measurement.

The renderer takes top-origin ``(page, bbox)`` rectangles produced by
``project_text_spans``. Each source page is rendered locally, redaction pixels
are burned into that raster, and a new PDF is assembled with pikepdf. The
source content streams, embedded search indexes, thumbnails, annotations, and
other alternate representations are therefore not copied into the output.

For usability, extractable WinAnsi words outside every redaction rectangle are
rebuilt as an invisible, clean text layer. Words touching a redaction rectangle
are omitted. Opaque vector rectangles are also drawn over the burned-in pixels,
which makes the redaction visually explicit and independently verifiable.

Every output is checked before it is atomically published:

* selected source words must be absent from the complete output text layer;
* every rectangle must contain no residual text and have an opaque box; and
* page count, page geometry, and pixels outside redaction rectangles must meet
  deterministic layout-fidelity thresholds.

All processing is local. Reports contain geometry, counts, thresholds, and
hashes only; they never include source or residual plaintext.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import MissingDependencyError
from .verify_pdf import (
    PdfFidelityReport,
    PdfTextRemovalReport,
    verify_redacted_pdf,
    verify_redacted_text_removed,
)

_PDF_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'
_BBOX_FIELDS = ("x0", "top", "x1", "bottom")
_DEFAULT_RENDER_DPI = 144
_DEFAULT_PIXEL_TOLERANCE = 4
_DEFAULT_MAX_OUTSIDE_CHANGED_FRACTION = 0.0005
_DEFAULT_MAX_PAGES = 100
_DEFAULT_MAX_PAGE_PIXELS = 40_000_000
_DEFAULT_MAX_TOTAL_PIXELS = 100_000_000
_DEFAULT_MAX_REGIONS = 10_000
_VERIFICATION_DPI = 150
_PAGE_SIZE_TOLERANCE_POINTS = 0.01
_REDACTION_SAFETY_PADDING_POINTS = 1.0
_MASK_PADDING_POINTS = 1.0
_WINANSI_FONT_RESOURCE = "/OpenMedSafeText"


class PdfLayoutFidelityError(RuntimeError):
    """Raised when non-redacted page layout exceeds a fidelity threshold."""

    def __init__(self, report: "PdfLayoutFidelityReport") -> None:
        self.report = report
        super().__init__(report.summary())


class PdfRenderVerificationError(RuntimeError):
    """Raised when a rendered PDF fails any mandatory safety verification."""

    def __init__(self, result: "PdfRedactionResult") -> None:
        self.result = result
        super().__init__(result.summary())


@dataclass(frozen=True)
class PdfRedactionRegion:
    """One top-origin PDF redaction rectangle in page-coordinate points."""

    page: int
    bbox: tuple[float, float, float, float]
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return PHI-safe geometry and an optional label digest."""
        payload: dict[str, Any] = {"page": self.page, "bbox": list(self.bbox)}
        if self.label is not None:
            payload["label_sha256"] = _sha256_text(self.label)
        return payload


@dataclass(frozen=True)
class PdfPageFidelity:
    """Deterministic layout-fidelity metrics for one page."""

    page: int
    original_size: tuple[float, float]
    redacted_size: tuple[float, float]
    size_preserved: bool
    outside_pixel_count: int
    outside_changed_pixel_count: int
    outside_changed_fraction: float
    outside_mean_absolute_error: float
    max_outside_changed_fraction: float

    @property
    def measurable(self) -> bool:
        """Whether at least one non-redacted pixel was available to compare."""
        return self.outside_pixel_count > 0

    @property
    def passed(self) -> bool:
        """True when geometry and non-redacted pixels meet the configured gate."""
        return self.size_preserved and (
            not self.measurable
            or self.outside_changed_fraction <= self.max_outside_changed_fraction
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stable numeric evidence without file paths or page text."""
        return {
            "page": self.page,
            "passed": self.passed,
            "measurable": self.measurable,
            "original_size": [_rounded(value) for value in self.original_size],
            "redacted_size": [_rounded(value) for value in self.redacted_size],
            "size_preserved": self.size_preserved,
            "outside_pixel_count": self.outside_pixel_count,
            "outside_changed_pixel_count": self.outside_changed_pixel_count,
            "outside_changed_fraction": _rounded(self.outside_changed_fraction),
            "outside_mean_absolute_error": _rounded(self.outside_mean_absolute_error),
        }


@dataclass(frozen=True)
class PdfLayoutFidelityReport:
    """Page-count, geometry, and non-redacted-pixel fidelity report."""

    pages: tuple[PdfPageFidelity, ...]
    original_page_count: int
    redacted_page_count: int
    render_dpi: int
    pixel_tolerance: int
    max_outside_changed_fraction: float
    max_pages: int
    max_page_pixels: int
    max_total_pixels: int

    @property
    def pagination_preserved(self) -> bool:
        """Whether the output has exactly the source page count."""
        return self.original_page_count == self.redacted_page_count

    @property
    def failing_pages(self) -> tuple[PdfPageFidelity, ...]:
        """Return pages that violate geometry or pixel thresholds."""
        return tuple(page for page in self.pages if not page.passed)

    @property
    def passed(self) -> bool:
        """True only when pagination and every page pass the fidelity gate."""
        return (
            self.original_page_count > 0
            and self.pagination_preserved
            and len(self.pages) == self.original_page_count
            and not self.failing_pages
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, PHI-safe regression evidence."""
        return {
            "check": "redacted_pdf_layout_fidelity",
            "passed": self.passed,
            "pagination_preserved": self.pagination_preserved,
            "original_page_count": self.original_page_count,
            "redacted_page_count": self.redacted_page_count,
            "failing_page_count": len(self.failing_pages),
            "render_dpi": self.render_dpi,
            "pixel_tolerance": self.pixel_tolerance,
            "max_outside_changed_fraction": _rounded(self.max_outside_changed_fraction),
            "limits": {
                "max_pages": self.max_pages,
                "max_page_pixels": self.max_page_pixels,
                "max_total_pixels": self.max_total_pixels,
            },
            "pages": [page.to_dict() for page in self.pages],
        }

    def summary(self) -> str:
        """Return a plaintext-free fidelity summary."""
        if self.passed:
            return (
                "PDF layout-fidelity check PASSED for all "
                f"{self.original_page_count} page(s)."
            )
        return (
            "PDF layout-fidelity check FAILED: "
            f"page counts {self.original_page_count}/{self.redacted_page_count}; "
            f"{len(self.failing_pages)} measured page(s) exceeded the gate."
        )

    def raise_for_regression(self) -> "PdfLayoutFidelityReport":
        """Raise :class:`PdfLayoutFidelityError` when the report does not pass."""
        if not self.passed:
            raise PdfLayoutFidelityError(self)
        return self


@dataclass(frozen=True)
class PdfRedactionResult:
    """Verified output plus PHI-safe text, box, and layout evidence."""

    output_path: Path
    page_count: int
    region_count: int
    render_dpi: int
    output_sha256: str
    region_fidelity: PdfFidelityReport
    text_removal: PdfTextRemovalReport
    layout_fidelity: PdfLayoutFidelityReport

    @property
    def passed(self) -> bool:
        """Whether all mandatory redaction and layout checks passed."""
        return (
            self.region_fidelity.passed
            and self.text_removal.passed
            and self.layout_fidelity.passed
        )

    def to_dict(self) -> dict[str, Any]:
        """Return audit-safe evidence without echoing a potentially identifying path."""
        return {
            "check": "redacted_pdf_render",
            "passed": self.passed,
            "page_count": self.page_count,
            "region_count": self.region_count,
            "render_dpi": self.render_dpi,
            "output_sha256": self.output_sha256,
            "region_fidelity": self.region_fidelity.to_dict(),
            "text_removal": self.text_removal.to_dict(),
            "layout_fidelity": self.layout_fidelity.to_dict(),
        }

    def summary(self) -> str:
        """Return a plaintext-free render verification summary."""
        state = "PASSED" if self.passed else "FAILED"
        return (
            f"PDF redaction render {state}: {self.page_count} page(s), "
            f"{self.region_count} region(s); region/text/layout checks "
            f"{self.region_fidelity.passed}/{self.text_removal.passed}/"
            f"{self.layout_fidelity.passed}."
        )


def render_redacted_pdf(
    source: str | Path,
    output: str | Path,
    regions: Iterable[Any],
    *,
    render_dpi: int = _DEFAULT_RENDER_DPI,
    fidelity_dpi: int | None = None,
    pixel_tolerance: int = _DEFAULT_PIXEL_TOLERANCE,
    max_outside_changed_fraction: float = (_DEFAULT_MAX_OUTSIDE_CHANGED_FRACTION),
    max_pages: int = _DEFAULT_MAX_PAGES,
    max_page_pixels: int = _DEFAULT_MAX_PAGE_PIXELS,
    max_total_pixels: int = _DEFAULT_MAX_TOTAL_PIXELS,
    max_regions: int = _DEFAULT_MAX_REGIONS,
    overwrite: bool = False,
) -> PdfRedactionResult:
    """Render and verify a clean redacted PDF from projected rectangles.

    Args:
        source: Source digital PDF. Processing is local and never uses a network.
        output: Destination PDF. It is published atomically only after every
            mandatory verification passes.
        regions: Iterable of ``(page, bbox)`` tuples, mappings/objects with
            ``page`` and ``bbox``, or ``ProjectedRectangle`` instances. Bboxes
            use pdfplumber's top-origin ``(x0, top, x1, bottom)`` coordinates.
        render_dpi: Resolution used to burn each source page into safe pixels.
        fidelity_dpi: Resolution used by the independent regression check.
            Defaults to ``render_dpi``.
        pixel_tolerance: Maximum per-channel 0-255 difference treated as stable.
        max_outside_changed_fraction: Maximum fraction of pixels outside all
            redaction boxes that may differ.
        max_pages: Maximum number of source pages accepted for one render.
        max_page_pixels: Maximum rendered pixels accepted for any page.
        max_total_pixels: Maximum rendered pixels accepted across all pages.
        max_regions: Maximum number of distinct redaction rectangles.
        overwrite: Permit atomically replacing an existing output file.

    Returns:
        A :class:`PdfRedactionResult` containing PHI-safe verification evidence.

    Raises:
        ValueError: If regions, thresholds, page indexes, or bboxes are invalid.
        FileExistsError: If ``output`` exists and ``overwrite`` is false.
        PdfRenderVerificationError: If text removal, boxes, or layout fail.
        MissingDependencyError: If the ``multimodal`` PDF stack is unavailable.
    """
    render_dpi = _validate_dpi(render_dpi, name="render_dpi")
    fidelity_dpi = _validate_dpi(
        render_dpi if fidelity_dpi is None else fidelity_dpi,
        name="fidelity_dpi",
    )
    pixel_tolerance = _validate_pixel_tolerance(pixel_tolerance)
    max_outside_changed_fraction = _validate_fraction(
        max_outside_changed_fraction,
        name="max_outside_changed_fraction",
    )
    max_pages = _validate_positive_limit(max_pages, name="max_pages")
    max_page_pixels = _validate_positive_limit(max_page_pixels, name="max_page_pixels")
    max_total_pixels = _validate_positive_limit(
        max_total_pixels, name="max_total_pixels"
    )
    max_regions = _validate_positive_limit(max_regions, name="max_regions")
    normalized_regions = _normalize_regions(regions)
    if not normalized_regions:
        raise ValueError("At least one redaction region is required")
    if len(normalized_regions) > max_regions:
        raise ValueError(
            f"PDF redaction region count exceeds max_regions={max_regions}"
        )

    source_path = Path(source)
    output_path = Path(output)
    _validate_paths(source_path, output_path, overwrite=overwrite)
    pdfplumber, pikepdf, image_draw = _import_render_stack()

    output_pdf = pikepdf.Pdf.new()
    try:
        try:
            _reject_type3_fonts(source_path, pikepdf)
            with pdfplumber.open(source_path) as source_pdf:
                pages = tuple(getattr(source_pdf, "pages", ()))
                if not pages:
                    raise ValueError("Source PDF contains no pages")
                _validate_region_pages(normalized_regions, pages)
                _validate_page_budget(
                    pages,
                    render_dpi=max(render_dpi, fidelity_dpi, _VERIFICATION_DPI),
                    max_pages=max_pages,
                    max_page_pixels=max_page_pixels,
                    max_total_pixels=max_total_pixels,
                )
                normalized_regions = _pad_regions(normalized_regions, pages)
                for page_index, page in enumerate(pages):
                    page_regions = tuple(
                        region
                        for region in normalized_regions
                        if region.page == page_index
                    )
                    width, height = _page_size(page)
                    image = _render_page(page, resolution=render_dpi)
                    _burn_redactions(
                        image,
                        page_regions,
                        width=width,
                        height=height,
                        image_draw=image_draw,
                    )
                    safe_words = _safe_text_words(page, page_regions)
                    _append_page(
                        output_pdf,
                        pikepdf,
                        image,
                        width=width,
                        height=height,
                        regions=page_regions,
                        safe_words=safe_words,
                    )
        except (ValueError, IndexError):
            raise
        except Exception:
            raise RuntimeError("Source PDF could not be rendered safely") from None

        temporary_path = _temporary_output_path(output_path)
        try:
            try:
                output_pdf.save(
                    temporary_path,
                    force_version="1.7",
                    compress_streams=True,
                    object_stream_mode=pikepdf.ObjectStreamMode.disable,
                    recompress_flate=True,
                    deterministic_id=True,
                )
                result = _verify_rendered_output(
                    source_path,
                    temporary_path,
                    output_path,
                    normalized_regions,
                    page_count=len(output_pdf.pages),
                    render_dpi=render_dpi,
                    fidelity_dpi=fidelity_dpi,
                    pixel_tolerance=pixel_tolerance,
                    max_outside_changed_fraction=max_outside_changed_fraction,
                    max_pages=max_pages,
                    max_page_pixels=max_page_pixels,
                    max_total_pixels=max_total_pixels,
                    pdfplumber=pdfplumber,
                )
            except PdfRenderVerificationError:
                raise
            except Exception:
                raise RuntimeError(
                    "Redacted PDF could not be verified safely"
                ) from None
            if not result.passed:
                raise PdfRenderVerificationError(result)
            try:
                _publish_temporary_output(
                    temporary_path,
                    output_path,
                    overwrite=overwrite,
                )
            except FileExistsError:
                raise
            except OSError:
                raise RuntimeError(
                    "Verified PDF could not be published safely"
                ) from None
            return result
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        output_pdf.close()


def write_redacted_pdf(
    source: str | Path,
    output: str | Path,
    regions: Iterable[Any],
    **kwargs: Any,
) -> PdfRedactionResult:
    """Compatibility-style write helper delegating to :func:`render_redacted_pdf`."""
    return render_redacted_pdf(source, output, regions, **kwargs)


def measure_pdf_layout_fidelity(
    original: str | Path,
    redacted: str | Path,
    regions: Iterable[Any],
    *,
    render_dpi: int = _DEFAULT_RENDER_DPI,
    pixel_tolerance: int = _DEFAULT_PIXEL_TOLERANCE,
    max_outside_changed_fraction: float = (_DEFAULT_MAX_OUTSIDE_CHANGED_FRACTION),
    max_pages: int = _DEFAULT_MAX_PAGES,
    max_page_pixels: int = _DEFAULT_MAX_PAGE_PIXELS,
    max_total_pixels: int = _DEFAULT_MAX_TOTAL_PIXELS,
    strict: bool = False,
) -> PdfLayoutFidelityReport:
    """Measure page geometry and pixels outside redaction rectangles.

    The comparison is deterministic for a fixed local PDF stack. Pixels inside
    each requested rectangle, plus a one-point antialiasing margin, are masked
    out. The remaining pixels form an enforceable regression gate. No OCR or
    page text is included in the returned report.
    """
    render_dpi = _validate_dpi(render_dpi, name="render_dpi")
    pixel_tolerance = _validate_pixel_tolerance(pixel_tolerance)
    max_outside_changed_fraction = _validate_fraction(
        max_outside_changed_fraction,
        name="max_outside_changed_fraction",
    )
    max_pages = _validate_positive_limit(max_pages, name="max_pages")
    max_page_pixels = _validate_positive_limit(max_page_pixels, name="max_page_pixels")
    max_total_pixels = _validate_positive_limit(
        max_total_pixels, name="max_total_pixels"
    )
    normalized_regions = _normalize_regions(regions)
    if not normalized_regions:
        raise ValueError("At least one redaction region is required")

    pdfplumber, image_chops, image_draw, image_stat = _import_measure_stack()
    with (
        pdfplumber.open(original) as original_pdf,
        pdfplumber.open(redacted) as redacted_pdf,
    ):
        original_pages = tuple(getattr(original_pdf, "pages", ()))
        redacted_pages = tuple(getattr(redacted_pdf, "pages", ()))
        _validate_region_pages(normalized_regions, original_pages)
        _validate_page_budget(
            original_pages,
            render_dpi=render_dpi,
            max_pages=max_pages,
            max_page_pixels=max_page_pixels,
            max_total_pixels=max_total_pixels,
        )
        _validate_page_budget(
            redacted_pages,
            render_dpi=render_dpi,
            max_pages=max_pages,
            max_page_pixels=max_page_pixels,
            max_total_pixels=max_total_pixels,
        )
        pages: list[PdfPageFidelity] = []
        for page_index, (before_page, after_page) in enumerate(
            zip(original_pages, redacted_pages)
        ):
            original_size = _page_size(before_page)
            redacted_size = _page_size(after_page)
            size_preserved = all(
                abs(left - right) <= _PAGE_SIZE_TOLERANCE_POINTS
                for left, right in zip(original_size, redacted_size)
            )
            before = _render_page(before_page, resolution=render_dpi)
            after = _render_page(after_page, resolution=render_dpi)
            page_regions = tuple(
                region for region in normalized_regions if region.page == page_index
            )
            metrics = _outside_pixel_metrics(
                before,
                after,
                page_regions,
                page_size=original_size,
                pixel_tolerance=pixel_tolerance,
                image_chops=image_chops,
                image_draw=image_draw,
                image_stat=image_stat,
            )
            pages.append(
                PdfPageFidelity(
                    page=page_index,
                    original_size=original_size,
                    redacted_size=redacted_size,
                    size_preserved=size_preserved,
                    outside_pixel_count=metrics[0],
                    outside_changed_pixel_count=metrics[1],
                    outside_changed_fraction=metrics[2],
                    outside_mean_absolute_error=metrics[3],
                    max_outside_changed_fraction=max_outside_changed_fraction,
                )
            )

    report = PdfLayoutFidelityReport(
        pages=tuple(pages),
        original_page_count=len(original_pages),
        redacted_page_count=len(redacted_pages),
        render_dpi=render_dpi,
        pixel_tolerance=pixel_tolerance,
        max_outside_changed_fraction=max_outside_changed_fraction,
        max_pages=max_pages,
        max_page_pixels=max_page_pixels,
        max_total_pixels=max_total_pixels,
    )
    if strict:
        report.raise_for_regression()
    return report


def _verify_rendered_output(
    source: Path,
    temporary: Path,
    output: Path,
    regions: tuple[PdfRedactionRegion, ...],
    *,
    page_count: int,
    render_dpi: int,
    fidelity_dpi: int,
    pixel_tolerance: int,
    max_outside_changed_fraction: float,
    max_pages: int,
    max_page_pixels: int,
    max_total_pixels: int,
    pdfplumber: Any,
) -> PdfRedactionResult:
    region_payload = tuple(region.to_dict() for region in regions)
    region_fidelity = verify_redacted_pdf(
        source,
        temporary,
        region_payload,
        rasterizer=_cached_region_rasterizer(pdfplumber),
    )
    text_removal = verify_redacted_text_removed(source, temporary, region_payload)
    layout_fidelity = measure_pdf_layout_fidelity(
        source,
        temporary,
        region_payload,
        render_dpi=fidelity_dpi,
        pixel_tolerance=pixel_tolerance,
        max_outside_changed_fraction=max_outside_changed_fraction,
        max_pages=max_pages,
        max_page_pixels=max_page_pixels,
        max_total_pixels=max_total_pixels,
    )
    return PdfRedactionResult(
        output_path=output,
        page_count=page_count,
        region_count=len(regions),
        render_dpi=render_dpi,
        output_sha256=_file_sha256(temporary),
        region_fidelity=region_fidelity,
        text_removal=text_removal,
        layout_fidelity=layout_fidelity,
    )


def _append_page(
    pdf: Any,
    pikepdf: Any,
    image: Any,
    *,
    width: float,
    height: float,
    regions: tuple[PdfRedactionRegion, ...],
    safe_words: tuple[Mapping[str, Any], ...],
) -> None:
    page = pdf.add_blank_page(page_size=(width, height))
    image_stream = pdf.make_stream(image.tobytes())
    image_stream.Type = pikepdf.Name("/XObject")
    image_stream.Subtype = pikepdf.Name("/Image")
    image_stream.Width = image.width
    image_stream.Height = image.height
    image_stream.ColorSpace = pikepdf.Name("/DeviceRGB")
    image_stream.BitsPerComponent = 8

    font = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Helvetica"),
                "/Encoding": pikepdf.Name("/WinAnsiEncoding"),
            }
        )
    )
    page.Resources = pikepdf.Dictionary(
        {
            "/XObject": pikepdf.Dictionary({"/Im0": image_stream}),
            "/Font": pikepdf.Dictionary({_WINANSI_FONT_RESOURCE: font}),
        }
    )
    page.Contents = pdf.make_stream(
        _page_content(width, height, regions=regions, safe_words=safe_words)
    )


def _page_content(
    width: float,
    height: float,
    *,
    regions: tuple[PdfRedactionRegion, ...],
    safe_words: tuple[Mapping[str, Any], ...],
) -> bytes:
    parts = [
        b"q\n",
        (f"{_pdf_number(width)} 0 0 {_pdf_number(height)} 0 0 cm\n").encode("ascii"),
        b"/Im0 Do\nQ\n",
    ]
    text_commands = _safe_text_commands(safe_words, page_height=height)
    if text_commands:
        parts.append(text_commands)
    if regions:
        parts.append(b"0 0 0 rg\n")
        for region in regions:
            x0, top, x1, bottom = region.bbox
            y = height - bottom
            parts.append(
                (
                    f"{_pdf_number(x0)} {_pdf_number(y)} "
                    f"{_pdf_number(x1 - x0)} {_pdf_number(bottom - top)} re f\n"
                ).encode("ascii")
            )
    return b"".join(parts)


def _safe_text_commands(
    words: tuple[Mapping[str, Any], ...], *, page_height: float
) -> bytes:
    commands: list[bytes] = []
    for word in words:
        text = str(word.get("text", "")).strip()
        try:
            encoded = text.encode("cp1252")
        except UnicodeEncodeError:
            # The raster remains authoritative for visible multilingual content.
            # Never emit a lossy or incorrectly mapped text layer.
            continue
        bbox = _mapping_bbox(word)
        if not encoded or bbox is None:
            continue
        x0, top, _x1, bottom = bbox
        font_size = max(1.0, bottom - top)
        baseline = page_height - bottom + (font_size * 0.18)
        commands.extend(
            [
                b"BT\n",
                f"{_WINANSI_FONT_RESOURCE} {_pdf_number(font_size)} Tf\n".encode(
                    "ascii"
                ),
                b"3 Tr\n",
                (f"1 0 0 1 {_pdf_number(x0)} {_pdf_number(baseline)} Tm\n").encode(
                    "ascii"
                ),
                b"<" + encoded.hex().upper().encode("ascii") + b"> Tj\n",
                b"ET\n",
            ]
        )
    return b"".join(commands)


def _safe_text_words(
    page: Any, regions: tuple[PdfRedactionRegion, ...]
) -> tuple[Mapping[str, Any], ...]:
    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=True,
    )
    safe: list[Mapping[str, Any]] = []
    for word in words:
        if not str(word.get("text", "")).strip():
            continue
        bbox = _mapping_bbox(word)
        if bbox is None:
            continue
        if any(_intersection_area(bbox, region.bbox) > 0 for region in regions):
            continue
        safe.append(word)
    return tuple(safe)


def _burn_redactions(
    image: Any,
    regions: tuple[PdfRedactionRegion, ...],
    *,
    width: float,
    height: float,
    image_draw: Any,
) -> None:
    draw = image_draw.Draw(image)
    scale_x = image.width / width
    scale_y = image.height / height
    for region in regions:
        x0, top, x1, bottom = region.bbox
        left = max(0, math.floor(x0 * scale_x))
        upper = max(0, math.floor(top * scale_y))
        right = min(image.width, math.ceil(x1 * scale_x))
        lower = min(image.height, math.ceil(bottom * scale_y))
        if left >= right or upper >= lower:
            raise ValueError(f"Redaction region on page {region.page} is empty")
        draw.rectangle((left, upper, right - 1, lower - 1), fill=(0, 0, 0))


def _outside_pixel_metrics(
    before: Any,
    after: Any,
    regions: tuple[PdfRedactionRegion, ...],
    *,
    page_size: tuple[float, float],
    pixel_tolerance: int,
    image_chops: Any,
    image_draw: Any,
    image_stat: Any,
) -> tuple[int, int, float, float]:
    if before.size != after.size:
        pixel_count = before.width * before.height
        return pixel_count, pixel_count, 1.0, 1.0

    outside_mask = before.getchannel("R").point(lambda _value: 255)
    draw = image_draw.Draw(outside_mask)
    scale_x = before.width / page_size[0]
    scale_y = before.height / page_size[1]
    padding_x = math.ceil(_MASK_PADDING_POINTS * scale_x)
    padding_y = math.ceil(_MASK_PADDING_POINTS * scale_y)
    for region in regions:
        x0, top, x1, bottom = region.bbox
        draw.rectangle(
            (
                max(0, math.floor(x0 * scale_x) - padding_x),
                max(0, math.floor(top * scale_y) - padding_y),
                min(before.width - 1, math.ceil(x1 * scale_x) + padding_x),
                min(before.height - 1, math.ceil(bottom * scale_y) + padding_y),
            ),
            fill=0,
        )

    outside_pixel_count = outside_mask.histogram()[255]
    if outside_pixel_count == 0:
        return 0, 0, 0.0, 0.0

    difference = image_chops.difference(before, after)
    red, green, blue = difference.split()
    maximum = image_chops.lighter(red, image_chops.lighter(green, blue))
    changed = maximum.point(
        lambda value: 255 if value > pixel_tolerance else 0,
        mode="L",
    )
    changed_outside = image_chops.multiply(changed, outside_mask)
    outside_changed_pixel_count = changed_outside.histogram()[255]
    means = image_stat.Stat(difference, mask=outside_mask).mean
    mean_absolute_error = sum(means) / (len(means) * 255.0)
    return (
        outside_pixel_count,
        outside_changed_pixel_count,
        outside_changed_pixel_count / outside_pixel_count,
        mean_absolute_error,
    )


def _cached_region_rasterizer(
    pdfplumber: Any,
    *,
    resolution: int = _VERIFICATION_DPI,
) -> Any:
    cached_page = -1
    cached_images: dict[str, tuple[Any, tuple[float, float]]] = {}

    def rasterize(
        path: str | Path,
        page: int,
        region: tuple[float, float, float, float],
    ) -> bytes:
        nonlocal cached_page
        if page != cached_page:
            cached_images.clear()
            cached_page = page
        key = os.fspath(path)
        cached = cached_images.get(key)
        if cached is None:
            with pdfplumber.open(path) as pdf:
                page_obj = pdf.pages[page]
                page_size = _page_size(page_obj)
                image = _render_page(page_obj, resolution=resolution)
            cached = (image, page_size)
            cached_images[key] = cached

        image, page_size = cached
        scale_x = image.width / page_size[0]
        scale_y = image.height / page_size[1]
        crop = image.crop(
            (
                max(0, math.floor(region[0] * scale_x)),
                max(0, math.floor(region[1] * scale_y)),
                min(image.width, math.ceil(region[2] * scale_x)),
                min(image.height, math.ceil(region[3] * scale_y)),
            )
        )
        return crop.convert("L").tobytes()

    return rasterize


def _normalize_regions(regions: Iterable[Any]) -> tuple[PdfRedactionRegion, ...]:
    normalized = {_coerce_region(region) for region in regions}
    return tuple(
        sorted(
            normalized,
            key=lambda region: (region.page, region.bbox, region.label or ""),
        )
    )


def _pad_regions(
    regions: tuple[PdfRedactionRegion, ...],
    pages: Sequence[Any],
) -> tuple[PdfRedactionRegion, ...]:
    padded: list[PdfRedactionRegion] = []
    for region in regions:
        width, height = _page_size(pages[region.page])
        x0, top, x1, bottom = region.bbox
        padded.append(
            PdfRedactionRegion(
                page=region.page,
                bbox=(
                    max(0.0, x0 - _REDACTION_SAFETY_PADDING_POINTS),
                    max(0.0, top - _REDACTION_SAFETY_PADDING_POINTS),
                    min(width, x1 + _REDACTION_SAFETY_PADDING_POINTS),
                    min(height, bottom + _REDACTION_SAFETY_PADDING_POINTS),
                ),
                label=region.label,
            )
        )
    return tuple(padded)


def _coerce_region(value: Any) -> PdfRedactionRegion:
    if isinstance(value, PdfRedactionRegion):
        region = value
    elif isinstance(value, Mapping):
        region = PdfRedactionRegion(
            page=_coerce_page(value.get("page")),
            bbox=_coerce_bbox(value.get("bbox")),
            label=_coerce_label(value.get("label", value.get("entity_type"))),
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 2:
            raise ValueError("A PDF redaction tuple must contain page and bbox")
        region = PdfRedactionRegion(
            page=_coerce_page(value[0]),
            bbox=_coerce_bbox(value[1]),
            label=_coerce_label(value[2] if len(value) > 2 else None),
        )
    else:
        region = PdfRedactionRegion(
            page=_coerce_page(getattr(value, "page", None)),
            bbox=_coerce_bbox(getattr(value, "bbox", None)),
            label=_coerce_label(
                getattr(value, "label", getattr(value, "entity_type", None))
            ),
        )
    if region.page < 0:
        raise ValueError("PDF redaction page indexes must be non-negative")
    return region


def _coerce_page(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("PDF redaction page indexes must be integers")
    try:
        page = int(value)
    except (TypeError, ValueError):
        raise ValueError("PDF redaction page indexes must be integers") from None
    if value is not None and isinstance(value, float) and not value.is_integer():
        raise ValueError("PDF redaction page indexes must be integers")
    return page


def _coerce_bbox(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        try:
            coordinates = tuple(float(value[field]) for field in _BBOX_FIELDS)
        except (KeyError, TypeError, ValueError):
            raise ValueError("PDF redaction bboxes require x0/top/x1/bottom") from None
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 4:
            raise ValueError("PDF redaction bboxes require four coordinates")
        try:
            coordinates = tuple(float(item) for item in tuple(value)[:4])
        except (TypeError, ValueError):
            raise ValueError("PDF redaction bbox coordinates must be numeric") from None
    else:
        raise ValueError("PDF redaction bboxes require four coordinates")
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise ValueError("PDF redaction bbox coordinates must be finite")
    x0, top, x1, bottom = coordinates
    if x0 >= x1 or top >= bottom:
        raise ValueError("PDF redaction bboxes must have positive width and height")
    return coordinates  # type: ignore[return-value]


def _coerce_label(value: Any) -> str | None:
    return None if value is None else str(value)


def _validate_region_pages(
    regions: tuple[PdfRedactionRegion, ...], pages: Sequence[Any]
) -> None:
    if not pages:
        raise ValueError("Source PDF contains no pages")
    for region in regions:
        if region.page >= len(pages):
            raise ValueError(
                f"Redaction page {region.page} is outside the {len(pages)}-page PDF"
            )
        width, height = _page_size(pages[region.page])
        x0, top, x1, bottom = region.bbox
        if x0 < 0 or top < 0 or x1 > width or bottom > height:
            raise ValueError(
                f"Redaction bbox on page {region.page} is outside page bounds"
            )


def _validate_page_budget(
    pages: Sequence[Any],
    *,
    render_dpi: int,
    max_pages: int,
    max_page_pixels: int,
    max_total_pixels: int,
) -> None:
    if len(pages) > max_pages:
        raise ValueError(f"PDF page count exceeds max_pages={max_pages}")
    total_pixels = 0
    for page_index, page in enumerate(pages):
        width, height = _page_size(page)
        page_pixels = _rendered_pixel_count(width, height, render_dpi)
        if page_pixels > max_page_pixels:
            raise ValueError(
                f"Rendered page {page_index} exceeds max_page_pixels={max_page_pixels}"
            )
        total_pixels += page_pixels
        if total_pixels > max_total_pixels:
            raise ValueError(
                f"Rendered PDF exceeds max_total_pixels={max_total_pixels}"
            )


def _reject_type3_fonts(source: Path, pikepdf: Any) -> None:
    with pikepdf.open(source) as pdf:
        for obj in pdf.objects:
            try:
                object_type = str(obj.get("/Type", ""))
                subtype = str(obj.get("/Subtype", ""))
            except (AttributeError, TypeError, ValueError):
                continue
            if object_type == "/Font" and subtype == "/Type3":
                raise ValueError(
                    "Type 3 fonts are not supported for safe PDF redaction"
                )


def _validate_paths(source: Path, output: Path, *, overwrite: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError("Source PDF is not a readable regular file")
    if source.resolve() == output.resolve():
        raise ValueError("Source and output PDF paths must differ")
    if not output.parent.is_dir():
        raise FileNotFoundError("Output PDF parent directory does not exist")
    if output.exists() and not overwrite:
        raise FileExistsError("Output PDF already exists")


def _validate_dpi(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer from 72 through 600")
    try:
        dpi = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer from 72 through 600") from None
    if dpi != value or not 72 <= dpi <= 600:
        raise ValueError(f"{name} must be an integer from 72 through 600")
    return dpi


def _validate_pixel_tolerance(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("pixel_tolerance must be an integer from 0 through 255")
    try:
        tolerance = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "pixel_tolerance must be an integer from 0 through 255"
        ) from None
    if tolerance != value or not 0 <= tolerance <= 255:
        raise ValueError("pixel_tolerance must be an integer from 0 through 255")
    return tolerance


def _validate_fraction(value: Any, *, name: str) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be between 0 and 1") from None
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return fraction


def _validate_positive_limit(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        limit = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if limit != value or limit <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return limit


def _page_size(page: Any) -> tuple[float, float]:
    width = float(page.width)
    height = float(page.height)
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("PDF page dimensions must be finite and positive")
    return width, height


def _rendered_pixel_count(width: float, height: float, render_dpi: int) -> int:
    pixel_width = math.ceil(width * render_dpi / 72.0)
    pixel_height = math.ceil(height * render_dpi / 72.0)
    return pixel_width * pixel_height


def _render_page(page: Any, *, resolution: int) -> Any:
    image = page.to_image(resolution=resolution, antialias=True).original
    return image.convert("RGB")


def _mapping_bbox(value: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        bbox = tuple(float(value[field]) for field in _BBOX_FIELDS)
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        return None
    return bbox  # type: ignore[return-value]


def _intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    top = max(left[1], right[1])
    x1 = min(left[2], right[2])
    bottom = min(left[3], right[3])
    return max(0.0, x1 - x0) * max(0.0, bottom - top)


def _temporary_output_path(output: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        prefix=".openmed-redacted-",
        suffix=".pdf",
        dir=output.parent,
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    return path


def _publish_temporary_output(
    temporary: Path,
    output: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        os.replace(temporary, output)
        return
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise FileExistsError("Output PDF already exists") from None


def _import_render_stack() -> tuple[Any, Any, Any]:
    pdfplumber = _import_optional("pdfplumber", dependency="pdfplumber")
    pikepdf = _import_optional("pikepdf", dependency="pikepdf")
    image_draw = _import_optional("PIL.ImageDraw", dependency="Pillow")
    return pdfplumber, pikepdf, image_draw


def _import_measure_stack() -> tuple[Any, Any, Any, Any]:
    pdfplumber = _import_optional("pdfplumber", dependency="pdfplumber")
    image_chops = _import_optional("PIL.ImageChops", dependency="Pillow")
    image_draw = _import_optional("PIL.ImageDraw", dependency="Pillow")
    image_stat = _import_optional("PIL.ImageStat", dependency="Pillow")
    return pdfplumber, image_chops, image_draw, image_stat


def _import_optional(module: str, *, dependency: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised without the extra.
        raise MissingDependencyError(
            dependency=dependency,
            instruction=_PDF_INSTALL_HINT,
        ) from exc


def _pdf_number(value: float) -> str:
    rounded = round(float(value), 6)
    if rounded == 0:
        return "0"
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def _rounded(value: float) -> float:
    return round(float(value), 8)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PdfLayoutFidelityError",
    "PdfLayoutFidelityReport",
    "PdfPageFidelity",
    "PdfRedactionRegion",
    "PdfRedactionResult",
    "PdfRenderVerificationError",
    "measure_pdf_layout_fidelity",
    "render_redacted_pdf",
    "write_redacted_pdf",
]
