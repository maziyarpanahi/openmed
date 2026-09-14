"""Bounded, memory-only raster redaction with verified PNG/PDF encoding.

This exports already rendered pages. It neither detects identifiers nor certifies
OCR completeness. All source text layers, metadata and auxiliary representations
are excluded; the PDF output deliberately has no searchable text layer.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Literal

from .render_pdf import _append_page, _import_optional, _page_content


class RasterExportError(ValueError):
    """An export failed a resource, geometry, cancellation or encoding check."""


@dataclass(frozen=True)
class RasterRedactionPage:
    """One source raster and top-left, right/bottom-exclusive pixel rectangles.

    Args:
        image: A single-frame Pillow image; ownership remains with the caller.
        regions: Integer pixel rectangles. Padding is applied by the exporter.
        size_points: Source PDF page dimensions, required for PDF export. Never
            infer physical dimensions from an image with unknown resolution.
    """

    image: Any = field(repr=False)
    regions: Sequence[tuple[int, int, int, int]] = field(default_factory=tuple)
    size_points: tuple[float, float] | None = None


@dataclass(frozen=True)
class RasterExportResult:
    """Verified bytes and geometry-only evidence; the content repr is private."""

    content: bytes = field(repr=False)
    media_type: str
    pages: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return bounded evidence without output bytes or source identifiers."""
        return {
            "media_type": self.media_type,
            "byte_count": len(self.content),
            "sha256": hashlib.sha256(self.content).hexdigest(),
            "page_count": len(self.pages),
            "pages": list(self.pages),
            "pixel_verification": "exact_decoded_raster",
            "text_layer": "absent",
            "source_metadata": "not_copied",
        }


class _BoundedBuffer(BytesIO):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + len(data) > self.limit:
            raise RasterExportError("raster_output_limit")
        return super().write(data)


def _cancel(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise RasterExportError("raster_cancelled")


def _rectangles(regions, width, height, padding, remaining):
    if not isinstance(regions, Sequence) or len(regions) > remaining:
        raise RasterExportError("raster_region_limit")
    normalized = []
    for bbox in regions:
        if (
            not isinstance(bbox, (tuple, list))
            or len(bbox) != 4
            or any(type(v) is not int for v in bbox)
        ):
            raise RasterExportError("raster_invalid_geometry")
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise RasterExportError("raster_invalid_geometry")
        normalized.append(
            (
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(width, x1 + padding),
                min(height, y1 + padding),
            )
        )
    return tuple(sorted(set(normalized)))


def _rgb_copy(image, image_module):
    # New RGB storage strips info/EXIF/ICC and hidden transparent pixel values.
    with image.convert("RGBA") as rgba:
        with image_module.new("RGBA", image.size, (255, 255, 255, 255)) as white:
            with image_module.alpha_composite(white, rgba) as flattened:
                with flattened.convert("RGB") as rgb:
                    return image_module.frombytes("RGB", rgb.size, rgb.tobytes())


def _burn(image, regions):
    for bbox in regions:
        image.paste((0, 0, 0), bbox)


def _verify_pixels(source, redacted, regions):
    chops = _import_optional("PIL.ImageChops", dependency="Pillow")
    with chops.difference(source, redacted) as difference:
        for bbox in regions:
            with redacted.crop(bbox) as crop:
                if crop.getextrema() != ((0, 0), (0, 0), (0, 0)):
                    raise RasterExportError("raster_redaction_verification_failed")
            difference.paste((0, 0, 0), bbox)
        if difference.getbbox() is not None:
            raise RasterExportError("raster_outside_pixels_changed")


def _verify_png(content, expected, image_module):
    cursor = 8
    kinds = []
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise RasterExportError("raster_encoding_verification_failed")
    while cursor + 12 <= len(content):
        size = int.from_bytes(content[cursor : cursor + 4], "big")
        kind = content[cursor + 4 : cursor + 8]
        kinds.append(kind)
        cursor += 12 + size
    if (
        cursor != len(content)
        or not kinds
        or kinds[0] != b"IHDR"
        or kinds[-1] != b"IEND"
        or any(kind not in {b"IHDR", b"IDAT", b"IEND"} for kind in kinds)
    ):
        raise RasterExportError("raster_encoding_verification_failed")
    with image_module.open(BytesIO(content)) as decoded:
        if (
            decoded.mode != "RGB"
            or decoded.size != tuple(expected["size_pixels"])
            or hashlib.sha256(decoded.tobytes()).hexdigest()
            != expected["pixels_sha256"]
        ):
            raise RasterExportError("raster_encoding_verification_failed")


def _verify_pdf(content, expected, pikepdf, cancel_check):
    """Accept only this emitter's small image-only object graph and exact pixels."""
    with pikepdf.Pdf.open(BytesIO(content)) as pdf:

        def require(condition):
            if not condition:
                raise RasterExportError("raster_encoding_verification_failed")

        require(set(pdf.Root.keys()) == {"/Type", "/Pages"})
        require(set(pdf.trailer.keys()) <= {"/Root", "/Size", "/ID"})
        require(set(pdf.Root.Pages.keys()) == {"/Type", "/Count", "/Kids"})
        require(
            pdf.Root.Type == pikepdf.Name.Catalog
            and pdf.Root.Pages.Type == pikepdf.Name.Pages
        )
        require(
            len(pdf.pages) == len(expected)
            and int(pdf.Root.Pages.Count) == len(expected)
        )
        known = {pdf.Root.objgen, pdf.Root.Pages.objgen}
        for page, record in zip(pdf.pages, expected, strict=True):
            _cancel(cancel_check)
            require(
                set(page.obj.keys())
                == {"/Type", "/Parent", "/MediaBox", "/Resources", "/Contents"}
            )
            require(
                page.Type == pikepdf.Name.Page
                and page.Parent.objgen == pdf.Root.Pages.objgen
            )
            require(set(page.Resources.keys()) == {"/XObject", "/Font"})
            require(set(page.Resources.XObject.keys()) == {"/Im0"})
            require(set(page.Resources.Font.keys()) == {"/OpenMedSafeText"})
            font = page.Resources.Font["/OpenMedSafeText"]
            require(set(font.keys()) == {"/Type", "/Subtype", "/BaseFont", "/Encoding"})
            require(
                font.Type == pikepdf.Name.Font
                and font.Subtype == pikepdf.Name.Type1
                and font.BaseFont == pikepdf.Name.Helvetica
                and font.Encoding == pikepdf.Name.WinAnsiEncoding
            )
            width, height = record["size_points"]
            require(
                len(page.MediaBox) == 4
                and all(
                    abs(float(a) - b) <= 0.000001
                    for a, b in zip(page.MediaBox, [0, 0, width, height], strict=True)
                )
            )
            stream = page.Resources.XObject["/Im0"]
            require(
                set(stream.keys())
                <= {
                    "/Type",
                    "/Subtype",
                    "/Width",
                    "/Height",
                    "/ColorSpace",
                    "/BitsPerComponent",
                    "/Length",
                    "/Filter",
                }
            )
            require(
                stream.Type == pikepdf.Name.XObject
                and stream.Subtype == pikepdf.Name.Image
                and stream.ColorSpace == pikepdf.Name.DeviceRGB
                and int(stream.BitsPerComponent) == 8
            )
            require([int(stream.Width), int(stream.Height)] == record["size_pixels"])
            require(
                hashlib.sha256(stream.read_bytes()).hexdigest()
                == record["pixels_sha256"]
            )
            require(set(page.Contents.keys()) <= {"/Length", "/Filter"})
            require(
                page.Contents.read_bytes()
                == _page_content(width, height, regions=(), safe_words=())
            )
            known.update(
                (page.objgen, stream.objgen, font.objgen, page.Contents.objgen)
            )
        # Reject auxiliary objects, including otherwise unreferenced streams.
        require({obj.objgen for obj in pdf.objects} == known)


def render_redacted_raster_pages(
    pages: Iterable[RasterRedactionPage],
    *,
    output_format: Literal["pdf", "png"],
    padding_pixels: int = 3,
    max_pages: int = 8,
    max_page_pixels: int = 12_000_000,
    max_total_pixels: int = 72_000_000,
    max_regions: int = 25_000,
    max_output_bytes: int = 32_000_000,
    cancel_check: Callable[[], bool] | None = None,
) -> RasterExportResult:
    """Burn selected pixels and encode fresh, verified PDF or PNG bytes in memory.

    Args:
        pages: Bounded iterable of page rasters. Input images are never modified.
        output_format: PDF for pages with explicit physical sizes; PNG for one image.
        padding_pixels: Extra pixels around each selected rectangle, from 0 to 32.
        max_pages: Maximum source pages.
        max_page_pixels: Maximum pixels on a single page.
        max_total_pixels: Maximum pixels across pages.
        max_regions: Maximum input rectangles across pages, including duplicates.
        max_output_bytes: Maximum encoded file size, enforced during writing.
        cancel_check: Optional cooperative cancellation callback.

    Returns:
        Verified file bytes and coordinate/count/hash evidence. The PDF has no
        text layer, annotations, attachments or original metadata. Exact decoded
        raster verification does not qualify detection, OCR or PDF viewer fidelity.

    Raises:
        RasterExportError: Invalid geometry, exhausted limits or failed verification.
        MissingDependencyError: Pillow or the optional pikepdf PDF writer is absent.
    """
    if (
        output_format not in {"pdf", "png"}
        or type(padding_pixels) is not int
        or not 0 <= padding_pixels <= 32
    ):
        raise RasterExportError("raster_invalid_options")
    if any(
        type(v) is not int or v < 1
        for v in (
            max_pages,
            max_page_pixels,
            max_total_pixels,
            max_regions,
            max_output_bytes,
        )
    ):
        raise RasterExportError("raster_invalid_limits")
    image_module = _import_optional("PIL.Image", dependency="Pillow")
    pikepdf = (
        _import_optional("pikepdf", dependency="pikepdf")
        if output_format == "pdf"
        else None
    )
    pdf = pikepdf.Pdf.new() if pikepdf is not None else None
    records = []
    total_pixels = total_regions = 0
    output = _BoundedBuffer(max_output_bytes)
    try:
        for index, page in enumerate(pages):
            _cancel(cancel_check)
            if not isinstance(page, RasterRedactionPage) or not isinstance(
                page.image, image_module.Image
            ):
                raise RasterExportError("raster_invalid_page")
            if index >= max_pages or (output_format == "png" and index > 0):
                raise RasterExportError("raster_page_limit")
            width, height = page.image.size
            if width < 1 or height < 1 or width * height > max_page_pixels:
                raise RasterExportError("raster_page_pixel_limit")
            total_pixels += width * height
            if total_pixels > max_total_pixels:
                raise RasterExportError("raster_total_pixel_limit")
            if (
                getattr(page.image, "n_frames", 1) != 1
                or page.image.getexif().get(274, 1) != 1
            ):
                raise RasterExportError("raster_source_requires_normalization")
            points = page.size_points
            if output_format == "pdf" and (
                not isinstance(points, (tuple, list))
                or len(points) != 2
                or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    or not 3 <= v <= 14400
                    for v in points
                )
            ):
                raise RasterExportError("raster_invalid_page_size")
            regions = _rectangles(
                page.regions, width, height, padding_pixels, max_regions - total_regions
            )
            total_regions += len(page.regions)
            with _rgb_copy(page.image, image_module) as source:
                with source.copy() as redacted:
                    _burn(redacted, regions)
                    _verify_pixels(source, redacted, regions)
                    record = {
                        "page": index,
                        "size_pixels": [width, height],
                        "size_points": list(points) if output_format == "pdf" else None,
                        "regions_pixels": [list(bbox) for bbox in regions],
                        "padding_pixels": padding_pixels,
                        "pixels_sha256": hashlib.sha256(redacted.tobytes()).hexdigest(),
                    }
                    records.append(record)
                    if output_format == "png":
                        redacted.save(output, format="PNG", optimize=False)
                    else:
                        _append_page(
                            pdf,
                            pikepdf,
                            redacted,
                            width=points[0],
                            height=points[1],
                            regions=(),
                            safe_words=(),
                        )
            _cancel(cancel_check)
        if not records:
            raise RasterExportError("raster_no_pages")
        if pdf is not None:
            pdf.save(
                output,
                force_version="1.7",
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.disable,
                recompress_flate=True,
                deterministic_id=True,
            )
        content = output.getvalue()
        _cancel(cancel_check)
        if pikepdf is None:
            _verify_png(content, records[0], image_module)
        else:
            _verify_pdf(content, records, pikepdf, cancel_check)
        _cancel(cancel_check)
        return RasterExportResult(
            content, "application/pdf" if pikepdf else "image/png", tuple(records)
        )
    except RasterExportError:
        raise
    except Exception:
        raise RasterExportError("raster_export_failed") from None
    finally:
        output.close()
        if pdf is not None:
            pdf.close()
