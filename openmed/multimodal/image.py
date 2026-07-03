"""Plain raster-image PHI redaction with pixel boxes.

PNG, JPEG, and TIFF images can contain PHI both as burned-in pixels and as
embedded metadata. This module OCRs image frames, projects detected PHI spans
back to OCR word boxes, draws opaque rectangles over those boxes, saves a fresh
metadata-free image, and can re-OCR the redacted output to check for residual
PHI.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .base import ExtractedDocument, SourceSpan, register_handler
from .documents_pdf import ProjectedRectangle, project_text_spans
from .exceptions import MissingDependencyError, UnsupportedDocumentError
from .ocr import OcrEngine, OcrResult, ocr

_PILLOW_HINT = 'Install with: pip install "openmed[multimodal]".'
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_PHI_METADATA_KEYS = frozenset(
    {
        "exif",
        "xmp",
        "xml:com.adobe.xmp",
        "iptc",
        "photoshop",
        "comment",
        "description",
        "author",
        "artist",
        "copyright",
    }
)


@dataclass(frozen=True)
class ImageMetadataReport:
    """PHI-bearing metadata keys found on a redacted output image."""

    residual_keys: tuple[str, ...] = ()
    checked_keys: tuple[str, ...] = field(default_factory=tuple)

    @property
    def residual_count(self) -> int:
        """Number of PHI-bearing metadata keys still present."""

        return len(self.residual_keys)

    @property
    def clean(self) -> bool:
        """Whether no PHI-bearing image metadata keys remain."""

        return self.residual_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable PHI-safe report."""

        return {
            "type": "image_metadata_report",
            "clean": self.clean,
            "residual_count": self.residual_count,
            "residual_keys": list(self.residual_keys),
            "checked_keys": list(self.checked_keys),
        }


@dataclass(frozen=True)
class ResidualPhi:
    """A residual PHI span detected after image pixel redaction."""

    start: int
    end: int
    page: int | None = None
    label: str | None = None
    confidence: float | None = None
    text_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-safe residual finding."""

        payload: dict[str, Any] = {"start": self.start, "end": self.end}
        if self.page is not None:
            payload["page"] = self.page
        if self.label is not None:
            payload["label"] = self.label
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.text_sha256 is not None:
            payload["text_sha256"] = self.text_sha256
        return payload


@dataclass(frozen=True)
class ResidualPhiReport:
    """Result of re-OCRing a redacted image and detecting residual PHI."""

    residual_phi: tuple[ResidualPhi, ...] = ()
    frame_count: int = 0
    word_count: int = 0

    @property
    def residual_count(self) -> int:
        """Number of residual PHI spans."""

        return len(self.residual_phi)

    @property
    def clean(self) -> bool:
        """Whether verification found no residual PHI."""

        return self.residual_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable PHI-safe report."""

        return {
            "type": "image_redaction_residual_report",
            "clean": self.clean,
            "residual_count": self.residual_count,
            "frame_count": self.frame_count,
            "word_count": self.word_count,
            "residual_phi": [residual.to_dict() for residual in self.residual_phi],
        }


class ImageRedactionVerificationError(ValueError):
    """Raised when re-OCR verification finds residual PHI."""

    def __init__(self, report: ResidualPhiReport) -> None:
        super().__init__(
            "Residual PHI remained after image redaction: "
            f"{report.residual_count} residual span(s)."
        )
        self.report = report


@dataclass(frozen=True)
class RedactedImage:
    """A redacted image payload plus PHI-safe audit metadata."""

    source_path: Path
    redacted_bytes: bytes
    extracted_document: ExtractedDocument
    redaction_boxes: tuple[ProjectedRectangle, ...]
    frame_count: int
    image_format: str
    changed_pixel_count: int
    changed_pixels_by_frame: tuple[int, ...]
    metadata_report: ImageMetadataReport
    residual_report: ResidualPhiReport | None = None
    output_path: Path | None = None

    @property
    def modified_pixels(self) -> bool:
        """Whether any source pixels changed during redaction."""

        return self.changed_pixel_count > 0

    def to_document(self) -> ExtractedDocument:
        """Bridge the redaction result into the multimodal document contract."""

        metadata = dict(self.extracted_document.metadata)
        metadata.update(
            {
                "format": "image",
                "image_format": self.image_format,
                "frame_count": self.frame_count,
                "detected_span_count": len(self.redaction_boxes),
                "redaction_rectangles": [
                    rectangle.to_dict() for rectangle in self.redaction_boxes
                ],
                "pixel_redaction": {
                    "modified_pixels": self.modified_pixels,
                    "changed_pixel_count": self.changed_pixel_count,
                    "changed_pixels_by_frame": list(self.changed_pixels_by_frame),
                },
                "metadata_report": self.metadata_report.to_dict(),
                "redacted_image_sha256": hashlib.sha256(
                    self.redacted_bytes
                ).hexdigest(),
                "redacted_image_bytes": self.redacted_bytes,
            }
        )
        if self.residual_report is not None:
            metadata["residual_report"] = self.residual_report.to_dict()
        if self.output_path is not None:
            metadata["output_suffix"] = self.output_path.suffix.lower()
        return ExtractedDocument(
            text=self.extracted_document.text,
            spans=self.extracted_document.spans,
            metadata=metadata,
        )


def redact_image(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    policy: Any | None = None,
    models: Any | None = None,
    lang: str | None = None,
    ocr_engine: str | OcrEngine | None = None,
    verification_ocr_engine: str | OcrEngine | None = None,
    box_color: tuple[int, int, int] = (0, 0, 0),
    verify: bool | None = None,
) -> RedactedImage:
    """Redact burned-in PHI from a PNG, JPEG, or TIFF image.

    Args:
        path: Source raster image.
        output_path: Optional destination for the redacted image. When omitted,
            the redacted bytes are returned without writing a file.
        policy: Optional policy marker copied into result metadata.
        models: Supplied OpenMed PII model object or mapping. Detector callables
            are resolved from keys/attributes such as ``detector``,
            ``extract_pii``, or ``analyze_text``. Optional ``ocr_engine``,
            ``verification_ocr_engine``, and ``verify`` entries are honored.
        lang: Optional OpenMed language code forwarded to OCR and detection.
        ocr_engine: OCR engine name or instance for the first OCR pass.
        verification_ocr_engine: Optional engine for the re-OCR pass.
        box_color: RGB fill for opaque redaction rectangles.
        verify: Whether to re-OCR the redacted output. Defaults to true when a
            detector is available.

    Returns:
        A :class:`RedactedImage` with redacted bytes, pixel-diff counts, boxes,
        and residual verification metadata.
    """

    source = Path(path)
    target = Path(output_path) if output_path is not None else None
    image_module, image_chops, image_draw, image_sequence = _import_pillow()
    detector = _resolve_detector(models)
    first_pass_engine = (
        ocr_engine if ocr_engine is not None else _model_option(models, "ocr_engine")
    )
    verification_engine = (
        verification_ocr_engine
        if verification_ocr_engine is not None
        else _model_option(models, "verification_ocr_engine")
    )
    if verification_engine is None:
        verification_engine = first_pass_engine
    if verify is None:
        verify = bool(_model_option(models, "verify", default=detector is not None))

    languages = [lang] if lang else None
    with image_module.open(source) as image:
        source_format = _image_format(image, source)
        output_format = _output_format(target or source, source_format)
        frames = [frame.copy() for frame in image_sequence.Iterator(image)]

    if len(frames) > 1 and output_format != "TIFF":
        raise ValueError("multi-frame images can only be redacted to TIFF output")

    frame_results = tuple(
        ocr(frame, engine=first_pass_engine, languages=languages) for frame in frames
    )
    document = _document_from_frame_results(
        frame_results,
        image_format=output_format,
        frame_count=len(frames),
    )
    if policy is not None:
        document = ExtractedDocument(
            text=document.text,
            spans=document.spans,
            metadata={**dict(document.metadata), "policy": policy},
        )

    entities = _iter_entities(_detect_entities(document, detector, lang))
    boxes = project_text_spans(document, entities)
    redacted_frames = [
        _prepare_frame_for_output(frame, output_format) for frame in frames
    ]
    _draw_boxes(redacted_frames, boxes, box_color=box_color, image_draw=image_draw)
    changed_by_frame = tuple(
        _changed_pixel_count(original, redacted, image_chops=image_chops)
        for original, redacted in zip(frames, redacted_frames)
    )
    redacted_bytes = _encode_frames(redacted_frames, output_format)
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(redacted_bytes)

    metadata_report = verify_image_metadata(redacted_bytes)
    residual_report = None
    if verify:
        residual_report = verify_image_redaction(
            redacted_bytes,
            models=models,
            lang=lang,
            ocr_engine=verification_engine,
        )
        assert_no_residual_phi(residual_report)

    return RedactedImage(
        source_path=source,
        output_path=target,
        redacted_bytes=redacted_bytes,
        extracted_document=document,
        redaction_boxes=boxes,
        frame_count=len(frames),
        image_format=output_format,
        changed_pixel_count=sum(changed_by_frame),
        changed_pixels_by_frame=changed_by_frame,
        metadata_report=metadata_report,
        residual_report=residual_report,
    )


def verify_image_redaction(
    image: bytes | str | Path | Any,
    *,
    models: Any | None,
    lang: str | None = None,
    ocr_engine: str | OcrEngine | None = None,
) -> ResidualPhiReport:
    """Re-OCR a redacted image and report any detected residual PHI."""

    image_module, _, _, image_sequence = _import_pillow()
    detector = _resolve_detector(models)
    if detector is None:
        return ResidualPhiReport()

    languages = [lang] if lang else None
    with _open_image(image, image_module) as opened:
        image_format = _image_format_from_loaded(opened)
        frames = [frame.copy() for frame in image_sequence.Iterator(opened)]

    frame_results = tuple(
        ocr(frame, engine=ocr_engine, languages=languages) for frame in frames
    )
    document = _document_from_frame_results(
        frame_results,
        image_format=image_format,
        frame_count=len(frames),
    )
    entities = _iter_entities(_detect_entities(document, detector, lang))
    residuals = tuple(_residual_from_entity(document, entity) for entity in entities)
    return ResidualPhiReport(
        residual_phi=tuple(residual for residual in residuals if residual is not None),
        frame_count=len(frames),
        word_count=len(document.spans),
    )


def assert_no_residual_phi(report: ResidualPhiReport) -> ResidualPhiReport:
    """Raise if a residual PHI verification report is not clean."""

    if not report.clean:
        raise ImageRedactionVerificationError(report)
    return report


def verify_image_metadata(image: bytes | str | Path | Any) -> ImageMetadataReport:
    """Return PHI-bearing image metadata keys still present on an image."""

    image_module, _, _, _ = _import_pillow()
    with _open_image(image, image_module) as opened:
        info = dict(getattr(opened, "info", {}))
    checked = tuple(sorted(str(key).lower() for key in info))
    residual = tuple(key for key in checked if key in _PHI_METADATA_KEYS)
    return ImageMetadataReport(residual_keys=residual, checked_keys=checked)


def _import_pillow() -> tuple[Any, Any, Any, Any]:
    try:
        return (
            importlib.import_module("PIL.Image"),
            importlib.import_module("PIL.ImageChops"),
            importlib.import_module("PIL.ImageDraw"),
            importlib.import_module("PIL.ImageSequence"),
        )
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="Pillow", instruction=_PILLOW_HINT
        ) from exc


def _open_image(image: bytes | str | Path | Any, image_module: Any) -> Any:
    if isinstance(image, bytes):
        return image_module.open(BytesIO(image))
    return image_module.open(image)


def _image_format(image: Any, path: Path) -> str:
    return _normalize_image_format(getattr(image, "format", None), path)


def _image_format_from_loaded(image: Any) -> str:
    return _normalize_image_format(getattr(image, "format", None), Path("image.png"))


def _output_format(path: Path, fallback: str) -> str:
    return _normalize_image_format(None, path, fallback=fallback)


def _normalize_image_format(
    image_format: str | None,
    path: Path,
    *,
    fallback: str | None = None,
) -> str:
    normalized = str(image_format or "").upper()
    if normalized == "MPO":
        normalized = "JPEG"
    if normalized in {"PNG", "JPEG", "TIFF"}:
        return normalized

    suffix = path.suffix.lower()
    if suffix == ".png":
        return "PNG"
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if suffix in {".tif", ".tiff"}:
        return "TIFF"
    if fallback is not None:
        return fallback
    raise UnsupportedDocumentError(
        f"No image redaction handler registered for extension {suffix or '(none)'!r}."
    )


def _document_from_frame_results(
    results: Sequence[OcrResult],
    *,
    image_format: str,
    frame_count: int,
) -> ExtractedDocument:
    parts: list[str] = []
    spans: list[SourceSpan] = []
    cursor = 0
    word_count = 0
    for frame_index, result in enumerate(results):
        words = tuple(result.words)
        if frame_index > 0 and parts and words:
            parts.append("\n")
            cursor += 1
        for word_index, word in enumerate(words):
            if parts and not parts[-1].endswith("\n"):
                parts.append(" ")
                cursor += 1
            text = str(word.text)
            start = cursor
            parts.append(text)
            cursor += len(text)
            spans.append(
                SourceSpan(
                    start=start,
                    end=cursor,
                    page=frame_index,
                    bbox=word.bbox,
                    metadata={
                        "format": "image",
                        "block_type": "ocr_word",
                        "confidence": word.confidence,
                        "frame_word_index": word_index,
                        "document_word_index": word_count,
                    },
                )
            )
            word_count += 1
    return ExtractedDocument(
        text="".join(parts),
        spans=tuple(spans),
        metadata={
            "format": "image",
            "image_format": image_format,
            "frame_count": frame_count,
            "word_count": word_count,
        },
    )


def _detect_entities(
    document: ExtractedDocument,
    detector: Any,
    lang: str | None,
) -> Any:
    if detector is None:
        return ()
    try:
        return detector(document.text, lang=lang)
    except TypeError:
        return detector(document.text)


def _resolve_detector(models: Any) -> Any:
    if models is None:
        return None
    if callable(models):
        return models
    if isinstance(models, Mapping):
        for key in ("detector", "extract_pii", "analyze_text", "predict_entities"):
            candidate = models.get(key)
            if callable(candidate):
                return candidate
        return None
    for name in (
        "detect",
        "extract_pii",
        "analyze_text",
        "predict_entities",
        "predict",
    ):
        candidate = getattr(models, name, None)
        if callable(candidate):
            return candidate
    return None


def _iter_entities(result: Any) -> tuple[Any, ...]:
    if result is None:
        return ()
    entities = getattr(result, "entities", None)
    if entities is not None:
        return tuple(entities)
    pii_entities = getattr(result, "pii_entities", None)
    if pii_entities is not None:
        return tuple(pii_entities)
    if isinstance(result, Mapping):
        for key in ("entities", "pii_entities", "spans"):
            entities = result.get(key)
            if entities is not None:
                return tuple(entities)
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, bytearray)):
        return tuple(result)
    return ()


def _model_option(models: Any, name: str, default: Any = None) -> Any:
    if isinstance(models, Mapping):
        return models.get(name, default)
    return getattr(models, name, default)


def _prepare_frame_for_output(frame: Any, image_format: str) -> Any:
    if image_format == "JPEG":
        if frame.mode not in {"L", "RGB", "CMYK"}:
            frame = frame.convert("RGB")
    elif frame.mode not in {"RGB", "RGBA", "L"}:
        frame = frame.convert("RGBA" if "A" in frame.getbands() else "RGB")
    else:
        frame = frame.copy()
    frame.info.clear()
    return frame


def _draw_boxes(
    frames: Sequence[Any],
    boxes: Sequence[ProjectedRectangle],
    *,
    box_color: tuple[int, int, int],
    image_draw: Any,
) -> None:
    for box in boxes:
        if box.page < 0 or box.page >= len(frames):
            continue
        frame = frames[box.page]
        rectangle = _clip_bbox(box.bbox, frame.size)
        if rectangle is None:
            continue
        draw = image_draw.Draw(frame)
        draw.rectangle(rectangle, fill=_fill_for_mode(box_color, frame.mode))


def _clip_bbox(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    width, height = size
    x0, y0, x1, y1 = bbox
    left = max(0, min(width, math.floor(x0)))
    top = max(0, min(height, math.floor(y0)))
    right = max(0, min(width, math.ceil(x1)))
    bottom = max(0, min(height, math.ceil(y1)))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _fill_for_mode(box_color: tuple[int, int, int], mode: str) -> Any:
    if mode == "RGBA":
        return (*box_color, 255)
    if mode == "L":
        return 0
    return box_color


def _changed_pixel_count(original: Any, redacted: Any, *, image_chops: Any) -> int:
    left = original.convert("RGB")
    right = redacted.convert("RGB")
    diff = image_chops.difference(left, right)
    if diff.getbbox() is None:
        return 0
    pixels = (
        diff.get_flattened_data()
        if hasattr(diff, "get_flattened_data")
        else diff.getdata()
    )
    return sum(1 for pixel in pixels if pixel != (0, 0, 0))


def _encode_frames(frames: Sequence[Any], image_format: str) -> bytes:
    buffer = BytesIO()
    first, *rest = [frame.copy() for frame in frames]
    for frame in (first, *rest):
        frame.info.clear()
    if image_format == "JPEG" and first.mode not in {"L", "RGB", "CMYK"}:
        first = first.convert("RGB")
    if image_format == "TIFF" and rest:
        first.save(
            buffer,
            format=image_format,
            save_all=True,
            append_images=rest,
        )
    else:
        first.save(buffer, format=image_format)
    return buffer.getvalue()


def _residual_from_entity(
    document: ExtractedDocument,
    entity: Any,
) -> ResidualPhi | None:
    coerced = _coerce_entity(entity)
    if coerced is None:
        return None
    start, end, label, confidence = coerced
    if end <= start:
        return None
    text = document.text[start:end]
    location = document.location_at(start)
    return ResidualPhi(
        start=start,
        end=end,
        page=location.page if location is not None else None,
        label=label,
        confidence=confidence,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _coerce_entity(
    span: Any,
) -> tuple[int, int, str | None, float | None] | None:
    if isinstance(span, Sequence) and not isinstance(span, (str, bytes, bytearray)):
        if len(span) >= 2:
            return int(span[0]), int(span[1]), None, None
        return None

    if isinstance(span, Mapping):
        start = span.get("start")
        end = span.get("end")
        label = span.get("label", span.get("entity_type"))
        confidence = span.get("confidence", span.get("score"))
    else:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        label = getattr(span, "label", getattr(span, "entity_type", None))
        confidence = getattr(span, "confidence", getattr(span, "score", None))

    if start is None or end is None:
        return None
    return (
        int(start),
        int(end),
        _coerce_optional_str(label),
        _coerce_confidence(confidence),
    )


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _image_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    detector = _resolve_detector(models)
    if detector is None:
        languages = [lang] if lang else None
        return ocr(
            path,
            engine=_model_option(models, "ocr_engine"),
            languages=languages,
        ).to_document()

    result = redact_image(path, policy=policy, models=models, lang=lang)
    return result.to_document()


register_handler(_IMAGE_EXTENSIONS, _image_handler)


__all__ = [
    "ImageMetadataReport",
    "ImageRedactionVerificationError",
    "RedactedImage",
    "ResidualPhi",
    "ResidualPhiReport",
    "assert_no_residual_phi",
    "redact_image",
    "verify_image_metadata",
    "verify_image_redaction",
]
