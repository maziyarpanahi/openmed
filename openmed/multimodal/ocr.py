"""OCR contract and swappable engine adapters for the multimodal subsystem.

Provides one OCR result shape (per-word text + bbox + confidence + page) with
interchangeable backends (docTR, Tesseract, EasyOCR, PaddleOCR) plus a deterministic
in-memory fake engine for tests. OCR output bridges into
:class:`ExtractedDocument` so scanned/image text flows through
``redact_document`` and detected PHI projects back to source pixel bounding
boxes.

Like the rest of the package, this module imports no heavy dependency at module
load time: each engine imports its backend lazily and raises a clear,
actionable error when the dependency (or the system Tesseract binary) is
missing.

Language packs: ``ocr(..., languages=[...])`` selects OCR languages by OpenMed
PII language code (en, fr, de, it, es, nl, hi, te, pt, ar, ja, tr) and maps them
to each backend's identifiers. The language data itself is not bundled and must
be installed separately: Tesseract needs the matching ``traineddata`` files
(e.g. ``apt-get install tesseract-ocr-fra`` or the language pack for your OS),
EasyOCR downloads its detection/recognition models on first use, and PaddleOCR
downloads the recognition model for the requested language on first use.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .base import ExtractedDocument, register_handler
from .exceptions import MissingDependencyError


@dataclass(frozen=True)
class OcrWord:
    """A single recognized word with its pixel location and confidence."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float
    page: int = 0


@dataclass(frozen=True)
class OcrResult:
    """The common OCR contract shared by every engine adapter."""

    words: tuple[OcrWord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The recognized words joined into a single normalized string."""
        return " ".join(word.text for word in self.words)

    def to_document(self, *, separator: str = " ") -> ExtractedDocument:
        """Bridge OCR words into an :class:`ExtractedDocument`.

        Each word becomes a block so its ``SourceSpan`` carries the source pixel
        bbox and page, letting downstream redaction project a detected PHI
        offset back to its location in the image.
        """
        blocks = [
            {
                "text": word.text,
                "page": word.page,
                "bbox": word.bbox,
                "metadata": {"confidence": word.confidence},
            }
            for word in self.words
        ]
        return ExtractedDocument.from_blocks(
            blocks, separator=separator, metadata=dict(self.metadata)
        )


@runtime_checkable
class OcrEngine(Protocol):
    """Structural contract every OCR backend implements."""

    name: str

    def recognize(
        self, image: Any, *, languages: Sequence[str] | None = None
    ) -> OcrResult: ...


class FakeOcrEngine:
    """Deterministic in-memory engine for tests; returns fixed words.

    Records the ``languages`` of the most recent call as ``last_languages`` so
    tests can assert that a language selection reaches the adapter.
    """

    name = "fake"

    def __init__(self, words: Iterable[OcrWord], **metadata: Any) -> None:
        self._words = tuple(words)
        self._metadata = {"engine": "fake", **metadata}
        self.last_languages: list[str] | None = None

    def recognize(
        self, image: Any, *, languages: Sequence[str] | None = None
    ) -> OcrResult:
        self.last_languages = list(languages) if languages is not None else None
        metadata = {**self._metadata, "languages": self.last_languages}
        return OcrResult(words=self._words, metadata=metadata)


def _import_backend(module: str, instruction: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised when extra absent
        raise MissingDependencyError(
            dependency=module, instruction=instruction
        ) from exc


_TESSERACT_HINT = (
    'Install with: pip install "openmed[multimodal]" and install the system '
    "Tesseract binary (e.g. `brew install tesseract` or `apt-get install "
    "tesseract-ocr`)."
)
_DOCTR_HINT = 'Install with: pip install "openmed[multimodal]".'
_PADDLE_HINT = 'Install with: pip install "openmed[ocr-paddle]".'
_EASYOCR_HINT = 'Install with: pip install "openmed[multimodal]".'


# --- language mapping -------------------------------------------------------

DEFAULT_OCR_LANGUAGE = "en"

# OpenMed PII language code -> Tesseract traineddata code (ISO 639-2/T).
_TESSERACT_LANGUAGES: dict[str, str] = {
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "es": "spa",
    "nl": "nld",
    "hi": "hin",
    "te": "tel",
    "pt": "por",
    "ar": "ara",
    "ja": "jpn",
    "tr": "tur",
}

# OpenMed PII language code -> PaddleOCR ``lang`` identifier.
_PADDLE_LANGUAGES: dict[str, str] = {
    "en": "en",
    "fr": "fr",
    "de": "german",
    "it": "it",
    "es": "es",
    "nl": "nl",
    "hi": "hi",
    "te": "te",
    "pt": "pt",
    "ar": "ar",
    "ja": "japan",
    "tr": "tr",
}

# OpenMed PII language code -> EasyOCR language identifier.
_EASYOCR_LANGUAGES: dict[str, str] = {
    "en": "en",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "es": "es",
    "nl": "nl",
    "hi": "hi",
    "te": "te",
    "pt": "pt",
    "ar": "ar",
    "ja": "ja",
    "tr": "tr",
}

# The OpenMed PII languages with OCR-engine coverage.
SUPPORTED_OCR_LANGUAGES: tuple[str, ...] = tuple(_TESSERACT_LANGUAGES)


def _normalize_languages(languages: str | Sequence[str] | None) -> list[str]:
    """Normalize a language selection to a list of OpenMed codes (English default)."""
    if languages is None:
        return [DEFAULT_OCR_LANGUAGE]
    if isinstance(languages, str):
        languages = [languages]
    normalized = [str(code).strip().lower() for code in languages if str(code).strip()]
    return normalized or [DEFAULT_OCR_LANGUAGE]


def _lookup_language(mapping: Mapping[str, str], code: str) -> str:
    try:
        return mapping[code]
    except KeyError:
        raise ValueError(
            f"Unsupported OCR language {code!r}. "
            f"Supported OpenMed language codes: {', '.join(SUPPORTED_OCR_LANGUAGES)}."
        ) from None


def tesseract_language(languages: str | Sequence[str] | None = None) -> str:
    """Map OpenMed language code(s) to a Tesseract ``lang`` string (``eng+fra``)."""
    codes = _normalize_languages(languages)
    return "+".join(_lookup_language(_TESSERACT_LANGUAGES, code) for code in codes)


def paddle_language(languages: str | Sequence[str] | None = None) -> str:
    """Map OpenMed language code(s) to a PaddleOCR ``lang`` identifier.

    PaddleOCR loads a single recognition language per instance, so the first
    requested language is used.
    """
    codes = _normalize_languages(languages)
    return _lookup_language(_PADDLE_LANGUAGES, codes[0])


def easyocr_languages(languages: str | Sequence[str] | None = None) -> list[str]:
    """Map OpenMed language code(s) to EasyOCR language identifiers."""
    codes = _normalize_languages(languages)
    return [_lookup_language(_EASYOCR_LANGUAGES, code) for code in codes]


DocTrPredictorFactory = Callable[..., Callable[[Any], Any]]
DocTrDocumentLoader = Callable[[Any], Any]


class DocTrEngine:
    """OCR backend backed by python-doctr."""

    name = "doctr"

    def __init__(
        self,
        *,
        predictor_factory: DocTrPredictorFactory | None = None,
        document_loader: DocTrDocumentLoader | None = None,
    ) -> None:
        self._predictor_factory = predictor_factory
        self._document_loader = document_loader

    def recognize(
        self, image: Any, *, languages: Sequence[str] | None = None
    ) -> OcrResult:
        return run_doctr_ocr(
            image,
            predictor_factory=self._predictor_factory,
            document_loader=self._document_loader,
        )


class TesseractEngine:
    """OCR backend backed by pytesseract / the Tesseract binary."""

    name = "tesseract"

    def recognize(
        self, image: Any, *, languages: Sequence[str] | None = None
    ) -> OcrResult:
        pytesseract = _import_backend("pytesseract", _TESSERACT_HINT)
        loaded = _load_tesseract_image(image)
        try:
            data = pytesseract.image_to_data(
                loaded,
                lang=tesseract_language(languages),
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            tesseract_error = getattr(
                getattr(pytesseract, "pytesseract", None),
                "TesseractNotFoundError",
                None,
            )
            if tesseract_error is not None and isinstance(exc, tesseract_error):
                raise MissingDependencyError(
                    dependency="tesseract", instruction=_TESSERACT_HINT
                ) from exc
            raise

        words: list[OcrWord] = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            left, top = float(data["left"][i]), float(data["top"][i])
            width, height = float(data["width"][i]), float(data["height"][i])
            conf = float(data["conf"][i])
            page_nums = data.get("page_num")
            words.append(
                OcrWord(
                    text=text,
                    bbox=(left, top, left + width, top + height),
                    confidence=max(conf, 0.0) / 100.0,
                    page=_zero_based_page(page_nums[i]) if page_nums else 0,
                )
            )
        return OcrResult(words=tuple(words), metadata={"engine": self.name})


class EasyOcrEngine:
    """OCR backend backed by EasyOCR."""

    name = "easyocr"

    def recognize(
        self, image: Any, *, languages: Sequence[str] | None = None
    ) -> OcrResult:
        easyocr = _import_backend("easyocr", _EASYOCR_HINT)
        engine = easyocr.Reader(easyocr_languages(languages))
        predictions = engine.readtext(_load_easyocr_image(image))

        words: list[OcrWord] = []
        for box, text, confidence in _iter_easyocr_predictions(predictions):
            bbox = _polygon_bbox(box)
            for word in _split_detected_text(str(text), bbox):
                words.append(
                    OcrWord(
                        text=word[0],
                        bbox=word[1],
                        confidence=float(confidence),
                        page=0,
                    )
                )
        return OcrResult(words=tuple(words), metadata={"engine": self.name})


class PaddleOcrEngine:
    """OCR backend backed by PaddleOCR."""

    name = "paddleocr"

    def recognize(
        self, image: Any, *, languages: Sequence[str] | None = None
    ) -> OcrResult:
        paddleocr = _import_backend("paddleocr", _PADDLE_HINT)
        engine = paddleocr.PaddleOCR(show_log=False, lang=paddle_language(languages))
        loaded = _load_paddle_image(image)
        predictions = engine.ocr(loaded)
        words: list[OcrWord] = []
        for page_index, page in _iter_paddle_pages(predictions):
            for box, (text, confidence) in page or []:
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]
                words.append(
                    OcrWord(
                        text=str(text),
                        bbox=(min(xs), min(ys), max(xs), max(ys)),
                        confidence=float(confidence),
                        page=page_index,
                    )
                )
        return OcrResult(words=tuple(words), metadata={"engine": self.name})


def run_doctr_ocr(
    image: Any,
    *,
    predictor_factory: DocTrPredictorFactory | None = None,
    document_loader: DocTrDocumentLoader | None = None,
) -> OcrResult:
    """Run the docTR OCR adapter and return the shared OCR result contract."""
    factory = predictor_factory or _load_doctr_predictor()
    loader = document_loader or _load_doctr_document
    predictor = factory(pretrained=True)
    document = loader(image)
    return _result_from_doctr_document(predictor(document))


def _load_doctr_predictor() -> DocTrPredictorFactory:
    try:
        models = importlib.import_module("doctr.models")
    except ImportError as exc:  # pragma: no cover - exercised when extra absent
        raise _missing_doctr_dependency() from exc
    return models.ocr_predictor


def _load_doctr_document(image: Any) -> Any:
    try:
        document_file = importlib.import_module("doctr.io").DocumentFile
    except ImportError as exc:  # pragma: no cover - exercised when extra absent
        raise _missing_doctr_dependency() from exc
    normalized = _normalize_doctr_image_input(image)
    if isinstance(normalized, list):
        return document_file.from_images(normalized)
    return normalized


def _missing_doctr_dependency() -> MissingDependencyError:
    return MissingDependencyError(dependency="python-doctr", instruction=_DOCTR_HINT)


def _normalize_doctr_image_input(image: Any) -> Any:
    if isinstance(image, (str, Path)):
        return [str(image)]
    if isinstance(image, (list, tuple)):
        return [str(item) if isinstance(item, Path) else item for item in image]
    return image


def _result_from_doctr_document(document: Any) -> OcrResult:
    words: list[OcrWord] = []
    for page_index, page in enumerate(document.pages):
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    words.append(
                        OcrWord(
                            text=str(word.value),
                            bbox=_absolute_doctr_bbox(word.geometry, page.dimensions),
                            confidence=float(word.confidence),
                            page=page_index,
                        )
                    )
    return OcrResult(words=tuple(words), metadata={"engine": DocTrEngine.name})


def _absolute_doctr_bbox(
    geometry: tuple[tuple[float, float], tuple[float, float]],
    dimensions: tuple[int | float, int | float],
) -> tuple[float, float, float, float]:
    (rel_xmin, rel_ymin), (rel_xmax, rel_ymax) = geometry
    page_height, page_width = dimensions
    return (
        float(rel_xmin) * float(page_width),
        float(rel_ymin) * float(page_height),
        float(rel_xmax) * float(page_width),
        float(rel_ymax) * float(page_height),
    )


def _load_tesseract_image(image: Any) -> Any:
    """Load a path/bytes into a PIL image for pytesseract; pass through others."""
    if isinstance(image, (str, Path)):
        PIL_Image = _import_backend("PIL.Image", _TESSERACT_HINT)
        return PIL_Image.open(image)
    return image


def _load_paddle_image(image: Any) -> Any:
    """Normalize pathlib paths for PaddleOCR without forcing Pillow."""
    if isinstance(image, Path):
        return str(image)
    return image


def _load_easyocr_image(image: Any) -> Any:
    """Normalize pathlib paths for EasyOCR without forcing Pillow."""
    if isinstance(image, Path):
        return str(image)
    return image


def _zero_based_page(page_num: Any) -> int:
    """Convert OCR backend page numbers to the SourceSpan zero-based contract."""
    return max(int(page_num) - 1, 0)


def _looks_like_paddle_detection(item: Any) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return False
    box, text_confidence = item
    if not isinstance(box, (list, tuple)) or not box:
        return False
    first_point = box[0]
    return (
        isinstance(first_point, (list, tuple))
        and len(first_point) >= 2
        and isinstance(text_confidence, (list, tuple))
        and len(text_confidence) >= 2
    )


def _iter_paddle_pages(predictions: Any) -> Iterable[tuple[int, Any]]:
    """Yield page-indexed PaddleOCR detections for common v2 output shapes."""
    if not predictions:
        return ()
    if isinstance(predictions, (list, tuple)) and _looks_like_paddle_detection(
        predictions[0]
    ):
        return ((0, predictions),)
    return enumerate(predictions)


def _iter_easyocr_predictions(predictions: Any) -> Iterable[tuple[Any, Any, Any]]:
    """Yield EasyOCR ``(polygon, text, confidence)`` triples."""
    if not predictions:
        return ()
    return (
        (prediction[0], prediction[1], prediction[2])
        for prediction in predictions
        if isinstance(prediction, (list, tuple)) and len(prediction) >= 3
    )


def _polygon_bbox(
    polygon: Sequence[Sequence[Any]],
) -> tuple[float, float, float, float]:
    """Convert an OCR polygon into the contract's axis-aligned pixel bbox."""
    points = [(float(point[0]), float(point[1])) for point in polygon]
    if not points:
        raise ValueError("OCR polygon must contain at least one point.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _split_detected_text(
    text: str, bbox: tuple[float, float, float, float]
) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    """Split a detected text run into word boxes using horizontal proportions."""
    words = text.split()
    if not words:
        return ()
    if len(words) == 1:
        return ((words[0], bbox),)

    x0, y0, x1, y1 = bbox
    total = sum(len(word) for word in words)
    width = x1 - x0
    cursor = x0
    split_words: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, word in enumerate(words):
        if index == len(words) - 1:
            word_x1 = x1
        else:
            word_x1 = cursor + (width * len(word) / total)
        split_words.append((word, (cursor, y0, word_x1, y1)))
        cursor = word_x1
    return tuple(split_words)


EngineFactory = Callable[[], OcrEngine]

_ENGINES: dict[str, EngineFactory] = {
    "doctr": DocTrEngine,
    "tesseract": TesseractEngine,
    "easyocr": EasyOcrEngine,
    "paddleocr": PaddleOcrEngine,
}

# Backend import module per engine, used for availability-based auto-selection.
_ENGINE_MODULES = {
    "doctr": "doctr",
    "tesseract": "pytesseract",
    "easyocr": "easyocr",
    "paddleocr": "paddleocr",
}

# Auto-selection priority when no engine is requested.
_AUTO_ORDER = ("doctr", "tesseract", "easyocr", "paddleocr")


def register_ocr_engine(name: str, factory: EngineFactory) -> None:
    """Register an OCR engine factory under ``name``."""
    _ENGINES[name] = factory


def available_ocr_engines() -> tuple[str, ...]:
    """Return installed OCR engines in the default auto-detection order."""
    return tuple(name for name in _AUTO_ORDER if _engine_available(name))


def _engine_available(name: str) -> bool:
    module = _ENGINE_MODULES.get(name)
    return module is not None and importlib.util.find_spec(module) is not None


def resolve_engine(engine: str | OcrEngine | None = None) -> OcrEngine:
    """Resolve ``engine`` (name, instance, or ``None`` for auto-select)."""
    if isinstance(engine, OcrEngine) and not isinstance(engine, str):
        return engine
    if isinstance(engine, str):
        factory = _ENGINES.get(engine)
        if factory is None:
            raise ValueError(
                f"Unknown OCR engine {engine!r}. "
                f"Available: {', '.join(sorted(_ENGINES))}."
            )
        return factory()
    # Auto-select the first installed engine.
    for name in _AUTO_ORDER:
        if _engine_available(name):
            return _ENGINES[name]()
    raise MissingDependencyError(
        dependency="python-doctr, pytesseract, easyocr, or paddleocr",
        instruction=(
            "No OCR engine is installed. Install docTR, Tesseract, or EasyOCR via "
            'pip install "openmed[multimodal]" (Tesseract also needs its '
            'system binary), or PaddleOCR via pip install "openmed[ocr-paddle]".'
        ),
    )


def ocr(
    image: Any,
    *,
    engine: str | OcrEngine | None = None,
    languages: str | Sequence[str] | None = None,
) -> OcrResult:
    """Run OCR on ``image`` and return an :class:`OcrResult`.

    ``engine`` may be an engine name, an :class:`OcrEngine` instance, or
    ``None`` to auto-select the first installed backend. ``languages`` selects
    the OCR languages by OpenMed PII language code (e.g. ``["fr"]``); each
    adapter maps them to its own identifiers. Defaults to English.
    """
    return resolve_engine(engine).recognize(
        image, languages=_normalize_languages(languages)
    )


# --- redact_document bridge -------------------------------------------------

_IMAGE_EXTENSIONS = (
    ".bmp",
    ".gif",
    ".webp",
)


def _ocr_image_handler(
    path: Any, *, policy: Any = None, models: Any = None, lang: str | None = None
) -> ExtractedDocument:
    """redact_document handler for image files: OCR then bridge to a document.

    ``lang`` (an OpenMed language code from ``redact_document(lang=...)``) is
    forwarded to OCR so scans are read in the requested language.
    """
    languages = [lang] if lang else None
    return ocr(path, languages=languages).to_document()


register_handler(_IMAGE_EXTENSIONS, _ocr_image_handler)
