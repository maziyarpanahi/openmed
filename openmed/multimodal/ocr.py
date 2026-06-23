"""OCR contract and swappable engine adapters for the multimodal subsystem.

Provides one OCR result shape (per-word text + bbox + confidence + page) with
interchangeable backends (Tesseract, PaddleOCR) plus a deterministic in-memory
fake engine for tests. OCR output bridges into :class:`ExtractedDocument` so
scanned/image text flows through ``redact_document`` and detected PHI projects
back to source pixel bounding boxes.

Like the rest of the package, this module imports no heavy dependency at module
load time: each engine imports its backend lazily and raises a clear,
actionable error when the dependency (or the system Tesseract binary) is
missing.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, runtime_checkable

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

    def recognize(self, image: Any) -> OcrResult: ...


class FakeOcrEngine:
    """Deterministic in-memory engine for tests; returns fixed words."""

    name = "fake"

    def __init__(self, words: Iterable[OcrWord], **metadata: Any) -> None:
        self._words = tuple(words)
        self._metadata = {"engine": "fake", **metadata}

    def recognize(self, image: Any) -> OcrResult:
        return OcrResult(words=self._words, metadata=dict(self._metadata))


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
_PADDLE_HINT = 'Install with: pip install "openmed[ocr-paddle]".'


class TesseractEngine:
    """OCR backend backed by pytesseract / the Tesseract binary."""

    name = "tesseract"

    def recognize(self, image: Any) -> OcrResult:
        pytesseract = _import_backend("pytesseract", _TESSERACT_HINT)
        loaded = _load_image(image)
        data = pytesseract.image_to_data(loaded, output_type=pytesseract.Output.DICT)
        words: list[OcrWord] = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            left, top = float(data["left"][i]), float(data["top"][i])
            width, height = float(data["width"][i]), float(data["height"][i])
            conf = float(data["conf"][i])
            words.append(
                OcrWord(
                    text=text,
                    bbox=(left, top, left + width, top + height),
                    confidence=max(conf, 0.0) / 100.0,
                    page=int(data.get("page_num", [0])[i]) if "page_num" in data else 0,
                )
            )
        return OcrResult(words=tuple(words), metadata={"engine": self.name})


class PaddleOcrEngine:
    """OCR backend backed by PaddleOCR."""

    name = "paddleocr"

    def recognize(self, image: Any) -> OcrResult:
        paddleocr = _import_backend("paddleocr", _PADDLE_HINT)
        engine = paddleocr.PaddleOCR(show_log=False)
        loaded = _load_image(image)
        predictions = engine.ocr(loaded)
        words: list[OcrWord] = []
        for page_index, page in enumerate(predictions or []):
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


def _load_image(image: Any) -> Any:
    """Load a path/bytes into a PIL image; pass through anything else."""
    from pathlib import Path

    if isinstance(image, (str, Path)):
        PIL_Image = _import_backend("PIL.Image", _TESSERACT_HINT)
        return PIL_Image.open(image)
    return image


EngineFactory = Callable[[], OcrEngine]

_ENGINES: dict[str, EngineFactory] = {
    "tesseract": TesseractEngine,
    "paddleocr": PaddleOcrEngine,
}

# Backend import module per engine, used for availability-based auto-selection.
_ENGINE_MODULES = {"tesseract": "pytesseract", "paddleocr": "paddleocr"}

# Auto-selection priority when no engine is requested.
_AUTO_ORDER = ("tesseract", "paddleocr")


def register_ocr_engine(name: str, factory: EngineFactory) -> None:
    """Register an OCR engine factory under ``name``."""
    _ENGINES[name] = factory


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
        dependency="pytesseract or paddleocr",
        instruction=(
            "No OCR engine is installed. Install Tesseract via "
            'pip install "openmed[multimodal]" (plus the system Tesseract '
            'binary), or PaddleOCR via pip install "openmed[ocr-paddle]".'
        ),
    )


def ocr(image: Any, *, engine: str | OcrEngine | None = None) -> OcrResult:
    """Run OCR on ``image`` and return an :class:`OcrResult`.

    ``engine`` may be an engine name, an :class:`OcrEngine` instance, or
    ``None`` to auto-select the first installed backend.
    """
    return resolve_engine(engine).recognize(image)


# --- redact_document bridge -------------------------------------------------

_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
)


def _ocr_image_handler(
    path: Any, *, policy: Any = None, models: Any = None
) -> ExtractedDocument:
    """redact_document handler for image files: OCR then bridge to a document."""
    return ocr(path).to_document()


register_handler(_IMAGE_EXTENSIONS, _ocr_image_handler)
