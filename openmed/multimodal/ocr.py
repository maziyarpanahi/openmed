"""OCR engine adapters for multimodal document intake.

The OCR contract is intentionally small: each engine returns one
:class:`OcrResult` per recognized word with normalized text, an absolute pixel
bounding box, a confidence score, and a 0-based page index. Engine dependencies
are imported lazily so importing :mod:`openmed.multimodal.ocr` never downloads
models or requires optional OCR packages to be installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "OcrResult",
    "available_ocr_engines",
    "ocr",
    "run_doctr_ocr",
]

_DOCTR_ENGINE = "doctr"
_AUTO_ENGINE = "auto"
_ENGINE_ORDER = (_DOCTR_ENGINE,)
_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'

ImageInput = str | Path | Sequence[str | Path]
PredictorFactory = Callable[..., Callable[[list[str]], Any]]


@dataclass(frozen=True)
class OcrResult:
    """One OCR word result using absolute pixel coordinates."""

    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    page: int


def available_ocr_engines() -> tuple[str, ...]:
    """Return installed OCR engines in the default auto-detection order."""

    return tuple(engine for engine in _ENGINE_ORDER if _engine_available(engine))


def ocr(image: ImageInput, *, engine: str = _AUTO_ENGINE) -> list[OcrResult]:
    """Run OCR over ``image`` with a selected engine.

    ``engine="auto"`` chooses the first installed engine from the default
    detection order. ``engine="doctr"`` explicitly selects the docTR adapter.
    """

    resolved_engine = _resolve_engine(engine)
    if resolved_engine == _DOCTR_ENGINE:
        return run_doctr_ocr(image)
    raise ValueError(f"Unsupported OCR engine: {engine!r}")


def run_doctr_ocr(
    image: ImageInput,
    *,
    predictor_factory: PredictorFactory | None = None,
) -> list[OcrResult]:
    """Run the docTR OCR adapter and return normalized word results."""

    factory = predictor_factory or _load_doctr_predictor()
    predictor = factory(pretrained=True)
    document = predictor(_normalize_image_input(image))
    return _results_from_doctr_document(document)


def _resolve_engine(engine: str) -> str:
    normalized = engine.lower()
    if normalized == _AUTO_ENGINE:
        available = available_ocr_engines()
        if not available:
            raise _missing_doctr_error()
        return available[0]
    if normalized == _DOCTR_ENGINE:
        if not _engine_available(normalized):
            raise _missing_doctr_error()
        return normalized
    raise ValueError(f"Unsupported OCR engine: {engine!r}")


def _engine_available(engine: str) -> bool:
    if engine == _DOCTR_ENGINE:
        return importlib.util.find_spec("doctr") is not None
    return False


def _load_doctr_predictor() -> PredictorFactory:
    try:
        from doctr.models import ocr_predictor
    except ImportError as exc:  # pragma: no cover - covered by explicit guard tests
        raise _missing_doctr_error() from exc
    return ocr_predictor


def _missing_doctr_error() -> ImportError:
    return ImportError(
        "The docTR OCR engine requires the optional 'python-doctr' package. "
        f"{_INSTALL_HINT}"
    )


def _normalize_image_input(image: ImageInput) -> list[str]:
    if isinstance(image, (str, Path)):
        return [str(image)]
    return [str(path) for path in image]


def _results_from_doctr_document(document: Any) -> list[OcrResult]:
    results: list[OcrResult] = []
    for page_index, page in enumerate(document.pages):
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    results.append(
                        OcrResult(
                            text=str(word.value),
                            bbox=_absolute_bbox(word.geometry, page.dimensions),
                            confidence=float(word.confidence),
                            page=page_index,
                        )
                    )
    return results


def _absolute_bbox(
    geometry: tuple[tuple[float, float], tuple[float, float]],
    dimensions: tuple[int, int],
) -> tuple[int, int, int, int]:
    (rel_xmin, rel_ymin), (rel_xmax, rel_ymax) = geometry
    page_height, page_width = dimensions
    return (
        int(round(rel_xmin * page_width)),
        int(round(rel_ymin * page_height)),
        int(round(rel_xmax * page_width)),
        int(round(rel_ymax * page_height)),
    )
