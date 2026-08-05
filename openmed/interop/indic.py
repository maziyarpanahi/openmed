"""Lazy adapters for optional Indic text dependencies."""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any

_MISSING_EXTRA_MESSAGE = (
    "Indic segmentation requires the 'indic' extra. "
    "Install with `pip install openmed[indic]`."
)


def segment(text: str, *, language: str = "hi") -> tuple[str, ...]:
    """Segment Indic text without loading indic-nlp-library during core import.

    Args:
        text: Text to segment.
        language: Indic NLP Library language code.

    Returns:
        Segmented tokens in source order.

    Raises:
        ImportError: If the ``indic`` extra is not installed.
    """

    tokenizer = _load_dependency("indicnlp.tokenize.indic_tokenize")
    return tuple(tokenizer.trivial_tokenize(text, lang=language))


def transliterate(text: str, *, source: str, target: str) -> str:
    """Transliterate text between Indic NLP Library script codes.

    Args:
        text: Text to transliterate.
        source: Source script or language code.
        target: Target script or language code.

    Returns:
        Transliterated text.

    Raises:
        ImportError: If the ``indic`` extra is not installed.
    """

    module = _load_dependency("indicnlp.transliterate.unicode_transliterate")
    transliterator = module.UnicodeIndicTransliterator
    return str(transliterator.transliterate(text, source, target))


def _load_dependency(name: str) -> Any:
    try:
        return _import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(_MISSING_EXTRA_MESSAGE) from exc


__all__ = ["segment", "transliterate"]
