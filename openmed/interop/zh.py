"""Lazy adapters for optional Chinese text dependencies."""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any

_MISSING_EXTRA_MESSAGE = (
    "Chinese segmentation requires the 'zh' extra. "
    "Install with `pip install openmed[zh]`."
)


def segment(
    text: str, *, cut_all: bool = False, use_hmm: bool = True
) -> tuple[str, ...]:
    """Segment Chinese text with jieba without loading it during core import.

    Args:
        text: Text to segment.
        cut_all: Whether jieba should return every possible word match.
        use_hmm: Whether jieba may use its hidden Markov model for unknown words.

    Returns:
        Segmented tokens in source order.

    Raises:
        ImportError: If the ``zh`` extra is not installed.
    """

    jieba = _load_dependency("jieba")
    return tuple(jieba.lcut(text, cut_all=cut_all, HMM=use_hmm))


def convert_script(text: str, *, config: str = "s2t") -> str:
    """Convert Chinese text between OpenCC-supported script variants.

    Args:
        text: Text to convert.
        config: OpenCC conversion configuration, such as ``s2t`` or ``t2s``.

    Returns:
        Converted text.

    Raises:
        ImportError: If the ``zh`` extra is not installed.
    """

    opencc = _load_dependency("opencc")
    return str(opencc.OpenCC(config).convert(text))


def to_pinyin(text: str) -> tuple[str, ...]:
    """Return dependency-default pinyin syllables for Chinese text.

    Args:
        text: Text to transliterate.

    Returns:
        Pinyin syllables in source order.

    Raises:
        ImportError: If the ``zh`` extra is not installed.
    """

    pypinyin = _load_dependency("pypinyin")
    return tuple(str(syllable) for syllable in pypinyin.lazy_pinyin(text))


def _load_dependency(name: str) -> Any:
    try:
        return _import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(_MISSING_EXTRA_MESSAGE) from exc


__all__ = ["convert_script", "segment", "to_pinyin"]
