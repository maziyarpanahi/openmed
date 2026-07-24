"""Word-aware Pinyin romanization for Chinese text and name matching.

``pypinyin`` remains an optional dependency and is imported only when Han text
is romanized.  Chinese segmentation happens first so polyphonic characters are
resolved with their word context instead of being transliterated one code point
at a time.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from enum import Enum
from functools import lru_cache
from importlib import import_module as _import_module
from typing import TYPE_CHECKING, TypeAlias, overload

if TYPE_CHECKING:
    from .zh_segmentation import ChineseSegmenter


PINYIN_KEY_VERSION = "zh-pinyin-v1"
"""Version prefix for stable, tone-insensitive Chinese name keys."""

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_MISSING_EXTRA_MESSAGE = (
    "Chinese Pinyin romanization requires the optional 'zh' extra. "
    'Install it with `pip install "openmed[zh]"`.'
)
_PINYIN_NOTICE_EMITTED = False

PinyinReading: TypeAlias = tuple[str, ...]
HeteronymReading: TypeAlias = tuple[tuple[str, ...], ...]


class PinyinStyle(str, Enum):
    """Supported Pinyin tone representations."""

    NORMAL = "normal"
    TONE_MARK = "tone"
    TONE = "tone"
    TONE3 = "tone3"


class PinyinUnavailableError(ImportError):
    """Raised when a caller requests romanization without ``openmed[zh]``."""


class PinyinUnavailableWarning(UserWarning):
    """Warn that a deterministic non-romanized key is being used."""


def _coerce_style(style: PinyinStyle | str) -> PinyinStyle:
    if isinstance(style, PinyinStyle):
        return style
    normalized = str(style).strip().lower().replace("-", "_")
    aliases = {
        "normal": PinyinStyle.NORMAL,
        "tone": PinyinStyle.TONE_MARK,
        "tone_mark": PinyinStyle.TONE_MARK,
        "tonemark": PinyinStyle.TONE_MARK,
        "tone3": PinyinStyle.TONE3,
        "numeric": PinyinStyle.TONE3,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = "NORMAL, TONE_MARK, or TONE3"
        raise ValueError(f"Unsupported Pinyin style {style!r}; use {choices}") from exc


def _load_pypinyin():
    try:
        return _import_module("pypinyin")
    except (ImportError, OSError) as exc:
        raise PinyinUnavailableError(_MISSING_EXTRA_MESSAGE) from exc


def _pypinyin_style(module, style: PinyinStyle):
    return {
        PinyinStyle.NORMAL: module.Style.NORMAL,
        PinyinStyle.TONE_MARK: module.Style.TONE,
        PinyinStyle.TONE3: module.Style.TONE3,
    }[style]


@lru_cache(maxsize=1)
def _default_segmenter() -> ChineseSegmenter:
    from .zh_segmentation import create_chinese_segmenter

    return create_chinese_segmenter()


def _word_tokens(text: str, segmenter: ChineseSegmenter | None) -> tuple[str, ...]:
    if segmenter is None:
        segmenter = _default_segmenter()
    return tuple(token.text for token in segmenter.segment(text))


@overload
def to_pinyin(
    text: str,
    *,
    style: PinyinStyle | str = PinyinStyle.NORMAL,
    heteronym: bool = False,
    segmenter: ChineseSegmenter | None = None,
) -> PinyinReading: ...


@overload
def to_pinyin(
    text: str,
    *,
    style: PinyinStyle | str = PinyinStyle.NORMAL,
    heteronym: bool,
    segmenter: ChineseSegmenter | None = None,
) -> PinyinReading | HeteronymReading: ...


def to_pinyin(
    text: str,
    *,
    style: PinyinStyle | str = PinyinStyle.NORMAL,
    heteronym: bool = False,
    segmenter: ChineseSegmenter | None = None,
) -> PinyinReading | HeteronymReading:
    """Romanize Chinese text after word segmentation.

    Args:
        text: Source text to romanize.
        style: ``NORMAL`` (no tones), ``TONE_MARK`` (diacritics), or ``TONE3``
            (trailing tone numbers).
        heteronym: Return every reading reported for each source character.
        segmenter: Optional Chinese segmenter. The default is the configured
            jieba-backed segmenter from :mod:`openmed.processing.zh_segmentation`.

    Returns:
        A tuple of syllables when ``heteronym`` is false. When true, each tuple
        item contains all readings for the corresponding source character.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: If ``style`` is unsupported.
        PinyinUnavailableError: If the optional ``zh`` extra is unavailable.
    """

    if not isinstance(text, str):
        raise TypeError("Pinyin input must be a string")
    selected_style = _coerce_style(style)
    if not text:
        return ()

    pypinyin = _load_pypinyin()
    pinyin_style = _pypinyin_style(pypinyin, selected_style)
    readings: list[str] | list[tuple[str, ...]] = []
    for word in _word_tokens(text, segmenter):
        word_readings = pypinyin.pinyin(
            word,
            style=pinyin_style,
            heteronym=heteronym,
            strict=False,
        )
        if heteronym:
            readings.extend(
                tuple(dict.fromkeys(str(reading) for reading in alternatives))
                for alternatives in word_readings
            )
        else:
            readings.extend(
                str(alternatives[0]) for alternatives in word_readings if alternatives
            )
    return tuple(readings)


def _fold_romanization(text: str) -> str:
    folded = unicodedata.normalize(
        "NFKD",
        text.casefold().replace("ü", "u").replace("v", "u"),
    )
    return "".join(character for character in folded if "a" <= character <= "z")


def _fallback_han_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    compact = "".join(character for character in normalized if not character.isspace())
    return f"zh-han-v1:{compact}"


def pinyin_fuzzy_key(text: str) -> str:
    """Return a stable tone-insensitive comparison key for a Chinese name.

    Han surfaces are romanized with word context. Already-romanized spellings
    are case-folded and stripped of tone marks, tone numbers, punctuation, and
    spacing, so ``王芳``, ``Wáng Fāng``, and ``wang2 fang1`` compare equally.
    When ``pypinyin`` is unavailable, Han text uses a deterministic normalized
    fallback key and a one-time actionable warning; no source text is persisted
    by this helper.

    Args:
        text: Han or romanized name surface.

    Returns:
        A versioned comparison key.
    """

    if not isinstance(text, str):
        raise TypeError("Pinyin fuzzy-key input must be a string")
    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized:
        raise ValueError("Pinyin fuzzy-key input must be non-empty")

    if _HAN_RE.search(normalized):
        try:
            syllables = to_pinyin(normalized, style=PinyinStyle.NORMAL)
        except PinyinUnavailableError:
            global _PINYIN_NOTICE_EMITTED
            if not _PINYIN_NOTICE_EMITTED:
                warnings.warn(
                    f"{_MISSING_EXTRA_MESSAGE} "
                    "Using a deterministic Han-surface key instead.",
                    PinyinUnavailableWarning,
                    stacklevel=2,
                )
                _PINYIN_NOTICE_EMITTED = True
            return _fallback_han_key(normalized)
        folded = _fold_romanization(" ".join(syllables))
    else:
        folded = _fold_romanization(normalized)

    if not folded:
        return _fallback_han_key(normalized)
    return f"{PINYIN_KEY_VERSION}:{folded}"


__all__ = [
    "HeteronymReading",
    "PINYIN_KEY_VERSION",
    "PinyinReading",
    "PinyinStyle",
    "PinyinUnavailableError",
    "PinyinUnavailableWarning",
    "pinyin_fuzzy_key",
    "to_pinyin",
]
