"""Render surrogates in the digit system of their source span.

Detection folds native digits to ASCII for matching, but the surrogate
generators and the date shifter work on ASCII digits. In a note written with
Persian, Arabic-Indic, Devanagari, Bengali, Thai, Myanmar or other native
digits, ASCII surrogates stand out and reveal every replaced span. This module
keeps the generators ASCII-only: the source span is folded to ASCII before
generation and validation, and the generated value is rendered back in the
span's digit system afterwards.

Rules:

* A digit system is a run of ten Unicode decimal digits (general category
  ``Nd``). Unicode keeps each run contiguous, so every system can be folded
  and rendered. Its name comes from the Unicode name of its zero:
  ``arabic_indic``, ``extended_arabic_indic`` (Persian and Urdu),
  ``devanagari``, ``bengali``, ``thai``, ``myanmar``, ``fullwidth``, and so
  on.
* A span with no decimal digits, or whose decimal digits are all ASCII, has
  no source digit style. :func:`detect_source_digit_style` returns ``None``
  and callers keep their ASCII behavior unchanged.
* A mixed-digit span uses its dominant system, the one with the most digits.
  A tie goes to the system whose first digit comes earliest in the span. The
  whole surrogate is rendered in that one system.
* Ethiopic numerals (U+1369 to U+137C) are additive, have no zero and are not
  positional, so they are not a decimal digit system. They are never folded
  or rendered, and a span that uses them is passed through unchanged.
* The Arabic date separator and the fullwidth solidus, hyphen-minus and full
  stop are folded to their ASCII counterparts together with the digits and
  restored in the output. A native separator is left alone when its ASCII
  counterpart, or another native separator with the same counterpart, also
  occurs in the span.
* When the source is a set of digit groups joined by separators and the
  generator returns a bare digit run with the same digit count, the source's
  grouping and separators are applied to the result. A grouped native-digit
  Aadhaar keeps its 4-4-4 layout this way.

The helpers keep no state and store no source text. A
:class:`SourceDigitStyle` holds only a digit-system name, separator pairs and
digit-group lengths.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

__all__ = [
    "ASCII_DIGITS",
    "DigitSystem",
    "SourceDigitStyle",
    "detect_digit_system",
    "detect_source_digit_style",
    "digit_system_of",
    "fold_digits",
    "render_digits",
    "render_in_source_digits",
]


@dataclass(frozen=True)
class DigitSystem:
    """A positional decimal digit system.

    Attributes:
        name: Stable lowercase name derived from the Unicode name of the
            system's zero, for example ``"extended_arabic_indic"``.
        zero: The system's digit zero. Digits one to nine follow it.
    """

    name: str
    zero: str

    def digit(self, value: int) -> str:
        """Return the digit for ``value`` (0 to 9) in this system."""
        if not 0 <= value <= 9:
            raise ValueError("digit value must be between 0 and 9")
        return chr(ord(self.zero) + value)


ASCII_DIGITS = DigitSystem(name="ascii", zero="0")
"""The ASCII digits ``0`` to ``9``."""

_NATIVE_SEPARATORS: Mapping[str, str] = MappingProxyType(
    {
        "\N{ARABIC DATE SEPARATOR}": "/",
        "\N{FULLWIDTH HYPHEN-MINUS}": "-",
        "\N{FULLWIDTH FULL STOP}": ".",
        "\N{FULLWIDTH SOLIDUS}": "/",
    }
)
_GROUPED_DIGITS_RE = re.compile(
    r"[0-9]+(?:[ ./\N{NO-BREAK SPACE}\N{NARROW NO-BREAK SPACE}-]+[0-9]+)+"
)
_GROUPING_TOKEN_RE = re.compile(r"[0-9]+|[^0-9]+")

GroupingSegment = int | str


@lru_cache(maxsize=None)
def _system_for_zero(zero: str) -> DigitSystem:
    if zero == "0":
        return ASCII_DIGITS
    unicode_name = unicodedata.name(zero, "")
    prefix = unicode_name.removesuffix(" DIGIT ZERO")
    slug = re.sub(r"[^a-z0-9]+", "_", prefix.casefold()).strip("_")
    return DigitSystem(name=slug or f"u{ord(zero):04x}", zero=zero)


def _decimal_value(char: str) -> int | None:
    if not char.isdecimal():
        return None
    return unicodedata.decimal(char, None)


def digit_system_of(char: str) -> DigitSystem | None:
    """Return the decimal digit system of one character.

    Args:
        char: A single character.

    Returns:
        The character's :class:`DigitSystem`, or ``None`` when it is not a
        Unicode decimal digit. Ethiopic numerals and other non-positional
        numerals return ``None``.
    """
    if len(char) != 1:
        return None
    value = _decimal_value(char)
    if value is None:
        return None
    return _system_for_zero(chr(ord(char) - value))


def detect_digit_system(text: str) -> DigitSystem | None:
    """Return the dominant decimal digit system of ``text``.

    The dominant system has the most digits in ``text``. A tie goes to the
    system whose first digit occurs earliest.

    Args:
        text: Source text, usually one detected span.

    Returns:
        The dominant :class:`DigitSystem`, or ``None`` when ``text`` has no
        decimal digits.
    """
    counts: dict[DigitSystem, int] = {}
    first_index: dict[DigitSystem, int] = {}
    for index, char in enumerate(text):
        system = digit_system_of(char)
        if system is None:
            continue
        counts[system] = counts.get(system, 0) + 1
        first_index.setdefault(system, index)
    if not counts:
        return None
    return min(counts, key=lambda system: (-counts[system], first_index[system]))


def fold_digits(text: str) -> str:
    """Fold every Unicode decimal digit in ``text`` to ASCII.

    The fold is length-preserving. Non-decimal numerals, such as Ethiopic
    numerals or superscripts, are left unchanged.
    """
    return "".join(_fold_char(char) for char in text)


def _fold_char(char: str) -> str:
    value = _decimal_value(char)
    return char if value is None else str(value)


def render_digits(text: str, system: DigitSystem) -> str:
    """Render every Unicode decimal digit in ``text`` in ``system``.

    Args:
        text: Text whose digits may come from any decimal digit system.
        system: Target digit system.

    Returns:
        ``text`` with each decimal digit replaced by the digit of the same
        value in ``system``. Other characters are unchanged.
    """
    rendered: list[str] = []
    for char in text:
        value = _decimal_value(char)
        rendered.append(char if value is None else system.digit(value))
    return "".join(rendered)


@dataclass(frozen=True)
class SourceDigitStyle:
    """Digit system and separator pattern of one source span.

    Attributes:
        system: Dominant digit system of the span.
        separators: ``(native, ascii)`` pairs for native separators that are
            folded to ASCII before generation and restored afterwards.
        grouping: Digit-group lengths and ASCII separators of the folded
            span, set only when the span consists of digit groups joined by
            separators. Used to regroup bare digit surrogates.
    """

    system: DigitSystem
    separators: tuple[tuple[str, str], ...] = ()
    grouping: tuple[GroupingSegment, ...] = ()

    def fold(self, text: str) -> str:
        """Fold ``text`` to ASCII digits and ASCII separators."""
        folded = fold_digits(text)
        for native, ascii_separator in self.separators:
            folded = folded.replace(native, ascii_separator)
        return folded

    def render(self, text: str) -> str:
        """Render a generated ASCII value in this source style.

        Args:
            text: Generator output, normally ASCII digits and separators.

        Returns:
            ``text`` regrouped like the source when it is a bare digit run of
            the same length, with native separators restored and every digit
            rendered in :attr:`system`.
        """
        rendered = _apply_grouping(fold_digits(text), self.grouping)
        for native, ascii_separator in self.separators:
            rendered = rendered.replace(ascii_separator, native)
        return render_digits(rendered, self.system)


def detect_source_digit_style(text: str) -> SourceDigitStyle | None:
    """Describe the digit system and separator pattern of a source span.

    Args:
        text: One detected source span.

    Returns:
        A :class:`SourceDigitStyle` when ``text`` contains at least one
        non-ASCII decimal digit, otherwise ``None``. ASCII-only and digit-free
        spans therefore keep their existing behavior.
    """
    if not isinstance(text, str) or not any(
        not char.isascii() and char.isdecimal() for char in text
    ):
        return None
    system = detect_digit_system(text)
    if system is None:  # pragma: no cover - guarded by the check above
        return None
    separators = _native_separator_pairs(text)
    folded = SourceDigitStyle(system=system, separators=separators).fold(text)
    return SourceDigitStyle(
        system=system,
        separators=separators,
        grouping=_grouping(folded.strip()),
    )


def render_in_source_digits(
    source: str,
    generate: Callable[[str], str | None],
) -> str | None:
    """Run ``generate`` on the ASCII fold of ``source`` and render the result.

    Args:
        source: Detected source span.
        generate: Callable that turns an ASCII-digit source into a surrogate
            or shifted value, or returns ``None``.

    Returns:
        ``generate(source)`` unchanged when ``source`` has no non-ASCII
        decimal digits. Otherwise the result of ``generate`` on the folded
        source, rendered in the source's digit style, or ``None`` when
        ``generate`` returns ``None``.
    """
    style = detect_source_digit_style(source)
    if style is None:
        return generate(source)
    generated = generate(style.fold(source))
    if generated is None:
        return None
    return style.render(generated)


def _native_separator_pairs(text: str) -> tuple[tuple[str, str], ...]:
    natives_by_ascii: dict[str, list[str]] = {}
    for native, ascii_separator in _NATIVE_SEPARATORS.items():
        if native in text:
            natives_by_ascii.setdefault(ascii_separator, []).append(native)
    return tuple(
        (natives[0], ascii_separator)
        for ascii_separator, natives in sorted(natives_by_ascii.items())
        if len(natives) == 1 and ascii_separator not in text
    )


def _grouping(folded: str) -> tuple[GroupingSegment, ...]:
    if _GROUPED_DIGITS_RE.fullmatch(folded) is None:
        return ()
    return tuple(
        len(token) if token.isdigit() else token
        for token in _GROUPING_TOKEN_RE.findall(folded)
    )


def _apply_grouping(text: str, grouping: tuple[GroupingSegment, ...]) -> str:
    if not grouping or not (text.isascii() and text.isdigit()):
        return text
    group_lengths = [segment for segment in grouping if isinstance(segment, int)]
    if len(text) != sum(group_lengths):
        return text
    regrouped: list[str] = []
    position = 0
    for segment in grouping:
        if isinstance(segment, int):
            regrouped.append(text[position : position + segment])
            position += segment
        else:
            regrouped.append(segment)
    return "".join(regrouped)
