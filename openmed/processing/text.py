"""Text processing utilities, including offset-safe Indic normalization."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


INDIC_SCRIPTS = (
    "Devanagari",
    "Bengali",
    "Gurmukhi",
    "Gujarati",
    "Odia",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
)

_SCRIPT_BLOCKS: dict[str, tuple[int, int]] = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}

_SCRIPT_ALIASES = {
    "as": "Bengali",
    "Assamese": "Bengali",
    "Bengali/Assamese": "Bengali",
    "bn": "Bengali",
    "Deva": "Devanagari",
    "gu": "Gujarati",
    "hi": "Devanagari",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Devanagari",
    "ne": "Devanagari",
    "or": "Odia",
    "Oriya": "Odia",
    "pa": "Gurmukhi",
    "ta": "Tamil",
    "te": "Telugu",
}

_NUKTA_OFFSETS = {
    "Devanagari": 0x3C,
    "Bengali": 0x3C,
    "Gurmukhi": 0x3C,
    "Gujarati": 0x3C,
    "Odia": 0x3C,
}

_NUKTA_DECOMPOSITIONS: dict[str, dict[str, str]] = {
    "Devanagari": {
        "\u0929": "\u0928\u093c",
        "\u0931": "\u0930\u093c",
        "\u0934": "\u0933\u093c",
        "\u0958": "\u0915\u093c",
        "\u0959": "\u0916\u093c",
        "\u095a": "\u0917\u093c",
        "\u095b": "\u091c\u093c",
        "\u095c": "\u0921\u093c",
        "\u095d": "\u0922\u093c",
        "\u095e": "\u092b\u093c",
        "\u095f": "\u092f\u093c",
    },
    "Bengali": {
        "\u09dc": "\u09a1\u09bc",
        "\u09dd": "\u09a2\u09bc",
        "\u09df": "\u09af\u09bc",
    },
    "Gurmukhi": {
        "\u0a33": "\u0a32\u0a3c",
        "\u0a36": "\u0a38\u0a3c",
        "\u0a59": "\u0a16\u0a3c",
        "\u0a5a": "\u0a17\u0a3c",
        "\u0a5b": "\u0a1c\u0a3c",
        "\u0a5e": "\u0a2b\u0a3c",
    },
    "Odia": {
        "\u0b5c": "\u0b21\u0b3c",
        "\u0b5d": "\u0b22\u0b3c",
    },
}

# These small, data-only tables are independently implemented from the
# permissively licensed Indic NLP Library normalizer behavior. See
# docs/indic-normalization.md for provenance and the intentional differences.
_SEQUENCE_REPLACEMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "Gurmukhi": (
        ("\u0a05\u0a3e", "\u0a06"),
        ("\u0a72\u0a3f", "\u0a07"),
        ("\u0a72\u0a40", "\u0a08"),
        ("\u0a73\u0a41", "\u0a09"),
        ("\u0a73\u0a42", "\u0a0a"),
        ("\u0a72\u0a47", "\u0a0f"),
        ("\u0a05\u0a48", "\u0a10"),
        ("\u0a73\u0a4b", "\u0a13"),
        ("\u0a05\u0a4c", "\u0a14"),
    ),
    "Odia": (
        ("\u0b05\u0b3e", "\u0b06"),
        ("\u0b0f\u0b57", "\u0b10"),
        ("\u0b13\u0b57", "\u0b14"),
        ("\u0b47\u0b56", "\u0b58"),
        ("\u0b56\u0b47", "\u0b58"),
        ("\u0b47\u0b3e", "\u0b4b"),
        ("\u0b3e\u0b47", "\u0b4b"),
        ("\u0b47\u0b57", "\u0b4c"),
        ("\u0b57\u0b47", "\u0b4c"),
    ),
    "Bengali": (
        ("\u09c7\u09be", "\u09cb"),
        ("\u09be\u09c7", "\u09cb"),
        ("\u09c7\u09d7", "\u09cc"),
        ("\u09d7\u09c7", "\u09cc"),
    ),
    "Tamil": (
        ("\u0b92\u0bd7", "\u0b94"),
        ("\u0bc6\u0bbe", "\u0bca"),
        ("\u0bbe\u0bc6", "\u0bca"),
        ("\u0bc7\u0bbe", "\u0bcb"),
        ("\u0bbe\u0bc7", "\u0bcb"),
        ("\u0bc6\u0bd7", "\u0bcc"),
        ("\u0bd7\u0bc6", "\u0bcc"),
    ),
    "Telugu": (
        ("\u0c46\u0c56", "\u0c48"),
        ("\u0c56\u0c46", "\u0c48"),
    ),
    "Kannada": (
        ("\u0cbf\u0cd5", "\u0cc0"),
        ("\u0cd5\u0cbf", "\u0cc0"),
        ("\u0cc6\u0cd5", "\u0cc7"),
        ("\u0cd5\u0cc6", "\u0cc7"),
        ("\u0cc6\u0cd6", "\u0cc8"),
        ("\u0cd6\u0cc6", "\u0cc8"),
        ("\u0cc6\u0cc2", "\u0cca"),
        ("\u0cc2\u0cc6", "\u0cca"),
        ("\u0cca\u0cd5", "\u0ccb"),
        ("\u0cd5\u0cca", "\u0ccb"),
    ),
    "Malayalam": (
        ("\u0d46\u0d3e", "\u0d4a"),
        ("\u0d3e\u0d46", "\u0d4a"),
        ("\u0d47\u0d3e", "\u0d4b"),
        ("\u0d3e\u0d47", "\u0d4b"),
        ("\u0d46\u0d57", "\u0d4c"),
        ("\u0d57\u0d46", "\u0d4c"),
        ("\u0d57", "\u0d4c"),
    ),
}

_MALAYALAM_CHILLUS = {
    "\u0d23\u0d4d\u200d": "\u0d7a",
    "\u0d28\u0d4d\u200d": "\u0d7b",
    "\u0d30\u0d4d\u200d": "\u0d7c",
    "\u0d32\u0d4d\u200d": "\u0d7d",
    "\u0d33\u0d4d\u200d": "\u0d7e",
    "\u0d15\u0d4d\u200d": "\u0d7f",
}

_SCRIPT_POORNA_VIRAMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "Bengali": (("\u09e4", "\u0964"), ("\u09e5", "\u0965")),
    "Gurmukhi": (("\u0a64", "\u0964"), ("\u0a65", "\u0965")),
    "Gujarati": (("\u0ae4", "\u0964"), ("\u0ae5", "\u0965")),
    "Odia": (("\u0b64", "\u0964"), ("\u0b65", "\u0965")),
    "Tamil": (("\u0be4", "\u0964"), ("\u0be5", "\u0965")),
    "Telugu": (("\u0c64", "\u0964"), ("\u0c65", "\u0965")),
    "Kannada": (("\u0ce4", "\u0964"), ("\u0ce5", "\u0965")),
    "Malayalam": (("\u0d64", "\u0964"), ("\u0d65", "\u0965")),
}

_VISARGAS = {
    "Devanagari": "\u0903",
    "Bengali": "\u0983",
    "Gurmukhi": "\u0a03",
    "Gujarati": "\u0a83",
    "Odia": "\u0b03",
    "Tamil": "\u0b83",
    "Telugu": "\u0c03",
    "Kannada": "\u0c83",
    "Malayalam": "\u0d03",
}

_VIRAMAS = {
    "Devanagari": "\u094d",
    "Bengali": "\u09cd",
    "Gurmukhi": "\u0a4d",
    "Gujarati": "\u0acd",
    "Odia": "\u0b4d",
    "Tamil": "\u0bcd",
    "Telugu": "\u0c4d",
    "Kannada": "\u0ccd",
    "Malayalam": "\u0d4d",
}

_JOINERS = frozenset({"\u200c", "\u200d"})


@dataclass(frozen=True)
class IndicNormalization:
    """Normalized text plus a map back to raw character boundaries."""

    text: str
    original_length: int
    offset_starts: tuple[int, ...]
    offset_ends: tuple[int, ...]
    changes: int = 0
    removed_joiners: int = 0
    scripts: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Return whether canonicalization changed the input."""

        return self.changes > 0

    def remap_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a normalized half-open span to raw grapheme boundaries."""

        safe_start = max(0, min(int(start), len(self.text)))
        safe_end = max(safe_start, min(int(end), len(self.text)))
        if not self.offset_starts:
            return 0, 0
        if safe_start >= len(self.offset_starts):
            original_start = self.original_length
        else:
            original_start = self.offset_starts[safe_start]
        if safe_end <= 0:
            original_end = original_start
        elif safe_end - 1 >= len(self.offset_ends):
            original_end = self.original_length
        else:
            original_end = self.offset_ends[safe_end - 1]
        return original_start, max(original_start, original_end)


@dataclass(frozen=True)
class _MappedChar:
    char: str
    start: int
    end: int


class IndicNormalizer:
    """Canonicalize nine Brahmi-derived Unicode scripts without network access.

    NFC is applied before script-specific rules. Defaults collapse common
    encoding variants used to evade clinical PII detection while retaining
    every script-bearing code point unless a rule replaces it with a canonical
    script-bearing equivalent. Nukta removal is opt-in.

    Args:
        remove_nuktas: Remove script nuktas after decomposition. Defaults to
            ``False`` so phonemic distinctions are retained.
        nasals_mode: ``"to_anusvara"`` (default), ``"to_nasal_consonants"``,
            or ``"preserve"``. Indic NLP Library spelling aliases are accepted.
        normalize_chandra: Map chandrabindu/chandra forms to canonical anusvara
            and vowel forms.
        normalize_vowel_ending: Add a script-appropriate explicit ending to
            words ending in a consonant. Disabled by default.
        joiner_policy: Strip ZWJ/ZWNJ variants by default, or ``"preserve"``.
    """

    _NASAL_MODES = {
        "do_nothing": "preserve",
        "preserve": "preserve",
        "to_anusvaara_relaxed": "to_anusvara_relaxed",
        "to_anusvaara_strict": "to_anusvara",
        "to_anusvara": "to_anusvara",
        "to_anusvara_relaxed": "to_anusvara_relaxed",
        "to_nasal_consonants": "to_nasal_consonants",
    }

    def __init__(
        self,
        *,
        remove_nuktas: bool = False,
        nasals_mode: str = "to_anusvara",
        normalize_chandra: bool = True,
        normalize_vowel_ending: bool = False,
        joiner_policy: str = "strip",
    ) -> None:
        try:
            self.nasals_mode = self._NASAL_MODES[nasals_mode]
        except KeyError as exc:
            choices = ", ".join(sorted(self._NASAL_MODES))
            raise ValueError(
                f"Unsupported nasals_mode; choose one of: {choices}"
            ) from exc
        if joiner_policy not in {"preserve", "strip"}:
            raise ValueError("joiner_policy must be 'preserve' or 'strip'")
        self.remove_nuktas = bool(remove_nuktas)
        self.normalize_chandra = bool(normalize_chandra)
        self.normalize_vowel_ending = bool(normalize_vowel_ending)
        self.joiner_policy = joiner_policy

    def normalize(self, text: str, script: str | None = None) -> str:
        """Return canonical text, detecting script runs when needed."""

        return self.normalize_with_offsets(text, script=script).text

    def normalize_with_offsets(
        self,
        text: str,
        script: str | None = None,
    ) -> IndicNormalization:
        """Return canonical text with normalized-to-raw offset mappings."""

        if not isinstance(text, str):
            text = str(text)
        if not text:
            return IndicNormalization("", 0, (), ())

        if script is not None:
            canonical_script = self._canonical_script(script)
            return self._normalize_run(text, canonical_script)

        # Imported lazily to keep text processing usable without initializing
        # the full core package and to avoid a module import cycle.
        from openmed.core.script_detect import segment_by_script

        chars: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        changes = 0
        removed_joiners = 0
        scripts: list[str] = []
        for run_start, run_end, run_script in segment_by_script(text):
            result = self._normalize_run(text[run_start:run_end], run_script)
            chars.append(result.text)
            starts.extend(run_start + value for value in result.offset_starts)
            ends.extend(run_start + value for value in result.offset_ends)
            changes += result.changes
            removed_joiners += result.removed_joiners
            if run_script in INDIC_SCRIPTS and run_script not in scripts:
                scripts.append(run_script)
        return IndicNormalization(
            text="".join(chars),
            original_length=len(text),
            offset_starts=tuple(starts),
            offset_ends=tuple(ends),
            changes=changes,
            removed_joiners=removed_joiners,
            scripts=tuple(scripts),
        )

    @staticmethod
    def _canonical_script(script: str) -> str:
        canonical = _SCRIPT_ALIASES.get(script, script)
        if canonical not in _SCRIPT_BLOCKS:
            return canonical
        return canonical

    def _normalize_run(self, text: str, script: str) -> IndicNormalization:
        units, changes = _nfc_units(text)
        if script not in INDIC_SCRIPTS:
            return _normalization_from_units(
                units,
                original_length=len(text),
                changes=changes,
            )

        script_changes = 0
        initial_joiners = sum(unit.char in _JOINERS for unit in units)
        removed_joiners = 0

        if script == "Malayalam":
            for old, new in _MALAYALAM_CHILLUS.items():
                units, count = _replace_units(units, old, new)
                script_changes += count

        for old, new in _NUKTA_DECOMPOSITIONS.get(script, {}).items():
            units, count = _replace_units(units, old, new)
            script_changes += count

        if script == "Devanagari":
            units, count = _replace_units(units, "\u0972", "\u090f")
            script_changes += count

        if script == "Gurmukhi":
            units, count = _canonicalize_gurmukhi_addak(units)
            script_changes += count
            units, count = _replace_units(units, "\u0a70", "\u0a02")
            script_changes += count

        for old, new in _SEQUENCE_REPLACEMENTS.get(script, ()):
            units, count = _replace_units(units, old, new)
            script_changes += count

        if script == "Odia":
            for old in ("\u0b35", "\u0b71"):
                units, count = _replace_units(units, old, "\u0b2c")
                script_changes += count

        if self.normalize_chandra:
            units, count = _normalize_chandras(units, script)
            script_changes += count

        units, count = _normalize_nasals(units, script, self.nasals_mode)
        script_changes += count

        if self.remove_nuktas and script in _NUKTA_OFFSETS:
            nukta = chr(_SCRIPT_BLOCKS[script][0] + _NUKTA_OFFSETS[script])
            units, count = _drop_units(units, frozenset({nukta}))
            script_changes += count

        for old, new in _SCRIPT_POORNA_VIRAMAS.get(script, ()):
            units, count = _replace_units(units, old, new)
            script_changes += count

        if script == "Bengali":
            for old in ("\u09f7", "|"):
                units, count = _replace_units(units, old, "\u0964")
                script_changes += count
        elif script == "Odia":
            for old in ("\u0b7c", "|"):
                units, count = _replace_units(units, old, "\u0964")
                script_changes += count
        else:
            units, count = _replace_units(units, "|", "\u0964")
            script_changes += count

        if self.joiner_policy == "strip":
            units, count = _drop_units(units, _JOINERS)
            script_changes += count

        units, count = _replace_colon_with_visarga(units, script)
        script_changes += count

        if self.normalize_vowel_ending:
            units, count = _normalize_vowel_endings(units, script)
            script_changes += count

        removed_joiners = initial_joiners - sum(unit.char in _JOINERS for unit in units)

        return _normalization_from_units(
            units,
            original_length=len(text),
            changes=changes + script_changes,
            removed_joiners=removed_joiners,
            scripts=(script,),
        )


def _nfc_units(text: str) -> tuple[list[_MappedChar], int]:
    """Apply NFC per base-plus-mark cluster and retain grapheme boundaries."""

    clusters: list[tuple[int, int]] = []
    cluster_start = 0
    for index in range(1, len(text)):
        if unicodedata.category(text[index])[0] != "M":
            clusters.append((cluster_start, index))
            cluster_start = index
    clusters.append((cluster_start, len(text)))

    units: list[_MappedChar] = []
    changes = 0
    for start, end in clusters:
        raw = text[start:end]
        normalized = unicodedata.normalize("NFC", raw)
        if normalized != raw:
            changes += 1
        units.extend(_MappedChar(char, start, end) for char in normalized)
    return units, changes


def _normalization_from_units(
    units: list[_MappedChar],
    *,
    original_length: int,
    changes: int,
    removed_joiners: int = 0,
    scripts: tuple[str, ...] = (),
) -> IndicNormalization:
    return IndicNormalization(
        text="".join(unit.char for unit in units),
        original_length=original_length,
        offset_starts=tuple(unit.start for unit in units),
        offset_ends=tuple(unit.end for unit in units),
        changes=changes,
        removed_joiners=removed_joiners,
        scripts=scripts,
    )


def _replacement_units(
    matched: list[_MappedChar], replacement: str
) -> list[_MappedChar]:
    start = min(unit.start for unit in matched)
    end = max(unit.end for unit in matched)
    return [_MappedChar(char, start, end) for char in replacement]


def _replace_units(
    units: list[_MappedChar],
    old: str,
    new: str,
) -> tuple[list[_MappedChar], int]:
    if not old:
        return units, 0
    output: list[_MappedChar] = []
    count = 0
    index = 0
    width = len(old)
    while index < len(units):
        matched = units[index : index + width]
        if len(matched) == width and "".join(unit.char for unit in matched) == old:
            output.extend(_replacement_units(matched, new))
            count += 1
            index += width
            continue
        output.append(units[index])
        index += 1
    return output, count


def _drop_units(
    units: list[_MappedChar],
    chars: frozenset[str],
) -> tuple[list[_MappedChar], int]:
    output = [unit for unit in units if unit.char not in chars]
    return output, len(units) - len(output)


def _canonicalize_gurmukhi_addak(
    units: list[_MappedChar],
) -> tuple[list[_MappedChar], int]:
    output: list[_MappedChar] = []
    count = 0
    index = 0
    while index < len(units):
        if (
            units[index].char == "\u0a71"
            and index + 1 < len(units)
            and 0x0A15 <= ord(units[index + 1].char) <= 0x0A39
        ):
            matched = units[index : index + 2]
            consonant = units[index + 1].char
            output.extend(_replacement_units(matched, consonant + "\u0a4d" + consonant))
            count += 1
            index += 2
            continue
        output.append(units[index])
        index += 1
    return output, count


def _normalize_chandras(
    units: list[_MappedChar],
    script: str,
) -> tuple[list[_MappedChar], int]:
    block_start = _SCRIPT_BLOCKS[script][0]
    substitutions = (
        (0x0D, 0x0F),
        (0x11, 0x13),
        (0x45, 0x47),
        (0x49, 0x4B),
        (0x00, 0x02),
        (0x01, 0x02),
    )
    changes = 0
    for source_offset, target_offset in substitutions:
        units, count = _replace_units(
            units,
            chr(block_start + source_offset),
            chr(block_start + target_offset),
        )
        changes += count
    return units, changes


def _normalize_nasals(
    units: list[_MappedChar],
    script: str,
    mode: str,
) -> tuple[list[_MappedChar], int]:
    if mode == "preserve":
        return units, 0

    block_start = _SCRIPT_BLOCKS[script][0]
    virama = _VIRAMAS[script]
    anusvara = chr(block_start + 0x02)
    signatures = (
        (0x19, 0x15, 0x18),
        (0x1E, 0x1A, 0x1D),
        (0x23, 0x1F, 0x22),
        (0x28, 0x24, 0x27),
        (0x29, 0x24, 0x27),
        (0x2E, 0x2A, 0x2D),
    )
    output: list[_MappedChar] = []
    count = 0
    index = 0

    if mode in {"to_anusvara", "to_anusvara_relaxed"}:
        nasal_offsets = {signature[0] for signature in signatures}
        while index < len(units):
            if index + 1 < len(units):
                nasal_offset = ord(units[index].char) - block_start
                is_nasal = nasal_offset in nasal_offsets
                if is_nasal and units[index + 1].char == virama:
                    strict_match = False
                    if index + 2 < len(units):
                        following = ord(units[index + 2].char) - block_start
                        strict_match = any(
                            nasal_offset == nasal and first <= following <= last
                            for nasal, first, last in signatures
                        )
                    if mode == "to_anusvara_relaxed" or strict_match:
                        matched = units[index : index + 2]
                        output.extend(_replacement_units(matched, anusvara))
                        count += 1
                        index += 2
                        continue
            output.append(units[index])
            index += 1
        return output, count

    while index < len(units):
        if units[index].char == anusvara and index + 1 < len(units):
            following = ord(units[index + 1].char) - block_start
            matching = next(
                (
                    nasal
                    for nasal, first, last in signatures
                    if first <= following <= last
                ),
                None,
            )
            if matching is not None:
                output.extend(
                    _replacement_units(
                        [units[index]],
                        chr(block_start + matching) + virama,
                    )
                )
                count += 1
                index += 1
                continue
        output.append(units[index])
        index += 1
    return output, count


def _replace_colon_with_visarga(
    units: list[_MappedChar],
    script: str,
) -> tuple[list[_MappedChar], int]:
    block_start, block_end = _SCRIPT_BLOCKS[script]
    output = list(units)
    count = 0
    for index in range(1, len(output)):
        previous = ord(output[index - 1].char)
        if output[index].char == ":" and block_start <= previous <= block_end:
            unit = output[index]
            output[index] = _MappedChar(_VISARGAS[script], unit.start, unit.end)
            count += 1
    return output, count


def _normalize_vowel_endings(
    units: list[_MappedChar],
    script: str,
) -> tuple[list[_MappedChar], int]:
    block_start = _SCRIPT_BLOCKS[script][0]
    dravidian = script in {"Tamil", "Telugu", "Kannada", "Malayalam"}
    ending = chr(block_start + 0x3E) if dravidian else _VIRAMAS[script]
    output: list[_MappedChar] = []
    changes = 0
    for index, unit in enumerate(units):
        output.append(unit)
        offset = ord(unit.char) - block_start
        next_is_boundary = index + 1 == len(units) or units[index + 1].char.isspace()
        if 0x15 <= offset <= 0x39 and next_is_boundary:
            output.append(_MappedChar(ending, unit.start, unit.end))
            changes += 1
    return output, changes


class TextProcessor:
    """Handles text preprocessing and cleaning for medical text analysis."""

    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        normalize_whitespace: bool = True,
    ):
        """Initialize text processor.

        Args:
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to remove punctuation.
            remove_numbers: Whether to remove numbers.
            normalize_whitespace: Whether to normalize whitespace.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace

        # Medical abbreviations that should be preserved
        self.medical_abbreviations = {
            "mg",
            "ml",
            "kg",
            "lb",
            "oz",
            "cm",
            "mm",
            "hr",
            "min",
            "bp",
            "hr",
            "rr",
            "temp",
            "o2",
            "co2",
            "hiv",
            "aids",
            "icu",
            "er",
            "or",
            "cbc",
            "ekg",
            "ecg",
            "mri",
            "ct",
            "x-ray",
            "ultrasound",
            "bmi",
            "copd",
            "chf",
            "mi",
            "stroke",
            "tia",
            "dvt",
            "pe",
            "uti",
            "copd",
        }

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned text.
        """
        if not isinstance(text, str):
            text = str(text)

        original_text = text

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text.strip())

        # Handle medical abbreviations before other processing
        protected_abbrevs = {}
        if not self.remove_punctuation:
            for i, abbrev in enumerate(self.medical_abbreviations):
                placeholder = f"__ABBREV_{i}__"
                text = re.sub(
                    rf"\b{re.escape(abbrev)}\b", placeholder, text, flags=re.IGNORECASE
                )
                protected_abbrevs[placeholder] = abbrev

        # Remove or clean numbers
        if self.remove_numbers:
            # Preserve medical measurements (e.g., "120/80", "98.6°F")
            text = re.sub(r"\b\d+(?:[./]\d+)*\b(?![°%])", " ", text)

        # Remove punctuation
        if self.remove_punctuation:
            # Keep hyphens in compound medical terms
            text = re.sub(r"[^\w\s\-]", " ", text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Restore protected abbreviations
        for placeholder, abbrev in protected_abbrevs.items():
            text = text.replace(placeholder, abbrev)

        # Final whitespace normalization
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text.strip())

        logger.debug(
            "Text cleaning completed: input_chars=%d output_chars=%d changed=%s",
            len(original_text),
            len(text),
            original_text != text,
        )
        return text

    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using medical text-aware rules.

        Args:
            text: Input text to segment.

        Returns:
            List of sentences.
        """
        # Medical abbreviations that shouldn't trigger sentence breaks
        abbrev_pattern = r"\b(?:" + "|".join(self.medical_abbreviations) + r")\."

        # Temporarily replace medical abbreviations
        text_modified = re.sub(
            abbrev_pattern,
            lambda m: m.group().replace(".", "___DOT___"),
            text,
            flags=re.IGNORECASE,
        )

        # Simple sentence segmentation
        sentences = re.split(r"[.!?]+\s+", text_modified)

        # Restore dots in abbreviations
        sentences = [s.replace("___DOT___", ".") for s in sentences if s.strip()]

        return sentences

    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract basic medical entities using regex patterns.

        Args:
            text: Input text.

        Returns:
            Dictionary of entity types and their matches.
        """
        entities = {
            "medications": [],
            "dosages": [],
            "vital_signs": [],
            "lab_values": [],
            "symptoms": [],
        }

        # Dosage patterns
        dosage_patterns = [
            r"\b\d+\s*(?:mg|ml|g|kg|mcg|units?)\b",
            r"\b\d+\.\d+\s*(?:mg|ml|g|kg|mcg|units?)\b",
        ]

        for pattern in dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dosages"].extend(matches)

        # Vital signs patterns
        vital_patterns = [
            r"\b(?:bp|blood pressure):?\s*\d+/\d+\b",
            r"\b(?:hr|heart rate):?\s*\d+\b",
            r"\b(?:temp|temperature):?\s*\d+\.?\d*\s*[°]?[fF]?\b",
            r"\b(?:rr|respiratory rate):?\s*\d+\b",
        ]

        for pattern in vital_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["vital_signs"].extend(matches)

        # Clean up duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities


def preprocess_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    normalize_whitespace: bool = True,
) -> str:
    """Convenience function for text preprocessing.

    Args:
        text: Input text.
        lowercase: Whether to convert to lowercase.
        remove_punctuation: Whether to remove punctuation.
        remove_numbers: Whether to remove numbers.
        normalize_whitespace: Whether to normalize whitespace.

    Returns:
        Preprocessed text.
    """
    processor = TextProcessor(
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers,
        normalize_whitespace=normalize_whitespace,
    )
    return processor.clean_text(text)


def postprocess_text(text: str, capitalize_first: bool = True) -> str:
    """Postprocess text for better readability.

    Args:
        text: Input text.
        capitalize_first: Whether to capitalize the first letter.

    Returns:
        Postprocessed text.
    """
    if not text:
        return text

    text = text.strip()

    if capitalize_first and text:
        text = text[0].upper() + text[1:]

    return text


# ---------------------------------------------------------------------------
# Indic native-digit folding
# ---------------------------------------------------------------------------

# Base code point of the contiguous 0..9 decimal block for each supported
# Indic script. Only positional decimal digits are folded; non-positional
# number signs (e.g. the Tamil traditional signs U+0BF0-0BF2) are left as-is.
_INDIC_DIGIT_BASES: Dict[str, int] = {
    "devanagari": 0x0966,
    "bengali": 0x09E6,
    "gurmukhi": 0x0A66,
    "gujarati": 0x0AE6,
    "odia": 0x0B66,
    "tamil": 0x0BE6,
    "telugu": 0x0C66,
    "kannada": 0x0CE6,
    "malayalam": 0x0D66,
}

#: The Indic scripts whose decimal digits are folded to ASCII.
INDIC_DIGIT_SCRIPTS = tuple(_INDIC_DIGIT_BASES)

# Translation table from every native decimal digit code point to its ASCII
# equivalent. Built once; folding is a single str.translate call.
_INDIC_DIGIT_TRANSLATION = {
    base + digit: ord(str(digit))
    for base in _INDIC_DIGIT_BASES.values()
    for digit in range(10)
}


@dataclass(frozen=True)
class DigitFolding:
    """Native-digit text folded to ASCII, with a source offset mapping.

    Indic decimal digits are single code points, so folding is strictly
    length-preserving and the offset mapping is the identity: a span detected on
    the folded ``text`` indexes the same characters in ``original``, whose native
    digits are preserved for the output.
    """

    text: str
    original: str

    def to_original_span(self, start: int, end: int) -> "tuple[int, int]":
        """Map a folded ``[start, end)`` span to the original text offsets."""

        if not (0 <= start <= end <= len(self.text)):
            raise ValueError("span must satisfy 0 <= start <= end <= len(text)")
        return start, end


def fold_indic_digits(text: str) -> DigitFolding:
    """Fold Indic native decimal digits to ASCII, preserving offsets.

    Maps all nine native digit sets (Devanagari, Bengali, Gurmukhi, Gujarati,
    Odia, Tamil, Telugu, Kannada, Malayalam) to ASCII while leaving every other
    character -- ASCII digits, letters, and non-positional number signs --
    untouched. Folding is idempotent and length-preserving; the returned
    :class:`DigitFolding` keeps the original surface so native digits survive in
    the output.
    """

    return DigitFolding(text=text.translate(_INDIC_DIGIT_TRANSLATION), original=text)


def detect_with_digit_folding(
    text: str,
    matcher: Callable[[str], Iterable[Sequence[object]]],
) -> "list[tuple[object, ...]]":
    """Run ``matcher`` on digit-folded ``text`` and map spans back to the source.

    ``matcher`` takes the folded text and returns items whose first two elements
    are the ``(start, end)`` span; any trailing elements are preserved. Each
    result is ``(original_start, original_end, *trailing)``, so ASCII-only
    validators and regexes detect native-digit PHI while spans still index the
    original native-digit text.
    """

    folding = fold_indic_digits(text)
    results: "list[tuple[object, ...]]" = []
    for item in matcher(folding.text):
        start = int(item[0])  # type: ignore[call-overload]
        end = int(item[1])  # type: ignore[call-overload]
        original_start, original_end = folding.to_original_span(start, end)
        results.append((original_start, original_end, *tuple(item[2:])))
    return results
