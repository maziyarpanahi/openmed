"""Deterministic transliteration for the major Brahmi-derived Indic scripts.

The engine uses ISO 15919 as a common, NFC-normalized pivot for Devanagari,
Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, and Malayalam.  The
tables are a clean-room Unicode/ISO implementation interoperable with mapping
conventions used by Aksharamukha (AGPL-3.0) and the Indic NLP Library (MIT); no
project code, data bundle, copyleft component, or model weights are copied or
shipped. Neural or statistical transliterators remain user-supplied adapters.
Perso-Arabic Urdu is an explicit unsupported stub: callers receive
``ValueError`` and must provide an out-of-process adapter rather than silently
using an Indic-script table.

The default ``preserve`` schwa policy is orthographic and round-trip safe for
the supported subset.  The documented lossy cases are exposed as
``LOSSY_CASES``: script-specific letters without a one-to-one ISO pivot,
nukta/extended letters, Tamil consonant voicing and aspiration, source-policy
word-final schwa deletion, homorganic expansion of anusvara, and Malayalam
chillu letters.  Callers that need exact reconstruction should keep the default
schwa and anusvara policies and restrict input to ``LOSSY_CASES``-free text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, cast

RomanizationScheme = Literal["iso15919", "itrans", "harvard-kyoto"]
SchwaPolicy = Literal["preserve", "word-final", "source"]
AnusvaraPolicy = Literal["marker", "homorganic"]

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

LOSSY_CASES = (
    "nukta and script-specific extended consonants",
    "Tamil voicing and aspiration distinctions",
    "word-final schwa deletion with schwa_policy='word-final' or 'source'",
    "anusvara expansion with anusvara_policy='homorganic'",
    "Gurmukhi addak and Malayalam chillu letters",
)

_SCRIPT_BASES = {
    "Devanagari": 0x0900,
    "Bengali": 0x0980,
    "Gurmukhi": 0x0A00,
    "Gujarati": 0x0A80,
    "Odia": 0x0B00,
    "Tamil": 0x0B80,
    "Telugu": 0x0C00,
    "Kannada": 0x0C80,
    "Malayalam": 0x0D00,
}

_NORTHERN_SCHWA_SCRIPTS = frozenset(
    {"Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Odia"}
)

_INDEPENDENT_VOWEL_OFFSETS = {
    0x05: "a",
    0x06: "ā",
    0x07: "i",
    0x08: "ī",
    0x09: "u",
    0x0A: "ū",
    0x0B: "r̥",
    0x0C: "l̥",
    0x0E: "e",
    0x0F: "ē",
    0x10: "ai",
    0x12: "o",
    0x13: "ō",
    0x14: "au",
    0x60: "r̥̄",
    0x61: "l̥̄",
}

_DEPENDENT_VOWEL_OFFSETS = {
    0x3E: "ā",
    0x3F: "i",
    0x40: "ī",
    0x41: "u",
    0x42: "ū",
    0x43: "r̥",
    0x44: "r̥̄",
    0x46: "e",
    0x47: "ē",
    0x48: "ai",
    0x4A: "o",
    0x4B: "ō",
    0x4C: "au",
    0x62: "l̥",
    0x63: "l̥̄",
}

_CONSONANT_OFFSETS = {
    0x15: "k",
    0x16: "kh",
    0x17: "g",
    0x18: "gh",
    0x19: "ṅ",
    0x1A: "c",
    0x1B: "ch",
    0x1C: "j",
    0x1D: "jh",
    0x1E: "ñ",
    0x1F: "ṭ",
    0x20: "ṭh",
    0x21: "ḍ",
    0x22: "ḍh",
    0x23: "ṇ",
    0x24: "t",
    0x25: "th",
    0x26: "d",
    0x27: "dh",
    0x28: "n",
    0x29: "ṉ",
    0x2A: "p",
    0x2B: "ph",
    0x2C: "b",
    0x2D: "bh",
    0x2E: "m",
    0x2F: "y",
    0x30: "r",
    0x31: "ṟ",
    0x32: "l",
    0x33: "ḷ",
    0x34: "ḻ",
    0x35: "v",
    0x36: "ś",
    0x37: "ṣ",
    0x38: "s",
    0x39: "h",
}

_MARK_OFFSETS = {
    0x01: "m̐",
    0x02: "ṁ",
    0x03: "ḥ",
}

_ISO_VOWELS = frozenset(_INDEPENDENT_VOWEL_OFFSETS.values())
_ISO_CONSONANTS = frozenset(_CONSONANT_OFFSETS.values()) | frozenset(
    {"q", "x", "ġ", "z", "ṛ", "ṛh", "f", "ẏ"}
)
_ISO_MARKS = frozenset({"m̐", "ṁ", "ṃ", "ḥ"})
_ISO_TOKENS = tuple(
    sorted(_ISO_VOWELS | _ISO_CONSONANTS | _ISO_MARKS, key=len, reverse=True)
)

_ITRANS_TO_ISO = {
    "RRI": "r̥̄",
    "RRi": "r̥",
    "R^I": "r̥̄",
    "R^i": "r̥",
    "LLI": "l̥̄",
    "LLi": "l̥",
    "L^I": "l̥̄",
    "L^i": "l̥",
    "Ch": "ch",
    "chh": "ch",
    "ch": "c",
    "kh": "kh",
    "gh": "gh",
    "jh": "jh",
    "Th": "ṭh",
    "Dh": "ḍh",
    "th": "th",
    "dh": "dh",
    "ph": "ph",
    "bh": "bh",
    "~N": "ṅ",
    "~n": "ñ",
    ".N": "m̐",
    ".n": "ṁ",
    "Sh": "ṣ",
    "sh": "ś",
    "aa": "ā",
    "ii": "ī",
    "uu": "ū",
    "ai": "ai",
    "au": "au",
    "A": "ā",
    "I": "ī",
    "U": "ū",
    "T": "ṭ",
    "D": "ḍ",
    "N": "ṇ",
    "M": "ṁ",
    "H": "ḥ",
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "ē",
    "o": "ō",
    "k": "k",
    "g": "g",
    "c": "c",
    "j": "j",
    "t": "t",
    "d": "d",
    "n": "n",
    "p": "p",
    "b": "b",
    "m": "m",
    "y": "y",
    "r": "r",
    "l": "l",
    "v": "v",
    "w": "v",
    "s": "s",
    "h": "h",
}

_HK_TO_ISO = {
    "lRR": "l̥̄",
    "RR": "r̥̄",
    "lR": "l̥",
    "kh": "kh",
    "gh": "gh",
    "ch": "ch",
    "jh": "jh",
    "Th": "ṭh",
    "Dh": "ḍh",
    "th": "th",
    "dh": "dh",
    "ph": "ph",
    "bh": "bh",
    "ai": "ai",
    "au": "au",
    "A": "ā",
    "I": "ī",
    "U": "ū",
    "R": "r̥",
    "G": "ṅ",
    "J": "ñ",
    "T": "ṭ",
    "D": "ḍ",
    "N": "ṇ",
    "z": "ś",
    "S": "ṣ",
    "M": "ṁ",
    "H": "ḥ",
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "ē",
    "o": "ō",
    "k": "k",
    "g": "g",
    "c": "c",
    "j": "j",
    "t": "t",
    "d": "d",
    "n": "n",
    "p": "p",
    "b": "b",
    "m": "m",
    "y": "y",
    "r": "r",
    "l": "l",
    "v": "v",
    "s": "s",
    "h": "h",
}

_SCHEME_ALIASES = {
    "iso": "iso15919",
    "iso15919": "iso15919",
    "iso-15919": "iso15919",
    "itrans": "itrans",
    "harvardkyoto": "harvard-kyoto",
    "harvard-kyoto": "harvard-kyoto",
    "hk": "harvard-kyoto",
}

_TARGET_CONSONANT_ALIASES = {
    "Tamil": {
        "kh": "k",
        "g": "k",
        "gh": "k",
        "ch": "c",
        "jh": "j",
        "ṭh": "ṭ",
        "ḍ": "ṭ",
        "ḍh": "ṭ",
        "th": "t",
        "d": "t",
        "dh": "t",
        "ph": "p",
        "b": "p",
        "bh": "p",
    }
}

_HOMORGANIC_NASAL = {
    "k": "ṅ",
    "kh": "ṅ",
    "g": "ṅ",
    "gh": "ṅ",
    "c": "ñ",
    "ch": "ñ",
    "j": "ñ",
    "jh": "ñ",
    "ṭ": "ṇ",
    "ṭh": "ṇ",
    "ḍ": "ṇ",
    "ḍh": "ṇ",
    "t": "n",
    "th": "n",
    "d": "n",
    "dh": "n",
    "p": "m",
    "ph": "m",
    "b": "m",
    "bh": "m",
}


@dataclass(frozen=True)
class TransliterationResult:
    """ISO 15919 text plus an explicit output-to-source offset map."""

    text: str
    source_length: int
    offset_starts: tuple[int, ...]
    offset_ends: tuple[int, ...]
    source_scripts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.text) != len(self.offset_starts) or len(self.text) != len(
            self.offset_ends
        ):
            raise ValueError("offset map length must match transliterated text")

    def remap_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a half-open ISO-output span back to its source-text span."""

        safe_start = max(0, min(int(start), len(self.text)))
        safe_end = max(safe_start, min(int(end), len(self.text)))
        if safe_start == safe_end:
            if safe_start >= len(self.offset_starts):
                return self.source_length, self.source_length
            origin = self.offset_starts[safe_start]
            return origin, origin
        return (
            self.offset_starts[safe_start],
            max(self.offset_ends[safe_start:safe_end]),
        )


@dataclass(frozen=True)
class _Piece:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _ScriptTable:
    script: str
    base: int
    independent_vowels: dict[str, str]
    dependent_vowels: dict[str, str]
    consonants: dict[str, str]
    marks: dict[str, str]
    virama: str
    nukta: str | None
    independent_by_iso: dict[str, str]
    dependent_by_iso: dict[str, str]
    consonant_by_iso: dict[str, str]
    mark_by_iso: dict[str, str]


def _assigned_mapping(base: int, offsets: dict[int, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for offset, pivot in offsets.items():
        char = chr(base + offset)
        if unicodedata.category(char)[0] in {"L", "M"}:
            mapping[char] = pivot
    return mapping


def _reverse_first(mapping: dict[str, str]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    for char, pivot in mapping.items():
        reverse.setdefault(pivot, char)
    return reverse


def _build_table(script: str, base: int) -> _ScriptTable:
    independent = _assigned_mapping(base, _INDEPENDENT_VOWEL_OFFSETS)
    dependent = _assigned_mapping(base, _DEPENDENT_VOWEL_OFFSETS)
    consonants = _assigned_mapping(base, _CONSONANT_OFFSETS)
    marks = _assigned_mapping(base, _MARK_OFFSETS)
    if script == "Tamil":
        marks[chr(base + 0x03)] = "ḵ"
    return _ScriptTable(
        script=script,
        base=base,
        independent_vowels=independent,
        dependent_vowels=dependent,
        consonants=consonants,
        marks=marks,
        virama=chr(base + 0x4D),
        nukta=chr(base + 0x3C)
        if unicodedata.category(chr(base + 0x3C)).startswith("M")
        else None,
        independent_by_iso=_reverse_first(independent),
        dependent_by_iso=_reverse_first(dependent),
        consonant_by_iso=_reverse_first(consonants),
        mark_by_iso=_reverse_first(marks),
    )


_TABLES = {script: _build_table(script, base) for script, base in _SCRIPT_BASES.items()}


def to_latin(
    text: str,
    script: str | None = None,
    *,
    scheme: str = "iso15919",
    schwa_policy: SchwaPolicy = "preserve",
    anusvara_policy: AnusvaraPolicy = "marker",
) -> TransliterationResult:
    """Transliterate Indic or romanized text to the ISO 15919 pivot.

    Args:
        text: Source text, including mixed Indic and Latin runs.
        script: Optional source script. Indic script names, ``"Latin"``,
            ``"ITRANS"``, and ``"Harvard-Kyoto"`` are accepted. When omitted,
            each Indic run is detected independently.
        scheme: Romanized input scheme for Latin runs.
        schwa_policy: ``"preserve"`` for lossless orthographic output,
            ``"word-final"`` to drop final inherent vowels, or ``"source"``
            to apply word-final deletion only to northern scripts.
        anusvara_policy: Keep anusvara as ``ṁ`` or expand it to the following
            consonant's homorganic nasal.

    Returns:
        Transliteration text with an output-to-source offset map.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if schwa_policy not in {"preserve", "word-final", "source"}:
        raise ValueError("schwa_policy must be preserve, word-final, or source")
    if anusvara_policy not in {"marker", "homorganic"}:
        raise ValueError("anusvara_policy must be marker or homorganic")

    source_script, source_scheme = _resolve_source(script, scheme)
    pieces: list[_Piece] = []
    scripts_seen: list[str] = []
    index = 0
    while index < len(text):
        detected = _indic_script_for_char(text[index])
        if source_script in _TABLES and detected is None:
            detected = None
        elif source_script in _TABLES:
            detected = source_script

        if detected is not None:
            end = index + 1
            while end < len(text):
                next_script = _indic_script_for_char(text[end])
                if source_script in _TABLES:
                    if next_script != source_script:
                        break
                elif next_script != detected:
                    break
                end += 1
            pieces.extend(
                _romanize_indic_run(
                    text[index:end],
                    index,
                    _TABLES[detected],
                    schwa_policy=schwa_policy,
                    anusvara_policy=anusvara_policy,
                )
            )
            if detected not in scripts_seen:
                scripts_seen.append(detected)
            index = end
            continue

        if _is_latin_char(text[index]) or (
            unicodedata.category(text[index]).startswith("M") and pieces
        ):
            end = index + 1
            while end < len(text) and (
                _is_latin_char(text[end])
                or unicodedata.category(text[end]).startswith("M")
            ):
                end += 1
            pieces.extend(
                _romanized_pieces(text[index:end], index, scheme=source_scheme)
            )
            if "Latin" not in scripts_seen:
                scripts_seen.append("Latin")
            index = end
            continue

        pieces.append(_Piece(text[index], index, index + 1))
        index += 1

    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for piece in pieces:
        normalized = unicodedata.normalize("NFC", piece.text)
        output.append(normalized)
        starts.extend([piece.start] * len(normalized))
        ends.extend([piece.end] * len(normalized))
    return TransliterationResult(
        text="".join(output),
        source_length=len(text),
        offset_starts=tuple(starts),
        offset_ends=tuple(ends),
        source_scripts=tuple(scripts_seen),
    )


def from_latin(
    text: str,
    target_script: str,
    *,
    scheme: str = "iso15919",
) -> str:
    """Transliterate ISO 15919, ITRANS, or Harvard-Kyoto text to an Indic script.

    Args:
        text: Romanized source text.
        target_script: One of :data:`INDIC_SCRIPTS`.
        scheme: Input romanization scheme.

    Returns:
        NFC-normalized target-script text.
    """

    script = _normalize_indic_script(target_script)
    iso_text = romanized_to_iso15919(text, scheme=scheme)
    tokens = _tokenize_iso(iso_text)
    table = _TABLES[script]
    output: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _ISO_CONSONANTS:
            output.append(_target_consonant(token, table))
            if index + 1 < len(tokens) and tokens[index + 1] in _ISO_VOWELS:
                vowel = tokens[index + 1]
                if vowel != "a":
                    sign = table.dependent_by_iso.get(vowel)
                    if sign is None:
                        raise ValueError(
                            f"{script} has no dependent vowel for ISO token {vowel!r}"
                        )
                    output.append(sign)
                index += 2
                continue
            output.append(table.virama)
            index += 1
            continue
        if token in _ISO_VOWELS:
            independent_vowel = table.independent_by_iso.get(token)
            if independent_vowel is None:
                raise ValueError(
                    f"{script} has no independent vowel for ISO token {token!r}"
                )
            output.append(independent_vowel)
            index += 1
            continue
        if token in _ISO_MARKS:
            mark = table.mark_by_iso.get("ṁ" if token == "ṃ" else token)
            if mark is None:
                raise ValueError(f"{script} has no mark for ISO token {token!r}")
            output.append(mark)
            index += 1
            continue
        output.append(token)
        index += 1
    return unicodedata.normalize("NFC", "".join(output))


def transliterate(
    text: str,
    target_script: str,
    *,
    source_script: str | None = None,
    scheme: str = "iso15919",
    schwa_policy: SchwaPolicy = "preserve",
    anusvara_policy: AnusvaraPolicy = "marker",
) -> str:
    """Transliterate text between supported scripts through ISO 15919.

    Args:
        text: Source text.
        target_script: Indic script name or ``"Latin"``.
        source_script: Optional explicit source script.
        scheme: Romanized input scheme when the source is Latin.
        schwa_policy: Inherent-vowel handling policy.
        anusvara_policy: Anusvara handling policy.

    Returns:
        Transliterated text.
    """

    pivot = to_latin(
        text,
        source_script,
        scheme=scheme,
        schwa_policy=schwa_policy,
        anusvara_policy=anusvara_policy,
    ).text
    if _normalized_name(target_script) in {"latin", "iso", "iso15919"}:
        return pivot
    return from_latin(pivot, target_script)


def romanized_to_iso15919(text: str, *, scheme: str = "iso15919") -> str:
    """Normalize ISO 15919, ITRANS, or Harvard-Kyoto input to ISO 15919.

    Args:
        text: Romanized text.
        scheme: Source romanization scheme.

    Returns:
        NFC-normalized ISO 15919 text.
    """

    return unicodedata.normalize(
        "NFC",
        "".join(piece.text for piece in _romanized_pieces(text, 0, scheme=scheme)),
    )


def transliteration_key(
    text: str,
    script: str | None = None,
    *,
    scheme: str = "iso15919",
) -> str:
    """Return a deterministic cross-script join key for an in-memory surface.

    The returned value contains normalized source-equivalent text and is not a
    privacy-preserving digest. Persisted callers should HMAC it with a
    deployment secret, as :class:`openmed.core.surrogate_vault.SurrogateVault`
    does.

    Args:
        text: Indic or romanized source surface.
        script: Optional source script or romanization scheme.
        scheme: Romanized input scheme when ``script`` does not name one.

    Returns:
        An ``iso15919:``-prefixed canonical key.
    """

    pivot = to_latin(text, script, scheme=scheme).text
    pivot = unicodedata.normalize("NFC", pivot).casefold()
    pivot = pivot.replace("ṃ", "ṁ")
    pivot = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", pivot)
    pivot = " ".join(pivot.split())
    return f"iso15919:{pivot}"


def _romanize_indic_run(
    run: str,
    source_start: int,
    table: _ScriptTable,
    *,
    schwa_policy: SchwaPolicy,
    anusvara_policy: AnusvaraPolicy,
) -> list[_Piece]:
    pieces: list[_Piece] = []
    index = 0
    while index < len(run):
        char = run[index]
        absolute = source_start + index
        consonant = table.consonants.get(char)
        if consonant is not None:
            consumed_end = index + 1
            if (
                table.nukta is not None
                and consumed_end < len(run)
                and run[consumed_end] == table.nukta
            ):
                consonant += "̣"
                consumed_end += 1
            next_char = run[consumed_end] if consumed_end < len(run) else None
            if next_char == table.virama:
                pieces.append(
                    _Piece(
                        consonant,
                        absolute,
                        source_start + consumed_end + 1,
                    )
                )
                index = consumed_end + 1
                continue
            pieces.append(_Piece(consonant, absolute, source_start + consumed_end))
            if next_char in table.dependent_vowels:
                pieces.append(
                    _Piece(
                        table.dependent_vowels[next_char],
                        source_start + consumed_end,
                        source_start + consumed_end + 1,
                    )
                )
                index = consumed_end + 1
                continue
            if not _delete_inherent_vowel(
                table.script, run, consumed_end, schwa_policy
            ):
                pieces.append(_Piece("a", absolute, source_start + consumed_end))
            index = consumed_end
            continue

        independent = table.independent_vowels.get(char)
        if independent is not None:
            pieces.append(_Piece(independent, absolute, absolute + 1))
            index += 1
            continue

        mark = table.marks.get(char)
        if mark is not None:
            if mark == "ṁ" and anusvara_policy == "homorganic":
                following = _following_consonant(run, index + 1, table)
                mark = _HOMORGANIC_NASAL.get(following or "", mark)
            pieces.append(_Piece(mark, absolute, absolute + 1))
            index += 1
            continue

        digit = ord(char) - (table.base + 0x66)
        if 0 <= digit <= 9:
            pieces.append(_Piece(str(digit), absolute, absolute + 1))
        elif table.script == "Gurmukhi" and char == chr(table.base + 0x71):
            following = _following_consonant(run, index + 1, table)
            pieces.append(_Piece(following or "", absolute, absolute + 1))
        elif char not in {table.virama, table.nukta}:
            pieces.append(_Piece(char, absolute, absolute + 1))
        index += 1
    return pieces


def _delete_inherent_vowel(
    script: str,
    run: str,
    next_index: int,
    policy: SchwaPolicy,
) -> bool:
    if any(unicodedata.category(char)[0] in {"L", "M"} for char in run[next_index:]):
        return False
    if policy == "word-final":
        return True
    return policy == "source" and script in _NORTHERN_SCHWA_SCRIPTS


def _following_consonant(run: str, index: int, table: _ScriptTable) -> str | None:
    while index < len(run) and run[index] in {table.virama, table.nukta}:
        index += 1
    if index >= len(run):
        return None
    return table.consonants.get(run[index])


def _romanized_pieces(text: str, source_start: int, *, scheme: str) -> list[_Piece]:
    normalized_scheme = _normalize_scheme(scheme)
    if normalized_scheme == "iso15919":
        pieces: list[_Piece] = []
        index = 0
        while index < len(text):
            end = index + 1
            while end < len(text) and unicodedata.combining(text[end]):
                end += 1
            folded = unicodedata.normalize("NFC", text[index:end].casefold())
            pieces.append(_Piece(folded, source_start + index, source_start + end))
            index = end
        return pieces

    mapping = _ITRANS_TO_ISO if normalized_scheme == "itrans" else _HK_TO_ISO
    source_tokens = tuple(sorted(mapping, key=len, reverse=True))
    pieces = []
    index = 0
    while index < len(text):
        token = next(
            (
                candidate
                for candidate in source_tokens
                if text.startswith(candidate, index)
            ),
            None,
        )
        if token is None:
            pieces.append(
                _Piece(
                    text[index].casefold(),
                    source_start + index,
                    source_start + index + 1,
                )
            )
            index += 1
            continue
        pieces.append(
            _Piece(
                mapping[token],
                source_start + index,
                source_start + index + len(token),
            )
        )
        index += len(token)
    return pieces


def _tokenize_iso(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold().replace("ṃ", "ṁ")
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        token = next(
            (
                candidate
                for candidate in _ISO_TOKENS
                if normalized.startswith(candidate, index)
            ),
            None,
        )
        if token is None:
            tokens.append(normalized[index])
            index += 1
            continue
        tokens.append(token)
        index += len(token)
    return tokens


def _target_consonant(token: str, table: _ScriptTable) -> str:
    consonant = table.consonant_by_iso.get(token)
    if consonant is not None:
        return consonant
    alias = _TARGET_CONSONANT_ALIASES.get(table.script, {}).get(token)
    if alias is not None and alias in table.consonant_by_iso:
        return table.consonant_by_iso[alias]
    raise ValueError(f"{table.script} has no consonant for ISO token {token!r}")


def _indic_script_for_char(char: str) -> str | None:
    codepoint = ord(char)
    for script, base in _SCRIPT_BASES.items():
        if base <= codepoint <= base + 0x7F:
            return script
    return None


def _is_latin_char(char: str) -> bool:
    return "LATIN" in unicodedata.name(char, "")


def _resolve_source(script: str | None, scheme: str) -> tuple[str | None, str]:
    normalized_scheme = _normalize_scheme(scheme)
    if script is None:
        return None, normalized_scheme
    name = _normalized_name(script)
    for indic_script in INDIC_SCRIPTS:
        if name == _normalized_name(indic_script) or (
            name == "oriya" and indic_script == "Odia"
        ):
            return indic_script, normalized_scheme
    if name == "latin":
        return "Latin", normalized_scheme
    if name in _SCHEME_ALIASES:
        return "Latin", _SCHEME_ALIASES[name]
    raise ValueError(f"unsupported source script or scheme: {script!r}")


def _normalize_indic_script(script: str) -> str:
    name = _normalized_name(script)
    for supported in INDIC_SCRIPTS:
        if name == _normalized_name(supported) or (
            name == "oriya" and supported == "Odia"
        ):
            return supported
    raise ValueError(
        f"unsupported target script {script!r}; expected one of {INDIC_SCRIPTS!r}"
    )


def _normalize_scheme(scheme: str) -> RomanizationScheme:
    name = _normalized_name(scheme)
    normalized = _SCHEME_ALIASES.get(name)
    if normalized is None:
        raise ValueError("scheme must be iso15919, itrans, or harvard-kyoto")
    return cast(RomanizationScheme, normalized)


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


__all__ = [
    "AnusvaraPolicy",
    "INDIC_SCRIPTS",
    "LOSSY_CASES",
    "RomanizationScheme",
    "SchwaPolicy",
    "TransliterationResult",
    "from_latin",
    "romanized_to_iso15919",
    "to_latin",
    "transliterate",
    "transliteration_key",
]
