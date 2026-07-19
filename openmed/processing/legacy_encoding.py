"""Convert legacy Devanagari encodings to Unicode with byte offsets.

The ISCII assignments in this module follow the public Indian standard
IS 13194:1991 and the Unicode Consortium's published description of ISCII
semantics.  The tables were constructed for OpenMed from those public code
assignments; they are original project data and are distributed under
OpenMed's Apache-2.0 license.

No proprietary legacy-font table is bundled.  :class:`LegacyFontMap` loads a
caller-supplied JSON or YAML data file instead.  This keeps font-specific
licensing and provenance with the user who is entitled to the source font or
mapping data.

Vedic stress marks are decoded when their standard ISCII extension sequences
are encountered, but full Vedic round-trip coverage is outside this module's
lossless guarantee.  The guarantee covers the standard non-Vedic Devanagari
repertoire represented below, excluding contextual ``INV`` presentation
controls.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import yaml

from .text import normalize_indic_text

LegacyEncoding = Literal["unicode", "iscii", "legacy-font"]
ErrorMode = Literal["strict", "replace"]

ISCII_MAPPING_PROVENANCE = (
    "IS 13194:1991 public-standard assignments; OpenMed-original table, Apache-2.0"
)

_ISCII_ATR = 0xEF
_ISCII_EXT = 0xF0
_ISCII_DEVANAGARI = 0x42
_ISCII_INV = 0xD9
_ISCII_HALANT = 0xE8
_ISCII_NUKTA = 0xE9

# One-byte ISCII-1991 Devanagari assignments. ASCII remains ASCII and is not
# duplicated here. Context-sensitive two-byte forms are listed separately.
_ISCII_TO_UNICODE: dict[int, str] = {
    0xA0: "\u00a0",
    0xA1: "\u0901",
    0xA2: "\u0902",
    0xA3: "\u0903",
    0xA4: "\u0905",
    0xA5: "\u0906",
    0xA6: "\u0907",
    0xA7: "\u0908",
    0xA8: "\u0909",
    0xA9: "\u090a",
    0xAA: "\u090b",
    0xAB: "\u090e",
    0xAC: "\u090f",
    0xAD: "\u0910",
    0xAE: "\u090d",
    0xAF: "\u0912",
    0xB0: "\u0913",
    0xB1: "\u0914",
    0xB2: "\u0911",
    0xB3: "\u0915",
    0xB4: "\u0916",
    0xB5: "\u0917",
    0xB6: "\u0918",
    0xB7: "\u0919",
    0xB8: "\u091a",
    0xB9: "\u091b",
    0xBA: "\u091c",
    0xBB: "\u091d",
    0xBC: "\u091e",
    0xBD: "\u091f",
    0xBE: "\u0920",
    0xBF: "\u0921",
    0xC0: "\u0922",
    0xC1: "\u0923",
    0xC2: "\u0924",
    0xC3: "\u0925",
    0xC4: "\u0926",
    0xC5: "\u0927",
    0xC6: "\u0928",
    0xC7: "\u0929",
    0xC8: "\u092a",
    0xC9: "\u092b",
    0xCA: "\u092c",
    0xCB: "\u092d",
    0xCC: "\u092e",
    0xCD: "\u092f",
    0xCE: "\u095f",
    0xCF: "\u0930",
    0xD0: "\u0931",
    0xD1: "\u0932",
    0xD2: "\u0933",
    0xD3: "\u0934",
    0xD4: "\u0935",
    0xD5: "\u0936",
    0xD6: "\u0937",
    0xD7: "\u0938",
    0xD8: "\u0939",
    0xD9: "\u200d",
    0xDA: "\u093e",
    0xDB: "\u093f",
    0xDC: "\u0940",
    0xDD: "\u0941",
    0xDE: "\u0942",
    0xDF: "\u0943",
    0xE0: "\u0946",
    0xE1: "\u0947",
    0xE2: "\u0948",
    0xE3: "\u0945",
    0xE4: "\u094a",
    0xE5: "\u094b",
    0xE6: "\u094c",
    0xE7: "\u0949",
    0xE8: "\u094d",
    0xE9: "\u093c",
    0xEA: "\u0964",
    0xF1: "\u0966",
    0xF2: "\u0967",
    0xF3: "\u0968",
    0xF4: "\u0969",
    0xF5: "\u096a",
    0xF6: "\u096b",
    0xF7: "\u096c",
    0xF8: "\u096d",
    0xF9: "\u096e",
    0xFA: "\u096f",
}

_ISCII_SEQUENCE_TO_UNICODE: dict[bytes, str] = {
    b"\xa4\xe0": "\u0904",
    b"\xa6\xe9": "\u090c",
    b"\xa1\xe9": "\u0950",
    b"\xaa\xe9": "\u0960",
    b"\xa7\xe9": "\u0961",
    b"\xb3\xe9": "\u0958",
    b"\xb4\xe9": "\u0959",
    b"\xb5\xe9": "\u095a",
    b"\xba\xe9": "\u095b",
    b"\xbf\xe9": "\u095c",
    b"\xc0\xe9": "\u095d",
    b"\xc9\xe9": "\u095e",
    b"\xdb\xe9": "\u0962",
    b"\xdc\xe9": "\u0963",
    b"\xdf\xe9": "\u0944",
    b"\xea\xe9": "\u093d",
    b"\xea\xea": "\u0965",
}

_ISCII_EXTENSION_TO_UNICODE = {
    0xB5: "\u0951",  # Vedic: Udatta (documented limitation).
    0xB8: "\u0952",  # Vedic: Anudatta (documented limitation).
    0xBF: "\u0970",  # Devanagari abbreviation sign.
}

_UNICODE_SPECIAL_TO_ISCII: dict[str, bytes] = {
    "\u0904": b"\xa4\xe0",
    "\u090c": b"\xa6\xe9",
    "\u093d": b"\xea\xe9",
    "\u0944": b"\xdf\xe9",
    "\u0950": b"\xa1\xe9",
    "\u0951": b"\xf0\xb5",
    "\u0952": b"\xf0\xb8",
    "\u0960": b"\xaa\xe9",
    "\u0961": b"\xa7\xe9",
    "\u0962": b"\xdb\xe9",
    "\u0963": b"\xdc\xe9",
    "\u0965": b"\xea\xea",
    "\u0970": b"\xf0\xbf",
}

# U+095F has a canonical decomposition to YA + NUKTA under NFC, while ISCII
# assigns the letter its own byte. Prefer that canonical byte when encoding.
_UNICODE_SEQUENCE_TO_ISCII = {"\u092f\u093c": b"\xce"}

_UNICODE_TO_ISCII = {
    unicode_char: iscii_byte
    for iscii_byte, unicode_char in _ISCII_TO_UNICODE.items()
    if iscii_byte != _ISCII_INV
}

_ISCII_CORE_LETTERS = frozenset(range(0xB3, 0xD9))
_ISCII_UNDEFINED = frozenset(range(0xEB, 0xEF)) | frozenset(range(0xFB, 0x100))
_LEGACY_GAP_BYTES = frozenset(b" \t\r\n-_/.,:;()[]{}")


@dataclass(frozen=True)
class ConversionOffsetMap:
    """Map converted character spans to original byte spans and back."""

    original_length: int
    converted_to_original_spans: tuple[tuple[int, int], ...]
    original_to_converted: tuple[int | None, ...]

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map converted ``[start, end)`` offsets to source-byte offsets."""

        length = len(self.converted_to_original_spans)
        if not (0 <= start <= end <= length):
            raise ValueError("span must satisfy 0 <= start <= end <= len(text)")
        if start == end:
            if start < length:
                anchor = self.converted_to_original_spans[start][0]
            elif length:
                anchor = self.converted_to_original_spans[-1][1]
            else:
                anchor = 0
            return anchor, anchor
        spans = self.converted_to_original_spans[start:end]
        return min(item[0] for item in spans), max(item[1] for item in spans)

    def to_converted_span(self, start: int, end: int) -> tuple[int, int]:
        """Map source-byte ``[start, end)`` offsets to converted offsets."""

        if not (0 <= start <= end <= self.original_length):
            raise ValueError("span must satisfy 0 <= start <= end <= original_length")
        mapped = [
            index
            for index, (source_start, source_end) in enumerate(
                self.converted_to_original_spans
            )
            if source_start < end and source_end > start
        ]
        if mapped:
            return min(mapped), max(mapped) + 1
        cursor = start
        while cursor < self.original_length:
            value = self.original_to_converted[cursor]
            if value is not None:
                return value, value
            cursor += 1
        terminal = len(self.converted_to_original_spans)
        return terminal, terminal


@dataclass(frozen=True)
class LegacyConversion:
    """Unicode conversion result and alignment to the original byte stream."""

    text: str
    original: bytes
    encoding: LegacyEncoding
    offset_map: ConversionOffsetMap
    changed: bool

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a converted span to the original byte stream."""

        return self.offset_map.to_original_span(start, end)


@dataclass(frozen=True)
class LegacyFontMap:
    """Caller-supplied byte-to-Unicode mapping for one legacy font."""

    name: str
    mapping: Mapping[int, str]
    provenance: str = "user-supplied"

    def __post_init__(self) -> None:
        """Validate and freeze the mapping."""

        if not self.name.strip():
            raise ValueError("legacy font map name must not be empty")
        validated: dict[int, str] = {}
        for key, value in self.mapping.items():
            if isinstance(key, bool) or not isinstance(key, int) or not 0 <= key <= 255:
                raise ValueError("legacy font map keys must be byte values 0..255")
            if not isinstance(value, str) or not value:
                raise ValueError("legacy font map values must be non-empty strings")
            validated[key] = value
        if not validated:
            raise ValueError("legacy font map must contain at least one mapping")
        object.__setattr__(self, "mapping", MappingProxyType(validated))

    @classmethod
    def from_file(cls, path: str | Path) -> "LegacyFontMap":
        """Load a JSON or YAML legacy-font map.

        The file may contain a top-level ``mapping`` object plus optional
        ``name`` and ``provenance`` fields, or it may be the mapping object
        itself. Keys may be integers, one-byte characters, decimal strings, or
        hexadecimal strings such as ``"0x66"``.

        Args:
            path: JSON, YAML, or YML mapping file.

        Returns:
            Validated immutable legacy-font map.
        """

        source = Path(path)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        elif source.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        else:
            raise ValueError("legacy font map must be JSON, YAML, or YML")
        if not isinstance(payload, Mapping):
            raise ValueError("legacy font map file must contain an object")

        raw_mapping = payload.get("mapping", payload)
        if not isinstance(raw_mapping, Mapping):
            raise ValueError("legacy font map 'mapping' must be an object")
        mapping: dict[int, str] = {}
        for key, value in raw_mapping.items():
            byte_key = _parse_map_key(key)
            if byte_key in mapping:
                raise ValueError(f"duplicate legacy font map byte: {byte_key}")
            if not isinstance(value, str):
                raise ValueError("legacy font map values must be strings")
            mapping[byte_key] = value
        name = str(payload.get("name", source.stem))
        provenance = str(payload.get("provenance", f"user-supplied:{source.name}"))
        return cls(name=name, mapping=mapping, provenance=provenance)


def detect_legacy_encoding(
    data: bytes | str,
    *,
    legacy_font_map: LegacyFontMap | None = None,
) -> LegacyEncoding:
    """Conservatively detect ISCII or a supplied ASCII-remapped font.

    A Devanagari ISCII attribute sequence is definitive. Otherwise at least
    two valid high bytes, including a core ISCII consonant, are required. A
    legacy-font candidate needs a dense run of at least three mapped bytes and
    must produce Devanagari letters. These gates intentionally prefer a false
    negative over corrupting an ordinary Latin clinical note.
    """

    raw = _coerce_legacy_bytes(data)
    if raw is None:
        return "unicode"
    if _is_non_ascii_utf8(raw):
        return "unicode"
    if b"\xef\x42" in raw:
        return "iscii"

    high_bytes = [byte for byte in raw if byte >= 0xA0]
    valid_high = [byte for byte in high_bytes if byte in _ISCII_TO_UNICODE]
    if (
        len(valid_high) >= 2
        and len(valid_high) / max(1, len(high_bytes)) >= 0.8
        and any(byte in _ISCII_CORE_LETTERS for byte in valid_high)
        and not any(0x80 <= byte <= 0x9F for byte in raw)
        and not any(byte in _ISCII_UNDEFINED for byte in high_bytes)
    ):
        return "iscii"

    if legacy_font_map is not None and _legacy_font_candidate_runs(
        raw, legacy_font_map
    ):
        return "legacy-font"
    return "unicode"


def iscii_to_unicode(
    data: bytes,
    *,
    errors: ErrorMode = "strict",
) -> LegacyConversion:
    """Convert ISCII-1991 Devanagari bytes to NFC Unicode.

    Contextual nukta forms, explicit/soft virama controls, Devanagari script
    attributes, double danda, and supported extension sequences are handled.
    Every output character retains the source-byte span that produced it.
    """

    _validate_errors(errors)
    chars: list[str] = []
    origins: list[tuple[int, int]] = []
    index = 0
    while index < len(data):
        byte = data[index]
        if byte < 0x80:
            chars.append(chr(byte))
            origins.append((index, index + 1))
            index += 1
            continue

        if byte == _ISCII_ATR:
            if index + 1 >= len(data):
                index = _decode_error(data, index, errors, chars, origins)
                continue
            if data[index + 1] != _ISCII_DEVANAGARI:
                index = _decode_error(data, index, errors, chars, origins, length=2)
                continue
            index += 2
            continue

        if byte == _ISCII_EXT:
            if index + 1 >= len(data):
                index = _decode_error(data, index, errors, chars, origins)
                continue
            extension = _ISCII_EXTENSION_TO_UNICODE.get(data[index + 1])
            if extension is None:
                index = _decode_error(data, index, errors, chars, origins, length=2)
                continue
            chars.append(extension)
            origins.append((index, index + 2))
            index += 2
            continue

        pair = data[index : index + 2]
        if pair == b"\xe8\xe8":
            chars.extend(("\u094d", "\u200c"))
            origins.extend(((index, index + 1), (index + 1, index + 2)))
            index += 2
            continue
        if pair == b"\xe8\xe9":
            chars.extend(("\u094d", "\u200d"))
            origins.extend(((index, index + 1), (index + 1, index + 2)))
            index += 2
            continue
        sequence_char = _ISCII_SEQUENCE_TO_UNICODE.get(pair)
        if sequence_char is not None:
            chars.append(sequence_char)
            origins.append((index, index + 2))
            index += 2
            continue

        unicode_char = _ISCII_TO_UNICODE.get(byte)
        if unicode_char is None:
            index = _decode_error(data, index, errors, chars, origins)
            continue
        chars.append(unicode_char)
        origins.append((index, index + 1))
        index += 1

    normalization = normalize_indic_text("".join(chars), char_origins=tuple(origins))
    offset_map = _build_offset_map(len(data), normalization.char_origins)
    return LegacyConversion(
        text=normalization.text,
        original=bytes(data),
        encoding="iscii",
        offset_map=offset_map,
        changed=True,
    )


def unicode_to_iscii(
    text: str,
    *,
    errors: ErrorMode = "strict",
    include_atr: bool = False,
) -> bytes:
    """Encode Unicode Devanagari to ISCII-1991 bytes.

    Args:
        text: Unicode text containing ASCII and supported Devanagari.
        errors: ``"strict"`` raises :class:`UnicodeEncodeError`; ``"replace"``
            emits ASCII ``?`` for unsupported code points.
        include_atr: Prefix the Devanagari script attribute ``EF 42``.

    Returns:
        ISCII byte stream.
    """

    _validate_errors(errors)
    normalized = normalize_indic_text(text).text
    output = bytearray(b"\xef\x42" if include_atr else b"")
    previous_char = ""
    index = 0
    while index < len(normalized):
        sequence = next(
            (
                (unicode_sequence, iscii_sequence)
                for unicode_sequence, iscii_sequence in _UNICODE_SEQUENCE_TO_ISCII.items()
                if normalized.startswith(unicode_sequence, index)
            ),
            None,
        )
        if sequence is not None:
            unicode_sequence, iscii_sequence = sequence
            output.extend(iscii_sequence)
            previous_char = unicode_sequence[-1]
            index += len(unicode_sequence)
            continue

        char = normalized[index]
        codepoint = ord(char)
        if codepoint < 0x80:
            output.append(codepoint)
        elif char == "\u200c" and previous_char == "\u094d":
            output.append(_ISCII_HALANT)
        elif char == "\u200d":
            output.append(_ISCII_NUKTA if previous_char == "\u094d" else _ISCII_INV)
        elif char in _UNICODE_SPECIAL_TO_ISCII:
            output.extend(_UNICODE_SPECIAL_TO_ISCII[char])
        elif char in _UNICODE_TO_ISCII:
            output.append(_UNICODE_TO_ISCII[char])
        elif errors == "replace":
            output.append(ord("?"))
        else:
            raise UnicodeEncodeError(
                "iscii-dev",
                normalized,
                index,
                index + 1,
                "character is not representable in ISCII Devanagari",
            )
        previous_char = char
        index += 1
    return bytes(output)


def convert_legacy_encoding(
    data: bytes | str,
    *,
    encoding: Literal["auto", "unicode", "iscii", "legacy-font"] = "auto",
    legacy_font_map: LegacyFontMap | None = None,
    errors: ErrorMode = "strict",
) -> LegacyConversion:
    """Convert detected ISCII or caller-mapped legacy-font runs to Unicode.

    ``bytes`` are preferred whenever original-stream byte offsets matter. A
    ``str`` containing Latin-1 code points is accepted for compatibility with
    systems that decoded legacy bytes without conversion; its code-point and
    original-byte offsets are then identical.
    """

    _validate_errors(errors)
    if encoding not in {"auto", "unicode", "iscii", "legacy-font"}:
        raise ValueError(f"unknown legacy encoding: {encoding!r}")

    raw = _coerce_legacy_bytes(data)
    if raw is None:
        if encoding in {"iscii", "legacy-font"}:
            raise ValueError("legacy conversion requires bytes or Latin-1 text")
        assert isinstance(data, str)
        return _unicode_identity(data)

    detected: LegacyEncoding
    if encoding == "auto":
        detected = detect_legacy_encoding(raw, legacy_font_map=legacy_font_map)
    else:
        detected = encoding

    if detected == "iscii":
        return iscii_to_unicode(raw, errors=errors)
    if detected == "legacy-font":
        if legacy_font_map is None:
            raise ValueError("legacy-font conversion requires legacy_font_map")
        return _convert_legacy_font(
            raw,
            legacy_font_map,
            auto_detected=encoding == "auto",
            errors=errors,
        )
    return _decode_unicode(raw, original=data, errors=errors)


def _convert_legacy_font(
    data: bytes,
    font_map: LegacyFontMap,
    *,
    auto_detected: bool,
    errors: ErrorMode,
) -> LegacyConversion:
    candidate_ranges = (
        _legacy_font_candidate_runs(data, font_map)
        if auto_detected
        else ((0, len(data)),)
    )
    chars: list[str] = []
    origins: list[tuple[int, int]] = []
    range_index = 0
    active = candidate_ranges[range_index] if candidate_ranges else None

    for index, byte in enumerate(data):
        while active is not None and index >= active[1]:
            range_index += 1
            active = (
                candidate_ranges[range_index]
                if range_index < len(candidate_ranges)
                else None
            )
        in_candidate = active is not None and active[0] <= index < active[1]
        replacement = font_map.mapping.get(byte) if in_candidate else None
        if replacement is None:
            if byte < 0x80:
                replacement = chr(byte)
            elif errors == "replace":
                replacement = "\ufffd"
            else:
                raise UnicodeDecodeError(
                    f"legacy-font:{font_map.name}",
                    data,
                    index,
                    index + 1,
                    "unmapped non-ASCII byte",
                )
        chars.extend(replacement)
        origins.extend((index, index + 1) for _ in replacement)

    normalization = normalize_indic_text("".join(chars), char_origins=tuple(origins))
    return LegacyConversion(
        text=normalization.text,
        original=data,
        encoding="legacy-font",
        offset_map=_build_offset_map(len(data), normalization.char_origins),
        changed=True,
    )


def _legacy_font_candidate_runs(
    data: bytes,
    font_map: LegacyFontMap,
) -> tuple[tuple[int, int], ...]:
    mapped_positions = [
        index for index, byte in enumerate(data) if byte in font_map.mapping
    ]
    if not mapped_positions:
        return ()

    groups: list[list[int]] = [[mapped_positions[0]]]
    for position in mapped_positions[1:]:
        previous = groups[-1][-1]
        gap = data[previous + 1 : position]
        if len(gap) <= 2 and all(byte in _LEGACY_GAP_BYTES for byte in gap):
            groups[-1].append(position)
        else:
            groups.append([position])

    candidates: list[tuple[int, int]] = []
    for positions in groups:
        start, end = positions[0], positions[-1] + 1
        mapped_count = len(positions)
        if mapped_count < 3 or len({data[index] for index in positions}) < 2:
            continue
        output = "".join(font_map.mapping[data[index]] for index in positions)
        devanagari_letters = sum(0x0904 <= ord(char) <= 0x0939 for char in output)
        alphanumeric = sum(chr(byte).isalnum() for byte in data[start:end])
        coverage = mapped_count / max(1, alphanumeric)
        if (
            devanagari_letters >= 2
            and coverage >= 0.7
            and _looks_suspicious_legacy_cluster(data[start:end])
        ):
            candidates.append((start, end))
    return tuple(candidates)


def _decode_unicode(
    data: bytes,
    *,
    original: bytes | str,
    errors: ErrorMode,
) -> LegacyConversion:
    if isinstance(original, str):
        text = original
        origins = tuple((index, index + 1) for index in range(len(text)))
        original_bytes = data
    else:
        error_handler = "strict" if errors == "strict" else "replace"
        text = data.decode("utf-8", errors=error_handler)
        original_bytes = data
        origins_list: list[tuple[int, int]] = []
        cursor = 0
        for char in text:
            width = len(char.encode("utf-8"))
            origins_list.append((cursor, min(len(data), cursor + width)))
            cursor += width
        origins = tuple(origins_list)
    return LegacyConversion(
        text=text,
        original=original_bytes,
        encoding="unicode",
        offset_map=_build_offset_map(len(original_bytes), origins),
        changed=False,
    )


def _unicode_identity(text: str) -> LegacyConversion:
    data = text.encode("utf-8")
    origins: list[tuple[int, int]] = []
    cursor = 0
    for char in text:
        width = len(char.encode("utf-8"))
        origins.append((cursor, cursor + width))
        cursor += width
    return LegacyConversion(
        text=text,
        original=data,
        encoding="unicode",
        offset_map=_build_offset_map(len(data), tuple(origins)),
        changed=False,
    )


def _build_offset_map(
    original_length: int,
    origins: tuple[tuple[int, int], ...],
) -> ConversionOffsetMap:
    original_to_converted: list[int | None] = [None] * original_length
    for converted_index, (start, end) in enumerate(origins):
        for original_index in range(start, end):
            if original_to_converted[original_index] is None:
                original_to_converted[original_index] = converted_index
    return ConversionOffsetMap(
        original_length=original_length,
        converted_to_original_spans=origins,
        original_to_converted=tuple(original_to_converted),
    )


def _decode_error(
    data: bytes,
    index: int,
    errors: ErrorMode,
    chars: list[str],
    origins: list[tuple[int, int]],
    *,
    length: int = 1,
) -> int:
    end = min(len(data), index + length)
    if errors == "strict":
        raise UnicodeDecodeError(
            "iscii-dev",
            data,
            index,
            end,
            "invalid or unsupported ISCII sequence",
        )
    chars.append("\ufffd")
    origins.append((index, end))
    return end


def _coerce_legacy_bytes(data: bytes | str) -> bytes | None:
    if isinstance(data, bytes):
        return data
    if not isinstance(data, str):
        raise TypeError("legacy input must be bytes or str")
    try:
        return data.encode("latin-1")
    except UnicodeEncodeError:
        return None


def _is_non_ascii_utf8(data: bytes) -> bool:
    if not any(byte >= 0x80 for byte in data):
        return False
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return any(ord(char) >= 0x80 for char in decoded)


def _looks_suspicious_legacy_cluster(data: bytes) -> bool:
    if any(byte >= 0x80 for byte in data):
        return True
    text = data.decode("ascii")
    if any(char in "`~[]{}\\|#$%^&*_+=/<>" for char in text):
        return True

    words = [word for word in _ascii_words(text) if len(word) >= 4]
    if not words:
        return False
    suspicious = 0
    for word in words:
        interior_case_change = any(
            left.islower() and right.isupper() for left, right in zip(word, word[1:])
        )
        vowel_ratio = sum(char.lower() in "aeiou" for char in word) / len(word)
        if interior_case_change or vowel_ratio < 0.2:
            suspicious += 1
    return suspicious / len(words) >= 0.6


def _ascii_words(text: str) -> tuple[str, ...]:
    words: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isalpha():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return tuple(words)


def _parse_map_key(key: object) -> int:
    if isinstance(key, bool):
        raise ValueError("legacy font map keys must identify one byte")
    if isinstance(key, int):
        value = key
    elif isinstance(key, str):
        if len(key) == 1:
            value = ord(key)
        elif key.lower().startswith("0x"):
            value = int(key, 16)
        elif key.isdecimal():
            value = int(key, 10)
        else:
            raise ValueError(f"invalid legacy font map key: {key!r}")
    else:
        raise ValueError("legacy font map keys must identify one byte")
    if not 0 <= value <= 255:
        raise ValueError("legacy font map keys must be byte values 0..255")
    return value


def _validate_errors(errors: str) -> None:
    if errors not in {"strict", "replace"}:
        raise ValueError("errors must be 'strict' or 'replace'")


__all__ = [
    "ConversionOffsetMap",
    "ISCII_MAPPING_PROVENANCE",
    "LegacyConversion",
    "LegacyEncoding",
    "LegacyFontMap",
    "convert_legacy_encoding",
    "detect_legacy_encoding",
    "iscii_to_unicode",
    "unicode_to_iscii",
]
