"""Detect Zawgyi-encoded Burmese and convert it to Unicode with source offsets.

Zawgyi is a widely deployed Burmese font encoding. It reuses code points of the
Unicode Myanmar block (U+1000-U+109F) with different meanings and stores text
in visual rather than logical order, so Zawgyi text decodes as apparently valid
Unicode while breaking Unicode-based segmenters, detectors and models.

Detection is deterministic and model-free. It counts structural signals that
are invalid in Unicode storage order but required by Zawgyi's visual order
(a vowel sign E or medial RA written before its consonant, and the Zawgyi
killer written as U+1039 at the end of a syllable), plus signals that only
Unicode storage produces (vowel sign E followed by a dependent sign, the
U+1004 U+103A U+1039 kinzi sequence, dot below before asat, and code points
the Zawgyi layout never assigns). The smoothed share of Zawgyi evidence is the
reported confidence. Text is converted only when that confidence reaches
:data:`ZAWGYI_CONFIDENCE_THRESHOLD` *and* at least two structural Zawgyi
signals are present. Zawgyi presentation forms (U+1060-U+1097 and friends)
only add weak evidence, because Mon, Shan, Karen and other Myanmar-script
languages use the same code points for their own letters in Unicode.

Provenance: the rule table and detector were written for OpenMed from the
publicly documented Zawgyi-One code-point layout, the Unicode Myanmar block
chart, and the storage order described in Unicode Technical Note #11
("Representing Myanmar in Unicode"). No third-party converter code, rule file
or detector model (including Google's myanmar-tools) is copied or bundled. The
table is original project data distributed under OpenMed's Apache-2.0
license; :data:`ZAWGYI_RULE_PROVENANCE` carries the machine-readable record.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

ZawgyiDecision = Literal["zawgyi", "unicode", "undetermined", "not-myanmar"]

#: Smoothed Zawgyi-evidence share required before text is converted.
ZAWGYI_CONFIDENCE_THRESHOLD = 0.85

#: License and provenance record for the bundled Zawgyi rule data.
ZAWGYI_RULE_PROVENANCE: Mapping[str, str] = MappingProxyType(
    {
        "name": "openmed-zawgyi-unicode-rules",
        "version": "1",
        "license": "Apache-2.0",
        "provenance": (
            "OpenMed-original rule table and detector built from the public "
            "Zawgyi-One code-point layout, the Unicode Myanmar block chart and "
            "Unicode Technical Note #11 storage order"
        ),
        "third_party_code": "none",
    }
)

# Conversion needs at least this many signals that are invalid in Unicode
# storage order. Presentation forms alone never qualify (see module docstring).
_MIN_STRUCTURAL_SIGNALS = 2
_FORM_WEIGHT = 0.5
_SMOOTHING = 0.5

_MYANMAR_RUN_RE = re.compile("[က-႟]+")
_MYANMAR_EXTENDED_RE = re.compile("[ꧠ-꧿ꩠ-ꩿ]")

# Unicode storage order of one Myanmar syllable (UTN #11). Zawgyi units are
# sorted by these slots inside each cluster; ties keep source order.
_KINZI = 0
_BASE = 1
_ASAT_CONTRACTION = 2
_STACK = 3
_MEDIAL_YA = 4
_MEDIAL_RA = 5
_MEDIAL_WA = 6
_MEDIAL_HA = 7
_VOWEL_E = 8
_VOWEL_UPPER = 9
_VOWEL_LOWER = 10
_VOWEL_AA = 11
_ANUSVARA = 12
_DOT_BELOW = 13
_ASAT = 14
_VISARGA = 15

_Units = tuple[tuple[str, int], ...]

_KINZI_UNITS: _Units = (("င", _KINZI), ("်", _KINZI), ("္", _KINZI))


def _stacked(consonant: str) -> _Units:
    return (("္", _STACK), (consonant, _STACK))


# Zawgyi signs typed before their consonant (visual order).
_ZAWGYI_PREBASE: Mapping[int, _Units] = MappingProxyType(
    {
        0x1031: (("ေ", _VOWEL_E),),
        0x103B: (("ြ", _MEDIAL_RA),),
        # Width and gap variants of the medial RA glyph.
        **{codepoint: (("ြ", _MEDIAL_RA),) for codepoint in range(0x107E, 0x1085)},
    }
)

# Zawgyi letter variants and ligatures that start a syllable.
_ZAWGYI_BASES: Mapping[int, _Units] = MappingProxyType(
    {
        0x106A: (("ဉ", _BASE),),
        0x106B: (("ည", _BASE),),
        0x1086: (("ဿ", _BASE),),
        0x108F: (("န", _BASE),),
        0x1090: (("ရ", _BASE),),
        0x106E: (("ဍ", _BASE), *_stacked("ဍ")),
        0x106F: (("ဍ", _BASE), *_stacked("ဎ")),
        0x1091: (("ဏ", _BASE), *_stacked("ဍ")),
        0x1092: (("ဋ", _BASE), *_stacked("ဌ")),
        0x1097: (("ဋ", _BASE), *_stacked("ဋ")),
    }
)

# Zawgyi subscript (stacked) consonant forms.
_ZAWGYI_STACKED_FORMS: Mapping[int, str] = MappingProxyType(
    {
        0x1060: "က",
        0x1061: "ခ",
        0x1062: "ဂ",
        0x1063: "ဃ",
        0x1065: "စ",
        0x1066: "ဆ",
        0x1067: "ဆ",
        0x1068: "ဇ",
        0x1069: "ဈ",
        0x106C: "ဋ",
        0x106D: "ဌ",
        0x1070: "ဏ",
        0x1071: "တ",
        0x1072: "တ",
        0x1073: "ထ",
        0x1074: "ထ",
        0x1075: "ဒ",
        0x1076: "ဓ",
        0x1077: "န",
        0x1078: "ပ",
        0x1079: "ဖ",
        0x107A: "ဗ",
        0x107B: "ဘ",
        0x1093: "ဘ",
        0x107C: "မ",
        0x1085: "လ",
    }
)

# Zawgyi signs typed after their consonant.
_ZAWGYI_MARKS: Mapping[int, _Units] = MappingProxyType(
    {
        0x102B: (("ါ", _VOWEL_AA),),
        0x102C: (("ာ", _VOWEL_AA),),
        0x102D: (("ိ", _VOWEL_UPPER),),
        0x102E: (("ီ", _VOWEL_UPPER),),
        0x1032: (("ဲ", _VOWEL_UPPER),),
        0x102F: (("ု", _VOWEL_LOWER),),
        0x1030: (("ူ", _VOWEL_LOWER),),
        0x1033: (("ု", _VOWEL_LOWER),),
        0x1034: (("ူ", _VOWEL_LOWER),),
        0x1036: (("ံ", _ANUSVARA),),
        0x1037: (("့", _DOT_BELOW),),
        0x1094: (("့", _DOT_BELOW),),
        0x1095: (("့", _DOT_BELOW),),
        0x1038: (("း", _VISARGA),),
        # Zawgyi writes the killer (Unicode asat) at the virama code point.
        0x1039: (("်", _ASAT),),
        0x103A: (("ျ", _MEDIAL_YA),),
        0x107D: (("ျ", _MEDIAL_YA),),
        0x103C: (("ွ", _MEDIAL_WA),),
        0x103D: (("ှ", _MEDIAL_HA),),
        0x1087: (("ှ", _MEDIAL_HA),),
        0x1088: (("ှ", _MEDIAL_HA), ("ု", _VOWEL_LOWER)),
        0x1089: (("ှ", _MEDIAL_HA), ("ူ", _VOWEL_LOWER)),
        0x108A: (("ွ", _MEDIAL_WA), ("ှ", _MEDIAL_HA)),
        0x105A: (("ါ", _VOWEL_AA), ("်", _ASAT)),
        0x108E: (("ိ", _VOWEL_UPPER), ("ံ", _ANUSVARA)),
        0x1064: _KINZI_UNITS,
        0x108B: (*_KINZI_UNITS, ("ိ", _VOWEL_UPPER)),
        0x108C: (*_KINZI_UNITS, ("ီ", _VOWEL_UPPER)),
        0x108D: (*_KINZI_UNITS, ("ံ", _ANUSVARA)),
        0x1096: (*_stacked("တ"), ("ွ", _MEDIAL_WA)),
        **{
            codepoint: _stacked(consonant)
            for codepoint, consonant in _ZAWGYI_STACKED_FORMS.items()
        },
    }
)

_ROLE_PREBASE = 0
_ROLE_BASE = 1
_ROLE_MARK = 2


def _role_for(codepoint: int) -> int:
    if codepoint in _ZAWGYI_PREBASE:
        return _ROLE_PREBASE
    if codepoint in _ZAWGYI_BASES:
        return _ROLE_BASE
    if codepoint in _ZAWGYI_MARKS:
        return _ROLE_MARK
    # Unmapped code points keep their Unicode meaning: letters, digits and
    # punctuation start a cluster, marks attach to the current one.
    if unicodedata.category(chr(codepoint)).startswith("M"):
        return _ROLE_MARK
    return _ROLE_BASE


_ROLES: Mapping[int, int] = MappingProxyType(
    {codepoint: _role_for(codepoint) for codepoint in range(0x1000, 0x10A0)}
)

_CONSONANTS = frozenset(range(0x1000, 0x1022))
# Myanmar letters (``Lo``) that Unicode storage allows before a dependent sign.
_UNICODE_LETTERS = frozenset(
    codepoint
    for codepoint in range(0x1000, 0x10A0)
    if unicodedata.category(chr(codepoint)) == "Lo"
)
_UNICODE_MEDIALS = frozenset(
    {0x103B, 0x103C, 0x103D, 0x103E, 0x105E, 0x105F, 0x1060, 0x1082}
)
_UNICODE_E_HOSTS = _UNICODE_LETTERS | _UNICODE_MEDIALS
_ZAWGYI_PREBASE_RA = frozenset({0x103B, *range(0x107E, 0x1085)})
_ZAWGYI_LETTER_FORMS = frozenset({0x106A, 0x106B, 0x108F, 0x1090})
_ZAWGYI_E_FOLLOWERS = _CONSONANTS | _ZAWGYI_PREBASE_RA | _ZAWGYI_LETTER_FORMS
# Digits ZERO and SEVEN are habitually typed for WA and RA; never treat a
# vowel sign E before them as Unicode-only evidence.
_AMBIGUOUS_E_FOLLOWERS = _ZAWGYI_E_FOLLOWERS | {0x1040, 0x1047}
_ZAWGYI_PRESENTATION_FORMS = frozenset({0x1033, 0x1034, 0x105A, *range(0x1060, 0x1098)})
# Code points the Zawgyi layout does not assign; they only occur in Unicode.
_UNICODE_ONLY = frozenset({0x1022, 0x1028, 0x1035, 0x103E, *range(0x1098, 0x10A0)})
_LOWER_VOWELS = frozenset({0x102F, 0x1030, 0x1033, 0x1034})


@dataclass(frozen=True)
class ZawgyiDetection:
    """Content-free Zawgyi detection decision with its evidence counts.

    Attributes:
        decision: ``"zawgyi"`` when Zawgyi is suspected with at least
            ``threshold`` confidence, ``"unicode"`` when Unicode Burmese is
            confirmed with the same margin, ``"undetermined"`` for short,
            mixed or otherwise ambiguous Myanmar text, and ``"not-myanmar"``
            when the text has no Myanmar-script code points.
        confidence: Smoothed share of Zawgyi evidence in ``[0, 1]``.
        threshold: Confidence margin required for a definite decision.
        myanmar_chars: Myanmar-script code points examined.
        zawgyi_signals: Structural signals that are invalid in Unicode order.
        zawgyi_forms: Zawgyi presentation-form code points (weak evidence).
        unicode_signals: Signals produced only by Unicode storage order.
    """

    decision: ZawgyiDecision
    confidence: float
    threshold: float
    myanmar_chars: int
    zawgyi_signals: int
    zawgyi_forms: int
    unicode_signals: int

    @property
    def suspected(self) -> bool:
        """Return whether the text should be treated as Zawgyi."""

        return self.decision == "zawgyi"

    def to_metadata(self) -> dict[str, object]:
        """Return the decision as PHI-free codes and counts."""

        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "myanmar_chars": self.myanmar_chars,
            "zawgyi_signals": self.zawgyi_signals,
            "zawgyi_forms": self.zawgyi_forms,
            "unicode_signals": self.unicode_signals,
        }


def detect_zawgyi(
    text: str,
    *,
    threshold: float = ZAWGYI_CONFIDENCE_THRESHOLD,
) -> ZawgyiDetection:
    """Deterministically classify Myanmar-script text as Zawgyi or Unicode.

    Args:
        text: Decoded text. Only Myanmar-script code points are examined.
        threshold: Confidence margin in ``(0.5, 1.0]`` required for a
            ``"zawgyi"`` or ``"unicode"`` decision. Text in between is
            reported as ``"undetermined"``.

    Returns:
        A content-free decision with evidence counts. With no Unicode
        evidence, the default threshold needs at least three Zawgyi signals,
        so very short runs are reported as ``"undetermined"``.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: If ``threshold`` is outside ``(0.5, 1.0]``.
    """

    if not isinstance(text, str):
        raise TypeError("zawgyi detection requires str input")
    if isinstance(threshold, bool) or not 0.5 < float(threshold) <= 1.0:
        raise ValueError("threshold must be greater than 0.5 and at most 1.0")
    threshold = float(threshold)

    extended = len(_MYANMAR_EXTENDED_RE.findall(text))
    myanmar_chars = extended
    zawgyi_signals = 0
    zawgyi_forms = 0
    unicode_signals = extended
    for match in _MYANMAR_RUN_RE.finditer(text):
        codepoints = [ord(char) for char in match.group()]
        myanmar_chars += len(codepoints)
        last = len(codepoints) - 1
        for index, codepoint in enumerate(codepoints):
            previous = codepoints[index - 1] if index else None
            following = codepoints[index + 1] if index < last else None
            if codepoint == 0x1031:
                if previous in _UNICODE_E_HOSTS:
                    if following not in _AMBIGUOUS_E_FOLLOWERS:
                        unicode_signals += 1
                elif following in _ZAWGYI_E_FOLLOWERS:
                    zawgyi_signals += 1
            elif codepoint == 0x103B:
                if previous not in _UNICODE_LETTERS:
                    if following in _CONSONANTS or following in _ZAWGYI_LETTER_FORMS:
                        zawgyi_signals += 1
                elif following in {0x103D, 0x103E}:
                    unicode_signals += 1
            elif codepoint == 0x1039:
                if following not in _UNICODE_LETTERS:
                    zawgyi_signals += 1
            elif codepoint == 0x1004:
                if following == 0x103A and index + 2 <= last:
                    if codepoints[index + 2] == 0x1039:
                        unicode_signals += 1
            elif codepoint == 0x1037 and following == 0x103A:
                unicode_signals += 1
            if codepoint in _UNICODE_ONLY:
                unicode_signals += 1
            if codepoint in _ZAWGYI_PRESENTATION_FORMS:
                zawgyi_forms += 1

    if myanmar_chars == 0:
        return ZawgyiDetection(
            decision="not-myanmar",
            confidence=0.0,
            threshold=threshold,
            myanmar_chars=0,
            zawgyi_signals=0,
            zawgyi_forms=0,
            unicode_signals=0,
        )

    zawgyi_score = zawgyi_signals + _FORM_WEIGHT * zawgyi_forms
    confidence = (zawgyi_score + _SMOOTHING) / (
        zawgyi_score + unicode_signals + 2 * _SMOOTHING
    )
    decision: ZawgyiDecision
    if zawgyi_signals >= _MIN_STRUCTURAL_SIGNALS and confidence >= threshold:
        decision = "zawgyi"
    elif 1.0 - confidence >= threshold:
        decision = "unicode"
    else:
        decision = "undetermined"
    return ZawgyiDetection(
        decision=decision,
        confidence=round(confidence, 4),
        threshold=threshold,
        myanmar_chars=myanmar_chars,
        zawgyi_signals=zawgyi_signals,
        zawgyi_forms=zawgyi_forms,
        unicode_signals=unicode_signals,
    )


def convert_zawgyi_with_origins(
    text: str,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Convert every Myanmar run of ``text`` from Zawgyi to Unicode order.

    Non-Myanmar characters are copied unchanged. Each Myanmar run is split
    into visual syllable clusters; every Zawgyi code point is expanded to its
    Unicode units, which are then sorted into UTN #11 storage order. Each
    converted cluster is emitted in canonical (NFC) order.

    Args:
        text: Decoded text to convert. No detection is performed.

    Returns:
        The converted text and, for each output character, the half-open
        ``[start, end)`` code-point span of ``text`` it derives from. Starts and
        ends never decrease, and a character moved by reordering maps to the
        smallest source range covering every character it moved across, so
        any converted span maps back to a source span that fully contains its
        origin.
    """

    if not isinstance(text, str):
        raise TypeError("zawgyi conversion requires str input")
    chars: list[str] = []
    origins: list[tuple[int, int]] = []
    cursor = 0
    for match in _MYANMAR_RUN_RE.finditer(text):
        start, end = match.span()
        chars.append(text[cursor:start])
        origins.extend((index, index + 1) for index in range(cursor, start))
        run_chars, run_origins = _convert_run(text, start, end)
        chars.extend(run_chars)
        origins.extend(_monotone_spans(run_origins))
        cursor = end
    chars.append(text[cursor:])
    origins.extend((index, index + 1) for index in range(cursor, len(text)))
    return "".join(chars), tuple(origins)


def _convert_run(
    text: str,
    start: int,
    end: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    chars: list[str] = []
    origins: list[tuple[int, int]] = []
    cluster: list[int] = []
    # "empty", "prebase" (only pre-base signs so far), "based" or "orphan".
    state = "empty"
    for index in range(start, end):
        role = _ROLES[ord(text[index])]
        if role != _ROLE_MARK and state not in {"empty", "prebase"}:
            _emit_cluster(text, cluster, state == "based", chars, origins)
            cluster = []
            state = "empty"
        cluster.append(index)
        if role == _ROLE_PREBASE:
            state = "prebase"
        elif role == _ROLE_BASE:
            state = "based"
        elif state != "based":
            state = "orphan"
    if cluster:
        _emit_cluster(text, cluster, state == "based", chars, origins)
    return chars, origins


def _emit_cluster(
    text: str,
    indices: list[int],
    has_base: bool,
    chars: list[str],
    origins: list[tuple[int, int]],
) -> None:
    units: list[tuple[int, int, str, int]] = []
    sticky_slot = _BASE
    last = len(indices) - 1
    for position, index in enumerate(indices):
        codepoint = ord(text[index])
        following = ord(text[indices[position + 1]]) if position < last else None
        expanded = _expand(codepoint, following, sticky_slot)
        for char, slot in expanded:
            units.append((slot, len(units), char, index))
        sticky_slot = expanded[-1][1]
    if has_base:
        units.sort(key=lambda unit: (unit[0], unit[1]))

    merged: list[tuple[str, int, int]] = []
    for _slot, _sequence, char, index in units:
        if char == "ီ" and merged and merged[-1][0] == "ဥ":
            # U+1025 U+102E canonically composes to U+1026 under NFC.
            _previous, source_start, source_end = merged[-1]
            merged[-1] = (
                "ဦ",
                min(source_start, index),
                max(source_end, index + 1),
            )
            continue
        merged.append((char, index, index + 1))
    _canonical_reorder(merged)
    for char, source_start, source_end in merged:
        chars.append(char)
        origins.append((source_start, source_end))


def _expand(codepoint: int, following: int | None, sticky_slot: int) -> _Units:
    prebase = _ZAWGYI_PREBASE.get(codepoint)
    if prebase is not None:
        return prebase
    base = _ZAWGYI_BASES.get(codepoint)
    if base is not None:
        return base
    if codepoint == 0x1025 and following in {0x1039, 0x102C}:
        # Zawgyi reuses LETTER U for NYA before a killer or vowel sign AA.
        return (("ဉ", _BASE),)
    if codepoint == 0x1039 and following in _LOWER_VOWELS:
        # Contractions such as NA + asat + U keep asat before the vowel.
        return (("်", _ASAT_CONTRACTION),)
    mark = _ZAWGYI_MARKS.get(codepoint)
    if mark is not None:
        return mark
    char = chr(codepoint)
    if _ROLES[codepoint] == _ROLE_BASE:
        return ((char, _BASE),)
    # Unmapped marks stay next to the unit they followed.
    return ((char, sticky_slot),)


def _canonical_reorder(units: list[tuple[str, int, int]]) -> None:
    """Stable-sort adjacent non-starters by combining class, as NFC does."""

    for index in range(1, len(units)):
        position = index
        current_class = unicodedata.combining(units[position][0])
        while position > 0 and current_class:
            previous_class = unicodedata.combining(units[position - 1][0])
            if previous_class <= current_class:
                break
            units[position - 1], units[position] = units[position], units[position - 1]
            position -= 1


def _monotone_spans(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Widen reordered spans so starts and ends never decrease."""

    starts = [start for start, _end in spans]
    ends = [end for _start, end in spans]
    for index in range(len(starts) - 2, -1, -1):
        starts[index] = min(starts[index], starts[index + 1])
    for index in range(1, len(ends)):
        ends[index] = max(ends[index], ends[index - 1])
    return list(zip(starts, ends))


__all__ = [
    "ZAWGYI_CONFIDENCE_THRESHOLD",
    "ZAWGYI_RULE_PROVENANCE",
    "ZawgyiDecision",
    "ZawgyiDetection",
    "convert_zawgyi_with_origins",
    "detect_zawgyi",
]
