"""Deterministic, privacy-conscious matching for Indic personal names.

The normalizer converts an Indic or Latin name surface into a conservative
Latin phonetic key.  Callers must treat that key as sensitive transient data;
the surrogate vault HMACs it before storage.  No model weights are bundled.
Applications may provide a local transliterator, while the built-in fallback
uses only :mod:`unicodedata` and supports the major Brahmic scripts.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

DEFAULT_INDIC_NAME_SIMILARITY_THRESHOLD = 0.80
INDIC_NAME_KEY_VERSION = "indic-name-v1"

INDIC_LANGUAGE_CODES = frozenset(
    {
        "as",
        "bn",
        "gu",
        "hi",
        "kn",
        "kok",
        "mai",
        "ml",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "ta",
        "te",
    }
)

_SCRIPT_PREFIXES = {
    "devanagari": "DEVANAGARI",
    "bengali": "BENGALI",
    "gurmukhi": "GURMUKHI",
    "gujarati": "GUJARATI",
    "odia": "ORIYA",
    "tamil": "TAMIL",
    "telugu": "TELUGU",
    "kannada": "KANNADA",
    "malayalam": "MALAYALAM",
}

_CONSONANTS = {
    "KA": "k",
    "KHA": "kh",
    "GA": "g",
    "GHA": "gh",
    "NGA": "ng",
    "CA": "ch",
    "CHA": "chh",
    "JA": "j",
    "JHA": "jh",
    "NYA": "ny",
    "TTA": "t",
    "TTHA": "th",
    "DDA": "d",
    "DDHA": "dh",
    "NNA": "n",
    "TA": "t",
    "THA": "th",
    "DA": "d",
    "DHA": "dh",
    "NA": "n",
    "PA": "p",
    "PHA": "ph",
    "BA": "b",
    "BHA": "bh",
    "MA": "m",
    "YA": "y",
    "RA": "r",
    "RRA": "r",
    "LA": "l",
    "LLA": "l",
    "VA": "v",
    "SHA": "sh",
    "SSA": "sh",
    "SA": "s",
    "HA": "h",
    "QA": "q",
    "KHHA": "kh",
    "GHHA": "gh",
    "ZA": "z",
    "FA": "f",
    "YYA": "y",
}

_VOWELS = {
    "A": "a",
    "AA": "aa",
    "I": "i",
    "II": "ii",
    "U": "u",
    "UU": "uu",
    "VOCALIC R": "ri",
    "VOCALIC RR": "ri",
    "VOCALIC L": "li",
    "E": "e",
    "AI": "ai",
    "O": "o",
    "AU": "au",
    "SHORT E": "e",
    "SHORT O": "o",
}

_LATIN_DIACRITIC_FOLDS = str.maketrans(
    {
        "ṛ": "ri",
        "ṝ": "ri",
        "ṟ": "r",
        "ḷ": "l",
        "ḹ": "l",
        "ṣ": "sh",
        "ś": "sh",
        "ṅ": "n",
        "ñ": "n",
        "ṇ": "n",
        "ṃ": "n",
        "ṁ": "n",
        "ḥ": "h",
    }
)

_ROMAN_CONSONANT_SUFFIX = {
    "k": "KA",
    "kh": "KHA",
    "g": "GA",
    "gh": "GHA",
    "ng": "NGA",
    "ch": "CA",
    "chh": "CHA",
    "c": "KA",
    "j": "JA",
    "jh": "JHA",
    "ny": "NYA",
    "t": "TA",
    "th": "THA",
    "d": "DA",
    "dh": "DHA",
    "n": "NA",
    "p": "PA",
    "ph": "PHA",
    "b": "BA",
    "bh": "BHA",
    "m": "MA",
    "y": "YA",
    "r": "RA",
    "l": "LA",
    "v": "VA",
    "w": "VA",
    "sh": "SHA",
    "s": "SA",
    "h": "HA",
    "q": "KA",
    "f": "PHA",
    "x": "KA",
    "z": "JA",
}
_ROMAN_VOWEL_SUFFIX = {
    "a": "A",
    "aa": "AA",
    "i": "I",
    "ii": "II",
    "u": "U",
    "uu": "UU",
    "e": "E",
    "ai": "AI",
    "o": "O",
    "au": "AU",
}
_ROMAN_TOKENS = tuple(
    sorted(
        {*_ROMAN_CONSONANT_SUFFIX, *_ROMAN_VOWEL_SUFFIX, "ksh"},
        key=len,
        reverse=True,
    )
)

Transliterator = Callable[[str], Any] | Any


@dataclass(frozen=True)
class IndicNameNormalizer:
    """Build conservative canonical keys and render Indic surrogate surfaces.

    Args:
        similarity_threshold: Minimum edit similarity required before a
            romanization fold (for example ``x`` to ``ksh``) is accepted.
            Higher values reduce matching and therefore reduce collision risk.
        transliterator: Optional local callable or object. A callable should
            return a Latin string. Objects may provide ``to_latin`` and
            ``from_latin`` methods, or a ``transliterate`` method accepting
            ``source_script`` and ``target_script`` keyword arguments.
    """

    similarity_threshold: float = DEFAULT_INDIC_NAME_SIMILARITY_THRESHOLD
    transliterator: Transliterator | None = None

    def __post_init__(self) -> None:
        _validate_threshold(self.similarity_threshold)

    def canonical_key(self, surface: str) -> str:
        """Return a versioned in-memory canonical match key for ``surface``."""

        latin = self.to_latin(surface)
        tokens = _latin_tokens(latin)
        folded = [
            _collision_safe_fold(token, threshold=self.similarity_threshold)
            for token in tokens
        ]
        return f"{INDIC_NAME_KEY_VERSION}:{' '.join(folded)}"

    def to_latin(self, surface: str) -> str:
        """Return deterministic Latin text using a local model or stdlib."""

        source = str(surface or "").strip()
        if not source:
            raise ValueError("name surface must be non-empty")
        script = detect_name_script(source)
        if script in _SCRIPT_PREFIXES and self.transliterator is not None:
            translated = _model_to_latin(self.transliterator, source, script)
            if translated:
                return translated
        return _stdlib_to_latin(source)

    def matches(self, left: str, right: str) -> bool:
        """Return whether two surfaces produce the same collision-safe key."""

        return self.canonical_key(left) == self.canonical_key(right)

    def render_surrogate(self, identity: str, *, source_surface: str) -> str:
        """Render a stored Latin surrogate in the source surface's script."""

        script = detect_name_script(source_surface)
        if script not in _SCRIPT_PREFIXES:
            return identity
        if self.transliterator is not None:
            translated = _model_from_latin(self.transliterator, identity, script)
            if translated and detect_name_script(translated) == script:
                return translated
        return _stdlib_from_latin(identity, script)


def canonical_indic_name_key(
    surface: str,
    *,
    similarity_threshold: float = DEFAULT_INDIC_NAME_SIMILARITY_THRESHOLD,
    transliterator: Transliterator | None = None,
) -> str:
    """Return a deterministic canonical key for an Indic name surface."""

    return IndicNameNormalizer(
        similarity_threshold=similarity_threshold,
        transliterator=transliterator,
    ).canonical_key(surface)


def indic_names_match(
    left: str,
    right: str,
    *,
    similarity_threshold: float = DEFAULT_INDIC_NAME_SIMILARITY_THRESHOLD,
    transliterator: Transliterator | None = None,
) -> bool:
    """Return whether two name surfaces share a conservative canonical key."""

    return IndicNameNormalizer(
        similarity_threshold=similarity_threshold,
        transliterator=transliterator,
    ).matches(left, right)


def detect_name_script(surface: str) -> str:
    """Return the Indic script name, ``latin``, or ``other`` for ``surface``."""

    indic_counts = {script: 0 for script in _SCRIPT_PREFIXES}
    latin_count = 0
    for char in str(surface or ""):
        name = unicodedata.name(char, "")
        for script, prefix in _SCRIPT_PREFIXES.items():
            if name.startswith(f"{prefix} "):
                indic_counts[script] += 1
                break
        else:
            if "LATIN" in name and unicodedata.category(char).startswith("L"):
                latin_count += 1
    script, count = max(indic_counts.items(), key=lambda item: item[1])
    if count:
        return script
    return "latin" if latin_count else "other"


def is_indic_name_candidate(surface: str, *, lang: str = "") -> bool:
    """Return whether transliteration-aware matching should apply."""

    language = str(lang or "").split("-", 1)[0].split("_", 1)[0].lower()
    return detect_name_script(surface) in _SCRIPT_PREFIXES or language in (
        INDIC_LANGUAGE_CODES
    )


def _stdlib_to_latin(surface: str) -> str:
    result: list[str] = []
    for char in unicodedata.normalize("NFC", surface):
        name = unicodedata.name(char, "")
        script_prefix = next(
            (
                prefix
                for prefix in _SCRIPT_PREFIXES.values()
                if name.startswith(f"{prefix} ")
            ),
            None,
        )
        if script_prefix is None:
            result.append(char)
            continue

        tail = name[len(script_prefix) + 1 :]
        if tail.startswith("LETTER "):
            suffix = tail.removeprefix("LETTER ")
            consonant = _CONSONANTS.get(suffix)
            if consonant is not None:
                result.append(f"{consonant}a")
                continue
            vowel = _VOWELS.get(suffix)
            if vowel is not None:
                result.append(vowel)
                continue
        if tail.startswith("VOWEL SIGN "):
            vowel = _VOWELS.get(tail.removeprefix("VOWEL SIGN "))
            if vowel is not None:
                _replace_inherent_vowel(result, vowel)
                continue
        if tail in {"SIGN VIRAMA", "SIGN HALANT"}:
            _replace_inherent_vowel(result, "")
        elif tail in {"SIGN ANUSVARA", "SIGN CANDRABINDU"}:
            result.append("n")
        elif tail == "SIGN VISARGA":
            result.append("h")
        elif tail in {"SIGN NUKTA", "SIGN AVAGRAHA"} or "JOINER" in tail:
            continue
    return "".join(result)


def _replace_inherent_vowel(result: list[str], vowel: str) -> None:
    if result and result[-1].endswith("a"):
        result[-1] = f"{result[-1][:-1]}{vowel}"


def _latin_tokens(value: str) -> list[str]:
    folded = value.casefold().translate(_LATIN_DIACRITIC_FOLDS)
    decomposed = unicodedata.normalize("NFKD", folded)
    ascii_value = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    tokens = re.findall(r"[a-z]+", ascii_value)
    if not tokens:
        raise ValueError("name surface must contain Latin or supported Indic letters")
    return tokens


def _collision_safe_fold(value: str, *, threshold: float) -> str:
    # ``x`` is the conventional Latin shorthand for the Indic ``ksh``
    # conjunct, so account for that zero-cost phonetic expansion before the
    # edit-similarity gate is applied to the remaining optional folds.
    phonetic_base = value.replace("x", "ksh").replace("q", "k")
    candidate = phonetic_base
    candidate = re.sub(r"ee", "i", candidate)
    candidate = re.sub(r"oo", "u", candidate)
    candidate = re.sub(r"ii+", "i", candidate)
    candidate = re.sub(r"uu+", "u", candidate)
    candidate = re.sub(r"aya$", "ay", candidate)
    candidate = re.sub(r"ai$", "ay", candidate)
    candidate = re.sub(r"aa$", "a", candidate)
    similarity = SequenceMatcher(
        None,
        phonetic_base,
        candidate,
        autojunk=False,
    ).ratio()
    return candidate if similarity >= threshold else phonetic_base


def _stdlib_from_latin(identity: str, script: str) -> str:
    prefix = _SCRIPT_PREFIXES[script]
    rendered = [
        _render_roman_word(word, prefix) if word else word
        for word in re.split(r"([A-Za-z]+)", identity)
    ]
    value = "".join(rendered)
    return value if value.strip() else identity


def _render_roman_word(word: str, prefix: str) -> str:
    tokens: list[str] = []
    lower = word.casefold()
    index = 0
    while index < len(lower):
        token = next(
            (item for item in _ROMAN_TOKENS if lower.startswith(item, index)),
            lower[index],
        )
        tokens.append(token)
        index += len(token)

    output: list[str] = []
    pending_consonant = False
    for token in tokens:
        consonant_suffix = (
            "KA" if token in {"ksh", "x"} else _ROMAN_CONSONANT_SUFFIX.get(token)
        )
        if consonant_suffix is not None:
            if pending_consonant:
                virama = _lookup(prefix, "SIGN VIRAMA")
                if virama:
                    output.append(virama)
            letter = _lookup_letter(prefix, consonant_suffix)
            if letter:
                output.append(letter)
                if token in {"ksh", "x"}:
                    virama = _lookup(prefix, "SIGN VIRAMA")
                    sha = _lookup_letter(prefix, "SHA")
                    if virama and sha:
                        output.extend((virama, sha))
                pending_consonant = True
            continue

        vowel_suffix = _ROMAN_VOWEL_SUFFIX.get(token)
        if vowel_suffix is None:
            output.append(token)
            pending_consonant = False
            continue
        if pending_consonant:
            if vowel_suffix != "A":
                sign = _lookup(prefix, f"VOWEL SIGN {vowel_suffix}")
                if sign:
                    output.append(sign)
            pending_consonant = False
        else:
            vowel = _lookup(prefix, f"LETTER {vowel_suffix}")
            if vowel:
                output.append(vowel)
    return "".join(output) or word


def _lookup_letter(prefix: str, suffix: str) -> str:
    alternatives = {
        "KHA": ("KHA", "KA"),
        "GHA": ("GHA", "GA"),
        "CHA": ("CHA", "CA"),
        "JHA": ("JHA", "JA"),
        "THA": ("THA", "TA"),
        "DHA": ("DHA", "DA"),
        "PHA": ("PHA", "PA"),
        "BHA": ("BHA", "BA"),
        "SHA": ("SHA", "SA"),
    }.get(suffix, (suffix,))
    for candidate in alternatives:
        value = _lookup(prefix, f"LETTER {candidate}")
        if value:
            return value
    return ""


def _lookup(prefix: str, suffix: str) -> str:
    try:
        return unicodedata.lookup(f"{prefix} {suffix}")
    except KeyError:
        return ""


def _model_to_latin(model: Transliterator, text: str, script: str) -> str:
    to_latin = getattr(model, "to_latin", None)
    if callable(to_latin):
        return _coerce_model_result(to_latin(text))
    method = getattr(model, "transliterate", None)
    if callable(method):
        return _try_model_calls(
            (
                lambda: method(
                    text,
                    source_script=script,
                    target_script="latin",
                ),
                lambda: method(text, target_script="latin"),
                lambda: method(text),
            )
        )
    if callable(model):
        return _coerce_model_result(model(text))
    return ""


def _model_from_latin(model: Transliterator, text: str, script: str) -> str:
    from_latin = getattr(model, "from_latin", None)
    if callable(from_latin):
        return _coerce_model_result(from_latin(text, script))
    method = getattr(model, "transliterate", None)
    if callable(method):
        return _try_model_calls(
            (
                lambda: method(
                    text,
                    source_script="latin",
                    target_script=script,
                ),
                lambda: method(text, target_script=script),
            )
        )
    return ""


def _try_model_calls(calls: Sequence[Callable[[], Any]]) -> str:
    for call in calls:
        try:
            value = _coerce_model_result(call())
        except (KeyError, TypeError, ValueError):
            continue
        if value:
            return value
    return ""


def _coerce_model_result(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("latin", "en", "text", "output"):
            if key in value:
                result = _coerce_model_result(value[key])
                if result:
                    return result
        for item in value.values():
            result = _coerce_model_result(item)
            if result:
                return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            result = _coerce_model_result(item)
            if result:
                return result
    return ""


def _validate_threshold(value: float) -> None:
    if not 0.5 <= float(value) <= 1.0:
        raise ValueError("similarity_threshold must be between 0.5 and 1.0")


__all__ = [
    "DEFAULT_INDIC_NAME_SIMILARITY_THRESHOLD",
    "INDIC_LANGUAGE_CODES",
    "INDIC_NAME_KEY_VERSION",
    "IndicNameNormalizer",
    "canonical_indic_name_key",
    "detect_name_script",
    "indic_names_match",
    "is_indic_name_candidate",
]
