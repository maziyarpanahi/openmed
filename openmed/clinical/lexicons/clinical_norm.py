"""Locale-aware clinical normalization lexicons.

The tables in this module are compact synthetic/permissive language packs for
deterministic unit, abbreviation, and number normalization. They deliberately
avoid restricted terminology assets and clinical corpus text.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field

AbbreviationExpansion = str

_ASCII_APOSTROPHE = "'"
_RIGHT_SINGLE_QUOTE = "\u2019"
_NO_BREAK_SPACE = "\u00a0"
_NARROW_NO_BREAK_SPACE = "\u202f"
_ARABIC_DECIMAL_SEPARATOR = "\u066b"
_ARABIC_THOUSANDS_SEPARATOR = "\u066c"


@dataclass(frozen=True)
class ClinicalNormLexicon:
    """Language pack for deterministic clinical value normalization."""

    language: str
    decimal_separator: str = "."
    thousands_separators: tuple[str, ...] = ()
    unit_aliases: Mapping[str, str] = field(default_factory=dict)
    abbreviation_expansions: Mapping[str, AbbreviationExpansion] = field(
        default_factory=dict
    )
    frequency_aliases: Mapping[str, str] = field(default_factory=dict)
    duration_unit_aliases: Mapping[str, str] = field(default_factory=dict)
    abnormal_flags: Mapping[str, str] = field(default_factory=dict)
    token_boundaries: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", normalize_language(self.language))
        object.__setattr__(
            self,
            "thousands_separators",
            tuple(dict.fromkeys(self.thousands_separators)),
        )
        object.__setattr__(
            self,
            "unit_aliases",
            {
                normalize_unit_surface(key): value
                for key, value in self.unit_aliases.items()
                if normalize_unit_surface(key)
            },
        )
        object.__setattr__(
            self,
            "abbreviation_expansions",
            {
                normalize_surface(key): value
                for key, value in self.abbreviation_expansions.items()
                if normalize_surface(key)
            },
        )
        object.__setattr__(
            self,
            "frequency_aliases",
            {
                normalize_surface(key): value
                for key, value in self.frequency_aliases.items()
                if normalize_surface(key)
            },
        )
        object.__setattr__(
            self,
            "duration_unit_aliases",
            {
                normalize_surface(key): value
                for key, value in self.duration_unit_aliases.items()
                if normalize_surface(key)
            },
        )
        object.__setattr__(
            self,
            "abnormal_flags",
            {
                normalize_surface(key): value
                for key, value in self.abnormal_flags.items()
                if normalize_surface(key)
            },
        )


def normalize_language(language: object | None = None) -> str:
    """Normalize a BCP-47-ish language code to its primary subtag."""

    if not isinstance(language, str):
        return "en"
    code = language.casefold().replace("_", "-").strip()
    return code.split("-", 1)[0] or "en"


def normalize_surface(text: object) -> str:
    """Return a matching key for language-pack surfaces."""

    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = normalized.replace(_NO_BREAK_SPACE, " ")
    normalized = normalized.replace(_NARROW_NO_BREAK_SPACE, " ")
    normalized = normalized.replace("\N{GREEK SMALL LETTER MU}", "u")
    normalized = normalized.replace("\N{MICRO SIGN}", "u")
    normalized = re.sub(r"[.,;:()\[\]{}_-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_unit_surface(text: object) -> str:
    """Return a matching key for unit surface variants."""

    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = normalized.replace(_NO_BREAK_SPACE, " ")
    normalized = normalized.replace(_NARROW_NO_BREAK_SPACE, " ")
    normalized = normalized.replace("\N{GREEK SMALL LETTER MU}", "u")
    normalized = normalized.replace("\N{MICRO SIGN}", "u")
    normalized = normalized.replace("\N{DEGREE SIGN}", "")
    normalized = re.sub(r"\s+per\s+", "/", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s*\*\s*", "*", normalized)
    normalized = re.sub(r"\s*\^\s*", "^", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def register_clinical_norm_lexicon(
    lexicon: ClinicalNormLexicon,
) -> ClinicalNormLexicon:
    """Register or replace a deterministic normalization language pack."""

    normalized = ClinicalNormLexicon(
        language=lexicon.language,
        decimal_separator=lexicon.decimal_separator,
        thousands_separators=tuple(lexicon.thousands_separators),
        unit_aliases=dict(lexicon.unit_aliases),
        abbreviation_expansions=dict(lexicon.abbreviation_expansions),
        frequency_aliases=dict(lexicon.frequency_aliases),
        duration_unit_aliases=dict(lexicon.duration_unit_aliases),
        abnormal_flags=dict(lexicon.abnormal_flags),
        token_boundaries=lexicon.token_boundaries,
    )
    _LEXICONS[normalized.language] = normalized
    return normalized


def get_clinical_norm_lexicon(
    language: object | None = None,
) -> ClinicalNormLexicon:
    """Return a normalization language pack, falling back to English."""

    code = normalize_language(language)
    return _LEXICONS.get(code, _LEXICONS["en"])


def available_clinical_norm_languages() -> tuple[str, ...]:
    """Return registered clinical normalization language codes."""

    return tuple(sorted(_LEXICONS))


def clinical_norm_lexicon_stats(
    language: object | None = None,
) -> Mapping[str, Mapping[str, int]]:
    """Return table sizes by language for deterministic normalization packs."""

    languages = (
        (normalize_language(language),)
        if language
        else available_clinical_norm_languages()
    )
    stats: dict[str, dict[str, int]] = {}
    for code in languages:
        lexicon = get_clinical_norm_lexicon(code)
        stats[lexicon.language] = {
            "unit_aliases": len(lexicon.unit_aliases),
            "abbreviation_expansions": len(lexicon.abbreviation_expansions),
            "frequency_aliases": len(lexicon.frequency_aliases),
            "duration_unit_aliases": len(lexicon.duration_unit_aliases),
            "abnormal_flags": len(lexicon.abnormal_flags),
        }
    return stats


def canonical_unit_alias(
    unit: object,
    *,
    language: object | None = None,
) -> str | None:
    """Return a canonical UCUM surface for a localized unit alias."""

    key = normalize_unit_surface(unit)
    if not key:
        return None
    lexicon = get_clinical_norm_lexicon(language)
    alias = lexicon.unit_aliases.get(key)
    if alias is not None:
        return alias
    if lexicon.language != "en":
        return _LEXICONS["en"].unit_aliases.get(key)
    return None


def abbreviation_expansion(
    text: object,
    *,
    language: object | None = None,
) -> AbbreviationExpansion | None:
    """Return a normalized clinical abbreviation expansion."""

    key = normalize_surface(text)
    if not key:
        return None
    lexicon = get_clinical_norm_lexicon(language)
    expansion = lexicon.abbreviation_expansions.get(key)
    if expansion is not None:
        return expansion
    if lexicon.language != "en":
        return _LEXICONS["en"].abbreviation_expansions.get(key)
    return None


def abbreviation_surfaces(
    expansion: AbbreviationExpansion,
    *,
    language: object | None = None,
) -> tuple[str, ...]:
    """Return normalized surfaces that expand to ``expansion``."""

    lexicon = get_clinical_norm_lexicon(language)
    surfaces = [
        surface
        for surface, value in lexicon.abbreviation_expansions.items()
        if value == expansion
    ]
    if lexicon.language != "en":
        surfaces.extend(
            surface
            for surface, value in _LEXICONS["en"].abbreviation_expansions.items()
            if value == expansion
        )
    return tuple(dict.fromkeys(surfaces))


def abnormal_flag_alias(
    text: object,
    *,
    language: object | None = None,
) -> str | None:
    """Return a canonical abnormal flag for a localized flag string."""

    key = normalize_surface(text)
    if not key:
        return None
    lexicon = get_clinical_norm_lexicon(language)
    flag = lexicon.abnormal_flags.get(key)
    if flag is not None:
        return flag
    if lexicon.language != "en":
        return _LEXICONS["en"].abnormal_flags.get(key)
    return None


def localized_frequency_text(
    text: object,
    *,
    language: object | None = None,
) -> object:
    """Expand a localized frequency cue into the English sig cue space."""

    if not isinstance(text, str):
        return text
    key = normalize_surface(text)
    if not key:
        return text
    lexicon = get_clinical_norm_lexicon(language)
    alias = lexicon.frequency_aliases.get(key)
    if alias is not None:
        return alias
    if lexicon.language != "en":
        return _LEXICONS["en"].frequency_aliases.get(key, text)
    return text


def localized_duration_text(
    text: object,
    *,
    language: object | None = None,
) -> object:
    """Translate localized duration unit words into the English sig space."""

    if not isinstance(text, str):
        return text
    lexicon = get_clinical_norm_lexicon(language)
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    if not normalized:
        return text
    aliases = dict(_LEXICONS["en"].duration_unit_aliases)
    if lexicon.language != "en":
        aliases.update(lexicon.duration_unit_aliases)
    for surface, canonical in sorted(
        aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = _surface_pattern(surface, token_boundaries=lexicon.token_boundaries)
        normalized = pattern.sub(canonical, normalized)
    return normalized


_LOCALIZED_NUMBER = (
    r"[+-]?(?:"
    r"(?:\d{1,3}(?:[ \u00a0\u202f'’.,]\d{3})+|\d+)(?:[.,\u066b]\d+)?"
    r"|[.,\u066b]\d+)"
)
_MEASUREMENT_RE = re.compile(
    rf"^\s*(?P<value>{_LOCALIZED_NUMBER})\s*(?P<unit>.+?)\s*$",
    re.UNICODE,
)


def split_measurement_text(text: object) -> tuple[str, str] | None:
    """Split a localized measurement string into value and unit text."""

    if not isinstance(text, str):
        return None
    match = _MEASUREMENT_RE.fullmatch(unicodedata.normalize("NFKC", text))
    if match is None:
        return None
    return match.group("value").strip(), match.group("unit").strip()


def parse_locale_number(
    value: object,
    *,
    language: object | None = None,
) -> float | None:
    """Parse a finite number using locale punctuation and Unicode digits."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None

    text = _ascii_number_text(value)
    if not text:
        return None

    decimal_separator = get_clinical_norm_lexicon(language).decimal_separator
    normalized = _normalize_decimal_punctuation(text, decimal_separator)
    try:
        number = float(normalized)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _ascii_number_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    result: list[str] = []
    for char in normalized:
        if char in {_NO_BREAK_SPACE, _NARROW_NO_BREAK_SPACE}:
            result.append(" ")
            continue
        if char == _ARABIC_DECIMAL_SEPARATOR:
            result.append(",")
            continue
        if char == _ARABIC_THOUSANDS_SEPARATOR:
            result.append(",")
            continue
        try:
            digit = unicodedata.decimal(char)
        except (TypeError, ValueError):
            result.append(char)
        else:
            result.append(str(digit))
    return "".join(result).strip()


def _normalize_decimal_punctuation(text: str, decimal_separator: str) -> str:
    compact = text.replace(" ", "")
    compact = compact.replace(_NO_BREAK_SPACE, "")
    compact = compact.replace(_NARROW_NO_BREAK_SPACE, "")
    compact = compact.replace(_ASCII_APOSTROPHE, "")
    compact = compact.replace(_RIGHT_SINGLE_QUOTE, "")

    has_dot = "." in compact
    has_comma = "," in compact
    decimal: str | None = None
    if has_dot and has_comma:
        decimal = "." if compact.rfind(".") > compact.rfind(",") else ","
    elif has_comma:
        decimal = "," if decimal_separator == "," else None
        if decimal is None and not _looks_grouped(compact, ","):
            decimal = ","
    elif has_dot:
        decimal = "." if decimal_separator == "." else None
        if decimal is None and not _looks_grouped(compact, "."):
            decimal = "."

    if decimal is None:
        return compact.replace(",", "").replace(".", "")

    integer, fractional = compact.rsplit(decimal, 1)
    integer = integer.replace(",", "").replace(".", "")
    fractional = fractional.replace(",", "").replace(".", "")
    if not integer and text.strip().startswith(("+", "-")):
        integer = text.strip()[0]
    return f"{integer}.{fractional}"


def _looks_grouped(text: str, separator: str) -> bool:
    signless = text.lstrip("+-")
    groups = signless.split(separator)
    if len(groups) <= 1:
        return False
    return 1 <= len(groups[0]) <= 3 and all(
        len(group) == 3 and group.isdigit() for group in groups[1:]
    )


def _surface_pattern(surface: str, *, token_boundaries: bool) -> re.Pattern[str]:
    escaped = re.escape(surface)
    if not token_boundaries:
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


_COMMON_UNIT_ALIASES = {
    "percent": "%",
    "mg per dl": "mg/dL",
    "mg/dl": "mg/dL",
    "milligram/deciliter": "mg/dL",
    "milligrams/deciliter": "mg/dL",
    "g/dl": "g/dL",
    "g/l": "g/L",
    "mg/l": "mg/L",
    "ug/ml": "ug/mL",
    "mcg/ml": "ug/mL",
    "ng/ml": "ng/mL",
    "mmol/l": "mmol/L",
    "umol/l": "umol/L",
    "meq/l": "mEq/L",
    "u/l": "U/L",
    "iu/l": "U/L",
    "mm hg": "mmHg",
    "mmhg": "mmHg",
    "c": "Cel",
    "degc": "Cel",
    "celsius": "Cel",
    "f": "[degF]",
    "degf": "[degF]",
    "fahrenheit": "[degF]",
    "/min": "1/min",
    "per minute": "1/min",
    "bpm": "beat/min",
    "beats/min": "beat/min",
    "breaths/min": "breath/min",
    "cells/ul": "cell/uL",
}

ENGLISH_NORM_LEXICON = ClinicalNormLexicon(
    language="en",
    decimal_separator=".",
    thousands_separators=(",", _NO_BREAK_SPACE, _NARROW_NO_BREAK_SPACE),
    unit_aliases=_COMMON_UNIT_ALIASES,
    abbreviation_expansions={
        "bp": "blood_pressure",
        "b/p": "blood_pressure",
        "blood pressure": "blood_pressure",
        "hr": "heart_rate",
        "heart rate": "heart_rate",
        "pulse": "heart_rate",
        "rr": "respiratory_rate",
        "resp": "respiratory_rate",
        "respiratory rate": "respiratory_rate",
        "temp": "body_temperature",
        "temperature": "body_temperature",
        "t": "body_temperature",
        "spo2": "oxygen_saturation",
        "sp o2": "oxygen_saturation",
        "o2 sat": "oxygen_saturation",
        "oxygen saturation": "oxygen_saturation",
    },
    frequency_aliases={
        "as needed": "prn",
        "when needed": "prn",
    },
    duration_unit_aliases={
        "d": "days",
        "day": "days",
        "days": "days",
        "wk": "weeks",
        "week": "weeks",
        "weeks": "weeks",
    },
    abnormal_flags={
        "h": "high",
        "high": "high",
        "l": "low",
        "low": "low",
        "c": "critical",
        "crit": "critical",
        "critical": "critical",
        "n": "normal",
        "normal": "normal",
    },
)

SPANISH_NORM_LEXICON = ClinicalNormLexicon(
    language="es",
    decimal_separator=",",
    thousands_separators=(".", " ", _NO_BREAK_SPACE, _NARROW_NO_BREAK_SPACE),
    unit_aliases={
        **_COMMON_UNIT_ALIASES,
        "mg por dl": "mg/dL",
        "mg por decilitro": "mg/dL",
        "g por l": "g/L",
        "g por litro": "g/L",
        "mm de hg": "mmHg",
        "latidos/min": "beat/min",
        "latidos por minuto": "beat/min",
        "resp/min": "breath/min",
        "respiraciones/min": "breath/min",
    },
    abbreviation_expansions={
        "ta": "blood_pressure",
        "pa": "blood_pressure",
        "presion arterial": "blood_pressure",
        "presión arterial": "blood_pressure",
        "fc": "heart_rate",
        "frecuencia cardiaca": "heart_rate",
        "frecuencia cardíaca": "heart_rate",
        "fr": "respiratory_rate",
        "frecuencia respiratoria": "respiratory_rate",
        "t": "body_temperature",
        "temp": "body_temperature",
        "temperatura": "body_temperature",
        "sato2": "oxygen_saturation",
        "spo2": "oxygen_saturation",
    },
    frequency_aliases={
        "cada dia": "daily",
        "cada día": "daily",
        "una vez al dia": "once daily",
        "una vez al día": "once daily",
        "dos veces al dia": "twice daily",
        "dos veces al día": "twice daily",
        "cada 8 horas": "every 8 hours",
        "segun necesidad": "prn",
        "según necesidad": "prn",
    },
    duration_unit_aliases={
        "dia": "days",
        "día": "days",
        "dias": "days",
        "días": "days",
    },
    abnormal_flags={"alto": "high", "alta": "high", "bajo": "low", "baja": "low"},
)

FRENCH_NORM_LEXICON = ClinicalNormLexicon(
    language="fr",
    decimal_separator=",",
    thousands_separators=(" ", _NO_BREAK_SPACE, _NARROW_NO_BREAK_SPACE, "."),
    unit_aliases={
        **_COMMON_UNIT_ALIASES,
        "mg par dl": "mg/dL",
        "mg par decilitre": "mg/dL",
        "mg par décilitre": "mg/dL",
        "g par l": "g/L",
        "g par litre": "g/L",
        "mm de hg": "mmHg",
        "battements/min": "beat/min",
        "battements par minute": "beat/min",
        "respirations/min": "breath/min",
    },
    abbreviation_expansions={
        "ta": "blood_pressure",
        "tension arterielle": "blood_pressure",
        "tension artérielle": "blood_pressure",
        "fc": "heart_rate",
        "frequence cardiaque": "heart_rate",
        "fréquence cardiaque": "heart_rate",
        "fr": "respiratory_rate",
        "frequence respiratoire": "respiratory_rate",
        "fréquence respiratoire": "respiratory_rate",
        "t": "body_temperature",
        "temp": "body_temperature",
        "temperature": "body_temperature",
        "température": "body_temperature",
        "spo2": "oxygen_saturation",
        "saturation": "oxygen_saturation",
    },
    frequency_aliases={
        "tous les jours": "daily",
        "une fois par jour": "once daily",
        "deux fois par jour": "twice daily",
        "toutes les 8 heures": "every 8 hours",
        "si besoin": "prn",
    },
    duration_unit_aliases={"jour": "days", "jours": "days", "semaine": "weeks"},
    abnormal_flags={"haut": "high", "haute": "high", "bas": "low", "basse": "low"},
)

GERMAN_NORM_LEXICON = ClinicalNormLexicon(
    language="de",
    decimal_separator=",",
    thousands_separators=(".", " ", _NO_BREAK_SPACE, _NARROW_NO_BREAK_SPACE),
    unit_aliases={
        **_COMMON_UNIT_ALIASES,
        "mg pro dl": "mg/dL",
        "mg je dl": "mg/dL",
        "g pro l": "g/L",
        "g je l": "g/L",
        "mm quecksilber": "mmHg",
        "schläge/min": "beat/min",
        "schlaege/min": "beat/min",
        "schläge pro minute": "beat/min",
        "atemzüge/min": "breath/min",
        "atemzuege/min": "breath/min",
    },
    abbreviation_expansions={
        "rr": "blood_pressure",
        "blutdruck": "blood_pressure",
        "hf": "heart_rate",
        "herzfrequenz": "heart_rate",
        "af": "respiratory_rate",
        "atemfrequenz": "respiratory_rate",
        "temp": "body_temperature",
        "temperatur": "body_temperature",
        "spo2": "oxygen_saturation",
    },
    frequency_aliases={
        "taeglich": "daily",
        "täglich": "daily",
        "1x taeglich": "once daily",
        "1x täglich": "once daily",
        "2x taeglich": "twice daily",
        "2x täglich": "twice daily",
        "alle 8 stunden": "every 8 hours",
        "bei bedarf": "prn",
    },
    duration_unit_aliases={"tag": "days", "tage": "days", "woche": "weeks"},
    abnormal_flags={"hoch": "high", "niedrig": "low", "kritisch": "critical"},
)

CHINESE_NORM_LEXICON = ClinicalNormLexicon(
    language="zh",
    decimal_separator=".",
    thousands_separators=(",", _NO_BREAK_SPACE, _NARROW_NO_BREAK_SPACE),
    unit_aliases={
        **_COMMON_UNIT_ALIASES,
        "毫克/分升": "mg/dL",
        "毫克每分升": "mg/dL",
        "克/升": "g/L",
        "克每升": "g/L",
        "毫摩尔/升": "mmol/L",
        "毫米汞柱": "mmHg",
        "摄氏度": "Cel",
        "℃": "Cel",
        "次/分": "beat/min",
        "次/分钟": "beat/min",
    },
    abbreviation_expansions={
        "血压": "blood_pressure",
        "心率": "heart_rate",
        "脉搏": "heart_rate",
        "呼吸频率": "respiratory_rate",
        "呼吸": "respiratory_rate",
        "体温": "body_temperature",
        "血氧": "oxygen_saturation",
        "血氧饱和度": "oxygen_saturation",
        "spo2": "oxygen_saturation",
    },
    frequency_aliases={
        "每日一次": "once daily",
        "每日两次": "twice daily",
        "每日三次": "three times daily",
        "每8小时": "every 8 hours",
        "按需": "prn",
    },
    duration_unit_aliases={"天": "days", "日": "days", "周": "weeks"},
    abnormal_flags={"高": "high", "低": "low", "危急": "critical", "正常": "normal"},
    token_boundaries=False,
)

_LEXICONS: dict[str, ClinicalNormLexicon] = {}

for _lexicon in (
    ENGLISH_NORM_LEXICON,
    SPANISH_NORM_LEXICON,
    FRENCH_NORM_LEXICON,
    GERMAN_NORM_LEXICON,
    CHINESE_NORM_LEXICON,
):
    register_clinical_norm_lexicon(_lexicon)


__all__ = [
    "ClinicalNormLexicon",
    "abbreviation_expansion",
    "abbreviation_surfaces",
    "abnormal_flag_alias",
    "available_clinical_norm_languages",
    "canonical_unit_alias",
    "clinical_norm_lexicon_stats",
    "get_clinical_norm_lexicon",
    "localized_duration_text",
    "localized_frequency_text",
    "normalize_language",
    "normalize_surface",
    "normalize_unit_surface",
    "parse_locale_number",
    "register_clinical_norm_lexicon",
    "split_measurement_text",
]
