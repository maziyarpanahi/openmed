"""Multilingual clinical section header lexicons.

The section packs are compact synthetic surface-form tables for deterministic
clinical-note section detection. They deliberately avoid restricted clinical
corpora and gated terminology assets.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SectionLexicon:
    """Language-specific clinical section header surface forms."""

    language: str
    sections: Mapping[str, tuple[str, ...]]
    token_boundaries: bool = True


ENGLISH_SECTION_LEXICON = SectionLexicon(
    language="en",
    sections={
        "past_medical_history": (
            "Past Medical History",
            "PMH",
            "Medical History",
        ),
        "history": ("History",),
        "family_history": (
            "Family History",
            "Family Medical History",
            "FH",
        ),
        "social_history": (
            "Social History",
            "Social Hx",
            "SH",
        ),
        "history_of_present_illness": (
            "History of Present Illness",
            "HPI",
        ),
        "assessment": (
            "Assessment",
            "Impression",
            "Assessment and Plan",
        ),
        "plan": ("Plan",),
    },
)

SPANISH_SECTION_LEXICON = SectionLexicon(
    language="es",
    sections={
        "past_medical_history": (
            "Antecedentes",
            "Antecedentes personales",
            "Historia medica",
            "Historia médica",
        ),
        "family_history": (
            "Antecedentes familiares",
            "Historia familiar",
        ),
        "social_history": (
            "Antecedentes sociales",
            "Historia social",
            "Habitos",
            "Hábitos",
        ),
        "history_of_present_illness": (
            "Historia de la enfermedad actual",
            "Enfermedad actual",
            "Motivo de consulta",
        ),
        "assessment": (
            "Evaluacion",
            "Evaluación",
            "Impresion",
            "Impresión",
        ),
        "plan": ("Plan", "Plan terapeutico", "Plan terapéutico"),
    },
)

FRENCH_SECTION_LEXICON = SectionLexicon(
    language="fr",
    sections={
        "past_medical_history": (
            "Antécédents",
            "Antecedents",
            "Antécédents médicaux",
            "Antecedents medicaux",
        ),
        "family_history": (
            "Antécédents familiaux",
            "Antecedents familiaux",
            "Histoire familiale",
        ),
        "social_history": (
            "Antécédents sociaux",
            "Antecedents sociaux",
            "Histoire sociale",
        ),
        "history_of_present_illness": (
            "Histoire de la maladie actuelle",
            "Motif de consultation",
        ),
        "assessment": ("Évaluation", "Evaluation", "Impression"),
        "plan": ("Plan", "Plan de soins"),
    },
)

GERMAN_SECTION_LEXICON = SectionLexicon(
    language="de",
    sections={
        "past_medical_history": (
            "Anamnese",
            "Vorgeschichte",
            "Medizinische Vorgeschichte",
        ),
        "family_history": (
            "Familienanamnese",
            "Familiengeschichte",
        ),
        "social_history": (
            "Sozialanamnese",
            "Sozialgeschichte",
        ),
        "history_of_present_illness": (
            "Aktuelle Anamnese",
            "Aktuelle Beschwerden",
        ),
        "assessment": ("Beurteilung", "Einschätzung", "Einschaetzung"),
        "plan": ("Plan", "Therapieplan"),
    },
)

CHINESE_SECTION_LEXICON = SectionLexicon(
    language="zh",
    sections={
        "past_medical_history": ("既往史", "既往病史"),
        "family_history": ("家族史", "家族病史"),
        "social_history": ("社会史", "个人史"),
        "history_of_present_illness": ("现病史", "主诉"),
        "assessment": ("评估", "诊断印象"),
        "plan": ("计划", "治疗计划"),
    },
    token_boundaries=False,
)

ARABIC_SECTION_LEXICON = SectionLexicon(
    language="ar",
    sections={
        "past_medical_history": ("التاريخ المرضي", "السوابق المرضية"),
        "family_history": ("تاريخ عائلي", "التاريخ العائلي"),
        "social_history": ("التاريخ الاجتماعي",),
        "history_of_present_illness": ("القصة المرضية الحالية", "الشكوى الحالية"),
        "assessment": ("التقييم", "الانطباع"),
        "plan": ("الخطة", "خطة العلاج"),
    },
    token_boundaries=False,
)

HEBREW_SECTION_LEXICON = SectionLexicon(
    language="he",
    sections={
        "past_medical_history": ("היסטוריה רפואית", "עבר רפואי"),
        "family_history": ("היסטוריה משפחתית",),
        "social_history": ("היסטוריה חברתית",),
        "history_of_present_illness": ("מחלה נוכחית", "תלונה עיקרית"),
        "assessment": ("הערכה",),
        "plan": ("תוכנית", "תכנית טיפול"),
    },
    token_boundaries=False,
)

_LEXICONS: dict[str, SectionLexicon] = {}


def normalize_section_header(text: str) -> str:
    """Return a script-preserving normalized section header key."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if char == "_" or category[0] in {"P", "S", "Z"}:
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


def _normalize_language(language: str) -> str:
    code = language.casefold().replace("_", "-").strip()
    return code.split("-", 1)[0] or "en"


def register_section_lexicon(lexicon: SectionLexicon) -> SectionLexicon:
    """Register or replace a clinical section header lexicon."""

    code = _normalize_language(lexicon.language)
    sections: dict[str, tuple[str, ...]] = {}
    for canonical, headers in lexicon.sections.items():
        canonical_key = str(canonical).strip()
        if not canonical_key:
            raise ValueError("section lexicon canonical keys must be non-empty")
        header_values = tuple(str(header).strip() for header in headers if header)
        if not header_values:
            raise ValueError(f"section {canonical_key!r} must include headers")
        sections[canonical_key] = header_values
    normalized = SectionLexicon(
        language=code,
        sections=sections,
        token_boundaries=bool(lexicon.token_boundaries),
    )
    _LEXICONS[code] = normalized
    return normalized


def get_section_lexicon(language: str | None = None) -> SectionLexicon:
    """Return a section language pack, falling back to English."""

    code = _normalize_language(language or "en")
    return _LEXICONS.get(code, _LEXICONS["en"])


def available_section_languages() -> tuple[str, ...]:
    """Return registered section header language codes."""

    return tuple(sorted(_LEXICONS))


def section_header_aliases(language: str | None = None) -> dict[str, str]:
    """Return raw section header aliases mapped to canonical section keys."""

    languages = (
        (_normalize_language(language),) if language else available_section_languages()
    )
    aliases: dict[str, str] = {}
    for code in languages:
        lexicon = get_section_lexicon(code)
        for canonical, headers in lexicon.sections.items():
            aliases[canonical] = canonical
            for header in headers:
                aliases[header] = canonical
    return aliases


def normalized_section_header_aliases(language: str | None = None) -> dict[str, str]:
    """Return normalized section header aliases mapped to canonical keys."""

    return {
        normalize_section_header(header): canonical
        for header, canonical in section_header_aliases(language).items()
    }


def section_lexicon_stats(
    language: str | None = None,
) -> Mapping[str, Mapping[str, int]]:
    """Return section header counts by language and canonical section key."""

    languages = (
        (_normalize_language(language),) if language else available_section_languages()
    )
    stats: dict[str, dict[str, int]] = {}
    for code in languages:
        lexicon = get_section_lexicon(code)
        stats[lexicon.language] = {
            canonical: len(headers)
            for canonical, headers in sorted(lexicon.sections.items())
        }
    return stats


for _lexicon in (
    ENGLISH_SECTION_LEXICON,
    SPANISH_SECTION_LEXICON,
    FRENCH_SECTION_LEXICON,
    GERMAN_SECTION_LEXICON,
    CHINESE_SECTION_LEXICON,
    ARABIC_SECTION_LEXICON,
    HEBREW_SECTION_LEXICON,
):
    register_section_lexicon(_lexicon)


__all__ = [
    "SectionLexicon",
    "available_section_languages",
    "get_section_lexicon",
    "normalize_section_header",
    "normalized_section_header_aliases",
    "register_section_lexicon",
    "section_header_aliases",
    "section_lexicon_stats",
]
