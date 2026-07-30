"""Multilingual deterministic ConText cue lexicons.

The cue packs are compact, permissively redistributable surface-form tables
for the rule-based ConText layer. They are inspired by public NegEx/ConText
trigger translations and reviewed as synthetic lexicon entries, not copied
from restricted clinical corpora or gated terminology assets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalCueLexicon:
    """Language-specific cue pack for deterministic clinical ConText axes."""

    language: str
    negation: tuple[str, ...]
    pseudo_negation: tuple[str, ...]
    historical: tuple[str, ...]
    hypothetical: tuple[str, ...]
    recent: tuple[str, ...]
    uncertainty: tuple[str, ...]
    backward: tuple[str, ...]
    scope_terminators: tuple[str, ...]
    conjunction_terminators: tuple[str, ...]
    token_boundaries: bool = True


ENGLISH_CONTEXT_LEXICON = ClinicalCueLexicon(
    language="en",
    historical=(
        "history of",
        "hx of",
        "h/o",
        "status post",
        "s/p",
        "previous",
        "previously",
        "prior",
        "in the past",
        "past medical history",
        "pmh",
        "resolved",
    ),
    hypothetical=(
        "if",
        "should",
        "in case of",
        "in case",
        "in the event of",
        "unless",
    ),
    recent=(
        "active",
        "acute",
        "current",
        "currently",
        "new",
        "new onset",
        "ongoing",
    ),
    uncertainty=(
        "cannot exclude",
        "can't exclude",
        "concern for",
        "concerning for",
        "suspicious for",
        "suspicion for",
        "worrisome for",
        "question of",
        "to rule out",
        "rule out",
        "not ruled out",
        "not yet ruled out",
        "not completely ruled out",
        "not been ruled out",
        "cannot be excluded",
        "can't be excluded",
        "in the event of",
        "in case of",
        "in case",
        "versus",
        "probable",
        "probably",
        "possible",
        "possibly",
        "suspected",
        "suspect",
        "unlikely",
        "likely",
        "unless",
        "should",
        "might",
        "could",
        "may",
        "r/o",
        "vs",
        "if",
    ),
    negation=(
        "no evidence of",
        "no evidence",
        "no sign of",
        "no signs of",
        "negative for",
        "absence of",
        "free of",
        "ruled out",
        "not present",
        "denies",
        "denied",
        "deny",
        "without",
        "absent",
        "never",
        "none",
        "not",
        "no",
    ),
    pseudo_negation=(
        "no increase",
        "no interval increase",
        "no significant increase",
        "not ruled out",
        "not yet ruled out",
        "not completely ruled out",
        "not been ruled out",
        "cannot be excluded",
        "can't be excluded",
        "cannot exclude",
        "can't exclude",
    ),
    backward=(
        "absent",
        "can't be excluded",
        "cannot be excluded",
        "not been ruled out",
        "not completely ruled out",
        "not present",
        "not ruled out",
        "not yet ruled out",
        "resolved",
        "ruled out",
    ),
    scope_terminators=("and", "but", "however", "or"),
    conjunction_terminators=("but", "however", "although"),
)

SPANISH_CONTEXT_LEXICON = ClinicalCueLexicon(
    language="es",
    historical=(
        "antecedente de",
        "antecedentes de",
        "historia de",
        "previo",
        "previa",
        "previamente",
        "pasado",
        "resuelto",
    ),
    hypothetical=("si", "en caso de", "a menos que", "deberia", "debería"),
    recent=("activo", "activa", "agudo", "aguda", "actual", "actualmente", "nuevo"),
    uncertainty=(
        "posible",
        "probable",
        "sospecha de",
        "sospechoso de",
        "para descartar",
        "descartar",
        "no se descarta",
        "no puede descartarse",
        "no se puede excluir",
        "puede",
        "podria",
        "podría",
        "si",
        "en caso de",
    ),
    negation=(
        "sin evidencia de",
        "sin evidencia",
        "negativo para",
        "ausencia de",
        "descartado",
        "descartada",
        "niega",
        "nego",
        "negó",
        "no presenta",
        "sin",
        "no",
    ),
    pseudo_negation=(
        "no se descarta",
        "no puede descartarse",
        "no puede ser descartado",
        "no se puede excluir",
    ),
    backward=("descartado", "descartada", "resuelto"),
    scope_terminators=("y", "pero", "sin embargo", "o", "aunque"),
    conjunction_terminators=("pero", "sin embargo", "aunque"),
)

FRENCH_CONTEXT_LEXICON = ClinicalCueLexicon(
    language="fr",
    historical=(
        "antécédent de",
        "antécédents de",
        "histoire de",
        "ancien",
        "ancienne",
        "auparavant",
        "précédent",
        "résolu",
    ),
    hypothetical=("si", "en cas de", "à moins que", "devrait"),
    recent=("actif", "active", "aigu", "aiguë", "actuel", "nouveau", "en cours"),
    uncertainty=(
        "possible",
        "probable",
        "suspicion de",
        "suspect de",
        "à exclure",
        "exclure",
        "ne peut être exclue",
        "ne peut être exclu",
        "ne peut pas être exclue",
        "ne peut pas être exclu",
        "pourrait",
        "peut",
        "si",
        "en cas de",
    ),
    negation=(
        "aucune preuve de",
        "aucun signe de",
        "négatif pour",
        "absence de",
        "pas de",
        "sans",
        "nie",
        "nié",
        "aucune",
        "aucun",
        "non",
        "pas",
    ),
    pseudo_negation=(
        "ne peut être exclue",
        "ne peut être exclu",
        "ne peut pas être exclue",
        "ne peut pas être exclu",
        "pas exclu",
    ),
    backward=("exclu", "exclue", "résolu"),
    scope_terminators=("et", "mais", "cependant", "ou", "bien que"),
    conjunction_terminators=("mais", "cependant", "bien que"),
)

GERMAN_CONTEXT_LEXICON = ClinicalCueLexicon(
    language="de",
    historical=(
        "anamnese von",
        "vorgeschichte von",
        "zustand nach",
        "z.n.",
        "früher",
        "zuvor",
        "vorherige",
        "abgeklungen",
    ),
    hypothetical=("wenn", "falls", "im fall von", "sofern", "sollte"),
    recent=("aktiv", "akut", "aktuell", "derzeit", "neu", "laufend"),
    uncertainty=(
        "möglich",
        "wahrscheinlich",
        "verdacht auf",
        "ausschluss",
        "zum ausschluss",
        "nicht ausgeschlossen",
        "kann nicht ausgeschlossen werden",
        "nicht auszuschließen",
        "könnte",
        "kann",
        "wenn",
        "falls",
    ),
    negation=(
        "kein hinweis auf",
        "negativ für",
        "ausschluss von",
        "ausgeschlossen",
        "verneint",
        "ohne",
        "keinen",
        "keine",
        "kein",
        "nicht",
    ),
    pseudo_negation=(
        "nicht ausgeschlossen",
        "kann nicht ausgeschlossen werden",
        "nicht auszuschließen",
    ),
    backward=("ausgeschlossen", "abgeklungen"),
    scope_terminators=("und", "aber", "jedoch", "oder", "obwohl"),
    conjunction_terminators=("aber", "jedoch", "obwohl"),
)

CHINESE_CONTEXT_LEXICON = ClinicalCueLexicon(
    language="zh",
    historical=("既往史", "既往", "病史", "曾有", "既往有", "已缓解"),
    hypothetical=("如果", "若", "如", "除非"),
    recent=("急性", "当前", "目前", "新发", "活动性", "持续"),
    uncertainty=(
        "可能",
        "疑似",
        "考虑",
        "不能排除",
        "尚不能排除",
        "无法排除",
        "待排",
        "如果",
        "若",
    ),
    negation=(
        "无证据提示",
        "未见",
        "未发现",
        "阴性",
        "否认",
        "没有",
        "并非",
        "无",
        "不",
    ),
    pseudo_negation=("不能排除", "尚不能排除", "无法排除"),
    backward=("排除", "已缓解"),
    scope_terminators=("但", "但是", "然而", "和", "或", "以及"),
    conjunction_terminators=("但", "但是", "然而"),
    token_boundaries=False,
)

HINDI_CONTEXT_LEXICON = ClinicalCueLexicon(
    language="hi",
    historical=(
        "का इतिहास",
        "पूर्व इतिहास",
        "इतिहास",
        "पहले",
        "पूर्व",
        "पुराना",
        "ठीक हो चुका",
    ),
    hypothetical=("यदि", "अगर", "के मामले में", "जब तक", "हो तो"),
    recent=("सक्रिय", "तीव्र", "वर्तमान", "अभी", "नया", "चल रहा"),
    uncertainty=(
        "संभव",
        "संभावित",
        "संदेह",
        "संदेहास्पद",
        "खारिज करने के लिए",
        "इनकार नहीं किया जा सकता",
        "इंकार नहीं किया जा सकता",
        "हो सकता है",
        "यदि",
        "अगर",
    ),
    negation=(
        "कोई प्रमाण नहीं",
        "का कोई प्रमाण नहीं",
        "नकारात्मक",
        "इनकार करता है",
        "इनकार किया",
        "इंकार किया",
        "बिना",
        "नहीं",
    ),
    pseudo_negation=(
        "इनकार नहीं किया जा सकता",
        "इंकार नहीं किया जा सकता",
        "से इंकार नहीं किया जा सकता",
        "खारिज नहीं किया जा सकता",
        "बाहर नहीं किया जा सकता",
    ),
    backward=("खारिज", "ठीक हो चुका"),
    scope_terminators=("और", "लेकिन", "परंतु", "हालांकि"),
    conjunction_terminators=("लेकिन", "परंतु", "हालांकि"),
    token_boundaries=False,
)

_LEXICONS: dict[str, ClinicalCueLexicon] = {}


def _normalize_language(language: str) -> str:
    code = language.casefold().replace("_", "-").strip()
    return code.split("-", 1)[0] or "en"


def register_clinical_cue_lexicon(
    lexicon: ClinicalCueLexicon,
) -> ClinicalCueLexicon:
    """Register or replace a clinical ConText cue lexicon."""

    code = _normalize_language(lexicon.language)
    normalized = ClinicalCueLexicon(
        language=code,
        negation=tuple(lexicon.negation),
        pseudo_negation=tuple(lexicon.pseudo_negation),
        historical=tuple(lexicon.historical),
        hypothetical=tuple(lexicon.hypothetical),
        recent=tuple(lexicon.recent),
        uncertainty=tuple(lexicon.uncertainty),
        backward=tuple(lexicon.backward),
        scope_terminators=tuple(lexicon.scope_terminators),
        conjunction_terminators=tuple(lexicon.conjunction_terminators),
        token_boundaries=bool(lexicon.token_boundaries),
    )
    _LEXICONS[code] = normalized
    return normalized


def get_clinical_cue_lexicon(language: str | None = None) -> ClinicalCueLexicon:
    """Return a language pack, falling back to English for unknown codes."""

    code = _normalize_language(language or "en")
    return _LEXICONS.get(code, _LEXICONS["en"])


def available_clinical_cue_languages() -> tuple[str, ...]:
    """Return registered clinical ConText language codes."""

    return tuple(sorted(_LEXICONS))


def clinical_context_lexicon_stats(
    language: str | None = None,
) -> Mapping[str, Mapping[str, int]]:
    """Return cue counts by language and ConText axis."""

    languages = (
        (_normalize_language(language),)
        if language
        else available_clinical_cue_languages()
    )
    stats: dict[str, dict[str, int]] = {}
    for code in languages:
        lexicon = get_clinical_cue_lexicon(code)
        stats[lexicon.language] = {
            "negation": len(lexicon.negation),
            "pseudo_negation": len(lexicon.pseudo_negation),
            "historical": len(lexicon.historical),
            "hypothetical": len(lexicon.hypothetical),
            "recent": len(lexicon.recent),
            "uncertainty": len(lexicon.uncertainty),
            "backward": len(lexicon.backward),
            "scope_terminators": len(lexicon.scope_terminators),
            "conjunction_terminators": len(lexicon.conjunction_terminators),
        }
    return stats


for _lexicon in (
    ENGLISH_CONTEXT_LEXICON,
    SPANISH_CONTEXT_LEXICON,
    FRENCH_CONTEXT_LEXICON,
    GERMAN_CONTEXT_LEXICON,
    CHINESE_CONTEXT_LEXICON,
    HINDI_CONTEXT_LEXICON,
):
    register_clinical_cue_lexicon(_lexicon)


__all__ = [
    "ClinicalCueLexicon",
    "available_clinical_cue_languages",
    "clinical_context_lexicon_stats",
    "get_clinical_cue_lexicon",
    "register_clinical_cue_lexicon",
]
