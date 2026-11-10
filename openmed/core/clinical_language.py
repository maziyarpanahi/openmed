"""Explicit locale controls and conservative automatic clinical language routing."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence

from .language_pack_catalog import LANG_TO_LOCALE
from .language_router import (
    LanguagePrediction,
    LanguageRouter,
    LanguageRun,
    PyCLD2LanguageIdentifier,
)

_LANGUAGE = re.compile(r"([a-z]{2,3})(?:[-_]([a-z]{2}))?", re.I)
_LOCALES = frozenset(LANG_TO_LOCALE.values()) | {
    "de_AT",
    "de_CH",
    "en_GB",
    "en_CA",
    "en_AU",
    "fr_CA",
    "fr_CH",
    "fr_BE",
    "es_MX",
}


class ReliableClinicalLanguageIdentifier(PyCLD2LanguageIdentifier):
    """Use the existing optional CLD2 adapter with its reliability gate intact."""

    name = "pycld2:reliable"

    def identify(
        self, text: str, candidates: Sequence[str]
    ) -> LanguagePrediction | None:
        """Return a reliable language share, not a calibrated accuracy score."""
        module = self._load()
        if module is None or not text.strip():
            return None
        try:
            reliable, _bytes, details = module.detect(text, bestEffort=False)
        except Exception:
            return None
        if not reliable:
            return None
        ranked = [
            (float(percent) / 100, str(code).lower().split("-")[0])
            for _name, code, percent, _score in details
        ]
        if not ranked:
            return None
        confidence, code = max(ranked)
        if code not in candidates:
            return None
        return LanguagePrediction(code, max(0.0, min(1.0, confidence)))


@dataclass(frozen=True)
class ClinicalLanguage:
    """A language hint and source-aligned runs, separate from model qualification."""

    language: str
    locale: str
    confidence: float
    source: str
    runs: tuple[LanguageRun, ...]
    needs_review: bool
    mixed: bool = False


def normalize_clinical_language(language: str) -> tuple[str, str | None]:
    """Normalize DE/de-DE/de_DE without losing the requested locale."""
    if not isinstance(language, str):
        raise ValueError("language must be a language code or auto")
    if language.strip().lower() == "auto":
        return "auto", None
    match = _LANGUAGE.fullmatch(language.strip())
    if not match:
        raise ValueError(
            "language must use a registered language or language-region code"
        )
    code = match[1].lower()
    if code not in LANG_TO_LOCALE:
        raise ValueError("language has no registered clinical preprocessing pack")
    locale = f"{code}_{match[2].upper()}" if match[2] else None
    if locale and locale not in _LOCALES:
        raise ValueError("locale has no registered clinical formatting route")
    return code, locale


def resolve_clinical_language(
    text: str,
    *,
    language: str = "auto",
    locale: str | None = None,
    router: LanguageRouter | None = None,
) -> ClinicalLanguage:
    """Resolve explicit controls or conservative paragraph-level language hints.

    Automatic script/pack fallback remains review-required. Short or mixed
    text cannot silently inherit an English qualification. Returned confidence
    is a routing hint, not a measurement of de-identification accuracy.
    """
    code, requested_locale = normalize_clinical_language(language)
    if locale is not None:
        locale_code, explicit_locale = normalize_clinical_language(locale)
        if explicit_locale is None:
            raise ValueError("locale must contain a language and region")
        if code != "auto" and locale_code != code:
            raise ValueError("locale conflicts with language")
        if requested_locale and explicit_locale != requested_locale:
            raise ValueError("locale conflicts with the language-region code")
        requested_locale = explicit_locale
    if code != "auto":
        return ClinicalLanguage(
            code, requested_locale or LANG_TO_LOCALE[code], 1.0, "explicit", (), False
        )
    router = router or LanguageRouter(
        language_identifier=ReliableClinicalLanguageIdentifier()
    )
    runs = []
    # Split paragraphs before script routing so mixed German/English Latin
    # paragraphs are not collapsed into a single document-level guess.
    for paragraph in re.finditer(r"[^\r\n]+(?:\r\n|[\r\n]|$)|[\r\n]+", text):
        for run in router.route_runs(paragraph[0]):
            runs.append(
                replace(
                    run,
                    start=run.start + paragraph.start(),
                    end=run.end + paragraph.start(),
                )
            )
    lexical = [
        run for run in runs if any(char.isalpha() for char in text[run.start : run.end])
    ]
    if not lexical:
        return ClinicalLanguage(
            "und", requested_locale or "en_US", 0.0, "uncertain", tuple(runs), True
        )
    totals: dict[str, int] = {}
    for run in lexical:
        totals[run.language] = totals.get(run.language, 0) + run.end - run.start
    dominant = max(totals, key=totals.get)
    confidence = sum(run.confidence * (run.end - run.start) for run in lexical) / sum(
        totals.values()
    )
    uncertain = any(
        run.source.startswith("stdlib:") or run.confidence < 0.8 for run in lexical
    )
    mixed = len(totals) > 1
    if requested_locale and not requested_locale.startswith(dominant + "_"):
        raise ValueError("locale conflicts with the detected language")
    return ClinicalLanguage(
        dominant,
        requested_locale or LANG_TO_LOCALE[dominant],
        confidence,
        "auto",
        tuple(runs),
        uncertain or mixed,
        mixed,
    )


__all__ = [
    "ClinicalLanguage",
    "ReliableClinicalLanguageIdentifier",
    "normalize_clinical_language",
    "resolve_clinical_language",
]
