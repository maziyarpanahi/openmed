"""Mapping from OpenMed language codes to Faker locales.

Faker speaks locale strings like ``en_US``, ``pt_PT``, ``fr_FR``. This module
resolves OpenMed's ISO 639-1 codes (used everywhere else in the library) to
the most appropriate Faker locale.

Notes:
- Telugu (``te``) has no Faker locale; non-script-specific values fall back to
  ``en_IN`` and surface a one-time ``UserWarning``. Native Telugu name
  surrogates bypass that approximate path and therefore do not warn.
- Amharic (``am``) has no Faker locale; the conceptual ``am_ET`` locale uses
  ``en_KE`` as its runtime backend while curated Ethiopic surrogate data stays
  available through the language pack.
- Portuguese defaults to ``pt_PT``; pass ``locale="pt_BR"`` explicitly to
  generate Brazilian-Portuguese surrogates (matters for CPF/CNPJ context).
- Chinese resolves to ``zh_CN`` so PERSON/FIRST_NAME/LAST_NAME dispatch uses
  the surname-aware, Han-only surrogate generators rather than a Latin fallback.

Regression contract (OM-135):
- Every ``openmed.core.pii_i18n.SUPPORTED_LANGUAGES`` code must have a
  :data:`LANG_TO_LOCALE` entry whose locale exists in Faker (or is a
  documented approximation in :data:`_APPROXIMATE_LOCALES`).
- Every language with a validator-backed national ID must appear in
  :data:`NATIONAL_ID_PROVIDERS`, and its generated surrogates must round-trip
  that language's registered validator in :mod:`openmed.core.pii_i18n`.
  Some entries are national-ID-only and intentionally not full model-backed
  ``SUPPORTED_LANGUAGES`` yet.
- Only the documented approximations may emit the locale ``UserWarning``.
  ``tests/unit/core/test_locale_coherence.py`` gates all three so a new
  language pack mis-wired to a wrong locale or provider fails loudly. Use
  :func:`locale_coherence_report` for a per-language summary.
"""

from __future__ import annotations

import warnings
from typing import Final, Mapping

from ..language_pack_catalog import LANG_TO_LOCALE, NATIONAL_ID_PROVIDERS

ZH_CN_ADDRESS_LOCALE: Final = "zh_CN"
"""Faker locale used by Chinese hierarchical address surrogates."""

# Languages whose default locale is a known approximation rather than a
# direct match. Used to emit a one-time warning so callers can override.
_APPROXIMATE_LOCALES: Final = frozenset(
    {
        "af",
        "am",
        "as",
        "kn",
        "ml",
        "mr",
        "ms",
        "pa",
        "rw",
        "sr",
        "te",
        "ur",
        "xh",
    }
)


# Conceptual locale -> installed Faker locale. This keeps national-ID dispatch
# keyed by the target country while allowing generic names/addresses to use a
# nearby installed Faker backend.
FAKER_BACKEND_LOCALE: Final[Mapping[str, str]] = {
    "af_ZA": "zu_ZA",
    "am_ET": "en_KE",
    "ar_MA": "ar_EG",
    "as_IN": "bn_BD",
    "en_ET": "en_US",
    "en_TZ": "en_US",
    "en_UG": "en_US",
    "en_ZA": "zu_ZA",
    "en_GH": "tw_GH",
    "fr_MA": "fr_FR",
    "kn_IN": "en_IN",
    "ml_IN": "en_IN",
    "mr_IN": "hi_IN",
    "ms_MY": "id_ID",
    "pa_IN": "en_IN",
    "rw_RW": "en_US",
    "sr_RS": "hr_HR",
    "sw_TZ": "sw",
    "ur_PK": "en_PK",
    "xh_ZA": "zu_ZA",
}


# Region-qualified Arabic codes -> Faker locale. Bare ``ar`` stays ``ar_EG``
# (see LANG_TO_LOCALE); these let callers request a Gulf/Levant flavour so
# surrogate names, phones and addresses read in-region (OM-285).
AR_REGION_LOCALES: Final[Mapping[str, str]] = {
    "ar-EG": "ar_EG",  # Egypt (the bare-"ar" default, exposed explicitly too)
    "ar-SA": "ar_SA",  # Saudi Arabia
    "ar-AE": "ar_AE",  # United Arab Emirates
    "ar-JO": "ar_JO",  # Jordan
    "ar-PS": "ar_PS",  # Palestine
}


def _available_faker_locales() -> frozenset[str]:
    """Return the locales the installed Faker version actually ships."""
    try:
        from faker.config import AVAILABLE_LOCALES
    except Exception:  # pragma: no cover - Faker always ships this
        return frozenset()
    return frozenset(AVAILABLE_LOCALES)


# Validated once at import: the subset of AR_REGION_LOCALES whose Faker locale
# is installed. Region tags whose locale is missing fall back to ``ar_EG`` with
# a one-time warning at resolve time.
_AR_REGION_AVAILABLE: Final[Mapping[str, str]] = {
    tag: loc
    for tag, loc in AR_REGION_LOCALES.items()
    if loc in _available_faker_locales()
}


_warned: set[str] = set()


def _resolve_arabic_region(tag: str) -> str:
    """Resolve one region-qualified Arabic tag with a safe fallback."""
    locale = _AR_REGION_AVAILABLE.get(tag)
    if locale is not None:
        return locale
    if tag not in _warned:
        warnings.warn(
            f"OpenMed: no Faker locale available for Arabic region {tag!r}; "
            f"falling back to {LANG_TO_LOCALE['ar']!r}.",
            UserWarning,
            stacklevel=4,
        )
        _warned.add(tag)
    return LANG_TO_LOCALE["ar"]


def resolve_locale(
    lang: str,
    locale_override: str | None = None,
    *,
    warn_approximation: bool = True,
) -> str:
    """Resolve a Faker locale for ``lang``.

    Args:
        lang: ISO 639-1 language code (``en``, ``fr``, ``de``, ...).
        locale_override: Caller-supplied Faker locale (e.g. ``pt_BR``) or
            documented region tag (e.g. ``ar-SA``); takes precedence.
        warn_approximation: Emit the documented one-time warning when the
            resolved Faker locale approximates ``lang``. Script-specific
            providers set this false because they do not use Faker's name data.

    Returns:
        A Faker locale string.
    """
    if locale_override:
        if locale_override.startswith("ar-"):
            return _resolve_arabic_region(locale_override)
        return locale_override

    # Region-qualified Arabic codes (e.g. "ar-SA") select a Gulf/Levant Faker
    # locale. Bare "ar" does NOT start with "ar-", so it skips this branch and
    # keeps the existing ar_EG default below unchanged.
    if lang.startswith("ar-"):
        return _resolve_arabic_region(lang)

    normalized_lang = lang.replace("-", "_")
    if normalized_lang in FAKER_BACKEND_LOCALE:
        return normalized_lang

    locale = LANG_TO_LOCALE.get(lang)
    if locale is None:
        return LANG_TO_LOCALE["en"]

    if warn_approximation and lang in _APPROXIMATE_LOCALES and lang not in _warned:
        backend = FAKER_BACKEND_LOCALE.get(locale)
        backend_note = f" backed by {backend!r}" if backend else ""
        warnings.warn(
            f"OpenMed: language {lang!r} has no native Faker locale; "
            f"using {locale!r}{backend_note}. Pass locale=... to override.",
            UserWarning,
            stacklevel=3,
        )
        _warned.add(lang)

    return locale


def list_regional_locales(lang: str) -> list[str]:
    """Return the supported region-qualified codes for ``lang``.

    Currently only Arabic (``ar``) has regional overrides. Returns every
    documented tag, including ``ar-EG`` (the explicit form of the bare-``ar``
    default). A listed tag whose locale is absent from the installed Faker
    version resolves to ``ar_EG`` with a one-time warning.
    """
    if lang == "ar":
        return sorted(AR_REGION_LOCALES)
    return []


def resolve_faker_backend_locale(locale: str) -> str:
    """Return the installed Faker locale backing a conceptual locale."""

    return FAKER_BACKEND_LOCALE.get(locale, locale)


def locale_coherence_report() -> list[dict[str, object]]:
    """Return one locale-coherence row per supported or ID-only language.

    Each row is a plain JSON-friendly ``dict`` (so the status/leaderboard work
    can reuse it) with:

      - ``language``: the OpenMed ISO 639-1 code.
      - ``locale``: the default Faker locale it resolves to (no warning side
        effect — read straight from :data:`LANG_TO_LOCALE`).
      - ``approximate``: ``True`` when that default locale is a documented
        approximation rather than a native match.
      - ``id_providers``: national-ID Faker method names whose surrogates
        round-trip the language's registered checksum validator (empty when the
        language has no checksummed national-ID surrogate provider).
      - ``id_types``: stable registry ID types covered by ``id_providers``.
      - ``id_locale``: the Faker locale those providers are drawn from, or
        ``None``. Usually equals ``locale``; differs when the registered
        validators target another country's format (e.g. ``pt`` -> ``pt_BR``).
    """
    from ..pii_i18n import (  # lazy: avoid import cycle
        INDIC_NER_LANGUAGES,
        NATIONAL_ID_ONLY_LANGUAGES,
        SUPPORTED_LANGUAGES,
    )

    rows: list[dict[str, object]] = []
    reported_languages = (
        SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES | INDIC_NER_LANGUAGES
    )
    for lang in sorted(reported_languages):
        provider: tuple[str, str] | None = NATIONAL_ID_PROVIDERS.get(lang)
        id_locale, id_method = provider if provider else (None, None)
        id_types = [id_method] if id_method is not None else []
        id_methods = [id_method] if id_method is not None else []
        if lang in {"hi", "te"}:
            id_types = ["aadhaar", "pan", "gstin", "abha"]
            id_methods = ["aadhaar", "pan", "gstin", "abha"]
        rows.append(
            {
                "language": lang,
                "locale": LANG_TO_LOCALE.get(lang, LANG_TO_LOCALE["en"]),
                "approximate": lang in _APPROXIMATE_LOCALES,
                "id_providers": id_methods,
                "id_types": id_types,
                "id_locale": id_locale,
            }
        )
    return rows


__all__ = [
    "AR_REGION_LOCALES",
    "LANG_TO_LOCALE",
    "FAKER_BACKEND_LOCALE",
    "NATIONAL_ID_PROVIDERS",
    "ZH_CN_ADDRESS_LOCALE",
    "list_regional_locales",
    "locale_coherence_report",
    "resolve_faker_backend_locale",
    "resolve_locale",
]
