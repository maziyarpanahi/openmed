"""Mapping from OpenMed language codes to Faker locales.

Faker speaks locale strings like ``en_US``, ``pt_PT``, ``fr_FR``. This module
resolves OpenMed's ISO 639-1 codes (used everywhere else in the library) to
the most appropriate Faker locale.

Notes:
- Telugu (``te``) has no Faker locale; we fall back to ``en_IN`` so generated
  surrogates stay culturally adjacent. This is documented and surfaced to
  callers as a ``UserWarning`` the first time it's used.
- Portuguese defaults to ``pt_PT``; pass ``locale="pt_BR"`` explicitly to
  generate Brazilian-Portuguese surrogates (matters for CPF/CNPJ context).

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

# Default conceptual Faker locale per OpenMed language code. Some conceptual
# locales are backed by another installed Faker locale at runtime; see
# ``FAKER_BACKEND_LOCALE``.
LANG_TO_LOCALE: Final[Mapping[str, str]] = {
    "en": "en_US",
    "fr": "fr_FR",
    "de": "de_DE",
    "it": "it_IT",
    "es": "es_ES",
    "nl": "nl_NL",
    "hi": "hi_IN",
    "te": "en_IN",  # Faker has no Telugu locale; en_IN is the closest match
    "pt": "pt_PT",
    "ar": "ar_EG",  # Egypt is the most-populous Arabic-speaking country; override for Gulf/Levant locales.
    "he": "he_IL",
    "ja": "ja_JP",
    "zh": "zh_CN",  # CJK PERSON spans draw family-name-first Chinese surrogates
    "tr": "tr_TR",
    "id": "id_ID",
    "th": "th_TH",
    "pl": "pl_PL",
    "lv": "lv_LV",
    "ko": "ko_KR",
    "cs": "cs_CZ",
    "sk": "sk_SK",
    "ms": "ms_MY",
    "tl": "fil_PH",
    "da": "da_DK",
    "ro": "ro_RO",
    "fi": "fi_FI",
    "bg": "bg_BG",
    "hr": "hr_HR",
    "sr": "sr_RS",  # Faker has no Serbian locale; backed by hr_HR at runtime
    "hu": "hu_HU",
    "et": "et_EE",
    "el": "el_GR",
    "vi": "vi_VN",
}


# Languages whose default locale is a known approximation rather than a
# direct match. Used to emit a one-time warning so callers can override.
_APPROXIMATE_LOCALES: Final = frozenset({"te", "ms", "sr"})


# Conceptual locale -> installed Faker locale. This keeps national-ID dispatch
# keyed by the target country while allowing generic names/addresses to use a
# nearby installed Faker backend.
FAKER_BACKEND_LOCALE: Final[Mapping[str, str]] = {
    "ms_MY": "id_ID",
    "sr_RS": "hr_HR",
}


# Per-language national-ID surrogate providers — the single source of truth for
# the OM-135 round-trip fidelity suite. Maps each language that has a
# validator-backed national ID to the ``(faker_locale, faker_method)`` whose
# generated surrogates pass that language's registered validator(s) in
# :mod:`openmed.core.pii_i18n`. The locale here can differ from the language's
# default display locale when the registered validators target another country's
# format: Portuguese national-ID validation is Brazilian CPF/CNPJ, so ``pt``
# draws ID surrogates from ``pt_BR`` even though its default locale (names,
# addresses, ...) stays ``pt_PT``. The method must match the registry's
# locale-aware dispatch (``registry._LOCALE_ID_METHODS``); the regression suite
# asserts that and the round-trip.
NATIONAL_ID_PROVIDERS: Final[Mapping[str, tuple[str, str]]] = {
    "en": ("en_US", "ssn"),
    "fr": ("fr_FR", "ssn"),  # NIR / INSEE
    "de": ("de_DE", "german_steuer_id"),  # Steuer-ID
    "it": ("it_IT", "ssn"),  # Codice Fiscale
    "es": ("es_ES", "nie"),  # NIE
    "nl": ("nl_NL", "ssn"),  # BSN
    "hi": ("hi_IN", "aadhaar"),  # Aadhaar (Verhoeff)
    "te": ("en_IN", "aadhaar"),  # Aadhaar via approximate en_IN
    # CPF drives the OM-135 coherence round-trip; pt_PT surrogates draw NIF via
    # the locale-keyed registry dispatch (``registry._LOCALE_ID_METHODS``).
    "pt": ("pt_BR", "cpf"),
    "tr": ("tr_TR", "ssn"),  # TCKN
    "he": ("he_IL", "teudat_zehut"),  # Israeli Teudat Zehut
    "id": ("id_ID", "indonesian_nik"),  # NIK
    "th": ("th_TH", "thai_national_id"),  # Thai 13-digit national ID
    "pl": ("pl_PL", "pesel"),  # PESEL
    "lv": ("lv_LV", "personas_kods"),
    "ko": ("ko_KR", "korean_rrn"),  # RRN
    "cs": ("cs_CZ", "rodne_cislo"),  # Czech rodne cislo (shared provider)
    "sk": ("sk_SK", "rodne_cislo"),  # Slovak rodne cislo
    "ms": ("ms_MY", "mykad"),  # Malaysian MyKad / NRIC
    "tl": ("fil_PH", "philsys_psn"),  # Philippine PhilSys PSN
    "da": ("da_DK", "danish_cpr"),  # Danish CPR / personnummer
    "ro": ("ro_RO", "romanian_cnp"),  # CNP (Cod Numeric Personal)
    "fi": ("fi_FI", "ssn"),  # Finnish HETU (Faker's native fi_FI ssn)
    "bg": ("bg_BG", "egn"),  # Bulgarian EGN (unified civil number)
    "hr": ("hr_HR", "ssn"),  # Croatian OIB (Faker's native hr_HR ssn)
    "sr": ("sr_RS", "jmbg"),  # Serbian / ex-Yugoslav JMBG
    "hu": ("hu_HU", "hungarian_taj"),  # TAJ social-security identifier
    "et": ("et_EE", "isikukood"),  # Estonian isikukood
    "el": ("el_GR", "ssn"),  # Greek AMKA (Faker's native el_GR ssn)
    "vi": ("vi_VN", "vietnamese_cccd"),  # 12-digit CCCD
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


def resolve_locale(lang: str, locale_override: str | None = None) -> str:
    """Resolve a Faker locale for ``lang``.

    Args:
        lang: ISO 639-1 language code (``en``, ``fr``, ``de``, ...).
        locale_override: Caller-supplied Faker locale (e.g. ``pt_BR``) or
            documented region tag (e.g. ``ar-SA``); takes precedence.

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

    locale = LANG_TO_LOCALE.get(lang)
    if locale is None:
        return LANG_TO_LOCALE["en"]

    if lang in _APPROXIMATE_LOCALES and lang not in _warned:
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
      - ``id_locale``: the Faker locale those providers are drawn from, or
        ``None``. Usually equals ``locale``; differs when the registered
        validators target another country's format (e.g. ``pt`` -> ``pt_BR``).
    """
    from ..pii_i18n import (  # lazy: avoid import cycle
        NATIONAL_ID_ONLY_LANGUAGES,
        SUPPORTED_LANGUAGES,
    )

    rows: list[dict[str, object]] = []
    for lang in sorted(SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES):
        provider: tuple[str, str] | None = NATIONAL_ID_PROVIDERS.get(lang)
        id_locale, id_method = provider if provider else (None, None)
        rows.append(
            {
                "language": lang,
                "locale": LANG_TO_LOCALE.get(lang, LANG_TO_LOCALE["en"]),
                "approximate": lang in _APPROXIMATE_LOCALES,
                "id_providers": [id_method] if id_method else [],
                "id_locale": id_locale,
            }
        )
    return rows


__all__ = [
    "AR_REGION_LOCALES",
    "LANG_TO_LOCALE",
    "FAKER_BACKEND_LOCALE",
    "NATIONAL_ID_PROVIDERS",
    "list_regional_locales",
    "locale_coherence_report",
    "resolve_faker_backend_locale",
    "resolve_locale",
]
