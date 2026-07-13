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
