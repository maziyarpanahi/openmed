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
"""

from __future__ import annotations

import warnings
from typing import Final, Mapping


# Default Faker locale per OpenMed language code.
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
}


# Languages whose default locale is a known approximation rather than a
# direct match. Used to emit a one-time warning so callers can override.
_APPROXIMATE_LOCALES: Final = frozenset({"te"})

_warned: set[str] = set()


def resolve_locale(lang: str, locale_override: str | None = None) -> str:
    """Resolve a Faker locale for ``lang``.

    Args:
        lang: ISO 639-1 language code (``en``, ``fr``, ``de``, ...).
        locale_override: Caller-supplied locale (e.g. ``pt_BR``); takes
            precedence and skips the warning.

    Returns:
        A Faker locale string.
    """
    if locale_override:
        return locale_override

    locale = LANG_TO_LOCALE.get(lang)
    if locale is None:
        return LANG_TO_LOCALE["en"]

    if lang in _APPROXIMATE_LOCALES and lang not in _warned:
        warnings.warn(
            f"OpenMed: language {lang!r} has no native Faker locale; "
            f"falling back to {locale!r}. Pass locale=... to override.",
            UserWarning,
            stacklevel=3,
        )
        _warned.add(lang)

    return locale


__all__ = ["LANG_TO_LOCALE", "resolve_locale"]
