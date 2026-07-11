"""Unit tests for regional Arabic Faker-locale overrides (OM-285)."""

from __future__ import annotations

import warnings

import pytest

from openmed.core.anonymizer import locales
from openmed.core.anonymizer.locales import (
    AR_REGION_LOCALES,
    list_regional_locales,
    resolve_locale,
)


@pytest.fixture(autouse=True)
def _reset_warning_cache():
    """Start each test with a clean one-time-warning cache."""
    locales._warned.clear()
    yield
    locales._warned.clear()


def test_bare_ar_unchanged_and_silent():
    # Acceptance: resolve_locale('ar') still returns 'ar_EG' with no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn ANY warning into a test failure
        assert resolve_locale("ar") == "ar_EG"


def test_region_qualified_selects_regional_locale():
    # Acceptance: resolve_locale('ar-SA') returns 'ar_SA' (when Faker ships it).
    available = set(list_regional_locales("ar"))
    for tag, faker_locale in AR_REGION_LOCALES.items():
        expected = faker_locale if tag in available else "ar_EG"
        assert resolve_locale(tag) == expected
        locales._warned.clear()  # isolate per-tag warning state


def test_unknown_region_warns_once_and_falls_back():
    # Acceptance: unknown region warns and falls back to ar_EG...
    with pytest.warns(UserWarning):
        assert resolve_locale("ar-ZZ") == "ar_EG"
    # ...and does NOT warn again on a repeat call (one-time warning).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_locale("ar-ZZ") == "ar_EG"


def test_list_regional_locales():
    # Acceptance: returns the documented supported Arabic region tags.
    tags = list_regional_locales("ar")
    assert tags == sorted(tags)  # returned sorted
    assert set(tags) <= set(AR_REGION_LOCALES)  # only documented tags
    assert list_regional_locales("en") == []  # non-Arabic has none


def test_locale_override_takes_precedence():
    # An explicit override always wins, even over a region code.
    assert resolve_locale("ar-SA", locale_override="fr_FR") == "fr_FR"
