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
    available = set(locales._AR_REGION_AVAILABLE)
    for tag, faker_locale in AR_REGION_LOCALES.items():
        expected = faker_locale if tag in available else "ar_EG"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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


def test_list_regional_locales_includes_every_documented_tag():
    # Acceptance: returns the documented supported Arabic region tags.
    tags = list_regional_locales("ar")
    assert tags == sorted(AR_REGION_LOCALES)
    assert list_regional_locales("en") == []


def test_missing_documented_locale_warns_once_and_falls_back(monkeypatch):
    monkeypatch.setattr(locales, "_AR_REGION_AVAILABLE", {})
    with pytest.warns(UserWarning, match="ar-SA"):
        assert resolve_locale("ar-SA") == "ar_EG"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_locale("ar-SA") == "ar_EG"


def test_region_tag_locale_override_is_resolved(monkeypatch):
    monkeypatch.setattr(locales, "_AR_REGION_AVAILABLE", {"ar-AE": "ar_AE"})
    assert resolve_locale("ar", locale_override="ar-AE") == "ar_AE"


def test_faker_locale_override_takes_precedence():
    # A direct Faker locale override still wins over a language region code.
    assert resolve_locale("ar-SA", locale_override="fr_FR") == "fr_FR"
