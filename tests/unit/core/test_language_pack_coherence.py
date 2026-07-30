"""Tests for language-pack coherence validation and capability coverage (#1584).

The suite drives the coherence validator through a private
:class:`LanguagePackRegistry` (never the process-global one) so registrations
stay isolated, mirroring the #1586 contract tests. Every pack value is
synthetic.
"""

from __future__ import annotations

import json

import pytest

from openmed.core import (
    LanguagePack,
    LanguagePackCoherenceError,
    LanguagePackRegistry,
    check_language_pack_coherence,
    incoherent_packs,
    pack_coherence_report,
    require_language_pack_coherence,
)
from openmed.core.language_pack_catalog import SUPPORTED_LANGUAGES
from openmed.core.language_pack_coherence import (
    APPROXIMATED,
    CAPABILITY_SLOTS,
    FILLED,
    MISSING,
)


def _pack(code: str = "en", **overrides: object) -> LanguagePack:
    """Return a fully coherent synthetic pack unless overridden."""
    values: dict[str, object] = {
        "code": code,
        "scripts": ["Latin"],
        "default_model": "OpenMed/synthetic-pii",
        "segmenter_id": "unicode-sentence",
        "recognizers": ["regex", "model"],
        "surrogate_locale": "en_US",
        "national_id_providers": {"ssn": "en_US"},
        "policy_overrides": {"profile": "strict_no_leak"},
        "recall_floor_overrides": {"PERSON": 0.99},
    }
    values.update(overrides)
    return LanguagePack(**values)  # type: ignore[arg-type]


def _registry(*packs: LanguagePack) -> LanguagePackRegistry:
    registry = LanguagePackRegistry()
    for pack in packs:
        registry.register(pack)
    return registry


def _row_for(code: str, registry: LanguagePackRegistry) -> dict[str, object]:
    return {row["language"]: row for row in pack_coherence_report(registry=registry)}[
        code
    ]


# ---------------------------------------------------------------------------
# Coverage: the cross-script >= 5-filled-slots gate
# ---------------------------------------------------------------------------


def test_coherent_pack_reports_five_filled_slots_and_no_missing():
    registry = _registry(_pack("en"))
    row = _row_for("en", registry)

    assert row["coherent"] is True
    assert set(row["coverage"]["slots"]) == set(CAPABILITY_SLOTS)
    assert row["coverage"]["filled"] == 5
    assert row["coverage"]["missing"] == 0
    assert all(status == FILLED for status in row["coverage"]["slots"].values())


def test_approximate_surrogate_locale_is_explicit_and_never_silently_empty():
    # te -> en_IN is a documented approximation (a real Faker backend locale).
    registry = _registry(
        _pack(
            "te",
            scripts=["Telugu"],
            segmenter_id="pysbd",
            surrogate_locale="en_IN",
            national_id_providers={"aadhaar": "en_IN"},
            policy_overrides={"profile": "balanced"},
        )
    )
    row = _row_for("te", registry)

    assert row["coherent"] is True
    assert row["coverage"]["slots"]["surrogate_locale"] == APPROXIMATED
    assert row["surrogate_locale"]["status"] == APPROXIMATED
    assert row["coverage"]["approximated"] == 1
    assert row["coverage"]["missing"] == 0


def test_approximate_language_with_invalid_locale_fails_loudly():
    registry = _registry(
        _pack(
            "te",
            scripts=["Telugu"],
            surrogate_locale="not_A_REAL_LOCALE",
            national_id_providers={},
        )
    )
    row = _row_for("te", registry)

    assert row["coherent"] is False
    assert row["surrogate_locale"]["status"] == MISSING
    assert any("surrogate locale" in issue for issue in row["issues"])


def test_real_override_for_approximate_language_is_filled():
    registry = _registry(
        _pack(
            "te",
            scripts=["Telugu"],
            surrogate_locale="en_US",
            national_id_providers={},
        )
    )
    row = _row_for("te", registry)

    assert row["coherent"] is True
    assert row["surrogate_locale"]["status"] == FILLED


def test_policy_slot_missing_is_not_a_coherence_failure():
    registry = _registry(_pack("en", policy_overrides={}, recall_floor_overrides={}))
    row = _row_for("en", registry)

    assert row["coverage"]["slots"]["policy"] == MISSING
    assert row["coherent"] is True  # an absent optional slot is not incoherent
    assert row["national_id"]["status"] == FILLED


# ---------------------------------------------------------------------------
# Fail-loud coherence checks
# ---------------------------------------------------------------------------


def test_unresolvable_segmenter_fails_loudly():
    registry = _registry(_pack("en", segmenter_id="no-such-segmenter"))
    row = _row_for("en", registry)

    assert row["coherent"] is False
    assert row["coverage"]["slots"]["segmenter"] == MISSING
    assert any("segmenter" in issue for issue in row["issues"])


def test_national_id_provider_with_wrong_registry_dispatch_fails_loudly():
    # PESEL is not the registry's dispatch method for en_US; the mis-wiring is
    # caught before any surrogate is generated.
    registry = _registry(_pack("en", national_id_providers={"pesel": "en_US"}))
    row = _row_for("en", registry)

    assert row["coherent"] is False
    assert row["national_id"]["status"] == MISSING
    assert any("disagrees with registry dispatch" in issue for issue in row["issues"])


def test_national_id_provider_that_does_not_round_trip_fails_loudly():
    # Dispatch AGREES (_LOCALE_ID_METHODS["en_US"] == "ssn"), but a US-SSN
    # surrogate cannot pass the German Steuer-ID validator registered for 'de',
    # so the round-trip itself fails.
    registry = _registry(
        _pack(
            "de",
            surrogate_locale="de_DE",
            national_id_providers={"ssn": "en_US"},
            recall_floor_overrides={"PERSON": 0.9},
        )
    )
    row = _row_for("de", registry)

    assert row["coherent"] is False
    assert row["national_id"]["status"] == MISSING
    assert any("do not round-trip" in issue for issue in row["issues"])


def test_valid_national_id_provider_round_trips():
    registry = _registry(_pack("en", national_id_providers={"ssn": "en_US"}))
    row = _row_for("en", registry)

    assert row["coherent"] is True
    assert row["national_id"]["status"] == FILLED


def test_unknown_policy_profile_fails_loudly():
    registry = _registry(_pack("en", policy_overrides={"profile": "made_up"}))
    row = _row_for("en", registry)

    assert row["coherent"] is False
    assert any("policy profile" in issue for issue in row["issues"])


def test_unknown_policy_override_key_fails_loudly():
    registry = _registry(_pack("en", policy_overrides={"garbage": "value"}))
    row = _row_for("en", registry)

    assert row["coherent"] is False
    assert row["policy"]["status"] == MISSING
    assert any("unsupported policy override keys" in issue for issue in row["issues"])


def test_non_canonical_recall_floor_label_fails_loudly():
    registry = _registry(_pack("en", recall_floor_overrides={"NOT_A_LABEL": 0.5}))
    row = _row_for("en", registry)

    assert row["coherent"] is False
    assert any("non-canonical labels" in issue for issue in row["issues"])


# ---------------------------------------------------------------------------
# Check entry points
# ---------------------------------------------------------------------------


def test_check_returns_zero_when_all_coherent():
    registry = _registry(
        _pack("en"),
        _pack("fr", surrogate_locale="fr_FR", national_id_providers={"ssn": "fr_FR"}),
    )
    assert check_language_pack_coherence(registry=registry) == 0
    assert incoherent_packs(registry=registry) == []
    require_language_pack_coherence(registry=registry)  # does not raise


def test_check_returns_nonzero_and_require_raises_on_incoherence():
    registry = _registry(
        _pack("en"),
        _pack("xx", scripts=["Synthetic"], segmenter_id="broken"),
    )
    assert check_language_pack_coherence(registry=registry) == 1

    with pytest.raises(LanguagePackCoherenceError, match="xx"):
        require_language_pack_coherence(registry=registry)

    failures = incoherent_packs(registry=registry)
    assert [row["language"] for row in failures] == ["xx"]


# ---------------------------------------------------------------------------
# Report contract: deterministic + JSON serializable
# ---------------------------------------------------------------------------


def test_report_is_sorted_by_code_and_deterministic():
    registry = LanguagePackRegistry()
    registry.register(
        _pack("fr", surrogate_locale="fr_FR", national_id_providers={"ssn": "fr_FR"})
    )
    registry.register(_pack("en"))
    registry.register(
        _pack(
            "de",
            surrogate_locale="de_DE",
            national_id_providers={"german_steuer_id": "de_DE"},
        )
    )

    first = pack_coherence_report(registry=registry)
    second = pack_coherence_report(registry=registry)

    assert [row["language"] for row in first] == ["de", "en", "fr"]
    assert first == second


def test_report_is_json_serializable():
    registry = _registry(_pack("en"), _pack("xx", segmenter_id="broken"))
    # Must not raise; the status/coverage tooling serializes this.
    json.dumps(pack_coherence_report(registry=registry))


def test_empty_registry_reports_no_rows_and_is_coherent():
    registry = LanguagePackRegistry()
    assert pack_coherence_report(registry=registry) == []
    assert check_language_pack_coherence(registry=registry) == 0


def test_builtin_language_pack_catalog_is_coherent():
    rows = pack_coherence_report()

    assert len(rows) == len(SUPPORTED_LANGUAGES)
    assert [row["language"] for row in rows] == sorted(row["language"] for row in rows)
    assert [row for row in rows if not row["coherent"]] == []
