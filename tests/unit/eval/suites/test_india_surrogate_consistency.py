"""Tests for the India cross-document surrogate-consistency gate."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Iterator

import pytest

from openmed.eval.datasets import load_india_clinical_phi_corpus
from openmed.eval.datasets.clinical_phi import (
    IndiaClinicalPHIIdentity,
    IndiaClinicalPHIIdentityAlias,
)
from openmed.eval.harness import BenchmarkFixture, run_benchmark
from openmed.eval.suites import (
    DEFAULT_SUITES,
    IDENTIFIER_LINKAGE_LANGUAGE_SCOPED,
    IDENTITY_SPLIT_ACROSS_SCRIPTS,
    INDIA_CLINICAL_PHI_LEAKAGE,
    INDIA_SURROGATE_CONSISTENCY,
    LANGUAGE_SCOPED_IDENTIFIER_KEYS,
    LINKED_IDENTIFIER_TYPES,
    SURROGATE_NOT_REUSED,
    TRANSLITERATION_AWARE_NAME_MATCHING,
    IndiaSurrogateConsistencyResult,
    assert_india_surrogate_consistency_gate,
    evaluate_india_surrogate_consistency,
    india_surrogate_consistency_metadata,
    load_suite_fixtures,
    run_india_clinical_suite_report,
    run_india_surrogate_consistency_gate,
    suite_metadata,
    validate_suite_name,
)

# A fictional, deliberately non-matching persona, mirroring the
# ``negative_surfaces`` convention of the committed indic_name_variants
# fixture. It is not a real person and is used only to split an identity.
UNRELATED_SYNTHETIC_PERSONA = "Rohan Verma"


def _alias_surfaces() -> set[str]:
    corpus = load_india_clinical_phi_corpus()
    return {
        alias.text
        for identity in corpus.manifest.cross_document_identities
        for alias in identity.aliases
    }


def _direct_identifier_surfaces() -> set[str]:
    corpus = load_india_clinical_phi_corpus()
    return {
        span.text
        for record in corpus.records
        for span in record.gold_spans
        if span.metadata.get("direct_identifier") is True
    }


def _json_strings(payload: Any) -> Iterator[str]:
    """Yield every key and every scalar of a serialized payload."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from _json_strings(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _json_strings(item)
    else:
        yield str(payload)


def _corpus_with_split_identity():
    """Return the corpus with one extra, unrelated alias in the group."""

    corpus = load_india_clinical_phi_corpus()
    identity = corpus.manifest.cross_document_identities[0]
    broken = IndiaClinicalPHIIdentity(
        group_id=identity.group_id,
        aliases=identity.aliases
        + (
            IndiaClinicalPHIIdentityAlias(
                document_id=identity.aliases[0].document_id,
                text=UNRELATED_SYNTHETIC_PERSONA,
                script="Latin",
            ),
        ),
    )
    manifest = replace(corpus.manifest, cross_document_identities=(broken,))
    return replace(corpus, manifest=manifest)


def test_shipped_corpus_yields_one_identity_across_scripts() -> None:
    result = run_india_surrogate_consistency_gate()

    assert isinstance(result, IndiaSurrogateConsistencyResult)
    assert result.passed is True
    assert result.failures == ()
    assert result.group_count == 1
    assert result.alias_count == 3
    assert result.scripts == ("Devanagari", "Latin", "Tamil")

    verdict = result.identity_verdicts[0]
    assert verdict.group_id == "india-person-001"
    # The core acceptance criterion: three aliases, three documents, three
    # scripts, exactly one surrogate identity.
    assert verdict.identity_count == 1
    assert verdict.surrogate_count == 1
    assert verdict.document_count == 3
    assert verdict.script_mismatch_count == 0
    assert verdict.alias_leak_count == 0
    assert verdict.passed is True


def test_repeated_identifiers_share_one_surrogate_and_stay_valid() -> None:
    result = run_india_surrogate_consistency_gate()

    by_type = {v.identifier_type: v for v in result.identifier_verdicts}
    assert set(by_type) == set(LINKED_IDENTIFIER_TYPES)

    # Aadhaar and ABHA each appear in two documents in the shipped corpus.
    assert by_type["aadhaar"].occurrence_count == 2
    assert by_type["aadhaar"].repeated_source_count == 1
    assert by_type["abha"].occurrence_count == 2
    assert by_type["abha"].repeated_source_count == 1

    for verdict in result.identifier_verdicts:
        # No two distinct sources may share a surrogate, and linkage must hold
        # at the granularity the vault guarantees, (value, language).
        assert verdict.collision_count == 0
        assert verdict.unstable_surrogate_count == 0
        # Every generated surrogate satisfies its own shape/checksum validator.
        assert verdict.invalid_surrogate_count == 0
        assert verdict.passed is True

    # ABHA repeats within a single language, so it links to one surrogate.
    assert by_type["abha"].distinct_surrogate_count == 1
    # Aadhaar repeats across Hindi and Tamil, which the vault keys separately.
    assert by_type["aadhaar"].cross_language_split_count == 1


def test_gate_is_deterministic_and_negatives_do_not_collide() -> None:
    first = run_india_surrogate_consistency_gate()
    second = run_india_surrogate_consistency_gate()

    assert first.deterministic is True
    assert first.to_dict() == second.to_dict()
    assert first.negative_identifier_count > 0
    assert first.negative_collision_count == 0
    assert first.negative_invalid_count == 0


def test_transliteration_aware_divergence_is_recorded_but_not_gated() -> None:
    result = run_india_surrogate_consistency_gate()

    assert result.passed is True
    assert result.mode == "default"
    divergence = next(
        d
        for d in result.known_divergences
        if d.mode == TRANSLITERATION_AWARE_NAME_MATCHING
    )
    assert divergence.mode == TRANSLITERATION_AWARE_NAME_MATCHING
    assert divergence.group_id == "india-person-001"
    assert divergence.reason == IDENTITY_SPLIT_ACROSS_SCRIPTS
    assert divergence.expected_identity_count == 1
    assert divergence.observed_identity_count == 3
    # Recorded for transparency; it must never contribute to the verdict.
    assert divergence.gated is False


def test_transliteration_aware_path_still_splits_the_corpus_identity() -> None:
    """Guard the recorded divergence so it cannot rot silently.

    The opt-in transliteration-aware matcher folds the Devanagari and Tamil
    aliases to different Latin keys than the Latin alias, so the same person
    yields three identities. If that is ever fixed in
    ``openmed.core.indic_name_match``, this test fails loudly and forces the
    recorded ``known_divergences`` entry to be revisited and removed.
    """

    result = evaluate_india_surrogate_consistency(
        mode=TRANSLITERATION_AWARE_NAME_MATCHING,
    )

    assert result.mode == TRANSLITERATION_AWARE_NAME_MATCHING
    assert result.identity_verdicts[0].identity_count == 3
    assert result.passed is False
    assert any(
        failure.endswith(IDENTITY_SPLIT_ACROSS_SCRIPTS) for failure in result.failures
    )
    # Evaluated standalone, the alternate path does not recurse into itself,
    # so it records no transliteration divergence of its own.
    assert not [
        d
        for d in result.known_divergences
        if d.mode == TRANSLITERATION_AWARE_NAME_MATCHING
    ]


def test_gate_fails_closed_when_an_alias_maps_to_another_identity() -> None:
    result = evaluate_india_surrogate_consistency(_corpus_with_split_identity())

    assert result.passed is False
    verdict = result.identity_verdicts[0]
    assert verdict.alias_count == 4
    assert verdict.identity_count == 2
    assert verdict.surrogate_count == 2
    assert verdict.passed is False
    assert f"india-person-001:{IDENTITY_SPLIT_ACROSS_SCRIPTS}" in result.failures


def test_gate_fails_closed_when_the_product_stops_rendering_indic_scripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real rendering regression, not a relabelled fixture, must fail.

    Forcing the vault's Indic renderer to emit its Latin input models a
    regression in which Devanagari and Tamil documents receive Latin-script
    surrogates.
    """

    monkeypatch.setattr(
        "openmed.core.surrogate_vault.render_indian_name",
        lambda identity, script: identity,
    )

    result = evaluate_india_surrogate_consistency()

    verdict = result.identity_verdicts[0]
    # The identity still collapses; only the rendered script is wrong.
    assert verdict.identity_count == 1
    assert verdict.script_mismatch_count == 2
    assert verdict.passed is False
    assert result.passed is False


def test_one_surrogate_claim_fails_when_the_store_drops_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store whose writes vanish must not report one surrogate.

    This is the regression that a bare ``len(set(stored))`` cannot see: every
    lookup misses, so each alias is minted a fresh unrelated name while the
    aliases still share a single key.
    """

    from openmed.core.surrogate_vault import InMemorySurrogateStore

    monkeypatch.setattr(
        InMemorySurrogateStore,
        "set",
        lambda self, *args, **kwargs: None,
    )

    result = evaluate_india_surrogate_consistency()

    verdict = result.identity_verdicts[0]
    assert verdict.identity_count == 1, "key equality is unaffected by the fault"
    # Both independent detectors fire: nothing was stored, and re-asking the
    # vault returned a different surface.
    assert verdict.missing_surrogate_count == 3
    assert verdict.surrogate_count == 0
    # The repeat probe is stable here because the name factory is itself
    # deterministic per source; the missing-store detector is what bites.
    assert verdict.passed is False
    assert result.passed is False
    assert f"india-person-001:{SURROGATE_NOT_REUSED}" in result.failures


def test_identifier_linkage_fails_when_the_store_drops_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated identifiers must not silently re-mint under a broken store."""

    from openmed.core.surrogate_vault import InMemorySurrogateStore

    monkeypatch.setattr(
        InMemorySurrogateStore,
        "set",
        lambda self, *args, **kwargs: None,
    )

    result = evaluate_india_surrogate_consistency()

    by_type = {v.identifier_type: v for v in result.identifier_verdicts}
    # ABHA repeats within one language, so its linkage is gated.
    assert by_type["abha"].unstable_surrogate_count == 1
    assert by_type["abha"].passed is False
    assert result.passed is False


def test_language_scoped_identifier_keys_are_recorded_not_gated() -> None:
    """Guard the second recorded divergence so it cannot rot silently.

    The vault normalizes the key language for names but keeps the document
    language for structured identifiers, so one Aadhaar used in a Hindi and a
    Tamil note yields two surrogates. If that is ever unified, this test fails
    loudly and forces the divergence record to be revisited.
    """

    result = run_india_surrogate_consistency_gate()

    by_type = {v.identifier_type: v for v in result.identifier_verdicts}
    assert by_type["aadhaar"].cross_language_split_count == 1
    # Recorded, and deliberately excluded from the verdict.
    assert by_type["aadhaar"].passed is True
    assert result.passed is True

    divergence = next(
        d for d in result.known_divergences if d.mode == LANGUAGE_SCOPED_IDENTIFIER_KEYS
    )
    assert divergence.group_id == "aadhaar"
    assert divergence.reason == IDENTIFIER_LINKAGE_LANGUAGE_SCOPED
    assert divergence.observed_identity_count == 2
    assert divergence.gated is False


def test_a_surrogate_that_is_only_a_fold_of_the_alias_is_a_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical-fold leakage must be caught, not just exact substrings."""

    from openmed.eval.suites import india_surrogate_consistency as module

    # The transliterated fold of the real name is trivially re-identifiable.
    assert module._leaks_alias("Arav Sarma", "Aarav Sharma") is True
    assert module._leaks_alias("Aarav Sharma", "Aarav Sharma") is True
    # An unrelated synthetic surface is not a leak.
    assert module._leaks_alias(UNRELATED_SYNTHETIC_PERSONA, "Aarav Sharma") is False


@pytest.mark.parametrize("mode", ["transliteration-aware", "totally-bogus", ""])
def test_unknown_matching_modes_are_rejected(mode: str) -> None:
    """An unvalidated mode would stamp a green verdict with a bad label."""

    with pytest.raises(ValueError, match="unknown_matching_mode"):
        run_india_surrogate_consistency_gate(mode=mode)


def test_negative_identifier_validator_regression_fails_rather_than_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider regression must produce a verdict, not an exception."""

    from openmed.eval.suites import india_surrogate_consistency as module

    monkeypatch.setitem(module._IDENTIFIER_VALIDATORS, "pan", lambda value: False)

    result = evaluate_india_surrogate_consistency()

    assert result.negative_invalid_count > 0
    assert result.passed is False


def test_assert_helper_raises_without_leaking_raw_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = _corpus_with_split_identity()
    monkeypatch.setattr(
        "openmed.eval.datasets.clinical_phi.load_india_clinical_phi_corpus",
        lambda *args, **kwargs: broken,
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_india_surrogate_consistency_gate()

    message = str(excinfo.value)
    assert "india-person-001" in message
    assert IDENTITY_SPLIT_ACROSS_SCRIPTS in message
    for surface in _alias_surfaces() | {UNRELATED_SYNTHETIC_PERSONA}:
        assert surface not in message


def test_assert_helper_returns_passing_result_on_clean_corpus() -> None:
    result = assert_india_surrogate_consistency_gate()

    assert result.passed is True


def test_combined_report_exposes_all_three_verdicts() -> None:
    report = run_india_clinical_suite_report()
    payload = report.to_dict()

    assert payload["suite"] == "india_clinical"
    assert payload["passed"] is True
    assert payload["policy"] == "india_dpdp_act"
    # One report, three verdicts.
    assert payload["policy_coverage"]["passed"] is True
    assert payload["policy_coverage"]["per_label"]
    assert payload["residual_leakage"]["passed"] is True
    assert payload["residual_leakage"]["residual_leak_count"] == 0
    assert payload["surrogate_consistency"]["passed"] is True

    boundary = payload["safety_boundary"]
    assert boundary["synthetic_only"] is True
    assert boundary["contains_real_phi"] is False
    assert boundary["contains_dua_data"] is False
    assert boundary["assist_only_non_decisional"] is True
    assert boundary["excludes_real_hospital_data"] is True
    assert boundary["execution"] == "local-offline-deterministic"


def test_combined_report_has_no_raw_alias_surfaces() -> None:
    payload = run_india_clinical_suite_report().to_dict()
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # Both keys and values are inspected: content that rides through as a dict
    # key is invisible to a values-only scan.
    fragments = list(_json_strings(payload))

    for surface in _alias_surfaces() | _direct_identifier_surfaces():
        assert surface not in serialized
        assert not any(surface in fragment for fragment in fragments)


def test_registered_in_default_suites_and_discoverable() -> None:
    assert INDIA_SURROGATE_CONSISTENCY in DEFAULT_SUITES
    assert INDIA_CLINICAL_PHI_LEAKAGE in DEFAULT_SUITES
    assert validate_suite_name(INDIA_SURROGATE_CONSISTENCY) == (
        INDIA_SURROGATE_CONSISTENCY
    )
    assert validate_suite_name(INDIA_CLINICAL_PHI_LEAKAGE) == (
        INDIA_CLINICAL_PHI_LEAKAGE
    )


@pytest.mark.parametrize(
    "suite",
    [INDIA_SURROGATE_CONSISTENCY, INDIA_CLINICAL_PHI_LEAKAGE],
)
def test_standard_harness_runs_the_suite_without_special_wiring(suite: str) -> None:
    fixtures = load_suite_fixtures(suite)

    assert len(fixtures) == 3
    assert all(isinstance(fixture, BenchmarkFixture) for fixture in fixtures)

    def runner(fixture: BenchmarkFixture, model_name: str, device: str):
        return [
            {
                "start": span.start,
                "end": span.end,
                "label": span.label,
                "text": fixture.text[span.start : span.end],
            }
            for span in fixture.gold_spans
        ]

    report = run_benchmark(
        fixtures,
        suite=suite,
        model_name="stub",
        device="cpu",
        runner=runner,
        metadata=suite_metadata(suite),
    )

    assert report.suite == suite
    assert report.fixture_count == 3
    assert report.metrics["exact_span_f1"]["f1"] == pytest.approx(1.0)


def test_metadata_is_offline_and_raw_text_free() -> None:
    metadata = india_surrogate_consistency_metadata()

    assert metadata["suite"] == INDIA_SURROGATE_CONSISTENCY
    assert metadata["synthetic"] is True
    assert metadata["required_identity_count_per_group"] == 1
    assert metadata["required_negative_collisions"] == 0
    assert metadata["gated_mode"] == "default"
    assert metadata["recorded_not_gated_modes"] == [TRANSLITERATION_AWARE_NAME_MATCHING]
    assert sorted(metadata["linked_identifier_types"]) == sorted(
        LINKED_IDENTIFIER_TYPES
    )

    serialized = json.dumps(metadata, ensure_ascii=False)
    for surface in _alias_surfaces() | _direct_identifier_surfaces():
        assert surface not in serialized


def test_metadata_hashes_operator_controlled_fixture_paths() -> None:
    metadata = india_surrogate_consistency_metadata(
        fixture_path="/home/alice/private-synthetic-india.jsonl"
    )

    assert metadata["fixture_path_hash"].startswith("sha256:")
    serialized = json.dumps(metadata, ensure_ascii=False)
    assert "alice" not in serialized
    assert "private-synthetic-india.jsonl" not in serialized
