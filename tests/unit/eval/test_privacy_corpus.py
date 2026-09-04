"""Focused tests for the synthetic privacy regression corpus manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.eval.privacy_corpus import (
    PRIVACY_CORPUS_SCHEMA_VERSION,
    PrivacyCase,
    PrivacyCorpusManifest,
    PrivacyFindingExpectation,
    PrivacyPolicyProfile,
    build_privacy_corpus_manifest,
    compute_privacy_corpus_manifest_hash,
    compute_privacy_fixture_hash,
    default_privacy_corpus_manifest,
    load_privacy_corpus_manifest,
    make_privacy_case,
    privacy_corpus_coverage,
    validate_privacy_corpus_manifest,
    write_privacy_corpus_manifest,
)


def _profile() -> PrivacyPolicyProfile:
    return PrivacyPolicyProfile(
        profile_id="strict_redaction",
        required_categories=("direct_identifier",),
        required_severities=("critical",),
    )


def _case(fixture: str = "synthetic-fixture-a") -> PrivacyCase:
    return make_privacy_case(
        "direct_identifier_case",
        fixture,
        category="direct_identifier",
        policy_profile_id="strict_redaction",
        severity="critical",
        expected_findings=(
            PrivacyFindingExpectation(
                finding_id="critical_leakage",
                severity="critical",
                expected_count=0,
                critical_leakage=True,
            ),
        ),
    )


def test_default_manifest_is_deterministic_and_complete() -> None:
    first = default_privacy_corpus_manifest()
    second = load_privacy_corpus_manifest()

    assert first == second
    assert first.schema_version == PRIVACY_CORPUS_SCHEMA_VERSION
    assert first.synthetic_only is True
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first.manifest_hash)
    report = validate_privacy_corpus_manifest(first)
    assert report.valid is True
    assert report.missing_categories == ()
    assert report.missing_severities == ()
    assert report.expected_critical_leakage == 0

    serialized = json.dumps(first.to_dict(), sort_keys=True)
    assert "seed-a" not in serialized
    assert "seed-b" not in serialized
    assert all("text" not in case.to_dict() for case in first.cases)


def test_builder_hashes_fixture_without_persisting_source_text() -> None:
    fixture = "synthetic-private-value-001"
    manifest = build_privacy_corpus_manifest(
        [_case(fixture)],
        [_profile()],
    )
    serialized = json.dumps(manifest.to_dict(), sort_keys=True)

    assert fixture not in serialized
    assert manifest.cases[0].fixture_length == len(fixture)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", manifest.cases[0].fixture_hash)
    assert compute_privacy_corpus_manifest_hash(manifest) == manifest.manifest_hash
    assert compute_privacy_fixture_hash(fixture) == manifest.cases[0].fixture_hash


def test_manifest_hash_is_independent_of_input_order() -> None:
    first = build_privacy_corpus_manifest(
        [
            _case("synthetic-a"),
            make_privacy_case(
                "direct_identifier_case_2",
                "synthetic-b",
                category="direct_identifier",
                policy_profile_id="strict_redaction",
                severity="critical",
                expected_findings=(
                    PrivacyFindingExpectation(
                        finding_id="critical_leakage",
                        severity="critical",
                        expected_count=0,
                        critical_leakage=True,
                    ),
                ),
            ),
        ],
        [_profile()],
    )
    second = build_privacy_corpus_manifest(
        list(reversed(first.cases)),
        list(reversed(first.policy_profiles)),
    )

    assert first == second


def test_fixture_changes_produce_a_new_content_hash() -> None:
    assert compute_privacy_fixture_hash("synthetic-a") != compute_privacy_fixture_hash(
        "synthetic-b"
    )


def test_persisted_manifest_rejects_raw_fixture_fields_without_echoing_values() -> None:
    sensitive_sentinel = "synthetic-value-must-not-be-echoed"
    payload = default_privacy_corpus_manifest().to_dict()
    payload["cases"][0]["text"] = sensitive_sentinel

    with pytest.raises(ValueError, match="unsupported fields") as error:
        PrivacyCorpusManifest.from_mapping(payload)

    assert sensitive_sentinel not in str(error.value)


def test_manifest_round_trip_is_deterministic(tmp_path: Path) -> None:
    manifest = default_privacy_corpus_manifest()
    path = write_privacy_corpus_manifest(tmp_path / "privacy-corpus.json", manifest)

    loaded = load_privacy_corpus_manifest(path)

    assert loaded == manifest
    assert json.loads(path.read_text(encoding="utf-8")) == manifest.to_dict()


def test_coverage_report_exposes_profile_gaps_without_fixture_values() -> None:
    case = _case()
    profile = PrivacyPolicyProfile(
        profile_id="strict_redaction",
        required_categories=("direct_identifier", "quasi_identifier"),
        required_severities=("critical", "high"),
    )
    manifest = PrivacyCorpusManifest(
        manifest_id="incomplete",
        cases=(case,),
        policy_profiles=(profile,),
        required_categories=("direct_identifier", "quasi_identifier"),
        manifest_hash="sha256:" + "0" * 64,
    )

    report = privacy_corpus_coverage(manifest)

    assert report.valid is False
    assert report.missing_categories == ("quasi_identifier",)
    assert report.profile_category_gaps == ("strict_redaction",)
    assert "synthetic-fixture-a" not in json.dumps(report.to_dict())


def test_non_finite_fixture_values_fail_without_echoing_input() -> None:
    sensitive_sentinel = "synthetic-not-in-error"

    with pytest.raises(ValueError, match="finite JSON") as error:
        compute_privacy_fixture_hash(
            {"text": sensitive_sentinel, "score": float("nan")}
        )

    assert sensitive_sentinel not in str(error.value)
