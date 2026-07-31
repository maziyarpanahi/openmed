"""Tests for the no-PHI production drift monitor.

Every aggregate below is synthetic and generated algorithmically; no real
production text or PHI is used. The suite proves three properties the retrain
flywheel depends on: drift records carry only aggregates/hashes/offsets (never
raw text), a synthetic drifted label distribution crosses the divergence
threshold with the correct dominant label, and divergence re-derives identically
offline from the same committed aggregates.
"""

from __future__ import annotations

import json
import socket

import pytest

from openmed.core.offline import OFFLINE_ENV_VAR
from openmed.eval.drift_monitor import (
    DEFAULT_DRIFT_THRESHOLD,
    DRIFT_FAMILIES,
    DRIFT_TRIGGER_SCHEMA_VERSION,
    LABEL_RATE_FAMILY,
    VERDICT_DRIFT,
    VERDICT_STABLE,
    DriftAggregateWindow,
    DriftPrivacyError,
    assert_no_raw_text,
    build_drift_window,
    compute_drift_report,
    load_drift_reference,
)

# A synthetic PHI-shaped string that must never survive into monitor state.
PHI_TEXT = "Patient Evelyn Quantum, MRN ZQ-7391, SSN 042-66-9001."
PHI_SUBSTRINGS = ("Evelyn Quantum", "ZQ-7391", "042-66-9001", "Patient")

_BASE_LABELS = {
    "ADDRESS": 1300,
    "AGE": 900,
    "DATE": 2600,
    "EMAIL": 700,
    "ID_NUM": 600,
    "MRN": 1500,
    "NAME": 3200,
    "ORGANIZATION": 1000,
    "PHONE": 1100,
    "SSN": 400,
}
_BASE_SCORES = [40, 55, 70, 110, 180, 320, 680, 1600, 4000, 6245]
_BASE_LENGTHS = {
    "len_0000_0064": 1800,
    "len_0064_0128": 3400,
    "len_0128_0256": 4200,
    "len_0256_0512": 2400,
    "len_0512_1024": 1000,
    "len_1024_plus": 500,
}
_BASE_SCRIPTS = {
    "Arabic": 900,
    "Cyrillic": 300,
    "Devanagari": 700,
    "Han": 1600,
    "Latin": 9800,
}


def _baseline_window(window_id: str = "baseline") -> DriftAggregateWindow:
    return build_drift_window(
        window_id,
        label_counts=dict(_BASE_LABELS),
        score_histogram=list(_BASE_SCORES),
        length_histogram=dict(_BASE_LENGTHS),
        script_histogram=dict(_BASE_SCRIPTS),
    )


def _stable_window() -> DriftAggregateWindow:
    # Same proportions at a different scale: divergence must stay ~0.
    return build_drift_window(
        "stable",
        label_counts={key: value * 2 for key, value in _BASE_LABELS.items()},
        score_histogram=[value * 2 for value in _BASE_SCORES],
        length_histogram={key: value * 2 for key, value in _BASE_LENGTHS.items()},
        script_histogram={key: value * 2 for key, value in _BASE_SCRIPTS.items()},
    )


def _label_drifted_window() -> DriftAggregateWindow:
    # NAME predictions spike hard; the dominant drifting label must be NAME.
    labels = dict(_BASE_LABELS)
    labels["NAME"] = 9000
    labels["DATE"] = 900
    labels["MRN"] = 500
    return build_drift_window(
        "observation-drifted",
        label_counts=labels,
        score_histogram=list(_BASE_SCORES),
        length_histogram=dict(_BASE_LENGTHS),
        script_histogram=dict(_BASE_SCRIPTS),
    )


def test_stable_window_stays_below_threshold():
    report = compute_drift_report(_baseline_window(), _stable_window())
    assert report.verdict == VERDICT_STABLE
    assert not report.drift_detected
    assert report.max_divergence < report.warning_threshold
    assert report.dominant_drifting_label is None


def test_label_drift_crosses_threshold_with_correct_dominant_label():
    report = compute_drift_report(_baseline_window(), _label_drifted_window())

    assert report.verdict == VERDICT_DRIFT
    assert report.drift_detected
    assert report.max_divergence >= DEFAULT_DRIFT_THRESHOLD
    assert report.dominant_family == LABEL_RATE_FAMILY
    assert report.dominant_drifting_label == "NAME"

    label_family = next(
        family for family in report.families if family.family == LABEL_RATE_FAMILY
    )
    assert label_family.verdict == VERDICT_DRIFT
    assert label_family.dominant_bucket == "NAME"


def test_divergence_is_reproducible_offline():
    reference = _baseline_window()
    observation = _label_drifted_window()

    first = compute_drift_report(
        reference, observation, generated_at="2026-07-24T00:00:00+00:00"
    )
    second = compute_drift_report(
        reference, observation, generated_at="2026-07-24T00:00:00+00:00"
    )
    assert first.to_dict() == second.to_dict()
    assert first.per_family_divergence == second.per_family_divergence


def test_all_families_are_scored():
    report = compute_drift_report(_baseline_window(), _label_drifted_window())
    assert tuple(family.family for family in report.families) == DRIFT_FAMILIES


def test_trigger_signal_is_directly_consumable():
    report = compute_drift_report(_baseline_window(), _label_drifted_window())
    signal = report.to_trigger_signal()
    payload = signal.to_dict()

    assert payload["schema_version"] == DRIFT_TRIGGER_SCHEMA_VERSION
    assert payload["drift_detected"] is True
    assert payload["verdict"] == VERDICT_DRIFT
    assert set(payload["per_family_divergence"]) == set(DRIFT_FAMILIES)
    assert payload["dominant_drifting_label"] == "NAME"
    assert payload["max_divergence"] == report.max_divergence
    # The boundary shape must round-trip through JSON for the trigger to read it.
    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# No-PHI structural guarantees
# ---------------------------------------------------------------------------


def _assert_no_phi_substrings(blob: str) -> None:
    leaked = [substring for substring in PHI_SUBSTRINGS if substring in blob]
    assert leaked == [], f"raw PHI leaked into drift record: {leaked!r}"


def test_serialized_record_contains_only_aggregates():
    report = compute_drift_report(_baseline_window(), _label_drifted_window())
    blob = report.to_json()
    _assert_no_phi_substrings(blob)
    # The serializer re-runs the structural guard; this must not raise.
    assert_no_raw_text(report.to_dict(), where="drift report")
    assert_no_raw_text(report.to_trigger_signal().to_dict(), where="trigger signal")


def test_window_rejects_forbidden_raw_text_key():
    payload = {
        "window_id": "leaky",
        "label_counts": {"NAME": 10},
        "text": PHI_TEXT,
    }
    with pytest.raises(DriftPrivacyError):
        DriftAggregateWindow.from_mapping(payload)


def test_window_rejects_nested_raw_span_field():
    payload = {
        "window_id": "leaky",
        "label_counts": {"NAME": 10},
        "metadata": {"spans": [PHI_TEXT]},
    }
    with pytest.raises(DriftPrivacyError):
        DriftAggregateWindow.from_mapping(payload)


def test_window_rejects_free_text_value_without_forbidden_key():
    payload = {
        "window_id": "leaky",
        "label_counts": {"NAME": 10},
        "feature_hashes": {"corpus": PHI_TEXT},
    }
    with pytest.raises(DriftPrivacyError):
        DriftAggregateWindow.from_mapping(payload)


def test_window_rejects_free_text_identifier():
    with pytest.raises(DriftPrivacyError):
        build_drift_window(
            "patient Evelyn Quantum note",
            label_counts={"NAME": 10},
            score_histogram=[10],
            length_histogram={"len_0000_0064": 10},
            script_histogram={"Latin": 10},
        )


def test_guard_detects_planted_raw_phi():
    with pytest.raises(DriftPrivacyError):
        assert_no_raw_text({"label_counts": {"note": PHI_TEXT}})
    with pytest.raises(DriftPrivacyError):
        assert_no_raw_text({"window_id": PHI_TEXT})


def test_guard_allows_safe_aggregate_payload():
    assert_no_raw_text(
        {
            "window_id": "reference-v2",
            "sample_count": 100,
            "label_counts": {"NAME": 40, "DATE": 60},
            "score_histogram": [1, 2, 3],
            "feature_hashes": {"corpus": "sha256:deadbeef"},
            "generated_at": "2026-07-24T00:00:00+00:00",
        }
    )


# ---------------------------------------------------------------------------
# Local-first / offline guarantees
# ---------------------------------------------------------------------------


def test_committed_reference_window_loads():
    reference = load_drift_reference()
    assert reference.window_id == "reference-v2-2026-07-24"
    assert reference.sample_count == sum(reference.label_counts.values())
    assert len(reference.score_histogram) == 10


def test_computation_opens_no_socket(monkeypatch):
    monkeypatch.delenv(OFFLINE_ENV_VAR, raising=False)
    attempts: list[tuple] = []

    def fail_socket(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("network egress attempted from drift monitor")

    monkeypatch.setattr(socket.socket, "connect", fail_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)

    report = compute_drift_report(
        _baseline_window(), _label_drifted_window(), local_only=True
    )
    assert report.drift_detected
    assert attempts == []
