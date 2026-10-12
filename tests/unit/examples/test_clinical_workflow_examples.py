"""Offline safety and negative controls for the five workflow examples."""

import builtins
import importlib
import json
import re
import socket
import urllib.request

import pytest

CASES = (
    ("prior_auth_completeness", "complete", "schema_version_mismatch"),
    ("chart_abstraction_evidence", "finalized", "evidence_not_finalizable"),
    ("cohort_explanations", "eligible", "time_window_mismatch"),
    ("quality_measure_evidence", "verified", "signature_mismatch"),
    ("trial_eligibility_review", "agreement", "unknown_criterion"),
)


@pytest.mark.parametrize(("name", "success", "reason"), CASES)
def test_examples_are_deterministic_in_memory_and_value_free(
    name, success, reason, monkeypatch, capsys
):
    example = importlib.import_module(f"examples.{name}")

    def unexpected_io(*args, **kwargs):
        raise AssertionError("workflow example attempted external IO")

    monkeypatch.setattr(builtins, "open", unexpected_io)
    monkeypatch.setattr(socket, "socket", unexpected_io)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_io)
    first = example.main()
    output = capsys.readouterr().out
    assert example.main() == first
    assert capsys.readouterr().out == output
    assert json.loads(output) == first
    assert first["workflow_id"] == name
    assert first["passed"]["code"] == success
    assert first["fail_closed"]["code"] == reason

    # A closed output vocabulary catches new plaintext, paths, dates, keys,
    # scores, counts or offsets accidentally copied from an evidence packet.
    allowed = {
        name,
        success,
        reason,
        "registry.synthetic_field",
        "review_not_approved",
        "clinical.inclusion",
        "clinical.exclusion",
        "met",
        "not_met",
        "outcome_conflict",
    }

    def check_leaves(value):
        if isinstance(value, dict):
            for item in value.values():
                check_leaves(item)
        elif isinstance(value, list):
            for item in value:
                check_leaves(item)
        else:
            assert isinstance(value, str)
            assert value in allowed or re.fullmatch(r"sha256:[0-9a-f]{64}", value)

    check_leaves(first)
    assert "synthetic-offline-example-key" not in output
    assert "2026-01-01" not in output
    if name == "chart_abstraction_evidence":
        assert first["fail_closed"]["issues"] == [
            {"code": "review_not_approved", "field_id": "registry.synthetic_field"}
        ]
    if name == "cohort_explanations":
        assert first["passed"]["criteria"] == [
            {"criterion_id": "clinical.exclusion", "code": "not_met"},
            {"criterion_id": "clinical.inclusion", "code": "met"},
        ]
    if name == "trial_eligibility_review":
        assert first["review_required"]["causes"] == ["outcome_conflict"]


@pytest.mark.parametrize(("name", "success", "reason"), CASES)
def test_examples_detect_a_disabled_rejection_gate(name, success, reason, monkeypatch):
    example = importlib.import_module(f"examples.{name}")
    if name == "chart_abstraction_evidence":
        target, attribute = example.ChartAbstractionEvidence, "finalize"
    elif name == "quality_measure_evidence":
        target, attribute = example.QualityMeasureEvidencePacket, "verify"
    else:
        target = example
        attribute = {
            "prior_auth_completeness": "score_prior_authorization_packet",
            "cohort_explanations": "explain_criterion_membership",
            "trial_eligibility_review": "build_trial_eligibility_review_packet",
        }[name]
    original = getattr(target, attribute)
    calls = 0
    result = None

    def bypass_rejection(*args, **kwargs):
        nonlocal calls, result
        calls += 1
        # Trial comparison has two valid builds before its rejection case.
        if calls <= (2 if name == "trial_eligibility_review" else 1):
            result = original(*args, **kwargs)
        return result

    monkeypatch.setattr(target, attribute, bypass_rejection)
    with pytest.raises(AssertionError, match="not_rejected"):
        example.run_example()
