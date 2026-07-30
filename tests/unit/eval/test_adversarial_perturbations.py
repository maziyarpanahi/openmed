"""Tests for the adversarial Unicode-evasion perturbation suite."""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys

import pytest

from openmed.eval.adversarial_perturbations import (
    DEFAULT_ADVERSARIAL_PERTURBATIONS,
    AdversarialPerturbationGateError,
    adversarial_perturbation_report,
    bidi_control_wrapping_perturbation,
    combining_mark_injection_perturbation,
    homoglyph_substitution_perturbation,
    zero_width_injection_perturbation,
)
from openmed.eval.harness import BenchmarkFixture

_ALL_OPERATORS = [
    homoglyph_substitution_perturbation(probability=1.0),
    zero_width_injection_perturbation(probability=1.0),
    combining_mark_injection_perturbation(probability=1.0),
    bidi_control_wrapping_perturbation(),
]


def _fixture() -> BenchmarkFixture:
    return BenchmarkFixture.from_mapping(
        {
            "id": "note-1",
            "text": "Patient John Doe has MRN A1001 today.",
            "language": "en",
            "gold_spans": [
                {"start": 8, "end": 16, "label": "PERSON"},
                {"start": 25, "end": 30, "label": "ID_NUM"},
            ],
        }
    )


def _naive_runner(fixture, model_name, device):
    """Detector that only matches the pristine (unperturbed) surfaces."""
    assert device == "cpu"
    out = []
    for value, label in (("John Doe", "PERSON"), ("A1001", "ID_NUM")):
        start = fixture.text.find(value)
        if start >= 0:
            out.append({"start": start, "end": start + len(value), "label": label})
    return out


def _oracle_runner(fixture, model_name, device):
    """Detector that recovers every gold span regardless of perturbation."""
    return [
        {"start": span.start, "end": span.end, "label": span.label}
        for span in fixture.gold_spans
    ]


@pytest.mark.parametrize("operator", _ALL_OPERATORS)
def test_operators_are_seeded_and_reproject_spans_without_drift(operator):
    fixture = _fixture()

    first = operator(fixture, random.Random(17))
    second = operator(fixture, random.Random(17))

    assert first.text == second.text
    assert first.gold_spans == second.gold_spans
    assert first.text != fixture.text
    for span in first.gold_spans:
        assert span.text == first.text[span.start : span.end]
        assert span.end > span.start


@pytest.mark.parametrize("operator", _ALL_OPERATORS)
def test_operators_actually_perturb_each_gold_span(operator):
    fixture = _fixture()

    perturbed = operator(fixture, random.Random(3))

    # Every gold identifier surface is corrupted at the byte level.
    for clean_span, dirty_span in zip(fixture.gold_spans, perturbed.gold_spans):
        assert dirty_span.text != clean_span.text


@pytest.mark.parametrize(
    "name",
    [
        "homoglyph_substitution",
        "zero_width_injection",
        "combining_mark_injection",
        "bidi_control_wrapping",
    ],
)
def test_operators_are_byte_identical_across_processes(name):
    fixture = _fixture()
    operator = next(p for p in DEFAULT_ADVERSARIAL_PERTURBATIONS if p.name == name)
    in_process = operator(fixture, random.Random(99)).text

    def _subprocess_text(hashseed: str) -> str:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hashseed
        snippet = (
            "import random, json;"
            "from openmed.eval.harness import BenchmarkFixture;"
            "from openmed.eval.adversarial_perturbations import"
            " DEFAULT_ADVERSARIAL_PERTURBATIONS as P;"
            "fx=BenchmarkFixture.from_mapping("
            "{'id':'note-1','text':'Patient John Doe has MRN A1001 today.',"
            "'language':'en','gold_spans':["
            "{'start':8,'end':16,'label':'PERSON'},"
            "{'start':25,'end':30,'label':'ID_NUM'}]});"
            f"op=next(p for p in P if p.name=={name!r});"
            "print(json.dumps(op(fx, random.Random(99)).text))"
        )
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads(result.stdout)

    # Hash randomization must not perturb the deterministic output.
    assert _subprocess_text("0") == in_process
    assert _subprocess_text("123456") == in_process


def test_report_flags_over_budget_operators_against_clean_baseline():
    report = adversarial_perturbation_report(
        "exact",
        [_fixture()],
        runner=_naive_runner,
        suite_name="synthetic",
    )

    assert report.clean.overall == pytest.approx(0.0)
    assert not report.passed
    # Every operator evades the pristine-surface detector and raises leakage.
    assert set(report.violations) == {
        "bidi_control_wrapping",
        "combining_mark_injection",
        "homoglyph_substitution",
        "zero_width_injection",
    }
    for variant in report.variants:
        assert variant.flagged
        assert variant.delta_overall > 0.0
        # Report is keyed by operator and label.
        assert set(variant.flagged_labels) == {"ID_NUM", "PERSON"}
        assert set(variant.delta_by_label) == {"ID_NUM", "PERSON"}
        assert all(delta >= 0.0 for delta in variant.delta_by_label.values())


def test_report_passes_when_detector_recovers_perturbed_spans():
    report = adversarial_perturbation_report(
        "exact",
        [_fixture()],
        runner=_oracle_runner,
    )

    assert report.passed
    assert not report.violations
    for variant in report.variants:
        assert not variant.flagged
        assert variant.flagged_labels == ()
        assert variant.delta_overall == pytest.approx(0.0)
        assert variant.leakage.overall == pytest.approx(0.0)


def test_report_tolerates_deltas_within_configured_budget():
    report = adversarial_perturbation_report(
        "exact",
        [_fixture()],
        runner=_naive_runner,
        leakage_delta_budget=1.0,
    )

    # A budget of 1.0 admits full-leakage deltas, so nothing is flagged.
    assert report.passed
    assert not report.violations


def test_gate_raises_on_violation():
    with pytest.raises(
        AdversarialPerturbationGateError,
        match="adversarial-perturbation leakage gate failed",
    ):
        adversarial_perturbation_report(
            "exact",
            [_fixture()],
            runner=_naive_runner,
            raise_on_violation=True,
        )


def test_report_is_offline_and_serializes_without_raw_phi(monkeypatch):
    def _no_network(*args, **kwargs):
        raise AssertionError("adversarial-perturbation suite must run offline")

    monkeypatch.setattr(socket, "socket", _no_network)

    report = adversarial_perturbation_report(
        "exact",
        [_fixture()],
        runner=_naive_runner,
        suite_name="synthetic",
    )

    payload = json.dumps(report.to_dict(), sort_keys=True)
    assert "John Doe" not in payload
    assert "A1001" not in payload
    assert report.to_dict()["variants"]["homoglyph_substitution"]["flagged"] is True
    assert report.to_dict()["leakage_delta_budget"] == 0.0


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        adversarial_perturbation_report(
            "exact",
            [_fixture()],
            runner=_naive_runner,
            leakage_delta_budget=-0.1,
        )
