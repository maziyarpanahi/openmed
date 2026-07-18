"""Tests for inter-annotator agreement metrics on extraction gold sets."""

from __future__ import annotations

import json

import pytest

from openmed.eval.metrics import (
    InterAnnotatorAgreement,
    cohen_kappa_agreement,
    fleiss_kappa_agreement,
    inter_annotator_agreement,
)

# Annotators are lists of (start, end, label) span tuples.
PERSON_A = [(0, 5, "PERSON"), (10, 15, "DATE")]
PERSON_B = [(0, 5, "PERSON"), (10, 15, "LOCATION")]


# --------------------------------------------------------------------------
# Cohen kappa (two annotators) -- hand-computed
# --------------------------------------------------------------------------


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa_agreement(PERSON_A, PERSON_A) == pytest.approx(1.0)


def test_cohen_kappa_label_disagreement():
    # Items (0,5) agree; (10,15) disagree (DATE vs LOCATION).
    # p_o = 0.5, p_e = 0.25, kappa = (0.5-0.25)/(1-0.25) = 1/3.
    assert cohen_kappa_agreement(PERSON_A, PERSON_B) == pytest.approx(1 / 3, abs=1e-6)


def test_cohen_kappa_span_presence_disagreement():
    a = [(0, 5, "PERSON")]
    b = [(0, 5, "PERSON"), (10, 15, "DATE")]
    # (0,5) agree; (10,15) A missing vs B DATE -> p_o=0.5, p_e=0.25 -> 1/3.
    assert cohen_kappa_agreement(a, b) == pytest.approx(1 / 3, abs=1e-6)


def test_cohen_kappa_requires_two_annotators():
    with pytest.raises(ValueError):
        cohen_kappa_agreement(PERSON_A, PERSON_A, PERSON_A)


# --------------------------------------------------------------------------
# Fleiss kappa (three or more annotators) -- hand-computed
# --------------------------------------------------------------------------


def test_fleiss_kappa_perfect_agreement():
    raters = [PERSON_A, PERSON_A, PERSON_A]
    assert fleiss_kappa_agreement(raters) == pytest.approx(1.0)


def test_fleiss_kappa_partial_agreement():
    # (0,5): all PERSON; (10,15): two DATE, one LOCATION.
    # P_bar = (1 + 1/3)/2 = 2/3; P_e = 0.5^2 + (1/3)^2 + (1/6)^2 = 0.3889.
    # kappa = (2/3 - 0.3889)/(1 - 0.3889) = 0.4545...
    raters = [
        [(0, 5, "PERSON"), (10, 15, "DATE")],
        [(0, 5, "PERSON"), (10, 15, "DATE")],
        [(0, 5, "PERSON"), (10, 15, "LOCATION")],
    ]
    assert fleiss_kappa_agreement(raters) == pytest.approx(0.4545, abs=1e-3)


def test_fleiss_kappa_requires_three_annotators():
    with pytest.raises(ValueError):
        fleiss_kappa_agreement([PERSON_A, PERSON_B])


# --------------------------------------------------------------------------
# Report: overall / per-label / disagreements / span-F1
# --------------------------------------------------------------------------


def test_report_two_annotators_uses_cohen():
    report = inter_annotator_agreement([PERSON_A, PERSON_B])

    assert isinstance(report, InterAnnotatorAgreement)
    assert report.n_annotators == 2
    assert report.cohen_kappa == pytest.approx(1 / 3, abs=1e-6)
    assert report.fleiss_kappa is None
    # Perfectly agreed (0,5) span -> F1 counts it; (10,15) mismatched labels.
    assert 0.0 <= report.mean_span_f1 <= 1.0


def test_report_three_annotators_uses_fleiss():
    report = inter_annotator_agreement([PERSON_A, PERSON_A, PERSON_B])

    assert report.n_annotators == 3
    assert report.fleiss_kappa is not None
    assert report.cohen_kappa is None


def test_report_per_label_agreement():
    report = inter_annotator_agreement([PERSON_A, PERSON_B])

    # PERSON is agreed by both; DATE/LOCATION are the disagreement.
    assert report.per_label["PERSON"] == pytest.approx(1.0)
    assert report.per_label["DATE"] < 1.0


def test_disagreements_carry_offsets_and_labels_not_text():
    report = inter_annotator_agreement([PERSON_A, PERSON_B])

    assert report.disagreements
    example = report.disagreements[0]
    assert example["offset"] == (10, 15)
    assert set(example["labels"]) == {"DATE", "LOCATION"}
    # No raw clinical text leaks into the disagreement record.
    assert "text" not in example
    assert "text" not in json.dumps(report.to_dict())


def test_per_relation_agreement():
    # Relations keyed by type; annotators disagree on one relation's label.
    rel_a = {"drug_to_dose": [(0, 5, "confirmed")]}
    rel_b = {"drug_to_dose": [(0, 5, "refuted")]}
    report = inter_annotator_agreement([PERSON_A, PERSON_A], relations=[rel_a, rel_b])

    assert "drug_to_dose" in report.per_relation
    assert report.per_relation["drug_to_dose"] < 1.0
