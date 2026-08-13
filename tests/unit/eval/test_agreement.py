"""Focused synthetic tests for annotation agreement scoring."""

from __future__ import annotations

import json

import pytest

from openmed.eval.annotation.agreement import (
    AgreementReport,
    cohen_kappa,
    fleiss_kappa,
    inter_annotator_agreement,
)
from openmed.eval.error_analysis import mine_gate_failure_labeling_queue

ANNOTATOR_A = [(0, 5, "PERSON"), (10, 15, "DATE")]
ANNOTATOR_B = [(0, 5, "PERSON"), (10, 15, "LOCATION")]


def test_cohen_kappa_matches_hand_computed_fixture() -> None:
    # p_o = 1/2, p_e = 1/4, so kappa = (1/2 - 1/4) / (1 - 1/4) = 1/3.
    assert cohen_kappa(ANNOTATOR_A, ANNOTATOR_B) == pytest.approx(1 / 3)


def test_fleiss_kappa_matches_hand_computed_fixture() -> None:
    raters = [
        ANNOTATOR_A,
        ANNOTATOR_A,
        [(0, 5, "PERSON"), (10, 15, "LOCATION")],
    ]
    assert fleiss_kappa(raters) == pytest.approx(0.4545, abs=1e-3)


def test_perfect_and_total_disagreement_edge_cases() -> None:
    assert cohen_kappa(ANNOTATOR_A, ANNOTATOR_A) == pytest.approx(1.0)
    assert fleiss_kappa([ANNOTATOR_A, ANNOTATOR_A, ANNOTATOR_A]) == pytest.approx(1.0)
    total_disagreement_a = [(0, 1, "A"), (2, 3, "B")]
    total_disagreement_b = [(0, 1, "B"), (2, 3, "A")]
    assert cohen_kappa(total_disagreement_a, total_disagreement_b) == pytest.approx(
        -1.0
    )
    total_disagreement_raters = [
        [(0, 1, "A"), (2, 3, "B")],
        [(0, 1, "B"), (2, 3, "C")],
        [(0, 1, "C"), (2, 3, "A")],
    ]
    assert fleiss_kappa(total_disagreement_raters) == pytest.approx(-0.5)


def test_overlap_matching_handles_small_boundary_shift() -> None:
    shifted = [[(0, 5, "PERSON")], [(1, 6, "PERSON")]]

    exact = inter_annotator_agreement(shifted, match="exact")
    overlap = inter_annotator_agreement(shifted, match="overlap")

    assert exact.kappa == pytest.approx(-1.0)
    assert overlap.kappa == pytest.approx(1.0)
    assert overlap.mean_span_f1 == pytest.approx(1.0)


def test_report_breakdowns_disagreements_and_queue_are_phi_free() -> None:
    report = inter_annotator_agreement(
        [
            [
                {"start": 0, "end": 5, "label": "PERSON", "text": "Synthetic Name"},
                {"start": 10, "end": 15, "label": "DATE", "text": "Synthetic Date"},
            ],
            ANNOTATOR_B,
        ],
        relations=[
            {"drug_to_dose": [(0, 5, "confirmed")]},
            {"drug_to_dose": [(0, 5, "refuted")]},
        ],
    )

    assert isinstance(report, AgreementReport)
    assert report.per_label["PERSON"] == pytest.approx(1.0)
    assert report.per_label["DATE"] == pytest.approx(0.0)
    assert report.per_relation["drug_to_dose"] == pytest.approx(0.0)
    assert report.disagreements == (
        {"offset": (10, 15), "labels": ("DATE", "LOCATION")},
    )

    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)
    assert "Synthetic Name" not in serialized
    assert "Synthetic Date" not in serialized
    assert payload["disagreements"] == [
        {"offset": [10, 15], "labels": ["DATE", "LOCATION"]}
    ]
    assert report.to_active_learning_queue() == (
        {
            "end": 15,
            "kind": "annotator_disagreement",
            "label": "DATE",
            "matched_label": "LOCATION",
            "start": 10,
            "uncertainty": 1.0,
        },
    )


def test_report_materializes_annotation_generators_once() -> None:
    report = inter_annotator_agreement(
        [
            (span for span in ANNOTATOR_A),
            (span for span in ANNOTATOR_B),
        ]
    )

    assert report.n_items == 2
    assert report.kappa == pytest.approx(1 / 3)


def test_report_queue_candidates_are_consumable_by_labeling_queue() -> None:
    report = inter_annotator_agreement([ANNOTATOR_A, ANNOTATOR_B])

    queue = mine_gate_failure_labeling_queue(
        report.to_active_learning_queue(),
        gate_run_hash="synthetic-agreement-run",
    )

    assert len(queue.items) == 1
    assert queue.items[0].provenance["kind"] == "annotator_disagreement"
    assert queue.items[0].provenance["start"] == 10
    assert queue.items[0].provenance["end"] == 15


def test_invalid_alignment_and_annotator_counts_are_explicit() -> None:
    with pytest.raises(ValueError, match="at least two"):
        inter_annotator_agreement([ANNOTATOR_A])
    with pytest.raises(ValueError, match="either 'exact' or 'overlap'"):
        inter_annotator_agreement([ANNOTATOR_A, ANNOTATOR_B], match="token")  # type: ignore[arg-type]
