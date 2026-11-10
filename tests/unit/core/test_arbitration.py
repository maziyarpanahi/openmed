import pytest

from openmed.core.arbitration import (
    MODE_BALANCED,
    MODE_HIGH_RECALL_UNION,
    apply_calibration,
    arbitrate,
    arbitration_mode,
    fit,
    is_more_specific_label,
)
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash


def _span(
    *,
    start: int = 0,
    end: int = 8,
    label: str = "PERSON",
    detector: str = "model:tiny",
    score: float = 0.9,
) -> OpenMedSpan:
    surface = f"{start}:{end}:{label}"
    return OpenMedSpan(
        doc_id="doc-1",
        start=start,
        end=end,
        text_hash=hmac_text_hash(surface, "test-secret"),
        entity_type=label,
        canonical_label=label,
        score=score,
        detector=detector,
    )


def test_overlapping_rules_span_wins_against_model_span():
    rules = _span(start=0, end=8, detector="rules:mrn_luhn", score=0.55)
    model = _span(start=0, end=12, detector="model:base", score=0.99)

    assert arbitrate([model, rules], mode=MODE_BALANCED) == (rules,)


def test_high_recall_union_keeps_span_balanced_mode_drops():
    weak_span = _span(label="EMAIL", detector="model:tiny", score=0.2)

    assert arbitrate([weak_span], mode=MODE_BALANCED) == ()
    assert arbitrate([weak_span], mode=MODE_HIGH_RECALL_UNION) == (weak_span,)
    assert arbitrate([weak_span], strict_no_leak=True) == (weak_span,)
    assert arbitration_mode(strict_no_leak=True) == MODE_HIGH_RECALL_UNION


def test_more_specific_label_wins_overlap_before_score():
    date = _span(start=4, end=14, label="DATE", detector="model:base", score=0.99)
    dob = _span(
        start=4,
        end=12,
        label="DATE_OF_BIRTH",
        detector="model:tiny",
        score=0.6,
    )

    assert is_more_specific_label("DATE_OF_BIRTH", "DATE")
    assert arbitrate([date, dob], mode=MODE_BALANCED) == (dob,)


def test_exact_duplicate_spans_collapse_to_best_scored_span():
    lower = _span(detector="model:tiny", score=0.6)
    higher = _span(detector="model:base", score=0.8)

    assert arbitrate([lower, higher], mode=MODE_BALANCED) == (higher,)


def test_score_calibration_makes_detector_scores_comparable():
    calibrator = fit(
        [
            {"detector": "model:a", "score": 0.2, "target": 0},
            {"detector": "model:a", "score": 0.8, "target": 1},
            {"detector": "model:b", "score": 0.7, "target": 0},
            {"detector": "model:b", "score": 0.9, "target": 1},
        ]
    )
    a_span = _span(detector="model:a", score=0.8)
    b_span = _span(start=12, end=20, detector="model:b", score=0.9)

    calibrated_a, calibrated_b = apply_calibration([a_span, b_span], calibrator)

    assert calibrated_a.score == pytest.approx(calibrated_b.score)
    assert calibrated_a.score == pytest.approx(1.0)
    assert calibrated_b.metadata["score_calibration"]["version"] == calibrator.version
