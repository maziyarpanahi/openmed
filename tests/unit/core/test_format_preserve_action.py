from __future__ import annotations

import copy
import re
from datetime import datetime

import pytest

from openmed.core.anonymizer.format_preserve import extract_digit_groups
from openmed.core.arbitration import MODE_BALANCED
from openmed.core.labels import CANONICAL_LABELS
from openmed.core.pii import deidentify
from openmed.core.pipeline import Pipeline
from openmed.core.policy import _profile_from_mapping
from openmed.core.thresholds import validate_threshold_matrix
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _entity(text: str, surface: str, label: str) -> EntityPrediction:
    start = text.index(surface)
    return EntityPrediction(
        text=surface,
        label=label,
        start=start,
        end=start + len(surface),
        confidence=0.99,
    )


def _prediction(
    text: str,
    entities: list[EntityPrediction],
) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )


def _same_format(original: str, replacement: str) -> bool:
    if len(original) != len(replacement):
        return False
    for original_char, replacement_char in zip(original, replacement):
        if original_char.isdigit():
            if not replacement_char.isdigit():
                return False
        elif original_char.isalpha():
            if not replacement_char.isalpha():
                return False
            if original_char.isupper() != replacement_char.isupper():
                return False
        elif original_char != replacement_char:
            return False
    return True


def _profile_payload(
    action_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    actions = {label: "mask" for label in CANONICAL_LABELS}
    actions.update(action_overrides or {})
    return {
        "schema_version": 1,
        "name": "hipaa_safe_harbor",
        "posture": "test",
        "threshold_profile": "balanced",
        "default_action": "mask",
        "default_action_bias": "mask",
        "arbitration_mode": MODE_BALANCED,
        "safety_sweep_mandatory": False,
        "keep_mapping": False,
        "reversible_id": False,
        "forced_cascade_tiers": [],
        "actions": actions,
    }


def _threshold_matrix(action: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "profiles": {
            "balanced": {
                "recall_floor": 0.95,
                "default": {
                    "keep_floor": 0.7,
                    "escalate_below": 0.8,
                    "action": action,
                },
            }
        },
    }


def test_policy_profile_phone_format_preserve_keeps_shape_not_value():
    text = "Call +1 (415) 555-1234"
    original = "+1 (415) 555-1234"
    profile = _profile_from_mapping(
        _profile_payload({"PHONE": "format_preserve"}),
        source="test",
    )

    result = Pipeline(
        model_detector=lambda text, **kwargs: _prediction(
            text,
            [_entity(text, original, "PHONE")],
        ),
        policy=profile,
        use_safety_sweep=False,
    ).run(text, method="mask", consistent=True, seed=7)

    replacement = result.deidentification_result.pii_entities[0].redacted_text
    assert replacement is not None
    assert replacement != original
    assert result.redacted_text == f"Call {replacement}"
    assert re.sub(r"\d", "0", replacement) == re.sub(r"\d", "0", original)
    assert extract_digit_groups(replacement) == extract_digit_groups(original)
    assert result.deidentification_result.pii_entities[0].action == "format_preserve"


def test_deidentify_format_preserve_masks_unsupported_labels(monkeypatch):
    text = "MRN AB-1234 belongs to John Doe"
    original_id = "AB-1234"

    def fake_extract(text: str, *args: object, **kwargs: object) -> PredictionResult:
        return _prediction(
            text,
            [
                _entity(text, original_id, "ID_NUM"),
                _entity(text, "John Doe", "NAME"),
            ],
        )

    from openmed.core import pii

    monkeypatch.setattr(pii, "extract_pii", fake_extract)

    result = deidentify(
        text,
        method="format_preserve",
        use_safety_sweep=False,
        consistent=True,
        seed=42,
    )
    replacement = result.pii_entities[0].redacted_text

    assert replacement is not None
    assert _same_format(original_id, replacement)
    assert replacement != original_id
    assert result.deidentified_text == f"MRN {replacement} belongs to [NAME]"
    assert [entity.action for entity in result.pii_entities] == [
        "format_preserve",
        "mask",
    ]


def test_format_preserve_action_validation_accepts_only_known_actions():
    profile = _profile_from_mapping(
        _profile_payload({"PHONE": "format_preserve"}),
        source="test",
    )
    assert profile.action_for("PHONE") == "format_preserve"

    with pytest.raises(ValueError, match="actions.PHONE must be one of"):
        _profile_from_mapping(_profile_payload({"PHONE": "unknown"}), source="test")

    validate_threshold_matrix(_threshold_matrix("format_preserve"))

    invalid_matrix = copy.deepcopy(_threshold_matrix("format_preserve"))
    invalid_matrix["profiles"]["balanced"]["default"]["action"] = "unknown"
    with pytest.raises(ValueError, match="balanced.default.action"):
        validate_threshold_matrix(invalid_matrix)


def test_format_preserve_leakage_does_not_copy_identifier_digits(monkeypatch):
    text = "Card 4111-1111-1111-1111"
    original = "4111-1111-1111-1111"

    def fake_extract(text: str, *args: object, **kwargs: object) -> PredictionResult:
        return _prediction(text, [_entity(text, original, "CREDIT_CARD")])

    from openmed.core import pii

    monkeypatch.setattr(pii, "extract_pii", fake_extract)

    result = deidentify(
        text,
        method="format_preserve",
        use_safety_sweep=False,
        consistent=True,
        seed=11,
    )
    replacement = result.pii_entities[0].redacted_text

    assert replacement is not None
    assert _same_format(original, replacement)
    assert original not in result.deidentified_text
    for source_char, replacement_char in zip(original, replacement):
        if source_char.isdigit():
            assert replacement_char != source_char
