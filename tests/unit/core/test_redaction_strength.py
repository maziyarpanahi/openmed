from datetime import datetime

from openmed.core.labels import (
    AGE,
    DIRECT_IDENTIFIER,
    PERSON,
    QUASI_IDENTIFIER,
    RISK_HIGH,
    policy_label_for,
)
from openmed.core.pipeline import Pipeline
from openmed.core.policy import load_policy
from openmed.core.redaction_strength import (
    ActionMapComparison,
    action_meets_floor,
    action_strength,
    compare_action_maps,
    select_minimum_necessary,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult


def test_select_minimum_necessary_uses_policy_floors_without_mutating_source():
    strict = load_policy("hipaa_safe_harbor")
    target = load_policy("research_limited_dataset")
    original_actions = dict(strict.actions)

    derived = select_minimum_necessary(
        strict,
        target_posture=target,
    )

    assert derived is not strict
    assert dict(strict.actions) == original_actions
    assert derived.actions[PERSON] == target.policy_label_actions[DIRECT_IDENTIFIER]
    assert derived.actions[AGE] == target.policy_label_actions[QUASI_IDENTIFIER]
    assert action_strength(derived.actions[AGE]) < action_strength(strict.actions[AGE])

    direct_floor = target.policy_label_actions[DIRECT_IDENTIFIER]
    for label, action in derived.actions.items():
        if policy_label_for(label) == DIRECT_IDENTIFIER:
            assert action_meets_floor(action, direct_floor)


def test_select_minimum_necessary_can_raise_risk_override_to_direct_floor():
    derived = select_minimum_necessary(
        "clinical_minimal_redaction",
        target_posture="research_limited_dataset",
        risk_level_by_label={AGE: RISK_HIGH},
    )

    assert derived.actions[AGE] == "mask"


def test_derived_profile_drives_pipeline_and_preserves_source_profile():
    strict = load_policy("hipaa_safe_harbor")
    original_actions = dict(strict.actions)
    derived = select_minimum_necessary(
        strict,
        target_posture="research_limited_dataset",
    )
    text = "Patient John Doe is 47 years old"

    def model_detector(text, **kwargs):
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="John Doe",
                    label="NAME",
                    start=text.index("John Doe"),
                    end=text.index("John Doe") + len("John Doe"),
                    confidence=0.98,
                ),
                EntityPrediction(
                    text="47",
                    label="AGE",
                    start=text.index("47"),
                    end=text.index("47") + len("47"),
                    confidence=0.98,
                ),
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    result = Pipeline(
        model_detector=model_detector,
        policy=derived,
        use_safety_sweep=False,
    ).run(text, method="mask")

    policy_actions = {
        span.canonical_label: span.action
        for span in result.stage("policy_actions").spans
    }
    assert policy_actions[PERSON] == "mask"
    assert policy_actions[AGE] == "keep"
    assert result.redacted_text == "Patient [NAME] is 47 years old"
    assert dict(strict.actions) == original_actions


def test_compare_action_maps_orders_synthetic_inputs():
    baseline = {"direct": "mask", "quasi": "mask"}
    weaker = {"direct": "mask", "quasi": "keep"}
    stronger = {"direct": "redact", "quasi": "mask"}
    mixed = {"direct": "redact", "quasi": "keep"}

    assert compare_action_maps(weaker, baseline) == ActionMapComparison.WEAKER
    assert compare_action_maps(stronger, baseline) == ActionMapComparison.STRONGER
    assert compare_action_maps(mixed, baseline) == ActionMapComparison.INCOMPARABLE
    assert compare_action_maps(baseline, dict(baseline)) == ActionMapComparison.EQUAL
    assert (
        compare_action_maps({"direct": "mask"}, baseline)
        == ActionMapComparison.INCOMPARABLE
    )
