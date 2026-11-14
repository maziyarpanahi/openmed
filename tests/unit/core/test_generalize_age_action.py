"""Tests for the label-aware ``generalize`` action (Safe Harbor ages over 89)."""

from __future__ import annotations

from datetime import datetime

import pytest

from openmed.core.age_generalization import (
    AGE_OUTCOME_KEPT,
    AGE_OUTCOME_TOP_BAND,
    AGE_OUTCOME_UNPARSEABLE,
    AGE_TOP_BAND,
    AGE_TOP_THRESHOLD,
    generalize_age_span,
)
from openmed.core.arbitration import MODE_BALANCED
from openmed.core.labels import AGE, CANONICAL_LABELS
from openmed.core.pipeline import Pipeline
from openmed.core.policy import (
    _profile_from_mapping,
    compile_policy,
    list_policies,
    load_policy,
    policy_requirements,
)
from openmed.core.policy_diff import diff_policies
from openmed.core.policy_lint import lint_policy
from openmed.core.redaction_strength import action_strength
from openmed.core.schemas.span import ACTION_GENERALIZE, ACTION_VALUES
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.structured.hierarchies import functions as hierarchy_functions

SAFE_HARBOR = "hipaa_safe_harbor"


def _run(
    text: str,
    surfaces: list[tuple[str, str]],
    *,
    policy: object = SAFE_HARBOR,
    **run_kwargs: object,
):
    """Run the pipeline with a stub detector returning ``(surface, label)``."""

    def detector(text: str, **kwargs: object) -> PredictionResult:
        entities = []
        cursor = 0
        for surface, label in surfaces:
            start = text.index(surface, cursor)
            cursor = start + len(surface)
            entities.append(
                EntityPrediction(
                    text=surface,
                    label=label,
                    start=start,
                    end=cursor,
                    confidence=0.99,
                )
            )
        return PredictionResult(
            text=text,
            entities=entities,
            model_name="stub",
            timestamp=datetime.now().isoformat(),
        )

    return Pipeline(
        model_detector=detector,
        policy=policy,
        use_safety_sweep=False,
    ).run(text, method="mask", **run_kwargs)


def _actions(result) -> dict[str, str | None]:
    return {
        entity.text: entity.action
        for entity in result.deidentification_result.pii_entities
    }


@pytest.mark.parametrize(
    ("surface", "outcome", "rendered"),
    [
        ("54", AGE_OUTCOME_KEPT, "54"),
        ("54-year-old", AGE_OUTCOME_KEPT, "54-year-old"),
        ("0", AGE_OUTCOME_KEPT, "0"),
        ("89", AGE_OUTCOME_KEPT, "89"),
        ("90", AGE_OUTCOME_TOP_BAND, "90+"),
        ("93", AGE_OUTCOME_TOP_BAND, "90+"),
        ("93-year-old", AGE_OUTCOME_TOP_BAND, "90+-year-old"),
        ("93 y/o", AGE_OUTCOME_TOP_BAND, "90+ y/o"),
        ("aged 101", AGE_OUTCOME_TOP_BAND, "aged 90+"),
        ("93 ans", AGE_OUTCOME_TOP_BAND, "90+ ans"),
        ("९३", AGE_OUTCOME_TOP_BAND, "90+"),
        ("150", AGE_OUTCOME_TOP_BAND, "90+"),
        ("151", AGE_OUTCOME_UNPARSEABLE, None),
        ("1931", AGE_OUTCOME_UNPARSEABLE, None),
        ("ninety-three", AGE_OUTCOME_UNPARSEABLE, None),
        ("65-70", AGE_OUTCOME_UNPARSEABLE, None),
        ("92.5", AGE_OUTCOME_UNPARSEABLE, None),
        ("3 years 2 months", AGE_OUTCOME_UNPARSEABLE, None),
        ("", AGE_OUTCOME_UNPARSEABLE, None),
    ],
)
def test_generalize_age_span(surface: str, outcome: str, rendered: str | None) -> None:
    result = generalize_age_span(surface)

    assert result.outcome == outcome
    assert result.text == rendered


def test_text_and_structured_layers_share_one_top_band() -> None:
    assert hierarchy_functions.AGE_TOP_THRESHOLD == AGE_TOP_THRESHOLD == 90
    assert hierarchy_functions.AGE_TOP_BAND == AGE_TOP_BAND == "90+"
    assert hierarchy_functions._age_band(93, width=5) == AGE_TOP_BAND


def test_generalize_sits_between_format_preserve_and_replace() -> None:
    assert ACTION_GENERALIZE in ACTION_VALUES
    assert (
        action_strength("keep")
        < action_strength("format_preserve")
        < action_strength(ACTION_GENERALIZE)
        < action_strength("replace")
        < action_strength("mask")
    )


def test_safe_harbor_generalizes_age_and_other_profiles_do_not() -> None:
    assert load_policy(SAFE_HARBOR).actions[AGE] == ACTION_GENERALIZE
    for name in list_policies():
        if name == SAFE_HARBOR:
            continue
        profile = load_policy(name)
        assert ACTION_GENERALIZE not in set(profile.actions.values()), name
        assert profile.default_action != ACTION_GENERALIZE, name


def test_safe_harbor_profile_still_passes_policy_lint() -> None:
    report = lint_policy(SAFE_HARBOR)

    assert report["errors"] == []


def test_safe_harbor_compiles_with_generalize_as_the_age_requirement() -> None:
    requirements = {
        requirement.label: requirement.minimum_action
        for requirement in policy_requirements(SAFE_HARBOR)
    }

    assert requirements[AGE] == ACTION_GENERALIZE
    assert requirements["GENDER"] == "mask"
    proof = compile_policy(SAFE_HARBOR).proof
    assert proof.verified
    assert not proof.weak_actions


def test_policy_diff_reports_mask_to_generalize_as_weaker() -> None:
    payload = {
        "schema_version": 1,
        "name": SAFE_HARBOR,
        "posture": "test",
        "threshold_profile": "balanced",
        "default_action": "mask",
        "default_action_bias": "mask",
        "arbitration_mode": MODE_BALANCED,
        "safety_sweep_mandatory": False,
        "keep_mapping": False,
        "reversible_id": False,
        "forced_cascade_tiers": [],
        "actions": {label: "mask" for label in CANONICAL_LABELS},
    }
    masked = _profile_from_mapping(payload, source="test")
    generalized = _profile_from_mapping(
        {**payload, "actions": {**payload["actions"], AGE: ACTION_GENERALIZE}},
        source="test",
    )

    diff = diff_policies(masked, generalized)

    assert [
        (change.label, change.direction) for change in diff.changed_label_actions
    ] == [(AGE, "weaker")]


def test_age_89_and_below_stays_in_the_text() -> None:
    text = "A 54-year-old male presented with chest pain."

    result = _run(text, [("54", "AGE")])

    assert result.redacted_text == text
    entity = result.deidentification_result.pii_entities[0]
    assert entity.action == "keep"
    assert entity.redacted_text == "54"
    # A kept age must not be copied into the audit surrogate field.
    assert entity.surrogate is None


def test_age_over_89_renders_the_top_band() -> None:
    text = "A 93-year-old female and a 54-year-old male."

    result = _run(text, [("93", "AGE"), ("54", "AGE")])

    assert result.redacted_text == "A 90+-year-old female and a 54-year-old male."
    assert _actions(result) == {"93": ACTION_GENERALIZE, "54": "keep"}
    generalized = result.deidentification_result.pii_entities[0]
    assert generalized.redacted_text == AGE_TOP_BAND
    assert generalized.surrogate == AGE_TOP_BAND


def test_wide_age_span_keeps_its_token_shape() -> None:
    result = _run("Seen today: 93-year-old woman.", [("93-year-old", "AGE")])

    assert result.redacted_text == "Seen today: 90+-year-old woman."


def test_boundary_ages_89_and_90() -> None:
    result = _run("Ages 89 and 90 were recorded.", [("89", "AGE"), ("90", "AGE")])

    assert result.redacted_text == "Ages 89 and 90+ were recorded."
    assert _actions(result) == {"89": "keep", "90": ACTION_GENERALIZE}


def test_unparseable_age_falls_back_to_mask() -> None:
    text = "Mother aged ninety-three, sibling 65-70, born 1931."

    result = _run(
        text,
        [("ninety-three", "AGE"), ("65-70", "AGE"), ("1931", "AGE")],
    )

    assert result.redacted_text == "Mother aged [AGE], sibling [AGE], born [AGE]."
    assert set(_actions(result).values()) == {"mask"}
    for leaked in ("ninety-three", "65-70", "1931"):
        assert leaked not in result.redacted_text


def test_generalize_on_an_unsupported_label_falls_back_to_mask() -> None:
    payload = {
        "schema_version": 1,
        "name": SAFE_HARBOR,
        "posture": "test",
        "threshold_profile": "balanced",
        "default_action": "mask",
        "default_action_bias": "mask",
        "arbitration_mode": MODE_BALANCED,
        "safety_sweep_mandatory": False,
        "keep_mapping": False,
        "reversible_id": False,
        "forced_cascade_tiers": [],
        "actions": {
            **{label: "mask" for label in CANONICAL_LABELS},
            "ZIPCODE": ACTION_GENERALIZE,
        },
    }
    profile = _profile_from_mapping(payload, source="test")

    result = _run("ZIP 30301 on file.", [("30301", "ZIPCODE")], policy=profile)

    assert "30301" not in result.redacted_text
    assert _actions(result) == {"30301": "mask"}


def test_other_identifiers_are_still_masked_next_to_a_kept_age() -> None:
    text = "Alex Rivers, 54, phone 336-555-0187."

    result = _run(
        text,
        [("Alex Rivers", "PERSON"), ("54", "AGE"), ("336-555-0187", "PHONE")],
    )

    assert "Alex Rivers" not in result.redacted_text
    assert "336-555-0187" not in result.redacted_text
    assert ", 54, " in result.redacted_text


def test_generalized_ages_never_enter_the_reidentification_mapping() -> None:
    text = "Ages 93, 54, ninety-one and ninety-two."

    result = _run(
        text,
        [
            ("93", "AGE"),
            ("54", "AGE"),
            ("ninety-one", "AGE"),
            ("ninety-two", "AGE"),
        ],
        keep_mapping=True,
    )

    mapping = result.deidentification_result.mapping or {}
    assert result.redacted_text.startswith("Ages 90+, 54, ")
    assert sorted(mapping.values()) == ["ninety-one", "ninety-two"]
    # Masked fallbacks stay unique so the mapping remains reversible.
    assert len(mapping) == 2
