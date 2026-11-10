"""Offline tests for consent and data-use tag enforcement."""

from __future__ import annotations

import json
from datetime import datetime
from typing import cast

import pytest

from openmed.compliance import (
    DEFAULT_DATA_USE_POLICY,
    DEFAULT_DENIED_ACTIONS,
    DataUseAction,
    DataUsePolicy,
    DataUsePolicyViolation,
    DataUseTag,
)
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import PredictionResult


def _empty_prediction(text: str, model_name: str = "offline-stub") -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def _offline_pipeline(**kwargs):
    return Pipeline(
        model_detector=lambda text, **detector_kwargs: _empty_prediction(
            text,
            model_name=detector_kwargs["model_name"],
        ),
        **kwargs,
    )


def test_no_export_tag_blocks_fhir_export_before_text_processing():
    raw_phi = "Patient Jane Doe, MRN 123456"

    def must_not_run(*args, **kwargs):
        raise AssertionError("detector ran before data-use enforcement")

    pipeline = Pipeline(model_detector=must_not_run)

    with pytest.raises(DataUsePolicyViolation, match="no-export:fhir-export") as exc:
        pipeline.run(
            raw_phi,
            data_use_tags=[DataUseTag.NO_EXPORT],
            data_use_action=DataUseAction.FHIR_EXPORT,
        )

    report = exc.value.report.to_dict()
    assert report == {
        "tags": ["no-export"],
        "attempted_actions": ["fhir-export"],
        "decision": "deny",
        "violations": [
            {
                "tag": "no-export",
                "attempted_action": "fhir-export",
                "decision": "deny",
            }
        ],
    }
    assert raw_phi not in json.dumps(report)
    assert raw_phi not in str(exc.value)


def test_tags_propagate_to_context_output_and_audit_offline():
    observed_context_tags = None

    def policy_actions(spans, context):
        nonlocal observed_context_tags
        observed_context_tags = context.data_use_tags
        return spans

    result = _offline_pipeline(policy_actions=policy_actions).run(
        "Synthetic note without identifiers.",
        data_use_tags=["research_only", DataUseTag.NO_EXPORT],
        data_use_action=DataUseAction.RESEARCH,
    )

    expected_tags = ("no-export", "research-only")
    assert observed_context_tags == expected_tags
    assert result.data_use_tags == expected_tags
    assert result.data_use_decision == {
        "tags": list(expected_tags),
        "attempted_actions": ["research"],
        "decision": "allow",
        "violations": [],
    }
    assert result.audit_record["data_use"] == result.data_use_decision
    assert result.deidentification_result.metadata["data_use"] == (
        result.data_use_decision
    )


def test_permitted_processing_is_unaffected_by_no_export_constraint():
    pipeline = _offline_pipeline()
    untagged = pipeline.run("Synthetic note.")
    tagged = pipeline.run(
        "Synthetic note.",
        data_use_tags=[DataUseTag.NO_EXPORT],
        data_use_action=DataUseAction.PROCESS,
    )

    assert tagged.redacted_text == untagged.redacted_text
    assert tagged.spans == untagged.spans
    assert tagged.data_use_decision["decision"] == "allow"


def test_surrogate_vault_is_an_implicit_attempted_action():
    with pytest.raises(
        DataUsePolicyViolation,
        match="no-surrogate-vault:surrogate-vault",
    ) as exc:
        _offline_pipeline().run(
            "Synthetic note.",
            data_use_tags=[DataUseTag.NO_SURROGATE_VAULT],
            surrogate_vault=object(),
        )

    assert exc.value.report.to_dict()["attempted_actions"] == [
        "process",
        "surrogate-vault",
    ]


def test_consent_withdrawal_blocks_even_default_processing():
    with pytest.raises(DataUsePolicyViolation) as exc:
        _offline_pipeline().run(
            "Synthetic note.",
            data_use_tags=[DataUseTag.CONSENT_WITHDRAWN],
        )

    violation = exc.value.report.violations[0]
    assert violation.tag is DataUseTag.CONSENT_WITHDRAWN
    assert violation.attempted_action is DataUseAction.PROCESS
    assert violation.decision == "deny"


def test_unknown_tag_and_action_fail_closed():
    policy = DataUsePolicy()

    with pytest.raises(ValueError, match="fail closed"):
        policy.enforce(["no-exprot"], DataUseAction.EXPORT)
    with pytest.raises(ValueError, match="fail closed"):
        policy.enforce([DataUseTag.NO_EXPORT], "fhir-exprot")


def test_policy_can_add_an_organization_specific_denial():
    policy = DataUsePolicy(
        denied_actions={
            DataUseTag.RESEARCH_ONLY: [DataUseAction.RETENTION],
        }
    )

    with pytest.raises(DataUsePolicyViolation) as exc:
        policy.enforce(DataUseTag.RESEARCH_ONLY, DataUseAction.RETAIN)

    assert exc.value.report.to_dict()["violations"] == [
        {
            "tag": "research-only",
            "attempted_action": "retention",
            "decision": "deny",
        }
    ]


def test_exported_default_denials_cannot_be_weakened():
    mutable_defaults = cast(
        dict[DataUseTag, frozenset[DataUseAction]],
        DEFAULT_DENIED_ACTIONS,
    )

    with pytest.raises(TypeError):
        mutable_defaults[DataUseTag.NO_EXPORT] = frozenset()

    with pytest.raises(DataUsePolicyViolation):
        DEFAULT_DATA_USE_POLICY.enforce(
            DataUseTag.NO_EXPORT,
            DataUseAction.EXPORT,
        )


def test_policy_denials_cannot_be_mutated_after_construction():
    policy = DataUsePolicy()
    mutable_denials = cast(
        dict[DataUseTag, frozenset[DataUseAction]],
        policy.denied_actions,
    )

    with pytest.raises(TypeError):
        mutable_denials[DataUseTag.NO_EXPORT] = frozenset()

    with pytest.raises(DataUsePolicyViolation):
        policy.enforce(DataUseTag.NO_EXPORT, DataUseAction.EXPORT)
