from __future__ import annotations

from dataclasses import replace

import pytest

from openmed.eval.evidence_bundle import bundle_gate_evidence
from openmed.eval.harness import (
    DEFAULT_PIPELINE_EVAL_FIXTURE,
    PipelineEvalFixture,
    PipelineStageOutput,
    load_pipeline_eval_fixtures,
    run_pipeline_eval,
)
from openmed.eval.metrics import PIPELINE_EVAL_STAGES, PipelineFact
from openmed.eval.release_gates import (
    G15_E2E_FACT_F1_FLOOR,
    ReleaseGate,
    evaluate_end_to_end_pipeline_gate,
)


def _stage_runners(
    *,
    assertion_flip: bool = False,
    over_redact: bool = False,
):
    def deid(
        fixture: PipelineEvalFixture,
        _outputs: dict[str, PipelineStageOutput],
    ) -> tuple[PipelineFact, ...]:
        facts = fixture.gold_facts
        if over_redact:
            facts = tuple(fact for fact in facts if fact.fact_id != "fact-diabetes")
        return tuple(
            replace(
                fact,
                assertion="",
                code_system="",
                code="",
                resource_type="",
            )
            for fact in facts
        )

    def ner(
        _fixture: PipelineEvalFixture,
        outputs: dict[str, PipelineStageOutput],
    ) -> tuple[PipelineFact, ...]:
        return outputs["deid"].facts

    def assertion(
        fixture: PipelineEvalFixture,
        outputs: dict[str, PipelineStageOutput],
    ) -> tuple[PipelineFact, ...]:
        gold = {fact.fact_id: fact for fact in fixture.gold_facts}
        return tuple(
            replace(
                fact,
                assertion=(
                    "absent"
                    if assertion_flip and fact.fact_id == "fact-diabetes"
                    else gold[fact.fact_id].assertion
                ),
            )
            for fact in outputs["ner"].facts
        )

    def grounding(
        fixture: PipelineEvalFixture,
        outputs: dict[str, PipelineStageOutput],
    ) -> tuple[PipelineFact, ...]:
        gold = {fact.fact_id: fact for fact in fixture.gold_facts}
        return tuple(
            replace(
                fact,
                code_system=gold[fact.fact_id].code_system,
                code=gold[fact.fact_id].code,
            )
            for fact in outputs["assertion"].facts
        )

    def fhir(
        fixture: PipelineEvalFixture,
        outputs: dict[str, PipelineStageOutput],
    ) -> tuple[PipelineFact, ...]:
        gold = {fact.fact_id: fact for fact in fixture.gold_facts}
        return tuple(
            replace(fact, resource_type=gold[fact.fact_id].resource_type)
            for fact in outputs["grounding"].facts
        )

    return {
        "deid": deid,
        "ner": ner,
        "assertion": assertion,
        "grounding": grounding,
        "fhir": fhir,
    }


def _zero_stage_counts() -> dict[str, int]:
    return {stage: 0 for stage in PIPELINE_EVAL_STAGES}


def test_synthetic_fixture_runs_end_to_end_with_intermediate_outputs() -> None:
    fixtures = load_pipeline_eval_fixtures()

    report = run_pipeline_eval(fixtures, _stage_runners(), suite="synthetic-e2e")

    assert DEFAULT_PIPELINE_EVAL_FIXTURE.is_file()
    assert report.fixture_count == 1
    assert report.fact_level.f1 == pytest.approx(1.0)
    assert [output.stage for output in report.fixture_results[0].stage_outputs] == list(
        PIPELINE_EVAL_STAGES
    )
    assert report.attribution.total_end_to_end_errors == 0
    assert "Synthetic patient" not in report.to_json()


def test_assertion_flip_is_attributed_before_grounding() -> None:
    report = run_pipeline_eval(
        DEFAULT_PIPELINE_EVAL_FIXTURE,
        _stage_runners(assertion_flip=True),
    )

    attribution = report.attribution
    assert attribution.stage_error_counts["assertion"] == 1
    assert attribution.stage_error_counts["ner"] == 0
    assert attribution.stage_error_counts["grounding"] == 0
    assert attribution.findings[0].reason == "wrong_assertion"
    assert attribution.findings[0].mismatched_fields == ("assertion",)
    assert sum(attribution.stage_error_counts.values()) == (
        attribution.total_end_to_end_errors
    )


def test_deid_over_redaction_has_a_distinct_first_stage_bucket() -> None:
    report = run_pipeline_eval(
        DEFAULT_PIPELINE_EVAL_FIXTURE,
        _stage_runners(over_redact=True),
    )

    attribution = report.attribution
    assert attribution.stage_error_counts == {
        "deid": 1,
        "ner": 0,
        "assertion": 0,
        "grounding": 0,
        "fhir": 0,
    }
    assert attribution.findings[0].reason == "over_redaction"
    assert report.fact_level.f1 == pytest.approx(2 / 3)


def test_g15_fails_below_floor_and_on_stage_regression() -> None:
    below_floor = {
        "fact_f1": G15_E2E_FACT_F1_FLOOR - 0.01,
        "stage_error_counts": _zero_stage_counts(),
        "total_end_to_end_errors": 0,
    }
    floor_check = evaluate_end_to_end_pipeline_gate(below_floor)

    regressed_counts = _zero_stage_counts()
    regressed_counts["assertion"] = 1
    regression_metric = {
        "fact_f1": 1.0,
        "stage_error_counts": regressed_counts,
        "total_end_to_end_errors": 1,
    }
    regression_check = evaluate_end_to_end_pipeline_gate(
        regression_metric,
        {
            "fact_f1": 1.0,
            "stage_error_counts": _zero_stage_counts(),
            "total_end_to_end_errors": 0,
        },
    )

    assert floor_check.gate == "G15"
    assert not floor_check.passed
    assert "fact_f1" in floor_check.details["violations"]
    assert not regression_check.passed
    assert regression_check.details["violations"]["stage_regressions"] == {
        "assertion": {"baseline": 0, "observed": 1}
    }


def test_release_gate_and_evidence_bundle_surface_g15_attribution(tmp_path) -> None:
    report = run_pipeline_eval(
        DEFAULT_PIPELINE_EVAL_FIXTURE,
        _stage_runners(assertion_flip=True),
    )
    baseline_metric = {
        "fact_f1": 1.0,
        "stage_error_counts": _zero_stage_counts(),
        "total_end_to_end_errors": 0,
    }
    gate_report = ReleaseGate(signing_key="pipeline-unit-key").preview(
        {
            "metrics": {"end_to_end_pipeline": report.to_metric()},
            "metadata": {
                "family": "PII",
                "format": "mlx-fp",
                "repo_id": "OpenMed/pipeline-unit",
                "tier": "Tiny",
            },
        },
        {"metrics": {"end_to_end_pipeline": baseline_metric}},
    )
    g15 = next(check for check in gate_report.gate_results if check.gate == "G15")

    bundle = bundle_gate_evidence(
        gate_report,
        tmp_path / "bundle",
        pipeline_attribution=report.attribution,
    )

    assert not g15.passed
    assert g15.details["stage_error_counts"]["assertion"] == 1
    assert (
        bundle.manifest["pipeline_attribution"]["stage_error_counts"]["assertion"] == 1
    )
    assert bundle.manifest["gates"]["G15"]["status"] == "covered"
