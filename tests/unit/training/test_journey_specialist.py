"""Reproducibility, budget, holdout, and promotion tests for OJ-011."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
from importlib import resources

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.clinical.journey_contracts import sha256_digest
from openmed.eval.suites.journey_specialist import (
    SpecialistPrediction,
    SpecialistPromotionPolicy,
    build_journey_specialist_model_pack_entry,
    evaluate_journey_specialist_holdout,
    finalize_journey_specialist_run,
    render_journey_specialist_model_card,
)
from openmed.structured.store import StoreState
from openmed.training.journey_specialist import (
    JOURNEY_SPECIALIST_GPU_BUDGET_USD,
    JOURNEY_SPECIALIST_TASKS,
    GpuSpendLedger,
    JourneySpecialistConflictError,
    JourneySpecialistDeniedError,
    SpecialistTrainingExample,
    build_specialist_split,
    dry_run_journey_specialist_pack,
    load_journey_specialist_examples,
    load_journey_specialist_pack,
    load_journey_specialist_schema,
    render_journey_specialist_pack_card,
)

CODE_REVISION = "a" * 40
HOLDOUT_DIGEST = sha256_digest("synthetic-frozen-holdout")
CALIBRATION_DIGEST = sha256_digest("synthetic-frozen-calibration")


def _config() -> dict[str, object]:
    resource = resources.files("openmed.training.configs").joinpath(
        "journey_specialist_pack.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def _predictions(
    task: str,
    labels: tuple[str, ...],
    *,
    confidence: float = 0.96,
    quantized_error: bool = False,
) -> tuple[SpecialistPrediction, ...]:
    predictions: list[SpecialistPrediction] = []
    remaining = (1.0 - confidence) / (len(labels) - 1)
    for index, gold in enumerate(labels * 4):
        scores = {label: remaining for label in labels}
        scores[gold] = confidence
        quantized = dict(scores)
        if quantized_error and index == 0:
            wrong = next(label for label in labels if label != gold)
            quantized = {
                label: (0.02 if label not in {gold, wrong} else 0.04)
                for label in labels
            }
            quantized[wrong] = 1.0 - sum(
                value for label, value in quantized.items() if label != wrong
            )
        predictions.append(
            SpecialistPrediction(
                example_id=f"case_{index:04d}",
                task=task,
                gold_label=gold,
                scores=scores,
                quantized_scores=quantized,
                subgroups={"site": "synthetic", "format": "short"},
            )
        )
    return tuple(predictions)


def _report(
    task: str = "classification",
    *,
    baseline_macro_f1: float = 0.5,
    confidence: float = 0.96,
    quantized_error: bool = False,
):
    plan = load_journey_specialist_pack()
    recipe = plan.recipe_for(task)
    policy = SpecialistPromotionPolicy.from_recipe(
        recipe,
        baseline_macro_f1=baseline_macro_f1,
        candidate_alias=f"specialist.{task}",
        fallback_alias=f"rules.{task}",
    )
    result = evaluate_journey_specialist_holdout(
        _predictions(
            task,
            recipe.labels,
            confidence=confidence,
            quantized_error=quantized_error,
        ),
        recipe=recipe,
        policy=policy,
        dataset_digest=plan.dataset.content_digest,
        calibration_digest=CALIBRATION_DIGEST,
        holdout_digest=HOLDOUT_DIGEST,
    )
    assert result.ok and result.value is not None
    return plan, recipe, result.value


def test_bundled_specialist_pack_is_pinned_budgeted_and_schema_valid() -> None:
    plan = load_journey_specialist_pack()
    schema = load_journey_specialist_schema()
    validator_for(schema).check_schema(schema)
    validator = validator_for(schema)(schema)

    assert tuple(item.task for item in plan.recipes) == tuple(
        sorted(JOURNEY_SPECIALIST_TASKS)
    )
    assert plan.aggregate_gpu_budget_usd == JOURNEY_SPECIALIST_GPU_BUDGET_USD
    assert sum(item.estimated_cost_usd for item in plan.recipes) == 468.0
    assert plan.dataset.synthetic and plan.dataset.bundled
    assert plan.dataset.usage_lane == "distributable"
    assert plan.dataset.license == "cc0-1.0"
    assert not tuple(validator.iter_errors(plan.to_dict()))
    assert load_journey_specialist_pack(plan.to_dict()) == plan
    for recipe in plan.recipes:
        assert recipe.backbone.model_id == "microsoft/deberta-v3-small"
        assert recipe.backbone.revision == "a36c739020e01763fe789b4b85e2df55d6180012"
        assert recipe.backbone.license == "mit"
        assert recipe.adapter.method == "lora"
        assert recipe.mixed_precision == "bf16"
        assert recipe.estimated_gpu_hours <= recipe.stop_rule.maximum_gpu_hours
        assert {"safetensors", "onnx"}.issubset(recipe.export_formats)


def test_offline_dry_run_is_deterministic_raw_free_and_schema_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run attempted network access")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    first = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    second = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)

    assert first.ok and first.value is not None
    assert second.ok and second.value is not None
    assert first.value == second.value
    assert first.value.ledger.committed_cost_usd == 468.0
    assert first.value.ledger.remaining_cost_usd == 532.0
    assert len(first.value.manifests) == 4
    payload = first.value.to_json()
    assert "Synthetic finding" not in payload
    assert "source_text" not in payload
    schema = load_journey_specialist_schema()
    validator = validator_for(schema)(schema)
    assert not tuple(validator.iter_errors(first.value.to_dict()))
    assert all(
        not tuple(validator.iter_errors(item.to_dict()))
        for item in first.value.manifests
    )


def test_dataset_integrity_license_and_cap_fail_closed_with_typed_states() -> None:
    conflict = dry_run_journey_specialist_pack(
        code_revision=CODE_REVISION,
        dataset_payload=b"tampered-synthetic-dataset",
    )
    denied_config = _config()
    denied_config["dataset"]["license"] = "restricted"  # type: ignore[index]
    denied = dry_run_journey_specialist_pack(
        code_revision=CODE_REVISION,
        config_payload=denied_config,
    )
    eval_only_config = _config()
    eval_only_config["dataset"].update(  # type: ignore[union-attr]
        {
            "bundled": False,
            "license": "restricted",
            "synthetic": False,
            "usage_lane": "user_supplied_eval",
        }
    )
    eval_only = dry_run_journey_specialist_pack(
        code_revision=CODE_REVISION,
        config_payload=eval_only_config,
        dataset_payload=b"caller-supplied-eval-only",
    )
    budget_config = _config()
    budget_config["aggregate_gpu_budget_usd"] = 1000.01
    over_budget = dry_run_journey_specialist_pack(
        code_revision=CODE_REVISION,
        config_payload=budget_config,
    )
    unsupported_config = _config()
    unsupported_config["recipes"][0]["adapter"]["method"] = "full_tune"  # type: ignore[index]
    unsupported = dry_run_journey_specialist_pack(
        code_revision=CODE_REVISION,
        config_payload=unsupported_config,
    )
    malformed = dry_run_journey_specialist_pack(
        code_revision=CODE_REVISION,
        config_payload=b"{not-json",
    )

    assert (conflict.state, conflict.code) == (
        StoreState.CONFLICT,
        "specialist_plan_conflict",
    )
    assert (denied.state, denied.code) == (
        StoreState.DENIED,
        "specialist_plan_denied",
    )
    assert (eval_only.state, eval_only.code) == (
        StoreState.DENIED,
        "specialist_plan_denied",
    )
    assert (over_budget.state, over_budget.code) == (
        StoreState.DENIED,
        "specialist_plan_denied",
    )
    assert (unsupported.state, unsupported.code) == (
        StoreState.UNSUPPORTED,
        "specialist_plan_unsupported",
    )
    assert (malformed.state, malformed.code) == (
        StoreState.FAILURE,
        "specialist_plan_invalid",
    )


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_seeded_split_is_reproducible_and_stratified(seed: int) -> None:
    plan = load_journey_specialist_pack()
    recipe = replace(plan.recipe_for("assertion"), seed=seed)
    examples = load_journey_specialist_examples(plan.dataset)

    first = build_specialist_split(
        recipe, examples, dataset_digest=plan.dataset.content_digest
    )
    second = build_specialist_split(
        recipe, tuple(reversed(examples)), dataset_digest=plan.dataset.content_digest
    )

    assert first.ok and first.value is not None
    assert second.ok and second.value is not None
    assert first.value == second.value
    assert first.value.split_counts == {
        "holdout": 4,
        "train": 4,
        "validation": 4,
    }
    assert all(
        counts == {"holdout": 1, "train": 1, "validation": 1}
        for counts in first.value.label_counts.values()
    )


def test_split_reports_unknown_partial_and_conflict_without_false_success() -> None:
    plan = load_journey_specialist_pack()
    recipe = plan.recipe_for("assertion")
    missing = build_specialist_split(
        recipe, (), dataset_digest=plan.dataset.content_digest
    )
    partial_examples = tuple(
        SpecialistTrainingExample(
            example_id=f"tiny_{index:04d}",
            task="assertion",
            label=label,
            text="Synthetic.",
        )
        for index, label in enumerate(recipe.labels)
    )
    partial = build_specialist_split(
        recipe, partial_examples, dataset_digest=plan.dataset.content_digest
    )
    mismatched = build_specialist_split(
        recipe,
        (
            *partial_examples,
            SpecialistTrainingExample(
                example_id="extra_0001",
                task="assertion",
                label="unsupported_label",
                text="Synthetic.",
            ),
        ),
        dataset_digest=plan.dataset.content_digest,
    )

    assert (missing.state, missing.code) == (
        StoreState.UNKNOWN,
        "task_examples_missing",
    )
    assert (partial.state, partial.code) == (
        StoreState.PARTIAL,
        "task_split_insufficient",
    )
    assert (mismatched.state, mismatched.code) == (
        StoreState.CONFLICT,
        "task_labels_mismatch",
    )


@given(actual_cost=st.floats(min_value=1000.01, max_value=10000, allow_nan=False))
def test_shared_gpu_ledger_never_exceeds_hard_cap(actual_cost: float) -> None:
    dry_run = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    assert dry_run.ok and dry_run.value is not None
    run_id = dry_run.value.manifests[0].run_id

    with pytest.raises(JourneySpecialistDeniedError, match="exceeds"):
        dry_run.value.ledger.record_actual(
            run_id,
            actual_cost_usd=actual_cost,
            actual_gpu_hours=1,
        )


def test_cancelled_run_still_counts_measured_spend() -> None:
    dry = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    assert dry.value is not None
    cancelled = replace(
        dry.value.ledger.entries[0],
        state="cancelled",
        actual_cost_usd=1001.0,
        actual_gpu_hours=1.0,
    )
    with pytest.raises(JourneySpecialistDeniedError, match="exceeds"):
        GpuSpendLedger(entries=(cancelled,))


def test_completed_spend_is_idempotent_but_cannot_be_rewritten() -> None:
    dry = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    assert dry.value is not None
    run_id = dry.value.ledger.entries[0].run_id
    completed = dry.value.ledger.record_actual(
        run_id, actual_cost_usd=10.0, actual_gpu_hours=1.0
    )
    assert (
        completed.record_actual(run_id, actual_cost_usd=10.0, actual_gpu_hours=1.0)
        == completed
    )
    with pytest.raises(JourneySpecialistConflictError):
        completed.record_actual(run_id, actual_cost_usd=0.0, actual_gpu_hours=0.0)


def test_frozen_holdout_reports_metrics_calibration_slices_and_promotion() -> None:
    plan, recipe, report = _report("classification")
    schema = load_journey_specialist_schema()
    validator = validator_for(schema)(schema)

    assert report.promotion_decision == "promote"
    assert report.selected_alias == "specialist.classification"
    assert report.metrics["macro_f1"] == 1.0
    assert report.calibration["expected_calibration_error"] == pytest.approx(0.04)
    assert report.abstention["coverage"] == 1.0
    assert set(report.per_class) == set(recipe.labels)
    assert set(report.subgroup_slices) == {"format:short", "site:synthetic"}
    assert report.quantized_delta["macro_f1_delta"] == 0.0
    assert not tuple(validator.iter_errors(report.to_dict()))
    assert plan.dataset.content_digest == report.dataset_digest


def test_promotion_holds_and_selects_fallback_when_evidence_is_not_better() -> None:
    _, _, baseline_hold = _report(
        "classification",
        baseline_macro_f1=1.0,
    )
    _, _, calibration_hold = _report(
        "classification",
        confidence=0.60,
    )
    _, _, quantized_hold = _report(
        "classification",
        quantized_error=True,
    )

    assert baseline_hold.promotion_decision == "hold"
    assert baseline_hold.selected_alias == "rules.classification"
    assert "baseline_not_beaten" in baseline_hold.promotion_reasons
    assert calibration_hold.promotion_decision == "hold"
    assert "calibration_above_ceiling" in calibration_hold.promotion_reasons
    assert "coverage_below_floor" in calibration_hold.promotion_reasons
    assert quantized_hold.promotion_decision == "hold"
    assert "quantized_delta_above_ceiling" in quantized_hold.promotion_reasons
    assert quantized_hold.quantized_delta["maximum_per_class_recall_delta"] > 0


def test_holdout_missing_quantization_and_bad_task_are_typed() -> None:
    plan = load_journey_specialist_pack()
    recipe = plan.recipe_for("classification")
    policy = SpecialistPromotionPolicy.from_recipe(
        recipe,
        baseline_macro_f1=0.5,
        candidate_alias="specialist.classification",
        fallback_alias="rules.classification",
    )
    no_predictions = evaluate_journey_specialist_holdout(
        (),
        recipe=recipe,
        policy=policy,
        dataset_digest=plan.dataset.content_digest,
        calibration_digest=CALIBRATION_DIGEST,
        holdout_digest=HOLDOUT_DIGEST,
    )
    missing_quantized = tuple(
        replace(item, quantized_scores=None)
        for item in _predictions("classification", recipe.labels)
    )
    partial = evaluate_journey_specialist_holdout(
        missing_quantized,
        recipe=recipe,
        policy=policy,
        dataset_digest=plan.dataset.content_digest,
        calibration_digest=CALIBRATION_DIGEST,
        holdout_digest=HOLDOUT_DIGEST,
    )
    wrong_task_recipe = plan.recipe_for("assertion")
    unsupported = evaluate_journey_specialist_holdout(
        _predictions("classification", recipe.labels),
        recipe=wrong_task_recipe,
        policy=replace(
            policy, task="assertion", baseline_id="builtin/assertion_rules.v1"
        ),
        dataset_digest=plan.dataset.content_digest,
        calibration_digest=CALIBRATION_DIGEST,
        holdout_digest=HOLDOUT_DIGEST,
    )
    mismatched_baseline = evaluate_journey_specialist_holdout(
        _predictions("classification", recipe.labels),
        recipe=recipe,
        policy=replace(policy, baseline_id="builtin/different_rules.v1"),
        dataset_digest=plan.dataset.content_digest,
        calibration_digest=CALIBRATION_DIGEST,
        holdout_digest=HOLDOUT_DIGEST,
    )

    assert (no_predictions.state, no_predictions.code) == (
        StoreState.UNKNOWN,
        "holdout_predictions_missing",
    )
    assert (partial.state, partial.code) == (
        StoreState.PARTIAL,
        "quantized_holdout_incomplete",
    )
    assert (unsupported.state, unsupported.code) == (
        StoreState.UNSUPPORTED,
        "holdout_task_unsupported",
    )
    assert (mismatched_baseline.state, mismatched_baseline.code) == (
        StoreState.CONFLICT,
        "promotion_policy_mismatch",
    )


def test_run_finalization_pins_artifacts_spend_metrics_and_model_card() -> None:
    dry = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    assert dry.ok and dry.value is not None
    plan, recipe, report = _report("classification")
    manifest = next(item for item in dry.value.manifests if item.task == recipe.task)
    card = render_journey_specialist_model_card(manifest, report)
    card_digest = sha256_digest(card)
    artifacts = {
        "int8": sha256_digest("synthetic-int8-artifact"),
        "onnx": sha256_digest("synthetic-onnx-artifact"),
        "safetensors": sha256_digest("synthetic-safetensors-artifact"),
    }

    completed = finalize_journey_specialist_run(
        manifest,
        report,
        dry.value.ledger,
        actual_cost_usd=50.0,
        actual_gpu_hours=10.0,
        artifact_digests=artifacts,
        model_card_digest=card_digest,
    )

    assert completed.ok and completed.value is not None
    assert completed.value.manifest.promotion_decision == "promote"
    assert completed.value.manifest.metrics_digest == report.digest
    assert completed.value.manifest.failure_slices_digest == (
        report.failure_slices_digest
    )
    assert completed.value.manifest.model_card_digest == card_digest
    assert completed.value.manifest.actual_cost_usd == 50.0
    assert not completed.value.manifest.dry_run
    assert completed.value.ledger.committed_cost_usd == 428.0
    assert completed.value.manifest.ledger_digest == completed.value.ledger.digest
    assert (
        plan.dataset.content_digest == completed.value.manifest.dataset.content_digest
    )
    schema = load_journey_specialist_schema()
    validator = validator_for(schema)(schema)
    assert not tuple(validator.iter_errors(completed.value.manifest.to_dict()))
    assert "case_" not in card
    assert "Synthetic evidence" not in card


def test_run_finalization_refuses_missing_artifacts_and_budget_overrun() -> None:
    dry = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    assert dry.ok and dry.value is not None
    _, recipe, report = _report("classification")
    manifest = next(item for item in dry.value.manifests if item.task == recipe.task)
    missing = finalize_journey_specialist_run(
        manifest,
        report,
        dry.value.ledger,
        actual_cost_usd=50.0,
        actual_gpu_hours=10.0,
        artifact_digests={"onnx": sha256_digest("only-one")},
        model_card_digest=sha256_digest("card"),
    )
    over_budget = finalize_journey_specialist_run(
        manifest,
        report,
        dry.value.ledger,
        actual_cost_usd=1000.0,
        actual_gpu_hours=10.0,
        artifact_digests={
            "int8": sha256_digest("int8"),
            "onnx": sha256_digest("onnx"),
            "safetensors": sha256_digest("safetensors"),
        },
        model_card_digest=sha256_digest("card"),
    )
    matching_entry = next(
        entry for entry in dry.value.ledger.entries if entry.run_id == manifest.run_id
    )
    substituted_ledger = GpuSpendLedger(
        entries=(matching_entry,),
        cap_usd=dry.value.ledger.cap_usd,
    )
    wrong_ledger = finalize_journey_specialist_run(
        manifest,
        report,
        substituted_ledger,
        actual_cost_usd=50.0,
        actual_gpu_hours=10.0,
        artifact_digests={
            "int8": sha256_digest("int8"),
            "onnx": sha256_digest("onnx"),
            "safetensors": sha256_digest("safetensors"),
        },
        model_card_digest=sha256_digest("card"),
    )
    over_hours = finalize_journey_specialist_run(
        manifest,
        report,
        dry.value.ledger,
        actual_cost_usd=50.0,
        actual_gpu_hours=recipe.stop_rule.maximum_gpu_hours + 0.01,
        artifact_digests={
            "int8": sha256_digest("int8"),
            "onnx": sha256_digest("onnx"),
            "safetensors": sha256_digest("safetensors"),
        },
        model_card_digest=sha256_digest("card"),
    )

    assert (missing.state, missing.code) == (
        StoreState.CONFLICT,
        "run_artifact_conflict",
    )
    assert (over_budget.state, over_budget.code) == (
        StoreState.DENIED,
        "gpu_budget_exceeded",
    )
    assert (wrong_ledger.state, wrong_ledger.code) == (
        StoreState.CONFLICT,
        "run_ledger_mismatch",
    )
    assert (over_hours.state, over_hours.code) == (
        StoreState.DENIED,
        "training_stop_rule_exceeded",
    )


def test_held_run_cannot_become_a_model_pack_entry() -> None:
    dry = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    assert dry.ok and dry.value is not None
    _, recipe, report = _report("classification", baseline_macro_f1=1.0)
    manifest = next(item for item in dry.value.manifests if item.task == recipe.task)
    completed = finalize_journey_specialist_run(
        manifest,
        report,
        dry.value.ledger,
        actual_cost_usd=50.0,
        actual_gpu_hours=10.0,
        artifact_digests={
            "int8": sha256_digest("int8"),
            "onnx": sha256_digest("onnx"),
            "safetensors": sha256_digest("safetensors"),
        },
        model_card_digest=sha256_digest("model-card"),
    )
    assert completed.ok and completed.value is not None

    entry = build_journey_specialist_model_pack_entry(completed.value, report)

    assert (entry.state, entry.code) == (
        StoreState.DENIED,
        "specialist_not_promoted",
    )


def test_pack_card_is_reproducible_and_truthful_about_unpromoted_state() -> None:
    plan = load_journey_specialist_pack()
    dry = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    assert dry.ok and dry.value is not None

    first = render_journey_specialist_pack_card(plan, dry.value)
    second = render_journey_specialist_pack_card(plan, dry.value)

    assert first == second
    assert "no trained artifact is promoted" in first
    assert "$468.00" in first
    assert plan.digest in first
    assert plan.dataset.content_digest in first
