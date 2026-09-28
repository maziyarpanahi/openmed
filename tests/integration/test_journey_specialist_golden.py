"""Golden Journey coverage for training, evaluation, promotion, and routing."""

from __future__ import annotations

from pathlib import Path

from openmed.clinical.journey_contracts import sha256_digest
from openmed.clinical.model_packs import (
    CalibrationMetadata,
    ClinicalTaskRequest,
    ClinicalTaskRouter,
    FallbackPolicy,
    LocalArtifactBinding,
    ModelPackEntry,
    ModelPackManifest,
    QuantizationMetadata,
)
from openmed.eval.suites.journey_specialist import (
    SpecialistPrediction,
    SpecialistPromotionPolicy,
    build_journey_specialist_model_pack_entry,
    evaluate_journey_specialist_holdout,
    finalize_journey_specialist_run,
    render_journey_specialist_model_card,
)
from openmed.training.journey_specialist import (
    dry_run_journey_specialist_pack,
    load_journey_specialist_pack,
)

CODE_REVISION = "a" * 40
CALIBRATION_DIGEST = sha256_digest("synthetic-frozen-calibration")
HOLDOUT_DIGEST = sha256_digest("synthetic-frozen-holdout")


def test_promoted_specialist_routes_through_the_offline_golden_journey(
    tmp_path: Path,
) -> None:
    """Carry measured evidence into a digest-checked local routing decision."""

    dry = dry_run_journey_specialist_pack(code_revision=CODE_REVISION)
    assert dry.ok and dry.value is not None
    plan = load_journey_specialist_pack()
    recipe = plan.recipe_for("classification")
    predictions = []
    for index, gold_label in enumerate(recipe.labels * 4):
        other_probability = 0.04 / (len(recipe.labels) - 1)
        scores = {label: other_probability for label in recipe.labels}
        scores[gold_label] = 0.96
        predictions.append(
            SpecialistPrediction(
                example_id=f"golden_{index:04d}",
                task=recipe.task,
                gold_label=gold_label,
                scores=scores,
                quantized_scores=scores,
                subgroups={"site": "synthetic"},
            )
        )
    policy = SpecialistPromotionPolicy.from_recipe(
        recipe,
        baseline_macro_f1=0.5,
        candidate_alias="specialist.classification",
        fallback_alias="rules.classification",
    )
    evaluated = evaluate_journey_specialist_holdout(
        predictions,
        recipe=recipe,
        policy=policy,
        dataset_digest=plan.dataset.content_digest,
        calibration_digest=CALIBRATION_DIGEST,
        holdout_digest=HOLDOUT_DIGEST,
    )
    assert evaluated.ok and evaluated.value is not None
    report = evaluated.value
    assert report.promotion_decision == "promote"

    dry_manifest = next(
        item for item in dry.value.manifests if item.task == recipe.task
    )
    model_card = render_journey_specialist_model_card(dry_manifest, report)
    artifact_bytes = b"synthetic-int8-artifact"
    artifacts = {
        "int8": sha256_digest(artifact_bytes),
        "onnx": sha256_digest("synthetic-onnx-artifact"),
        "safetensors": sha256_digest("synthetic-safetensors-artifact"),
    }
    completed = finalize_journey_specialist_run(
        dry_manifest,
        report,
        dry.value.ledger,
        actual_cost_usd=50.0,
        actual_gpu_hours=10.0,
        artifact_digests=artifacts,
        model_card_digest=sha256_digest(model_card),
    )
    assert completed.ok and completed.value is not None
    entry = build_journey_specialist_model_pack_entry(completed.value, report)
    assert entry.ok and entry.value is not None
    candidate = entry.value

    fallback_digest = sha256_digest("classification-rules")
    fallback = ModelPackEntry(
        alias=policy.fallback_alias,
        task=recipe.task,
        family="deterministic_rules",
        artifact_id="builtin/rules.classification",
        artifact_kind="builtin",
        revision="builtin-v1",
        artifact_digest=fallback_digest,
        license="apache-2.0",
        runtime="builtin",
        model_kind="deterministic",
        output_schema=recipe.output_schema,
        output_schema_version="1.0.0",
        languages=("en",),
        domains=("clinical",),
        calibration=CalibrationMetadata(method="none"),
        quantization=QuantizationMetadata(mode="none"),
        fallback=FallbackPolicy(),
        priority=900,
    )
    pack = ModelPackManifest(
        pack_id="openmed.journey.runtime.v1",
        entries=(candidate, fallback),
    )
    artifact_path = tmp_path / "classification.int8"
    artifact_path.write_bytes(artifact_bytes)
    router = ClinicalTaskRouter(
        pack,
        bindings=(
            LocalArtifactBinding(
                alias=candidate.alias,
                runtime="onnx",
                path=artifact_path,
            ),
            LocalArtifactBinding(
                alias=fallback.alias,
                runtime="builtin",
                builtin_id=fallback.artifact_id,
                builtin_digest=fallback.artifact_digest,
            ),
        ),
        available_runtimes=("onnx", "builtin"),
    )

    route = router.route(
        ClinicalTaskRequest(
            task=recipe.task,
            output_schema=recipe.output_schema,
            output_schema_version="1.0.0",
            language="en",
            domain="clinical",
            requested_alias=candidate.alias,
        )
    )

    assert route.ok and route.value is not None
    assert route.value.alias == candidate.alias
    assert route.value.artifact_digest == artifacts["int8"]
    assert candidate.calibration.calibration_dataset_digest == CALIBRATION_DIGEST
    assert candidate.calibration.holdout_dataset_digest == HOLDOUT_DIGEST
    assert candidate.quantization.evaluation_digest == report.digest
