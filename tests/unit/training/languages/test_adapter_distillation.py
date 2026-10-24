"""Offline tests for language-family adapter distillation plans and gates."""

from __future__ import annotations

import json
import math
import random
import re
import socket

import pytest

from openmed.eval.tiers import TIERS
from openmed.training.adapters import AdapterMetadata
from openmed.training.distill import (
    compute_kd_loss,
    student_backbone_from_tiny_distill_preset,
)
from openmed.training.languages.adapter_distillation import (
    GATE_CHECKS,
    ROUTE_DONOR_FALLBACK,
    ROUTE_TARGET_ADAPTER,
    TEACHER_ROLE_DONOR,
    TEACHER_ROLE_PARENT,
    AdapterDistillationError,
    AdapterMetricRecord,
    DistillationGatePolicy,
    DistillationSource,
    build_adapter_distillation_plan,
    compute_language_kd_loss,
    ensemble_teacher_soft_labels,
    evaluate_adapter_distillation_gate,
    select_family_adapters,
)
from openmed.training.recipe import load_preset

STUDENT_ID = "openmed-test/indic-edge-student"
EVAL_DIGEST = "sha256:" + "e" * 64
DEVICE = "synthetic-phone-a"
ID2LABEL = {0: "O", 1: "SSN", 2: "PHONE"}


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _adapter(language: str, *, license_name: str = "apache-2.0") -> AdapterMetadata:
    return AdapterMetadata(
        adapter_id=f"openmed-test/indic-{language}",
        license=license_name,
        provenance=f"Synthetic offline {language} adapter fixture",
    )


def _installed() -> dict[str, AdapterMetadata]:
    return {
        "hi": _adapter("hi"),
        "te": _adapter("te"),
        "bn": _adapter("bn"),
        "es": AdapterMetadata(
            adapter_id="openmed-test/romance-es",
            provenance="Synthetic offline es adapter fixture",
        ),
    }


def _sources() -> list[DistillationSource]:
    return [
        DistillationSource(
            source_id="synthetic-indic-notes-v1",
            kind="synthetic",
            license="Apache-2.0",
            content_digest=_digest("a"),
            record_counts={"hi": 900, "te": 300, "bn": 200},
        ),
        DistillationSource(
            source_id="public-indic-ner-v2",
            kind="public",
            license="cc-by-4.0",
            content_digest=_digest("b"),
            record_counts={"hi": 5000},
        ),
    ]


def _plan(**overrides):
    options = {
        "plan_id": "indic-edge-v1",
        "family_id": "indic",
        "available_adapters": _installed(),
        "sources": _sources(),
        "student_id": STUDENT_ID,
    }
    options.update(overrides)
    return build_adapter_distillation_plan(**options)


def _record(model_id: str, language: str, **overrides) -> dict:
    payload = {
        "model_id": model_id,
        "language": language,
        "eval_set_digest": EVAL_DIGEST,
        "device_profile": DEVICE,
        "label_recall": {},
        "f1": 0.9,
        "size_bytes": 300_000_000,
        "peak_memory_mb": 420.0,
        "latency_p50_ms": 80.0,
        "latency_p95_ms": 190.0,
    }
    payload.update(overrides)
    return payload


def _records(plan, *, student=None, parent=None) -> list[dict]:
    """Return a passing synthetic metric set with per-language overrides."""

    student = student or {}
    parent = parent or {}
    rows = []
    for language in plan.languages:
        parent_recall = {label: 0.95 for label in plan.critical_labels}
        parent_recall["PERSON"] = 0.9
        rows.append(
            _record(
                plan.parent_for(language),
                language,
                label_recall=parent_recall,
                **parent.get(language, {}),
            )
        )
        student_values = {
            "label_recall": dict(parent_recall),
            "f1": 0.895,
            "size_bytes": 60_000_000,
            "peak_memory_mb": 150.0,
            "latency_p50_ms": 20.0,
            "latency_p95_ms": 50.0,
        }
        student_values.update(student.get(language, {}))
        rows.append(_record(STUDENT_ID, language, **student_values))
    return rows


def _failed(report) -> list[tuple]:
    return [
        (check.check, check.language, check.label, check.reason)
        for check in report.failed_checks
    ]


class _FakeTokenTagger:
    """Deterministic stand-in for a token classifier: one logit row per token."""

    def __init__(self, model_id: str, table: dict[int, tuple[float, ...]]) -> None:
        self.model_id = model_id
        self._table = table

    def __call__(self, token_ids: list[list[int]]) -> list[list[list[float]]]:
        return [[list(self._table[token]) for token in sample] for sample in token_ids]


# --- language-family adapter selection -------------------------------------


def test_selection_defaults_to_family_languages_with_installed_adapters() -> None:
    selection = select_family_adapters("indic", _installed())

    assert selection.family_id == "indic"
    assert selection.languages == ("bn", "hi", "te")
    assert selection.parent_adapter_ids == (
        "openmed-test/indic-bn",
        "openmed-test/indic-hi",
        "openmed-test/indic-te",
    )
    assert {route.route_mode for route in selection.routes} == {ROUTE_TARGET_ADAPTER}
    assert selection.to_dict()["routes"][0]["backbone_model_id"]


def test_selection_rejects_language_outside_family() -> None:
    with pytest.raises(AdapterDistillationError, match="outside family"):
        select_family_adapters("indic", _installed(), languages=("hi", "es"))


def test_selection_requires_opt_in_for_donor_fallback() -> None:
    with pytest.raises(AdapterDistillationError, match="allow_donor_fallback"):
        select_family_adapters("indic", _installed(), languages=("hi", "mr"))

    selection = select_family_adapters(
        "indic",
        _installed(),
        languages=("mr", "hi"),
        allow_donor_fallback=True,
    )

    marathi = selection.route_for("mr")
    assert marathi.route_mode == ROUTE_DONOR_FALLBACK
    assert marathi.adapter_id == "openmed-test/indic-hi"
    assert marathi.adapter_language == "hi"
    assert selection.parent_adapter_ids == ("openmed-test/indic-hi",)
    assert selection.languages_for_parent("openmed-test/indic-hi") == ("hi", "mr")


def test_selection_rejects_unknown_family_unservable_and_restricted_teachers() -> None:
    with pytest.raises(AdapterDistillationError, match="taxonomy"):
        select_family_adapters("klingon", _installed())
    with pytest.raises(AdapterDistillationError, match="can serve"):
        select_family_adapters(
            "indic", _installed(), languages=("ml",), allow_donor_fallback=True
        )
    restricted = dict(_installed(), te=_adapter("te", license_name="gpl-3.0"))
    with pytest.raises(AdapterDistillationError, match="permissive"):
        select_family_adapters("indic", restricted)


# --- teacher ensemble to student plan --------------------------------------


def test_plan_defaults_come_from_tiny_distill_preset() -> None:
    preset = load_preset("tiny_distill")

    plan = _plan()

    assert plan.student_backbone == student_backbone_from_tiny_distill_preset()
    assert plan.critical_labels == tuple(sorted(preset.loss.critical_labels))
    assert plan.critical_label_weight == preset.loss.critical_label_weight
    assert plan.seed == preset.seed
    assert plan.recipe_preset == "tiny_distill"
    assert plan.gate == DistillationGatePolicy.from_tier("Tiny")
    assert plan.gate.max_peak_memory_mb == TIERS["Tiny"]["ram_mb_max"]
    assert plan.entity_type_weights().weight_for_label("B-SSN") == 3.0
    assert plan.entity_type_weights().weight_for_label("PERSON") == 1.0


def test_plan_is_deterministic_and_path_free(tmp_path) -> None:
    first = _plan()
    second = _plan(
        sources=list(reversed(_sources())),
        available_adapters=dict(reversed(list(_installed().items()))),
    )

    assert first.to_json() == second.to_json()
    assert first.plan_digest == second.plan_digest
    payload = first.to_dict()
    assert payload["plan_digest"] == first.plan_digest
    assert payload["network_accessed"] is False
    assert payload["training_started"] is False
    assert "path" not in first.to_json()
    assert str(tmp_path) not in first.to_json()
    written = first.write_json(tmp_path / "plan" / "indic.json")
    assert json.loads(written.read_text(encoding="utf-8")) == payload


def test_language_sampling_smooths_toward_low_resource_languages() -> None:
    plan = _plan()
    totals = plan.language_record_counts()
    weights = plan.language_sampling_weights()

    assert totals == {"bn": 200, "hi": 5900, "te": 300}
    assert math.isclose(sum(weights.values()), 1.0)
    proportional_te = totals["te"] / sum(totals.values())
    assert weights["te"] > proportional_te
    assert weights["hi"] < totals["hi"] / sum(totals.values())

    shards = plan.shards()
    assert math.isclose(sum(shard.sampling_probability for shard in shards), 1.0)
    assert [(shard.language, shard.source_id) for shard in shards] == [
        ("bn", "synthetic-indic-notes-v1"),
        ("hi", "public-indic-ner-v2"),
        ("hi", "synthetic-indic-notes-v1"),
        ("te", "synthetic-indic-notes-v1"),
    ]
    assert plan.to_dict()["language_sampling"]["synthetic_share"]["bn"] == 1.0

    proportional = _plan(language_sampling_exponent=1.0)
    assert math.isclose(proportional.language_sampling_weights()["te"], proportional_te)


def test_donor_teachers_join_the_ensemble_with_normalized_weights() -> None:
    plan = _plan(donor_teacher_weight=0.5)

    telugu = plan.teachers_for("te")
    assert [(t.adapter_id, t.role) for t in telugu] == [
        ("openmed-test/indic-te", TEACHER_ROLE_PARENT),
        ("openmed-test/indic-hi", TEACHER_ROLE_DONOR),
    ]
    assert [t.weight for t in telugu] == pytest.approx([2 / 3, 1 / 3])
    assert [t.adapter_id for t in plan.teachers_for("hi")] == ["openmed-test/indic-hi"]
    assert plan.parent_for("te") == "openmed-test/indic-te"
    assert _plan().teachers_for("te")[0].weight == 1.0


def test_sources_refuse_restricted_corpora_and_non_open_licenses() -> None:
    def source(**overrides) -> DistillationSource:
        options = {
            "source_id": "public-indic-ner-v2",
            "kind": "public",
            "license": "MIT",
            "content_digest": _digest("c"),
            "record_counts": {"hi": 10},
        }
        options.update(overrides)
        return DistillationSource(**options)

    for restricted in ("mimic-iv-notes", "n2c2-2018-hi", "umls_terms", "i2b2"):
        with pytest.raises(AdapterDistillationError, match="restricted"):
            source(source_id=restricted)
    for license_name in ("GPL-3.0", "CC-BY-SA-4.0", "proprietary", None):
        with pytest.raises(AdapterDistillationError, match="license"):
            source(license=license_name)
    with pytest.raises(AdapterDistillationError, match="kind"):
        source(kind="clinical-notes")
    with pytest.raises(AdapterDistillationError, match="digest"):
        source(content_digest="md5:1234")
    with pytest.raises(AdapterDistillationError, match="between 1"):
        source(record_counts={"hi": 0})
    with pytest.raises(AdapterDistillationError, match="identifier"):
        source(source_id="https://example.org/corpus")
    assert source(license="cc0-1.0").license == "CC0-1.0"

    with pytest.raises(AdapterDistillationError) as error:
        source(source_id="Jane Roe 123-45-6789 export")
    assert "Jane" not in str(error.value)
    assert "6789" not in str(error.value)


def test_plan_requires_synthetic_coverage_for_every_language() -> None:
    sources = [
        DistillationSource(
            source_id="synthetic-indic-notes-v1",
            kind="synthetic",
            license="Apache-2.0",
            content_digest=_digest("a"),
            record_counts={"hi": 900, "te": 300},
        ),
        DistillationSource(
            source_id="public-bengali-ner",
            kind="public",
            license="MIT",
            content_digest=_digest("b"),
            record_counts={"bn": 4000},
        ),
    ]

    with pytest.raises(AdapterDistillationError, match="'bn' needs synthetic"):
        _plan(sources=sources)


def test_plan_rejects_foreign_languages_reused_ids_and_bad_hyperparameters() -> None:
    foreign = _sources() + [
        DistillationSource(
            source_id="synthetic-romance-v1",
            kind="synthetic",
            license="Apache-2.0",
            content_digest=_digest("d"),
            record_counts={"es": 100},
        )
    ]
    with pytest.raises(AdapterDistillationError, match="outside the plan"):
        _plan(sources=foreign)
    with pytest.raises(AdapterDistillationError, match="differ from every parent"):
        _plan(student_id="openmed-test/indic-hi")
    with pytest.raises(AdapterDistillationError, match="unique"):
        _plan(sources=_sources() + _sources()[:1])
    with pytest.raises(AdapterDistillationError, match="at least 1"):
        _plan(critical_label_weight=0.5)
    with pytest.raises(AdapterDistillationError, match="alpha"):
        _plan(alpha=1.5)
    with pytest.raises(AdapterDistillationError, match="student_license"):
        _plan(student_license="gpl-3.0")
    with pytest.raises(AdapterDistillationError, match="student_backbone"):
        _plan(student_backbone="/Users/someone/models/student")


# --- per-language teacher-ensemble KD ---------------------------------------


def test_language_kd_loss_distills_from_the_planned_teacher_ensemble() -> None:
    plan = _plan(donor_teacher_weight=1.0)
    telugu_teacher = _FakeTokenTagger(
        "openmed-test/indic-te", {0: (3.0, 0.0, 0.0), 1: (0.0, 4.0, 0.5)}
    )
    hindi_donor = _FakeTokenTagger(
        "openmed-test/indic-hi", {0: (2.0, 0.5, 0.0), 1: (0.0, 2.0, 1.0)}
    )
    tokens = [[0, 1, 0]]
    hard_labels = [[0, 1, 0]]
    teacher_logits = {
        teacher.model_id: teacher(tokens) for teacher in (telugu_teacher, hindi_donor)
    }

    ensemble = ensemble_teacher_soft_labels(
        teacher_logits,
        weights={"openmed-test/indic-te": 0.5, "openmed-test/indic-hi": 0.5},
        temperature=plan.temperature,
    )
    for row in ensemble[0]:
        assert math.isclose(sum(row), 1.0)
    matching_student = [
        [[plan.temperature * math.log(p) for p in row] for row in ensemble[0]]
    ]
    uniform_student = [[[0.0, 0.0, 0.0] for _ in tokens[0]]]

    matched = compute_language_kd_loss(
        plan,
        language="te",
        student_logits=matching_student,
        hard_labels=hard_labels,
        teacher_logits=teacher_logits,
        id2label=ID2LABEL,
    )
    uniform = compute_language_kd_loss(
        plan,
        language="te",
        student_logits=uniform_student,
        hard_labels=hard_labels,
        teacher_logits=teacher_logits,
        id2label=ID2LABEL,
    )

    assert matched.soft_kl == pytest.approx(0.0, abs=1e-9)
    assert uniform.soft_kl > 0.0
    assert uniform.total > matched.total
    direct = compute_kd_loss(
        student_logits=uniform_student,
        hard_labels=hard_labels,
        teacher_soft_labels=ensemble,
        temperature=plan.temperature,
        alpha=plan.alpha,
        span_loss_weight=plan.span_loss_weight,
        entity_type_weights=plan.entity_type_weights(),
        id2label=ID2LABEL,
    )
    assert uniform.to_dict() == direct.to_dict()
    unweighted = compute_kd_loss(
        student_logits=uniform_student,
        hard_labels=hard_labels,
        teacher_soft_labels=ensemble,
        temperature=plan.temperature,
        alpha=plan.alpha,
        id2label=ID2LABEL,
    )
    assert uniform.soft_kl != pytest.approx(unweighted.soft_kl)


def test_language_kd_loss_refuses_foreign_or_missing_teachers() -> None:
    plan = _plan()
    hindi = _FakeTokenTagger("openmed-test/indic-hi", {0: (1.0, 0.0, 0.0)})
    telugu = _FakeTokenTagger("openmed-test/indic-te", {0: (1.0, 0.0, 0.0)})
    batch = {"student_logits": [[[0.0, 0.0, 0.0]]], "hard_labels": [[0]]}

    with pytest.raises(AdapterDistillationError, match="planned teachers"):
        compute_language_kd_loss(
            plan,
            language="hi",
            teacher_logits={telugu.model_id: telugu([[0]])},
            **batch,
        )
    with pytest.raises(AdapterDistillationError, match="planned teachers"):
        compute_language_kd_loss(
            plan,
            language="hi",
            teacher_logits={
                hindi.model_id: hindi([[0]]),
                telugu.model_id: telugu([[0]]),
            },
            **batch,
        )
    with pytest.raises(AdapterDistillationError, match="not part of the plan"):
        compute_language_kd_loss(
            plan, language="mr", teacher_logits={hindi.model_id: hindi([[0]])}, **batch
        )


def test_ensemble_soft_labels_validate_shapes_and_weights() -> None:
    narrow = {"a": [[[1.0, 0.0]]], "b": [[[1.0, 0.0, 0.0]]]}
    with pytest.raises(AdapterDistillationError, match="shape"):
        ensemble_teacher_soft_labels(narrow, weights={"a": 1, "b": 1}, temperature=1)
    same = {"a": [[[1.0, 0.0]]], "b": [[[0.0, 1.0]]]}
    with pytest.raises(AdapterDistillationError, match="cover exactly"):
        ensemble_teacher_soft_labels(same, weights={"a": 1}, temperature=1)
    with pytest.raises(AdapterDistillationError, match="all be zero"):
        ensemble_teacher_soft_labels(same, weights={"a": 0, "b": 0}, temperature=1)
    averaged = ensemble_teacher_soft_labels(
        same, weights={"a": 3, "b": 1}, temperature=1
    )
    first = math.exp(1.0) / (math.exp(1.0) + 1.0)
    assert averaged[0][0][0] == pytest.approx(0.75 * first + 0.25 * (1 - first))


# --- promotion gate ----------------------------------------------------------


def test_gate_promotes_student_that_holds_recall_and_fits_budgets() -> None:
    plan = _plan()

    report = evaluate_adapter_distillation_gate(plan, _records(plan))

    assert report.promoted is True
    assert report.blocking_reasons == ()
    payload = report.to_dict()
    assert payload["decision"] == "promote"
    assert payload["plan_digest"] == plan.plan_digest
    assert payload["tier"] == "Tiny"
    checks = {check.check for check in report.checks}
    assert checks == set(GATE_CHECKS) - {"measurement_integrity"}
    per_language_recall = [
        c for c in report.checks if c.check == "critical_label_recall"
    ]
    assert len(per_language_recall) == len(plan.languages) * len(plan.critical_labels)
    assert [language for language, _ in report.evidence] == list(plan.languages)
    assert all(evidence.recall_gate_passed for _, evidence in report.evidence)
    assert report.evidence[0][1].teacher_id == "openmed-test/indic-bn"


def test_smaller_faster_student_losing_critical_recall_in_one_language_is_rejected():
    plan = _plan()
    excellent = {label: 1.0 for label in plan.critical_labels} | {"PERSON": 1.0}
    student = {
        language: {
            "label_recall": dict(excellent),
            "f1": 0.97,
            "size_bytes": 20_000_000,
            "latency_p50_ms": 5.0,
            "latency_p95_ms": 12.0,
        }
        for language in plan.languages
    }
    student["te"]["label_recall"]["SSN"] = 0.80

    report = evaluate_adapter_distillation_gate(plan, _records(plan, student=student))

    assert report.promoted is False
    assert report.to_dict()["decision"] == "reject"
    assert _failed(report) == [
        ("critical_label_recall", "te", "SSN", "critical_recall_drop")
    ]
    assert report.blocking_reasons == ("critical_recall_drop",)
    size_and_latency = [
        check
        for check in report.checks
        if check.check
        in {"artifact_size_vs_parent", "latency_p95_vs_parent", "latency_p95_budget"}
    ]
    assert size_and_latency and all(check.passed for check in size_and_latency)
    telugu_evidence = dict(report.evidence)["te"]
    assert telugu_evidence.critical_label_drops == ("SSN",)


def test_critical_recall_is_non_compensable_across_languages_and_labels() -> None:
    plan = _plan()
    parent_rows = _records(plan)
    student = {
        language: {"label_recall": {label: 1.0 for label in plan.critical_labels}}
        for language in plan.languages
    }
    student["bn"]["label_recall"]["EMAIL"] = 0.94

    rows = _records(plan, student=student)
    report = evaluate_adapter_distillation_gate(plan, rows)

    def mean_recall(model_rows) -> float:
        values = [
            value
            for row in model_rows
            for label, value in row["label_recall"].items()
            if label in plan.critical_labels
        ]
        return sum(values) / len(values)

    student_rows = [row for row in rows if row["model_id"] == STUDENT_ID]
    parents = [row for row in parent_rows if row["model_id"] != STUDENT_ID]
    assert mean_recall(student_rows) > mean_recall(parents)
    assert report.promoted is False
    assert _failed(report) == [
        ("critical_label_recall", "bn", "EMAIL", "critical_recall_drop")
    ]


def test_critical_recall_tolerance_is_pinned_in_the_plan() -> None:
    plan = _plan(gate=DistillationGatePolicy.from_tier(critical_recall_tolerance=0.05))
    student = {"te": {"label_recall": {label: 0.92 for label in plan.critical_labels}}}

    report = evaluate_adapter_distillation_gate(plan, _records(plan, student=student))

    assert report.promoted is True
    assert plan.to_dict()["gate"]["critical_recall_tolerance"] == 0.05
    assert plan.plan_digest != _plan().plan_digest


def test_gate_rejects_cross_language_interference() -> None:
    plan = _plan()
    student = {"bn": {"f1": 0.85}, "hi": {"f1": 0.88}}

    report = evaluate_adapter_distillation_gate(plan, _records(plan, student=student))

    assert _failed(report) == [
        ("cross_language_interference", "bn", None, "f1_drop_exceeds_budget")
    ]
    hindi = next(
        check
        for check in report.checks
        if check.check == "cross_language_interference" and check.language == "hi"
    )
    assert hindi.passed is True
    assert hindi.observed == pytest.approx(-0.02)
    assert hindi.threshold == -0.02


def test_gate_rejects_size_memory_and_latency_budget_violations() -> None:
    plan = _plan()
    student = {language: {"size_bytes": 310_000_000} for language in plan.languages}
    student["hi"].update(
        peak_memory_mb=360.0, latency_p50_ms=70.0, latency_p95_ms=140.0
    )
    student["te"].update(latency_p95_ms=160.0)
    parent = {"bn": {"latency_p50_ms": 20.0, "latency_p95_ms": 40.0}}

    report = evaluate_adapter_distillation_gate(
        plan, _records(plan, student=student, parent=parent)
    )

    assert _failed(report) == [
        ("artifact_size_vs_parent", None, None, "size_exceeds_parent"),
        ("artifact_size_vs_parent", None, None, "size_exceeds_parent"),
        ("artifact_size_vs_parent", None, None, "size_exceeds_parent"),
        ("latency_p95_vs_parent", "bn", None, "latency_exceeds_parent"),
        ("peak_memory_budget", "hi", None, "memory_exceeds_tier_budget"),
        ("latency_p50_budget", "hi", None, "latency_exceeds_tier_budget"),
        ("latency_p95_budget", "te", None, "latency_exceeds_tier_budget"),
    ]
    assert report.blocking_reasons == (
        "latency_exceeds_parent",
        "latency_exceeds_tier_budget",
        "memory_exceeds_tier_budget",
        "size_exceeds_parent",
    )


def test_gate_fails_closed_on_missing_or_incomparable_measurements() -> None:
    plan = _plan()
    rows = _records(plan, student={"bn": {"eval_set_digest": _digest("f")}})
    rows = [
        row
        for row in rows
        if not (row["model_id"] == STUDENT_ID and row["language"] == "te")
    ]
    hindi_student = next(
        row for row in rows if row["model_id"] == STUDENT_ID and row["language"] == "hi"
    )
    del hindi_student["label_recall"]["SSN"]

    report = evaluate_adapter_distillation_gate(plan, rows)

    assert report.promoted is False
    assert _failed(report) == [
        ("measurement_integrity", "bn", None, "eval_set_mismatch"),
        ("critical_label_recall", "hi", "SSN", "critical_label_unmeasured"),
        ("measurement_integrity", "te", None, "missing_student_record"),
    ]

    mixed = _records(plan, student={"hi": {"device_profile": "synthetic-laptop-b"}})
    mixed_report = evaluate_adapter_distillation_gate(plan, mixed)
    assert _failed(mixed_report) == [
        ("measurement_integrity", None, None, "device_profile_mismatch")
    ]

    empty = evaluate_adapter_distillation_gate(plan, [])
    assert empty.promoted is False
    assert set(empty.blocking_reasons) == {
        "missing_parent_record",
        "missing_student_record",
    }


def test_gate_flags_inconsistent_size_measurements() -> None:
    plan = _plan()
    rows = _records(plan, student={"hi": {"size_bytes": 61_000_000}})

    report = evaluate_adapter_distillation_gate(plan, rows)

    assert _failed(report) == [
        ("measurement_integrity", None, None, "inconsistent_size_measurement")
    ]


def test_gate_rejects_malformed_duplicated_or_foreign_records() -> None:
    plan = _plan()
    good = _records(plan)

    with pytest.raises(AdapterDistillationError, match="outside the plan"):
        evaluate_adapter_distillation_gate(
            plan, good + [_record("openmed-test/other-model", "hi")]
        )
    with pytest.raises(AdapterDistillationError, match="duplicate"):
        evaluate_adapter_distillation_gate(plan, good + good[:1])
    with pytest.raises(AdapterDistillationError, match="language is outside"):
        evaluate_adapter_distillation_gate(plan, good + [_record(STUDENT_ID, "mr")])
    with pytest.raises(AdapterDistillationError, match="does not serve"):
        evaluate_adapter_distillation_gate(
            plan, good + [_record("openmed-test/indic-hi", "te")]
        )
    with pytest.raises(AdapterDistillationError) as unexpected:
        evaluate_adapter_distillation_gate(
            plan, [dict(good[0], note_text="Jane Roe 123-45-6789")]
        )
    assert "Jane" not in str(unexpected.value)
    assert "note_text" not in str(unexpected.value)
    for field, value in (
        ("f1", float("nan")),
        ("f1", 1.5),
        ("size_bytes", True),
        ("latency_p95_ms", 10.0),
        ("eval_set_digest", "not-a-digest"),
        ("label_recall", {"SSN": -0.1}),
        ("label_recall", {"ssn number": 0.9}),
    ):
        with pytest.raises(AdapterDistillationError):
            AdapterMetricRecord.from_dict(dict(good[0], **{field: value}))
    with pytest.raises(AdapterDistillationError, match="missing"):
        AdapterMetricRecord.from_dict({"model_id": STUDENT_ID})


def test_gate_report_is_deterministic_and_contains_no_raw_values(tmp_path) -> None:
    plan = _plan()
    rows = _records(plan, student={"te": {"label_recall": {"SSN": 0.5}}})
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)

    first = evaluate_adapter_distillation_gate(plan, rows)
    second = evaluate_adapter_distillation_gate(
        plan, [AdapterMetricRecord.from_dict(row) for row in shuffled]
    )

    assert first.to_json() == second.to_json()
    text = first.to_json()
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", text)
    assert not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    for check in first.to_dict()["checks"]:
        assert set(check) == {
            "check",
            "label",
            "language",
            "observed",
            "parent_adapter_id",
            "passed",
            "reason",
            "threshold",
        }
    written = first.write_json(tmp_path / "gate.json")
    assert json.loads(written.read_text(encoding="utf-8")) == first.to_dict()


def test_plan_and_gate_make_no_network_calls(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    plan = _plan(donor_teacher_weight=0.5)
    report = evaluate_adapter_distillation_gate(plan, _records(plan))

    assert report.promoted is True
    assert report.to_dict()["network_accessed"] is False


def test_gate_policy_budgets_come_from_the_tier_table() -> None:
    tiny = DistillationGatePolicy.from_tier("tiny")
    nano = DistillationGatePolicy.from_tier("Tiny", sub_tier="nano")

    assert tiny.tier == "Tiny"
    assert tiny.max_peak_memory_mb == TIERS["Tiny"]["ram_mb_max"]
    assert tiny.max_latency_p50_ms == TIERS["Tiny"]["p50_ms_max"]
    assert tiny.max_latency_p95_ms == TIERS["Tiny"]["p95_ms_max"]
    assert nano.tier == "Tiny/Nano"
    assert nano.max_peak_memory_mb == 150.0
    with pytest.raises(AdapterDistillationError, match="tier"):
        DistillationGatePolicy.from_tier("Watch")
    with pytest.raises(AdapterDistillationError, match="sub_tier"):
        DistillationGatePolicy.from_tier("Base", sub_tier="Nano")
    with pytest.raises(AdapterDistillationError, match="max_f1_drop"):
        DistillationGatePolicy.from_tier(max_f1_drop=-0.1)
    with pytest.raises(AdapterDistillationError, match="max_size_ratio"):
        DistillationGatePolicy.from_tier(max_size_ratio=float("inf"))
