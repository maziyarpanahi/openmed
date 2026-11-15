"""Synthetic tests for the OpenMedKit real-device qualification harness.

Every plan, evidence record and digest here is synthetic. Negative controls
cover missing targets, stale digests, simulator-only evidence, identifier
leakage and each deliberately failed gate, not only the happy path.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from openmed.eval import kit_qualification as kq
from openmed.eval.kit_support_matrix import (
    identifier_shaped,
    load_kit_support_matrix,
    validate_kit_support_matrix,
)
from openmed.eval.suites import edge_health
from openmed.eval.suites.openphi_vision import ReleaseProvenance, VisionGateError

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "kit_qualification_report.py"
SPEC = importlib.util.spec_from_file_location("kit_qualification_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)

NNAPI = "android-arm64-onnx-nnapi"
IOS_COREML = "swift-ios-coreml"
SIM_COREML = "swift-ios-simulator-coreml"
MAC_MLX = "swift-macos-mlx"
NARROWED = (NNAPI, IOS_COREML, SIM_COREML, MAC_MLX)

HARNESS = "4" * 64
FIXTURE = "2" * 64
POLICY = "3" * 64
BUNDLES = {
    "bert-int8-onnx-android": ("onnx-android", "1" * 64),
    "bert-int8-coreml": ("coreml", "5" * 64),
    "bert-int8-mlx": ("mlx", "6" * 64),
}
BUNDLE_FOR = {
    NNAPI: "bert-int8-onnx-android",
    IOS_COREML: "bert-int8-coreml",
    SIM_COREML: "bert-int8-coreml",
    MAC_MLX: "bert-int8-mlx",
}
AXES: dict[str, dict[str, Any]] = {
    NNAPI: {
        "device_class": "android-device",
        "abi": "arm64-v8a",
        "platform": {"name": "android", "version": "34"},
        "runtime": {"pin": "onnxruntime-android", "version": "1.25.1"},
        "provider": "nnapi",
    },
    IOS_COREML: {
        "device_class": "iphone-ipad",
        "abi": "arm64",
        "platform": {"name": "ios", "version": "17.5"},
        "runtime": {"pin": "coreml-system", "version": "17.5"},
        "provider": "coreml",
    },
    SIM_COREML: {
        "device_class": "ios-simulator",
        "abi": "arm64",
        "platform": {"name": "ios", "version": "17.5"},
        "runtime": {"pin": "coreml-system", "version": "17.5"},
        "provider": "coreml",
    },
    MAC_MLX: {
        "device_class": "apple-silicon-mac",
        "abi": "arm64",
        "platform": {"name": "macos", "version": "15.1"},
        "runtime": {"pin": "mlx-swift", "version": "0.31.3"},
        "provider": "metal",
    },
}
GATES = {
    "minimum_direct_identifier_recall": 0.95,
    "maximum_cold_latency_ms": 5000,
    "maximum_warm_p95_latency_ms": 150,
    "maximum_sustained_p95_latency_ms": 200,
    "minimum_sustained_seconds": 300,
    "maximum_cancel_latency_ms": 250,
    "maximum_thermal_level": 2,
    "require_thermal": True,
    "require_measured_energy": False,
}
MIB = 1024 * 1024


def _rows(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["cell"]: row for row in kq.qualification_cells(matrix)}


def _cell(matrix: dict[str, Any], cell_id: str) -> dict[str, Any]:
    return next(cell for cell in matrix["cells"] if cell["id"] == cell_id)


@pytest.fixture
def matrix() -> dict[str, Any]:
    """The committed matrix with four cells narrowed to one bundle each."""
    document = copy.deepcopy(load_kit_support_matrix())
    for cell_id in NARROWED:
        cell = _cell(document, cell_id)
        cell["families"] = ["bert"]
        cell["precisions"] = ["int8"]
    assert validate_kit_support_matrix(document) == ()
    return document


def make_plan(matrix: dict[str, Any], *cells: str) -> dict[str, Any]:
    rows = _rows(matrix)
    entries = []
    for cell_id in cells or (NNAPI,):
        entry: dict[str, Any] = {
            "cell": cell_id,
            "cell_digest": rows[cell_id]["cell_digest"],
            "bundles": [BUNDLE_FOR[cell_id]],
        }
        if rows[cell_id]["device_memory_required"]:
            entry["gates"] = {"maximum_device_memory_mb": 256}
        entries.append(entry)
    return {
        "schema": kq.PLAN_SCHEMA,
        "release": matrix["release"],
        "harness_digest": HARNESS,
        "bundles": [
            {
                "id": bundle_id,
                "family": "bert",
                "artifact": artifact,
                "precision": "int8",
                "digest": digest,
            }
            for bundle_id, (artifact, digest) in BUNDLES.items()
        ],
        "fixtures": [{"id": "synthetic-notes", "digest": FIXTURE}],
        "policies": [{"id": "mask-default", "digest": POLICY}],
        "gates": dict(GATES),
        "cells": entries,
    }


def passing_metrics() -> dict[str, Any]:
    return {
        "direct_identifier_recall": {
            "status": "measured",
            "detected": 99,
            "expected": 100,
        },
        "leakage": {"status": "measured", "assessed": 100, "critical": 0, "any_phi": 0},
        "cold_latency_ms": {"status": "measured", "value": 900.0},
        "warm_latency_ms": {"status": "measured", "n": 20, "p50": 40.0, "p95": 60.0},
        "sustained_latency_ms": {
            "status": "measured",
            "n": 500,
            "p95": 80.0,
            "duration_s": 600.0,
        },
        "cancel_latency_ms": {"status": "measured", "n": 5, "p95": 30.0},
        "memory_peak_bytes": {
            "status": "measured",
            "value": 200 * MIB,
            "scope": "process_lifetime_highwater",
        },
        "device_memory_peak_bytes": {
            "status": "measured",
            "value": 100 * MIB,
            "scope": "allocator_peak",
        },
        "thermal": {"status": "measured", "max_level": 1},
        "energy": {"status": "proxy", "proxy": "process_cpu_seconds", "value": 12.5},
        "provider_fallback": {"status": "measured", "observed": False},
    }


def make_record(
    planned: str = NNAPI, /, *, execution: str = "physical-device", **overrides: Any
) -> dict[str, Any]:
    """Return a passing record for a planned cell; ``cell=`` may override it."""
    axes = copy.deepcopy(AXES[planned])
    bundle_id = BUNDLE_FOR[planned]
    record: dict[str, Any] = {
        "schema": kq.EVIDENCE_SCHEMA,
        "recorded_on": "2026-11-08",
        "cell": planned,
        "execution": execution,
        **axes,
        "bundle": {"id": bundle_id, "digest": BUNDLES[bundle_id][1]},
        "precision": "int8",
        "fixture": {"id": "synthetic-notes", "digest": FIXTURE},
        "policy": {"id": "mask-default", "digest": POLICY},
        "harness_digest": HARNESS,
        "run": {"status": "completed", "failures": []},
        "metrics": passing_metrics(),
    }
    record.update(overrides)
    return record


def simulator_record(cell: str = IOS_COREML, **overrides: Any) -> dict[str, Any]:
    return make_record(
        cell, execution="simulator", device_class="ios-simulator", **overrides
    )


def _target(report: dict[str, Any], cell: str) -> dict[str, Any]:
    return next(row for row in report["targets"] if row["cell"] == cell)


def _cell_row(report: dict[str, Any], cell: str) -> dict[str, Any]:
    return next(row for row in report["cells"] if row["cell"] == cell)


def _gate(record_row: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in record_row["gates"] if row["gate"] == name)


# --- enumeration and plans --------------------------------------------------


def test_matrix_cells_enumerate_with_mobile_edge_health_tiers() -> None:
    committed = load_kit_support_matrix()
    rows = _rows(committed)
    inference_pins = {
        pin["id"] for pin in committed["runtime_pins"] if pin["kind"] == "inference"
    }
    expected = [
        cell["id"] for cell in committed["cells"] if cell["runtime"] in inference_pins
    ]
    assert list(rows) == expected
    assert "key-store-ios-keychain" not in rows
    assert "document-intake-android-mlkit-latin" not in rows
    tiers = {cell: row["edge_health_tier"] for cell, row in rows.items()}
    assert tiers["android-arm64-onnx-cpu"] == "android_ort_cpu"
    assert tiers[NNAPI] == "android_ort_nnapi"
    assert tiers["android-arm64-onnx-qnn"] == "android_ort_qnn"
    assert tiers["android-emulator-onnx-cpu"] == "android_ort_cpu"
    assert tiers["swift-ios-mlx"] == "ios_device_mlx"
    assert tiers[IOS_COREML] == "ios_device_core_ml"
    assert tiers["swift-watchos-coreml-nano"] == "ios_device_core_ml"
    assert tiers["flutter-ios-onnx-cpu"] == "ios_device_cpu"
    assert tiers[SIM_COREML] == "core_ml"
    assert tiers[MAC_MLX] == "mlx"
    assert tiers["web-chromium-webgpu"] == "browser"
    assert tiers["python-linux-torch-cpu"] == "cpu"
    assert rows[NNAPI]["device_memory_required"]
    assert not rows["android-arm64-onnx-cpu"]["device_memory_required"]
    assert rows[SIM_COREML]["virtual_device_class"]
    assert rows[NNAPI]["ram_ceiling_mb"] == 350
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", row["cell_digest"]) for row in rows.values()
    )


def test_plan_skeleton_lists_every_cell_and_stays_invalid_until_filled() -> None:
    committed = load_kit_support_matrix()
    skeleton = kq.plan_skeleton(committed)

    assert [entry["cell"] for entry in skeleton["cells"]] == list(_rows(committed))
    assert set(skeleton["gates"]) == set(kq.REQUIRED_GATES)
    assert all(value is None for value in skeleton["gates"].values())
    problems = kq.validate_qualification_plan(skeleton, committed)
    assert "plan_bundle_empty" in problems
    assert "plan_fixture_empty" in problems
    assert "plan_gate_invalid:plan:require_thermal" in problems


def test_cell_digest_tracks_runtime_pins_but_not_status(
    matrix: dict[str, Any],
) -> None:
    before = kq.cell_digest(_cell(matrix, NNAPI), matrix)
    promoted = copy.deepcopy(matrix)
    _cell(promoted, NNAPI)["note"] = "Edited note."
    _cell(promoted, NNAPI)["status"] = "supported"
    assert kq.cell_digest(_cell(promoted, NNAPI), promoted) == before

    bumped = copy.deepcopy(matrix)
    pin = next(p for p in bumped["runtime_pins"] if p["id"] == "onnxruntime-android")
    pin["version"] = "1.26.0"
    assert kq.cell_digest(_cell(bumped, NNAPI), bumped) != before


def test_valid_plan_expands_every_target_axis(matrix: dict[str, Any]) -> None:
    plan = make_plan(matrix, NNAPI)
    plan["fixtures"].append({"id": "synthetic-notes-long", "digest": "7" * 64})
    plan["policies"].append({"id": "redact-strict", "digest": "8" * 64})
    plan["cells"][0]["platform_versions"] = ["29", "34"]

    assert kq.validate_qualification_plan(plan, matrix) == ()
    targets = kq.qualification_targets(plan, matrix)
    assert len(targets) == 8
    assert {target["platform_version"] for target in targets} == {"29", "34"}
    assert {(t["fixture"], t["policy"]) for t in targets} == {
        ("synthetic-notes", "mask-default"),
        ("synthetic-notes", "redact-strict"),
        ("synthetic-notes-long", "mask-default"),
        ("synthetic-notes-long", "redact-strict"),
    }
    assert {target["precision"] for target in targets} == {"int8"}
    assert kq.effective_gates(plan, NNAPI)["maximum_device_memory_mb"] == 256


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(plan: dict[str, Any]) -> None:
        target: Any = plan
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _drop(path: tuple[Any, ...]) -> Callable[[dict[str, Any]], None]:
    def mutate(plan: dict[str, Any]) -> None:
        target: Any = plan
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_set(("cells", 0, "cell"), "no-such-cell"), "plan_cell_unknown:0"),
        (_set(("cells", 0, "cell"), "android-armeabi-v7a"), "plan_cell_unknown:0"),
        (
            _set(("cells", 0, "cell"), "key-store-ios-keychain"),
            "plan_cell_out_of_scope:key-store-ios-keychain",
        ),
        (
            _set(("cells", 0, "bundles"), ["bert-int8-coreml"]),
            f"plan_bundle_mismatch:{NNAPI}:bert-int8-coreml",
        ),
        (
            _set(("cells", 0, "gates", "maximum_memory_peak_mb"), 351),
            f"plan_gate_looser_than_tier:{NNAPI}:maximum_memory_peak_mb",
        ),
        (
            _drop(("cells", 0, "gates")),
            f"plan_gate_missing:{NNAPI}:maximum_device_memory_mb",
        ),
        (_drop(("gates", "require_thermal")), "plan_gate_missing:plan:require_thermal"),
        (
            _set(("gates", "minimum_direct_identifier_recall"), 1.5),
            "plan_gate_invalid:plan:minimum_direct_identifier_recall",
        ),
        (_set(("bundles", 0, "id"), "sm-g991b"), "plan_bundle_id_invalid:0"),
        (_set(("bundles", 0, "id"), "udid-bundle"), "plan_bundle_id_invalid:0"),
        (
            _set(("bundles", 0, "digest"), "not-a-digest"),
            "plan_bundle_digest_invalid:bert-int8-onnx-android",
        ),
        (_set(("fixtures",), []), "plan_fixture_empty"),
        (_set(("release",), "v3.5"), "plan_release_mismatch"),
        (
            _set(("schema",), "openmed.kit_qualification.plan.v0"),
            "plan_schema_unsupported",
        ),
        (
            _set(("cells", 0, "platform_versions"), ["25"]),
            f"plan_cell_platform_versions_invalid:{NNAPI}",
        ),
        (
            _set(("cells", 0, "platform_versions"), ["35"]),
            f"plan_cell_platform_versions_invalid:{NNAPI}",
        ),
        (
            _set(("cells", 0, "fixtures"), ["missing"]),
            f"plan_cell_fixtures_invalid:{NNAPI}",
        ),
        (_set(("harness_digest",), "abc"), "plan_harness_digest_invalid"),
        (_set(("cells", 0, "device_name"), "x"), "plan_cell_invalid:0"),
    ],
)
def test_plan_validation_rejects_bad_plans_without_echoing_values(
    matrix: dict[str, Any], mutate: Callable[[dict[str, Any]], None], code: str
) -> None:
    plan = make_plan(matrix, NNAPI)
    mutate(plan)

    problems = kq.validate_qualification_plan(plan, matrix)
    assert code in problems
    assert not any("no-such-cell" in problem for problem in problems)
    assert not any("sm-g991b" in problem for problem in problems)
    with pytest.raises(kq.KitQualificationError):
        kq.collect_qualification(matrix, plan, [])


def test_plan_validation_refuses_non_objects(matrix: dict[str, Any]) -> None:
    assert kq.validate_qualification_plan([], matrix) == ("plan_not_object",)  # type: ignore[arg-type]


def test_invalid_matrix_is_refused(matrix: dict[str, Any]) -> None:
    broken = copy.deepcopy(matrix)
    _cell(broken, NNAPI)["status"] = "supported"
    with pytest.raises(kq.KitQualificationError) as caught:
        kq.collect_qualification(broken, make_plan(matrix), [])
    assert all(problem.startswith("matrix:") for problem in caught.value.problems)


# --- qualification and proposals -------------------------------------------


def test_physical_device_evidence_qualifies_and_proposes_promotion(
    matrix: dict[str, Any],
) -> None:
    original = copy.deepcopy(matrix)
    report = kq.collect_qualification(matrix, make_plan(matrix), [make_record()])

    target = _target(report, NNAPI)
    assert target["status"] == kq.QUALIFIED
    assert target["evidence"]["device"]["passed"] == 1
    record_row = target["records"][0]
    assert record_row["outcome"] == "passed"
    assert [gate["gate"] for gate in record_row["gates"]] == list(kq.GATES)
    assert _gate(record_row, "memory")["limit"] == 350 * MIB
    assert _gate(record_row, "provider_binding")["status"] == "passed"
    row = _cell_row(report, NNAPI)
    assert row["qualified"] and row["status"] == kq.QUALIFIED
    assert report["summary"]["cells_qualified"] == 1
    assert report["summary"]["records_by_class"]["device"] == 1
    assert report["requires_review"] and not report["clinical_readiness"]
    assert not report["auto_promotion"] and not report["independent_validation"]

    proposal = kq.propose_matrix_updates(
        matrix, report, evidence_refs=["docs/openmedkit-device-qualification.md"]
    )
    assert proposal["auto_applied"] is False
    assert proposal["promotions"] == [
        {
            "cell": NNAPI,
            "from": "experimental",
            "to": "supported",
            "evidence": f"device-run-{NNAPI}",
            "qualified_targets": 1,
        }
    ]
    assert proposal["surface_claims"] == [
        {"surface": "android", "from": "experimental-only", "to": "supported"}
    ]
    [evidence] = proposal["evidence"]
    assert evidence["kind"] == "device-run"
    assert "device-qualification" in evidence["covers"]

    updated = kq.apply_matrix_proposal(matrix, proposal)
    assert validate_kit_support_matrix(updated) == ()
    assert _cell(updated, NNAPI)["status"] == "supported"
    assert _cell(updated, NNAPI)["evidence"] == [f"device-run-{NNAPI}"]
    assert matrix == original


def test_missing_targets_and_unplanned_cells_are_reported(
    matrix: dict[str, Any],
) -> None:
    plan = make_plan(matrix, NNAPI, IOS_COREML)
    report = kq.collect_qualification(matrix, plan, [make_record()])

    assert _target(report, IOS_COREML)["status"] == kq.MISSING
    assert _target(report, IOS_COREML)["reasons"] == ["no_evidence"]
    assert _cell_row(report, IOS_COREML)["status"] == kq.MISSING
    assert not _cell_row(report, IOS_COREML)["qualified"]
    summary = report["summary"]
    assert summary["cells_unplanned"] == summary["cells_in_scope"] - 2
    assert _cell_row(report, "swift-ios-mlx")["status"] == kq.UNPLANNED
    assert summary["cells_out_of_scope"] == len(report["out_of_scope_cells"]) == 5
    assert "python-linux-torch-cpu" in report["supported_without_device_qualification"]

    proposal = kq.propose_matrix_updates(
        matrix, report, evidence_refs=["docs/openmedkit-device-qualification.md"]
    )
    assert [row["cell"] for row in proposal["promotions"]] == [NNAPI]


def test_unplanned_family_or_precision_leaves_the_cell_incomplete(
    matrix: dict[str, Any],
) -> None:
    _cell(matrix, NNAPI)["precisions"] = ["fp32", "int8"]
    report = kq.collect_qualification(matrix, make_plan(matrix), [make_record()])

    assert _target(report, NNAPI)["status"] == kq.QUALIFIED
    row = _cell_row(report, NNAPI)
    assert row["status"] == kq.INCOMPLETE and not row["qualified"]
    assert row["coverage_gaps"] == ["family_precision_unplanned:bert:fp32"]
    proposal = kq.propose_matrix_updates(
        matrix, report, evidence_refs=["docs/openmedkit-device-qualification.md"]
    )
    assert proposal["promotions"] == [] and proposal["evidence"] == []


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        (
            {"bundle": {"id": "bert-int8-onnx-android", "digest": "9" * 64}},
            "bundle_digest_mismatch",
        ),
        (
            {"fixture": {"id": "synthetic-notes", "digest": "9" * 64}},
            "fixture_digest_mismatch",
        ),
        (
            {"policy": {"id": "mask-default", "digest": "9" * 64}},
            "policy_digest_mismatch",
        ),
        (
            {"runtime": {"pin": "onnxruntime-android", "version": "1.24.0"}},
            "runtime_version_mismatch",
        ),
        ({"harness_digest": "9" * 64}, "harness_digest_mismatch"),
    ],
)
def test_stale_digests_never_qualify(
    matrix: dict[str, Any], overrides: dict[str, Any], code: str
) -> None:
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(**overrides)]
    )

    target = _target(report, NNAPI)
    assert target["status"] == kq.STALE
    assert code in target["reasons"]
    [record_row] = target["records"]
    assert record_row["outcome"] == "stale" and record_row["gates"] == []
    assert not _cell_row(report, NNAPI)["qualified"]


def test_stale_records_do_not_hide_a_fresh_pass(matrix: dict[str, Any]) -> None:
    stale = make_record(harness_digest="9" * 64)
    report = kq.collect_qualification(matrix, make_plan(matrix), [stale, make_record()])

    target = _target(report, NNAPI)
    assert target["status"] == kq.QUALIFIED
    assert target["evidence"]["device"] == {
        "passed": 1,
        "failed": 0,
        "not_run": 0,
        "stale": 1,
    }


def test_changed_cell_definition_makes_planned_targets_stale(
    matrix: dict[str, Any],
) -> None:
    plan = make_plan(matrix)
    _cell(matrix, NNAPI)["tier"] = "Base"

    report = kq.collect_qualification(matrix, plan, [make_record()])
    assert _target(report, NNAPI)["status"] == kq.STALE
    assert _target(report, NNAPI)["reasons"] == ["cell_definition_changed"]
    assert _cell_row(report, NNAPI)["plan_fresh"] is False


# --- simulator, emulator and fake-runtime evidence --------------------------


def test_simulator_evidence_never_qualifies_a_physical_cell(
    matrix: dict[str, Any],
) -> None:
    plan = make_plan(matrix, IOS_COREML)
    report = kq.collect_qualification(matrix, plan, [simulator_record()])

    target = _target(report, IOS_COREML)
    assert target["status"] == kq.SIMULATOR_ONLY
    assert target["reasons"] == ["device_evidence_missing"]
    assert target["evidence"]["simulator"]["passed"] == 1
    assert target["records"][0]["evidence_class"] == "simulator"
    assert report["summary"]["records_by_class"] == {
        "device": 0,
        "simulator": 1,
        "fake_runtime": 0,
    }
    proposal = kq.propose_matrix_updates(
        matrix, report, evidence_refs=["docs/openmedkit-device-qualification.md"]
    )
    assert proposal["promotions"] == [] and proposal["evidence"] == []


def test_simulator_class_cells_can_never_be_promoted(matrix: dict[str, Any]) -> None:
    plan = make_plan(matrix, SIM_COREML)
    report = kq.collect_qualification(matrix, plan, [simulator_record(SIM_COREML)])
    assert _target(report, SIM_COREML)["status"] == kq.SIMULATOR_ONLY
    assert _cell_row(report, SIM_COREML)["virtual_device_class"]

    physical = make_record(SIM_COREML, device_class="iphone-ipad")
    report = kq.collect_qualification(matrix, plan, [physical])
    assert report["rejected_records"][0]["reasons"] == [
        "execution_device_class_mismatch"
    ]
    assert _target(report, SIM_COREML)["status"] == kq.MISSING


def test_emulator_and_fake_runtime_evidence_are_labelled_separately(
    matrix: dict[str, Any],
) -> None:
    emulator = make_record(
        execution="emulator", device_class="android-emulator", abi="x86_64"
    )
    report = kq.collect_qualification(matrix, make_plan(matrix), [emulator])
    assert _target(report, NNAPI)["status"] == kq.SIMULATOR_ONLY

    fake = make_record(execution="fake-runtime")
    report = kq.collect_qualification(matrix, make_plan(matrix), [fake])
    assert _target(report, NNAPI)["status"] == kq.FAKE_RUNTIME_ONLY
    assert _target(report, NNAPI)["records"][0]["evidence_class"] == "fake_runtime"


@pytest.mark.parametrize(
    ("execution", "device_class"),
    [
        ("physical-device", "ios-simulator"),
        ("physical-device", "android-emulator"),
        ("simulator", "iphone-ipad"),
        ("simulator", "android-emulator"),
        ("emulator", "android-device"),
    ],
)
def test_execution_must_agree_with_the_device_class(
    execution: str, device_class: str
) -> None:
    record = make_record(IOS_COREML, execution=execution, device_class=device_class)
    assert "evidence_execution_device_class_mismatch" in kq.validate_evidence_record(
        record
    )


def test_simulator_passes_do_not_offset_a_device_failure(
    matrix: dict[str, Any],
) -> None:
    plan = make_plan(matrix, IOS_COREML)
    failing = make_record(IOS_COREML, metrics=_changed("cold_latency_ms", value=9e3))
    report = kq.collect_qualification(matrix, plan, [simulator_record(), failing])
    assert _target(report, IOS_COREML)["status"] == kq.FAILED

    sim_failing = simulator_record(metrics=_changed("cold_latency_ms", value=9e3))
    report = kq.collect_qualification(
        matrix, plan, [sim_failing, make_record(IOS_COREML)]
    )
    assert _target(report, IOS_COREML)["status"] == kq.QUALIFIED


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"platform": {"name": "ios", "version": "17.5"}}, "platform_mismatch"),
        (
            {"platform": {"name": "android", "version": "25"}},
            "platform_version_out_of_range",
        ),
        ({"provider": "cpu"}, "provider_mismatch"),
        (
            {"runtime": {"pin": "onnxruntime-web", "version": "1.25.1"}},
            "runtime_mismatch",
        ),
        ({"abi": "x86_64"}, "abi_mismatch"),
        ({"device_class": "iphone-ipad"}, "device_class_mismatch"),
        ({"precision": "fp32"}, "precision_mismatch"),
        (
            {"bundle": {"id": "bert-int8-coreml", "digest": "5" * 64}},
            "bundle_not_planned",
        ),
        ({"fixture": {"id": "other-notes", "digest": FIXTURE}}, "fixture_not_planned"),
        ({"policy": {"id": "other-policy", "digest": POLICY}}, "policy_not_planned"),
        ({"cell": "android-arm64-onnx-cpu"}, "cell_not_planned"),
        ({"cell": "key-store-android-tee"}, "cell_out_of_scope"),
        ({"cell": "android-armeabi-v7a"}, "cell_unsupported"),
        ({"cell": "no-such-cell-xyz"}, "cell_unknown"),
    ],
)
def test_records_that_do_not_fit_are_rejected(
    matrix: dict[str, Any], overrides: dict[str, Any], code: str
) -> None:
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(**overrides)]
    )

    [rejected] = report["rejected_records"]
    assert code in rejected["reasons"]
    assert re.fullmatch(r"[0-9a-f]{64}", rejected["record"])
    assert _target(report, NNAPI)["status"] == kq.MISSING
    assert "no-such-cell-xyz" not in json.dumps(report)


def test_platform_versions_are_matched_by_prefix(matrix: dict[str, Any]) -> None:
    plan = make_plan(matrix)
    plan["cells"][0]["platform_versions"] = ["29", "34"]
    report = kq.collect_qualification(
        matrix,
        plan,
        [make_record(platform={"name": "android", "version": "34.1"})],
    )
    statuses = {t["platform_version"]: t["status"] for t in report["targets"]}
    assert statuses == {"29": kq.MISSING, "34": kq.QUALIFIED}
    assert not _cell_row(report, NNAPI)["qualified"]

    report = kq.collect_qualification(
        matrix, plan, [make_record(platform={"name": "android", "version": "31"})]
    )
    assert report["rejected_records"][0]["reasons"] == ["platform_version_unplanned"]


# --- gates ------------------------------------------------------------------


def _unavailable(name: str) -> dict[str, Any]:
    metrics = passing_metrics()
    metrics[name] = {"status": "unavailable"}
    return metrics


def _changed(name: str, **fields: Any) -> dict[str, Any]:
    metrics = passing_metrics()
    metrics[name] = {**metrics[name], **fields}
    return metrics


@pytest.mark.parametrize(
    ("metrics", "gate", "code"),
    [
        (_changed("direct_identifier_recall", detected=90), "recall", "recall"),
        (
            _unavailable("direct_identifier_recall"),
            "recall",
            "missing_privacy_evidence",
        ),
        (
            _changed("leakage", critical=1, any_phi=1),
            "leakage",
            "critical_or_document_phi_leakage",
        ),
        (_changed("leakage", any_phi=1), "leakage", "critical_or_document_phi_leakage"),
        (_changed("leakage", assessed=0), "leakage", "unassessed_or_failed_units"),
        (_unavailable("leakage"), "leakage", "missing_privacy_evidence"),
        (_changed("cold_latency_ms", value=5001.0), "cold_latency", "cold_latency"),
        (_unavailable("cold_latency_ms"), "cold_latency", "missing_latency_evidence"),
        (_changed("warm_latency_ms", p95=151.0), "warm_latency", "latency"),
        (_changed("warm_latency_ms", n=2), "warm_latency", "incomplete_measurements"),
        (
            _changed("sustained_latency_ms", duration_s=299.0),
            "sustained_duration",
            "sustained_duration",
        ),
        (
            _changed("sustained_latency_ms", p95=201.0),
            "sustained_latency",
            "sustained_latency",
        ),
        (_changed("cancel_latency_ms", p95=251.0), "cancel_latency", "cancel_latency"),
        (
            _unavailable("cancel_latency_ms"),
            "cancel_latency",
            "missing_latency_evidence",
        ),
        (_changed("memory_peak_bytes", value=351 * MIB), "memory", "resource_budget"),
        (_unavailable("memory_peak_bytes"), "memory", "missing_resource_evidence"),
        (
            _changed("device_memory_peak_bytes", value=257 * MIB),
            "device_memory",
            "resource_budget",
        ),
        (
            _unavailable("device_memory_peak_bytes"),
            "device_memory",
            "missing_resource_evidence",
        ),
        (_changed("thermal", max_level=3), "thermal", "resource_budget"),
        (_unavailable("thermal"), "thermal", "missing_resource_evidence"),
        (
            _changed("provider_fallback", observed=True),
            "provider_binding",
            "runtime_fallback",
        ),
        (
            _unavailable("provider_fallback"),
            "provider_binding",
            "missing_provider_evidence",
        ),
    ],
)
def test_each_deliberately_failed_gate_fails_on_its_own(
    matrix: dict[str, Any], metrics: dict[str, Any], gate: str, code: str
) -> None:
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(metrics=metrics)]
    )

    target = _target(report, NNAPI)
    assert target["status"] == kq.FAILED
    assert code in target["reasons"]
    [record_row] = target["records"]
    failing = [
        row["gate"]
        for row in record_row["gates"]
        if row["status"]
        in (
            "failed",
            "missing",
        )
    ]
    assert failing == [gate]
    assert code in _gate(record_row, gate)["reasons"]
    assert not _cell_row(report, NNAPI)["qualified"]


def test_run_failures_and_required_energy_fail_their_gates(
    matrix: dict[str, Any],
) -> None:
    failed_run = make_record(run={"status": "failed", "failures": ["out_of_memory"]})
    report = kq.collect_qualification(matrix, make_plan(matrix), [failed_run])
    assert _target(report, NNAPI)["reasons"] == ["out_of_memory"]
    assert _gate(_target(report, NNAPI)["records"][0], "run")["status"] == "failed"

    plan = make_plan(matrix)
    plan["gates"]["require_measured_energy"] = True
    report = kq.collect_qualification(matrix, plan, [make_record()])
    target = _target(report, NNAPI)
    assert target["status"] == kq.FAILED
    energy = _gate(target["records"][0], "energy")
    assert energy["status"] == "missing"
    assert energy["measurement"] == "proxy:process_cpu_seconds"

    measured = _changed("energy")
    measured["energy"] = {"status": "measured", "joules": 41.5}
    report = kq.collect_qualification(matrix, plan, [make_record(metrics=measured)])
    assert _target(report, NNAPI)["status"] == kq.QUALIFIED
    assert _target(report, NNAPI)["records"][0]["energy_measured"] is True


def test_one_failed_run_blocks_a_passing_run(matrix: dict[str, Any]) -> None:
    failing = make_record(metrics=_changed("thermal", max_level=3))
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(), failing]
    )
    assert _target(report, NNAPI)["status"] == kq.FAILED
    assert _target(report, NNAPI)["evidence"]["device"]["passed"] == 1


def test_measured_and_proxy_metrics_are_labelled(matrix: dict[str, Any]) -> None:
    plan = make_plan(matrix)
    plan["gates"]["require_thermal"] = False
    report = kq.collect_qualification(
        matrix, plan, [make_record(metrics=_unavailable("thermal"))]
    )

    [record_row] = _target(report, NNAPI)["records"]
    assert record_row["outcome"] == "passed"
    assert record_row["thermal_measured"] is False
    assert record_row["energy_measured"] is False
    assert record_row["energy_measurement"] == "proxy:process_cpu_seconds"
    assert record_row["device_memory_measured"] is True
    assert _gate(record_row, "thermal")["status"] == "informational"
    assert _gate(record_row, "energy")["status"] == "informational"


# --- not_run and EdgeHealthBench --------------------------------------------


def _profile(tier: edge_health.DeviceTier, **overrides: Any) -> Any:
    provenance = ReleaseProvenance(
        model_digest="a" * 64,
        preprocessing_digest="b" * 64,
        hardware_digest="c" * 64,
        runtime_digest="d" * 64,
        model_license_digest="e" * 64,
        dataset_license_digest="f" * 64,
    )
    values: dict[str, Any] = {
        "tier": tier,
        "provenance": provenance,
        "dataset_digest": "1" * 64,
        "workload_digest": "2" * 64,
        "reference_output_digest": "3" * 64,
        "units_per_trial": 2,
        "trials": 3,
        "warmups": 0,
        "maximum_latency_ms": 100,
        "minimum_throughput": 1,
        "maximum_rss_bytes": 2000,
        "maximum_thermal_level": 1,
        "require_thermal": True,
        "minimum_quality_rate": 1,
        "minimum_utility_rate": 1,
        "maximum_abstention_rate": 0,
        "maximum_device_memory_bytes": 2000,
    }
    values.update(overrides)
    return edge_health.DeviceProfile(**values)


def test_not_run_never_counts_as_a_pass(matrix: dict[str, Any]) -> None:
    not_run = make_record(
        run={"status": "not_run", "failures": ["provider_unavailable"]}
    )
    report = kq.collect_qualification(matrix, make_plan(matrix), [not_run])

    target = _target(report, NNAPI)
    assert target["status"] == kq.NOT_RUN
    [record_row] = target["records"]
    assert record_row["reasons"] == ["not_run", "provider_unavailable"]
    assert record_row["gates"] == []

    unmeasured = edge_health.run_device_profile(
        _profile(edge_health.DeviceTier.ANDROID_ORT_NNAPI), None
    )
    block = kq.edge_health_block(json.loads(json.dumps(unmeasured)))
    assert block["status"] == "not_run" and block["tier"] == "android_ort_nnapi"
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(edge_health=block)]
    )
    assert _target(report, NNAPI)["status"] == kq.NOT_RUN


def test_edge_health_summaries_bind_tier_and_failures(matrix: dict[str, Any]) -> None:
    block = {
        "schema": kq.EDGE_HEALTH_SCHEMA,
        "tier": "android_ort_nnapi",
        "status": "passed",
        "failures": [],
        "report_digest": "a" * 64,
    }
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(edge_health=block)]
    )
    assert _target(report, NNAPI)["status"] == kq.QUALIFIED

    cpu_tier = {**block, "tier": "cpu"}
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(edge_health=cpu_tier)]
    )
    assert _target(report, NNAPI)["reasons"] == ["edge_health_tier_mismatch"]

    failed = {**block, "status": "failed", "failures": ["missing_resource_evidence"]}
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(edge_health=failed)]
    )
    assert _target(report, NNAPI)["reasons"] == ["missing_resource_evidence"]

    invalid = {**block, "failures": ["missing_resource_evidence"]}
    assert "evidence_edge_health_invalid" in kq.validate_evidence_record(
        make_record(edge_health=invalid)
    )


def test_edge_health_block_rechecks_the_report_digest() -> None:
    report = edge_health.run_device_profile(
        _profile(edge_health.DeviceTier.IOS_DEVICE_CORE_ML), None
    )
    assert kq.edge_health_block(report)["tier"] == "ios_device_core_ml"

    edited = {**report, "status": "passed", "passed": True}
    with pytest.raises(kq.KitQualificationError) as caught:
        kq.edge_health_block(edited)
    assert caught.value.problems == ("edge_health_report_digest_mismatch",)
    with pytest.raises(kq.KitQualificationError):
        kq.edge_health_block({**report, "schema": "openmed.edge_health.v0"})


def test_mobile_device_tiers_and_allocator_rule() -> None:
    tiers = edge_health.DeviceTier
    assert {tier.value for tier in edge_health.MOBILE_DEVICE_TIERS} == {
        "android_ort_cpu",
        "android_ort_nnapi",
        "android_ort_qnn",
        "ios_device_cpu",
        "ios_device_mlx",
        "ios_device_core_ml",
    }
    for tier in edge_health.DEVICE_MEMORY_TIERS:
        with pytest.raises(VisionGateError):
            _profile(tier, maximum_device_memory_bytes=None)
    for tier in (tiers.ANDROID_ORT_CPU, tiers.IOS_DEVICE_CPU, tiers.CPU):
        assert tier not in edge_health.DEVICE_MEMORY_TIERS
        _profile(tier, maximum_device_memory_bytes=None)
    assert {tiers.ANDROID_ORT_NNAPI, tiers.ANDROID_ORT_QNN} <= (
        edge_health.DEVICE_MEMORY_TIERS
    )


def test_edge_health_failure_codes_cover_what_the_harness_reports() -> None:
    source = Path(edge_health.__file__).read_text(encoding="utf-8")
    emitted = set(re.findall(r'failed\.add\("([a-z_]+)"\)', source))
    emitted |= set(re.findall(r'return "([a-z_]+)"', source))
    assert emitted == set(edge_health.FAILURE_CODES)
    assert edge_health.FAILURE_CODES <= kq.RUN_FAILURE_CODES


# --- privacy and determinism ------------------------------------------------


@pytest.mark.parametrize(
    ("value", "shaped"),
    [
        ("SM-G991B", True),
        ("SM-S918", True),
        ("GT-I9300", True),
        ("iPhone15,2", True),
        ("udid-1234", True),
        ("android-device", False),
        ("onnxruntime-android", False),
        ("bert-int8-onnx-android", False),
    ],
)
def test_identifier_shapes_include_model_codes_with_suffixes(
    value: str, shaped: bool
) -> None:
    assert identifier_shaped(value) is shaped


def test_records_with_identifiers_are_rejected_without_echo(
    matrix: dict[str, Any],
) -> None:
    named = make_record(device_name="Jane Example's phone", serial="C02XK0ABJGH5")
    model_bundle = make_record(bundle={"id": "sm-g991b", "digest": "1" * 64})
    keyword = make_record(fixture={"id": "udid-notes", "digest": FIXTURE})
    hex_id = make_record(policy={"id": "a1b2c3d4e5f60718", "digest": POLICY})

    assert "evidence_field_unknown" in kq.validate_evidence_record(named)
    assert "evidence_bundle_invalid" in kq.validate_evidence_record(model_bundle)
    assert "evidence_fixture_invalid" in kq.validate_evidence_record(keyword)
    assert "evidence_policy_invalid" in kq.validate_evidence_record(hex_id)
    report = kq.collect_qualification(
        matrix, make_plan(matrix), [named, model_bundle, keyword, hex_id]
    )
    text = json.dumps(report)
    assert report["summary"]["records_rejected"] == 4
    for value in ("Jane", "C02XK0ABJGH5", "sm-g991b", "udid-notes", "a1b2c3d4e5f60718"):
        assert value not in text
    assert "/Users/" not in text and "/home/" not in text


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda r: r.update(schema="openmed.kit_qualification.evidence.v0"),
            "evidence_schema_unsupported",
        ),
        (lambda r: r.update(recorded_on="2026-02-30"), "evidence_recorded_on_invalid"),
        (lambda r: r.update(execution="cloud"), "evidence_execution_invalid"),
        (
            lambda r: r.update(device_class=["android-device"]),
            "evidence_device_class_invalid",
        ),
        (lambda r: r.update(platform={"name": "android"}), "evidence_platform_invalid"),
        (
            lambda r: r.update(runtime={"pin": "x", "version": "v1"}),
            "evidence_runtime_invalid",
        ),
        (
            lambda r: r.update(run={"status": "completed", "failures": ["latency"]}),
            "evidence_run_invalid",
        ),
        (
            lambda r: r.update(run={"status": "failed", "failures": []}),
            "evidence_run_invalid",
        ),
        (
            lambda r: r.update(run={"status": "failed", "failures": ["free text"]}),
            "evidence_run_invalid",
        ),
        (lambda r: r.pop("metrics"), "evidence_field_missing:metrics"),
        (lambda r: r["metrics"].pop("thermal"), "evidence_metric_missing:thermal"),
        (
            lambda r: r["metrics"].update(extra={"status": "unavailable"}),
            "evidence_metrics_field_unknown",
        ),
        (
            lambda r: r["metrics"]["leakage"].update(critical=2, any_phi=1),
            "evidence_metric_invalid:leakage",
        ),
        (
            lambda r: r["metrics"]["warm_latency_ms"].update(p50=90.0, p95=60.0),
            "evidence_metric_invalid:warm_latency_ms",
        ),
        (
            lambda r: r["metrics"]["direct_identifier_recall"].update(detected=101),
            "evidence_metric_invalid:direct_identifier_recall",
        ),
        (
            lambda r: r["metrics"]["cold_latency_ms"].update(value=float("nan")),
            "evidence_metric_invalid:cold_latency_ms",
        ),
        (
            lambda r: r["metrics"]["memory_peak_bytes"].update(value=True),
            "evidence_metric_invalid:memory_peak_bytes",
        ),
        (
            lambda r: r["metrics"]["thermal"].update(status="proxy"),
            "evidence_metric_invalid:thermal",
        ),
        (
            lambda r: r["metrics"]["thermal"].update(max_level=4),
            "evidence_metric_invalid:thermal",
        ),
        (
            lambda r: r["metrics"].update(
                energy={"status": "proxy", "proxy": "guess", "value": 1.0}
            ),
            "evidence_metric_invalid:energy",
        ),
        (
            lambda r: r["metrics"].update(
                thermal={"status": "unavailable", "max_level": 0}
            ),
            "evidence_metric_invalid:thermal",
        ),
    ],
)
def test_evidence_schema_is_closed_and_bounded(
    mutate: Callable[[dict[str, Any]], Any], code: str
) -> None:
    record = make_record()
    assert kq.validate_evidence_record(record) == ()
    mutate(record)
    assert code in kq.validate_evidence_record(record)


def test_reports_are_deterministic_and_count_bad_inputs(
    matrix: dict[str, Any],
) -> None:
    records: list[Any] = [
        make_record(),
        make_record(
            execution="emulator", device_class="android-emulator", abi="x86_64"
        ),
        None,
        ["not", "an", "object"],
        make_record(cell="no-such-cell-xyz"),
    ]
    forward = kq.collect_qualification(matrix, make_plan(matrix), records)
    backward = kq.collect_qualification(matrix, make_plan(matrix), records[::-1])
    assert forward == backward
    assert forward["report_digest"] == kq._hash(
        {k: v for k, v in forward.items() if k != "report_digest"}
    )
    reasons = sorted(r for row in forward["rejected_records"] for r in row["reasons"])
    assert reasons == ["cell_unknown", "evidence_not_object", "evidence_unreadable"]

    duplicate = kq.collect_qualification(
        matrix, make_plan(matrix), [make_record(), make_record()]
    )
    assert duplicate["rejected_records"][0]["reasons"] == ["evidence_duplicate"]
    assert (
        kq.evidence_record_digest(make_record())
        == (duplicate["targets"][0]["records"][0]["record"])
    )
    with pytest.raises(kq.KitQualificationError) as caught:
        kq.collect_qualification(
            matrix, make_plan(matrix), [None] * (kq.MAX_EVIDENCE_RECORDS + 1)
        )
    assert caught.value.problems == ("evidence_too_many",)


# --- proposal safety --------------------------------------------------------


def test_proposals_refuse_edited_or_mismatched_reports(matrix: dict[str, Any]) -> None:
    report = kq.collect_qualification(matrix, make_plan(matrix), [make_record()])
    refs = ["docs/openmedkit-device-qualification.md"]

    edited = copy.deepcopy(report)
    _cell_row(edited, IOS_COREML)["qualified"] = True
    with pytest.raises(kq.KitQualificationError) as caught:
        kq.propose_matrix_updates(matrix, edited, evidence_refs=refs)
    assert caught.value.problems == ("report_digest_mismatch",)

    other = copy.deepcopy(matrix)
    _cell(other, NNAPI)["note"] = "Edited after collection."
    with pytest.raises(kq.KitQualificationError) as caught:
        kq.propose_matrix_updates(other, report, evidence_refs=refs)
    assert caught.value.problems == ("report_matrix_mismatch",)

    with pytest.raises(kq.KitQualificationError) as caught:
        kq.propose_matrix_updates(matrix, report, evidence_refs=[])
    assert caught.value.problems == ("proposal_evidence_refs_missing",)

    with pytest.raises(kq.KitQualificationError) as caught:
        kq.propose_matrix_updates(matrix, report, evidence_refs=["/tmp/report.json"])
    assert f"proposed_matrix:evidence_refs_invalid:device-run-{NNAPI}" in (
        caught.value.problems
    )

    proposal = kq.propose_matrix_updates(matrix, report, evidence_refs=refs)
    tampered = {**proposal, "promotions": [*proposal["promotions"]] * 2}
    with pytest.raises(kq.KitQualificationError) as caught:
        kq.apply_matrix_proposal(matrix, tampered)
    assert caught.value.problems == ("proposal_digest_mismatch",)
    applied = kq.apply_matrix_proposal(matrix, proposal)
    with pytest.raises(kq.KitQualificationError) as caught:
        kq.apply_matrix_proposal(applied, proposal)
    assert caught.value.problems == ("proposal_matrix_mismatch",)


def test_supported_cells_gain_evidence_or_review_but_are_never_demoted(
    matrix: dict[str, Any],
) -> None:
    refs = ["docs/openmedkit-device-qualification.md"]
    plan = make_plan(matrix, MAC_MLX)
    report = kq.collect_qualification(matrix, plan, [make_record(MAC_MLX)])
    proposal = kq.propose_matrix_updates(matrix, report, evidence_refs=refs)
    assert proposal["promotions"] == [] and proposal["surface_claims"] == []
    assert proposal["evidence_additions"] == [
        {"cell": MAC_MLX, "evidence": f"device-run-{MAC_MLX}"}
    ]
    updated = kq.apply_matrix_proposal(matrix, proposal)
    assert _cell(updated, MAC_MLX)["evidence"] == [
        "ci-swift-macos",
        f"device-run-{MAC_MLX}",
    ]

    failing = make_record(MAC_MLX, metrics=_changed("cold_latency_ms", value=9e3))
    report = kq.collect_qualification(matrix, plan, [failing])
    assert MAC_MLX in report["supported_without_device_qualification"]
    proposal = kq.propose_matrix_updates(matrix, report, evidence_refs=refs)
    assert proposal["reviews"] == [
        {"cell": MAC_MLX, "status": kq.FAILED, "action": "review"}
    ]
    assert proposal["evidence"] == [] and proposal["evidence_additions"] == []
    assert _cell(kq.apply_matrix_proposal(matrix, proposal), MAC_MLX)["status"] == (
        "supported"
    )


# --- command line -----------------------------------------------------------


def _write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_cli_writes_report_proposal_and_proposed_matrix(
    tmp_path: Path, matrix: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    matrix_path = _write(tmp_path / "matrix.json", matrix)
    plan_path = _write(tmp_path / "plan.json", make_plan(matrix))
    runs = tmp_path / "runs"
    runs.mkdir()
    _write(runs / "a.json", make_record())
    (runs / "b.json").write_text("{not json", encoding="utf-8")
    out = tmp_path / "out"

    status = cli.main(
        [
            "--matrix",
            str(matrix_path),
            "--plan",
            str(plan_path),
            "--evidence",
            str(runs),
            "--output",
            str(out / "report.json"),
            "--proposal",
            str(out / "proposal.json"),
            "--proposed-matrix",
            str(out / "matrix.json"),
            "--evidence-ref",
            "docs/openmedkit-device-qualification.md",
        ]
    )

    assert status == 0
    printed = capsys.readouterr().out
    assert "evidence_unreadable" in printed
    assert "1 promotions" in printed and "not applied" in printed
    assert str(tmp_path) not in printed and "a.json" not in printed
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["cells_qualified"] == 1
    assert str(tmp_path) not in json.dumps(report)
    proposed = json.loads((out / "matrix.json").read_text(encoding="utf-8"))
    assert validate_kit_support_matrix(proposed) == ()
    assert _cell(proposed, NNAPI)["status"] == "supported"
    assert json.loads(matrix_path.read_text(encoding="utf-8")) == matrix


def test_cli_require_qualified_and_overwrite_guard(
    tmp_path: Path, matrix: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    matrix_path = _write(tmp_path / "matrix.json", matrix)
    plan_path = _write(tmp_path / "plan.json", make_plan(matrix, NNAPI, IOS_COREML))
    record_path = _write(tmp_path / "run.json", make_record())
    base = ["--matrix", str(matrix_path), "--plan", str(plan_path)]

    assert cli.main([*base, "--evidence", str(record_path), "--require-qualified"]) == 1
    assert f"not_qualified:{IOS_COREML}" in capsys.readouterr().out

    before = matrix_path.read_bytes()
    guarded = cli.main(
        [
            *base,
            "--evidence",
            str(record_path),
            "--proposed-matrix",
            str(matrix_path),
            "--evidence-ref",
            "docs/openmedkit-device-qualification.md",
        ]
    )
    assert guarded == 1
    assert "proposed_matrix_would_overwrite_matrix" in capsys.readouterr().out
    assert matrix_path.read_bytes() == before

    with pytest.raises(SystemExit):
        cli.main([*base, "--proposal", str(tmp_path / "p.json")])


def test_cli_plan_skeleton_and_invalid_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skeleton = tmp_path / "skeleton.json"
    assert cli.main(["--plan-skeleton", str(skeleton)]) == 0
    document = json.loads(skeleton.read_text(encoding="utf-8"))
    assert document["schema"] == kq.PLAN_SCHEMA and document["cells"]

    capsys.readouterr()
    assert cli.main(["--plan", str(skeleton)]) == 1
    printed = capsys.readouterr().out
    assert "kit qualification: plan_bundle_empty" in printed
    assert str(tmp_path) not in printed

    assert cli.main(["--plan", str(tmp_path / "missing.json")]) == 1
    assert "plan_unreadable" in capsys.readouterr().out
    assert cli.main(["--plan", str(skeleton), "--evidence", str(tmp_path / "x")]) == 1
    assert "evidence_source_missing" in capsys.readouterr().out
