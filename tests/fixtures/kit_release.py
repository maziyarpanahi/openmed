"""Synthetic builders for the v4.0 OpenMedKit release-gate tests.

Every artifact, record, digest and commit here is synthetic. The signing key
below is committed on purpose and is TEST-ONLY: it signs fixture packets so
offline verification can be exercised, and it must never sign real release
evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openmed.eval import kit_qualification as kq
from openmed.eval.kit_release import KIT_RELEASE_TEST_LANE_SCHEMA
from openmed.eval.kit_support_matrix import (
    SUPPORTED,
    load_kit_support_matrix,
    validate_kit_support_matrix,
)

# TEST-ONLY committed key. Never use it for a real release packet.
TEST_ONLY_SIGNING_KEY = b"test-only-openmed-kit-release-signing-key-never-for-release"
TEST_ONLY_KEY_ID = "test-only-kit-release"
OTHER_TEST_ONLY_KEY = b"test-only-other-kit-release-key-for-negative-controls"

CANDIDATE = "3932" * 10
OTHER_COMMIT = "5481" * 10
EVALUATED_AT = "2026-11-08T00:00:00Z"
GENERATED_AT = "2026-11-07T12:00:00Z"
HARNESS = "4" * 64
FIXTURE_DIGEST = "2" * 64
POLICY_DIGEST = "3" * 64
MIB = 1024 * 1024
IOS_COREML = "swift-ios-coreml"
SIM_COREML = "swift-ios-simulator-coreml"

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

# Surface -> (format, synthetic file name) for the kit artifacts the fixture
# ships for every surface with a supported cell.
ARTIFACTS = {
    "python": ("python-wheel", "openmed-4.0.0-py3-none-any.whl"),
    "swift": ("swiftpm-source", "OpenMedKit-4.0.0-source.tar"),
    "android": ("aar", "openmedkit-4.0.0-release.aar"),
}
DIGEST_ONLY_FILES = {
    "bundle_manifest": ("bundle-bert-int8", "bundles/bert-int8.manifest.json"),
    "parity_fixture": ("offset-parity-vectors", "parity/offset-vectors.json"),
    "license_inventory": ("kit-license-inventory", "licenses/inventory.json"),
    "threat_model": ("kit-threat-model", "security/openmedkit-threat-model.md"),
}


def head_resolver(commit: str = CANDIDATE) -> Callable[[Path], str]:
    """Return an injected ``HEAD`` resolver pinned to *commit*."""

    return lambda _root: commit


def synthetic_matrix(
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Return a validated copy of the committed matrix, optionally edited."""

    matrix = copy.deepcopy(load_kit_support_matrix())
    if mutate is not None:
        mutate(matrix)
    problems = validate_kit_support_matrix(matrix)
    assert problems == (), problems
    return matrix


def promote_cell(cell_id: str, *evidence: str) -> Callable[[dict[str, Any]], None]:
    """Return a matrix edit that marks one cell ``supported``."""

    def mutate(matrix: dict[str, Any]) -> None:
        cell = next(row for row in matrix["cells"] if row["id"] == cell_id)
        cell["status"] = SUPPORTED
        cell["evidence"] = list(evidence)
        for surface in matrix["surfaces"]:
            if surface["id"] == cell["surface"]:
                surface["support_claim"] = "supported"

    return mutate


def passing_metrics() -> dict[str, Any]:
    """Metrics that pass :data:`GATES` on every tier the fixture uses."""

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


def qualification_report(
    matrix: Mapping[str, Any],
    *,
    executions: Mapping[str, str] | None = None,
    unplanned: tuple[str, ...] = (),
    without_records: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Collect a real qualification report for every supported inference cell.

    Each supported cell gets one bundle per declared family and precision and
    one record per target. Records are physical-device runs unless
    *executions* names ``simulator``, ``emulator`` or ``fake-runtime`` for a
    cell. Cells in *unplanned* are left out of the plan; cells in
    *without_records* are planned but get no records.
    """

    executions = dict(executions or {})
    rows = [
        row
        for row in kq.qualification_cells(matrix)
        if row["matrix_status"] == SUPPORTED and row["cell"] not in unplanned
    ]
    bundles: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for row in rows:
        names = []
        for family in row["families"]:
            for precision in row["precisions"]:
                name = f"{row['artifact']}-{family}-{precision}"
                digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
                bundles[name] = {
                    "id": name,
                    "family": family,
                    "artifact": row["artifact"],
                    "precision": precision,
                    "digest": digest,
                }
                names.append(name)
        entry: dict[str, Any] = {
            "cell": row["cell"],
            "cell_digest": row["cell_digest"],
            "bundles": names,
        }
        if row["device_memory_required"]:
            entry["gates"] = {"maximum_device_memory_mb": 256}
        entries.append(entry)
    plan = {
        "schema": kq.PLAN_SCHEMA,
        "release": matrix["release"],
        "harness_digest": HARNESS,
        "bundles": list(bundles.values()),
        "fixtures": [{"id": "synthetic-notes", "digest": FIXTURE_DIGEST}],
        "policies": [{"id": "mask-default", "digest": POLICY_DIGEST}],
        "gates": dict(GATES),
        "cells": entries,
    }
    assert kq.validate_qualification_plan(plan, matrix) == ()
    pins = {pin["id"]: pin for pin in matrix["runtime_pins"]}
    records = []
    for row, entry in zip(rows, entries):
        if row["cell"] in without_records:
            continue
        execution = executions.get(row["cell"], kq.PHYSICAL_DEVICE)
        device_class = row["device_class"]
        if execution in (kq.SIMULATOR, kq.EMULATOR) and (
            device_class not in kq.VIRTUAL_DEVICE_CLASSES
        ):
            device_class = next(
                virtual
                for virtual, physical in kq.VIRTUAL_DEVICE_CLASSES.items()
                if physical == row["device_class"]
            )
        for name in entry["bundles"]:
            records.append(
                {
                    "schema": kq.EVIDENCE_SCHEMA,
                    "recorded_on": "2026-11-07",
                    "cell": row["cell"],
                    "execution": execution,
                    "device_class": device_class,
                    "abi": row["abi"],
                    "platform": {
                        "name": row["platform"]["name"],
                        "version": row["platform"].get("min", "1.0"),
                    },
                    "runtime": {
                        "pin": row["runtime"],
                        "version": _runtime_version(pins[row["runtime"]]),
                    },
                    "provider": row["provider"],
                    "bundle": {"id": name, "digest": bundles[name]["digest"]},
                    "precision": bundles[name]["precision"],
                    "fixture": {"id": "synthetic-notes", "digest": FIXTURE_DIGEST},
                    "policy": {"id": "mask-default", "digest": POLICY_DIGEST},
                    "harness_digest": HARNESS,
                    "run": {"status": "completed", "failures": []},
                    "metrics": passing_metrics(),
                }
            )
    return kq.collect_qualification(matrix, plan, records)


def lane_record(
    lane: str,
    *,
    commit: str = CANDIDATE,
    state: str = "success",
    failed: int = 0,
) -> dict[str, Any]:
    """Return one synthetic test-lane result."""

    return {
        "schema": KIT_RELEASE_TEST_LANE_SCHEMA,
        "lane": lane,
        "candidate_commit": commit,
        "state": state,
        "test_count": 120,
        "failed_count": failed,
        "skipped_count": 3,
    }


def write_evidence(
    root: Path,
    *,
    matrix: Mapping[str, Any] | None = None,
    report: Mapping[str, Any] | None = None,
    contents: Mapping[str, bytes] | None = None,
    prefix: str = "release-evidence",
) -> dict[str, Any]:
    """Write synthetic evidence beneath *root* and return a passing manifest.

    Args:
        root: Candidate checkout directory.
        matrix: Matrix to freeze; defaults to the committed matrix.
        report: Qualification report; defaults to a passing report.
        contents: Optional bytes for digest-only kinds, by kind.
        prefix: Directory beneath *root* that holds the evidence.

    Returns:
        A complete manifest whose every entry matches the written files.
    """

    matrix = dict(matrix or synthetic_matrix())
    report = dict(report or qualification_report(matrix))
    contents = dict(contents or {})
    evidence: list[dict[str, Any]] = []

    def add(entry_id: str, kind: str, relative: str, data: bytes, **extra: str):
        path = root / prefix / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        evidence.append(
            {
                "id": entry_id,
                "kind": kind,
                "path": f"{prefix}/{relative}",
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "candidate_commit": CANDIDATE,
                "generated_at": GENERATED_AT,
                **extra,
            }
        )

    supported_surfaces = sorted(
        {cell["surface"] for cell in matrix["cells"] if cell["status"] == SUPPORTED}
    )
    for surface in supported_surfaces:
        artifact_format, name = ARTIFACTS[surface]
        add(
            f"{surface}-kit",
            "kit_artifact",
            f"kit/{name}",
            f"synthetic {artifact_format} archive for {surface}\n".encode(),
            surface=surface,
            format=artifact_format,
        )
    add("support-matrix", "support_matrix", "matrix.json", _json_bytes(matrix))
    add(
        "device-qualification",
        "qualification_report",
        "qualification-report.json",
        _json_bytes(report),
    )
    for kind, (entry_id, relative) in DIGEST_ONLY_FILES.items():
        data = contents.get(kind, f"synthetic {kind} evidence\n".encode())
        add(entry_id, kind, relative, data)
    kinds = {row["id"]: row["kind"] for row in matrix["evidence"]}
    lanes = sorted(
        {
            ref
            for cell in matrix["cells"]
            if cell["status"] == SUPPORTED
            for ref in cell["evidence"]
            if kinds[ref] == "ci-lane"
        }
    )
    for lane in lanes:
        add(
            f"lane-{lane}",
            "test_lane",
            f"lanes/{lane}.json",
            _json_bytes(lane_record(lane)),
        )
    return {
        "schema_version": "1.0.0",
        "compatibility_policy": "same_major",
        "release": {
            "version": "4.0.0",
            "matrix_release": "v4.0",
            "candidate_commit": CANDIDATE,
            "evaluated_at": EVALUATED_AT,
            "source_date_epoch": int(
                datetime(2026, 11, 8, tzinfo=timezone.utc).timestamp()
            ),
        },
        "policy": {"max_evidence_age_seconds": 7 * 86_400},
        "evidence": evidence,
        "moved_items": [
            {
                "issue": "#3936",
                "priority": "P1",
                "destination": "v4.1",
                "rationale": "toolchain_unavailable",
                "narrowed_claims": ["react-native"],
            }
        ],
    }


def entry(manifest: Mapping[str, Any], entry_id: str) -> dict[str, Any]:
    """Return the mutable manifest entry named *entry_id*."""

    return next(row for row in manifest["evidence"] if row["id"] == entry_id)


def rewrite(
    root: Path, manifest: Mapping[str, Any], entry_id: str, data: bytes
) -> None:
    """Replace an evidence file and re-pin its digest in the manifest."""

    row = entry(manifest, entry_id)
    (root / row["path"]).write_bytes(data)
    row["sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()


def git_checkout(root: Path) -> str:
    """Initialise *root* as a one-commit Git checkout and return ``HEAD``."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "README.txt").write_text("synthetic kit release checkout\n", "utf-8")
    for arguments in (
        ("init", "-q"),
        ("config", "user.name", "OpenMed Test"),
        ("config", "user.email", "test@openmed.invalid"),
        ("config", "commit.gpgsign", "false"),
        ("add", "README.txt"),
        ("commit", "-q", "-m", "synthetic kit release candidate"),
    ):
        _git(root, *arguments)
    return _git(root, "rev-parse", "HEAD")


def retarget(manifest: dict[str, Any], root: Path, commit: str) -> None:
    """Bind a manifest and its lane records to another candidate commit."""

    manifest["release"]["candidate_commit"] = commit
    for row in manifest["evidence"]:
        row["candidate_commit"] = commit
        if row["kind"] == "test_lane":
            record = json.loads((root / row["path"]).read_text("utf-8"))
            record["candidate_commit"] = commit
            rewrite(root, manifest, row["id"], _json_bytes(record))


def _runtime_version(pin: Mapping[str, Any]) -> str:
    text = str(pin["version"]).lstrip(">=^~")
    return text if text and text[0].isdigit() else "17.5"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip()


__all__ = [
    "CANDIDATE",
    "EVALUATED_AT",
    "IOS_COREML",
    "OTHER_COMMIT",
    "OTHER_TEST_ONLY_KEY",
    "SIM_COREML",
    "TEST_ONLY_KEY_ID",
    "TEST_ONLY_SIGNING_KEY",
    "entry",
    "git_checkout",
    "head_resolver",
    "lane_record",
    "promote_cell",
    "qualification_report",
    "retarget",
    "rewrite",
    "synthetic_matrix",
    "write_evidence",
]
