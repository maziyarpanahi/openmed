"""The seeded service fault matrix meets documented outcomes, offline.

Rows that master cannot meet yet are declared as known gaps in the plan, with
the owning issue, and pinned below with ``xfail(strict=True)`` so the matrix
flags them the moment the owning work lands.
"""

from __future__ import annotations

import os
import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from openmed.core.offline import HF_OFFLINE_ENV_VARS
from openmed.service import jobs
from tests.faults.plan import (
    CATEGORIES,
    OUTCOMES,
    PHASES,
    FaultPlan,
    canonical_json,
)
from tests.faults.runner import main, run_case, run_plan, validate_manifest
from tests.faults.scenarios import CANARIES

PLAN = FaultPlan.load()
# Closed bug reports whose behaviour the owning issue now covers.
FOLDED_REPORTS = {"jobs-queue.shutdown-cancels-queued": "#3467 folded into #3574"}


@pytest.fixture(scope="module")
def matrix(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OPENMED_PROFILE", "test")
        workdir = tmp_path_factory.mktemp("fault-matrix")
        environment = {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS}
        primitives = (
            os.replace,
            os.fsync,
            tempfile.NamedTemporaryFile,
            Path.read_text,
            socket.create_connection,
            socket.socket.connect,
            jobs.ThreadPoolExecutor,
        )
        manifest = run_plan(PLAN, workdir)
        after = (
            os.replace,
            os.fsync,
            tempfile.NamedTemporaryFile,
            Path.read_text,
            socket.create_connection,
            socket.socket.connect,
            jobs.ThreadPoolExecutor,
        )
        assert after == primitives
        assert jobs.ThreadPoolExecutor is ThreadPoolExecutor
        assert {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS} == (
            environment
        )
        return {"manifest": manifest, "workdir": str(workdir)}


def test_manifest_conforms_and_carries_no_values(matrix: dict[str, Any]) -> None:
    manifest = matrix["manifest"]
    validate_manifest(manifest)
    encoded = canonical_json(manifest).decode("ascii")

    for canary in CANARIES:
        assert canary not in encoded
    assert matrix["workdir"] not in encoded
    assert os.sep + "state" not in encoded
    assert "Errno" not in encoded
    assert manifest["plan_digest"] == PLAN.digest()
    assert [row["case_id"] for row in manifest["cases"]] == [
        case.case_id for case in PLAN.cases
    ]


def test_documented_cases_reach_their_outcome(
    matrix: dict[str, Any],
    subtests,
) -> None:
    for row in matrix["manifest"]["cases"]:
        with subtests.test(case=row["case_id"]):
            if row["known_gap_issue"] is None:
                assert row["status"] == "passed", row
                assert row["observed_outcome"] == row["expected_outcome"]
            else:
                assert row["status"] in {"known_gap", "passed"}, row
    assert matrix["manifest"]["summary"]["failed"] == 0


def test_every_category_has_positive_and_negative_controls(
    matrix: dict[str, Any],
) -> None:
    rows = matrix["manifest"]["cases"]
    for category in CATEGORIES:
        controlled = [
            row
            for row in rows
            if row["category"] == category
            and row["status"] == "passed"
            and row["control_seam_hits"] > 0
            and row["control_outcome"] == row["baseline_outcome"]
            and row["faults_fired"] > 0
        ]
        assert controlled, category
    for row in rows:
        if row["status"] == "passed":
            assert row["control_seam_hits"] > 0 and row["faults_fired"] > 0, row


def test_commit_points_produce_one_documented_result(matrix: dict[str, Any]) -> None:
    rows = matrix["manifest"]["cases"]
    phases_by_component: dict[str, set[str]] = {}
    for row in rows:
        assert row["expected_outcome"] in OUTCOMES
        assert row["phase"] in PHASES
        phases_by_component.setdefault(row["component"], set()).add(row["phase"])
        if row["status"] == "passed":
            assert row["observed_outcome"] == row["expected_outcome"]
        if row["require_no_residue"] and row["status"] == "passed":
            assert row["residue_count"] == 0
    assert all(len(phases) >= 2 for phases in phases_by_component.values())
    for row in rows:
        if row["category"] in {"cancellation", "process"} and row["status"] == "passed":
            assert row["observed_outcome"] != "ambiguous"


def test_matrix_is_reproducible_offline_and_across_seeds(
    matrix: dict[str, Any],
    tmp_path: Path,
) -> None:
    output = tmp_path / "service-fault-manifest.json"
    assert main(["--output", str(output)]) == 0
    assert output.read_bytes() == canonical_json(matrix["manifest"]) + b"\n"

    reseeded = run_plan(PLAN.with_seed(PLAN.seed + 101), tmp_path / "reseeded")
    original = matrix["manifest"]["cases"]
    assert [row["status"] for row in reseeded["cases"]] == [
        row["status"] for row in original
    ]
    assert all(
        left["case_digest"] != right["case_digest"]
        for left, right in zip(reseeded["cases"], original, strict=True)
    )


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param(
            case.case_id,
            id=case.case_id,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    f"#{case.known_gap.issue}: {case.known_gap.reason_code}"
                    + (
                        f" ({FOLDED_REPORTS[case.case_id]})"
                        if case.case_id in FOLDED_REPORTS
                        else ""
                    )
                ),
            ),
        )
        for case in PLAN.cases
        if case.known_gap is not None
    ],
)
def test_known_gap_reaches_documented_outcome(case_id: str, tmp_path: Path) -> None:
    row = run_case(PLAN.resolve(PLAN.case(case_id)), tmp_path)
    assert row["status"] == "passed", row
