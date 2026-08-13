"""Offline contract tests for DP synthetic tabular generation (issue #1270)."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from openmed.eval.utility import (
    DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING,
    DEFAULT_SYNTHETIC_MARGINAL_MAE_CAP,
    membership_inference_risk_report,
    synthetic_tabular_utility_report,
)
from openmed.interop.bridges.dp_synth import (
    DPSynthBridge,
    DPSynthEngineUnavailable,
    DPSynthLicenseError,
    DPSynthProtocolError,
)
from openmed.risk import BudgetExceeded, DPGenerationBudgetAccountant, EpsilonPolicy
from openmed.structured import (
    SyntheticDataGateError,
    generate_synthetic,
    read_table,
    write_table,
)

ENGINE_SOURCE = r"""
import json
import pathlib
import sys

capture = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
request = json.load(sys.stdin)
if request["operation"] == "capabilities":
    license_id = "GPL-3.0-only" if mode == "gpl" else "Apache-2.0"
    response = {
        "protocol_version": 1,
        "engine": {
            "name": "synthetic-offline-test-engine",
            "version": "1.0",
            "license": license_id,
            "family": "graphical",
        },
        "capabilities": {
            "input_contract": "aggregate-statistics-only",
            "accepts_raw_rows": False,
        },
    }
else:
    capture.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    if mode == "shifted":
        base = [
            {
                "age": "99",
                "score": "999",
                "group": "shifted",
                "marker": "synthetic-shift",
            }
        ]
    else:
        base = [
            {"age": "30", "score": "110", "group": "a", "marker": "holdout-1"},
            {"age": "31", "score": "111", "group": "b", "marker": "holdout-2"},
            {"age": "32", "score": "112", "group": "a", "marker": "holdout-3"},
        ]
    rows = [dict(base[index % len(base)]) for index in range(request["row_count"])]
    response = {
        "protocol_version": 1,
        "rows": rows,
        "privacy": {
            "epsilon_spent": request["privacy"]["epsilon"],
            "delta_spent": request["privacy"]["delta"],
        },
    }
json.dump(response, sys.stdout, allow_nan=False, sort_keys=True)
"""


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(9):
        rows.append(
            {
                "age": 21 + index,
                "score": 101 + index,
                "group": "a" if index % 2 == 0 else "b",
                "marker": "private-source-canary" if index == 0 else f"train-{index}",
            }
        )
    rows.extend(
        [
            {"age": 30, "score": 110, "group": "a", "marker": "holdout-1"},
            {"age": 31, "score": 111, "group": "b", "marker": "holdout-2"},
            {"age": 32, "score": 112, "group": "a", "marker": "holdout-3"},
        ]
    )
    return rows


def _source_table(tmp_path: Path) -> Path:
    source = tmp_path / "source.csv"
    write_table(source, _source_rows())
    return source


def _engine_command(
    tmp_path: Path,
    *,
    mode: str = "good",
) -> tuple[tuple[str, ...], Path]:
    script = tmp_path / f"engine-{mode}.py"
    capture = tmp_path / f"capture-{mode}.json"
    script.write_text(ENGINE_SOURCE, encoding="utf-8")
    return (sys.executable, str(script), str(capture), mode), capture


def _all_mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        keys.update(str(key).casefold() for key in value)
        for item in value.values():
            keys.update(_all_mapping_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            keys.update(_all_mapping_keys(item))
    return keys


def test_generate_synthetic_writes_csv_privacy_report_and_aggregate_payload(
    tmp_path: Path,
) -> None:
    source = _source_table(tmp_path)
    command, capture = _engine_command(tmp_path)
    output = tmp_path / "release.csv"
    report_path = tmp_path / "release.privacy.json"

    result = generate_synthetic(
        source,
        epsilon=0.05,
        delta=1e-8,
        output_path=output,
        report_path=report_path,
        engine_command=command,
    )

    assert result.output_path == output
    assert result.report_path == report_path
    assert result.row_count == len(_source_rows())
    assert len(read_table(output)) == len(_source_rows())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["privacy"]["epsilon_spent"] == 0.05
    assert report["privacy"]["delta_spent"] == 1e-8
    assert report["engine"]["license"] == "Apache-2.0"
    assert report["gates"] == {
        "membership_inference_passed": True,
        "passed": True,
        "utility_passed": True,
    }
    assert report["output"]["sha256"]
    assert "private-source-canary" not in json.dumps(report, sort_keys=True)

    request = json.loads(capture.read_text(encoding="utf-8"))
    assert set(request) == {
        "operation",
        "privacy",
        "protocol_version",
        "row_count",
        "schema",
        "seed",
        "statistics",
    }
    assert request["operation"] == "fit_synthesize"
    assert request["statistics"]["source_row_count"] == 9
    assert request["row_count"] == 12
    forbidden = {"data", "raw_rows", "records", "rows", "source_path", "source_rows"}
    assert _all_mapping_keys(request).isdisjoint(forbidden)
    assert str(source) not in json.dumps(request, sort_keys=True)


def test_generate_synthetic_writes_parquet_when_columnar_extra_is_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    source = _source_table(tmp_path)
    command, _capture = _engine_command(tmp_path, mode="parquet")
    output = tmp_path / "release.parquet"

    result = generate_synthetic(
        source,
        epsilon=0.05,
        delta=1e-8,
        output_path=output,
        engine_command=command,
    )

    assert result.output_path == output
    assert len(read_table(output)) == 12
    assert result.privacy_report["output"]["format"] == "parquet"


def test_budget_gate_blocks_before_aggregate_payload_is_sent(tmp_path: Path) -> None:
    source = _source_table(tmp_path)
    command, capture = _engine_command(tmp_path, mode="budget")
    accountant = DPGenerationBudgetAccountant(
        {
            "local": EpsilonPolicy(
                scope="local",
                max_epsilon=0.01,
                max_delta=1e-6,
                composition="basic",
            )
        }
    )

    with pytest.raises(BudgetExceeded):
        generate_synthetic(
            source,
            epsilon=0.02,
            delta=1e-8,
            output_path=tmp_path / "blocked.csv",
            scope="local",
            accountant=accountant,
            engine_command=command,
        )

    assert accountant.ledger == ()
    assert not capture.exists()
    assert not (tmp_path / "blocked.csv").exists()


def test_missing_optional_engine_has_actionable_error_and_core_imports(
    tmp_path: Path,
) -> None:
    source = _source_table(tmp_path)

    with pytest.raises(DPSynthEngineUnavailable) as raised:
        generate_synthetic(
            source,
            epsilon=0.05,
            delta=1e-8,
            engine_command=str(tmp_path / "missing-engine"),
        )

    assert "Install a separate" in str(raised.value)
    assert "engine_command" in str(raised.value)
    import openmed

    assert openmed is not None


def test_non_permissive_engine_is_refused_before_fitting(tmp_path: Path) -> None:
    source = _source_table(tmp_path)
    command, capture = _engine_command(tmp_path, mode="gpl")

    with pytest.raises(DPSynthLicenseError, match="permissive SPDX"):
        generate_synthetic(
            source,
            epsilon=0.05,
            delta=1e-8,
            engine_command=command,
        )

    assert not capture.exists()


def test_bridge_rejects_raw_row_shaped_statistics(tmp_path: Path) -> None:
    command, capture = _engine_command(tmp_path, mode="raw")
    bridge = DPSynthBridge(command)

    with pytest.raises(DPSynthProtocolError, match="raw-row-shaped"):
        bridge.fit_synthesize(
            [{"name": "age", "kind": "integer", "nullable": False}],
            {"rows": [{"age": 30}]},
            epsilon=0.1,
            delta=1e-8,
            row_count=1,
        )

    assert not capture.exists()


def test_utility_and_membership_regressions_fail_documented_gates() -> None:
    rows = _source_rows()
    members = rows[:9]
    nonmembers = rows[9:]
    memorized = [dict(row) for row in members]
    shifted = [
        {"age": 99, "score": 999, "group": "shifted", "marker": "shifted"}
        for _ in range(12)
    ]

    membership = membership_inference_risk_report(
        members,
        nonmembers,
        memorized,
    )
    utility = synthetic_tabular_utility_report(nonmembers, shifted)

    assert membership.advantage == pytest.approx(1.0)
    assert membership.advantage_ceiling == DEFAULT_MEMBERSHIP_ADVANTAGE_CEILING
    assert membership.passed is False
    assert utility.one_way_marginal_mae > DEFAULT_SYNTHETIC_MARGINAL_MAE_CAP
    assert utility.two_way_marginal_mae > DEFAULT_SYNTHETIC_MARGINAL_MAE_CAP
    assert utility.passed is False


def test_generate_synthetic_does_not_write_a_failed_release(tmp_path: Path) -> None:
    source = _source_table(tmp_path)
    command, _capture = _engine_command(tmp_path, mode="shifted")
    output = tmp_path / "unsafe.csv"
    report = tmp_path / "unsafe.privacy.json"

    with pytest.raises(SyntheticDataGateError) as raised:
        generate_synthetic(
            source,
            epsilon=0.05,
            delta=1e-8,
            output_path=output,
            report_path=report,
            engine_command=command,
        )

    assert raised.value.privacy_report["gates"]["passed"] is False
    assert not output.exists()
    assert not report.exists()


def test_public_dp_synthetic_api_is_exported() -> None:
    import openmed.eval as evaluation
    import openmed.structured as structured

    assert "generate_synthetic" in structured.__all__
    assert structured.generate_synthetic is generate_synthetic
    assert "synthetic_tabular_utility_report" in evaluation.__all__
    assert "membership_inference_risk_report" in evaluation.__all__
