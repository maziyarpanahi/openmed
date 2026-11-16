"""The Swift contract-gate runner: consumer classes, the swift test call, checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "release"
RESULT_SCHEMA = (
    ROOT / "tests" / "fixtures" / "parity" / "contract_gate_result.schema.json"
)
WORKFLOW = ROOT / ".github" / "workflows" / "contract-gate.yml"
MINI_MANIFEST = "fx/contract_gate.json"
CONSUMER = "swift/OpenMedKit/Tests/OpenMedKitTests/AlphaParityTests.swift"
# Synthetic planted text: it must never reach a log line.
SENTINEL = "Quorble Zyxwick MRN-PLANTED-0042"


SPEC = importlib.util.spec_from_file_location(
    "contract_gate_swift", SCRIPTS / "contract_gate_swift.py"
)
assert SPEC is not None and SPEC.loader is not None
swift_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = swift_gate
SPEC.loader.exec_module(swift_gate)
gate = swift_gate.gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _required(consumers: list[str]) -> dict[str, Any]:
    return {"status": "required", "consumers": consumers}


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    (tmp_path / "fx").mkdir()
    shutil.copyfile(RESULT_SCHEMA, tmp_path / "fx" / "contract_gate_result.schema.json")
    _write_json(
        tmp_path / "fx" / "alpha.json",
        {"version": 1, "cases": [{"id": "a1", "text": SENTINEL}, {"id": "a2"}]},
    )
    _write_json(tmp_path / "fx" / "beta.json", {"version": 1, "items": [{}]})
    consumer = tmp_path / CONSUMER
    consumer.parent.mkdir(parents=True)
    consumer.write_text(
        "#if canImport(XCTest)\n"
        "    final class AlphaParityTests: ContractGateTestCase {}\n"
        "#endif\n"
        "final class AlphaHelperTests: XCTestCase {}\n"
        "final class AlphaVectorTests : ContractGateTestCase {}\n",
        encoding="utf-8",
    )

    def listed(path: str) -> dict[str, str]:
        return {"path": path, "sha256": _sha(tmp_path / path)}

    _write_json(
        tmp_path / MINI_MANIFEST,
        {
            "schema": gate.MANIFEST_SCHEMA,
            "version": gate.MANIFEST_VERSION,
            "algorithm": "sha256",
            "roots": ["fx"],
            "result_schema": listed("fx/contract_gate_result.schema.json"),
            "runtimes": {
                "python": {"reporter": "active", "command": "run-python"},
                "swift": {"reporter": "active", "command": "run-swift"},
            },
            "fixtures": [
                {
                    **listed("fx/alpha.json"),
                    "id": "alpha",
                    "version": 1,
                    "case_groups": [{"key": "cases", "id_field": "id"}],
                    "runtimes": {
                        "python": _required(["tests/test_alpha.py"]),
                        "swift": _required([CONSUMER]),
                    },
                },
                {
                    **listed("fx/beta.json"),
                    "id": "beta",
                    "version": 1,
                    "case_groups": [{"key": "items"}],
                    "runtimes": {
                        "python": _required(["tests/test_beta.py"]),
                        "swift": {
                            "status": "not_applicable",
                            "reason": "Synthetic Python-only contract.",
                        },
                    },
                },
            ],
        },
    )
    return tmp_path


def _result(root: Path, *, failed: bool = False) -> dict[str, Any]:
    alpha = root / "fx" / "alpha.json"
    return {
        "schema": gate.RESULT_SCHEMA,
        "schema_version": gate.RESULT_SCHEMA_VERSION,
        "runtime": "swift",
        "manifest_sha256": _sha(root / MINI_MANIFEST),
        "fixtures": [
            {
                "fixture_id": "alpha",
                "fixture_sha256": _sha(alpha),
                "passed": ["cases/a1"] if failed else ["cases/a1", "cases/a2"],
                "failed": (
                    [{"case_id": "cases/a2", "reason": "consumer_failed"}]
                    if failed
                    else []
                ),
                "skipped": [],
                "tests": {"passed": 1, "failed": int(failed), "skipped": 0},
            }
        ],
    }


def _runner(document: Any | None, calls: list[tuple[list[str], dict[str, str]]]):
    def run(command: Sequence[str], cwd: Path, env: Mapping[str, str]) -> int:
        calls.append((list(command), dict(env)))
        if document is not None:
            target = Path(env[gate.RESULTS_DIR_ENV]) / "swift.json"
            target.write_text(
                document if isinstance(document, str) else json.dumps(document),
                encoding="utf-8",
            )
        return 1

    return run


def test_consumer_classes_come_from_declared_swift_consumer_files(
    mini_repo: Path,
) -> None:
    classes = swift_gate.consumer_classes(mini_repo, MINI_MANIFEST)

    assert classes == [
        ("OpenMedKitTests", "AlphaParityTests"),
        ("OpenMedKitTests", "AlphaVectorTests"),
    ]
    assert swift_gate.xctest_filter(classes) == (
        r"^(OpenMedKitTests\.(AlphaParityTests|AlphaVectorTests))/"
    )


def test_missing_consumer_files_are_left_to_the_reporter(mini_repo: Path) -> None:
    (mini_repo / CONSUMER).unlink()

    assert swift_gate.consumer_classes(mini_repo, MINI_MANIFEST) == []
    with pytest.raises(gate.ContractGateError) as raised:
        swift_gate.run_swift(
            mini_repo, MINI_MANIFEST, mini_repo / "results", runner=_runner(None, [])
        )
    assert raised.value.code == "consumer_not_adopted"


@pytest.mark.parametrize(
    "consumer",
    [
        "tests/unit/test_alpha.py",
        "swift/OpenMedKit/Tests/AlphaParityTests.swift",
        "swift/OpenMedKit/Tests/OpenMedKitTests/alpha.json",
    ],
)
def test_consumers_outside_a_swift_test_target_are_rejected(
    mini_repo: Path, consumer: str
) -> None:
    path = mini_repo / MINI_MANIFEST
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fixtures"][0]["runtimes"]["swift"]["consumers"] = [consumer]
    _write_json(path, document)

    with pytest.raises(gate.ContractGateError) as raised:
        swift_gate.consumer_classes(mini_repo, MINI_MANIFEST)
    assert (raised.value.code, raised.value.field_path) == (
        "consumer_path_invalid",
        consumer,
    )


def test_run_sets_the_results_directory_and_filters_the_consumers(
    mini_repo: Path, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    results = mini_repo / "results"
    results.mkdir()
    (results / "swift.json").write_text("stale", encoding="utf-8")

    document, status = swift_gate.run_swift(
        mini_repo,
        MINI_MANIFEST,
        results,
        scratch_path=tmp_path / "scratch",
        skip_build=True,
        runner=_runner(_result(mini_repo), calls),
    )

    assert status == 1
    assert document["runtime"] == "swift"
    [(command, env)] = calls
    assert command == [
        "swift",
        "test",
        "--package-path",
        str(mini_repo),
        "--scratch-path",
        str(tmp_path / "scratch"),
        "--skip-build",
        "--filter",
        r"^(OpenMedKitTests\.(AlphaParityTests|AlphaVectorTests))/",
    ]
    assert env[gate.RESULTS_DIR_ENV] == str(results.resolve())


def test_cli_reports_counts_and_check_fails_on_a_failed_case(
    mini_repo: Path, monkeypatch, capsys
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(
        swift_gate, "run_swift_test", _runner(_result(mini_repo, failed=True), calls)
    )
    args = ["--root", str(mini_repo), "--manifest", MINI_MANIFEST]
    monkeypatch.setenv(gate.RESULTS_DIR_ENV, str(mini_repo / "results"))

    assert swift_gate.main(args) == 0
    assert swift_gate.main([*args, "--check"]) == 1
    output = capsys.readouterr().out
    assert "fixture=alpha passed=1 failed=1 skipped=0" in output
    assert "swift_test_exit=1 wrote swift.json" in output
    assert SENTINEL not in output
    assert str(mini_repo) not in output


@pytest.mark.parametrize(
    ("document", "code"),
    [
        (None, "result_missing"),
        ("{not json", "result_unreadable"),
        ({"schema": "openmed.contract_gate.result"}, "invalid_result"),
    ],
)
def test_runs_without_a_valid_result_exit_2(
    mini_repo: Path, monkeypatch, capsys, document: Any, code: str
) -> None:
    monkeypatch.setattr(swift_gate, "run_swift_test", _runner(document, []))
    results = mini_repo / "results"

    status = swift_gate.main(
        [
            "--root",
            str(mini_repo),
            "--manifest",
            MINI_MANIFEST,
            "--results-dir",
            str(results),
        ]
    )

    assert status == 2
    assert f"contract_gate run-swift: {code}" in capsys.readouterr().err


def test_a_result_for_another_runtime_is_invalid(mini_repo: Path) -> None:
    document = {**_result(mini_repo), "runtime": "kotlin"}

    with pytest.raises(gate.ContractGateError) as raised:
        swift_gate.run_swift(
            mini_repo, MINI_MANIFEST, mini_repo / "r", runner=_runner(document, [])
        )
    assert raised.value.code == "invalid_result"


def test_cli_needs_a_results_directory(mini_repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv(gate.RESULTS_DIR_ENV, raising=False)

    assert swift_gate.main(["--root", str(mini_repo), "--manifest", MINI_MANIFEST]) == 2
    assert gate.RESULTS_DIR_ENV in capsys.readouterr().err


def test_print_filter_needs_no_swift_toolchain(mini_repo: Path, capsys) -> None:
    status = swift_gate.main(
        ["--root", str(mini_repo), "--manifest", MINI_MANIFEST, "--print-filter"]
    )

    assert status == 0
    assert capsys.readouterr().out.strip().startswith(r"^(OpenMedKitTests\.(")


# --- The committed checkout -------------------------------------------------


def test_every_required_swift_cell_names_an_adopted_consumer_class() -> None:
    manifest = gate.load_manifest()
    classes = swift_gate.consumer_classes()

    assert manifest.runtimes["swift"].reporter == gate.REPORTER_ACTIVE
    assert ("OpenMedKitTests", "OffsetContractParityTests") in classes
    for entry in manifest.fixtures:
        cell = entry.cells["swift"]
        if cell.status != gate.STATUS_REQUIRED:
            continue
        assert cell.consumers, entry.id
        for consumer in cell.consumers:
            source = (ROOT / consumer).read_text(encoding="utf-8")
            assert ": ContractGateTestCase" in source, consumer
            assert "ContractGateReporter.read(" in source, consumer


def test_contract_gate_workflow_runs_the_swift_reporter() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    job = jobs["swift-results"]
    steps = {step.get("name"): step for step in job["steps"]}

    assert job["runs-on"].startswith("macos")
    assert "swift-results" in jobs["contract-gate"]["needs"]
    assert "swift build --build-tests" in steps["Build the Swift test bundle"]["run"]
    run = steps["Write the Swift result file"]["run"]
    assert "contract_gate_swift.py" in run
    assert "--results-dir contract-gate-results" in run
    assert "--skip-build" in run
    upload = steps["Upload the Swift result file"]
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "contract-gate-result-swift"
    assert upload["with"]["path"] == "contract-gate-results/swift.json"
