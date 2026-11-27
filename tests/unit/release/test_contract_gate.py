"""Contract-gate manifest, result schema and Python runner."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import pytest
from jsonschema.validators import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "contract_gate.py"
MANIFEST = ROOT / "tests" / "fixtures" / "parity" / "contract_gate.json"
RESULT_SCHEMA = (
    ROOT / "tests" / "fixtures" / "parity" / "contract_gate_result.schema.json"
)
PARITY_ROOTS = (ROOT / "fixtures" / "parity", ROOT / "tests" / "fixtures" / "parity")

SPEC = importlib.util.spec_from_file_location("contract_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

# Synthetic planted text: it must never reach a result file or a log line.
SENTINEL = "Quorble Zyxwick MRN-PLANTED-0042"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


ALPHA_CONSUMER = """
import json
from pathlib import Path

import pytest

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fx" / "alpha.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["id"])
def test_case(case):
    assert not case.get("broken"), case["text"]


def test_fixture_level():
    assert FIXTURE["version"] == 1
"""

BETA_CONSUMER = """
import pytest


def test_items():
    assert True


@pytest.mark.skip(reason="synthetic skip")
def test_skipped():
    raise AssertionError
"""


def _mini_manifest(root: Path) -> dict[str, Any]:
    def listed(path: str) -> dict[str, str]:
        return {"path": path, "sha256": _sha(root / path)}

    return {
        "schema": gate.MANIFEST_SCHEMA,
        "version": gate.MANIFEST_VERSION,
        "algorithm": "sha256",
        "roots": ["fx"],
        "result_schema": listed("fx/contract_gate_result.schema.json"),
        "runtimes": {
            "python": {"reporter": "active", "command": "contract_gate run-python"},
            "swift": {"reporter": "pending", "tracking": "#1"},
        },
        "fixtures": [
            {
                **listed("fx/alpha.json"),
                "id": "alpha",
                "version": 1,
                "case_groups": [
                    {"key": "cases", "id_field": "id"},
                    {"key": "rejections", "id_field": "id"},
                ],
                "companions": [listed("fx/alpha_note.txt")],
                "runtimes": {
                    "python": {
                        "status": "required",
                        "consumers": ["tests/test_alpha.py"],
                    },
                    "swift": {"status": "required", "consumers": []},
                },
            },
            {
                **listed("fx/nested/beta.json"),
                "id": "beta",
                "version": 3,
                "case_groups": [{"key": "items"}, {"key": "settings"}],
                "runtimes": {
                    "python": {
                        "status": "required",
                        "consumers": ["tests/test_beta.py"],
                    },
                    "swift": {
                        "status": "not_applicable",
                        "reason": "Synthetic Python-only contract.",
                    },
                },
            },
        ],
    }


def _build_mini_repo(root: Path, *, broken: bool = False) -> Path:
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    shutil.copyfile(RESULT_SCHEMA, root / "fx" / "contract_gate_result.schema.json")
    _write_json(
        root / "fx" / "alpha.json",
        {
            "version": 1,
            "cases": [
                {"id": "a1", "text": SENTINEL},
                {"id": "a2", "text": SENTINEL, "broken": broken},
                {"id": "a3", "text": SENTINEL},
            ],
            "rejections": [{"id": "r1", "text": SENTINEL}],
        },
    )
    (root / "fx" / "alpha_note.txt").write_text(SENTINEL + "\n", encoding="utf-8")
    _write_json(
        root / "fx" / "nested" / "beta.json",
        {"version": 3, "items": [{"value": SENTINEL}, {"value": 2}], "settings": {}},
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_alpha.py").write_text(ALPHA_CONSUMER, encoding="utf-8")
    (root / "tests" / "test_beta.py").write_text(BETA_CONSUMER, encoding="utf-8")
    _write_json(root / "fx" / "contract_gate.json", _mini_manifest(root))
    return root


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    (tmp_path / "fx").mkdir()
    return _build_mini_repo(tmp_path)


MINI_MANIFEST = "fx/contract_gate.json"


def _mini_codes(root: Path) -> list[str]:
    return [finding.code for finding in gate.verify(root, MINI_MANIFEST)]


def _edit_manifest(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = root / MINI_MANIFEST
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    _write_json(path, document)


def _junit(*cases: tuple[str, str, str]) -> str:
    children = {
        "passed": "",
        "failed": '<failure message="synthetic"/>',
        "error": '<error message="synthetic"/>',
        "skipped": '<skipped message="synthetic"/>',
    }
    body = "".join(
        f"<testcase classname={quoteattr(classname)} name={quoteattr(name)}>"
        f"{children[outcome]}</testcase>"
        for classname, name, outcome in cases
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        f'<testsuite name="pytest">{body}</testsuite></testsuites>'
    )


def _fake_runner(report: str | None, calls: list[tuple[str, ...]] | None = None):
    def runner(root: Path, consumers, junit_path: Path) -> int:
        if calls is not None:
            calls.append(tuple(consumers))
        if report is not None:
            junit_path.write_text(report, encoding="utf-8")
        return 0

    return runner


def _by_fixture(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["fixture_id"]: item for item in result["fixtures"]}


def _schema() -> dict[str, Any]:
    return json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))


# --- The committed manifest -------------------------------------------------


def test_committed_manifest_verifies_against_the_checkout(capsys) -> None:
    assert gate.verify() == []
    assert gate.main(["verify"]) == 0
    output = capsys.readouterr().out
    assert "contract_gate verify: ok fixtures=" in output
    assert str(ROOT) not in output


def test_manifest_lists_every_file_under_both_parity_roots() -> None:
    manifest = gate.load_manifest()
    listed = {manifest.path, manifest.result_schema.path}
    for entry in manifest.fixtures:
        listed.add(entry.path)
        listed.update(item.path for item in entry.companions)
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for base in PARITY_ROOTS
        for path in base.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and "__pycache__" not in path.parts
    }

    assert set(manifest.roots) == {"fixtures/parity", "tests/fixtures/parity"}
    assert listed == on_disk


def test_every_fixture_declares_each_runtime_with_reasons_for_exclusions() -> None:
    manifest = gate.load_manifest()

    assert set(manifest.runtimes) == {"python", "swift", "kotlin", "web"}
    assert manifest.runtimes["python"].reporter == gate.REPORTER_ACTIVE
    for entry in manifest.fixtures:
        assert set(entry.cells) == set(manifest.runtimes)
        assert entry.cells["python"].status == gate.STATUS_REQUIRED
        assert entry.cells["python"].consumers
        for cell in entry.cells.values():
            if cell.status == gate.STATUS_NOT_APPLICABLE:
                assert cell.reason and not cell.consumers
            else:
                assert cell.reason is None
    # The offset contract is required on web even though no web reporter
    # reads it yet: the gate must show that cell as missing, not hide it.
    assert manifest.fixture("offset_contract").cells["web"].status == "required"


def test_committed_manifest_is_in_the_canonical_refresh_format() -> None:
    text = MANIFEST.read_text(encoding="utf-8")

    assert gate.dump_json(json.loads(text)) == text


def test_committed_case_ids_are_unique_and_schema_safe() -> None:
    manifest = gate.load_manifest()
    case_schema = {**_schema()["$defs"]["case_id"]}
    validator = Draft202012Validator(case_schema)
    for entry in manifest.fixtures:
        document = json.loads((ROOT / entry.path).read_text(encoding="utf-8"))
        case_ids = gate.expected_case_ids(entry, document)
        assert case_ids, entry.id
        assert len(case_ids) == len(set(case_ids))
        for case_id in case_ids:
            validator.validate(case_id)


def test_result_schema_is_a_versioned_draft_2020_12_schema() -> None:
    schema = _schema()

    Draft202012Validator.check_schema(schema)
    assert schema["$id"].endswith("/contract_gate_result.schema.json")
    assert schema["schema_version"] == gate.RESULT_SCHEMA_VERSION
    assert schema["properties"]["schema"]["const"] == gate.RESULT_SCHEMA
    assert schema["properties"]["schema_version"]["const"] == (
        gate.RESULT_SCHEMA_VERSION
    )
    assert schema["additionalProperties"] is False
    fixture_result = schema["$defs"]["fixture_result"]
    assert fixture_result["additionalProperties"] is False
    assert schema["$defs"]["case_outcome"]["additionalProperties"] is False


# --- verify on a synthetic repository ---------------------------------------


def test_synthetic_manifest_verifies(mini_repo: Path) -> None:
    assert _mini_codes(mini_repo) == []


def test_unlisted_file_fails_verify(mini_repo: Path) -> None:
    _write_json(mini_repo / "fx" / "nested" / "gamma.json", {"version": 1})
    (mini_repo / "fx" / ".DS_Store").write_bytes(b"\0")

    findings = gate.verify(mini_repo, MINI_MANIFEST)

    assert [(f.code, f.path) for f in findings] == [
        ("unlisted_file", "fx/nested/gamma.json")
    ]


@pytest.mark.parametrize(
    "relative", ["fx/alpha.json", "fx/alpha_note.txt", "fx/nested/beta.json"]
)
def test_fixture_edited_without_its_digest_fails_until_refresh(
    mini_repo: Path, relative: str
) -> None:
    target = mini_repo / relative
    target.write_bytes(target.read_bytes().replace(b"Quorble", b"Quarble"))

    findings = gate.verify(mini_repo, MINI_MANIFEST)

    assert [(f.code, f.path) for f in findings] == [("digest_mismatch", relative)]
    assert gate.refresh(mini_repo, MINI_MANIFEST) == [relative]
    assert _mini_codes(mini_repo) == []
    assert gate.refresh(mini_repo, MINI_MANIFEST) == []


def test_missing_listed_file_fails_verify(mini_repo: Path) -> None:
    (mini_repo / "fx" / "alpha_note.txt").unlink()

    assert _mini_codes(mini_repo) == ["file_missing"]
    with pytest.raises(gate.ContractGateError) as error:
        gate.refresh(mini_repo, MINI_MANIFEST)
    assert error.value.code == "file_missing"


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(document: dict[str, Any]) -> None:
        target = document
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return mutate


_DELETE = object()
_BETA_SWIFT = ("fixtures", 1, "runtimes", "swift")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_set((*_BETA_SWIFT, "reason"), _DELETE), "not_applicable_without_reason"),
        (_set((*_BETA_SWIFT, "reason"), "  "), "not_applicable_without_reason"),
        (
            _set((*_BETA_SWIFT, "consumers"), ["tests/test_beta.py"]),
            "not_applicable_with_consumers",
        ),
        (_set(_BETA_SWIFT, _DELETE), "runtime_cell_missing"),
        (
            _set(("fixtures", 1, "runtimes", "kotlin"), {"status": "required"}),
            "unknown_runtime",
        ),
        (_set((*_BETA_SWIFT, "status"), "optional"), "invalid_status"),
        (_set(("fixtures", 1, "id"), "alpha"), "duplicate_fixture_id"),
        (_set(("fixtures", 1, "id"), "Beta Fixture"), "fixture_id_invalid"),
        (_set(("fixtures", 1, "path"), "fx/../fx/nested/beta.json"), "path_invalid"),
        (_set(("fixtures", 1, "path"), "/fx/nested/beta.json"), "path_invalid"),
        (_set(("fixtures", 1, "case_groups"), []), "case_groups_missing"),
        (_set(("fixtures", 1, "text"), SENTINEL), "unknown_field"),
        (_set(("schema",), "openmed.other"), "unknown_schema"),
        (_set(("runtimes", "swift", "tracking"), _DELETE), "reporter_tracking_missing"),
        (_set(("runtimes", "python", "command"), _DELETE), "reporter_command_missing"),
        (_set(("runtimes", "python"), _DELETE), "runtime_cell_missing"),
    ],
)
def test_malformed_manifest_entries_fail_verify(
    mini_repo: Path, mutate: Callable[[dict[str, Any]], None], code: str
) -> None:
    _edit_manifest(mini_repo, mutate)

    findings = gate.verify(mini_repo, MINI_MANIFEST)

    assert [finding.code for finding in findings] == [code]
    assert SENTINEL not in findings[0].render()


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_set(("fixtures", 1, "version"), 4), "version_mismatch"),
        (
            _set(("fixtures", 0, "case_groups", 1, "key"), "missing"),
            "case_group_missing",
        ),
        (
            _set(("fixtures", 0, "case_groups", 1, "id_field"), "text"),
            "case_id_invalid",
        ),
        (
            _set(("fixtures", 0, "runtimes", "python", "consumers"), ["tests/none.py"]),
            "consumer_missing",
        ),
        (
            _set(("fixtures", 0, "runtimes", "python", "consumers"), []),
            "consumer_required",
        ),
        (_set(("fixtures", 1, "path"), "tests/test_beta.py"), "path_outside_roots"),
        (
            _set(
                ("fixtures", 1, "companions"),
                [{"path": "fx/alpha_note.txt", "sha256": "0" * 64}],
            ),
            "duplicate_path",
        ),
    ],
)
def test_inconsistent_fixtures_and_cells_fail_verify(
    mini_repo: Path, mutate: Callable[[dict[str, Any]], None], code: str
) -> None:
    _edit_manifest(mini_repo, mutate)

    assert code in _mini_codes(mini_repo)


def test_duplicate_case_ids_fail_verify(mini_repo: Path) -> None:
    path = mini_repo / "fx" / "alpha.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cases"][2]["id"] = "a1"
    _write_json(path, document)
    gate.refresh(mini_repo, MINI_MANIFEST)

    assert _mini_codes(mini_repo) == ["case_id_duplicate"]


def test_case_id_with_a_trailing_newline_fails_verify(mini_repo: Path) -> None:
    # Swift, Kotlin and web match the whole id, so Python must not accept the
    # line break that a ``$`` anchor would let through.
    path = mini_repo / "fx" / "alpha.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["cases"][2]["id"] = "a3\n"
    _write_json(path, document)
    gate.refresh(mini_repo, MINI_MANIFEST)

    assert _mini_codes(mini_repo) == ["case_id_invalid"]


def test_required_cell_of_a_pending_runtime_may_wait_for_its_consumer(
    mini_repo: Path,
) -> None:
    manifest = gate.load_manifest(mini_repo, MINI_MANIFEST)

    assert manifest.runtimes["swift"].reporter == gate.REPORTER_PENDING
    assert manifest.fixture("alpha").cells["swift"].consumers == ()
    assert _mini_codes(mini_repo) == []


def test_unreadable_manifest_exits_2(tmp_path: Path, capsys) -> None:
    (tmp_path / "fx").mkdir()
    (tmp_path / MINI_MANIFEST).write_text("{", encoding="utf-8")

    assert (
        gate.main(["--root", str(tmp_path), "--manifest", MINI_MANIFEST, "verify"]) == 2
    )
    assert "manifest_unreadable" in capsys.readouterr().err


def test_verify_cli_reports_codes_and_relative_paths_only(
    mini_repo: Path, capsys
) -> None:
    (mini_repo / "fx" / "alpha.json").write_text(
        json.dumps({"version": 1, "cases": [], "note": SENTINEL}), encoding="utf-8"
    )

    status = gate.main(
        ["--root", str(mini_repo), "--manifest", MINI_MANIFEST, "verify"]
    )

    output = capsys.readouterr().out
    assert status == 1
    assert "digest_mismatch fixture=alpha path=fx/alpha.json" in output
    assert SENTINEL not in output
    assert str(mini_repo) not in output


# --- case ids ---------------------------------------------------------------


def test_case_ids_follow_keyed_index_and_object_groups(mini_repo: Path) -> None:
    manifest = gate.load_manifest(mini_repo, MINI_MANIFEST)
    alpha = json.loads((mini_repo / "fx" / "alpha.json").read_text(encoding="utf-8"))
    beta = json.loads(
        (mini_repo / "fx" / "nested" / "beta.json").read_text(encoding="utf-8")
    )

    assert gate.expected_case_ids(manifest.fixture("alpha"), alpha) == (
        "cases/a1",
        "cases/a2",
        "cases/a3",
        "rejections/r1",
    )
    assert gate.expected_case_ids(manifest.fixture("beta"), beta) == (
        "items/0",
        "items/1",
        "settings",
    )


# --- run-python with an injected pytest runner ------------------------------

ALPHA = "tests.test_alpha"
BETA = "tests.test_beta"


def _run(mini_repo: Path, report: str | None, calls=None) -> dict[str, Any]:
    result = gate.run_python(
        mini_repo, MINI_MANIFEST, runner=_fake_runner(report, calls)
    )
    assert gate.validate_result(result, _schema()) == []
    assert SENTINEL not in json.dumps(result)
    return result


def test_runner_runs_every_python_consumer_once_and_records_digests(
    mini_repo: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    report = _junit(
        (ALPHA, "test_case[a1]", "passed"),
        (ALPHA, "test_case[a2]", "passed"),
        (ALPHA, "test_case[a3]", "passed"),
        (ALPHA, "test_fixture_level", "passed"),
        (BETA, "test_items", "passed"),
        (BETA, "test_skipped", "skipped"),
    )

    result = _run(mini_repo, report, calls)
    fixtures = _by_fixture(result)

    assert calls == [("tests/test_alpha.py", "tests/test_beta.py")]
    assert result["runtime"] == "python"
    assert result["manifest_sha256"] == _sha(mini_repo / MINI_MANIFEST)
    assert fixtures["alpha"]["fixture_sha256"] == _sha(mini_repo / "fx/alpha.json")
    assert fixtures["alpha"]["passed"] == [
        "cases/a1",
        "cases/a2",
        "cases/a3",
        "rejections/r1",
    ]
    assert fixtures["alpha"]["tests"] == {"passed": 4, "failed": 0, "skipped": 0}
    assert fixtures["beta"]["passed"] == ["items/0", "items/1", "settings"]
    assert fixtures["beta"]["tests"] == {"passed": 1, "failed": 0, "skipped": 1}
    assert fixtures["alpha"]["failed"] == fixtures["alpha"]["skipped"] == []


def test_parametrized_failure_fails_only_its_case(mini_repo: Path) -> None:
    report = _junit(
        (ALPHA, "test_case[a1]", "passed"),
        (ALPHA, "test_case[a2]", "failed"),
        (ALPHA, "test_case[cases/a3]", "error"),
        (ALPHA, "test_fixture_level", "passed"),
        (BETA, "test_items", "passed"),
    )

    alpha = _by_fixture(_run(mini_repo, report))["alpha"]

    assert alpha["passed"] == ["cases/a1", "rejections/r1"]
    assert alpha["failed"] == [
        {"case_id": "cases/a2", "reason": "consumer_failed"},
        {"case_id": "cases/a3", "reason": "consumer_error"},
    ]


def test_fixture_level_failure_fails_every_case(mini_repo: Path) -> None:
    report = _junit(
        (ALPHA, "test_case[a1]", "passed"),
        (ALPHA, "test_fixture_level", "failed"),
        (BETA, "test_items", "passed"),
    )

    alpha = _by_fixture(_run(mini_repo, report))["alpha"]

    assert alpha["passed"] == []
    assert {item["reason"] for item in alpha["failed"]} == {"consumer_failed"}
    assert len(alpha["failed"]) == 4


def test_skipped_and_unexercised_cases_are_not_passes(mini_repo: Path) -> None:
    report = _junit(
        (ALPHA, "test_case[a1]", "passed"),
        (ALPHA, "test_case[a2]", "skipped"),
        (BETA, "test_items", "skipped"),
    )

    fixtures = _by_fixture(_run(mini_repo, report))

    assert fixtures["alpha"]["passed"] == ["cases/a1"]
    assert fixtures["alpha"]["skipped"] == [
        {"case_id": "cases/a2", "reason": "consumer_skipped"}
    ]
    assert fixtures["alpha"]["failed"] == [
        {"case_id": "cases/a3", "reason": "case_not_exercised"},
        {"case_id": "rejections/r1", "reason": "case_not_exercised"},
    ]
    assert [item["reason"] for item in fixtures["beta"]["skipped"]] == [
        "consumer_skipped"
    ] * 3


def test_index_case_ids_only_match_their_full_id(mini_repo: Path) -> None:
    report = _junit(
        (ALPHA, "test_fixture_level", "passed"),
        (BETA, "test_value[items/1]", "failed"),
        (BETA, "test_value[0]", "passed"),
    )

    beta = _by_fixture(_run(mini_repo, report))["beta"]

    # "0" is not a case id: the test covers every beta case, while the full
    # id "items/1" narrows the failure to that case.
    assert beta["passed"] == ["items/0", "settings"]
    assert beta["failed"] == [{"case_id": "items/1", "reason": "consumer_failed"}]


def test_collection_error_and_missing_report_fail_closed(mini_repo: Path) -> None:
    collection = _junit(("", ALPHA, "error"), (BETA, "test_items", "passed"))

    fixtures = _by_fixture(_run(mini_repo, collection))
    assert {item["reason"] for item in fixtures["alpha"]["failed"]} == {
        "consumer_error"
    }
    assert fixtures["beta"]["passed"] == ["items/0", "items/1", "settings"]

    missing = _by_fixture(_run(mini_repo, None))
    for item in missing.values():
        assert item["passed"] == []
        assert {case["reason"] for case in item["failed"]} == {"consumer_not_collected"}


def test_missing_consumer_and_unreadable_fixture_are_recorded(
    mini_repo: Path,
) -> None:
    (mini_repo / "tests" / "test_beta.py").unlink()
    (mini_repo / "fx" / "alpha.json").write_text("{", encoding="utf-8")

    fixtures = _by_fixture(_run(mini_repo, _junit()))

    assert fixtures["alpha"]["errors"] == ["fixture_unreadable"]
    assert fixtures["alpha"]["passed"] == fixtures["alpha"]["failed"] == []
    assert {item["reason"] for item in fixtures["beta"]["failed"]} == {
        "consumer_missing"
    }


def test_fixture_filter_and_not_applicable_runtime(mini_repo: Path) -> None:
    calls: list[tuple[str, ...]] = []
    result = gate.run_python(
        mini_repo,
        MINI_MANIFEST,
        fixture_ids=["beta"],
        runner=_fake_runner(_junit((BETA, "test_items", "passed")), calls),
    )

    assert calls == [("tests/test_beta.py",)]
    assert [item["fixture_id"] for item in result["fixtures"]] == ["beta"]
    with pytest.raises(gate.ContractGateError) as error:
        gate.run_python(mini_repo, MINI_MANIFEST, fixture_ids=["gamma"])
    assert error.value.code == "unknown_fixture"


def test_parse_junit_maps_params_classes_and_collection_errors(
    tmp_path: Path,
) -> None:
    report = tmp_path / "junit.xml"
    report.write_text(
        _junit(
            ("tests.unit.core.test_x", "test_case[cjk-01]", "passed"),
            ("tests.unit.core.test_x.TestGroup", "test_method", "skipped"),
            ("", "tests.unit.core.test_y", "error"),
        ),
        encoding="utf-8",
    )

    outcomes = gate.parse_junit(report)

    assert outcomes == [
        gate.ConsumerOutcome("tests.unit.core.test_x", "test_case", "cjk-01", "passed"),
        gate.ConsumerOutcome(
            "tests.unit.core.test_x.TestGroup", "test_method", None, "skipped"
        ),
        gate.ConsumerOutcome("tests.unit.core.test_y", "", None, "error"),
    ]
    assert gate._belongs_to(outcomes[1].classname, ("tests.unit.core.test_x",))
    assert not gate._belongs_to("tests.unit.core.test_xy", ("tests.unit.core.test_x",))


# --- run-python end to end --------------------------------------------------


@pytest.mark.integration
def test_run_python_cli_runs_real_consumers_without_echoing_values(
    tmp_path: Path, capsys
) -> None:
    repo = tmp_path / "repo"
    (repo / "fx").mkdir(parents=True)
    _build_mini_repo(repo, broken=True)
    output = tmp_path / "out" / "python.json"
    base = ["--root", str(repo), "--manifest", MINI_MANIFEST, "run-python"]

    assert gate.main([*base, "--output", str(output)]) == 0
    assert gate.main([*base, "--output", str(output), "--check"]) == 1

    text = output.read_text(encoding="utf-8")
    result = json.loads(text)
    fixtures = _by_fixture(result)
    logs = capsys.readouterr()
    assert gate.validate_result(result, _schema()) == []
    assert fixtures["alpha"]["passed"] == ["cases/a1", "cases/a3", "rejections/r1"]
    assert fixtures["alpha"]["failed"] == [
        {"case_id": "cases/a2", "reason": "consumer_failed"}
    ]
    assert fixtures["beta"]["passed"] == ["items/0", "items/1", "settings"]
    assert SENTINEL not in text + logs.out + logs.err
    assert str(tmp_path) not in logs.out + logs.err


@pytest.mark.integration
def test_run_python_records_the_committed_offset_contract(tmp_path: Path) -> None:
    result = gate.run_python(fixture_ids=["offset_contract"])
    fixture = result["fixtures"][0]
    document = json.loads(
        (ROOT / "tests" / "fixtures" / "parity" / "offset_contract.json").read_text(
            encoding="utf-8"
        )
    )

    assert gate.validate_result(result, _schema()) == []
    assert (
        fixture["fixture_sha256"]
        == gate.load_manifest().fixture("offset_contract").sha256
    )
    assert fixture["passed"] == [f"cases/{case['id']}" for case in document["cases"]]
    assert fixture["failed"] == fixture["skipped"] == []
    assert fixture["tests"]["passed"] > len(document["cases"])


def test_run_python_cli_needs_an_output(mini_repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv(gate.RESULTS_DIR_ENV, raising=False)

    status = gate.main(
        ["--root", str(mini_repo), "--manifest", MINI_MANIFEST, "run-python"]
    )

    assert status == 2
    assert gate.RESULTS_DIR_ENV in capsys.readouterr().err


def test_results_dir_environment_names_the_runtime_file(
    mini_repo: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(gate.RESULTS_DIR_ENV, str(tmp_path / "results"))
    monkeypatch.setattr(
        gate, "run_pytest", _fake_runner(_junit((BETA, "test_items", "passed")))
    )

    status = gate.main(
        ["--root", str(mini_repo), "--manifest", MINI_MANIFEST, "run-python"]
    )

    assert status == 0
    assert (tmp_path / "results" / "python.json").is_file()


# --- validate-result --------------------------------------------------------


def _valid_result() -> dict[str, Any]:
    return {
        "schema": gate.RESULT_SCHEMA,
        "schema_version": gate.RESULT_SCHEMA_VERSION,
        "runtime": "swift",
        "manifest_sha256": "a" * 64,
        "fixtures": [
            {
                "fixture_id": "offset_contract",
                "fixture_sha256": "b" * 64,
                "passed": ["cases/cjk-01"],
                "failed": [{"case_id": "cases/cjk-02", "reason": "offset_mismatch"}],
                "skipped": [],
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda doc: doc["fixtures"][0].update(text=SENTINEL),
            "result_schema_violation",
        ),
        (
            lambda doc: doc["fixtures"][0]["failed"][0].update(message=SENTINEL),
            "result_schema_violation",
        ),
        (
            lambda doc: doc["fixtures"][0]["failed"][0].update(reason=SENTINEL),
            "result_schema_violation",
        ),
        (
            lambda doc: doc["fixtures"][0]["passed"].append(SENTINEL),
            "result_schema_violation",
        ),
        (lambda doc: doc.update(schema_version=2), "result_schema_violation"),
        (
            lambda doc: doc.update(manifest_sha256="sha256:abc"),
            "result_schema_violation",
        ),
        (
            lambda doc: doc["fixtures"][0]["passed"].append("cases/cjk-02"),
            "case_reported_twice",
        ),
        (
            lambda doc: doc["fixtures"].append(copy.deepcopy(doc["fixtures"][0])),
            "fixture_reported_twice",
        ),
    ],
)
def test_invalid_results_fail_validation_without_echoing_values(mutate, code) -> None:
    document = _valid_result()
    assert gate.validate_result(document, _schema()) == []
    mutate(document)

    findings = gate.validate_result(document, _schema())

    assert findings and {finding.code for finding in findings} == {code}
    assert all(SENTINEL not in finding.render() for finding in findings)


def test_validate_result_cli_exit_codes(tmp_path: Path, capsys) -> None:
    good = tmp_path / "swift.json"
    bad = tmp_path / "web.json"
    broken = tmp_path / "kotlin.json"
    _write_json(good, _valid_result())
    invalid = _valid_result()
    invalid["fixtures"][0]["deidentified_text"] = SENTINEL
    _write_json(bad, invalid)
    broken.write_text("{", encoding="utf-8")

    assert gate.main(["validate-result", str(good)]) == 0
    assert gate.main(["validate-result", str(good), str(bad)]) == 1
    assert gate.main(["validate-result", str(broken)]) == 2
    output = capsys.readouterr().out
    assert "ok file=swift.json" in output
    assert "result_schema_violation" in output
    assert "result_unreadable file=kotlin.json" in output
    assert SENTINEL not in output
    assert str(tmp_path) not in output
