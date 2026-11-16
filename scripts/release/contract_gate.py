#!/usr/bin/env python3
"""Verify the shared parity fixtures and record Python contract-gate results.

The contract gate is one manifest, ``tests/fixtures/parity/contract_gate.json``.
It names every file under the shared parity roots, pins each fixture's SHA-256
digest and version, says how a fixture's case ids are derived, and declares for
every runtime (Python, Swift, Kotlin, web) whether the fixture is ``required``
or ``not_applicable`` (with a reason). Each runtime reporter writes one result
file that matches ``tests/fixtures/parity/contract_gate_result.schema.json``:
fixture ids, the digest of the fixture the runtime actually read, the case ids
that passed, failed or were skipped, and reason codes. Result files never carry
fixture text, inputs, outputs or assertion messages. ``docs/contract-gate.md``
documents the manifest format, the case-id rules and the reporter contract.

Usage::

    python scripts/release/contract_gate.py verify
    python scripts/release/contract_gate.py run-python --results-dir results
    python scripts/release/contract_gate.py validate-result results/python.json
    python scripts/release/contract_gate.py refresh

``verify`` fails when a file under a parity root is not listed, a listed file
is missing or its digest is stale, a fixture's version or case ids are
inconsistent, or a runtime cell is malformed. ``run-python`` runs the declared
Python consumer tests once through pytest and maps each test outcome onto the
fixture's case ids. ``refresh`` rewrites the digests of files that are already
listed; it never adds a file or decides which runtimes a fixture needs.

Output carries reason codes, fixture ids, case ids, counts and repository
relative paths only. The script makes no network call.

Exit codes: 0 on success, 1 when verification fails or a ``--check`` run has a
case that did not pass, 2 when the manifest or an input cannot be read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "tests/fixtures/parity/contract_gate.json"
MANIFEST_SCHEMA = "openmed.contract_gate.manifest"
MANIFEST_VERSION = 1
RESULT_SCHEMA = "openmed.contract_gate.result"
RESULT_SCHEMA_VERSION = 1
RESULTS_DIR_ENV = "OPENMED_CONTRACT_GATE_RESULTS_DIR"
PYTHON_RUNTIME = "python"

STATUS_REQUIRED = "required"
STATUS_NOT_APPLICABLE = "not_applicable"
CELL_STATUSES = (STATUS_REQUIRED, STATUS_NOT_APPLICABLE)
REPORTER_ACTIVE = "active"
REPORTER_PENDING = "pending"
REPORTER_STATES = (REPORTER_ACTIVE, REPORTER_PENDING)

# Reason codes the Python runner writes. Other reporters may add their own
# codes; every code must match ``REASON_PATTERN``.
REASON_CONSUMER_FAILED = "consumer_failed"
REASON_CONSUMER_ERROR = "consumer_error"
REASON_CONSUMER_SKIPPED = "consumer_skipped"
REASON_CONSUMER_MISSING = "consumer_missing"
REASON_CONSUMER_NOT_COLLECTED = "consumer_not_collected"
REASON_CASE_NOT_EXERCISED = "case_not_exercised"
REASON_FIXTURE_UNREADABLE = "fixture_unreadable"

ID_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
RUNTIME_PATTERN = r"^[a-z][a-z0-9-]{0,31}$"
REASON_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
CASE_PART_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ID_RE = re.compile(ID_PATTERN)
_RUNTIME_RE = re.compile(RUNTIME_PATTERN)
_CASE_PART_RE = re.compile(CASE_PART_PATTERN)
_SHA256_RE = re.compile(SHA256_PATTERN)
_REASON_TEXT_LIMIT = 240

_MANIFEST_KEYS = frozenset(
    {"schema", "version", "algorithm", "roots", "result_schema", "runtimes"}
    | {"fixtures", "description"}
)
_RUNTIME_KEYS = frozenset({"reporter", "command", "tracking"})
_FIXTURE_KEYS = frozenset(
    {"id", "path", "sha256", "version", "case_groups", "companions", "runtimes"}
)
_CELL_KEYS = frozenset({"status", "consumers", "reason"})
_FILE_KEYS = frozenset({"path", "sha256"})
_GROUP_KEYS = frozenset({"key", "id_field"})


class ContractGateError(ValueError):
    """A value-free contract-gate failure.

    Attributes:
        code: Stable snake_case reason code.
        field_path: JSON path inside the manifest, or ``"$"``.
    """

    def __init__(self, code: str, field_path: str = "$") -> None:
        super().__init__(f"{code} at {field_path}")
        self.code = code
        self.field_path = field_path


@dataclass(frozen=True)
class Finding:
    """One value-free verification finding.

    Attributes:
        code: Stable snake_case reason code.
        fixture: Fixture id the finding belongs to, when known.
        path: Repository-relative file path or manifest JSON path.
    """

    code: str
    fixture: str | None = None
    path: str | None = None

    def render(self) -> str:
        """Return the finding as one ``code key=value`` log line."""
        parts = [self.code]
        if self.fixture is not None:
            parts.append(f"fixture={self.fixture}")
        if self.path is not None:
            parts.append(f"path={self.path}")
        return " ".join(parts)


@dataclass(frozen=True)
class CaseGroup:
    """Where a fixture keeps one family of cases.

    Attributes:
        key: Top-level fixture key holding a list or an object.
        id_field: Field naming each list item's id, or ``None`` to use the
            list index. Ignored when ``key`` holds an object.
    """

    key: str
    id_field: str | None = None


@dataclass(frozen=True)
class RuntimeCell:
    """One fixture-by-runtime cell of the contract matrix.

    Attributes:
        status: ``required`` or ``not_applicable``.
        consumers: Repository-relative test files that consume the fixture.
        reason: Why the fixture does not apply, for ``not_applicable`` cells.
    """

    status: str
    consumers: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class RuntimeDeclaration:
    """A runtime the gate knows about and the state of its reporter.

    Attributes:
        reporter: ``active`` once the runtime writes result files, otherwise
            ``pending``.
        command: Command that writes the runtime's result file, when active.
        tracking: Issue reference for a pending reporter.
    """

    reporter: str
    command: str | None = None
    tracking: str | None = None


@dataclass(frozen=True)
class ListedFile:
    """A file the manifest pins by digest.

    Attributes:
        path: Repository-relative POSIX path.
        sha256: Expected lowercase hex SHA-256 digest of the file bytes.
    """

    path: str
    sha256: str


@dataclass(frozen=True)
class FixtureEntry:
    """One shared parity fixture.

    Attributes:
        id: Stable snake_case fixture id.
        path: Repository-relative path of the fixture document.
        sha256: Expected digest of the fixture document.
        version: Fixture contract version.
        case_groups: Where the fixture's cases live.
        companions: Supporting files pinned with the fixture.
        cells: Runtime cells keyed by runtime id.
    """

    id: str
    path: str
    sha256: str
    version: int
    case_groups: tuple[CaseGroup, ...]
    companions: tuple[ListedFile, ...]
    cells: Mapping[str, RuntimeCell]


@dataclass(frozen=True)
class Manifest:
    """A parsed contract-gate manifest.

    Attributes:
        path: Repository-relative manifest path.
        sha256: Digest of the manifest bytes that were parsed.
        roots: Repository-relative parity roots the manifest covers.
        result_schema: The pinned result JSON Schema.
        runtimes: Runtime declarations keyed by runtime id.
        fixtures: Fixtures in manifest order.
    """

    path: str
    sha256: str
    roots: tuple[str, ...]
    result_schema: ListedFile
    runtimes: Mapping[str, RuntimeDeclaration]
    fixtures: tuple[FixtureEntry, ...]

    def fixture(self, fixture_id: str) -> FixtureEntry:
        """Return one fixture by id.

        Args:
            fixture_id: The fixture id.

        Returns:
            The matching fixture entry.

        Raises:
            ContractGateError: ``unknown_fixture`` when no fixture has the id.
        """
        for entry in self.fixtures:
            if entry.id == fixture_id:
                return entry
        raise ContractGateError("unknown_fixture")


@dataclass(frozen=True)
class ConsumerOutcome:
    """One pytest test outcome read back from a JUnit report.

    Attributes:
        classname: Dotted module path, plus a class name when the test is a
            method.
        name: Test function name without its parameter id.
        param: The parameter id inside ``[...]``, if any.
        outcome: ``passed``, ``failed``, ``error`` or ``skipped``.
    """

    classname: str
    name: str
    param: str | None
    outcome: str


PytestRunner = Callable[[Path, Sequence[str], Path], int]


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(document: Any) -> str:
    """Serialise a manifest or result document the way the gate commits it.

    Args:
        document: JSON-compatible value.

    Returns:
        Two-space indented ASCII JSON with a trailing newline.
    """
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"


def load_manifest(root: Path = ROOT, manifest: str = DEFAULT_MANIFEST) -> Manifest:
    """Read and structurally validate a contract-gate manifest.

    Args:
        root: Repository root that every manifest path is relative to.
        manifest: Repository-relative manifest path.

    Returns:
        The parsed manifest. Digests and files are not checked here; use
        :func:`verify` for that.

    Raises:
        ContractGateError: ``manifest_unreadable`` when the file cannot be read
            or decoded, otherwise a structural code with the JSON path.
    """
    manifest_path = _relative_path(manifest, "$manifest")
    try:
        raw = (root / manifest_path).read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractGateError("manifest_unreadable") from error
    return parse_manifest(
        document, path=manifest_path, sha256=hashlib.sha256(raw).hexdigest()
    )


def parse_manifest(
    document: Any, *, path: str = DEFAULT_MANIFEST, sha256: str = "0" * 64
) -> Manifest:
    """Validate the structure of a decoded manifest.

    Args:
        document: Decoded manifest JSON.
        path: Repository-relative manifest path, for self-exclusion.
        sha256: Digest of the manifest bytes.

    Returns:
        The parsed manifest.

    Raises:
        ContractGateError: A structural code such as ``manifest_invalid``,
            ``unknown_schema``, ``not_applicable_without_reason`` or
            ``runtime_cell_missing``, with the JSON path that failed.
    """
    if not isinstance(document, Mapping):
        raise ContractGateError("manifest_invalid")
    _closed(document, _MANIFEST_KEYS, "$")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ContractGateError("unknown_schema", "$.schema")
    if document.get("version") != MANIFEST_VERSION:
        raise ContractGateError("unsupported_version", "$.version")
    if document.get("algorithm") != "sha256":
        raise ContractGateError("unsupported_algorithm", "$.algorithm")
    roots = _string_list(document.get("roots"), "$.roots", minimum=1)
    roots = tuple(_relative_path(item, f"$.roots[{i}]") for i, item in enumerate(roots))
    result_schema = _listed_file(document.get("result_schema"), "$.result_schema")
    runtimes = _runtimes(document.get("runtimes"))
    raw_fixtures = document.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise ContractGateError("manifest_invalid", "$.fixtures")
    fixtures = tuple(
        _fixture(item, f"$.fixtures[{index}]", runtimes)
        for index, item in enumerate(raw_fixtures)
    )
    seen: set[str] = set()
    for index, entry in enumerate(fixtures):
        if entry.id in seen:
            raise ContractGateError("duplicate_fixture_id", f"$.fixtures[{index}].id")
        seen.add(entry.id)
    return Manifest(
        path=path,
        sha256=sha256,
        roots=roots,
        result_schema=result_schema,
        runtimes=runtimes,
        fixtures=fixtures,
    )


def expected_case_ids(entry: FixtureEntry, document: Any) -> tuple[str, ...]:
    """Derive a fixture's case ids from its content.

    A list group with an ``id_field`` yields ``<key>/<item id>``; a list group
    without one yields ``<key>/<index>``; an object group yields ``<key>``.
    Every runtime reporter must derive the same ids.

    Args:
        entry: The fixture's manifest entry.
        document: The decoded fixture document.

    Returns:
        Case ids in fixture order.

    Raises:
        ContractGateError: ``case_group_missing``, ``case_id_invalid`` or
            ``case_id_duplicate`` with the fixture JSON path.
    """
    if not isinstance(document, Mapping):
        raise ContractGateError("case_group_missing")
    ids: list[str] = []
    for group in entry.case_groups:
        value = document.get(group.key)
        path = f"$.{group.key}"
        if isinstance(value, Mapping):
            ids.append(group.key)
            continue
        if not isinstance(value, list):
            raise ContractGateError("case_group_missing", path)
        for index, item in enumerate(value):
            if group.id_field is None:
                part = str(index)
            else:
                raw = item.get(group.id_field) if isinstance(item, Mapping) else None
                if not isinstance(raw, str) or not _CASE_PART_RE.match(raw):
                    raise ContractGateError(
                        "case_id_invalid", f"{path}[{index}].{group.id_field}"
                    )
                part = raw
            ids.append(f"{group.key}/{part}")
    seen: set[str] = set()
    for case_id in ids:
        if case_id in seen:
            raise ContractGateError("case_id_duplicate", f"$.{case_id.split('/')[0]}")
        seen.add(case_id)
    return tuple(ids)


def verify(root: Path = ROOT, manifest: str = DEFAULT_MANIFEST) -> list[Finding]:
    """Check the manifest against the files on disk.

    Args:
        root: Repository root.
        manifest: Repository-relative manifest path.

    Returns:
        Findings in a stable order; an empty list means the gate is current.

    Raises:
        ContractGateError: ``manifest_unreadable`` when the manifest cannot be
            read or decoded.
    """
    try:
        parsed = load_manifest(root, manifest)
    except ContractGateError as error:
        if error.code == "manifest_unreadable":
            raise
        return [Finding(error.code, path=error.field_path)]
    findings: list[Finding] = []
    for root_path in parsed.roots:
        if not (root / root_path).is_dir():
            findings.append(Finding("root_missing", path=root_path))
    listed: dict[str, str | None] = {parsed.result_schema.path: None}
    files: list[tuple[ListedFile, str | None]] = [(parsed.result_schema, None)]
    for entry in parsed.fixtures:
        for item in (ListedFile(entry.path, entry.sha256), *entry.companions):
            if item.path in listed or item.path == parsed.path:
                findings.append(Finding("duplicate_path", entry.id, item.path))
                continue
            listed[item.path] = entry.id
            files.append((item, entry.id))
    for item, fixture_id in files:
        if not _under_roots(item.path, parsed.roots):
            findings.append(Finding("path_outside_roots", fixture_id, item.path))
        target = root / item.path
        if not target.is_file():
            findings.append(Finding("file_missing", fixture_id, item.path))
        elif sha256_file(target) != item.sha256:
            findings.append(Finding("digest_mismatch", fixture_id, item.path))
    for path in sorted(_files_under_roots(root, parsed.roots)):
        if path != parsed.path and path not in listed:
            findings.append(Finding("unlisted_file", path=path))
    for entry in parsed.fixtures:
        findings.extend(_verify_fixture(root, parsed, entry))
    return findings


def run_python(
    root: Path = ROOT,
    manifest: str = DEFAULT_MANIFEST,
    *,
    fixture_ids: Sequence[str] | None = None,
    runner: PytestRunner | None = None,
) -> dict[str, Any]:
    """Run the Python consumers of every required fixture and build a result.

    The declared consumer test files run in one pytest process. A test whose
    parameter id equals a case id (the bare id or the full ``<key>/<id>``)
    covers only that case; every other test in a consumer file covers all of
    the fixture's cases. A case passes when at least one covering test passed
    and none failed or errored; it is skipped when every covering test was
    skipped.

    Args:
        root: Repository root.
        manifest: Repository-relative manifest path.
        fixture_ids: Restrict the run to these fixtures. Defaults to every
            fixture whose Python cell is ``required``.
        runner: Callable ``(root, consumers, junit_path) -> exit code`` that
            runs pytest and writes a JUnit XML report. Defaults to a pytest
            subprocess.

    Returns:
        A result document matching the contract-gate result schema.

    Raises:
        ContractGateError: ``manifest_unreadable``, a structural manifest code,
            ``unknown_fixture`` or ``runtime_not_required``.
    """
    parsed = load_manifest(root, manifest)
    if fixture_ids:
        entries = [parsed.fixture(fixture_id) for fixture_id in fixture_ids]
        for entry in entries:
            if entry.cells[PYTHON_RUNTIME].status != STATUS_REQUIRED:
                raise ContractGateError("runtime_not_required")
    else:
        entries = [
            entry
            for entry in parsed.fixtures
            if entry.cells[PYTHON_RUNTIME].status == STATUS_REQUIRED
        ]
    consumers = sorted(
        {
            consumer
            for entry in entries
            for consumer in entry.cells[PYTHON_RUNTIME].consumers
            if (root / consumer).is_file()
        }
    )
    tests: list[ConsumerOutcome] | None = []
    if consumers:
        tests = _collect_outcomes(root, consumers, runner or run_pytest)
    results = [_fixture_result(root, entry, tests) for entry in entries]
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "runtime": PYTHON_RUNTIME,
        "manifest_sha256": parsed.sha256,
        "fixtures": results,
    }


def run_pytest(root: Path, consumers: Sequence[str], junit_path: Path) -> int:
    """Run pytest over consumer files and write a JUnit XML report.

    Console output is discarded so the gate never echoes fixture values;
    rerun the consumer files with pytest directly to see assertion details.

    Args:
        root: Repository root used as the pytest root directory.
        consumers: Repository-relative test files.
        junit_path: Where pytest writes the JUnit XML report.

    Returns:
        The pytest exit code.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((str(root), existing)) if existing else str(root)
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *consumers,
        "-q",
        "--tb=no",
        "-p",
        "no:cacheprovider",
        f"--rootdir={root}",
        f"--junitxml={junit_path}",
        "-o",
        "junit_family=xunit2",
        "-o",
        "junit_logging=no",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode


def parse_junit(path: Path) -> list[ConsumerOutcome]:
    """Read test outcomes from a pytest JUnit XML report.

    Args:
        path: The report written by ``pytest --junitxml``.

    Returns:
        One outcome per ``testcase`` element. A module that failed to collect
        is reported with its module path as ``classname`` and outcome
        ``error``.

    Raises:
        ContractGateError: ``consumer_not_collected`` when the report is
            missing or not valid XML.
    """
    try:
        # The report is written locally by the pytest process started above.
        tree = ElementTree.parse(path)
    except (OSError, ElementTree.ParseError) as error:
        raise ContractGateError(REASON_CONSUMER_NOT_COLLECTED) from error
    outcomes: list[ConsumerOutcome] = []
    for case in tree.getroot().iter("testcase"):
        classname = case.get("classname") or ""
        full_name = case.get("name") or ""
        if not classname:
            classname, full_name = full_name, ""
        name, param = _split_param(full_name)
        tags = {child.tag for child in case}
        if "failure" in tags:
            outcome = "failed"
        elif "error" in tags:
            outcome = "error"
        elif "skipped" in tags:
            outcome = "skipped"
        else:
            outcome = "passed"
        outcomes.append(ConsumerOutcome(classname, name, param, outcome))
    return outcomes


def validate_result(document: Any, schema: Mapping[str, Any]) -> list[Finding]:
    """Validate a runtime result document.

    Args:
        document: Decoded result document.
        schema: The contract-gate result JSON Schema.

    Returns:
        Value-free findings: ``result_schema_violation`` with the JSON path and
        failing keyword, or ``case_reported_twice`` and ``fixture_reported_twice``
        consistency findings. An empty list means the document is valid.
    """
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    findings = [
        Finding(
            "result_schema_violation",
            path=f"{_json_path(error.absolute_path)} keyword={error.validator}",
        )
        for error in sorted(
            validator.iter_errors(document), key=lambda item: list(item.absolute_path)
        )
    ]
    if findings:
        return findings
    seen_fixtures: set[str] = set()
    for index, result in enumerate(document["fixtures"]):
        fixture_id = result["fixture_id"]
        if fixture_id in seen_fixtures:
            findings.append(
                Finding("fixture_reported_twice", fixture_id, f"$.fixtures[{index}]")
            )
        seen_fixtures.add(fixture_id)
        reported = [
            *result["passed"],
            *(item["case_id"] for item in result["failed"]),
            *(item["case_id"] for item in result["skipped"]),
        ]
        if len(reported) != len(set(reported)):
            findings.append(
                Finding("case_reported_twice", fixture_id, f"$.fixtures[{index}]")
            )
    return findings


def load_result_schema(
    root: Path = ROOT, manifest: str = DEFAULT_MANIFEST
) -> dict[str, Any]:
    """Load the result JSON Schema pinned by the manifest.

    Args:
        root: Repository root.
        manifest: Repository-relative manifest path.

    Returns:
        The decoded JSON Schema.

    Raises:
        ContractGateError: ``manifest_unreadable`` or ``result_schema_unreadable``.
    """
    parsed = load_manifest(root, manifest)
    try:
        return json.loads((root / parsed.result_schema.path).read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractGateError("result_schema_unreadable") from error


def refresh(root: Path = ROOT, manifest: str = DEFAULT_MANIFEST) -> list[str]:
    """Rewrite the digests of files the manifest already lists.

    Args:
        root: Repository root.
        manifest: Repository-relative manifest path.

    Returns:
        Repository-relative paths whose digest changed.

    Raises:
        ContractGateError: ``manifest_unreadable``, a structural manifest code,
            or ``file_missing`` with the path of a listed file that is absent.
    """
    parsed = load_manifest(root, manifest)
    document = json.loads((root / parsed.path).read_text("utf-8"))
    changed: list[str] = []

    def update(item: dict[str, Any]) -> None:
        target = root / item["path"]
        if not target.is_file():
            raise ContractGateError("file_missing", item["path"])
        digest = sha256_file(target)
        if item["sha256"] != digest:
            item["sha256"] = digest
            changed.append(item["path"])

    update(document["result_schema"])
    for fixture in document["fixtures"]:
        update(fixture)
        for companion in fixture.get("companions", []):
            update(companion)
    if changed:
        (root / parsed.path).write_text(dump_json(document), encoding="utf-8")
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the contract-gate command line.

    Args:
        argv: Arguments without the program name. Defaults to ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else ROOT
    try:
        return int(args.handler(args, root))
    except ContractGateError as error:
        print(f"contract_gate: {error.code} path={error.field_path}", file=sys.stderr)
        return 2


def _cmd_verify(args: argparse.Namespace, root: Path) -> int:
    findings = verify(root, args.manifest)
    for finding in findings:
        print(f"contract_gate verify: {finding.render()}")
    if findings:
        print(f"contract_gate verify: failed findings={len(findings)}")
        return 1
    parsed = load_manifest(root, args.manifest)
    files = 1 + sum(1 + len(entry.companions) for entry in parsed.fixtures)
    print(f"contract_gate verify: ok fixtures={len(parsed.fixtures)} files={files}")
    return 0


def _cmd_run_python(args: argparse.Namespace, root: Path) -> int:
    output = _output_path(args)
    if output is None:
        print(
            f"contract_gate run-python: pass --output or --results-dir, or set "
            f"{RESULTS_DIR_ENV}",
            file=sys.stderr,
        )
        return 2
    result = run_python(root, args.manifest, fixture_ids=args.fixture or None)
    findings = validate_result(result, load_result_schema(root, args.manifest))
    if findings:  # pragma: no cover - the runner only writes schema-valid ids
        for finding in findings:
            print(f"contract_gate run-python: {finding.render()}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump_json(result), encoding="utf-8")
    incomplete = 0
    for item in result["fixtures"]:
        print(
            "contract_gate run-python: "
            f"fixture={item['fixture_id']} passed={len(item['passed'])} "
            f"failed={len(item['failed'])} skipped={len(item['skipped'])}"
        )
        if item["failed"] or item["skipped"] or item.get("errors"):
            incomplete += 1
    print(f"contract_gate run-python: wrote {output.name}")
    return 1 if args.check and incomplete else 0


def _cmd_validate_result(args: argparse.Namespace, root: Path) -> int:
    schema = load_result_schema(root, args.manifest)
    status = 0
    for name in args.results:
        path = Path(name)
        try:
            document = json.loads(path.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            print(f"contract_gate validate-result: result_unreadable file={path.name}")
            status = max(status, 2)
            continue
        findings = validate_result(document, schema)
        for finding in findings:
            print(f"contract_gate validate-result: {finding.render()} file={path.name}")
        if findings:
            status = max(status, 1)
        else:
            print(f"contract_gate validate-result: ok file={path.name}")
    return status


def _cmd_refresh(args: argparse.Namespace, root: Path) -> int:
    changed = refresh(root, args.manifest)
    for path in changed:
        print(f"contract_gate refresh: updated path={path}")
    print(f"contract_gate refresh: changed={len(changed)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contract_gate.py",
        description="Verify shared parity fixtures and write contract results.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Repository-relative manifest path.",
    )
    parser.add_argument("--root", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "verify", help="Check digests, listings and runtime cells."
    ).set_defaults(handler=_cmd_verify)
    run = commands.add_parser(
        "run-python", help="Run the Python consumers and write python.json."
    )
    target = run.add_mutually_exclusive_group()
    target.add_argument("--output", help="Result file to write.")
    target.add_argument(
        "--results-dir",
        help=f"Directory for python.json (default: ${RESULTS_DIR_ENV}).",
    )
    run.add_argument(
        "--fixture",
        action="append",
        help="Only run this fixture id; may be repeated.",
    )
    run.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when any case did not pass.",
    )
    run.set_defaults(handler=_cmd_run_python)
    validate = commands.add_parser(
        "validate-result", help="Validate result files against the schema."
    )
    validate.add_argument("results", nargs="+", help="Result files.")
    validate.set_defaults(handler=_cmd_validate_result)
    commands.add_parser(
        "refresh", help="Rewrite digests of files the manifest already lists."
    ).set_defaults(handler=_cmd_refresh)
    return parser


def _output_path(args: argparse.Namespace) -> Path | None:
    if args.output:
        return Path(args.output)
    directory = args.results_dir or os.environ.get(RESULTS_DIR_ENV)
    if not directory:
        return None
    return Path(directory) / f"{PYTHON_RUNTIME}.json"


def _collect_outcomes(
    root: Path, consumers: Sequence[str], runner: PytestRunner
) -> list[ConsumerOutcome] | None:
    with tempfile.TemporaryDirectory(prefix="contract-gate-") as directory:
        junit_path = Path(directory) / "junit.xml"
        runner(root, consumers, junit_path)
        try:
            return parse_junit(junit_path)
        except ContractGateError:
            return None


def _fixture_result(
    root: Path, entry: FixtureEntry, tests: list[ConsumerOutcome] | None
) -> dict[str, Any]:
    target = root / entry.path
    result: dict[str, Any] = {
        "fixture_id": entry.id,
        "fixture_sha256": "0" * 64,
        "passed": [],
        "failed": [],
        "skipped": [],
    }
    try:
        raw = target.read_bytes()
        result["fixture_sha256"] = hashlib.sha256(raw).hexdigest()
        case_ids = expected_case_ids(entry, json.loads(raw.decode("utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ContractGateError):
        result["errors"] = [REASON_FIXTURE_UNREADABLE]
        return result
    consumers = entry.cells[PYTHON_RUNTIME].consumers
    present = [consumer for consumer in consumers if (root / consumer).is_file()]
    if len(present) != len(consumers) or not consumers:
        result["failed"] = [
            {"case_id": case_id, "reason": REASON_CONSUMER_MISSING}
            for case_id in case_ids
        ]
        return result
    modules = tuple(_module_name(consumer) for consumer in consumers)
    matched = (
        [test for test in tests if _belongs_to(test.classname, modules)]
        if tests is not None
        else []
    )
    if not matched:
        result["failed"] = [
            {"case_id": case_id, "reason": REASON_CONSUMER_NOT_COLLECTED}
            for case_id in case_ids
        ]
        return result
    passed, failed, skipped = _case_outcomes(entry, case_ids, matched)
    result["passed"] = passed
    result["failed"] = failed
    result["skipped"] = skipped
    result["tests"] = {
        "passed": sum(test.outcome == "passed" for test in matched),
        "failed": sum(test.outcome in {"failed", "error"} for test in matched),
        "skipped": sum(test.outcome == "skipped" for test in matched),
    }
    return result


def _case_outcomes(
    entry: FixtureEntry, case_ids: Sequence[str], tests: Sequence[ConsumerOutcome]
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    keyed_groups = {group.key for group in entry.case_groups if group.id_field}
    by_key: dict[str, list[str]] = {}
    for case_id in case_ids:
        keys = {case_id}
        group, _, bare = case_id.partition("/")
        if bare and group in keyed_groups:
            keys.add(bare)
        for key in keys:
            by_key.setdefault(key, []).append(case_id)
    covering: dict[str, list[ConsumerOutcome]] = {case_id: [] for case_id in case_ids}
    for test in tests:
        targets = by_key.get(test.param) if test.param is not None else None
        for case_id in targets or case_ids:
            covering[case_id].append(test)
    passed: list[str] = []
    failed: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for case_id in case_ids:
        outcomes = {test.outcome for test in covering[case_id]}
        if "failed" in outcomes:
            failed.append({"case_id": case_id, "reason": REASON_CONSUMER_FAILED})
        elif "error" in outcomes:
            failed.append({"case_id": case_id, "reason": REASON_CONSUMER_ERROR})
        elif "passed" in outcomes:
            passed.append(case_id)
        elif "skipped" in outcomes:
            skipped.append({"case_id": case_id, "reason": REASON_CONSUMER_SKIPPED})
        else:
            failed.append({"case_id": case_id, "reason": REASON_CASE_NOT_EXERCISED})
    return passed, failed, skipped


def _verify_fixture(
    root: Path, manifest: Manifest, entry: FixtureEntry
) -> list[Finding]:
    findings: list[Finding] = []
    for runtime, cell in entry.cells.items():
        for consumer in cell.consumers:
            if not (root / consumer).is_file():
                findings.append(Finding("consumer_missing", entry.id, consumer))
        reporter = manifest.runtimes[runtime].reporter
        if (
            cell.status == STATUS_REQUIRED
            and reporter == REPORTER_ACTIVE
            and not cell.consumers
        ):
            findings.append(
                Finding("consumer_required", entry.id, f"$.runtimes.{runtime}")
            )
    target = root / entry.path
    if not target.is_file():
        return findings
    try:
        document = json.loads(target.read_text("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        findings.append(Finding("fixture_unreadable", entry.id, entry.path))
        return findings
    if isinstance(document, Mapping):
        version = document.get("version")
        if isinstance(version, int) and not isinstance(version, bool):
            if version != entry.version:
                findings.append(Finding("version_mismatch", entry.id, entry.path))
    try:
        expected_case_ids(entry, document)
    except ContractGateError as error:
        findings.append(Finding(error.code, entry.id, error.field_path))
    return findings


def _runtimes(value: Any) -> dict[str, RuntimeDeclaration]:
    if not isinstance(value, Mapping) or not value:
        raise ContractGateError("manifest_invalid", "$.runtimes")
    runtimes: dict[str, RuntimeDeclaration] = {}
    for runtime, raw in value.items():
        path = f"$.runtimes.{runtime}"
        if not isinstance(runtime, str) or not _RUNTIME_RE.match(runtime):
            raise ContractGateError("unknown_runtime", "$.runtimes")
        if not isinstance(raw, Mapping):
            raise ContractGateError("manifest_invalid", path)
        _closed(raw, _RUNTIME_KEYS, path)
        reporter = raw.get("reporter")
        if reporter not in REPORTER_STATES:
            raise ContractGateError("invalid_reporter_state", f"{path}.reporter")
        command = _optional_text(raw.get("command"), f"{path}.command")
        tracking = _optional_text(raw.get("tracking"), f"{path}.tracking")
        if reporter == REPORTER_ACTIVE and command is None:
            raise ContractGateError("reporter_command_missing", f"{path}.command")
        if reporter == REPORTER_PENDING and tracking is None:
            raise ContractGateError("reporter_tracking_missing", f"{path}.tracking")
        runtimes[runtime] = RuntimeDeclaration(reporter, command, tracking)
    if PYTHON_RUNTIME not in runtimes:
        raise ContractGateError("runtime_cell_missing", f"$.runtimes.{PYTHON_RUNTIME}")
    return runtimes


def _fixture(
    value: Any, path: str, runtimes: Mapping[str, RuntimeDeclaration]
) -> FixtureEntry:
    if not isinstance(value, Mapping):
        raise ContractGateError("manifest_invalid", path)
    _closed(value, _FIXTURE_KEYS, path)
    fixture_id = value.get("id")
    if not isinstance(fixture_id, str) or not _ID_RE.match(fixture_id):
        raise ContractGateError("fixture_id_invalid", f"{path}.id")
    listed = _listed_file(value, path, closed=False)
    version = value.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ContractGateError("manifest_invalid", f"{path}.version")
    raw_groups = value.get("case_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ContractGateError("case_groups_missing", f"{path}.case_groups")
    groups = tuple(
        _case_group(item, f"{path}.case_groups[{index}]")
        for index, item in enumerate(raw_groups)
    )
    if len({group.key for group in groups}) != len(groups):
        raise ContractGateError("manifest_invalid", f"{path}.case_groups")
    raw_companions = value.get("companions", [])
    if not isinstance(raw_companions, list):
        raise ContractGateError("manifest_invalid", f"{path}.companions")
    companions = tuple(
        _listed_file(item, f"{path}.companions[{index}]")
        for index, item in enumerate(raw_companions)
    )
    raw_cells = value.get("runtimes")
    if not isinstance(raw_cells, Mapping):
        raise ContractGateError("manifest_invalid", f"{path}.runtimes")
    for runtime in raw_cells:
        if runtime not in runtimes:
            raise ContractGateError("unknown_runtime", f"{path}.runtimes.{runtime}")
    cells = {}
    for runtime in runtimes:
        if runtime not in raw_cells:
            raise ContractGateError(
                "runtime_cell_missing", f"{path}.runtimes.{runtime}"
            )
        cells[runtime] = _cell(raw_cells[runtime], f"{path}.runtimes.{runtime}")
    return FixtureEntry(
        id=fixture_id,
        path=listed.path,
        sha256=listed.sha256,
        version=version,
        case_groups=groups,
        companions=companions,
        cells=cells,
    )


def _cell(value: Any, path: str) -> RuntimeCell:
    if not isinstance(value, Mapping):
        raise ContractGateError("manifest_invalid", path)
    _closed(value, _CELL_KEYS, path)
    status = value.get("status")
    if status not in CELL_STATUSES:
        raise ContractGateError("invalid_status", f"{path}.status")
    consumers = tuple(
        _relative_path(item, f"{path}.consumers[{index}]")
        for index, item in enumerate(
            _string_list(value.get("consumers", []), f"{path}.consumers")
        )
    )
    if len(set(consumers)) != len(consumers):
        raise ContractGateError("manifest_invalid", f"{path}.consumers")
    reason = value.get("reason")
    if status == STATUS_NOT_APPLICABLE:
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _REASON_TEXT_LIMIT
        ):
            raise ContractGateError("not_applicable_without_reason", f"{path}.reason")
        if consumers:
            raise ContractGateError("not_applicable_with_consumers", path)
    elif reason is not None:
        raise ContractGateError("manifest_invalid", f"{path}.reason")
    return RuntimeCell(status=status, consumers=consumers, reason=reason)


def _case_group(value: Any, path: str) -> CaseGroup:
    if not isinstance(value, Mapping):
        raise ContractGateError("manifest_invalid", path)
    _closed(value, _GROUP_KEYS, path)
    key = value.get("key")
    if not isinstance(key, str) or not _ID_RE.match(key):
        raise ContractGateError("manifest_invalid", f"{path}.key")
    id_field = value.get("id_field")
    if id_field is not None and (not isinstance(id_field, str) or not id_field):
        raise ContractGateError("manifest_invalid", f"{path}.id_field")
    return CaseGroup(key=key, id_field=id_field)


def _listed_file(value: Any, path: str, *, closed: bool = True) -> ListedFile:
    if not isinstance(value, Mapping):
        raise ContractGateError("manifest_invalid", path)
    if closed:
        _closed(value, _FILE_KEYS, path)
    file_path = _relative_path(value.get("path"), f"{path}.path")
    digest = value.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.match(digest):
        raise ContractGateError("manifest_invalid", f"{path}.sha256")
    return ListedFile(path=file_path, sha256=digest)


def _relative_path(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractGateError("path_invalid", path)
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise ContractGateError("path_invalid", path)
    return value


def _string_list(value: Any, path: str, *, minimum: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ContractGateError("manifest_invalid", path)
    if not all(isinstance(item, str) for item in value):
        raise ContractGateError("manifest_invalid", path)
    return list(value)


def _optional_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ContractGateError("manifest_invalid", path)
    return value


def _closed(value: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ContractGateError("unknown_field", f"{path}.{unknown[0]}")


def _under_roots(path: str, roots: Sequence[str]) -> bool:
    parts = PurePosixPath(path).parts
    for root_path in roots:
        prefix = PurePosixPath(root_path).parts
        if len(parts) > len(prefix) and parts[: len(prefix)] == prefix:
            return True
    return False


def _files_under_roots(root: Path, roots: Sequence[str]) -> set[str]:
    files: set[str] = set()
    for root_path in roots:
        base = root / root_path
        if not base.is_dir():
            continue
        for candidate in base.rglob("*"):
            relative = candidate.relative_to(root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if "__pycache__" in relative.parts or not candidate.is_file():
                continue
            files.add(relative.as_posix())
    return files


def _module_name(consumer: str) -> str:
    pure = PurePosixPath(consumer)
    return ".".join(pure.with_suffix("").parts)


def _belongs_to(classname: str, modules: Sequence[str]) -> bool:
    return any(
        classname == module or classname.startswith(module + ".") for module in modules
    )


def _split_param(name: str) -> tuple[str, str | None]:
    if name.endswith("]") and "[" in name:
        index = name.index("[")
        return name[:index], name[index + 1 : -1]
    return name, None


def _json_path(parts: Iterable[Any]) -> str:
    rendered = "$"
    for part in parts:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


if __name__ == "__main__":
    raise SystemExit(main())
