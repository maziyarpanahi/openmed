"""The error-code registry is complete, consistent, and agrees with the code.

The registry in ``openmed/core/error_codes.json`` describes current behaviour.
These tests derive each code set from the real tables and source in the
package, never from a hand list, so a code added or renamed anywhere fails
here until the registry records it. Surface behaviour (REST, gRPC, GraphQL,
MCP) is exercised in ``tests/unit/service/test_error_registry_surfaces.py``.
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import inspect
import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterator

import pytest
from jsonschema import Draft202012Validator

from openmed.cli import _output as cli_output
from openmed.core import errors as taxonomy
from openmed.core.error_registry import (
    CATEGORIES,
    ERROR_REGISTRY_CONTRACT,
    ErrorCodeRegistry,
    ErrorRegistryError,
    load_error_registry,
    load_error_registry_schema,
    lookup_error_code,
    parse_error_registry,
)
from openmed.core.schemas import (
    SCHEMA_NAMES,
    build_schema_snapshot,
    load_schema,
    load_schema_snapshot,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "openmed" / "core" / "error_codes.json"
TAXONOMY_FIXTURE = ROOT / "tests" / "fixtures" / "error_taxonomy.json"
PROTOCOL_MAP = ROOT / "docs" / "api" / "protocol-map.json"
ERRORS_DOC = ROOT / "docs" / "api" / "errors.md"
SENTINEL = "SYNTHETIC_RAW_PHI_SENTINEL_R2"
#: Classes that take their public code as a constructor argument.
_INSTANCE_CODE_PARAMETERS = frozenset({"code", "reason"})


@pytest.fixture(scope="module")
def registry() -> ErrorCodeRegistry:
    return load_error_registry()


@pytest.fixture
def document() -> dict[str, Any]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _import_path(path: str) -> Any:
    module_name, _, attribute = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


def _taxonomy_closure() -> set[str]:
    """Return every class name in ``openmed/`` rooted at a taxonomy class."""
    names = set(taxonomy.ERROR_CODES)
    bases: dict[str, set[str]] = {}
    for path in sorted((ROOT / "openmed").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases.setdefault(node.name, set()).update(
                    base.id if isinstance(base, ast.Name) else base.attr
                    for base in node.bases
                    if isinstance(base, (ast.Name, ast.Attribute))
                )
    changed = True
    while changed:
        changed = False
        for name, parents in bases.items():
            if name not in names and parents & names:
                names.add(name)
                changed = True
    return names


def _taxonomy_call_codes() -> Iterator[tuple[str, str]]:
    """Yield ``(class name, code)`` for taxonomy calls with a literal code."""
    names = _taxonomy_closure()
    for path in sorted((ROOT / "openmed").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else None
            )
            if name not in names:
                continue
            for keyword in node.keywords:
                if keyword.arg == "code" and isinstance(keyword.value, ast.Constant):
                    yield name, str(keyword.value.value)


def _cli_literal_codes() -> dict[str, set[int]]:
    """Return every literal ``CliError`` code with its literal exit statuses."""
    exits = {
        "EXIT_OK": cli_output.EXIT_OK,
        "EXIT_ERROR": cli_output.EXIT_ERROR,
        "EXIT_USAGE": cli_output.EXIT_USAGE,
    }
    codes: dict[str, set[int]] = {}
    for path in sorted((ROOT / "openmed").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else None
            )
            if name != "CliError":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            code = keywords.get("code")
            if not isinstance(code, ast.Constant):
                continue
            exit_node = keywords.get("exit_code")
            if exit_node is None:
                status = cli_output.EXIT_ERROR
            elif isinstance(exit_node, ast.Name) and exit_node.id in exits:
                status = exits[exit_node.id]
            elif isinstance(exit_node, ast.Constant):
                status = int(exit_node.value)
            else:
                continue
            codes.setdefault(str(code.value), set()).add(status)
    return codes


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------


def test_registry_document_matches_its_json_schema(document) -> None:
    schema = load_error_registry_schema()
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: list(error.absolute_path),
    )
    assert [list(error.absolute_path) for error in errors] == []
    assert document["contract"] == ERROR_REGISTRY_CONTRACT
    assert tuple(document["categories"]) == CATEGORIES


def test_registry_schema_is_bundled_and_fingerprinted() -> None:
    assert "error_code_registry" in SCHEMA_NAMES
    schema = load_schema("error_code_registry")
    assert schema == load_error_registry_schema()
    assert (
        load_schema_snapshot()["error_code_registry"]
        == build_schema_snapshot({"error_code_registry": schema})["error_code_registry"]
    )


def test_categories_match_the_protocol_map() -> None:
    protocol_map = json.loads(PROTOCOL_MAP.read_text(encoding="utf-8"))
    assert tuple(protocol_map["categories"]) == CATEGORIES


def test_codes_are_sorted_unique_and_never_reused(document, registry) -> None:
    codes = [entry["code"] for entry in document["codes"]]
    assert codes == sorted(codes)
    assert len({code.casefold() for code in codes}) == len(codes)
    assert not {code.casefold() for code in codes} & {
        code.casefold() for code in document["retired_codes"]
    }
    assert tuple(registry.codes) == tuple(codes)
    assert lookup_error_code("input_error").code == "input_error"
    with pytest.raises(KeyError) as excinfo:
        registry.entry(SENTINEL)
    assert SENTINEL not in str(excinfo.value)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda d: d["codes"].append(copy.deepcopy(d["codes"][0])),
            id="repeated-code",
        ),
        pytest.param(
            lambda d: d["codes"].append(
                {**copy.deepcopy(d["codes"][0]), "code": d["codes"][0]["code"].upper()}
            ),
            id="case-folded-code",
        ),
        pytest.param(
            lambda d: d["retired_codes"].append(d["codes"][0]["code"]),
            id="reused-retired-code",
        ),
        pytest.param(
            lambda d: d["codes"][0].update(
                retryable=True, retryable_basis="server_fault"
            ),
            id="retryable-contradicts-basis",
        ),
        pytest.param(
            lambda d: d["codes"][0].update(
                retryable=None, retryable_basis="server_fault"
            ),
            id="unknown-flag-with-evidence",
        ),
        pytest.param(
            lambda d: d["codes"][0]["divergences"].append("synthetic_unknown"),
            id="unknown-divergence",
        ),
        pytest.param(
            lambda d: d["divergences"].append(
                {**copy.deepcopy(d["divergences"][0]), "id": "synthetic_unused"}
            ),
            id="unused-divergence",
        ),
        pytest.param(
            lambda d: d["codes"][0].update(category=SENTINEL),
            id="unknown-category",
        ),
        pytest.param(
            lambda d: d["codes"][0].update(code=SENTINEL + "-not-a-code"),
            id="malformed-code",
        ),
        pytest.param(
            lambda d: d["codes"][0]["http"].append({"code": "input_error"}),
            id="response-without-status",
        ),
        pytest.param(
            lambda d: d.update(contract_version=SENTINEL),
            id="non-semver-version",
        ),
    ],
)
def test_parser_rejects_invalid_documents_without_echoing_values(
    document, mutate
) -> None:
    mutated = copy.deepcopy(document)
    mutate(mutated)
    with pytest.raises(ErrorRegistryError) as excinfo:
        parse_error_registry(mutated)
    assert SENTINEL not in str(excinfo.value)
    assert document["codes"][0]["code"] not in str(excinfo.value)


def test_parser_rejects_a_portable_code_without_a_flag_or_class(document) -> None:
    mutated = copy.deepcopy(document)
    portable = next(entry for entry in mutated["codes"] if entry["portable"])
    portable["python"] = []
    with pytest.raises(ErrorRegistryError):
        parse_error_registry(mutated)


def test_loader_rejects_repeated_json_keys(tmp_path: Path) -> None:
    text = DATA_PATH.read_text(encoding="utf-8")
    repeated = text.replace(
        '"contract": "openmed.errors.code-registry",',
        '"contract": "openmed.errors.code-registry",\n'
        '  "contract": "openmed.errors.code-registry",',
        1,
    )
    path = tmp_path / "registry.json"
    path.write_text(repeated, encoding="utf-8")
    with pytest.raises(ErrorRegistryError):
        load_error_registry(path)
    with pytest.raises(ErrorRegistryError):
        load_error_registry(tmp_path / "missing.json")


def test_retryable_follows_its_basis(registry) -> None:
    for entry in registry.codes.values():
        assert (entry.retryable is None) == (entry.retryable_basis == "undetermined")
        if any(mapping.retry_after for mapping in entry.http) or any(
            mapping.retry_after for mapping in entry.graphql
        ):
            assert entry.retryable is True, entry.code


def test_divergences_point_at_real_protocol_map_entries(registry) -> None:
    protocol_map = json.loads(PROTOCOL_MAP.read_text(encoding="utf-8"))
    known = {item["id"] for item in protocol_map["known_divergences"]}
    for divergence in registry.divergences.values():
        if divergence.protocol_map is not None:
            assert divergence.protocol_map in known, divergence.id


def test_protocol_map_follow_ups_no_longer_wait_on_the_registry() -> None:
    protocol_map = json.loads(PROTOCOL_MAP.read_text(encoding="utf-8"))
    for item in protocol_map["known_divergences"]:
        assert "R2" not in item["follow_up"], item["id"]


# ---------------------------------------------------------------------------
# Python surface
# ---------------------------------------------------------------------------


def test_every_taxonomy_class_is_registered(registry) -> None:
    for name, code in taxonomy.ERROR_CODES.items():
        entry = registry.entry(code)
        assert f"openmed.core.errors.{name}" in entry.python
        assert entry.mcp, code
        assert entry.http_status is not None, code


def test_taxonomy_fixture_agrees_with_the_registry(registry) -> None:
    fixture = json.loads(TAXONOMY_FIXTURE.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        entry = registry.entry(case["code"])
        assert f"openmed.core.errors.{case['class']}" in entry.python


def test_literal_taxonomy_codes_are_registered(registry) -> None:
    found = set(_taxonomy_call_codes())
    assert found, "the taxonomy call scan found nothing"
    for class_name, code in sorted(found):
        entry = registry.entry(code)
        assert any(path.endswith("." + class_name) for path in entry.python), code


def test_python_class_paths_import_and_carry_their_code(registry) -> None:
    for entry in registry.codes.values():
        for path in entry.python:
            cls = _import_path(path)
            assert inspect.isclass(cls) and issubclass(cls, BaseException), path
            declared = [
                value
                for attribute in ("code", "error_code")
                if isinstance(value := getattr(cls, attribute, None), str)
            ]
            if declared and entry.code not in declared:
                parameters = set(inspect.signature(cls.__init__).parameters)
                assert parameters & _INSTANCE_CODE_PARAMETERS, (entry.code, path)


def _exception_classes(*packages: str) -> Iterator[type[BaseException]]:
    import pkgutil

    seen: set[type[BaseException]] = set()
    for package_name in packages:
        package = importlib.import_module(package_name)
        modules = [package]
        if hasattr(package, "__path__"):
            for info in pkgutil.walk_packages(package.__path__, package_name + "."):
                if ".proto." in info.name:
                    continue
                modules.append(importlib.import_module(info.name))
        for module in modules:
            for value in vars(module).values():
                if (
                    inspect.isclass(value)
                    and issubclass(value, BaseException)
                    and value.__module__ == module.__name__
                    and value not in seen
                ):
                    seen.add(value)
                    yield value


def test_class_declared_codes_are_registered(registry) -> None:
    """Every exception that declares a code is listed under that code."""
    checked = 0
    for cls in _exception_classes(
        "openmed.core.errors", "openmed.service", "openmed.mcp"
    ):
        for attribute in ("code", "error_code"):
            value = cls.__dict__.get(attribute)
            if not isinstance(value, str):
                continue
            path = f"{cls.__module__}.{cls.__qualname__}"
            entry = registry.entry(value)
            if not cls.__name__.startswith("_"):
                assert path in entry.python, (value, path)
            checked += 1
    assert checked >= len(taxonomy.ERROR_CODES)


def test_drain_refusal_class_carries_the_registered_code(registry) -> None:
    from openmed.service.drain import SHUTTING_DOWN, ServiceShuttingDownError

    entry = registry.entry(ServiceShuttingDownError.reason_code)
    assert entry.code == SHUTTING_DOWN
    assert "openmed.service.drain.ServiceShuttingDownError" in entry.python


def test_portable_codes_are_taxonomy_codes_with_complete_mappings(registry) -> None:
    portable = registry.portable_codes()
    assert portable
    assert {entry.code for entry in portable} <= set(taxonomy.ERROR_CODES.values())
    for entry in portable:
        assert entry.retryable is not None, entry.code
        assert entry.http_status is not None, entry.code
        assert entry.cli_exit_code == cli_output.EXIT_ERROR, entry.code
        assert entry.category in CATEGORIES


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_exit_codes_match_the_dispatcher(registry) -> None:
    assert dict(registry.exit_codes) == {
        cli_output.EXIT_OK: "success",
        cli_output.EXIT_ERROR: "failure",
        cli_output.EXIT_USAGE: "usage",
    }


def _run_cli_with_failing_handler(monkeypatch, error: BaseException) -> tuple:
    cli_main = importlib.import_module("openmed.cli.main")

    def failing_handler(_args: Any) -> int:
        raise error

    monkeypatch.setattr(cli_main, "_handle_analyze", failing_handler)
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        status = cli_main.main(["analyze", "--text", "synthetic", "--json"])
    return status, json.loads(stdout.getvalue())


@pytest.mark.parametrize("class_name", sorted(taxonomy.ERROR_CODES))
def test_cli_reports_a_taxonomy_error_as_registered(
    monkeypatch, registry, class_name
) -> None:
    cls = getattr(taxonomy, class_name)
    error = cls(f"Synthetic {SENTINEL} failure.")
    status, envelope = _run_cli_with_failing_handler(monkeypatch, error)
    mapping = registry.entry(taxonomy.ERROR_CODES[class_name]).cli[0]
    assert status == mapping.exit_code
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == mapping.code
    assert SENTINEL not in json.dumps(envelope)


def test_cli_reports_an_unavailable_command_as_registered(
    monkeypatch, registry
) -> None:
    from openmed.cli.main import _UnavailableCommandError

    status, envelope = _run_cli_with_failing_handler(
        monkeypatch, _UnavailableCommandError("Synthetic command is unavailable.")
    )
    mapping = registry.entry("not_implemented").cli[0]
    assert (status, envelope["error"]["code"]) == (mapping.exit_code, mapping.code)


def test_cli_literal_codes_that_are_registry_codes_are_recorded(registry) -> None:
    literal = _cli_literal_codes()
    assert literal, "the CliError scan found nothing"
    recorded = {
        (mapping.code, mapping.exit_code)
        for entry in registry.codes.values()
        for mapping in entry.cli
    }
    for code, statuses in sorted(literal.items()):
        if code not in registry.codes:
            continue
        for status in statuses:
            assert (code, status) in recorded, (code, status)
        assert any(mapping.code == code for mapping in registry.entry(code).cli)
    cli_codes = {code for code, _ in recorded}
    assert cli_codes <= set(literal) | {"runtime_error", "not_implemented"}


# ---------------------------------------------------------------------------
# Generated kit enums
# ---------------------------------------------------------------------------


def _generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "generate_error_code_enums", ROOT / "scripts" / "generate_error_code_enums.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_kit_enums_are_current() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_error_code_enums.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_generated_kit_enums_hold_exactly_the_portable_codes(registry) -> None:
    generator = _generator()
    rendered = generator.render_all(registry)
    assert rendered == generator.render_all(registry)
    assert set(rendered) == {
        generator.SWIFT_PATH,
        generator.KOTLIN_PATH,
        generator.TYPESCRIPT_PATH,
    }
    portable = {entry.code for entry in registry.portable_codes()}
    for text in rendered.values():
        quoted = set(re.findall(r'"([A-Za-z][A-Za-z0-9_]*)"', text))
        assert quoted == portable
        assert registry.contract_version in text
    kotlin = rendered[generator.KOTLIN_PATH]
    for entry in registry.portable_codes():
        flag = "true" if entry.retryable else "false"
        assert f'{entry.code.upper()}("{entry.code}", {flag})' in kotlin
        assert f"  {entry.code}: {flag}," in rendered[generator.TYPESCRIPT_PATH]


def test_generator_check_names_only_the_stale_file(tmp_path: Path) -> None:
    generator = _generator()
    for relative, text in generator.render_all(load_error_registry()).items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    assert generator.stale_files(tmp_path) == []
    stale = tmp_path / generator.KOTLIN_PATH
    stale.write_text(stale.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    (tmp_path / generator.TYPESCRIPT_PATH).unlink()
    assert generator.stale_files(tmp_path) == [
        generator.KOTLIN_PATH.as_posix(),
        generator.TYPESCRIPT_PATH.as_posix(),
    ]


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


def _markdown_table(text: str, header: str) -> list[list[str]]:
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_errors_doc_taxonomy_table_agrees_with_the_registry(registry) -> None:
    text = ERRORS_DOC.read_text(encoding="utf-8")
    rows = _markdown_table(text, "| Python exception |")
    assert len(rows) == len(taxonomy.ERROR_CODES)
    for exception, code, _bases, status, mcp_code in rows:
        name = exception.strip("`")
        entry = registry.entry(code.strip("`"))
        assert f"openmed.core.errors.{name}" in entry.python
        assert int(status) == entry.http_status
        assert mcp_code.strip("`") == entry.mcp[0].code


def test_errors_doc_describes_the_registry_and_portable_codes(registry) -> None:
    text = ERRORS_DOC.read_text(encoding="utf-8")
    assert "openmed/core/error_codes.json" in text
    assert "scripts/generate_error_code_enums.py" in text
    rows = _markdown_table(text, "| Portable code |")
    documented = {row[0].strip("`"): row for row in rows}
    assert set(documented) == {entry.code for entry in registry.portable_codes()}
    for entry in registry.portable_codes():
        _code, category, status, exit_code, retryable = documented[entry.code]
        assert category.strip("`") == entry.category
        assert int(status) == entry.http_status
        assert int(exit_code) == entry.cli_exit_code
        assert retryable == ("yes" if entry.retryable else "no")
    for divergence in registry.divergences:
        assert f"`{divergence}`" in text, divergence


def test_registry_text_is_value_free(document) -> None:
    text = DATA_PATH.read_text(encoding="utf-8")
    assert SENTINEL not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    assert "/Users/" not in text and "C:\\" not in text
