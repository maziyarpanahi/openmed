"""Drift and fail-closed checks for the generated TypeScript contract types."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
CODEGEN_DIR = ROOT / "scripts" / "codegen"
SCRIPT = CODEGEN_DIR / "emit_ts_types.py"
SCHEMA_DIR = ROOT / "openmed" / "core" / "schemas" / "json"
GENERATED = ROOT / "js" / "openmedkit-web" / "src" / "generated" / "contract-types.ts"
WEB_TYPES = ROOT / "js" / "openmedkit-web" / "src" / "types.ts"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"


def _load(name: str, path: Path) -> ModuleType:
    if str(CODEGEN_DIR) not in sys.path:
        sys.path.insert(0, str(CODEGEN_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract_schema = _load("contract_schema", CODEGEN_DIR / "contract_schema.py")
emit_ts_types = _load("emit_ts_types", SCRIPT)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _copy_schemas(tmp_path: Path) -> Path:
    target = tmp_path / "schemas"
    target.mkdir()
    for name in contract_schema.CORE_SCHEMAS:
        file_name = contract_schema.schema_file_name(name)
        shutil.copyfile(SCHEMA_DIR / file_name, target / file_name)
    return target


def _edit_schema(schema_dir: Path, name: str, edit: Any) -> None:
    path = schema_dir / contract_schema.schema_file_name(name)
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def _ts_const_array(source: str, name: str) -> list[str]:
    match = re.search(
        rf"^export const {name} = \[\n(?P<body>.*?)\n\] as const;$",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{name} is missing"
    return re.findall(r'^  "([^"]+)",$', match.group("body"), re.MULTILINE)


def test_committed_typescript_contract_types_are_current() -> None:
    result = _run("--check")

    assert result.returncode == 0, result.stderr
    assert GENERATED.read_text(encoding="utf-8") == emit_ts_types.render(
        contract_schema.load_contract_model()
    )


def test_generation_is_deterministic_and_covers_every_core_schema() -> None:
    first = emit_ts_types.render(contract_schema.load_contract_model())
    second = emit_ts_types.render(contract_schema.load_contract_model())

    assert first == second
    assert first.endswith("\n") and not first.endswith("\n\n")
    for name in contract_schema.CORE_SCHEMAS:
        title = json.loads(
            (SCHEMA_DIR / contract_schema.schema_file_name(name)).read_text("utf-8")
        )["title"]
        interface = "OpenMedContract" + title.removeprefix("OpenMed")
        assert f"export interface {interface} {{" in first, name


def test_generated_module_contains_only_types_and_const_arrays() -> None:
    source = GENERATED.read_text(encoding="utf-8")
    exports = re.findall(r"^export .*$", source, re.MULTILINE)

    assert exports
    for line in exports:
        assert re.match(
            r"^export (interface \w+ \{|type \w+ = \(typeof \w+\)\[number\];"
            r"|const \w+ = \[)$",
            line,
        ), line
    assert "import " not in source and "require(" not in source
    assert source.count("] as const;") == len(contract_schema.NAMED_ENUMS)


def test_generated_lists_match_the_schemas_and_core_taxonomy() -> None:
    from openmed.core.labels import CANONICAL_LABELS, POLICY_LABELS

    source = GENERATED.read_text(encoding="utf-8")
    span = json.loads((SCHEMA_DIR / "span.schema.json").read_text("utf-8"))

    canonical = _ts_const_array(source, "CANONICAL_LABELS")
    policy = _ts_const_array(source, "POLICY_LABELS")
    actions = _ts_const_array(source, "SPAN_ACTIONS")
    assert canonical == span["properties"]["canonical_label"]["enum"]
    assert policy == span["properties"]["policy_label"]["enum"]
    assert actions == span["properties"]["action"]["enum"]
    assert set(canonical) <= CANONICAL_LABELS
    assert set(policy) == POLICY_LABELS


def test_shared_enums_reuse_one_named_type() -> None:
    source = GENERATED.read_text(encoding="utf-8")

    assert "  policy_label: PolicyLabel;" in source
    assert "  action: SpanAction;" in source
    assert "  default_action: SpanAction;" in source
    assert "  canonical_label: CanonicalLabel;" in source
    # entity.schema.json keeps canonical_label free-form.
    entity = source.split("export interface OpenMedContractEntity {", 1)[1]
    assert entity.split("}", 1)[0].count("canonical_label: string;") == 1


def test_adding_a_property_changes_output_and_fails_check(tmp_path: Path) -> None:
    schema_dir = _copy_schemas(tmp_path)

    def add_property(document: dict[str, Any]) -> None:
        document["properties"]["synthetic_flag"] = {"type": ["boolean", "null"]}
        document["required"].append("synthetic_flag")

    _edit_schema(schema_dir, "span", add_property)
    model = contract_schema.load_contract_model(schema_dir)
    rendered = emit_ts_types.render(model)
    assert rendered != GENERATED.read_text(encoding="utf-8")
    assert "  synthetic_flag: boolean | null;" in rendered

    stale = _run("--check", "--schema-dir", str(schema_dir))
    assert stale.returncode == 1
    assert "js/openmedkit-web/src/generated/contract-types.ts" in stale.stderr
    assert "scripts/codegen/emit_ts_types.py" in stale.stderr
    assert "synthetic_flag" not in stale.stderr

    output = tmp_path / "contract-types.ts"
    args = ("--schema-dir", str(schema_dir), "--output", str(output))
    assert _run(*args).returncode == 0
    assert output.read_text(encoding="utf-8") == rendered
    assert _run("--check", *args).returncode == 0


def test_check_reports_a_missing_generated_file(tmp_path: Path) -> None:
    result = _run("--check", "--output", str(tmp_path / "missing.ts"))

    assert result.returncode == 1
    assert "stale" in result.stderr


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (
            lambda doc: doc["properties"]["score"].update(
                {"type": ["number", "string"]}
            ),
            "type_union_unsupported",
        ),
        (
            lambda doc: doc["properties"].update(
                {"ref": {"$ref": "#/properties/start"}}
            ),
            "unsupported_keyword",
        ),
        (
            lambda doc: doc["properties"]["detector"].update(
                {"oneOf": [{"type": "string"}]}
            ),
            "unsupported_keyword",
        ),
        (
            lambda doc: doc.update({"additionalProperties": True}),
            "open_record_unsupported",
        ),
        (
            lambda doc: doc["properties"]["action"].update({"enum": [1, 2]}),
            "enum_not_string",
        ),
        (lambda doc: doc["properties"]["action"].pop("enum"), "named_enum_missing"),
        (
            lambda doc: doc["required"].append("undeclared"),
            "required_property_undeclared",
        ),
        (lambda doc: doc.pop("schema_version"), "schema_version_missing"),
    ],
)
def test_unsupported_schema_constructs_fail_closed(
    tmp_path: Path, edit: Any, code: str
) -> None:
    schema_dir = _copy_schemas(tmp_path)
    _edit_schema(schema_dir, "span", edit)

    with pytest.raises(contract_schema.ContractSchemaError) as raised:
        contract_schema.load_contract_model(schema_dir)

    assert raised.value.code == code
    assert raised.value.location.startswith("span#")
    result = _run("--check", "--schema-dir", str(schema_dir))
    assert result.returncode == 2
    assert code in result.stderr


def test_non_string_enum_on_an_unnamed_property_fails_closed(tmp_path: Path) -> None:
    schema_dir = _copy_schemas(tmp_path)
    _edit_schema(
        schema_dir,
        "audit",
        lambda doc: doc["properties"]["actor"].update({"enum": [1, 2]}),
    )

    with pytest.raises(contract_schema.ContractSchemaError) as raised:
        contract_schema.load_contract_model(schema_dir)

    assert raised.value.code == "enum_not_string"


def test_missing_schema_fails_closed(tmp_path: Path) -> None:
    schema_dir = _copy_schemas(tmp_path)
    (schema_dir / "code.schema.json").unlink()

    with pytest.raises(contract_schema.ContractSchemaError) as raised:
        contract_schema.load_contract_model(schema_dir)

    assert (raised.value.code, raised.value.location) == ("schema_missing", "code#")


def test_pattern_prefixes_become_template_literal_types() -> None:
    source = GENERATED.read_text(encoding="utf-8")

    assert "  text_hash: `hmac-sha256:${string}`;" in source
    assert "  integrity_hash: `sha256:${string}`;" in source
    # A pattern without a literal prefix stays a plain string.
    assert "  country_code: string;" in source


def test_web_span_type_is_guarded_against_the_generated_contract() -> None:
    source = WEB_TYPES.read_text(encoding="utf-8")

    assert 'from "./generated/contract-types";' in source
    assert "MutuallyAssignable<OpenMedSpan, OpenMedContractSpan>" in source
    for name in ("CANONICAL_LABELS", "POLICY_LABELS", "SPAN_ACTIONS"):
        assert f"export const {name}" not in source, name


def test_ci_and_make_run_every_contract_type_emitter_check() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    makefile = MAKEFILE.read_text(encoding="utf-8")

    loop = "for emitter in scripts/codegen/emit_*_types.py; do"
    assert loop in workflow
    assert 'uv run python "$emitter" --check' in workflow
    assert "contract-types-check:" in makefile
    assert loop in makefile
