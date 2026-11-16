"""Drift and fail-closed checks for the generated Swift contract types."""

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
SCRIPT = CODEGEN_DIR / "emit_swift_types.py"
SCHEMA_DIR = ROOT / "openmed" / "core" / "schemas" / "json"
SWIFT_SOURCES = ROOT / "swift" / "OpenMedKit" / "Sources"
GENERATED = SWIFT_SOURCES / "OpenMedKit" / "Generated" / "ContractTypes.swift"
SWIFT_TEST = (
    ROOT
    / "swift"
    / "OpenMedKit"
    / "Tests"
    / "OpenMedKitTests"
    / "ContractTypesTests.swift"
)


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
emit_swift_types = _load("emit_swift_types", SCRIPT)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _schemas_with(tmp_path: Path, name: str, edit: Any) -> Path:
    target = tmp_path / "schemas"
    target.mkdir()
    for schema in contract_schema.CORE_SCHEMAS:
        file_name = contract_schema.schema_file_name(schema)
        shutil.copyfile(SCHEMA_DIR / file_name, target / file_name)
    path = target / contract_schema.schema_file_name(name)
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target


def _struct(source: str, name: str) -> str:
    start = source.index(f"public struct {name}: Codable, Equatable, Sendable {{\n")
    return source[start : source.index("\n}\n", start)]


def test_committed_swift_contract_types_are_current() -> None:
    result = _run("--check")

    assert result.returncode == 0, result.stderr
    assert GENERATED.read_text(encoding="utf-8") == emit_swift_types.render(
        contract_schema.load_contract_model()
    )


def test_generation_is_deterministic_and_covers_every_core_schema() -> None:
    model = contract_schema.load_contract_model()
    first = emit_swift_types.render(model)

    assert first == emit_swift_types.render(contract_schema.load_contract_model())
    assert first.endswith("}\n")
    for record in model.records:
        assert (
            f"public struct OpenMedContract{record.name}: Codable, Equatable, "
            "Sendable {" in first
        ), record.name


def test_every_generated_public_type_is_prefixed_and_unique_in_the_module() -> None:
    source = GENERATED.read_text(encoding="utf-8")
    declared = re.findall(r"^public (?:struct|enum) (\w+)", source, re.MULTILINE)

    assert declared and len(declared) == len(set(declared))
    assert all(name.startswith("OpenMedContract") for name in declared)
    for path in SWIFT_SOURCES.rglob("*.swift"):
        if path == GENERATED:
            continue
        text = path.read_text(encoding="utf-8")
        for name in declared:
            assert not re.search(rf"\b(?:struct|enum|class) {name}\b", text), name


def test_nullable_required_fields_keep_explicit_nulls() -> None:
    span = _struct(GENERATED.read_text(encoding="utf-8"), "OpenMedContractSpan")

    assert "public let score: Double?" in span
    assert "if try container.decodeNil(forKey: .score) {" in span
    assert "try container.encodeNil(forKey: .score)" in span
    assert "decodeIfPresent" not in span and "encodeIfPresent" not in span
    assert 'case textHash = "text_hash"' in span
    assert "public let evidence: [String: OpenMedContractJSONValue]" in span
    assert "rejectUndeclaredKeys(" in span


def test_constants_are_fixed_and_checked_on_decode() -> None:
    source = GENERATED.read_text(encoding="utf-8")
    span = _struct(source, "OpenMedContractSpan")
    execution = _struct(source, "OpenMedContractDataResidencyAttestationExecution")

    assert "self.schemaVersion = 1" in span
    assert "guard self.schemaVersion == 1 else {" in span
    assert 'debugDescription: "schema_version must be 1"' in span
    assert "schemaVersion: Int," not in span.split("public init(from")[0]
    assert "guard self.hostLocal == true else {" in execution


def test_shared_enums_are_named_and_value_free_on_failure() -> None:
    source = GENERATED.read_text(encoding="utf-8")

    for name in ("CanonicalLabel", "PolicyLabel", "SpanAction"):
        assert (
            f"public enum OpenMedContract{name}: String, Codable, CaseIterable, "
            "Sendable {" in source
        )
        assert f'debugDescription: "Unknown OpenMedContract{name} value"' in source
    assert "public let defaultAction: OpenMedContractSpanAction" in source
    assert 'case formatPreserve = "format_preserve"' in source
    assert 'case configEnvironment = "config+environment"' in source


def test_adding_a_property_changes_output_and_fails_check(tmp_path: Path) -> None:
    def add(document: dict[str, Any]) -> None:
        document["properties"]["synthetic_count"] = {"type": "integer"}
        document["required"].append("synthetic_count")

    schema_dir = _schemas_with(tmp_path, "relation", add)
    rendered = emit_swift_types.render(contract_schema.load_contract_model(schema_dir))

    relation = _struct(rendered, "OpenMedContractRelation")
    assert "public let syntheticCount: Int" in relation
    assert 'case syntheticCount = "synthetic_count"' in relation

    stale = _run("--check", "--schema-dir", str(schema_dir))
    assert stale.returncode == 1
    assert "Generated/ContractTypes.swift" in stale.stderr
    assert "synthetic_count" not in stale.stderr

    output = tmp_path / "ContractTypes.swift"
    args = ("--schema-dir", str(schema_dir), "--output", str(output))
    assert _run(*args).returncode == 0
    assert _run("--check", *args).returncode == 0


def test_optional_property_uses_if_present_coding(tmp_path: Path) -> None:
    def add(document: dict[str, Any]) -> None:
        document["properties"]["synthetic_note"] = {"type": "string"}

    schema_dir = _schemas_with(tmp_path, "code", add)
    code = _struct(
        emit_swift_types.render(contract_schema.load_contract_model(schema_dir)),
        "OpenMedContractCode",
    )

    assert "public let syntheticNote: String?" in code
    assert "syntheticNote: String? = nil" in code
    assert "decodeIfPresent(String.self, forKey: .syntheticNote)" in code
    assert "encodeIfPresent(self.syntheticNote, forKey: .syntheticNote)" in code


def test_swift_keywords_are_escaped(tmp_path: Path) -> None:
    def add(document: dict[str, Any]) -> None:
        document["properties"]["default"] = {"type": "boolean"}
        document["required"].append("default")

    schema_dir = _schemas_with(tmp_path, "provenance", add)
    provenance = _struct(
        emit_swift_types.render(contract_schema.load_contract_model(schema_dir)),
        "OpenMedContractProvenance",
    )

    assert "public let `default`: Bool" in provenance
    assert 'case `default` = "default"' in provenance
    assert "self.default = `default`" in provenance


def test_enum_case_name_collisions_fail_closed(tmp_path: Path) -> None:
    def add(document: dict[str, Any]) -> None:
        document["properties"]["synthetic_kind"] = {
            "type": "string",
            "enum": ["a_b", "a-b"],
        }
        document["required"].append("synthetic_kind")

    schema_dir = _schemas_with(tmp_path, "audit", add)

    with pytest.raises(contract_schema.ContractSchemaError) as raised:
        emit_swift_types.render(contract_schema.load_contract_model(schema_dir))

    assert raised.value.code == "enum_case_name_collision"
    assert "a_b" not in str(raised.value)


def test_swift_decode_tests_cover_every_core_schema_and_the_parity_fixture() -> None:
    source = SWIFT_TEST.read_text(encoding="utf-8")
    model = contract_schema.load_contract_model()

    for record in model.roots:
        assert (
            f'"{record.schema}": roundTrip(OpenMedContract{record.name}.self)' in source
        ), record.schema
    assert "fixtures/parity/expected_spans.json" in source
    assert "tests/fixtures/contracts/core_contract_examples.json" in source
