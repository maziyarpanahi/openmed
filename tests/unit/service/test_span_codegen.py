"""The protobuf ``OpenMedSpan`` message is generated from the span schema."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.core.schemas import load_schema
from openmed.service.proto.generated import openmed_pb2
from scripts.codegen import span_proto
from scripts.codegen.span_proto import (
    DEFAULT_FIELD_MAP_PATH,
    DEFAULT_PROTO_PATH,
    DEFAULT_SCHEMA_PATH,
    FieldMap,
    ReservedField,
    SpanCodegenError,
    check,
    extract_message,
    generate,
    load_field_map,
    proto_field,
    render_field_map,
    render_message,
    update_field_map,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "codegen" / "span_proto.py"
# The shipped wire contract: name -> (number, proto type, explicit presence).
SHIPPED_FIELDS = {
    "schema_version": (1, "int32", False),
    "doc_id": (2, "string", False),
    "start": (3, "int64", False),
    "end": (4, "int64", False),
    "text_hash": (5, "string", False),
    "entity_type": (6, "string", False),
    "canonical_label": (7, "string", False),
    "policy_label": (8, "string", True),
    "regulatory_tags": (9, "repeated string", False),
    "score": (10, "double", True),
    "detector": (11, "string", True),
    "evidence": (12, "google.protobuf.Struct", False),
    "action": (13, "string", False),
    "replacement": (14, "string", True),
    "reversible_id": (15, "string", True),
    "section": (16, "string", True),
    "metadata": (17, "google.protobuf.Struct", False),
}
_PROTO_TYPES = {
    1: "double",
    3: "int64",
    5: "int32",
    8: "bool",
    9: "string",
}


def _copy_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "schema_path": tmp_path / "span.schema.json",
        "proto_path": tmp_path / "openmed.proto",
        "field_map_path": tmp_path / "openmed_span_fields.json",
    }
    shutil.copyfile(DEFAULT_SCHEMA_PATH, paths["schema_path"])
    shutil.copyfile(DEFAULT_PROTO_PATH, paths["proto_path"])
    shutil.copyfile(DEFAULT_FIELD_MAP_PATH, paths["field_map_path"])
    return paths


def _edit_schema(path: Path, edit) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    edit(schema)
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


def _descriptor_fields() -> dict[str, tuple[int, str, bool]]:
    fields: dict[str, tuple[int, str, bool]] = {}
    for item in openmed_pb2.OpenMedSpan.DESCRIPTOR.fields:
        if item.message_type is not None:
            proto_type = item.message_type.full_name
        else:
            proto_type = _PROTO_TYPES[item.type]
        repeated = getattr(item, "is_repeated", None)
        if repeated is None:  # protobuf releases before is_repeated
            repeated = item.label == item.LABEL_REPEATED
        if repeated:
            proto_type = f"repeated {proto_type}"
        presence = item.containing_oneof is not None
        fields[item.name] = (item.number, proto_type, presence)
    return fields


def test_committed_block_is_byte_identical_to_the_generated_message() -> None:
    schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    committed = extract_message(DEFAULT_PROTO_PATH.read_text(encoding="utf-8"))

    assert committed == render_message(schema, load_field_map())
    assert check() == []
    assert render_field_map(load_field_map()) == DEFAULT_FIELD_MAP_PATH.read_text(
        encoding="utf-8"
    )


def test_check_mode_passes_on_the_committed_files() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_field_numbers_and_types_one_to_seventeen_are_unchanged() -> None:
    assert _descriptor_fields() == SHIPPED_FIELDS
    assert dict(load_field_map().fields) == {
        name: number for name, (number, _, _) in SHIPPED_FIELDS.items()
    }


def test_every_schema_property_appears_on_the_message() -> None:
    schema = load_schema("span")

    assert set(schema["properties"]) == set(_descriptor_fields())


def test_added_property_takes_the_next_free_number(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    _edit_schema(
        paths["schema_path"],
        lambda schema: schema["properties"].update(
            {"synthetic_note_ref": {"type": ["string", "null"]}}
        ),
    )

    assert check(**paths) != []
    assert generate(**paths) is True
    assert check(**paths) == []
    field_map = load_field_map(paths["field_map_path"])
    assert field_map.fields["synthetic_note_ref"] == 18
    block = extract_message(paths["proto_path"].read_text(encoding="utf-8"))
    assert "  optional string synthetic_note_ref = 18;\n" in block
    for name, (number, _, _) in SHIPPED_FIELDS.items():
        assert f" {name} = {number};\n" in block


def test_removed_property_becomes_reserved_and_is_never_reused(
    tmp_path: Path,
) -> None:
    paths = _copy_inputs(tmp_path)

    def drop_section(schema: dict[str, Any]) -> None:
        del schema["properties"]["section"]
        schema["required"].remove("section")

    _edit_schema(paths["schema_path"], drop_section)
    assert generate(**paths) is True
    field_map = load_field_map(paths["field_map_path"])
    assert "section" not in field_map.fields
    assert field_map.reserved == (ReservedField(number=16, name="section"),)
    block = extract_message(paths["proto_path"].read_text(encoding="utf-8"))
    assert '  reserved 16;\n  reserved "section";\n' in block
    assert " section = 16;" not in block

    _edit_schema(
        paths["schema_path"],
        lambda schema: schema["properties"].update(
            {"synthetic_added": {"type": "boolean"}}
        ),
    )
    generate(**paths)
    assert load_field_map(paths["field_map_path"]).fields["synthetic_added"] == 18

    _edit_schema(
        paths["schema_path"],
        lambda schema: schema["properties"].update({"section": {"type": "string"}}),
    )
    with pytest.raises(SpanCodegenError, match="reserved field name"):
        generate(**paths)


def test_schema_change_without_regeneration_fails_the_check(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    _edit_schema(
        paths["schema_path"],
        lambda schema: schema["properties"]["score"].update({"type": "number"}),
    )

    assert check(**paths) == ["openmed.proto OpenMedSpan block differs from the schema"]
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--schema",
            str(paths["schema_path"]),
            "--proto",
            str(paths["proto_path"]),
            "--field-map",
            str(paths["field_map_path"]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "make grpc-proto" in result.stderr


def test_check_reports_unnumbered_properties(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    _edit_schema(
        paths["schema_path"],
        lambda schema: schema["properties"].update({"synthetic_x": {"type": "string"}}),
    )

    problems = check(**paths)

    assert len(problems) == 1
    assert "synthetic_x" in problems[0]


@pytest.mark.parametrize(
    ("prop", "required", "expected"),
    [
        ({"type": "string"}, True, ("", "string")),
        ({"type": "string"}, False, ("optional ", "string")),
        ({"type": ["string", "null"]}, True, ("optional ", "string")),
        ({"type": "boolean"}, True, ("", "bool")),
        ({"type": ["number", "null"]}, True, ("optional ", "double")),
        ({"type": "integer", "const": 1}, True, ("", "int32")),
        ({"type": "integer", "enum": [1, 2]}, True, ("", "int32")),
        (
            {"type": "integer", "minimum": 0, "maximum": 10},
            True,
            ("", "int32"),
        ),
        ({"type": "integer", "minimum": 0}, True, ("", "int64")),
        (
            {"type": "integer", "minimum": 0, "maximum": 2**40},
            True,
            ("", "int64"),
        ),
        ({"type": "object"}, False, ("", "google.protobuf.Struct")),
        (
            {"type": "array", "items": {"type": "string"}},
            False,
            ("repeated ", "string"),
        ),
        (
            {"type": "array", "items": {"type": "object"}},
            True,
            ("repeated ", "google.protobuf.Struct"),
        ),
    ],
)
def test_scalar_and_presence_mapping(
    prop: dict[str, Any], required: bool, expected: tuple[str, str]
) -> None:
    assert proto_field("synthetic", prop, required=required) == expected


@pytest.mark.parametrize(
    "prop",
    [
        {"type": ["string", "integer"]},
        {"type": "null"},
        {},
        {"type": "array"},
        {"type": "array", "items": {"type": ["string", "null"]}},
    ],
)
def test_unsupported_property_types_are_rejected(prop: dict[str, Any]) -> None:
    with pytest.raises(SpanCodegenError):
        proto_field("synthetic", prop, required=True)


def test_explicit_presence_keeps_shipped_optional_fields() -> None:
    schema = load_schema("span")
    field_map = load_field_map()

    assert field_map.explicit_presence == frozenset({"policy_label"})
    assert "policy_label" in schema["required"]
    assert proto_field(
        "policy_label",
        schema["properties"]["policy_label"],
        required=True,
        explicit_presence=True,
    ) == ("optional ", "string")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fields": {"a": 1, "b": 1}},
        {"fields": {"a": 1}, "reserved": (ReservedField(number=1, name="b"),)},
        {"fields": {"a": 0}},
        {"fields": {"a": 19000}},
        {"fields": {"a": 1}, "reserved": (ReservedField(number=2, name="a"),)},
        {"fields": {"a": 1}, "explicit_presence": frozenset({"b"})},
        {"fields": {"Bad-Name": 1}},
    ],
)
def test_invalid_field_maps_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(SpanCodegenError):
        FieldMap(**kwargs)


def test_update_skips_the_implementation_reserved_range() -> None:
    field_map = FieldMap(fields={"a": 18999})
    schema = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}}}

    assert update_field_map(schema, field_map).fields["b"] == 20000


def test_generate_is_idempotent_on_the_committed_inputs(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    before = {name: path.read_bytes() for name, path in paths.items()}

    assert generate(**paths) is False
    assert {name: path.read_bytes() for name, path in paths.items()} == before


def test_proto_without_exactly_one_block_is_rejected() -> None:
    text = DEFAULT_PROTO_PATH.read_text(encoding="utf-8")
    block = extract_message(text)

    with pytest.raises(SpanCodegenError):
        extract_message(text.replace(block, ""))
    with pytest.raises(SpanCodegenError):
        extract_message(text + "\n" + block)


def test_main_returns_usage_error_for_an_unreadable_field_map(tmp_path: Path) -> None:
    paths = _copy_inputs(tmp_path)
    paths["field_map_path"].write_text("{}", encoding="utf-8")
    argv = [
        "--schema",
        str(paths["schema_path"]),
        "--proto",
        str(paths["proto_path"]),
        "--field-map",
        str(paths["field_map_path"]),
    ]
    proto_before = paths["proto_path"].read_bytes()

    assert span_proto.main(["--check", *argv]) == 2
    assert span_proto.main(argv) == 2
    assert paths["proto_path"].read_bytes() == proto_before
