"""GraphQL SDL and protobuf descriptor classification in the API gate.

Every fixture below is synthetic.  Each breaking class is exercised by a
baseline/candidate pair derived from one small schema, so a failure names the
exact rule that regressed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.service import api_compatibility as gate
from openmed.service.api_compatibility import (
    DEFAULT_GRAPHQL_BASELINE_PATH,
    ERROR_CATEGORIES_EXTENSION,
    GRAPHQL_SURFACE,
    GRPC_SURFACE,
    SCHEMA_SURFACES,
    APICompatibilityError,
    APIContractError,
    APISurfaceUnavailable,
    ProtoContract,
    check_api_compatibility,
    check_graphql_compatibility,
    check_grpc_compatibility,
    compare_contracts,
    compare_graphql_schemas,
    compare_proto_contracts,
    compile_proto_contract,
    load_proto_contract,
    main,
    normalize_proto_descriptors,
    write_schema_baselines,
)

ROOT = Path(__file__).resolve().parents[3]
SENTINELS = (
    "SYNTHETIC_DEFAULT_SENTINEL",
    "SYNTHETIC_OTHER_SENTINEL",
    "SYNTHETIC_DESCRIPTION_SENTINEL",
    "SYNTHETIC_COMMENT_SENTINEL",
)
SCHEMA_ISSUE_KEYS = {
    "change",
    "surface",
    "type_name",
    "member",
    "argument",
    "field_number",
}

BASE_SDL = '''\
"""SYNTHETIC_DESCRIPTION_SENTINEL"""
directive @audit(level: Int = 1) repeatable on FIELD_DEFINITION | OBJECT

interface Node {
  id: ID!
}

type Patient implements Node {
  id: ID!
  label: String!
  score: Float
  tags: [String!]!
  kind: RecordKind!
}

type Device implements Node {
  id: ID!
}

type Audit {
  value: Int
}

union Subject = Patient | Device

enum RecordKind {
  ALPHA
  BETA
}

input SearchInput {
  text: String!
  limit: Int = 10
  mode: String = "SYNTHETIC_DEFAULT_SENTINEL"
}

type Query {
  """SYNTHETIC_DESCRIPTION_SENTINEL"""
  search(input: SearchInput!, first: Int = 20, after: String, \
cursor: String! = "SYNTHETIC_DEFAULT_SENTINEL"): [Patient!]!
  subject(id: ID!): Subject
}
'''

BASE_PROTO = """\
syntax = "proto3";

package synthetic.contract.v1;

// SYNTHETIC_COMMENT_SENTINEL
service RecordService {
  rpc Fetch(FetchRequest) returns (FetchResponse);
  rpc Watch(FetchRequest) returns (stream FetchResponse);
}

service AuditService {
  rpc Ping(FetchRequest) returns (FetchResponse);
}

enum RecordKind {
  RECORD_KIND_UNSPECIFIED = 0;
  RECORD_KIND_ALPHA = 1;
  RECORD_KIND_BETA = 2;
  reserved 5;
  reserved "RECORD_KIND_RETIRED";
}

enum Priority {
  PRIORITY_UNSPECIFIED = 0;
}

message FetchRequest {
  string record_id = 1;
  optional int32 limit = 2;
  repeated string tags = 3;
  RecordKind kind = 4;
  oneof selector {
    string cursor = 5;
    int64 offset = 6;
  }
  reserved 9, 11 to 13;
  reserved "legacy_field";
}

message FetchResponse {
  string record_id = 1;
  double score = 2;
  Nested nested = 3;
  message Nested {
    string value = 1;
  }
}

message Unused {
  string value = 1;
}
"""


def _mutate(source: str, *edits: tuple[str, str]) -> str:
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


def _changes(issues: Any) -> set[str]:
    return {issue.change for issue in issues}


def _assert_identifier_only(payload: Any) -> None:
    text = json.dumps(payload)
    for sentinel in SENTINELS:
        assert sentinel not in text


def _compile(tmp_path: Path, source: str, name: str = "synthetic") -> ProtoContract:
    path = tmp_path / f"{name}.proto"
    path.write_text(source, encoding="utf-8")
    return compile_proto_contract(path)


def _rest_contract() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Synthetic service", "version": "0.0.0"},
        "paths": {"/health": {"get": {"responses": {"200": {}}}}},
        ERROR_CATEGORIES_EXTENSION: ["validation_error"],
    }


@pytest.fixture
def proto_toolchain() -> None:
    pytest.importorskip("grpc_tools")
    pytest.importorskip("google.protobuf")


# ---------------------------------------------------------------------------
# GraphQL SDL
# ---------------------------------------------------------------------------


GRAPHQL_BREAKING_CASES = [
    ("type_removed", [("type Audit {\n  value: Int\n}\n", "")], {"type_name": "Audit"}),
    (
        "type_kind_changed",
        [("type Audit {\n  value: Int\n}", "input Audit {\n  value: Int\n}")],
        {"type_name": "Audit"},
    ),
    (
        "field_removed",
        [("  score: Float\n", "")],
        {"type_name": "Patient", "member": "score"},
    ),
    (
        "field_type_changed",
        [("  score: Float\n", "  score: String\n")],
        {"type_name": "Patient", "member": "score"},
    ),
    (
        "field_type_changed",
        [("  tags: [String!]!\n", "  tags: String!\n")],
        {"type_name": "Patient", "member": "tags"},
    ),
    (
        "field_nullability_relaxed",
        [("  label: String!\n", "  label: String\n")],
        {"type_name": "Patient", "member": "label"},
    ),
    (
        "field_nullability_relaxed",
        [("  tags: [String!]!\n", "  tags: [String]!\n")],
        {"type_name": "Patient", "member": "tags"},
    ),
    (
        "argument_removed",
        [("first: Int = 20, ", "")],
        {"type_name": "Query", "member": "search", "argument": "first"},
    ),
    (
        "required_argument_added",
        [("subject(id: ID!)", "subject(id: ID!, region: String!)")],
        {"type_name": "Query", "member": "subject", "argument": "region"},
    ),
    (
        "argument_type_changed",
        [("first: Int = 20", 'first: String = "20"')],
        {"type_name": "Query", "member": "search", "argument": "first"},
    ),
    (
        "argument_nullability_tightened",
        [("first: Int = 20", "first: Int! = 20")],
        {"type_name": "Query", "member": "search", "argument": "first"},
    ),
    (
        "argument_became_required",
        [("after: String,", "after: String!,")],
        {"type_name": "Query", "member": "search", "argument": "after"},
    ),
    (
        "argument_became_required",
        [('cursor: String! = "SYNTHETIC_DEFAULT_SENTINEL"', "cursor: String!")],
        {"type_name": "Query", "member": "search", "argument": "cursor"},
    ),
    (
        "input_field_removed",
        [("  limit: Int = 10\n", "")],
        {"type_name": "SearchInput", "member": "limit"},
    ),
    (
        "required_input_field_added",
        [("  text: String!\n", "  text: String!\n  region: String!\n")],
        {"type_name": "SearchInput", "member": "region"},
    ),
    (
        "input_field_type_changed",
        [("  limit: Int = 10\n", "  limit: [Int] = [10]\n")],
        {"type_name": "SearchInput", "member": "limit"},
    ),
    (
        "input_field_nullability_tightened",
        [("  limit: Int = 10\n", "  limit: Int! = 10\n")],
        {"type_name": "SearchInput", "member": "limit"},
    ),
    (
        "enum_value_removed",
        [("  BETA\n", "")],
        {"type_name": "RecordKind", "member": "BETA"},
    ),
    (
        "union_member_removed",
        [("= Patient | Device", "= Patient")],
        {"type_name": "Subject", "member": "Device"},
    ),
    (
        "interface_removed",
        [("type Device implements Node {", "type Device {")],
        {"type_name": "Device", "member": "Node"},
    ),
    (
        "directive_removed",
        [
            (
                "directive @audit(level: Int = 1) repeatable on FIELD_DEFINITION"
                " | OBJECT\n",
                "",
            )
        ],
        {"type_name": "@audit"},
    ),
    (
        "directive_location_removed",
        [("on FIELD_DEFINITION | OBJECT", "on FIELD_DEFINITION")],
        {"type_name": "@audit", "member": "OBJECT"},
    ),
    (
        "directive_repeatable_removed",
        [(" repeatable on", " on")],
        {"type_name": "@audit"},
    ),
    (
        "directive_argument_removed",
        [("@audit(level: Int = 1)", "@audit")],
        {"type_name": "@audit", "argument": "level"},
    ),
    (
        "required_directive_argument_added",
        [("(level: Int = 1)", "(level: Int = 1, scope: String!)")],
        {"type_name": "@audit", "argument": "scope"},
    ),
]


@pytest.mark.parametrize(
    ("change", "edits", "location"),
    GRAPHQL_BREAKING_CASES,
    ids=[f"{case[0]}-{index}" for index, case in enumerate(GRAPHQL_BREAKING_CASES)],
)
def test_graphql_breaking_classes_fail(
    change: str, edits: list[tuple[str, str]], location: dict[str, str]
) -> None:
    report = compare_graphql_schemas(BASE_SDL, _mutate(BASE_SDL, *edits))

    assert report.is_checked
    assert not report.is_compatible
    assert [issue.to_dict() for issue in report.breaking] == [
        {"change": change, "surface": GRAPHQL_SURFACE, **location}
    ]
    _assert_identifier_only(report.to_dict())


GRAPHQL_ADDITIVE_CASES = [
    ("type_added", [("type Audit {", "type Extra {\n  value: Int\n}\n\ntype Audit {")]),
    ("field_added", [("  score: Float\n", "  score: Float\n  note: String\n")]),
    ("argument_added", [("subject(id: ID!)", "subject(id: ID!, verbose: Boolean)")]),
    ("argument_added", [("subject(id: ID!)", "subject(id: ID!, page: Int! = 1)")]),
    ("input_field_added", [("  text: String!\n", "  text: String!\n  offset: Int\n")]),
    ("enum_value_added", [("  BETA\n", "  BETA\n  GAMMA\n")]),
    (
        "union_member_added",
        [
            ("= Patient | Device", "= Patient | Device | Sensor"),
            ("type Audit {", "type Sensor {\n  id: ID!\n}\n\ntype Audit {"),
        ],
    ),
    (
        "interface_added",
        [
            (
                "type Patient implements Node {",
                "type Patient implements Node & Named {",
            ),
            ("type Audit {", "interface Named {\n  label: String!\n}\n\ntype Audit {"),
        ],
    ),
    (
        "directive_added",
        [("interface Node {", "directive @trace on FIELD\n\ninterface Node {")],
    ),
    ("directive_location_added", [("| OBJECT", "| OBJECT | ENUM")]),
    (
        "directive_argument_added",
        [("(level: Int = 1)", "(level: Int = 1, tag: String)")],
    ),
]


@pytest.mark.parametrize(
    ("change", "edits"),
    GRAPHQL_ADDITIVE_CASES,
    ids=[f"{case[0]}-{index}" for index, case in enumerate(GRAPHQL_ADDITIVE_CASES)],
)
def test_graphql_additive_changes_pass(
    change: str, edits: list[tuple[str, str]]
) -> None:
    report = compare_graphql_schemas(BASE_SDL, _mutate(BASE_SDL, *edits))

    assert report.is_compatible
    assert report.breaking == ()
    assert change in _changes(report.added)


@pytest.mark.parametrize(
    ("change", "edits"),
    [
        ("field_nullability_tightened", [("  score: Float\n", "  score: Float!\n")]),
        (
            "argument_nullability_relaxed",
            [("input: SearchInput!", "input: SearchInput")],
        ),
        ("argument_default_changed", [("first: Int = 20", "first: Int = 50")]),
        (
            "input_field_default_changed",
            [('"SYNTHETIC_DEFAULT_SENTINEL"\n}', '"SYNTHETIC_OTHER_SENTINEL"\n}')],
        ),
        (
            "input_field_nullability_relaxed",
            [("  text: String!\n", "  text: String\n")],
        ),
    ],
)
def test_graphql_compatible_non_additive_changes_pass_without_values(
    change: str, edits: list[tuple[str, str]]
) -> None:
    report = compare_graphql_schemas(BASE_SDL, _mutate(BASE_SDL, *edits))

    assert report.is_compatible
    assert _changes(report.compatible) == {change}
    _assert_identifier_only(report.to_dict())


def test_graphql_identical_and_description_only_changes_have_no_findings() -> None:
    candidate = BASE_SDL.replace(
        "SYNTHETIC_DESCRIPTION_SENTINEL", "SYNTHETIC_OTHER_SENTINEL"
    )

    first = compare_graphql_schemas(BASE_SDL, candidate)
    second = compare_graphql_schemas(BASE_SDL, candidate)

    assert (first.breaking, first.added, first.compatible) == ((), (), ())
    assert first.to_dict() == second.to_dict()
    assert first.before_definitions == first.after_definitions == 8


def test_graphql_report_contains_only_schema_identifiers() -> None:
    candidate = _mutate(
        BASE_SDL,
        ("  score: Float\n", ""),
        ("first: Int = 20", "first: Int = 21"),
        ('"SYNTHETIC_DEFAULT_SENTINEL"\n}', '"SYNTHETIC_OTHER_SENTINEL"\n}'),
        ("  BETA\n", "  GAMMA\n"),
    )

    report = compare_graphql_schemas(BASE_SDL, candidate)
    payload = report.to_dict()

    _assert_identifier_only(payload)
    for bucket in ("added", "breaking", "compatible"):
        for issue in payload[bucket]:
            assert set(issue) <= SCHEMA_ISSUE_KEYS
            assert all(isinstance(value, str) for value in issue.values())


def test_graphql_invalid_sdl_is_rejected_without_echoing_source() -> None:
    with pytest.raises(APIContractError) as raised:
        compare_graphql_schemas(BASE_SDL, "type SYNTHETIC_DEFAULT_SENTINEL {")

    assert "SYNTHETIC_DEFAULT_SENTINEL" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_graphql_surface_skips_with_controlled_status_without_graphql_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "graphql", None)

    report = compare_graphql_schemas(BASE_SDL, BASE_SDL)

    assert report.status == "skipped"
    assert report.reason == "graphql_core_unavailable"
    assert report.to_dict()["status"] == "skipped"


def test_graphql_surface_skips_when_live_runtime_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = tmp_path / "baseline.graphql"
    baseline.write_text(BASE_SDL, encoding="utf-8")

    def unavailable() -> str:
        raise ImportError("synthetic missing runtime")

    monkeypatch.setattr(gate, "build_live_graphql_sdl", unavailable)
    report = check_graphql_compatibility(baseline)

    assert (report.status, report.reason) == ("skipped", "graphql_runtime_unavailable")


def test_live_graphql_schema_and_committed_sdl_pass_the_baseline() -> None:
    live = check_graphql_compatibility()
    snapshot = compare_graphql_schemas(
        DEFAULT_GRAPHQL_BASELINE_PATH.read_text(encoding="utf-8"),
        (ROOT / "docs" / "api" / "graphql-schema.graphql").read_text(encoding="utf-8"),
    )

    assert live.is_checked and live.is_compatible, [
        issue.describe() for issue in live.breaking
    ]
    assert snapshot.is_checked and snapshot.is_compatible


# ---------------------------------------------------------------------------
# Protobuf descriptors
# ---------------------------------------------------------------------------


FETCH_REQUEST = "synthetic.contract.v1.FetchRequest"
FETCH_RESPONSE = "synthetic.contract.v1.FetchResponse"
RECORD_KIND = "synthetic.contract.v1.RecordKind"
RECORD_SERVICE = "synthetic.contract.v1.RecordService"

PROTO_BREAKING_CASES = [
    (
        {"field_removed"},
        [("  double score = 2;\n", "")],
        {"type_name": FETCH_RESPONSE, "member": "score", "field_number": 2},
    ),
    (
        {"field_number_changed"},
        [("double score = 2;", "double score = 7;")],
        {"type_name": FETCH_RESPONSE, "member": "score", "field_number": 2},
    ),
    (
        {"field_renamed"},
        [("double score = 2;", "double rank = 2;")],
        {"type_name": FETCH_RESPONSE, "member": "score", "field_number": 2},
    ),
    (
        {"field_type_changed"},
        [("double score = 2;", "float score = 2;")],
        {"type_name": FETCH_RESPONSE, "member": "score", "field_number": 2},
    ),
    (
        {"field_type_changed"},
        [("  Nested nested = 3;", "  Unused nested = 3;")],
        {"type_name": FETCH_RESPONSE, "member": "nested", "field_number": 3},
    ),
    (
        {"field_cardinality_changed"},
        [("repeated string tags = 3;", "string tags = 3;")],
        {"type_name": FETCH_REQUEST, "member": "tags", "field_number": 3},
    ),
    (
        {"field_presence_changed"},
        [("optional int32 limit = 2;", "int32 limit = 2;")],
        {"type_name": FETCH_REQUEST, "member": "limit", "field_number": 2},
    ),
    (
        {"field_oneof_changed"},
        [
            (
                "    string cursor = 5;\n    int64 offset = 6;\n  }\n",
                "    string cursor = 5;\n  }\n  int64 offset = 6;\n",
            )
        ],
        {"type_name": FETCH_REQUEST, "member": "offset", "field_number": 6},
    ),
    (
        {"reserved_range_removed"},
        [("reserved 9, 11 to 13;", "reserved 9, 11 to 12;")],
        {"type_name": FETCH_REQUEST, "field_number": 13},
    ),
    (
        {"reserved_range_removed", "reserved_number_reused"},
        [
            ("reserved 9, 11 to 13;", "reserved 11 to 13;\n  string revived = 9;"),
        ],
        {"type_name": FETCH_REQUEST, "field_number": 9},
    ),
    (
        {"reserved_name_removed", "reserved_name_reused"},
        [
            (
                '  reserved "legacy_field";\n',
                "  string legacy_field = 20;\n",
            )
        ],
        {"type_name": FETCH_REQUEST, "member": "legacy_field"},
    ),
    (
        {"reserved_range_removed", "reserved_number_reused"},
        [("  reserved 5;\n", "  RECORD_KIND_REVIVED = 5;\n")],
        {"type_name": RECORD_KIND, "field_number": 5},
    ),
    (
        {"message_removed"},
        [("message Unused {\n  string value = 1;\n}\n", "")],
        {"type_name": "synthetic.contract.v1.Unused"},
    ),
    (
        {"enum_removed"},
        [("enum Priority {\n  PRIORITY_UNSPECIFIED = 0;\n}\n", "")],
        {"type_name": "synthetic.contract.v1.Priority"},
    ),
    (
        {"enum_value_removed"},
        [("  RECORD_KIND_BETA = 2;\n", "")],
        {"type_name": RECORD_KIND, "member": "RECORD_KIND_BETA", "field_number": 2},
    ),
    (
        {"enum_value_number_changed"},
        [("RECORD_KIND_BETA = 2;", "RECORD_KIND_BETA = 3;")],
        {"type_name": RECORD_KIND, "member": "RECORD_KIND_BETA", "field_number": 2},
    ),
    (
        {"service_removed"},
        [
            (
                "service AuditService {\n"
                "  rpc Ping(FetchRequest) returns (FetchResponse);\n}\n",
                "",
            )
        ],
        {"type_name": "synthetic.contract.v1.AuditService"},
    ),
    (
        {"rpc_removed"},
        [("  rpc Watch(FetchRequest) returns (stream FetchResponse);\n", "")],
        {"type_name": RECORD_SERVICE, "member": "Watch"},
    ),
    (
        {"rpc_request_type_changed"},
        [("rpc Fetch(FetchRequest)", "rpc Fetch(Unused)")],
        {"type_name": RECORD_SERVICE, "member": "Fetch"},
    ),
    (
        {"rpc_response_type_changed"},
        [
            (
                "rpc Fetch(FetchRequest) returns (FetchResponse);",
                "rpc Fetch(FetchRequest) returns (Unused);",
            )
        ],
        {"type_name": RECORD_SERVICE, "member": "Fetch"},
    ),
    (
        {"rpc_streaming_changed"},
        [("returns (stream FetchResponse)", "returns (FetchResponse)")],
        {"type_name": RECORD_SERVICE, "member": "Watch"},
    ),
]


@pytest.mark.usefixtures("proto_toolchain")
@pytest.mark.parametrize(
    ("changes", "edits", "location"),
    PROTO_BREAKING_CASES,
    ids=[
        f"{'+'.join(sorted(case[0]))}-{index}"
        for index, case in enumerate(PROTO_BREAKING_CASES)
    ],
)
def test_protobuf_breaking_classes_fail(
    tmp_path: Path,
    changes: set[str],
    edits: list[tuple[str, str]],
    location: dict[str, Any],
) -> None:
    baseline = _compile(tmp_path, BASE_PROTO, "baseline")
    candidate = _compile(tmp_path, _mutate(BASE_PROTO, *edits), "candidate")

    report = compare_proto_contracts(baseline, candidate)

    assert report.is_checked
    assert not report.is_compatible
    assert _changes(report.breaking) == changes
    for issue in report.breaking:
        payload = issue.to_dict()
        assert payload["surface"] == GRPC_SURFACE
        assert set(payload) <= SCHEMA_ISSUE_KEYS
        assert {key: payload[key] for key in location if key in payload} == {
            key: value for key, value in location.items() if key in payload
        }
    _assert_identifier_only(report.to_dict())


def _insert_before(anchor: str, text: str) -> tuple[str, str]:
    return (anchor, f"{text}{anchor}")


PROTO_ADDITIVE_CASES = [
    ("field_added", _insert_before("  message Nested {", "  string note = 4;\n")),
    (
        "message_added",
        _insert_before("message Unused {", "message Extra {\n  string v = 1;\n}\n"),
    ),
    (
        "enum_added",
        _insert_before("enum Priority {", "enum Tier {\n  TIER_UNSPECIFIED = 0;\n}\n"),
    ),
    (
        "enum_value_added",
        _insert_before("  reserved 5;\n", "  RECORD_KIND_GAMMA = 3;\n"),
    ),
    (
        "rpc_added",
        _insert_before(
            "  rpc Watch(", "  rpc Count(FetchRequest) returns (FetchResponse);\n"
        ),
    ),
    (
        "service_added",
        _insert_before(
            "service AuditService {",
            "service ExtraService {\n"
            "  rpc Echo(FetchRequest) returns (FetchResponse);\n}\n",
        ),
    ),
    (
        "reserved_range_added",
        _insert_before("  message Nested {", "  reserved 30 to 32;\n"),
    ),
    (
        "reserved_name_added",
        _insert_before("  message Nested {", '  reserved "old_name";\n'),
    ),
]


@pytest.mark.usefixtures("proto_toolchain")
@pytest.mark.parametrize(
    ("change", "edit"),
    PROTO_ADDITIVE_CASES,
    ids=[case[0] for case in PROTO_ADDITIVE_CASES],
)
def test_protobuf_additive_changes_pass(
    tmp_path: Path, change: str, edit: tuple[str, str]
) -> None:
    baseline = _compile(tmp_path, BASE_PROTO, "baseline")
    candidate = _compile(tmp_path, _mutate(BASE_PROTO, edit), "candidate")

    report = compare_proto_contracts(baseline, candidate)

    assert report.is_compatible
    assert _changes(report.added) == {change}


@pytest.mark.usefixtures("proto_toolchain")
def test_protobuf_identical_contract_is_deterministic_and_round_trips(
    tmp_path: Path,
) -> None:
    contract = _compile(tmp_path, BASE_PROTO)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(contract.to_json(), encoding="utf-8")

    loaded = load_proto_contract(baseline_path)
    report = compare_proto_contracts(loaded, contract)

    assert loaded == contract
    assert loaded.to_json() == contract.to_json()
    assert (report.breaking, report.added, report.compatible) == ((), (), ())
    assert report.before_definitions == report.after_definitions == 8
    _assert_identifier_only(json.loads(contract.to_json()))


def test_protobuf_descriptor_objects_normalize_reserved_range_ends() -> None:
    descriptor_pb2 = pytest.importorskip("google.protobuf.descriptor_pb2")
    file = descriptor_pb2.FileDescriptorProto(name="synthetic.proto", package="pkg")
    message = file.message_type.add(name="Record")
    message.field.add(
        name="value",
        number=1,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
    )
    message.reserved_range.add(start=5, end=8)
    enum = file.enum_type.add(name="Kind")
    enum.value.add(name="KIND_UNSPECIFIED", number=0)
    enum.reserved_range.add(start=5, end=8)
    descriptor_set = descriptor_pb2.FileDescriptorSet(file=[file])

    from_object = normalize_proto_descriptors(descriptor_set)
    from_bytes = normalize_proto_descriptors(descriptor_set.SerializeToString())

    assert from_object == from_bytes
    assert from_object.messages[0].reserved_numbers == ((5, 7),)
    assert from_object.enums[0].reserved_numbers == ((5, 8),)

    candidate = descriptor_pb2.FileDescriptorSet()
    candidate.CopyFrom(descriptor_set)
    candidate.file[0].message_type[0].field.add(
        name="revived",
        number=7,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_INT32,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
    )
    del candidate.file[0].message_type[0].reserved_range[:]

    report = compare_proto_contracts(descriptor_set, candidate)

    assert {issue.change for issue in report.breaking} == {
        "reserved_number_reused",
        "reserved_range_removed",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "surface": "grpc"},
        {"schema_version": 1, "surface": "rest"},
        {"schema_version": 1, "surface": "grpc", "unexpected": []},
        {
            "schema_version": 1,
            "surface": "grpc",
            "messages": [{"name": "pkg.Bad name", "fields": []}],
        },
        {
            "schema_version": 1,
            "surface": "grpc",
            "messages": [
                {
                    "name": "pkg.M",
                    "fields": [
                        {"name": "a", "number": 1, "type": "string"},
                        {"name": "b", "number": 1, "type": "string"},
                    ],
                }
            ],
        },
        {
            "schema_version": 1,
            "surface": "grpc",
            "messages": [
                {
                    "name": "pkg.M",
                    "fields": [{"name": "a", "number": 1, "type": "message"}],
                }
            ],
        },
    ],
)
def test_invalid_protobuf_baseline_is_rejected_without_echoing_values(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(APIContractError) as raised:
        load_proto_contract(path)

    assert "Bad name" not in str(raised.value)
    assert str(raised.value) == "The protobuf contract baseline is invalid"


@pytest.mark.usefixtures("proto_toolchain")
def test_protobuf_compile_failure_is_a_controlled_contract_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "broken.proto"
    path.write_text('syntax = "proto3";\nmessage SYNTHETIC_DEFAULT_SENTINEL {', "utf-8")

    with pytest.raises(APIContractError) as raised:
        compile_proto_contract(path)

    assert "SYNTHETIC_DEFAULT_SENTINEL" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)


def test_protobuf_surface_skips_with_controlled_status_without_grpc_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "grpc_tools", None)

    with pytest.raises(APISurfaceUnavailable) as raised:
        compile_proto_contract()
    report = check_grpc_compatibility()

    assert raised.value.reason == "grpc_tools_unavailable"
    assert (report.status, report.reason) == ("skipped", "grpc_tools_unavailable")
    assert report.is_compatible and not report.is_checked


@pytest.mark.usefixtures("proto_toolchain")
def test_live_proto_source_passes_the_committed_baseline() -> None:
    report = check_grpc_compatibility()

    assert report.is_checked
    assert report.is_compatible, [issue.describe() for issue in report.breaking]
    assert report.before_definitions >= 12


# ---------------------------------------------------------------------------
# Combined report, CLI and baseline writer
# ---------------------------------------------------------------------------


@pytest.fixture
def gate_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    contract = _rest_contract()
    contract_path = tmp_path / "openapi.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "build_live_contract",
        lambda *args, **kwargs: gate.normalize_contract(contract),
    )
    graphql_baseline = tmp_path / "baseline.graphql"
    graphql_baseline.write_text(BASE_SDL, encoding="utf-8")
    graphql_candidate = tmp_path / "candidate.graphql"
    graphql_candidate.write_text(BASE_SDL, encoding="utf-8")
    proto_candidate = tmp_path / "candidate.proto"
    proto_candidate.write_text(BASE_PROTO, encoding="utf-8")
    return {
        "contract": contract_path,
        "graphql_baseline": graphql_baseline,
        "graphql_candidate": graphql_candidate,
        "proto_baseline": tmp_path / "baseline.json",
        "proto_candidate": proto_candidate,
    }


def _cli(files: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--contract",
        str(files["contract"]),
        "--graphql-baseline",
        str(files["graphql_baseline"]),
        "--graphql-schema",
        str(files["graphql_candidate"]),
        "--proto-baseline",
        str(files["proto_baseline"]),
        "--proto",
        str(files["proto_candidate"]),
        *extra,
    ]


def test_rest_only_report_shape_is_unchanged() -> None:
    payload = compare_contracts(_rest_contract(), _rest_contract()).to_dict()

    assert set(payload) == {
        "schema_version",
        "summary",
        "added",
        "breaking",
        "compatible",
    }
    assert set(payload["summary"]) == {
        "before_operations",
        "after_operations",
        "added",
        "breaking",
    }


@pytest.mark.usefixtures("proto_toolchain")
def test_combined_report_separates_surfaces_and_fails_closed(
    gate_files: dict[str, Path],
) -> None:
    gate_files["proto_baseline"].write_text(
        _compile(gate_files["contract"].parent, BASE_PROTO, "baseline").to_json(),
        encoding="utf-8",
    )
    candidate_sdl = _mutate(
        BASE_SDL, ("  score: Float\n", ""), ("  BETA\n", "  BETA\n  GAMMA\n")
    )
    kwargs: dict[str, Any] = {
        "surfaces": SCHEMA_SURFACES,
        "graphql_baseline_path": gate_files["graphql_baseline"],
        "current_graphql_sdl": candidate_sdl,
        "proto_baseline_path": gate_files["proto_baseline"],
        "proto_path": gate_files["proto_candidate"],
    }

    report = check_api_compatibility(gate_files["contract"], **kwargs)
    payload = report.to_dict()

    assert not report.is_compatible
    assert report.is_complete
    assert payload["breaking"] == []
    assert payload["summary"]["total_breaking"] == 1
    assert payload["summary"]["total_added"] == 1
    assert payload["surfaces"]["graphql"]["breaking"] == [
        {
            "change": "field_removed",
            "member": "score",
            "surface": "graphql",
            "type_name": "Patient",
        }
    ]
    assert payload["surfaces"]["grpc"]["status"] == "checked"
    assert (
        report.to_json()
        == check_api_compatibility(gate_files["contract"], **kwargs).to_json()
    )
    with pytest.raises(APICompatibilityError) as raised:
        report.assert_compatible()
    assert "graphql field_removed: Patient.score" in str(raised.value)
    _assert_identifier_only(payload)


def test_skipped_surface_is_explicit_and_fails_closed_unless_allowed(
    gate_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_files["proto_baseline"].write_text(
        json.dumps({"schema_version": 1, "surface": "grpc"}), encoding="utf-8"
    )
    monkeypatch.setitem(sys.modules, "grpc_tools", None)

    report = check_api_compatibility(
        gate_files["contract"],
        surfaces=[GRPC_SURFACE],
        proto_baseline_path=gate_files["proto_baseline"],
        proto_path=gate_files["proto_candidate"],
    )

    assert report.is_compatible
    assert report.skipped_surfaces == (GRPC_SURFACE,)
    assert report.to_dict()["surfaces"]["grpc"] == {
        "status": "skipped",
        "reason": "grpc_tools_unavailable",
        "summary": {
            "before_definitions": 0,
            "after_definitions": 0,
            "added": 0,
            "breaking": 0,
        },
        "added": [],
        "breaking": [],
        "compatible": [],
    }
    with pytest.raises(APICompatibilityError) as raised:
        report.assert_compatible()
    assert "grpc skipped: grpc_tools_unavailable" in str(raised.value)
    assert report.assert_compatible(allow_skipped=True) is report

    assert main(_cli(gate_files)) == 3
    assert main(_cli(gate_files, "--allow-skipped")) == 0


def test_unknown_surface_is_rejected() -> None:
    with pytest.raises(APIContractError):
        check_api_compatibility(surfaces=["soap"], current_contract=_rest_contract())


@pytest.mark.usefixtures("proto_toolchain")
def test_cli_reports_schema_breaks_and_additions_as_identifier_only_json(
    gate_files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    gate_files["proto_baseline"].write_text(
        _compile(gate_files["contract"].parent, BASE_PROTO, "baseline").to_json(),
        encoding="utf-8",
    )
    gate_files["proto_candidate"].write_text(
        _mutate(BASE_PROTO, ("double score = 2;", "double score = 7;")),
        encoding="utf-8",
    )
    gate_files["graphql_candidate"].write_text(
        _mutate(BASE_SDL, ("  BETA\n", "  BETA\n  GAMMA\n")), encoding="utf-8"
    )

    assert main(_cli(gate_files, "--json")) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["surfaces"]["graphql"]["breaking"] == []
    assert payload["surfaces"]["graphql"]["added"][0]["change"] == "enum_value_added"
    assert payload["surfaces"]["grpc"]["breaking"] == [
        {
            "change": "field_number_changed",
            "field_number": 2,
            "member": "score",
            "surface": "grpc",
            "type_name": FETCH_RESPONSE,
        }
    ]
    _assert_identifier_only(payload)
    assert str(gate_files["contract"].parent) not in json.dumps(payload)

    assert main(_cli(gate_files)) == 1
    text = capsys.readouterr().out
    assert "gRPC contract: breaking changes found" in text
    assert f"grpc field_number_changed: {FETCH_RESPONSE}.score #2" in text

    assert main(_cli(gate_files, "--rest-only", "--json")) == 0
    assert "surfaces" not in json.loads(capsys.readouterr().out)


def test_cli_invalid_schema_baseline_is_a_controlled_error(
    gate_files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    gate_files["proto_baseline"].write_text(
        '{"schema_version": 1, "surface": "SYNTHETIC_DEFAULT_SENTINEL"}',
        encoding="utf-8",
    )

    assert main(_cli(gate_files, "--json")) == 2
    output = capsys.readouterr().out

    assert json.loads(output) == {"schema_version": 1, "error": "invalid_contract"}
    assert "SYNTHETIC_DEFAULT_SENTINEL" not in output


@pytest.mark.usefixtures("proto_toolchain")
def test_baseline_writer_ratchets_and_refuses_unacknowledged_breaks(
    gate_files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    graphql_baseline = gate_files["graphql_baseline"]
    graphql_baseline.unlink()
    proto_baseline = gate_files["proto_baseline"]
    options: dict[str, Any] = {
        "graphql_baseline_path": graphql_baseline,
        "proto_baseline_path": proto_baseline,
        "proto_path": gate_files["proto_candidate"],
    }

    assert write_schema_baselines(current_graphql_sdl=BASE_SDL, **options) == ()
    assert graphql_baseline.read_text(encoding="utf-8") == BASE_SDL
    assert load_proto_contract(proto_baseline) == compile_proto_contract(
        gate_files["proto_candidate"]
    )

    broken_sdl = _mutate(BASE_SDL, ("  score: Float\n", ""))
    with pytest.raises(APICompatibilityError):
        write_schema_baselines(current_graphql_sdl=broken_sdl, **options)
    assert graphql_baseline.read_text(encoding="utf-8") == BASE_SDL

    gate_files["graphql_candidate"].write_text(broken_sdl, encoding="utf-8")
    assert main(_cli(gate_files, "--write-baselines")) == 1
    assert "breaking changes found" in capsys.readouterr().out
    assert graphql_baseline.read_text(encoding="utf-8") == BASE_SDL

    assert main(_cli(gate_files, "--write-baselines", "--allow-breaking")) == 0
    assert "1 accepted breaking" in capsys.readouterr().out
    assert graphql_baseline.read_text(encoding="utf-8") == broken_sdl
    assert main(_cli(gate_files)) == 0


@pytest.mark.usefixtures("proto_toolchain")
def test_committed_schema_baselines_match_the_gate_writer_format() -> None:
    baseline = gate.DEFAULT_PROTO_BASELINE_PATH.read_text(encoding="utf-8")

    sdl = DEFAULT_GRAPHQL_BASELINE_PATH.read_text(encoding="utf-8")

    assert load_proto_contract(gate.DEFAULT_PROTO_BASELINE_PATH).to_json() == baseline
    assert sdl == f"{sdl.rstrip()}\n"
    assert compare_graphql_schemas(sdl, sdl).breaking == ()
