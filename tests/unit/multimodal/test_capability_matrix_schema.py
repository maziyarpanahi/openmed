"""Schema interoperability, negative payloads and Python contract drift controls."""

import json
from dataclasses import fields

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry

from openmed.multimodal.capability_matrix import (
    COMPATIBILITY_REASON_OPTIONS,
    MAX_CAPABILITY_MATRIX_ENTRIES,
    CompatibilityEntry,
    ModalityCapability,
    ModalityProfile,
    ProviderCapabilityEnvelope,
    capability_matrix,
    render_capability_matrix_json,
)
from openmed.multimodal.capability_matrix_schema import (
    build_capability_matrix_schema,
    render_capability_matrix_schema,
)


def record(**changes):
    return (
        CompatibilityEntry("local_audio", "1.2.3", "mono", "exact", ()).to_dict()
        | changes
    )


@pytest.fixture
def validator():
    def no_network(uri):
        pytest.fail("Schema attempted external resolution")

    schema = build_capability_matrix_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=Registry(retrieve=no_network))


def test_real_cross_product_all_outcomes_validates_locally(validator):
    providers = [
        ProviderCapabilityEnvelope("missing", "1.0.0"),
        ProviderCapabilityEnvelope("none", "1.0.0", ()),
        ProviderCapabilityEnvelope(
            "exact", "1.0.0", (ModalityCapability("audio", ("mono",)),)
        ),
        ProviderCapabilityEnvelope(
            "transform", "1.0.0", (ModalityCapability("audio", (), ("mono",)),)
        ),
    ]
    profiles = [
        ModalityProfile("mono", "audio", ("mono",)),
        ModalityProfile("other", "audio", ("unsupported",)),
    ]
    rows = capability_matrix(providers, profiles)
    payload = json.loads(render_capability_matrix_json(rows))
    validator.validate(payload)
    assert {row["outcome"] for row in payload} == set(COMPATIBILITY_REASON_OPTIONS)
    assert render_capability_matrix_json(rows) == render_capability_matrix_json(
        reversed(rows)
    )
    validator.validate([])


def test_schema_fields_bounds_and_enums_track_python_records():
    schema = build_capability_matrix_schema()
    cell = schema["$defs"]["cell"]
    names = {field.name for field in fields(CompatibilityEntry)}
    assert names == set(record()) == set(cell["properties"]) == set(cell["required"])
    assert schema["maxItems"] == MAX_CAPABILITY_MATRIX_ENTRIES
    assert set(cell["properties"]["outcome"]["enum"]) == set(
        COMPATIBILITY_REASON_OPTIONS
    )
    assert set(cell["properties"]["reasons"]["items"]["enum"]) == {
        reason
        for options in COMPATIBILITY_REASON_OPTIONS.values()
        for option in options
        for reason in option
    }


def test_every_outcome_reason_pair_matches_runtime(validator):
    reasons = [
        (),
        ("capability_missing",),
        ("modality_unsupported",),
        ("feature_unsupported",),
        ("declared_transform_required",),
        ("unknown",),
        ("capability_missing", "feature_unsupported"),
    ]
    for outcome in [*COMPATIBILITY_REASON_OPTIONS, "unknown"]:
        for reason in reasons:
            try:
                entry = CompatibilityEntry("local", "1.0.0", "profile", outcome, reason)
            except ValueError:
                assert not validator.is_valid(
                    [record(outcome=outcome, reasons=list(reason))]
                )
            else:
                validator.validate([entry.to_dict()])


@pytest.mark.parametrize("field", ["provider_id", "profile_id", "capability_version"])
@pytest.mark.parametrize(
    "value",
    [
        "https://example.invalid/private?token=synthetic",
        "/private/models/weights.bin",
        "C:\\private\\weights.bin",
        "synthetic patient name",
        "normal\n",
        "a\x00",
        "",
        None,
        1,
        True,
        {},
        [],
        "x" * 65,
        "1.2.3\n",
    ],
)
def test_endpoint_path_and_malformed_fields_fail_without_network(
    validator, field, value
):
    assert not validator.is_valid([record(**{field: value})])
    with pytest.raises(ValueError):
        CompatibilityEntry(**record(**{field: value}))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"records": [record()]},
        "private",
        [None],
        [True],
        [[record()]],
        [record(endpoint="https://example.invalid")],
        [record(model_path="/private/model")],
        [record(outcome="qualified")],
        [record(reasons="capability_missing")],
        [record(reasons=["unknown"])],
        [record(reasons=[None])],
        [record(reasons=[{}])],
        [record(reasons=[], schema_version=1)],
    ],
)
def test_closed_payload_rejects_unrecognized_structure(validator, payload):
    assert not validator.is_valid(payload)


def test_required_fields_and_identical_duplicates_are_rejected(validator):
    for name in record():
        incomplete = record()
        del incomplete[name]
        assert not validator.is_valid([incomplete])
    assert not validator.is_valid([record(), record()])


def test_identifier_and_version_boundaries_match_runtime(validator):
    for code in ["a", "a" * 64, "a0_-", "A", "a" * 65, "a\n", "a\r", "é", "1a", "a.b"]:
        for version in [
            "0.0.0",
            "000001.999999.0",
            "1000000.0.0",
            "1.2",
            "1.2.3+build",
            "1.2.3\n",
            "１.2.3",
        ]:
            data = record(provider_id=code, capability_version=version)
            try:
                CompatibilityEntry(**data)
            except ValueError:
                assert not validator.is_valid([data])
            else:
                validator.validate([data])


def test_pair_uniqueness_stays_a_runtime_semantic_check(validator):
    rows = [
        CompatibilityEntry("local", "1.0.0", "mono", "exact", ()),
        CompatibilityEntry(
            "local", "1.0.0", "mono", "unsupported", ("feature_unsupported",)
        ),
    ]
    # Standard uniqueItems cannot express uniqueness of a subset of fields.
    validator.validate([row.to_dict() for row in rows])
    with pytest.raises(ValueError, match="duplicate_entry"):
        render_capability_matrix_json(rows)


def test_export_is_byte_stable_independent_and_uses_only_local_refs():
    first = render_capability_matrix_schema()
    changed = build_capability_matrix_schema()
    changed["$defs"]["cell"]["properties"]["outcome"]["enum"].append("tampered")
    assert render_capability_matrix_schema() == first
    assert json.loads(first) == build_capability_matrix_schema()

    def walk(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].startswith("#/$defs/")
                assert value["$ref"].removeprefix("#/$defs/") in changed["$defs"]
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(build_capability_matrix_schema())
