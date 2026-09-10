"""Synthetic, offline contracts for federated update metadata validation."""

from __future__ import annotations

import json
import traceback
from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import pytest

from openmed.training import federated_update_metadata as metadata_module
from openmed.training.federated_update_metadata import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)

MODEL = "sha256:" + "a" * 64
UPDATE = "sha256:" + "b" * 64


@pytest.fixture
def policy():
    return FederatedUpdatePolicy(
        model_digest=MODEL,
        parameters=(
            FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
            FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
        ),
        max_total_elements=14,
    )


@pytest.fixture
def payload():
    return {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": MODEL,
        "adapter_format": "dense",
        "parameters": [
            {"name": "adapter.lora_A.weight", "shape": [2, 3], "dtype": "float32"},
            {"name": "adapter.lora_B.weight", "shape": [4, 2], "dtype": "float32"},
        ],
        "total_elements": 14,
        "update_digest": UPDATE,
        "clipped": True,
    }


def test_dense_metadata_round_trips_with_canonical_order(payload, policy):
    reference = FederatedUpdateMetadata.from_dict(payload, policy=policy)
    assert reference.total_elements == 14
    assert [p.element_count for p in reference.parameters] == [6, 8]
    assert reference.to_dict() == payload
    for order in permutations(payload["parameters"]):
        changed = dict(reversed(list(payload.items())))
        changed["parameters"] = [dict(reversed(list(p.items()))) for p in order]
        parsed = FederatedUpdateMetadata.from_json(json.dumps(changed), policy=policy)
        assert parsed == reference
        assert parsed.to_json() == reference.to_json()
        assert (
            FederatedUpdateMetadata.from_json(parsed.to_json(), policy=policy) == parsed
        )
    assert reference.to_json().endswith("\n")


def test_input_and_serialized_copies_cannot_change_validated_metadata(payload, policy):
    result = FederatedUpdateMetadata.from_dict(payload, policy=policy)
    canonical = result.to_json()
    payload["parameters"][0]["shape"][0] = 99
    exported = result.to_dict()
    exported["parameters"][1]["name"] = "forbidden"
    assert result.to_json() == canonical
    with pytest.raises(FrozenInstanceError):
        result.total_elements = 1
    with pytest.raises(FrozenInstanceError):
        result.parameters[0].shape = (1,)
    assert not hasattr(result, "policy")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_supported_dtypes_require_matching_policy(payload, dtype):
    for p in payload["parameters"]:
        p["dtype"] = dtype
    policy = FederatedUpdatePolicy(
        MODEL,
        tuple(
            FederatedParameterMetadata(p["name"], tuple(p["shape"]), dtype)
            for p in payload["parameters"]
        ),
    )
    assert (
        FederatedUpdateMetadata.from_dict(payload, policy=policy).total_elements == 14
    )


@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "bool", "complex64", "fp32", "FLOAT32", "", None, [], 32]
)
def test_unsupported_dtypes_are_rejected(payload, policy, dtype):
    payload["parameters"][0]["dtype"] = dtype
    with pytest.raises(FederatedUpdateMetadataError, match="dtype"):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


@pytest.mark.parametrize(
    "change", ["unknown", "missing", "duplicate", "shape", "dtype"]
)
def test_parameter_contract_is_authoritative(payload, policy, change):
    if change == "unknown":
        payload["parameters"][0]["name"] = "base_model.weight"
    elif change == "missing":
        payload["parameters"].pop()
        payload["total_elements"] = 6
    elif change == "duplicate":
        payload["parameters"][1]["name"] = payload["parameters"][0]["name"]
    elif change == "shape":
        # Same element count is insufficient: dimension order must match too.
        payload["parameters"][0]["shape"] = [3, 2]
    else:
        payload["parameters"][0]["dtype"] = "float16"
    with pytest.raises(FederatedUpdateMetadataError):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


@pytest.mark.parametrize(
    "shape",
    [
        [],
        [0],
        [-1],
        [True, 6],
        [2.0, 3],
        ["2", 3],
        [[2], 3],
        [None],
        [1] * 9,
        [1 << 31],
        "2,3",
        {"0": 2},
        (2, 3),
        None,
    ],
)
def test_shapes_fail_closed(payload, policy, shape):
    payload["parameters"][0]["shape"] = shape
    with pytest.raises(FederatedUpdateMetadataError):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


@pytest.mark.parametrize("total", [13, 15, 0, -1, True, 14.0, "14", None, 1 << 63])
def test_total_elements_are_exact_integers(payload, policy, total):
    payload["total_elements"] = total
    with pytest.raises(FederatedUpdateMetadataError):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


def test_shape_multiplication_overflow_is_rejected_before_materialization():
    with pytest.raises(FederatedUpdateMetadataError, match="shape element count"):
        FederatedParameterMetadata("adapter.weight", ((1 << 31) - 1,) * 3, "float32")


def test_sum_overflow_and_policy_budget_are_checked_without_tensors(payload, policy):
    # Each product fits int64, but the three-parameter sum does not.
    large = tuple(
        FederatedParameterMetadata(
            f"adapter.{i}.weight", ((1 << 31) - 1,) * 2, "float32"
        )
        for i in range(3)
    )
    with pytest.raises(FederatedUpdateMetadataError, match="total element count"):
        FederatedUpdatePolicy(MODEL, large, max_total_elements=(1 << 63) - 1)
    payload["parameters"][0]["shape"] = [3, 3]
    payload["total_elements"] = 17
    with pytest.raises(
        FederatedUpdateMetadataError, match="total element count exceeds"
    ):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)
    with pytest.raises(
        FederatedUpdateMetadataError, match="total element count exceeds"
    ):
        replace(policy, max_total_elements=13)


def test_large_valid_counts_remain_exact_beyond_float_precision():
    parameter = FederatedParameterMetadata(
        "adapter.weight", ((1 << 27) - 1, (1 << 27) - 1), "float64"
    )
    total = ((1 << 27) - 1) ** 2
    policy = FederatedUpdatePolicy(MODEL, (parameter,), max_total_elements=total)
    result = FederatedUpdateMetadata(
        model_digest=MODEL,
        adapter_format="dense",
        parameters=(parameter,),
        total_elements=total,
        update_digest=UPDATE,
        clipped=True,
        policy=policy,
    )
    assert total > (1 << 53)
    assert (
        FederatedUpdateMetadata.from_json(
            result.to_json(), policy=policy
        ).total_elements
        == total
    )
    with pytest.raises(FederatedUpdateMetadataError, match="does not match shapes"):
        replace(result, total_elements=total - 1, policy=policy)


@pytest.mark.parametrize("field", ["model_digest", "update_digest"])
@pytest.mark.parametrize(
    "digest",
    [
        "a" * 64,
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
        "sha256:" + "g" * 64,
        MODEL + "\n",
        " " + MODEL,
        "sha512:" + "a" * 64,
        "https://example.invalid/model",
        "",
        None,
        [0],
    ],
)
def test_malformed_digests_are_rejected(payload, policy, field, digest):
    payload[field] = digest
    with pytest.raises(FederatedUpdateMetadataError, match="digest"):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


def test_well_formed_model_digest_must_match_coordinator_policy(payload, policy):
    payload["model_digest"] = UPDATE
    with pytest.raises(
        FederatedUpdateMetadataError, match="model digest does not match"
    ):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


def test_clipping_is_required_by_default_and_policy_controls_exception(payload, policy):
    payload["clipped"] = False
    with pytest.raises(FederatedUpdateMetadataError, match="declare clipping"):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)
    optional = replace(policy, require_clipped=False)
    result = FederatedUpdateMetadata.from_dict(payload, policy=optional)
    assert result.clipped is False
    assert result.to_dict()["clipped"] is False


@pytest.mark.parametrize("clipped", [1, 0, "true", "false", None, [], {}])
def test_clipping_status_is_a_strict_boolean(payload, policy, clipped):
    payload["clipped"] = clipped
    with pytest.raises(FederatedUpdateMetadataError, match="clipping status"):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("adapter_format", "sparse"),
        ("adapter_format", "safetensors"),
        ("adapter_format", None),
        ("schema_version", "v2"),
        ("schema_version", 1),
        ("schema_version", FEDERATED_UPDATE_METADATA_SCHEMA_VERSION + "\n"),
    ],
)
def test_unknown_formats_and_versions_are_rejected(payload, policy, field, value):
    payload[field] = value
    with pytest.raises(FederatedUpdateMetadataError):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensors", [[1, 2]]),
        ("values", [1, 2]),
        ("gradients", [0.1]),
        ("examples", ["synthetic private marker"]),
        ("client_id", "synthetic_client"),
        ("site_name", "synthetic_site"),
        ("path", "/synthetic/private"),
        ("endpoint", "https://example.invalid"),
        ("local_metrics", {"loss": 1}),
        ("allowed_parameters", []),
        ("require_clipped", False),
        ("policy", {}),
        ("notes", "synthetic private marker"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_forbidden_fields_are_rejected_at_every_record_level(
    payload, policy, field, value, nested
):
    target = payload["parameters"][0] if nested else payload
    target[field] = value
    for candidate in (payload, json.dumps(payload)):
        parse = (
            FederatedUpdateMetadata.from_json
            if isinstance(candidate, str)
            else FederatedUpdateMetadata.from_dict
        )
        with pytest.raises(FederatedUpdateMetadataError):
            parse(candidate, policy=policy)


@pytest.mark.parametrize("nested", [False, True])
def test_every_schema_field_is_required(payload, policy, nested):
    original = payload["parameters"][0] if nested else payload
    for field in original:
        candidate = json.loads(json.dumps(payload))
        target = candidate["parameters"][0] if nested else candidate
        del target[field]
        with pytest.raises(
            FederatedUpdateMetadataError, match="invalid metadata fields"
        ):
            FederatedUpdateMetadata.from_dict(candidate, policy=policy)


@pytest.mark.parametrize("location", ["top", "parameter"])
def test_duplicate_json_keys_are_rejected_even_when_values_agree(
    payload, policy, location
):
    encoded = json.dumps(payload)
    if location == "top":
        encoded = encoded.replace('"clipped": true', '"clipped": true, "clipped": true')
    else:
        encoded = encoded.replace(
            '"dtype": "float32"', '"dtype": "float32", "dtype": "float32"', 1
        )
    with pytest.raises(FederatedUpdateMetadataError, match="JSON"):
        FederatedUpdateMetadata.from_json(encoded, policy=policy)


@pytest.mark.parametrize(
    "number", ["NaN", "Infinity", "-Infinity", "14.0", "14e0", "1" * 1000, str(1 << 63)]
)
def test_json_rejects_non_integer_and_oversized_number_tokens(payload, policy, number):
    encoded = json.dumps(payload).replace(
        '"total_elements": 14', f'"total_elements": {number}'
    )
    with pytest.raises(FederatedUpdateMetadataError, match="JSON"):
        FederatedUpdateMetadata.from_json(encoded, policy=policy)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "{",
        "null",
        "[]",
        "true",
        "\ud800",
        b"{}",
        None,
        pytest.param(" " * (1024 * 1024 + 1), id="oversized-json"),
        "[" * 2000 + "]" * 2000,
    ],
)
def test_invalid_or_unbounded_json_fails_with_safe_errors(policy, raw):
    with pytest.raises(FederatedUpdateMetadataError):
        FederatedUpdateMetadata.from_json(raw, policy=policy)


def test_json_limit_counts_utf8_bytes(policy):
    encoded = '{"notes":"' + "\u2603" * 350000 + '"}'
    assert len(encoded) < 1024 * 1024 < len(encoded.encode("utf-8"))
    with pytest.raises(FederatedUpdateMetadataError, match="JSON"):
        FederatedUpdateMetadata.from_json(encoded, policy=policy)


@pytest.mark.parametrize("parameters", [[], {}, "private", [1, 2], [None], [True]])
def test_parameter_collection_must_contain_metadata_records(
    payload, policy, parameters
):
    payload["parameters"] = parameters
    with pytest.raises(FederatedUpdateMetadataError):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


@pytest.mark.parametrize(
    "name",
    [
        "/tmp/private",
        "C:\\private",
        "https://example.invalid",
        "client@example.invalid",
        "site name",
        "adapter.weight\n",
        "",
        "a" * 257,
        None,
    ],
)
def test_parameter_names_cannot_contain_paths_or_free_text(name):
    with pytest.raises(FederatedUpdateMetadataError, match="parameter name"):
        FederatedParameterMetadata(name, (1,), "float32")


def test_parameter_count_limit_is_enforced(payload, policy):
    parameters = tuple(
        FederatedParameterMetadata(f"adapter.{i}.weight", (1,), "float16")
        for i in range(1024)
    )
    limit_policy = FederatedUpdatePolicy(MODEL, parameters)
    result = FederatedUpdateMetadata(
        model_digest=MODEL,
        adapter_format="dense",
        parameters=parameters,
        total_elements=1024,
        update_digest=UPDATE,
        clipped=True,
        policy=limit_policy,
    )
    assert (
        FederatedUpdateMetadata.from_json(result.to_json(), policy=limit_policy)
        == result
    )
    with pytest.raises(FederatedUpdateMetadataError, match="parameter collection"):
        replace(
            limit_policy,
            parameters=parameters
            + (FederatedParameterMetadata("extra", (1,), "float16"),),
        )
    payload["parameters"] = [payload["parameters"][0]] * 1025
    with pytest.raises(FederatedUpdateMetadataError, match="parameter collection"):
        FederatedUpdateMetadata.from_dict(payload, policy=policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_total_elements", True),
        ("max_total_elements", 0),
        ("max_total_elements", 1 << 63),
        ("max_total_elements", 14.0),
        ("require_clipped", 1),
        ("model_digest", "invalid"),
        ("adapter_format", "sparse"),
        ("parameters", ()),
        ("parameters", []),
        ("parameters", ({"name": "anything"},)),
    ],
)
def test_policy_direct_construction_validates_all_fields(policy, field, value):
    with pytest.raises(FederatedUpdateMetadataError):
        replace(policy, **{field: value})
    with pytest.raises(FederatedUpdateMetadataError):
        replace(policy, parameters=(policy.parameters[0], policy.parameters[0]))


def test_direct_construction_cannot_bypass_policy_checks(payload, policy):
    valid = FederatedUpdateMetadata.from_dict(payload, policy=policy)
    for changes in (
        {"total_elements": 99},
        {"clipped": False},
        {"model_digest": UPDATE},
        {"schema_version": "bad"},
        {"parameters": policy.parameters[:1], "total_elements": 6},
    ):
        with pytest.raises(FederatedUpdateMetadataError):
            replace(valid, policy=policy, **changes)
    with pytest.raises(FederatedUpdateMetadataError, match="policy"):
        FederatedUpdateMetadata.from_dict(payload, policy=None)
    with pytest.raises(TypeError):
        FederatedUpdateMetadata(
            model_digest=MODEL,
            adapter_format="dense",
            parameters=policy.parameters,
            total_elements=14,
            update_digest=UPDATE,
            clipped=True,
        )


def test_errors_never_echo_rejected_identifiers_values_or_parser_excerpts(
    payload, policy, caplog, capsys
):
    marker = "SYNTHETIC_PRIVATE_SENTINEL_3010"
    for raw in (
        '{"' + marker + '":',
        '{"values":["' + marker + '"]}',
        json.dumps({**payload, "model_digest": marker}),
    ):
        with pytest.raises(FederatedUpdateMetadataError) as exc:
            FederatedUpdateMetadata.from_json(raw, policy=policy)
        assert marker not in str(exc.value)
        assert marker not in "".join(
            traceback.format_exception_only(type(exc.value), exc.value)
        )
        assert exc.value.__cause__ is None
    assert caplog.text == ""
    assert capsys.readouterr() == ("", "")


def test_lazy_training_exports_resolve():
    import openmed.training as training

    for name in metadata_module.__all__:
        assert name in training.__all__
        assert getattr(training, name) is getattr(metadata_module, name)
