"""Synthetic signed round/return transfer, corruption and independent trust controls."""

import hashlib
import json
import struct
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openmed.training.federated.airgap_bundle import (
    build_return_bundle,
    build_round_bundle,
    transfer_receipt,
    verify_return_bundle,
    verify_round_bundle,
)
from openmed.training.federated_metrics import (
    FederatedMetricKind,
    FederatedPrivacyMechanism,
    build_federated_metric_envelope,
)
from openmed.training.federated_schedule import FederatedRoundSchedule
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def components():
    # Published synthetic signing material is never a production trust anchor.
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    model = "sha256:" + "1" * 64
    policy = FederatedUpdatePolicy(
        model, (FederatedParameterMetadata("adapter.weight", (2,), "float32"),)
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    schedule = FederatedRoundSchedule(*(start + timedelta(minutes=n) for n in range(5)))
    code = {
        "runner": b"def run_synthetic(): return 1\n",
        "self_test": b"assert 1 + 1 == 2\n",
    }
    kwargs = dict(
        round_reference="sha256:" + "2" * 64,
        code=code,
        synthetic_self_tests=["self_test"],
        model_references=[model],
        schedule=schedule,
        update_policy=policy,
        signing_key=key,
    )
    verify = dict(
        public_key=key.public_key(),
        expected_code_digests={name: digest(raw) for name, raw in code.items()},
        expected_model_references=[model],
        expected_update_policy=policy,
        expected_round_reference=kwargs["round_reference"],
        expected_synthetic_self_tests=["self_test"],
    )
    return kwargs, verify


def test_signed_round_is_deterministic_and_deeply_immutable():
    kwargs, verify = components()
    first = build_round_bundle(**kwargs)
    assert first == build_round_bundle(**kwargs)
    body = verify_round_bundle(first, **verify)
    assert body["code"]["runner"] == kwargs["code"]["runner"]
    assert body["synthetic_self_tests"] == ("self_test",)
    with pytest.raises(TypeError):
        body["code"]["runner"] = b"changed"
    assert set(transfer_receipt(first)) == {"schema_version", "byte_count", "sha256"}


@pytest.mark.parametrize(
    "field",
    [
        "expected_code_digests",
        "expected_model_references",
        "expected_round_reference",
        "expected_synthetic_self_tests",
    ],
)
def test_independent_allowlists_cannot_be_supplied_by_bundle(field):
    kwargs, verify = components()
    bundle = build_round_bundle(**kwargs)
    if field == "expected_code_digests":
        verify[field] = {
            "runner": "sha256:" + "0" * 64,
            "self_test": "sha256:" + "0" * 64,
        }
    elif field == "expected_model_references":
        verify[field] = ["sha256:" + "0" * 64]
    elif field == "expected_round_reference":
        verify[field] = "sha256:" + "0" * 64
    else:
        verify[field] = ["runner"]
    with pytest.raises(ValueError, match="^airgap_bundle_invalid_contract$"):
        verify_round_bundle(bundle, **verify)


def test_wrong_signing_key_and_unsigned_tampering_rejected():
    kwargs, verify = components()
    bundle = build_round_bundle(**kwargs)
    verify["public_key"] = Ed25519PrivateKey.from_private_bytes(
        bytes(reversed(range(32)))
    ).public_key()
    with pytest.raises(ValueError):
        verify_round_bundle(bundle, **verify)
    kwargs, verify = components()
    mutated = bundle.replace(b"self_test", b"cache_dir", 1)
    with pytest.raises(ValueError):
        verify_round_bundle(mutated, **verify)


def test_signed_unknown_fields_and_omitted_tests_still_rejected():
    kwargs, verify = components()
    body = json.loads(build_round_bundle(**kwargs))
    body["body"]["raw_records"] = ["SYNTHETIC PRIVATE SENTINEL"]
    unsigned = {name: body[name] for name in ("schema_version", "kind", "body")}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    body["signature"] = (
        kwargs["signing_key"].sign(b"openmed.airgap.bundle.v1\0" + encoded).hex()
    )
    bundle = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError) as caught:
        verify_round_bundle(bundle, **verify)
    assert caught.value.__context__ is None
    assert "PRIVATE" not in str(caught.value)


def return_components(*, value=0.25, participants=8):
    kwargs, verify = components()
    raw = struct.pack("<ff", value, 0.5)
    policy = kwargs["update_policy"]
    metadata = FederatedUpdateMetadata(
        model_digest=policy.model_digest,
        adapter_format="dense",
        parameters=policy.parameters,
        total_elements=2,
        update_digest=digest(raw),
        clipped=True,
        policy=policy,
    )
    metric = build_federated_metric_envelope(
        metric_id="synthetic.rate",
        metric_kind=FederatedMetricKind.RATE,
        aggregate_value=0.5,
        clipping_lower_bound=0,
        clipping_upper_bound=1,
        privacy_mechanism=FederatedPrivacyMechanism.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=participants,
    )
    round_digest = transfer_receipt(build_round_bundle(**kwargs))["sha256"]
    ret = dict(
        round_bundle_digest=round_digest,
        metadata=metadata,
        update_content=raw,
        update_policy=policy,
        aggregate_metrics=[metric],
        signing_key=kwargs["signing_key"],
    )
    back = dict(
        public_key=verify["public_key"],
        expected_round_bundle_digest=round_digest,
        expected_update_policy=policy,
        allowed_metric_ids=["synthetic.rate"],
    )
    return ret, back


@pytest.mark.parametrize("participants", [1, 8])
def test_real_dense_update_and_existing_aggregate_suppression_roundtrip(participants):
    ret, back = return_components(participants=participants)
    first = build_return_bundle(**ret)
    assert first == build_return_bundle(**ret)
    body = verify_return_bundle(first, **back)
    assert body["update_content"] == ret["update_content"]
    assert body["aggregate_metrics"][0]["aggregate_value"] == (
        None if participants == 1 else 0.5
    )
    assert b"client_id" not in first and b"raw_records" not in first


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_actual_update_values_rejected(value):
    ret, _ = return_components(value=value)
    with pytest.raises(ValueError):
        build_return_bundle(**ret)


def test_exact_tensor_content_length_and_digest_are_verified():
    ret, _ = return_components()
    ret["update_content"] += b"unexpected tail"
    with pytest.raises(ValueError):
        build_return_bundle(**ret)


def test_return_bound_to_round_policy_metric_allowlist_and_direction():
    ret, back = return_components()
    bundle = build_return_bundle(**ret)
    for change in (
        {"expected_round_bundle_digest": "sha256:" + "0" * 64},
        {"allowed_metric_ids": []},
    ):
        with pytest.raises(ValueError):
            verify_return_bundle(bundle, **(back | change))
    kwargs, verify = components()
    with pytest.raises(ValueError):
        verify_round_bundle(bundle, **verify)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"body":1,"body":2}',
        b"[" * 2000 + b"]" * 2000,
        b"x" * 1_048_577,
        b"not json SYNTHETIC PRIVATE",
    ],
)
def test_malformed_transfer_is_bounded_and_value_free(raw):
    _, verify = components()
    with pytest.raises(ValueError, match="^airgap_bundle_invalid_contract$") as caught:
        verify_round_bundle(raw, **verify)
    assert caught.value.__context__ is None


def test_code_bounds_and_names_reject_paths_and_oversize_content():
    kwargs, _ = components()
    for code in (
        {"../private": b"x"},
        {"runner": b"x" * 262145},
        {1: b"x", "runner": b"x"},
    ):
        with pytest.raises(ValueError):
            build_round_bundle(**(kwargs | {"code": code}))
