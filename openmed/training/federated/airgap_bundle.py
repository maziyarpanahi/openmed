"""Signed, bounded offline round transfers with independently supplied allowlists."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from functools import wraps
from types import MappingProxyType

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from openmed.training.federated_metrics import FederatedMetricEnvelope
from openmed.training.federated_schedule import FederatedRoundSchedule
from openmed.training.federated_update_metadata import (
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

MAX_BUNDLE_BYTES = 1_048_576
MAX_CONTENT_BYTES = 262_144
_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DOMAIN = b"openmed.airgap.bundle.v1\0"


def _error() -> None:
    raise ValueError("airgap_bundle_invalid_contract")


def _contract_errors(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except Exception:
            failed = True
        else:
            failed = False
        if failed:
            _error()
        return result

    return guarded


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _code(value: object) -> None:
    if type(value) is not str or not _CODE.fullmatch(value):
        _error()


def _reference(value: object) -> None:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        _error()


def _encode(body: object) -> bytes:
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _policy(policy: FederatedUpdatePolicy) -> dict[str, object]:
    if not isinstance(policy, FederatedUpdatePolicy):
        _error()
    return {
        "model_digest": policy.model_digest,
        "adapter_format": policy.adapter_format,
        "parameters": [p.to_dict() for p in policy.parameters],
        "require_clipped": policy.require_clipped,
        "max_total_elements": policy.max_total_elements,
    }


def _closed(value: object, names: set[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != names:
        _error()


def _signed(kind: str, body: dict[str, object], key: Ed25519PrivateKey) -> bytes:
    if not isinstance(key, Ed25519PrivateKey):
        _error()
    payload = {"schema_version": 1, "kind": kind, "body": body}
    signature = key.sign(_DOMAIN + _encode(payload)).hex()
    encoded = _encode(payload | {"signature": signature})
    if len(encoded) > MAX_BUNDLE_BYTES:
        _error()
    return encoded


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            _error()
        result[name] = value
    return result


def _read(bundle: bytes, kind: str, key: Ed25519PublicKey) -> dict[str, object]:
    if (
        type(bundle) is not bytes
        or not 1 <= len(bundle) <= MAX_BUNDLE_BYTES
        or not isinstance(key, Ed25519PublicKey)
    ):
        _error()
    try:
        parsed = json.loads(
            bundle, object_pairs_hook=_pairs, parse_constant=lambda value: _error()
        )
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        _error()
    _closed(parsed, {"schema_version", "kind", "body", "signature"})
    if (
        type(parsed["schema_version"]) is not int
        or parsed["schema_version"] != 1
        or parsed["kind"] != kind
    ):
        _error()
    signature = parsed["signature"]
    if type(signature) is not str or not re.fullmatch(r"[0-9a-f]{128}", signature):
        _error()
    payload = {name: parsed[name] for name in ("schema_version", "kind", "body")}
    try:
        key.verify(bytes.fromhex(signature), _DOMAIN + _encode(payload))
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        _error()
    if _encode(parsed) != bundle:
        _error()
    return parsed["body"]


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({name: _freeze(item) for name, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _allowed_codes(values: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(values, Mapping) or not 1 <= len(values) <= 32:
        _error()
    for code, digest in values.items():
        _code(code)
        _reference(digest)
    return dict(values)


def _models(values: Sequence[str]) -> list[str]:
    if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 32:
        _error()
    for value in values:
        _reference(value)
    if len(set(values)) != len(values):
        _error()
    return sorted(values)


@_contract_errors
def build_round_bundle(
    *,
    round_reference: str,
    code: Mapping[str, bytes],
    synthetic_self_tests: Sequence[str],
    model_references: Sequence[str],
    schedule: FederatedRoundSchedule,
    update_policy: FederatedUpdatePolicy,
    signing_key: Ed25519PrivateKey,
) -> bytes:
    """Package explicitly approved code, model references, policy and synthetic tests.

    No filesystem traversal, extraction, downloads or code execution is performed.
    The signer must review source code and self-tests before providing them. Model
    weights and local datasets are never included by this transfer contract.
    """
    _reference(round_reference)
    if (
        not isinstance(code, Mapping)
        or not 1 <= len(code) <= 32
        or not isinstance(schedule, FederatedRoundSchedule)
    ):
        _error()
    models = _models(model_references)
    policy = _policy(update_policy)
    if update_policy.model_digest not in models:
        _error()
    if (
        not isinstance(synthetic_self_tests, (tuple, list))
        or not 1 <= len(synthetic_self_tests) <= 32
    ):
        _error()
    for name in synthetic_self_tests:
        _code(name)
    if len(set(synthetic_self_tests)) != len(synthetic_self_tests) or not set(
        synthetic_self_tests
    ) <= set(code):
        _error()
    records = {}
    total = 0
    for name in code:
        _code(name)
    for name, content in sorted(code.items()):
        _code(name)
        if type(content) is not bytes or not 1 <= len(content) <= MAX_CONTENT_BYTES:
            _error()
        total += len(content)
        if total > MAX_CONTENT_BYTES:
            _error()
        records[name] = {"sha256": _digest(content), "content_hex": content.hex()}
    body = {
        "round_reference": round_reference,
        "code": records,
        "synthetic_self_tests": sorted(synthetic_self_tests),
        "model_references": models,
        "schedule": schedule.to_dict(),
        "update_policy": policy,
    }
    return _signed("round", body, signing_key)


@_contract_errors
def verify_round_bundle(
    bundle: bytes,
    *,
    public_key: Ed25519PublicKey,
    expected_code_digests: Mapping[str, str],
    expected_model_references: Sequence[str],
    expected_update_policy: FederatedUpdatePolicy,
    expected_round_reference: str,
    expected_synthetic_self_tests: Sequence[str],
) -> Mapping[str, object]:
    """Authenticate and independently allowlist every transferred round component.

    Trust anchors and expected policy are caller-owned, never accepted from the
    bundle itself. Return deeply immutable validated data; never execute self-tests.
    """
    body = _read(bundle, "round", public_key)
    _closed(
        body,
        {
            "round_reference",
            "code",
            "synthetic_self_tests",
            "model_references",
            "schedule",
            "update_policy",
        },
    )
    _reference(expected_round_reference)
    if body["round_reference"] != expected_round_reference or _encode(
        body["update_policy"]
    ) != _encode(_policy(expected_update_policy)):
        _error()
    if body["model_references"] != _models(expected_model_references):
        _error()
    if expected_update_policy.model_digest not in body["model_references"]:
        _error()
    allowed = _allowed_codes(expected_code_digests)
    _closed(body["code"], set(allowed))
    code = {}
    total = 0
    for name, record in body["code"].items():
        _closed(record, {"sha256", "content_hex"})
        raw = record["content_hex"]
        if (
            type(raw) is not str
            or not 2 <= len(raw) <= MAX_CONTENT_BYTES * 2
            or len(raw) % 2
            or not re.fullmatch(r"[0-9a-f]+", raw)
        ):
            _error()
        content = bytes.fromhex(raw)
        total += len(content)
        if (
            total > MAX_CONTENT_BYTES
            or _digest(content) != record["sha256"]
            or record["sha256"] != allowed[name]
        ):
            _error()
        code[name] = content
    tests = body["synthetic_self_tests"]
    if (
        not isinstance(tests, list)
        or not 1 <= len(tests) <= 32
        or any(type(name) is not str or name not in allowed for name in tests)
        or len(set(tests)) != len(tests)
    ):
        _error()
    if (
        not isinstance(expected_synthetic_self_tests, (tuple, list))
        or not 1 <= len(expected_synthetic_self_tests) <= 32
    ):
        _error()
    for name in expected_synthetic_self_tests:
        _code(name)
    if len(set(expected_synthetic_self_tests)) != len(
        expected_synthetic_self_tests
    ) or tests != sorted(expected_synthetic_self_tests):
        _error()
    FederatedRoundSchedule.from_dict(body["schedule"])
    return _freeze(body | {"code": code})


def _dense_bytes(
    metadata: FederatedUpdateMetadata, content: bytes, policy: FederatedUpdatePolicy
) -> None:
    FederatedUpdateMetadata.from_dict(metadata.to_dict(), policy=policy)
    if (
        type(content) is not bytes
        or not 1 <= len(content) <= MAX_CONTENT_BYTES
        or metadata.update_digest != _digest(content)
    ):
        _error()
    offset = 0
    for parameter in metadata.parameters:
        if parameter.dtype not in {"float32", "float64"}:
            _error()
        width, fmt = (4, "<f") if parameter.dtype == "float32" else (8, "<d")
        end = offset + parameter.element_count * width
        if end > len(content) or any(
            not math.isfinite(value[0])
            for value in struct.iter_unpack(fmt, content[offset:end])
        ):
            _error()
        offset = end
    if offset != len(content):
        _error()


@_contract_errors
def build_return_bundle(
    *,
    round_bundle_digest: str,
    metadata: FederatedUpdateMetadata,
    update_content: bytes,
    update_policy: FederatedUpdatePolicy,
    aggregate_metrics: Sequence[FederatedMetricEnvelope],
    signing_key: Ed25519PrivateKey,
) -> bytes:
    """Export only one admitted dense update and existing aggregate-safe envelopes.

    This is a typed transport allowlist, not proof of clipping, differential
    privacy or absence of memorization in the actual update values.
    """
    _reference(round_bundle_digest)
    if not isinstance(metadata, FederatedUpdateMetadata):
        _error()
    _dense_bytes(metadata, update_content, update_policy)
    if (
        not isinstance(aggregate_metrics, (tuple, list))
        or len(aggregate_metrics) > 32
        or any(
            not isinstance(metric, FederatedMetricEnvelope)
            for metric in aggregate_metrics
        )
    ):
        _error()
    if len({metric.metric_id for metric in aggregate_metrics}) != len(
        aggregate_metrics
    ):
        _error()
    metrics = [
        FederatedMetricEnvelope.from_dict(metric.to_dict()).to_dict()
        for metric in aggregate_metrics
    ]
    body = {
        "round_bundle_digest": round_bundle_digest,
        "update_metadata": metadata.to_dict(),
        "update_content_hex": update_content.hex(),
        "aggregate_metrics": sorted(metrics, key=lambda row: row["metric_id"]),
    }
    return _signed("return", body, signing_key)


@_contract_errors
def verify_return_bundle(
    bundle: bytes,
    *,
    public_key: Ed25519PublicKey,
    expected_round_bundle_digest: str,
    expected_update_policy: FederatedUpdatePolicy,
    allowed_metric_ids: Sequence[str],
) -> Mapping[str, object]:
    """Verify exact round binding, finite typed dense data and safe aggregate schema."""
    body = _read(bundle, "return", public_key)
    _closed(
        body,
        {
            "round_bundle_digest",
            "update_metadata",
            "update_content_hex",
            "aggregate_metrics",
        },
    )
    _reference(expected_round_bundle_digest)
    if body["round_bundle_digest"] != expected_round_bundle_digest:
        _error()
    if (
        not isinstance(allowed_metric_ids, (tuple, list))
        or len(allowed_metric_ids) > 32
    ):
        _error()
    for value in allowed_metric_ids:
        _code(value)
    if len(set(allowed_metric_ids)) != len(allowed_metric_ids):
        _error()
    metadata = FederatedUpdateMetadata.from_dict(
        body["update_metadata"], policy=expected_update_policy
    )
    raw = body["update_content_hex"]
    if (
        type(raw) is not str
        or not 2 <= len(raw) <= MAX_CONTENT_BYTES * 2
        or len(raw) % 2
        or not re.fullmatch(r"[0-9a-f]+", raw)
    ):
        _error()
    content = bytes.fromhex(raw)
    _dense_bytes(metadata, content, expected_update_policy)
    metrics = body["aggregate_metrics"]
    if not isinstance(metrics, list) or len(metrics) > 32:
        _error()
    parsed = [FederatedMetricEnvelope.from_dict(metric) for metric in metrics]
    if len({metric.metric_id for metric in parsed}) != len(parsed) or any(
        metric.metric_id not in allowed_metric_ids for metric in parsed
    ):
        _error()
    return _freeze(body | {"update_content": content})


def transfer_receipt(bundle: bytes) -> dict[str, object]:
    """Return byte count/digest only; receipt is not signature or privacy approval."""
    if type(bundle) is not bytes or not 1 <= len(bundle) <= MAX_BUNDLE_BYTES:
        _error()
    return {"schema_version": 1, "byte_count": len(bundle), "sha256": _digest(bundle)}
