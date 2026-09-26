"""Metadata-only schema identity checks for federated update envelopes."""

from __future__ import annotations

import hashlib
import hmac
import re
from unittest.mock import patch

import pytest

from openmed.training.federated_schema_fingerprint import (
    fingerprint_update_schema,
    same_update_schema,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)


def _metadata(
    *,
    model: str = "a",
    update: str = "b",
    clipped: bool = True,
    name: str = "adapter.lora_A.weight",
    shape: tuple[int, ...] = (2, 3),
    dtype: str = "float32",
    reverse: bool = False,
) -> FederatedUpdateMetadata:
    parameters = (
        FederatedParameterMetadata(name, shape, dtype),
        FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
    )
    if reverse:
        parameters = tuple(reversed(parameters))
    policy = FederatedUpdatePolicy(
        model_digest="sha256:" + model * 64,
        parameters=parameters,
        require_clipped=clipped,
    )
    total = sum(parameter.element_count for parameter in parameters)
    return FederatedUpdateMetadata(
        model_digest=policy.model_digest,
        adapter_format=policy.adapter_format,
        parameters=parameters,
        total_elements=total,
        update_digest="sha256:" + update * 64,
        clipped=clipped,
        policy=policy,
    )


def test_golden_vector_is_stable_and_schema_only() -> None:
    metadata = _metadata()
    fingerprint = fingerprint_update_schema(metadata)

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint)
    assert fingerprint == (
        "sha256:33ce7495b3e804919db8b2c94f5b073055a2b0513b9e8fc06d940eee22906470"
    )

    real_sha256 = hashlib.sha256
    with patch(
        "openmed.training.federated_schema_fingerprint.hashlib.sha256",
        wraps=real_sha256,
    ) as digest:
        assert fingerprint_update_schema(metadata) == fingerprint
    hashed = digest.call_args.args[0]
    assert hashed.startswith(b"openmed.training.federated_schema_fingerprint.v1\0")
    assert b'"adapter_format":"dense"' in hashed
    assert b'"model_digest":"sha256:' in hashed
    assert b'"parameters":[' in hashed
    assert b"update_digest" not in hashed
    assert b"clipped" not in hashed
    for forbidden in (b"tensor", b"gradient", b"example", b"client_id", b"path"):
        assert forbidden not in hashed


def test_parameter_order_and_update_values_do_not_change_schema_identity() -> None:
    baseline = _metadata()
    for equivalent in (
        _metadata(reverse=True),
        _metadata(update="c"),
        _metadata(clipped=False),
        _metadata(update="d", clipped=False, reverse=True),
    ):
        assert fingerprint_update_schema(equivalent) == fingerprint_update_schema(
            baseline
        )
        assert same_update_schema(baseline, equivalent)


@pytest.mark.parametrize(
    "change",
    [
        {"name": "adapter.lora_C.weight"},
        {"shape": (3, 3)},
        {"dtype": "float16"},
        {"model": "c"},
    ],
)
def test_schema_changes_change_fingerprint(change: dict[str, object]) -> None:
    baseline = _metadata()
    changed = _metadata(**change)

    assert fingerprint_update_schema(changed) != fingerprint_update_schema(baseline)
    assert not same_update_schema(baseline, changed)


def test_comparison_uses_constant_time_primitive() -> None:
    first = _metadata()
    second = _metadata(model="c")

    with patch(
        "openmed.training.federated_schema_fingerprint.hmac.compare_digest",
        wraps=hmac.compare_digest,
    ) as compare:
        assert not same_update_schema(first, second)
    compare.assert_called_once_with(
        fingerprint_update_schema(first), fingerprint_update_schema(second)
    )


def test_only_validated_updates_are_accepted() -> None:
    with pytest.raises(TypeError, match="FederatedUpdateMetadata"):
        fingerprint_update_schema({"model_digest": "sha256:" + "a" * 64})

    metadata = _metadata()
    payload = metadata.to_dict()
    payload["adapter_format"] = "sparse"
    with pytest.raises(FederatedUpdateMetadataError, match="format"):
        FederatedUpdateMetadata.from_dict(
            payload,
            policy=FederatedUpdatePolicy(
                model_digest=metadata.model_digest,
                parameters=metadata.parameters,
            ),
        )
