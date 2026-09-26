"""Content-free identity for validated federated update schemas."""

from __future__ import annotations

import hashlib
import hmac
import json

from .federated_update_metadata import FederatedUpdateMetadata

_DOMAIN = b"openmed.training.federated_schema_fingerprint.v1\0"


def fingerprint_update_schema(metadata: FederatedUpdateMetadata) -> str:
    """Hash the validated model and parameter schema without update values.

    The update's declared content digest and clipping status describe an
    individual update, not its schema, and are deliberately excluded.
    """

    if type(metadata) is not FederatedUpdateMetadata:
        raise TypeError("metadata must be FederatedUpdateMetadata")
    schema = {
        "schema_version": metadata.schema_version,
        "model_digest": metadata.model_digest,
        "adapter_format": metadata.adapter_format,
        "parameters": [parameter.to_dict() for parameter in metadata.parameters],
    }
    canonical = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(_DOMAIN + canonical).hexdigest()


def same_update_schema(
    first: FederatedUpdateMetadata, second: FederatedUpdateMetadata
) -> bool:
    """Compare two validated update schemas with a constant-time primitive."""

    return hmac.compare_digest(
        fingerprint_update_schema(first), fingerprint_update_schema(second)
    )


__all__ = ["fingerprint_update_schema", "same_update_schema"]
