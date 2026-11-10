"""Content-free validation of declared dense adapter update metadata.

The caller supplies trusted expected metadata separately from the update. This
module never reads tensors or verifies their contents, lineage, or clipping.
"""

from __future__ import annotations

import json
import re
from dataclasses import InitVar, dataclass
from typing import Any, Final

FEDERATED_UPDATE_METADATA_SCHEMA_VERSION = (
    "openmed.training.federated_update_metadata.v1"
)

_MAX_ELEMENTS: Final = (1 << 63) - 1
_MAX_DIMENSION: Final = (1 << 31) - 1
_MAX_PARAMETERS: Final = 1024
_MAX_RANK: Final = 8
_MAX_JSON_BYTES: Final = 1024 * 1024
_DTYPES: Final = frozenset({"float16", "bfloat16", "float32", "float64"})
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NAME: Final = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*\Z"
)
_PARAMETER_FIELDS: Final = frozenset({"name", "shape", "dtype"})
_UPDATE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "model_digest",
        "adapter_format",
        "parameters",
        "total_elements",
        "update_digest",
        "clipped",
    }
)


class FederatedUpdateMetadataError(ValueError):
    """Raised for invalid metadata without including submitted values."""


@dataclass(frozen=True, slots=True)
class FederatedParameterMetadata:
    """A bounded parameter name, shape, and floating-point dtype.

    Args:
        name: Dotted ASCII parameter name from the coordinator's allowlist.
        shape: One to eight positive integer dimensions; scalar and empty
            tensors are not supported by this dense adapter format.
        dtype: One of float16, bfloat16, float32, or float64.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not 1 <= len(self.name) <= 256
            or _NAME.fullmatch(self.name) is None
        ):
            raise FederatedUpdateMetadataError("invalid parameter name")
        _shape_elements(self.shape)
        if type(self.dtype) is not str or self.dtype not in _DTYPES:
            raise FederatedUpdateMetadataError("unsupported parameter dtype")

    @property
    def element_count(self) -> int:
        """Return the exact checked product of the declared dimensions."""
        return _shape_elements(self.shape)

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping containing only parameter metadata."""
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True, slots=True)
class FederatedUpdatePolicy:
    """Trusted coordinator expectations supplied independently of an update.

    Args:
        model_digest: Expected lowercase SHA-256 reference for the base model.
        parameters: Complete allowlist of expected parameter names, shapes,
            and dtypes. Every entry must be present in a dense update.
        adapter_format: Supported representation, currently only ``dense``.
        require_clipped: Reject an update that does not declare clipping.
        max_total_elements: Positive declared element budget, at most 2**63 - 1.
    """

    model_digest: str
    parameters: tuple[FederatedParameterMetadata, ...]
    adapter_format: str = "dense"
    require_clipped: bool = True
    max_total_elements: int = 100_000_000

    def __post_init__(self) -> None:
        _require_digest(self.model_digest)
        _require_format(self.adapter_format)
        if type(self.require_clipped) is not bool:
            raise FederatedUpdateMetadataError("invalid clipping policy")
        _require_count(self.max_total_elements)
        ordered = _ordered_parameters(self.parameters)
        _total_elements(ordered, self.max_total_elements)
        object.__setattr__(self, "parameters", ordered)


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedUpdateMetadata:
    """An immutable envelope checked against a caller-owned policy.

    Args:
        model_digest: Declared base-model SHA-256 reference.
        adapter_format: Declared representation, currently only ``dense``.
        parameters: Complete declared parameter metadata.
        total_elements: Claimed sum of the products of parameter dimensions.
        update_digest: Declared update SHA-256 reference; not a content check.
        clipped: Whether the submitter declares the update has been clipped.
        policy: Trusted coordinator policy, required even for construction.
            It is used for validation and is not stored or serialized.
        schema_version: Exact supported metadata schema identifier.
    """

    model_digest: str
    adapter_format: str
    parameters: tuple[FederatedParameterMetadata, ...]
    total_elements: int
    update_digest: str
    clipped: bool
    policy: InitVar[FederatedUpdatePolicy]
    schema_version: str = FEDERATED_UPDATE_METADATA_SCHEMA_VERSION

    def __post_init__(self, policy: FederatedUpdatePolicy) -> None:
        if type(policy) is not FederatedUpdatePolicy:
            raise FederatedUpdateMetadataError("invalid update policy")
        if (
            type(self.schema_version) is not str
            or self.schema_version != FEDERATED_UPDATE_METADATA_SCHEMA_VERSION
        ):
            raise FederatedUpdateMetadataError("unsupported update metadata schema")
        _require_digest(self.model_digest)
        _require_digest(self.update_digest)
        _require_format(self.adapter_format)
        if self.model_digest != policy.model_digest:
            raise FederatedUpdateMetadataError("model digest does not match policy")
        if self.adapter_format != policy.adapter_format:
            raise FederatedUpdateMetadataError("adapter format does not match policy")
        if type(self.clipped) is not bool:
            raise FederatedUpdateMetadataError("invalid clipping status")
        if policy.require_clipped and not self.clipped:
            raise FederatedUpdateMetadataError("update must declare clipping")
        _require_count(self.total_elements)
        ordered = _ordered_parameters(self.parameters)
        total = _total_elements(ordered, policy.max_total_elements)
        if self.total_elements != total:
            raise FederatedUpdateMetadataError(
                "total element count does not match shapes"
            )
        if ordered != policy.parameters:
            raise FederatedUpdateMetadataError("parameters do not match policy")
        object.__setattr__(self, "parameters", ordered)

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh versioned mapping with parameters ordered by name."""
        return {
            "schema_version": self.schema_version,
            "model_digest": self.model_digest,
            "adapter_format": self.adapter_format,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "total_elements": self.total_elements,
            "update_digest": self.update_digest,
            "clipped": self.clipped,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with sorted keys and a trailing newline."""
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    @classmethod
    def from_dict(
        cls, payload: object, *, policy: FederatedUpdatePolicy
    ) -> FederatedUpdateMetadata:
        """Validate a JSON-style dictionary against separate trusted metadata.

        Args:
            payload: Built-in dictionary with exactly the envelope fields.
                Parameter records must be dictionaries with name, shape, and
                dtype only. Shapes must be JSON-style lists of integers.
            policy: Coordinator expectations; never obtain this from payload.

        Returns:
            Validated immutable metadata in canonical parameter order.

        Raises:
            FederatedUpdateMetadataError: If the schema or policy is violated.
        """
        payload = _require_fields(payload, _UPDATE_FIELDS)
        parameters = payload["parameters"]
        if type(parameters) is not list or not 1 <= len(parameters) <= _MAX_PARAMETERS:
            raise FederatedUpdateMetadataError("invalid parameter collection")
        parsed = []
        for parameter in parameters:
            parameter = _require_fields(parameter, _PARAMETER_FIELDS)
            shape = parameter["shape"]
            if type(shape) is not list or not 1 <= len(shape) <= _MAX_RANK:
                raise FederatedUpdateMetadataError("invalid parameter shape")
            parsed.append(
                FederatedParameterMetadata(
                    name=parameter["name"], shape=tuple(shape), dtype=parameter["dtype"]
                )
            )
        return cls(
            model_digest=payload["model_digest"],
            adapter_format=payload["adapter_format"],
            parameters=tuple(parsed),
            total_elements=payload["total_elements"],
            update_digest=payload["update_digest"],
            clipped=payload["clipped"],
            schema_version=payload["schema_version"],
            policy=policy,
        )

    @classmethod
    def from_json(
        cls, payload: str, *, policy: FederatedUpdatePolicy
    ) -> FederatedUpdateMetadata:
        """Parse bounded JSON, rejecting duplicate keys and non-integer numbers.

        Args:
            payload: UTF-8 encodable JSON text, at most one MiB.
            policy: Independently supplied coordinator expectations.

        Returns:
            Validated immutable update metadata.

        Raises:
            FederatedUpdateMetadataError: On invalid JSON, limits, or policy.
                Errors never include submitted keys, values, or parser excerpts.
        """
        if type(payload) is not str or len(payload) > _MAX_JSON_BYTES:
            raise FederatedUpdateMetadataError("invalid update metadata JSON")
        try:
            if len(payload.encode("utf-8")) > _MAX_JSON_BYTES:
                raise FederatedUpdateMetadataError("invalid update metadata JSON")
            decoded = json.loads(
                payload,
                object_pairs_hook=_strict_object,
                parse_int=_parse_integer,
                parse_float=_reject_number,
                parse_constant=_reject_number,
            )
        except (ValueError, RecursionError):
            raise FederatedUpdateMetadataError("invalid update metadata JSON") from None
        return cls.from_dict(decoded, policy=policy)


def _require_fields(payload: object, expected: frozenset[str]) -> dict[str, Any]:
    if (
        type(payload) is not dict
        or len(payload) != len(expected)
        or any(type(key) is not str for key in payload)
        or payload.keys() != expected
    ):
        raise FederatedUpdateMetadataError("invalid metadata fields")
    return payload


def _require_digest(value: object) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise FederatedUpdateMetadataError("invalid SHA-256 digest reference")


def _require_format(value: object) -> None:
    if type(value) is not str or value != "dense":
        raise FederatedUpdateMetadataError("unsupported adapter format")


def _require_count(value: object) -> None:
    if type(value) is not int or not 1 <= value <= _MAX_ELEMENTS:
        raise FederatedUpdateMetadataError("invalid element count")


def _shape_elements(shape: object) -> int:
    if type(shape) is not tuple or not 1 <= len(shape) <= _MAX_RANK:
        raise FederatedUpdateMetadataError("invalid parameter shape")
    product = 1
    for dimension in shape:
        if type(dimension) is not int or not 1 <= dimension <= _MAX_DIMENSION:
            raise FederatedUpdateMetadataError("invalid shape dimension")
        if product > _MAX_ELEMENTS // dimension:
            raise FederatedUpdateMetadataError("shape element count exceeds limit")
        product *= dimension
    return product


def _ordered_parameters(
    value: object,
) -> tuple[FederatedParameterMetadata, ...]:
    if type(value) is not tuple or not 1 <= len(value) <= _MAX_PARAMETERS:
        raise FederatedUpdateMetadataError("invalid parameter collection")
    if any(type(parameter) is not FederatedParameterMetadata for parameter in value):
        raise FederatedUpdateMetadataError("invalid parameter metadata")
    if len({parameter.name for parameter in value}) != len(value):
        raise FederatedUpdateMetadataError("duplicate parameter names")
    return tuple(sorted(value, key=lambda parameter: parameter.name))


def _total_elements(
    parameters: tuple[FederatedParameterMetadata, ...], limit: int
) -> int:
    total = 0
    for parameter in parameters:
        count = parameter.element_count
        if count > limit - total:
            raise FederatedUpdateMetadataError("total element count exceeds limit")
        total += count
    return total


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FederatedUpdateMetadataError("duplicate metadata fields")
        result[key] = value
    return result


def _parse_integer(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise FederatedUpdateMetadataError("invalid metadata integer")
    parsed = int(value)
    if not -_MAX_ELEMENTS <= parsed <= _MAX_ELEMENTS:
        raise FederatedUpdateMetadataError("invalid metadata integer")
    return parsed


def _reject_number(value: str) -> None:
    raise FederatedUpdateMetadataError("non-integer metadata number")


__all__ = [
    "FEDERATED_UPDATE_METADATA_SCHEMA_VERSION",
    "FederatedParameterMetadata",
    "FederatedUpdateMetadata",
    "FederatedUpdateMetadataError",
    "FederatedUpdatePolicy",
]
