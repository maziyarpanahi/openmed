"""Deterministic, content-free clipping of private adapter update deltas.

The coordinator supplies trusted global and per-layer norm bounds separately
from the update. This module applies the clipping arithmetic locally with
pure-Python floating point: it performs no network call, records no logs, and
never echoes submitted numbers, layer names, or shapes in errors or reports.

Clipping is a precondition for a bounded sensitivity argument, not a privacy
guarantee by itself. The caller remains responsible for the noise mechanism,
the participant threshold, and the adjacency relation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

UPDATE_CLIPPING_SCHEMA_VERSION = "openmed.training.federated.update_clipping.v1"

MAX_CLIPPING_LAYERS: Final = 1024
MAX_CLIPPING_ELEMENTS_PER_LAYER: Final = 1 << 24
MAX_CLIPPING_TOTAL_ELEMENTS: Final = 1 << 26
MAX_CLIPPING_NORM_BOUND: Final = 1.0e12
MAX_CLIPPING_VALUE: Final = 1.0e12

CLIPPING_REASON_CODES: Final = frozenset(
    {
        "within_bound",
        "zero_norm",
        "scaled_to_global_bound",
        "scaled_to_layer_bound",
    }
)

_FINGERPRINT_DOMAIN: Final = b"openmed.training.federated.update_clipping.v1\0"
_MAX_JSON_BYTES: Final = 1024 * 1024
# Two correction passes keep the scaled vector inside its bound after the
# rounding the scaling itself introduces; the slack below only tolerates the
# final double-precision rounding of that comparison.
_NORM_ROUNDING_SLACK: Final = 1.0 + 1.0e-12
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NAME: Final = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*\Z"
)
_POLICY_FIELDS: Final = frozenset(
    {"schema_version", "global_norm_bound", "per_layer_bounds"}
)
_BOUND_ENTRY_FIELDS: Final = frozenset({"layer", "norm_bound"})


class FederatedUpdateClippingError(ValueError):
    """Raised for invalid clipping input without including submitted values."""


@dataclass(frozen=True, slots=True)
class FederatedClippingPolicy:
    """Trusted global and per-layer L2 norm bounds.

    Args:
        global_norm_bound: Positive finite L2 bound applied to every layer
            that has no per-layer override.
        per_layer_bounds: Pairs of dotted layer name and positive finite L2
            bound that override the global bound for that layer. Entries are
            canonicalized into layer-name order and must be unique.
    """

    global_norm_bound: float
    per_layer_bounds: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "global_norm_bound", _require_norm_bound(self.global_norm_bound)
        )
        object.__setattr__(
            self, "per_layer_bounds", _require_bound_entries(self.per_layer_bounds)
        )

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Layer names that carry an explicit per-layer bound."""
        return tuple(name for name, _ in self.per_layer_bounds)

    def bound_for(self, layer_name: str) -> float:
        """Return the per-layer override, or the global bound.

        Args:
            layer_name: Dotted layer name from the coordinator's allowlist.

        Returns:
            The positive L2 norm bound that applies to the layer.

        Raises:
            FederatedUpdateClippingError: If the name is not a valid dotted
                ASCII identifier. The message never includes the name.
        """
        _require_layer_name(layer_name)
        for name, bound in self.per_layer_bounds:
            if name == layer_name:
                return bound
        return self.global_norm_bound

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready policy document with bounds and layer names."""
        return {
            "schema_version": UPDATE_CLIPPING_SCHEMA_VERSION,
            "global_norm_bound": self.global_norm_bound,
            "per_layer_bounds": [
                {"layer": name, "norm_bound": bound}
                for name, bound in self.per_layer_bounds
            ],
        }

    def to_json(self) -> str:
        """Return canonical policy JSON text, newline terminated."""
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    @classmethod
    def from_dict(cls, payload: object) -> FederatedClippingPolicy:
        """Validate a policy document and return the frozen policy.

        Args:
            payload: Mapping with exactly ``schema_version``,
                ``global_norm_bound``, and ``per_layer_bounds``.

        Returns:
            The validated policy with canonical per-layer ordering.

        Raises:
            FederatedUpdateClippingError: If fields, bounds, or layer names
                are malformed. Messages never include submitted values.
        """
        fields = _require_fields(
            payload, _POLICY_FIELDS, "invalid clipping policy fields"
        )
        if fields["schema_version"] != UPDATE_CLIPPING_SCHEMA_VERSION:
            raise FederatedUpdateClippingError("unsupported clipping policy version")
        entries = fields["per_layer_bounds"]
        if type(entries) is not list or len(entries) > MAX_CLIPPING_LAYERS:
            raise FederatedUpdateClippingError("invalid clipping policy fields")
        pairs: list[tuple[str, float]] = []
        for entry in entries:
            bounds = _require_fields(
                entry, _BOUND_ENTRY_FIELDS, "invalid clipping policy fields"
            )
            pairs.append((bounds["layer"], bounds["norm_bound"]))
        return cls(
            global_norm_bound=fields["global_norm_bound"],
            per_layer_bounds=tuple(pairs),
        )

    @classmethod
    def from_json(cls, payload: str) -> FederatedClippingPolicy:
        """Parse bounded JSON, rejecting duplicate keys and non-finite numbers.

        Args:
            payload: UTF-8 encodable JSON text, at most one MiB.

        Returns:
            The validated policy.

        Raises:
            FederatedUpdateClippingError: On invalid JSON or policy content.
                Errors never include submitted keys, values, or parser
                excerpts.
        """
        if type(payload) is not str:
            raise FederatedUpdateClippingError("invalid clipping policy JSON")
        try:
            if len(payload.encode("utf-8")) > _MAX_JSON_BYTES:
                raise FederatedUpdateClippingError("invalid clipping policy JSON")
            decoded = json.loads(
                payload,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (ValueError, RecursionError, OverflowError):
            raise FederatedUpdateClippingError("invalid clipping policy JSON") from None
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class FederatedLayerClipDiagnostics:
    """Content-free clipping diagnostics for one layer.

    Args:
        layer_name: Dotted layer name from the coordinator's allowlist.
        norm_bound: Positive L2 bound that was applied to the layer.
        element_count: Number of scalar values in the layer.
        clipped: Whether the layer norm exceeded its bound before clipping.
        reason_code: One of ``CLIPPING_REASON_CODES``.
    """

    layer_name: str
    norm_bound: float
    element_count: int
    clipped: bool
    reason_code: str

    def __post_init__(self) -> None:
        _require_layer_name(self.layer_name)
        _require_norm_bound(self.norm_bound)
        if (
            type(self.element_count) is not int
            or not 1 <= self.element_count <= MAX_CLIPPING_ELEMENTS_PER_LAYER
        ):
            raise FederatedUpdateClippingError("invalid layer element count")
        if type(self.clipped) is not bool:
            raise FederatedUpdateClippingError("invalid clipping status")
        if (
            type(self.reason_code) is not str
            or self.reason_code not in CLIPPING_REASON_CODES
        ):
            raise FederatedUpdateClippingError("invalid clipping reason code")
        if self.clipped != (self.reason_code.startswith("scaled_to_")):
            raise FederatedUpdateClippingError("inconsistent clipping diagnostics")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready diagnostic without any update values."""
        return {
            "layer": self.layer_name,
            "norm_bound": self.norm_bound,
            "element_count": self.element_count,
            "clipped": self.clipped,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class FederatedClippingReport:
    """Deterministic, content-free clipping report.

    Args:
        policy_digest: Domain-separated SHA-256 of the trusted policy.
        layers: Per-layer diagnostics in canonical layer-name order.
        schema_version: Clipping report schema version.
    """

    policy_digest: str
    layers: tuple[FederatedLayerClipDiagnostics, ...]
    schema_version: str = UPDATE_CLIPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != UPDATE_CLIPPING_SCHEMA_VERSION:
            raise FederatedUpdateClippingError("unsupported clipping report version")
        _require_digest(self.policy_digest)
        if (
            type(self.layers) is not tuple
            or not 1 <= len(self.layers) <= MAX_CLIPPING_LAYERS
        ):
            raise FederatedUpdateClippingError("invalid clipping report layers")
        names: list[str] = []
        total = 0
        for layer in self.layers:
            if type(layer) is not FederatedLayerClipDiagnostics:
                raise FederatedUpdateClippingError("invalid clipping report layers")
            names.append(layer.layer_name)
            total += layer.element_count
            if total > MAX_CLIPPING_TOTAL_ELEMENTS:
                raise FederatedUpdateClippingError("total element count exceeds limit")
        if names != sorted(names) or len(set(names)) != len(names):
            raise FederatedUpdateClippingError(
                "clipping report layers must be unique and ordered"
            )

    @property
    def layer_count(self) -> int:
        """Number of layers in the report."""
        return len(self.layers)

    @property
    def element_count(self) -> int:
        """Total number of scalar values across all layers."""
        return sum(layer.element_count for layer in self.layers)

    @property
    def clipped(self) -> bool:
        """Whether at least one layer norm exceeded its bound."""
        return any(layer.clipped for layer in self.layers)

    @property
    def clipped_layer_count(self) -> int:
        """Number of layers whose norm exceeded its bound."""
        return sum(1 for layer in self.layers if layer.clipped)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Reason codes in canonical layer order."""
        return tuple(layer.reason_code for layer in self.layers)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report without any update values."""
        return {
            "schema_version": self.schema_version,
            "policy_digest": self.policy_digest,
            "layer_count": self.layer_count,
            "element_count": self.element_count,
            "clipped": self.clipped,
            "clipped_layer_count": self.clipped_layer_count,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def to_json(self) -> str:
        """Return canonical report JSON text, newline terminated."""
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


@dataclass(frozen=True, slots=True)
class ClippedFederatedUpdate:
    """Clipped layer deltas paired with their content-free report.

    Args:
        layers: Pairs of dotted layer name and clipped scalar values in
            canonical layer-name order.
        report: Diagnostics that describe the clipping decision per layer.
    """

    layers: tuple[tuple[str, tuple[float, ...]], ...]
    report: FederatedClippingReport

    def __post_init__(self) -> None:
        if type(self.report) is not FederatedClippingReport:
            raise FederatedUpdateClippingError("invalid clipping report")
        normalized = _require_clipped_layers(self.layers)
        object.__setattr__(self, "layers", normalized)
        names = tuple(name for name, _ in normalized)
        if names != tuple(layer.layer_name for layer in self.report.layers):
            raise FederatedUpdateClippingError("clipped layers do not match the report")
        for (_, values), layer in zip(normalized, self.report.layers):
            if _l2_norm(values) > layer.norm_bound * _NORM_ROUNDING_SLACK:
                raise FederatedUpdateClippingError(
                    "clipped layer exceeds its norm bound"
                )

    def layer_deltas(self, layer_name: str) -> tuple[float, ...]:
        """Return the clipped values for one layer.

        Args:
            layer_name: Dotted layer name present in the update.

        Returns:
            The clipped scalar values for the layer.

        Raises:
            FederatedUpdateClippingError: If the name is invalid or absent.
        """
        _require_layer_name(layer_name)
        for name, values in self.layers:
            if name == layer_name:
                return values
        raise FederatedUpdateClippingError("unknown layer name")

    def to_mapping(self) -> dict[str, tuple[float, ...]]:
        """Return a fresh layer-name to clipped-values mapping."""
        return {name: values for name, values in self.layers}

    def to_dict(self) -> dict[str, Any]:
        """Return clipped values plus the content-free report."""
        return {
            "layers": [
                {"layer": name, "values": list(values)} for name, values in self.layers
            ],
            "report": self.report.to_dict(),
        }

    def to_json(self) -> str:
        """Return canonical JSON text for the clipped values and report."""
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def clip_federated_update(
    deltas: Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]],
    *,
    policy: FederatedClippingPolicy,
) -> ClippedFederatedUpdate:
    """Clip adapter deltas to the trusted global or per-layer L2 norm bound.

    A layer whose L2 norm exceeds its bound is scaled by ``bound / norm``; a
    layer within its bound is returned unchanged. Layer order in the result is
    canonical, so equal inputs produce identical output regardless of mapping
    insertion order.

    Args:
        deltas: Layer name to finite scalar values. Accepts a mapping, or a
            sequence of ``(name, values)`` pairs for JSON-shaped input.
        policy: Trusted coordinator bounds; never derive these from the update.

    Returns:
        Clipped deltas paired with a content-free per-layer report.

    Raises:
        FederatedUpdateClippingError: If the policy or any layer is malformed,
            non-finite, out of range, over the size limits, or missing a
            trusted per-layer bound. Messages never include submitted numbers,
            layer names, or shapes.
    """
    if type(policy) is not FederatedClippingPolicy:
        raise FederatedUpdateClippingError("invalid clipping policy")
    layers = _require_update_layers(deltas)
    submitted = {name for name, _ in layers}
    if any(name not in submitted for name in policy.layer_names):
        raise FederatedUpdateClippingError("update is missing a bounded layer")

    diagnostics: list[FederatedLayerClipDiagnostics] = []
    clipped_layers: list[tuple[str, tuple[float, ...]]] = []
    overrides = frozenset(policy.layer_names)
    for name, values in layers:
        bound = policy.bound_for(name)
        clipped_values, was_clipped, reason_code = _clip_layer(
            values, bound, per_layer=name in overrides
        )
        diagnostics.append(
            FederatedLayerClipDiagnostics(
                layer_name=name,
                norm_bound=bound,
                element_count=len(values),
                clipped=was_clipped,
                reason_code=reason_code,
            )
        )
        clipped_layers.append((name, clipped_values))
    report = FederatedClippingReport(
        policy_digest=fingerprint_clipping_policy(policy),
        layers=tuple(diagnostics),
    )
    return ClippedFederatedUpdate(layers=tuple(clipped_layers), report=report)


def fingerprint_clipping_policy(policy: FederatedClippingPolicy) -> str:
    """Return a domain-separated SHA-256 digest of the trusted policy.

    Args:
        policy: Trusted coordinator bounds.

    Returns:
        ``sha256:`` prefixed hex digest over the canonical policy JSON.

    Raises:
        FederatedUpdateClippingError: If ``policy`` is not a policy instance.
    """
    if type(policy) is not FederatedClippingPolicy:
        raise FederatedUpdateClippingError("invalid clipping policy")
    payload = policy.to_json().encode("utf-8")
    return "sha256:" + hashlib.sha256(_FINGERPRINT_DOMAIN + payload).hexdigest()


def _clip_layer(
    values: tuple[float, ...], bound: float, *, per_layer: bool
) -> tuple[tuple[float, ...], bool, str]:
    norm = _l2_norm(values)
    if norm == 0.0:
        return values, False, "zero_norm"
    if norm <= bound:
        return values, False, "within_bound"
    scale = bound / norm
    clipped = tuple(value * scale for value in values)
    for _ in range(2):
        residual = _l2_norm(clipped)
        if residual <= bound * _NORM_ROUNDING_SLACK:
            break
        correction = bound / residual
        clipped = tuple(value * correction for value in clipped)
    reason_code = "scaled_to_layer_bound" if per_layer else "scaled_to_global_bound"
    return clipped, True, reason_code


def _l2_norm(values: Sequence[float]) -> float:
    largest = max(abs(value) for value in values)
    if largest == 0.0:
        return 0.0
    return largest * math.sqrt(math.fsum((value / largest) ** 2 for value in values))


def _require_update_layers(
    value: object,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if type(value) is dict:
        if not 1 <= len(value) <= MAX_CLIPPING_LAYERS:
            raise FederatedUpdateClippingError("invalid update layers")
        entries: list[tuple[object, object]] = list(value.items())
    elif type(value) in (list, tuple):
        pairs = cast("Sequence[object]", value)
        if not 1 <= len(pairs) <= MAX_CLIPPING_LAYERS:
            raise FederatedUpdateClippingError("invalid update layers")
        entries = []
        for entry in pairs:
            if type(entry) not in (list, tuple):
                raise FederatedUpdateClippingError("invalid update layers")
            pair = cast("Sequence[object]", entry)
            if len(pair) != 2:
                raise FederatedUpdateClippingError("invalid update layers")
            entries.append((pair[0], pair[1]))
    else:
        raise FederatedUpdateClippingError("invalid update layers")

    names: list[str] = []
    layers: list[tuple[str, tuple[float, ...]]] = []
    total = 0
    for name, values in entries:
        layer_name = _require_layer_name(name)
        normalized = _require_delta_values(values)
        total += len(normalized)
        if total > MAX_CLIPPING_TOTAL_ELEMENTS:
            raise FederatedUpdateClippingError("total element count exceeds limit")
        names.append(layer_name)
        layers.append((layer_name, normalized))
    if len(set(names)) != len(names):
        raise FederatedUpdateClippingError("duplicate layer names")
    layers.sort(key=lambda layer: layer[0])
    return tuple(layers)


def _require_clipped_layers(
    value: object,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    if type(value) is not tuple or not 1 <= len(value) <= MAX_CLIPPING_LAYERS:
        raise FederatedUpdateClippingError("invalid clipped layers")
    names: list[str] = []
    layers: list[tuple[str, tuple[float, ...]]] = []
    for entry in value:
        if type(entry) is not tuple or len(entry) != 2:
            raise FederatedUpdateClippingError("invalid clipped layers")
        name = _require_layer_name(entry[0])
        layers.append((name, _require_clipped_values(entry[1])))
        names.append(name)
    if names != sorted(names) or len(set(names)) != len(names):
        raise FederatedUpdateClippingError("clipped layers must be unique and ordered")
    return tuple(layers)


def _require_delta_values(value: object) -> tuple[float, ...]:
    if type(value) not in (list, tuple):
        raise FederatedUpdateClippingError("invalid layer values")
    items = cast("Sequence[object]", value)
    if not 1 <= len(items) <= MAX_CLIPPING_ELEMENTS_PER_LAYER:
        raise FederatedUpdateClippingError("invalid layer values")
    normalized: list[float] = []
    for item in items:
        if type(item) is bool or type(item) not in (int, float):
            raise FederatedUpdateClippingError("invalid layer value")
        try:
            number = float(cast("int | float", item))
        except OverflowError:
            raise FederatedUpdateClippingError("invalid layer value") from None
        if not math.isfinite(number) or abs(number) > MAX_CLIPPING_VALUE:
            raise FederatedUpdateClippingError("invalid layer value")
        normalized.append(number)
    return tuple(normalized)


def _require_clipped_values(value: object) -> tuple[float, ...]:
    if (
        type(value) is not tuple
        or not 1 <= len(value) <= MAX_CLIPPING_ELEMENTS_PER_LAYER
    ):
        raise FederatedUpdateClippingError("invalid layer values")
    for item in value:
        if (
            type(item) is not float
            or not math.isfinite(item)
            or abs(item) > MAX_CLIPPING_VALUE
        ):
            raise FederatedUpdateClippingError("invalid layer value")
    return value


def _require_fields(
    payload: object, expected: frozenset[str], message: str
) -> dict[str, Any]:
    if (
        type(payload) is not dict
        or len(payload) != len(expected)
        or any(type(key) is not str for key in payload)
        or payload.keys() != expected
    ):
        raise FederatedUpdateClippingError(message)
    return payload


def _require_norm_bound(value: object) -> float:
    if type(value) is bool or type(value) not in (int, float):
        raise FederatedUpdateClippingError("invalid clipping norm bound")
    try:
        bound = float(cast("int | float", value))
    except OverflowError:
        raise FederatedUpdateClippingError("invalid clipping norm bound") from None
    if not math.isfinite(bound) or not 0.0 < bound <= MAX_CLIPPING_NORM_BOUND:
        raise FederatedUpdateClippingError("invalid clipping norm bound")
    return bound


def _require_bound_entries(value: object) -> tuple[tuple[str, float], ...]:
    if type(value) not in (list, tuple):
        raise FederatedUpdateClippingError("invalid clipping policy fields")
    entries = cast("Sequence[object]", value)
    if len(entries) > MAX_CLIPPING_LAYERS:
        raise FederatedUpdateClippingError("invalid clipping policy fields")
    pairs: list[tuple[str, float]] = []
    for entry in entries:
        if type(entry) not in (list, tuple):
            raise FederatedUpdateClippingError("invalid clipping policy fields")
        pair = cast("Sequence[object]", entry)
        if len(pair) != 2:
            raise FederatedUpdateClippingError("invalid clipping policy fields")
        pairs.append((_require_layer_name(pair[0]), _require_norm_bound(pair[1])))
    pairs.sort(key=lambda pair: pair[0])
    if len({name for name, _ in pairs}) != len(pairs):
        raise FederatedUpdateClippingError("duplicate clipping policy layers")
    return tuple(pairs)


def _require_layer_name(value: object) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 256
        or _NAME.fullmatch(value) is None
    ):
        raise FederatedUpdateClippingError("invalid layer name")
    return value


def _require_digest(value: object) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise FederatedUpdateClippingError("invalid SHA-256 digest reference")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FederatedUpdateClippingError("duplicate clipping policy fields")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FederatedUpdateClippingError("non-finite clipping policy number")


__all__ = [
    "CLIPPING_REASON_CODES",
    "MAX_CLIPPING_ELEMENTS_PER_LAYER",
    "MAX_CLIPPING_LAYERS",
    "MAX_CLIPPING_NORM_BOUND",
    "MAX_CLIPPING_TOTAL_ELEMENTS",
    "MAX_CLIPPING_VALUE",
    "UPDATE_CLIPPING_SCHEMA_VERSION",
    "ClippedFederatedUpdate",
    "FederatedClippingPolicy",
    "FederatedClippingReport",
    "FederatedLayerClipDiagnostics",
    "FederatedUpdateClippingError",
    "clip_federated_update",
    "fingerprint_clipping_policy",
]
