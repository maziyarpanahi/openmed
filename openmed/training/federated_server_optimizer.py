"""Deterministic server optimizer rules over aggregated adapter deltas.

A federated round produces one already-aggregated adapter delta. This module
owns the next step: it turns that aggregate into the next global adapter
through a declared server rule (server SGD, heavy-ball or Nesterov momentum,
or the FedAdagrad, FedAdam and FedYogi adaptive rules of Reddi et al.,
"Adaptive Federated Optimization", ICLR 2021) and checkpoints the server
optimizer state.

Numerics use only the standard library. Every operation is elementwise
IEEE-754 float64 arithmetic in a fixed order, so results are byte-identical
across runs, processes and hash seeds. Aggregation, coordinator execution and
lineage storage are out of scope and are supplied by the caller.

Delta convention: an aggregate delta is the aggregated client adapter minus
the current global adapter, so a server step moves the global adapter in the
delta direction (``x + learning_rate * direction``). With server SGD and a
learning rate of 1.0 the rule reduces to adding the aggregate delta.

Privacy: the server step reads only the released aggregate, prior server
state and public hyperparameters, so it is post-processing of the released
aggregate and adds no differential-privacy charge. Records state this through
``SERVER_UPDATE_PRIVACY_ACCOUNTING``. Server state, checkpoints and records
hold no per-client data: only the parameter schema, hyperparameter codes,
digests, a step counter and aggregate-derived optimizer moments.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, cast

from .federated_schema_fingerprint import fingerprint_update_schema
from .federated_update_metadata import FederatedUpdateMetadata

SERVER_OPTIMIZER_STATE_SCHEMA_VERSION: Final = (
    "openmed.training.federated_server_optimizer_state.v1"
)
SERVER_UPDATE_RECORD_SCHEMA_VERSION: Final = (
    "openmed.training.federated_server_update_record.v1"
)
SERVER_UPDATE_PRIVACY_ACCOUNTING: Final = "post_processing_no_additional_charge"

MAX_SERVER_OPTIMIZER_PARAMETERS: Final = 1024
MAX_SERVER_OPTIMIZER_ELEMENTS: Final = 1 << 21
MAX_SERVER_OPTIMIZER_STEPS: Final = (1 << 31) - 1

SERVER_OPTIMIZER_ERROR_CODES: Final = frozenset(
    {
        "invalid_rule",
        "invalid_hyperparameter",
        "invalid_hyperparameter_code",
        "invalid_parameter_schema",
        "parameter_limit_exceeded",
        "invalid_values",
        "schema_mismatch",
        "non_finite_value",
        "step_limit_exceeded",
        "invalid_state",
        "state_too_large",
        "unsupported_state_version",
        "state_integrity_mismatch",
        "state_lineage_mismatch",
        "state_rule_mismatch",
        "state_hyperparameter_mismatch",
        "state_schema_mismatch",
        "state_step_mismatch",
    }
)

_DOMAIN_PREFIX: Final = b"openmed.training.federated_server_optimizer."
_CONFIG_DOMAIN: Final = _DOMAIN_PREFIX + b"config.v1\0"
_SCHEMA_DOMAIN: Final = _DOMAIN_PREFIX + b"parameter_schema.v1\0"
_STATE_DOMAIN: Final = _DOMAIN_PREFIX + b"state.v1\0"
_ADAPTER_DOMAIN: Final = _DOMAIN_PREFIX + b"adapter_values.v1\0"
_DELTA_DOMAIN: Final = _DOMAIN_PREFIX + b"aggregate_delta.v1\0"

_NAME: Final = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*\Z"
)
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_NAME_LENGTH: Final = 256
_MAX_CODE_LENGTH: Final = 512
_MAX_LEARNING_RATE: Final = 100.0
_MAX_INITIAL_ACCUMULATOR: Final = 1e6
_CHECKPOINT_BASE_BYTES: Final = 1024 * 1024
_CHECKPOINT_BYTES_PER_ELEMENT: Final = 12
_STATE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "rule",
        "hyperparameter_code",
        "config_digest",
        "parameter_schema",
        "parameter_schema_digest",
        "step",
        "slots",
        "state_digest",
    }
)
_SCHEMA_FIELDS: Final = frozenset({"parameters", "update_schema_fingerprint"})


class ServerOptimizerError(ValueError):
    """Fixed-code refusal that never echoes submitted names or values.

    Args:
        code: Stable reason code from ``SERVER_OPTIMIZER_ERROR_CODES``. Unknown
            codes collapse to ``invalid_state``.
        field: Optional fixed hyperparameter field name, never caller data.
    """

    def __init__(self, code: str, field: str | None = None) -> None:
        self.code = code if code in SERVER_OPTIMIZER_ERROR_CODES else "invalid_state"
        self.field = field
        super().__init__(self.code if field is None else f"{self.code}: {field}")


class ServerOptimizerRule(str, Enum):
    """Declared server update rules.

    ``SGD`` applies ``x + lr * delta``. ``MOMENTUM`` keeps a heavy-ball
    velocity ``v = momentum * v + delta`` and applies ``x + lr * v``, or
    ``x + lr * (delta + momentum * v)`` when Nesterov momentum is declared.
    ``ADAGRAD``, ``ADAM`` and ``YOGI`` follow Algorithm 2 of Reddi et al.
    (2021) without bias correction: ``m = beta1 * m + (1 - beta1) * delta``,
    a rule-specific second moment ``v`` and ``x + lr * (m / (sqrt(v) + tau))``.
    """

    SGD = "sgd"
    MOMENTUM = "momentum"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    YOGI = "yogi"

    def __str__(self) -> str:
        return self.value


_ADAPTIVE: Final = frozenset(
    {ServerOptimizerRule.ADAGRAD, ServerOptimizerRule.ADAM, ServerOptimizerRule.YOGI}
)
_CODE_FIELDS: Final[Mapping[ServerOptimizerRule, tuple[str, ...]]] = MappingProxyType(
    {
        ServerOptimizerRule.SGD: ("learning_rate",),
        ServerOptimizerRule.MOMENTUM: ("learning_rate", "momentum", "nesterov"),
        ServerOptimizerRule.ADAGRAD: (
            "learning_rate",
            "beta1",
            "tau",
            "initial_accumulator",
        ),
        ServerOptimizerRule.ADAM: (
            "learning_rate",
            "beta1",
            "beta2",
            "tau",
            "initial_accumulator",
        ),
        ServerOptimizerRule.YOGI: (
            "learning_rate",
            "beta1",
            "beta2",
            "tau",
            "initial_accumulator",
        ),
    }
)
_SLOTS: Final[Mapping[ServerOptimizerRule, tuple[str, ...]]] = MappingProxyType(
    {
        ServerOptimizerRule.SGD: (),
        ServerOptimizerRule.MOMENTUM: ("velocity",),
        ServerOptimizerRule.ADAGRAD: ("first_moment", "second_moment"),
        ServerOptimizerRule.ADAM: ("first_moment", "second_moment"),
        ServerOptimizerRule.YOGI: ("first_moment", "second_moment"),
    }
)
_DEFAULT_MOMENTUM: Final = 0.9
_DEFAULT_BETA1: Final = 0.9
_DEFAULT_BETA2: Final = 0.99
_DEFAULT_TAU: Final = 1e-3


@dataclass(frozen=True)
class ServerOptimizerConfig:
    """Declared server rule with fully resolved, range-checked hyperparameters.

    Hyperparameters that do not apply to the declared rule must be left unset,
    so nothing is silently ignored. Unset values that apply take documented
    defaults (momentum 0.9; beta1 0.9, beta2 0.99, tau 1e-3 and an initial
    second-moment accumulator of ``tau * tau`` for adaptive rules).

    Args:
        rule: Server rule, as a ``ServerOptimizerRule`` or its string value.
        learning_rate: Server learning rate in ``(0, 100]``.
        momentum: Heavy-ball coefficient in ``[0, 1)``; ``MOMENTUM`` only.
        nesterov: Use Nesterov momentum; ``MOMENTUM`` only.
        beta1: First-moment decay in ``[0, 1)``; adaptive rules only.
        beta2: Second-moment decay in ``[0, 1)``; ``ADAM`` and ``YOGI`` only.
        tau: Adaptivity constant in ``(0, 1]``; adaptive rules only.
        initial_accumulator: Initial second moment, at least ``tau * tau`` and
            at most ``1e6``; adaptive rules only.
    """

    rule: ServerOptimizerRule
    learning_rate: float
    momentum: float | None = None
    nesterov: bool = False
    beta1: float | None = None
    beta2: float | None = None
    tau: float | None = None
    initial_accumulator: float | None = None

    def __post_init__(self) -> None:
        rule = _require_rule(self.rule)
        set_ = object.__setattr__
        set_(self, "rule", rule)
        set_(
            self,
            "learning_rate",
            _hyper(self.learning_rate, "learning_rate", 0.0, _MAX_LEARNING_RATE, True),
        )
        if type(self.nesterov) is not bool:
            raise ServerOptimizerError("invalid_hyperparameter", "nesterov")
        if rule is ServerOptimizerRule.MOMENTUM:
            momentum = _DEFAULT_MOMENTUM if self.momentum is None else self.momentum
            set_(self, "momentum", _hyper(momentum, "momentum", 0.0, 1.0, False))
        else:
            _require_unset(self.momentum, "momentum")
            if self.nesterov:
                raise ServerOptimizerError("invalid_hyperparameter", "nesterov")
        if rule in _ADAPTIVE:
            beta1 = _DEFAULT_BETA1 if self.beta1 is None else self.beta1
            set_(self, "beta1", _hyper(beta1, "beta1", 0.0, 1.0, False))
            tau = _DEFAULT_TAU if self.tau is None else self.tau
            tau = _hyper(tau, "tau", 0.0, 1.0, True)
            set_(self, "tau", tau)
            floor = tau * tau
            initial = self.initial_accumulator
            initial = (
                floor if initial is None else _finite(initial, "initial_accumulator")
            )
            if not floor <= initial <= _MAX_INITIAL_ACCUMULATOR:
                raise ServerOptimizerError(
                    "invalid_hyperparameter", "initial_accumulator"
                )
            set_(self, "initial_accumulator", initial)
        else:
            for name in ("beta1", "tau", "initial_accumulator"):
                _require_unset(getattr(self, name), name)
        if rule in (ServerOptimizerRule.ADAM, ServerOptimizerRule.YOGI):
            beta2 = _DEFAULT_BETA2 if self.beta2 is None else self.beta2
            set_(self, "beta2", _hyper(beta2, "beta2", 0.0, 1.0, False))
        else:
            _require_unset(self.beta2, "beta2")

    @property
    def hyperparameter_code(self) -> str:
        """Return the canonical ASCII declaration of the rule and its values.

        Fields appear in a fixed per-rule order and floats use exact
        ``float.hex`` encoding, for example
        ``rule=sgd;learning_rate=0x1.0000000000000p-1``.
        """

        parts = [f"rule={self.rule.value}"]
        for name in _CODE_FIELDS[self.rule]:
            value = getattr(self, name)
            encoded = ("1" if value else "0") if name == "nesterov" else value.hex()
            parts.append(f"{name}={encoded}")
        return ";".join(parts)

    @property
    def config_digest(self) -> str:
        """Return a domain-separated SHA-256 reference for the declaration."""

        return _sha256(_CONFIG_DOMAIN, self.hyperparameter_code.encode("ascii"))

    @property
    def slot_names(self) -> tuple[str, ...]:
        """Return the optimizer state slots the rule maintains."""

        return _SLOTS[self.rule]

    def to_dict(self) -> dict[str, str]:
        """Return the round-contract declaration: rule, code and digest."""

        return {
            "rule": self.rule.value,
            "hyperparameter_code": self.hyperparameter_code,
            "config_digest": self.config_digest,
        }

    @classmethod
    def from_hyperparameter_code(cls, code: str) -> ServerOptimizerConfig:
        """Parse a canonical hyperparameter code declared in a round contract.

        Args:
            code: Exactly the string produced by ``hyperparameter_code``.

        Returns:
            The matching configuration.

        Raises:
            ServerOptimizerError: If the code is not canonical, names an
                unknown rule or declares an out-of-range value.
        """

        if (
            type(code) is not str
            or not 0 < len(code) <= _MAX_CODE_LENGTH
            or not code.isascii()
        ):
            raise ServerOptimizerError("invalid_hyperparameter_code")
        pairs = [part.partition("=") for part in code.split(";")]
        if any(not sep for _, sep, _ in pairs) or pairs[0][0] != "rule":
            raise ServerOptimizerError("invalid_hyperparameter_code")
        rule = _require_rule(pairs[0][2])
        names = tuple(name for name, _, _ in pairs[1:])
        if names != _CODE_FIELDS[rule]:
            raise ServerOptimizerError("invalid_hyperparameter_code")
        values: dict[str, Any] = {}
        try:
            for name, _, raw in pairs[1:]:
                if name == "nesterov":
                    if raw not in ("0", "1"):
                        raise ValueError
                    values[name] = raw == "1"
                else:
                    values[name] = float.fromhex(raw)
        except (ValueError, OverflowError):
            raise ServerOptimizerError("invalid_hyperparameter_code") from None
        config = cls(rule=rule, **values)
        if config.hyperparameter_code != code:
            raise ServerOptimizerError("invalid_hyperparameter_code")
        return config


@dataclass(frozen=True)
class ServerParameterSchema:
    """Canonical adapter parameter names and flat element counts.

    Args:
        parameters: ``(name, element_count)`` pairs; stored sorted by name.
            Names are dotted ASCII identifiers of at most 256 characters.
        update_schema_fingerprint: Optional ``sha256:`` fingerprint of the
            validated update schema the adapter was declared with.
    """

    parameters: tuple[tuple[str, int], ...]
    update_schema_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, (tuple, list)) or not self.parameters:
            raise ServerOptimizerError("invalid_parameter_schema")
        if len(self.parameters) > MAX_SERVER_OPTIMIZER_PARAMETERS:
            raise ServerOptimizerError("parameter_limit_exceeded")
        ordered: list[tuple[str, int]] = []
        total = 0
        for entry in self.parameters:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ServerOptimizerError("invalid_parameter_schema")
            name, length = entry
            if (
                type(name) is not str
                or not 0 < len(name) <= _MAX_NAME_LENGTH
                or _NAME.fullmatch(name) is None
                or type(length) is not int
                or length < 1
            ):
                raise ServerOptimizerError("invalid_parameter_schema")
            total += length
            if total > MAX_SERVER_OPTIMIZER_ELEMENTS:
                raise ServerOptimizerError("parameter_limit_exceeded")
            ordered.append((name, length))
        ordered.sort()
        if len({name for name, _ in ordered}) != len(ordered):
            raise ServerOptimizerError("invalid_parameter_schema")
        fingerprint = self.update_schema_fingerprint
        if fingerprint is not None and (
            type(fingerprint) is not str or _DIGEST.fullmatch(fingerprint) is None
        ):
            raise ServerOptimizerError("invalid_parameter_schema")
        object.__setattr__(self, "parameters", tuple(ordered))

    @classmethod
    def from_lengths(
        cls,
        lengths: Mapping[str, int],
        *,
        update_schema_fingerprint: str | None = None,
    ) -> ServerParameterSchema:
        """Build a schema from a ``name -> element count`` mapping.

        Args:
            lengths: Parameter names mapped to positive flat element counts.
            update_schema_fingerprint: Optional validated update fingerprint.

        Returns:
            The canonical schema.
        """

        if not isinstance(lengths, Mapping):
            raise ServerOptimizerError("invalid_parameter_schema")
        return cls(
            tuple(lengths.items()), update_schema_fingerprint=update_schema_fingerprint
        )

    @classmethod
    def from_update_metadata(
        cls, metadata: FederatedUpdateMetadata
    ) -> ServerParameterSchema:
        """Derive the schema from validated federated update metadata.

        Each parameter's element count is the product of its declared shape,
        and the update schema fingerprint is bound into the schema digest.

        Args:
            metadata: Metadata already validated against a coordinator policy.

        Returns:
            The canonical schema bound to the update schema fingerprint.
        """

        if type(metadata) is not FederatedUpdateMetadata:
            raise ServerOptimizerError("invalid_parameter_schema")
        return cls(
            tuple(
                (parameter.name, parameter.element_count)
                for parameter in metadata.parameters
            ),
            update_schema_fingerprint=fingerprint_update_schema(metadata),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Return parameter names in canonical order."""

        return tuple(name for name, _ in self.parameters)

    @property
    def total_elements(self) -> int:
        """Return the total flat element count."""

        return sum(length for _, length in self.parameters)

    @property
    def digest(self) -> str:
        """Return a domain-separated SHA-256 reference for the schema."""

        return _sha256(_SCHEMA_DOMAIN, _canonical_json(self._to_payload()))

    def _to_payload(self) -> dict[str, Any]:
        return {
            "parameters": [[name, length] for name, length in self.parameters],
            "update_schema_fingerprint": self.update_schema_fingerprint,
        }


@dataclass(frozen=True, eq=False)
class ServerOptimizerState:
    """Immutable server optimizer state for one declared rule and schema.

    State holds only the declaration, the parameter schema, a step counter
    and aggregate-derived moment vectors. ``repr`` hides the vectors.

    Args:
        config: Declared server rule and hyperparameters.
        schema: Parameter schema the moment vectors follow.
        step: Number of server updates already applied.
        slots: Slot name mapped to ``parameter name -> float64 vector``; the
            slot set must match ``config.slot_names``.
    """

    config: ServerOptimizerConfig
    schema: ServerParameterSchema
    step: int
    slots: Mapping[str, Mapping[str, tuple[float, ...]]] = field(repr=False)
    state_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.config) is not ServerOptimizerConfig:
            raise ServerOptimizerError("invalid_state")
        if type(self.schema) is not ServerParameterSchema:
            raise ServerOptimizerError("invalid_state")
        if type(self.step) is not int or not 0 <= self.step:
            raise ServerOptimizerError("invalid_state")
        if self.step > MAX_SERVER_OPTIMIZER_STEPS:
            raise ServerOptimizerError("step_limit_exceeded")
        if not isinstance(self.slots, Mapping) or set(self.slots) != set(
            self.config.slot_names
        ):
            raise ServerOptimizerError("invalid_state")
        frozen: dict[str, Mapping[str, tuple[float, ...]]] = {}
        for slot in self.config.slot_names:
            try:
                values = _require_values(self.slots[slot], self.schema)
            except ServerOptimizerError:
                raise ServerOptimizerError("invalid_state") from None
            if slot == "second_moment" and any(
                value < 0.0 for vector in values.values() for value in vector
            ):
                raise ServerOptimizerError("invalid_state")
            frozen[slot] = MappingProxyType(values)
        object.__setattr__(self, "slots", MappingProxyType(frozen))
        body = _canonical_json(self._body())
        object.__setattr__(self, "state_digest", _sha256(_STATE_DOMAIN, body))

    def slot_values(self, slot: str) -> Mapping[str, tuple[float, ...]]:
        """Return one read-only slot as ``parameter name -> vector``.

        Args:
            slot: A name from ``config.slot_names``.

        Returns:
            The read-only mapping for the slot.
        """

        if slot not in self.slots:
            raise ServerOptimizerError("invalid_state")
        return self.slots[slot]

    def describe(self) -> dict[str, Any]:
        """Return lineage evidence: codes, digests and the step, no vectors."""

        return {
            "schema_version": SERVER_OPTIMIZER_STATE_SCHEMA_VERSION,
            "rule": self.config.rule.value,
            "hyperparameter_code": self.config.hyperparameter_code,
            "config_digest": self.config.config_digest,
            "parameter_schema_digest": self.schema.digest,
            "step": self.step,
            "state_digest": self.state_digest,
        }

    def to_checkpoint_bytes(self) -> bytes:
        """Return versioned canonical checkpoint bytes with the state digest.

        The payload is sorted-key compact ASCII JSON. Vectors are base64 of
        big-endian IEEE-754 float64 values, so restore is bit-exact.
        """

        payload = self._body()
        payload["state_digest"] = self.state_digest
        return _canonical_json(payload)

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": SERVER_OPTIMIZER_STATE_SCHEMA_VERSION,
            "rule": self.config.rule.value,
            "hyperparameter_code": self.config.hyperparameter_code,
            "config_digest": self.config.config_digest,
            "parameter_schema": self.schema._to_payload(),
            "parameter_schema_digest": self.schema.digest,
            "step": self.step,
            "slots": {
                slot: {name: _encode_vector(vector) for name, vector in values.items()}
                for slot, values in self.slots.items()
            },
        }


@dataclass(frozen=True)
class ServerUpdateRecord:
    """Digest-only evidence for one server update, for lineage binding.

    Args:
        rule: Declared rule value.
        hyperparameter_code: Canonical hyperparameter declaration.
        config_digest: Digest of the declaration.
        parameter_schema_digest: Digest of the parameter schema.
        step: Server step after this update.
        previous_state_digest: Optimizer state digest before the update.
        state_digest: Optimizer state digest after the update.
        input_adapter_digest: Digest of the global adapter before the update.
        aggregate_delta_digest: Digest of the aggregate delta applied.
        output_adapter_digest: Digest of the next global adapter.
        privacy_accounting: Always ``SERVER_UPDATE_PRIVACY_ACCOUNTING``.
        schema_version: Record schema identifier.
    """

    rule: str
    hyperparameter_code: str
    config_digest: str
    parameter_schema_digest: str
    step: int
    previous_state_digest: str
    state_digest: str
    input_adapter_digest: str
    aggregate_delta_digest: str
    output_adapter_digest: str
    privacy_accounting: str = SERVER_UPDATE_PRIVACY_ACCOUNTING
    schema_version: str = SERVER_UPDATE_RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a fresh mapping of codes, digests and step."""

        return {
            "schema_version": self.schema_version,
            "rule": self.rule,
            "hyperparameter_code": self.hyperparameter_code,
            "config_digest": self.config_digest,
            "parameter_schema_digest": self.parameter_schema_digest,
            "step": self.step,
            "previous_state_digest": self.previous_state_digest,
            "state_digest": self.state_digest,
            "input_adapter_digest": self.input_adapter_digest,
            "aggregate_delta_digest": self.aggregate_delta_digest,
            "output_adapter_digest": self.output_adapter_digest,
            "privacy_accounting": self.privacy_accounting,
        }

    def to_json(self) -> str:
        """Return deterministic sorted-key compact JSON."""

        return _canonical_json(self.to_dict()).decode("ascii")


@dataclass(frozen=True, eq=False)
class ServerUpdateResult:
    """The next global adapter, the next optimizer state and its record.

    Args:
        global_adapter: Read-only ``parameter name -> float64 vector``.
        state: Optimizer state after the update.
        record: Digest-only evidence for lineage binding.
    """

    global_adapter: Mapping[str, tuple[float, ...]] = field(repr=False)
    state: ServerOptimizerState
    record: ServerUpdateRecord


def initial_server_optimizer_state(
    config: ServerOptimizerConfig, schema: ServerParameterSchema
) -> ServerOptimizerState:
    """Return step-zero state: zero first moments and velocities, and second
    moments at ``config.initial_accumulator``.

    Args:
        config: Declared server rule.
        schema: Adapter parameter schema.

    Returns:
        The initial optimizer state.
    """

    if type(config) is not ServerOptimizerConfig:
        raise ServerOptimizerError("invalid_state")
    if type(schema) is not ServerParameterSchema:
        raise ServerOptimizerError("invalid_parameter_schema")
    slots: dict[str, dict[str, tuple[float, ...]]] = {}
    for slot in config.slot_names:
        fill = config.initial_accumulator if slot == "second_moment" else 0.0
        slots[slot] = {name: (fill,) * length for name, length in schema.parameters}
    return ServerOptimizerState(config=config, schema=schema, step=0, slots=slots)


def apply_server_update(
    state: ServerOptimizerState,
    global_adapter: Mapping[str, Sequence[float]],
    aggregate_delta: Mapping[str, Sequence[float]],
) -> ServerUpdateResult:
    """Apply one aggregated delta through the declared server rule.

    Both mappings must cover exactly the schema's parameter names with
    matching lengths and finite values. Elements are processed in schema
    order with a fixed operation order, so results are byte-identical.

    Args:
        state: Current optimizer state; it is not modified.
        global_adapter: Current global adapter, ``name -> flat float values``.
        aggregate_delta: Already-aggregated delta, ``name -> flat values``.

    Returns:
        The next global adapter, next state and a digest-only record.

    Raises:
        ServerOptimizerError: With a fixed reason code for invalid or
            mismatched inputs, non-finite results or the step limit.
    """

    if type(state) is not ServerOptimizerState:
        raise ServerOptimizerError("invalid_state")
    if state.step >= MAX_SERVER_OPTIMIZER_STEPS:
        raise ServerOptimizerError("step_limit_exceeded")
    schema = state.schema
    adapter = _require_values(global_adapter, schema)
    delta = _require_values(aggregate_delta, schema)
    config = state.config
    rule = config.rule
    lr = config.learning_rate
    next_adapter: dict[str, tuple[float, ...]] = {}
    next_slots: dict[str, dict[str, tuple[float, ...]]] = {
        slot: {} for slot in config.slot_names
    }
    for name in schema.names:
        x = adapter[name]
        d = delta[name]
        if rule is ServerOptimizerRule.SGD:
            new_x = tuple([xi + lr * di for xi, di in zip(x, d)])
        elif rule is ServerOptimizerRule.MOMENTUM:
            new_x, velocity = _momentum_step(
                x, d, state.slots["velocity"][name], lr, config
            )
            next_slots["velocity"][name] = velocity
        else:
            new_x, first, second = _adaptive_step(
                x,
                d,
                state.slots["first_moment"][name],
                state.slots["second_moment"][name],
                lr,
                config,
            )
            next_slots["first_moment"][name] = first
            next_slots["second_moment"][name] = second
        next_adapter[name] = new_x
    for vectors in (next_adapter, *next_slots.values()):
        for vector in vectors.values():
            if not all(map(math.isfinite, vector)):
                raise ServerOptimizerError("non_finite_value")
    next_state = ServerOptimizerState(
        config=config, schema=schema, step=state.step + 1, slots=next_slots
    )
    record = ServerUpdateRecord(
        rule=rule.value,
        hyperparameter_code=config.hyperparameter_code,
        config_digest=config.config_digest,
        parameter_schema_digest=schema.digest,
        step=next_state.step,
        previous_state_digest=state.state_digest,
        state_digest=next_state.state_digest,
        input_adapter_digest=_values_digest(_ADAPTER_DOMAIN, adapter, schema),
        aggregate_delta_digest=_values_digest(_DELTA_DOMAIN, delta, schema),
        output_adapter_digest=_values_digest(_ADAPTER_DOMAIN, next_adapter, schema),
    )
    return ServerUpdateResult(
        global_adapter=MappingProxyType(next_adapter), state=next_state, record=record
    )


def restore_server_optimizer_state(
    payload: bytes,
    *,
    config: ServerOptimizerConfig,
    schema: ServerParameterSchema,
    expected_step: int,
    expected_state_digest: str | None = None,
) -> ServerOptimizerState:
    """Restore checkpoint bytes only when they match the expected round.

    Checks run in a fixed order: size, strict canonical JSON, version,
    integrity digest, the optional lineage-bound state digest, then rule,
    hyperparameter code, parameter schema digest and step. The embedded
    digest detects corruption and edits that do not recompute it; pass the
    state digest recorded in lineage evidence to also reject a recomputed or
    substituted checkpoint.

    Args:
        payload: Bytes from ``ServerOptimizerState.to_checkpoint_bytes``.
        config: Configuration declared for the round being resumed.
        schema: Expected adapter parameter schema.
        expected_step: Step the coordinator expects to resume from.
        expected_state_digest: Optional state digest bound in lineage.

    Returns:
        The restored state, bit-identical to the checkpointed state.

    Raises:
        ServerOptimizerError: With a fixed reason code on any mismatch.
    """

    if type(config) is not ServerOptimizerConfig:
        raise ServerOptimizerError("invalid_state")
    if type(schema) is not ServerParameterSchema:
        raise ServerOptimizerError("invalid_parameter_schema")
    if type(expected_step) is not int or expected_step < 0:
        raise ServerOptimizerError("invalid_state")
    if expected_state_digest is not None and (
        type(expected_state_digest) is not str
        or _DIGEST.fullmatch(expected_state_digest) is None
    ):
        raise ServerOptimizerError("invalid_state")
    if type(payload) is not bytes:
        raise ServerOptimizerError("invalid_state")
    limit = (
        _CHECKPOINT_BASE_BYTES
        + schema.total_elements
        * max(len(config.slot_names), 1)
        * _CHECKPOINT_BYTES_PER_ELEMENT
    )
    if len(payload) > limit:
        raise ServerOptimizerError("state_too_large")
    try:
        decoded = json.loads(
            payload.decode("ascii"), object_pairs_hook=_strict_json_object
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ServerOptimizerError("invalid_state") from None
    if not isinstance(decoded, dict) or set(decoded) != _STATE_FIELDS:
        raise ServerOptimizerError("invalid_state")
    if decoded["schema_version"] != SERVER_OPTIMIZER_STATE_SCHEMA_VERSION:
        raise ServerOptimizerError("unsupported_state_version")
    try:
        canonical = _canonical_json(decoded)
    except (TypeError, ValueError):
        raise ServerOptimizerError("invalid_state") from None
    if canonical != payload:
        raise ServerOptimizerError("invalid_state")
    embedded = decoded.pop("state_digest")
    recomputed = _sha256(_STATE_DOMAIN, _canonical_json(decoded))
    if type(embedded) is not str or not hmac.compare_digest(embedded, recomputed):
        raise ServerOptimizerError("state_integrity_mismatch")
    if expected_state_digest is not None and not hmac.compare_digest(
        expected_state_digest, recomputed
    ):
        raise ServerOptimizerError("state_lineage_mismatch")
    if decoded["rule"] != config.rule.value:
        raise ServerOptimizerError("state_rule_mismatch")
    if decoded["hyperparameter_code"] != config.hyperparameter_code or (
        decoded["config_digest"] != config.config_digest
    ):
        raise ServerOptimizerError("state_hyperparameter_mismatch")
    if (
        decoded["parameter_schema_digest"] != schema.digest
        or decoded["parameter_schema"] != schema._to_payload()
    ):
        raise ServerOptimizerError("state_schema_mismatch")
    if type(decoded["step"]) is not int or decoded["step"] != expected_step:
        raise ServerOptimizerError("state_step_mismatch")
    slots = decoded["slots"]
    if not isinstance(slots, dict) or set(slots) != set(config.slot_names):
        raise ServerOptimizerError("invalid_state")
    restored_slots: dict[str, dict[str, tuple[float, ...]]] = {}
    for slot, encoded in slots.items():
        if not isinstance(encoded, dict) or set(encoded) != set(schema.names):
            raise ServerOptimizerError("invalid_state")
        restored_slots[slot] = {
            name: _decode_vector(encoded[name], length)
            for name, length in schema.parameters
        }
    state = ServerOptimizerState(
        config=config, schema=schema, step=expected_step, slots=restored_slots
    )
    if not hmac.compare_digest(state.state_digest, recomputed):
        raise ServerOptimizerError("state_integrity_mismatch")
    return state


def digest_adapter_values(
    values: Mapping[str, Sequence[float]], schema: ServerParameterSchema
) -> str:
    """Return the domain-separated digest of global adapter values.

    The digest covers the schema digest and each parameter's big-endian
    float64 bytes in schema order.

    Args:
        values: ``name -> flat float values`` matching ``schema``.
        schema: Parameter schema.

    Returns:
        A ``sha256:`` reference.
    """

    if type(schema) is not ServerParameterSchema:
        raise ServerOptimizerError("invalid_parameter_schema")
    return _values_digest(_ADAPTER_DOMAIN, _require_values(values, schema), schema)


def digest_aggregate_delta(
    values: Mapping[str, Sequence[float]], schema: ServerParameterSchema
) -> str:
    """Return the domain-separated digest of an aggregate delta.

    Uses a different domain than ``digest_adapter_values`` so a delta digest
    can never be confused with an adapter digest.

    Args:
        values: ``name -> flat float values`` matching ``schema``.
        schema: Parameter schema.

    Returns:
        A ``sha256:`` reference.
    """

    if type(schema) is not ServerParameterSchema:
        raise ServerOptimizerError("invalid_parameter_schema")
    return _values_digest(_DELTA_DOMAIN, _require_values(values, schema), schema)


def _momentum_step(
    x: tuple[float, ...],
    d: tuple[float, ...],
    v: tuple[float, ...],
    lr: float,
    config: ServerOptimizerConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mu = cast(float, config.momentum)
    velocity = [mu * vi + di for vi, di in zip(v, d)]
    if config.nesterov:
        new_x = [xi + lr * (di + mu * vi) for xi, di, vi in zip(x, d, velocity)]
    else:
        new_x = [xi + lr * vi for xi, vi in zip(x, velocity)]
    return tuple(new_x), tuple(velocity)


def _adaptive_step(
    x: tuple[float, ...],
    d: tuple[float, ...],
    m: tuple[float, ...],
    v: tuple[float, ...],
    lr: float,
    config: ServerOptimizerConfig,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    beta1 = cast(float, config.beta1)
    tau = cast(float, config.tau)
    one_minus_beta1 = 1.0 - beta1
    first = [beta1 * mi + one_minus_beta1 * di for mi, di in zip(m, d)]
    squares = [di * di for di in d]
    if config.rule is ServerOptimizerRule.ADAGRAD:
        second = [vi + sq for vi, sq in zip(v, squares)]
    else:
        beta2 = cast(float, config.beta2)
        one_minus_beta2 = 1.0 - beta2
        if config.rule is ServerOptimizerRule.ADAM:
            second = [beta2 * vi + one_minus_beta2 * sq for vi, sq in zip(v, squares)]
        else:
            second = [
                vi - one_minus_beta2 * sq * _sign(vi - sq) for vi, sq in zip(v, squares)
            ]
    new_x = [
        xi + lr * (mi / (math.sqrt(vi) + tau)) for xi, mi, vi in zip(x, first, second)
    ]
    return tuple(new_x), tuple(first), tuple(second)


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _require_rule(value: object) -> ServerOptimizerRule:
    if isinstance(value, ServerOptimizerRule):
        return value
    if type(value) is str:
        try:
            return ServerOptimizerRule(value)
        except ValueError:
            pass
    raise ServerOptimizerError("invalid_rule")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServerOptimizerError("invalid_hyperparameter", name)
    converted = float(value)  # type: ignore[arg-type]
    if not math.isfinite(converted):
        raise ServerOptimizerError("invalid_hyperparameter", name)
    return converted


def _hyper(value: object, name: str, low: float, high: float, low_open: bool) -> float:
    """Check ``(low, high]`` when ``low_open`` is true, else ``[low, high)``.

    Learning rate and tau use an open lower bound and inclusive upper bound;
    decay coefficients use an inclusive lower bound and open upper bound.
    """

    converted = _finite(value, name)
    if low_open:
        valid = low < converted <= high
    else:
        valid = low <= converted < high
    if not valid:
        raise ServerOptimizerError("invalid_hyperparameter", name)
    return converted


def _require_unset(value: object, name: str) -> None:
    if value is not None:
        raise ServerOptimizerError("invalid_hyperparameter", name)


def _require_values(
    values: object, schema: ServerParameterSchema
) -> dict[str, tuple[float, ...]]:
    if not isinstance(values, Mapping):
        raise ServerOptimizerError("invalid_values")
    if len(values) != len(schema.parameters) or set(values) != set(schema.names):
        raise ServerOptimizerError("schema_mismatch")
    result: dict[str, tuple[float, ...]] = {}
    for name, length in schema.parameters:
        vector = values[name]
        if isinstance(vector, (str, bytes, bytearray)) or not isinstance(
            vector, Sequence
        ):
            raise ServerOptimizerError("invalid_values")
        if len(vector) != length:
            raise ServerOptimizerError("schema_mismatch")
        converted: list[float] = []
        for item in vector:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ServerOptimizerError("invalid_values")
            number = float(item)
            if not math.isfinite(number):
                raise ServerOptimizerError("non_finite_value")
            converted.append(number)
        result[name] = tuple(converted)
    return result


def _encode_vector(vector: tuple[float, ...]) -> str:
    return base64.b64encode(struct.pack(f">{len(vector)}d", *vector)).decode("ascii")


def _decode_vector(encoded: object, length: int) -> tuple[float, ...]:
    if type(encoded) is not str or len(encoded) > 4 * ((8 * length + 2) // 3):
        raise ServerOptimizerError("invalid_state")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError):
        raise ServerOptimizerError("invalid_state") from None
    if len(raw) != 8 * length or base64.b64encode(raw).decode("ascii") != encoded:
        raise ServerOptimizerError("invalid_state")
    vector = struct.unpack(f">{length}d", raw)
    if not all(map(math.isfinite, vector)):
        raise ServerOptimizerError("invalid_state")
    return vector


def _values_digest(
    domain: bytes,
    values: Mapping[str, tuple[float, ...]],
    schema: ServerParameterSchema,
) -> str:
    hasher = hashlib.sha256(domain)
    hasher.update(schema.digest.encode("ascii"))
    for name, length in schema.parameters:
        hasher.update(b"\0" + name.encode("ascii") + b"\0")
        hasher.update(length.to_bytes(8, "big"))
        hasher.update(struct.pack(f">{length}d", *values[name]))
    return "sha256:" + hasher.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(domain: bytes, payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(domain + payload).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


__all__ = [
    "MAX_SERVER_OPTIMIZER_ELEMENTS",
    "MAX_SERVER_OPTIMIZER_PARAMETERS",
    "MAX_SERVER_OPTIMIZER_STEPS",
    "SERVER_OPTIMIZER_ERROR_CODES",
    "SERVER_OPTIMIZER_STATE_SCHEMA_VERSION",
    "SERVER_UPDATE_PRIVACY_ACCOUNTING",
    "SERVER_UPDATE_RECORD_SCHEMA_VERSION",
    "ServerOptimizerConfig",
    "ServerOptimizerError",
    "ServerOptimizerRule",
    "ServerOptimizerState",
    "ServerParameterSchema",
    "ServerUpdateRecord",
    "ServerUpdateResult",
    "apply_server_update",
    "digest_adapter_values",
    "digest_aggregate_delta",
    "initial_server_optimizer_state",
    "restore_server_optimizer_state",
]
