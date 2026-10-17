"""Minimum local contribution floors for federated update submissions.

A round contract declares floors for the number of local examples, local
optimization steps, and the effective batch size behind one update. The
client-side guard takes a site's exact local counts, refuses to seal a
declaration below any floor, and emits only power-of-two bands. The
coordinator-side check compares declared bands with its own contract. Exact
local counts are never stored, serialized, or included in errors.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Mapping

from .federated_metrics import FederatedPrivacyMechanism

FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION = (
    "openmed.training.federated_contribution_floor_policy.v1"
)
FEDERATED_CONTRIBUTION_SCHEMA_VERSION = "openmed.training.federated_contribution.v1"
MAX_FEDERATED_CONTRIBUTION_FLOOR: Final = 1 << 16

# Rounds without a declared DP mechanism rely on the floors alone, so they use
# the stricter values: 8 local steps over batches of at least 32 examples cover
# one pass over the 256-example floor.
DEFAULT_NON_DP_CONTRIBUTION_FLOORS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "min_local_examples": 256,
        "min_local_steps": 8,
        "min_effective_batch_size": 32,
    }
)
# A declared DP mechanism bounds per-example influence; these floors only keep
# sampling from degenerating to a handful of examples.
DEFAULT_DP_CONTRIBUTION_FLOORS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "min_local_examples": 64,
        "min_local_steps": 1,
        "min_effective_batch_size": 8,
    }
)

_MAX_COUNT: Final = (1 << 63) - 1
_MAX_JSON_BYTES: Final = 1024 * 1024
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DOMAIN: Final = b"openmed.training.federated_contribution_floor_policy.v1\0"
_DP_MECHANISMS: Final = frozenset(
    {FederatedPrivacyMechanism.LAPLACE, FederatedPrivacyMechanism.GAUSSIAN}
)
_FLOOR_FIELDS: Final = (
    "min_local_examples",
    "min_local_steps",
    "min_effective_batch_size",
)
_POLICY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "privacy_mechanism",
        "allow_floors_below_default",
        *_FLOOR_FIELDS,
    }
)
_BAND_FIELDS: Final = (
    "local_example_band",
    "local_step_band",
    "effective_batch_size_band",
)
_DECLARATION_FIELDS: Final = frozenset(
    {"schema_version", "contract_digest", "update_digest", *_BAND_FIELDS}
)


class FederatedContributionError(ValueError):
    """Raised for invalid contribution inputs without including their values."""


class FederatedContributionBand(str, Enum):
    """Power-of-two range containing a local count, named by its bounds."""

    FROM_1 = "1_to_under_2"
    FROM_2 = "2_to_under_4"
    FROM_4 = "4_to_under_8"
    FROM_8 = "8_to_under_16"
    FROM_16 = "16_to_under_32"
    FROM_32 = "32_to_under_64"
    FROM_64 = "64_to_under_128"
    FROM_128 = "128_to_under_256"
    FROM_256 = "256_to_under_512"
    FROM_512 = "512_to_under_1024"
    FROM_1024 = "1024_to_under_2048"
    FROM_2048 = "2048_to_under_4096"
    FROM_4096 = "4096_to_under_8192"
    FROM_8192 = "8192_to_under_16384"
    FROM_16384 = "16384_to_under_32768"
    FROM_32768 = "32768_to_under_65536"
    FROM_65536 = "65536_or_more"

    def __str__(self) -> str:
        return self.value

    @property
    def lower_bound(self) -> int:
        """Return the inclusive lower bound, which is also a valid floor."""
        return int(self.name.removeprefix("FROM_"))

    @classmethod
    def for_count(cls, count: int) -> FederatedContributionBand:
        """Return the band containing a positive local count.

        Args:
            count: Positive integer count, at most ``2**63 - 1``.

        Returns:
            The band whose lower bound is the largest power of two not above
            ``count``, capped at the open top band.

        Raises:
            FederatedContributionError: If ``count`` is not a bounded positive
                integer. The message never includes the value.
        """
        if type(count) is not int or not 1 <= count <= _MAX_COUNT:
            raise FederatedContributionError("invalid local count")
        bound = min(1 << (count.bit_length() - 1), MAX_FEDERATED_CONTRIBUTION_FLOOR)
        return cls[f"FROM_{bound}"]


_FLOOR_VALUES: Final = frozenset(band.lower_bound for band in FederatedContributionBand)


class FederatedContributionReasonCode(str, Enum):
    """Stable reasons a contribution is refused or rejected."""

    CONTRACT_MISMATCH = "contract_mismatch"
    UPDATE_MISMATCH = "update_mismatch"
    LOCAL_EXAMPLES_BELOW_FLOOR = "local_examples_below_floor"
    LOCAL_STEPS_BELOW_FLOOR = "local_steps_below_floor"
    EFFECTIVE_BATCH_SIZE_BELOW_FLOOR = "effective_batch_size_below_floor"

    def __str__(self) -> str:
        return self.value


_REASON_ORDER: Final = {
    code: index for index, code in enumerate(FederatedContributionReasonCode)
}
_FLOOR_REASONS: Final = (
    FederatedContributionReasonCode.LOCAL_EXAMPLES_BELOW_FLOOR,
    FederatedContributionReasonCode.LOCAL_STEPS_BELOW_FLOOR,
    FederatedContributionReasonCode.EFFECTIVE_BATCH_SIZE_BELOW_FLOOR,
)


class FederatedContributionRefusedError(FederatedContributionError):
    """Raised by the client guard when local counts are below a floor.

    Args:
        reason_codes: Failed floors in canonical reason-code order.

    Attributes:
        reason_codes: The same codes. No count or floor value is retained.
    """

    def __init__(
        self, reason_codes: tuple[FederatedContributionReasonCode, ...]
    ) -> None:
        super().__init__("federated contribution is below round floors")
        self.reason_codes = _canonical_reasons(reason_codes)

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.reason_codes,))


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedContributionFloorPolicy:
    """Round-contract floors for one federated update.

    Omitted floors resolve to the defaults for the declared privacy
    mechanism: ``DEFAULT_NON_DP_CONTRIBUTION_FLOORS`` when no DP mechanism is
    declared (``threshold_only``) and ``DEFAULT_DP_CONTRIBUTION_FLOORS`` for
    ``laplace`` or ``gaussian``. Every floor must be a band lower bound: a
    power of two from 1 to 65,536.

    Args:
        privacy_mechanism: Round privacy mechanism from the existing
            :class:`FederatedPrivacyMechanism` vocabulary.
        min_local_examples: Minimum local examples behind one update.
        min_local_steps: Minimum local optimization steps behind one update.
        min_effective_batch_size: Minimum examples per optimization step,
            across accumulation and devices. At most ``min_local_examples``.
        allow_floors_below_default: Explicit policy field required to set any
            floor below its default. It is rejected when no floor is lowered.
        schema_version: Exact supported contract schema identifier.
    """

    privacy_mechanism: FederatedPrivacyMechanism = (
        FederatedPrivacyMechanism.THRESHOLD_ONLY
    )
    min_local_examples: int | None = None
    min_local_steps: int | None = None
    min_effective_batch_size: int | None = None
    allow_floors_below_default: bool = False
    schema_version: str = FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION
        ):
            raise FederatedContributionError("unsupported contribution floor schema")
        if not isinstance(self.privacy_mechanism, FederatedPrivacyMechanism):
            raise FederatedContributionError("invalid privacy mechanism")
        if type(self.allow_floors_below_default) is not bool:
            raise FederatedContributionError("invalid floor override policy")
        defaults = _defaults_for(self.privacy_mechanism)
        lowered = False
        for field in _FLOOR_FIELDS:
            value = getattr(self, field)
            if value is None:
                value = defaults[field]
                object.__setattr__(self, field, value)
            elif type(value) is not int or value not in _FLOOR_VALUES:
                raise FederatedContributionError("invalid contribution floor")
            lowered = lowered or value < defaults[field]
        if lowered and not self.allow_floors_below_default:
            raise FederatedContributionError(
                "contribution floor below default requires explicit override"
            )
        if self.allow_floors_below_default and not lowered:
            raise FederatedContributionError(
                "floor override requires a floor below the default"
            )
        if self.min_effective_batch_size > self.min_local_examples:
            raise FederatedContributionError(
                "effective batch floor exceeds local example floor"
            )

    @property
    def dp_mechanism_declared(self) -> bool:
        """Return whether the round declares a DP noise mechanism."""
        return self.privacy_mechanism in _DP_MECHANISMS

    @property
    def contract_digest(self) -> str:
        """Return a domain-separated SHA-256 reference to the canonical contract."""
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(_DOMAIN + canonical).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh versioned mapping with every floor explicit."""
        return {
            "schema_version": self.schema_version,
            "privacy_mechanism": self.privacy_mechanism.value,
            "min_local_examples": self.min_local_examples,
            "min_local_steps": self.min_local_steps,
            "min_effective_batch_size": self.min_effective_batch_size,
            "allow_floors_below_default": self.allow_floors_below_default,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with sorted keys and a trailing newline."""
        return _render(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> FederatedContributionFloorPolicy:
        """Validate a closed contract mapping with every floor explicit.

        Args:
            payload: Built-in dictionary with exactly the contract fields.
                Floors must be integers; ``null`` does not select a default.

        Returns:
            The validated immutable contract.

        Raises:
            FederatedContributionError: If the schema or floor policy is
                violated. Messages never include submitted keys or values.
        """
        payload = _require_fields(payload, _POLICY_FIELDS)
        if any(type(payload[field]) is not int for field in _FLOOR_FIELDS):
            raise FederatedContributionError("invalid contribution floor")
        mechanism = payload["privacy_mechanism"]
        if type(mechanism) is not str:
            raise FederatedContributionError("invalid privacy mechanism")
        try:
            mechanism = FederatedPrivacyMechanism(mechanism)
        except ValueError:
            raise FederatedContributionError("invalid privacy mechanism") from None
        return cls(
            privacy_mechanism=mechanism,
            min_local_examples=payload["min_local_examples"],
            min_local_steps=payload["min_local_steps"],
            min_effective_batch_size=payload["min_effective_batch_size"],
            allow_floors_below_default=payload["allow_floors_below_default"],
            schema_version=payload["schema_version"],
        )

    @classmethod
    def from_json(cls, payload: str) -> FederatedContributionFloorPolicy:
        """Parse bounded strict JSON into a validated contract.

        Args:
            payload: JSON text, at most one MiB of UTF-8.

        Returns:
            The validated immutable contract.

        Raises:
            FederatedContributionError: On invalid JSON, duplicate keys,
                non-integer numbers, or policy violations.
        """
        return cls.from_dict(_parse_json(payload))


@dataclass(frozen=True, slots=True, kw_only=True)
class FederatedContributionDeclaration:
    """Band-only contribution record bound to one contract and one update.

    Produce it on the client with :func:`seal_federated_contribution`. Parsing
    a declaration checks structure only; floors are enforced by
    :func:`check_federated_contribution` against the coordinator's contract.

    Args:
        contract_digest: ``contract_digest`` of the contract used to seal.
        update_digest: Declared SHA-256 reference of the sealed update.
        local_example_band: Band containing the local example count.
        local_step_band: Band containing the local optimization step count.
        effective_batch_size_band: Band containing the effective batch size.
        schema_version: Exact supported declaration schema identifier.
    """

    contract_digest: str
    update_digest: str
    local_example_band: FederatedContributionBand
    local_step_band: FederatedContributionBand
    effective_batch_size_band: FederatedContributionBand
    schema_version: str = FEDERATED_CONTRIBUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != FEDERATED_CONTRIBUTION_SCHEMA_VERSION
        ):
            raise FederatedContributionError("unsupported contribution schema")
        _require_digest(self.contract_digest)
        _require_digest(self.update_digest)
        if any(
            type(getattr(self, field)) is not FederatedContributionBand
            for field in _BAND_FIELDS
        ):
            raise FederatedContributionError("invalid contribution band")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh versioned mapping containing only digests and bands."""
        return {
            "schema_version": self.schema_version,
            "contract_digest": self.contract_digest,
            "update_digest": self.update_digest,
            "local_example_band": self.local_example_band.value,
            "local_step_band": self.local_step_band.value,
            "effective_batch_size_band": self.effective_batch_size_band.value,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with sorted keys and a trailing newline."""
        return _render(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> FederatedContributionDeclaration:
        """Validate the structure of a closed declaration mapping.

        Args:
            payload: Built-in dictionary with exactly the declaration fields.

        Returns:
            A structurally valid declaration. Floors are not checked here.

        Raises:
            FederatedContributionError: If fields, digests, or bands are
                invalid. Messages never include submitted keys or values.
        """
        payload = _require_fields(payload, _DECLARATION_FIELDS)
        bands = {}
        for field in _BAND_FIELDS:
            value = payload[field]
            if type(value) is not str:
                raise FederatedContributionError("invalid contribution band")
            try:
                bands[field] = FederatedContributionBand(value)
            except ValueError:
                raise FederatedContributionError("invalid contribution band") from None
        return cls(
            contract_digest=payload["contract_digest"],
            update_digest=payload["update_digest"],
            schema_version=payload["schema_version"],
            **bands,
        )

    @classmethod
    def from_json(cls, payload: str) -> FederatedContributionDeclaration:
        """Parse bounded strict JSON into a structurally valid declaration.

        Args:
            payload: JSON text, at most one MiB of UTF-8.

        Returns:
            A structurally valid declaration. Floors are not checked here.

        Raises:
            FederatedContributionError: On invalid JSON, duplicate keys,
                non-integer numbers, or schema violations.
        """
        return cls.from_dict(_parse_json(payload))


@dataclass(frozen=True, slots=True)
class FederatedContributionDecision:
    """Coordinator outcome containing only stable reason codes.

    Args:
        reason_codes: Unique codes in canonical enum order; empty when the
            contribution is accepted.
    """

    reason_codes: tuple[FederatedContributionReasonCode, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", _canonical_reasons(self.reason_codes))

    @property
    def accepted(self) -> bool:
        """Return whether no reason code was recorded."""
        return not self.reason_codes

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping with the acceptance flag and reason codes."""
        return {
            "accepted": self.accepted,
            "reason_codes": [code.value for code in self.reason_codes],
        }


def seal_federated_contribution(
    contract: FederatedContributionFloorPolicy,
    *,
    update_digest: str,
    local_examples: int,
    local_steps: int,
    effective_batch_size: int,
) -> FederatedContributionDeclaration:
    """Client-side guard: seal a band-only declaration or refuse.

    Exact counts are used only for local comparison and banding. They are not
    stored on the result, logged, or included in any error.

    Args:
        contract: Round contract received from the coordinator.
        update_digest: Declared SHA-256 reference of the local update.
        local_examples: Exact number of local examples behind the update.
        local_steps: Exact number of local optimization steps.
        effective_batch_size: Exact examples per optimization step, across
            accumulation and devices. Must not exceed ``local_examples``.

    Returns:
        A declaration carrying the contract digest, update digest, and bands.

    Raises:
        FederatedContributionRefusedError: If any count is below its floor;
            ``reason_codes`` lists every failed floor.
        FederatedContributionError: If an input is malformed.
    """
    if type(contract) is not FederatedContributionFloorPolicy:
        raise FederatedContributionError("invalid contribution contract")
    _require_digest(update_digest)
    counts = (local_examples, local_steps, effective_batch_size)
    if any(type(count) is not int or not 0 <= count <= _MAX_COUNT for count in counts):
        raise FederatedContributionError("invalid local count")
    if effective_batch_size > local_examples:
        raise FederatedContributionError("effective batch size exceeds local examples")
    floors = tuple(getattr(contract, field) for field in _FLOOR_FIELDS)
    failed = tuple(
        reason
        for count, floor, reason in zip(counts, floors, _FLOOR_REASONS, strict=True)
        if count < floor
    )
    if failed:
        raise FederatedContributionRefusedError(failed)
    return FederatedContributionDeclaration(
        contract_digest=contract.contract_digest,
        update_digest=update_digest,
        local_example_band=FederatedContributionBand.for_count(local_examples),
        local_step_band=FederatedContributionBand.for_count(local_steps),
        effective_batch_size_band=FederatedContributionBand.for_count(
            effective_batch_size
        ),
    )


def check_federated_contribution(
    declaration: FederatedContributionDeclaration,
    *,
    contract: FederatedContributionFloorPolicy,
    update_digest: str,
) -> FederatedContributionDecision:
    """Coordinator-side check of declared bands against the round contract.

    Floors are band lower bounds, so a declared band is at or above a floor
    exactly when every count in it is. The check does not trust the client
    guard: it re-evaluates every floor and the contract and update bindings.

    Args:
        declaration: Structurally validated client declaration.
        contract: The coordinator's own round contract.
        update_digest: Expected SHA-256 reference of the submitted update.

    Returns:
        A decision listing every failed binding and floor.

    Raises:
        FederatedContributionError: If an input has the wrong type or digest.
    """
    if type(declaration) is not FederatedContributionDeclaration:
        raise FederatedContributionError("invalid contribution declaration")
    if type(contract) is not FederatedContributionFloorPolicy:
        raise FederatedContributionError("invalid contribution contract")
    _require_digest(update_digest)
    reasons = []
    if not hmac.compare_digest(declaration.contract_digest, contract.contract_digest):
        reasons.append(FederatedContributionReasonCode.CONTRACT_MISMATCH)
    if not hmac.compare_digest(declaration.update_digest, update_digest):
        reasons.append(FederatedContributionReasonCode.UPDATE_MISMATCH)
    for band_field, floor_field, reason in zip(
        _BAND_FIELDS, _FLOOR_FIELDS, _FLOOR_REASONS, strict=True
    ):
        floor = getattr(contract, floor_field)
        if getattr(declaration, band_field).lower_bound < floor:
            reasons.append(reason)
    return FederatedContributionDecision(tuple(reasons))


def _defaults_for(mechanism: FederatedPrivacyMechanism) -> Mapping[str, int]:
    # Any mechanism that is not a known DP mechanism gets the stricter floors.
    if mechanism in _DP_MECHANISMS:
        return DEFAULT_DP_CONTRIBUTION_FLOORS
    return DEFAULT_NON_DP_CONTRIBUTION_FLOORS


def _canonical_reasons(
    value: object,
) -> tuple[FederatedContributionReasonCode, ...]:
    if type(value) is not tuple or any(
        type(code) is not FederatedContributionReasonCode for code in value
    ):
        raise FederatedContributionError("invalid contribution reason codes")
    order = [_REASON_ORDER[code] for code in value]
    if order != sorted(set(order)):
        raise FederatedContributionError("contribution reason codes are not canonical")
    return value


def _require_digest(value: object) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise FederatedContributionError("invalid SHA-256 digest reference")


def _require_fields(payload: object, expected: frozenset[str]) -> dict[str, Any]:
    if (
        type(payload) is not dict
        or len(payload) != len(expected)
        or any(type(key) is not str for key in payload)
        or payload.keys() != expected
    ):
        raise FederatedContributionError("invalid contribution fields")
    return payload


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parse_json(payload: object) -> object:
    if type(payload) is not str or len(payload) > _MAX_JSON_BYTES:
        raise FederatedContributionError("invalid contribution JSON")
    try:
        if len(payload.encode("utf-8")) > _MAX_JSON_BYTES:
            raise FederatedContributionError("invalid contribution JSON")
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_int=_parse_integer,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except (ValueError, RecursionError):
        raise FederatedContributionError("invalid contribution JSON") from None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FederatedContributionError("duplicate contribution fields")
        result[key] = value
    return result


def _parse_integer(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise FederatedContributionError("invalid contribution integer")
    parsed = int(value)
    if not -_MAX_COUNT <= parsed <= _MAX_COUNT:
        raise FederatedContributionError("invalid contribution integer")
    return parsed


def _reject_number(value: str) -> None:
    raise FederatedContributionError("non-integer contribution number")


__all__ = [
    "DEFAULT_DP_CONTRIBUTION_FLOORS",
    "DEFAULT_NON_DP_CONTRIBUTION_FLOORS",
    "FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION",
    "FEDERATED_CONTRIBUTION_SCHEMA_VERSION",
    "MAX_FEDERATED_CONTRIBUTION_FLOOR",
    "FederatedContributionBand",
    "FederatedContributionDeclaration",
    "FederatedContributionDecision",
    "FederatedContributionError",
    "FederatedContributionFloorPolicy",
    "FederatedContributionReasonCode",
    "FederatedContributionRefusedError",
    "check_federated_contribution",
    "seal_federated_contribution",
]
