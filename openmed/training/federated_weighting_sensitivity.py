"""Check federated site weighting against declared DP sensitivity bounds.

A client-level differentially private weighted mean is only as private as the
largest influence one client can have on it. For clipped updates ``x_i`` with
``||x_i|| <= C`` and weights ``0 <= w_i <= w_max`` divided by a fixed,
data-independent denominator ``W``, adding or removing one client moves the
estimate by at most ``C * w_max / W``; replacing one client moves it by at most
``2 * C * w_max / W``. Weights that follow private record counts, floors that
lift a client above the cap, or a denominator equal to the realized total
weight break that bound.

This module compares metadata only: a weighting-policy declaration, the update
mechanism declaration, and the accountant declaration. It never reads tensors,
site identities, or record counts, and it does not apply noise or compute
weights. Arithmetic is exact with :class:`fractions.Fraction`. Results carry
stable reason codes and a value-free compatibility receipt that binds the
three declaration digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import islice
from types import MappingProxyType
from typing import Any, Final

FEDERATED_WEIGHTING_SENSITIVITY_SCHEMA_VERSION: Final = (
    "openmed.training.federated_weighting_sensitivity.v1"
)

REASON_CODES: Final = (
    "weighting_count_proportional",
    "weighting_exact_count_input",
    "weighting_undeclared_band",
    "weighting_uncapped",
    "weighting_floor_exceeds_cap",
    "weighting_data_dependent_normalization",
    "mechanism_unsupported",
    "mechanism_privacy_unit_unsupported",
    "accountant_privacy_unit_mismatch",
    "accountant_adjacency_mismatch",
    "mechanism_sensitivity_too_small",
    "accountant_noise_multiplier_overstated",
)

_WEIGHT_SOURCES: Final = frozenset({"declared_band", "record_count"})
_NORMALIZATIONS: Final = frozenset({"fixed_denominator", "realized_sum"})
_MECHANISMS: Final = frozenset({"gaussian", "laplace"})
_PRIVACY_UNITS: Final = frozenset({"client", "record"})
_ADJACENCY_FACTORS: Final = MappingProxyType({"add_remove": 1, "replace_one": 2})
_CODE: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_MAX_BANDS: Final = 64
_MAX_WEIGHT_INPUTS: Final = 4096
_MAX_MAGNITUDE: Final = 10**9
_MAX_DENOMINATOR: Final = 10**12
_DOMAIN: Final = "openmed.training.federated_weighting_sensitivity.v1"


class WeightingSensitivityError(ValueError):
    """Raised with a fixed code, and at most a field name, never a value.

    Attributes:
        code: Stable error or reason code.
        reason_codes: Every incompatibility reason when raised by
            :func:`require_weighting_sensitivity`; otherwise empty.
    """

    def __init__(
        self,
        code: str,
        field_name: str | None = None,
        *,
        reason_codes: tuple[str, ...] = (),
    ) -> None:
        self.code = code
        self.reason_codes = reason_codes
        super().__init__(code if field_name is None else f"{code}: {field_name}")


def _exact(
    value: object,
    *,
    code: str,
    field_name: str,
    allow_zero: bool = False,
) -> Fraction:
    """Read an int, Fraction or finite float as a bounded exact rational."""
    if type(value) is int or type(value) is Fraction:
        result = Fraction(value)
    elif type(value) is float and math.isfinite(value):
        # The shortest decimal repr is the declared value: 0.1 means 1/10.
        result = Fraction(repr(value))
    else:
        raise WeightingSensitivityError(code, field_name)
    lower_ok = result >= 0 if allow_zero else result > 0
    if not lower_ok or result > _MAX_MAGNITUDE or result.denominator > _MAX_DENOMINATOR:
        raise WeightingSensitivityError(code, field_name)
    return result


def _optional_exact(
    value: object, *, code: str, field_name: str, allow_zero: bool = False
) -> Fraction | None:
    if value is None:
        return None
    return _exact(value, code=code, field_name=field_name, allow_zero=allow_zero)


def _choice(value: object, allowed: frozenset[str], *, code: str, name: str) -> str:
    if type(value) is not str or value not in allowed:
        raise WeightingSensitivityError(code, name)
    return value


def _controlled_code(value: object, *, code: str, name: str) -> str:
    if type(value) is not str or _CODE.fullmatch(value) is None:
        raise WeightingSensitivityError(code, name)
    return value


def _exact_mapping(
    value: object, *, code: str, name: str, allow_zero: bool
) -> Mapping[str, Fraction]:
    if not isinstance(value, Mapping) or len(value) > _MAX_BANDS:
        raise WeightingSensitivityError(code, name)
    result = {
        _controlled_code(key, code=code, name=name): _exact(
            item, code=code, field_name=name, allow_zero=allow_zero
        )
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def _ratio(value: Fraction | None) -> list[int] | None:
    return None if value is None else [value.numerator, value.denominator]


def _digest(kind: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    prefix = f"{_DOMAIN}.{kind}\0".encode("ascii")
    return "sha256:" + hashlib.sha256(prefix + canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class WeightingPolicyDeclaration:
    """Declared site-weighting envelope for one DP aggregation round.

    Numeric fields are hidden from ``repr``. Weights are relative factors in
    the same unit as ``fixed_denominator``; they are never record counts.

    Args:
        weight_source: ``declared_band`` for coarse declared volume bands, or
            ``record_count`` for count-proportional weights. The latter is
            accepted only so the guard can reject it with a stable code.
        band_weights: Controlled band code to positive declared weight. Must
            be non-empty for ``declared_band`` and empty for ``record_count``.
        weight_cap: Positive maximum weight any client may receive, or
            ``None`` when the policy is uncapped.
        weight_floor: Optional non-negative minimum weight for every client.
        protected_slice_floors: Controlled protected-slice code to the
            non-negative minimum weight a client in that slice receives.
        normalization: ``fixed_denominator`` for a declared, data-independent
            denominator, or ``realized_sum`` for the realized total weight.
        fixed_denominator: Positive denominator, required exactly when
            ``normalization`` is ``fixed_denominator``.

    Raises:
        WeightingSensitivityError: With code ``invalid_weighting_policy`` and
            the offending field name when the declaration is malformed.
    """

    weight_source: str = "declared_band"
    band_weights: Mapping[str, Fraction] = field(default_factory=dict, repr=False)
    weight_cap: Fraction | None = field(default=None, repr=False)
    weight_floor: Fraction | None = field(default=None, repr=False)
    protected_slice_floors: Mapping[str, Fraction] = field(
        default_factory=dict, repr=False
    )
    normalization: str = "fixed_denominator"
    fixed_denominator: Fraction | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        code = "invalid_weighting_policy"
        source = _choice(
            self.weight_source, _WEIGHT_SOURCES, code=code, name="weight_source"
        )
        bands = _exact_mapping(
            self.band_weights, code=code, name="band_weights", allow_zero=False
        )
        if (source == "declared_band") != bool(bands):
            raise WeightingSensitivityError(code, "band_weights")
        floors = _exact_mapping(
            self.protected_slice_floors,
            code=code,
            name="protected_slice_floors",
            allow_zero=True,
        )
        normalization = _choice(
            self.normalization, _NORMALIZATIONS, code=code, name="normalization"
        )
        denominator = _optional_exact(
            self.fixed_denominator, code=code, field_name="fixed_denominator"
        )
        if (normalization == "fixed_denominator") != (denominator is not None):
            raise WeightingSensitivityError(code, "fixed_denominator")
        object.__setattr__(self, "band_weights", bands)
        object.__setattr__(
            self,
            "weight_cap",
            _optional_exact(self.weight_cap, code=code, field_name="weight_cap"),
        )
        object.__setattr__(
            self,
            "weight_floor",
            _optional_exact(
                self.weight_floor,
                code=code,
                field_name="weight_floor",
                allow_zero=True,
            ),
        )
        object.__setattr__(self, "protected_slice_floors", floors)
        object.__setattr__(self, "fixed_denominator", denominator)

    def max_weight(self) -> Fraction | None:
        """Return the largest weight any client can receive under the policy.

        Each client's weight is ``max(floors, min(band weight, cap))``, so the
        maximum is the larger of the largest capped band weight and every
        floor. Uncapped count-proportional weights have no declared maximum.

        Returns:
            The exact maximum weight, or ``None`` when it is unbounded.
        """
        if self.weight_source == "record_count":
            if self.weight_cap is None:
                return None
            ceiling = self.weight_cap
        else:
            ceiling = max(
                weight if self.weight_cap is None else min(weight, self.weight_cap)
                for weight in self.band_weights.values()
            )
        floors = [*self.protected_slice_floors.values()]
        if self.weight_floor is not None:
            floors.append(self.weight_floor)
        return max([ceiling, *floors])

    def digest(self) -> str:
        """Return a domain-separated SHA-256 reference for the declaration."""
        return _digest(
            "weighting_policy",
            {
                "weight_source": self.weight_source,
                "band_weights": {
                    key: _ratio(value) for key, value in self.band_weights.items()
                },
                "weight_cap": _ratio(self.weight_cap),
                "weight_floor": _ratio(self.weight_floor),
                "protected_slice_floors": {
                    key: _ratio(value)
                    for key, value in self.protected_slice_floors.items()
                },
                "normalization": self.normalization,
                "fixed_denominator": _ratio(self.fixed_denominator),
            },
        )


@dataclass(frozen=True, slots=True)
class UpdateMechanismDeclaration:
    """Declared noise mechanism applied to the normalized weighted aggregate.

    The mechanism adds noise with standard deviation
    ``noise_multiplier * sensitivity`` to the aggregate after division by the
    weighting denominator.

    Args:
        mechanism: ``gaussian``; ``laplace`` is parsed but unsupported because
            this guard derives L2 sensitivity.
        privacy_unit: ``client``; ``record`` is parsed but unsupported because
            site weighting is a client-level concern.
        adjacency: ``add_remove`` or ``replace_one`` one client.
        clip_norm: Positive L2 clipping norm applied to every client update.
        sensitivity: Positive L2 sensitivity the noise was calibrated for.
        noise_multiplier: Positive ratio of noise scale to ``sensitivity``.

    Raises:
        WeightingSensitivityError: With code ``invalid_mechanism_declaration``
            and the offending field name when the declaration is malformed.
    """

    mechanism: str
    privacy_unit: str
    adjacency: str
    clip_norm: Fraction
    sensitivity: Fraction
    noise_multiplier: Fraction

    def __post_init__(self) -> None:
        code = "invalid_mechanism_declaration"
        _choice(self.mechanism, _MECHANISMS, code=code, name="mechanism")
        _choice(self.privacy_unit, _PRIVACY_UNITS, code=code, name="privacy_unit")
        _choice(
            self.adjacency, frozenset(_ADJACENCY_FACTORS), code=code, name="adjacency"
        )
        for name in ("clip_norm", "sensitivity", "noise_multiplier"):
            value = _exact(getattr(self, name), code=code, field_name=name)
            object.__setattr__(self, name, value)

    def digest(self) -> str:
        """Return a domain-separated SHA-256 reference for the declaration."""
        return _digest(
            "mechanism",
            {
                "mechanism": self.mechanism,
                "privacy_unit": self.privacy_unit,
                "adjacency": self.adjacency,
                "clip_norm": _ratio(self.clip_norm),
                "sensitivity": _ratio(self.sensitivity),
                "noise_multiplier": _ratio(self.noise_multiplier),
            },
        )


@dataclass(frozen=True, slots=True)
class AccountantDeclaration:
    """Declared privacy-accountant inputs for the same aggregation round.

    Args:
        accountant: Controlled lowercase accountant code, such as ``rdp``.
        privacy_unit: ``client`` or ``record``; must match the mechanism.
        adjacency: ``add_remove`` or ``replace_one``; must match the mechanism.
        noise_multiplier: Positive noise-to-sensitivity ratio the accountant
            consumes. It must not exceed the effective multiplier.

    Raises:
        WeightingSensitivityError: With code ``invalid_accountant_declaration``
            and the offending field name when the declaration is malformed.
    """

    accountant: str
    privacy_unit: str
    adjacency: str
    noise_multiplier: Fraction

    def __post_init__(self) -> None:
        code = "invalid_accountant_declaration"
        _controlled_code(self.accountant, code=code, name="accountant")
        _choice(self.privacy_unit, _PRIVACY_UNITS, code=code, name="privacy_unit")
        _choice(
            self.adjacency, frozenset(_ADJACENCY_FACTORS), code=code, name="adjacency"
        )
        object.__setattr__(
            self,
            "noise_multiplier",
            _exact(self.noise_multiplier, code=code, field_name="noise_multiplier"),
        )

    def digest(self) -> str:
        """Return a domain-separated SHA-256 reference for the declaration."""
        return _digest(
            "accountant",
            {
                "accountant": self.accountant,
                "privacy_unit": self.privacy_unit,
                "adjacency": self.adjacency,
                "noise_multiplier": _ratio(self.noise_multiplier),
            },
        )


@dataclass(frozen=True, slots=True)
class WeightingCompatibilityReceipt:
    """Value-free evidence binding the three declaration digests.

    The receipt contains no site identity, record count, participation count,
    band assignment, or per-site weight; only codes and digests.

    Args:
        outcome: ``compatible`` or ``incompatible``.
        reason_codes: Ordered incompatibility codes from ``REASON_CODES``.
        weighting_policy_digest: Digest of the weighting-policy declaration.
        mechanism_digest: Digest of the mechanism declaration.
        accountant_digest: Digest of the accountant declaration.
        normalization: Declared normalization rule code.
        adjacency: Declared mechanism adjacency code.
        schema_version: Receipt schema identifier.
    """

    outcome: str
    reason_codes: tuple[str, ...]
    weighting_policy_digest: str
    mechanism_digest: str
    accountant_digest: str
    normalization: str
    adjacency: str
    schema_version: str = FEDERATED_WEIGHTING_SENSITIVITY_SCHEMA_VERSION

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "weighting_policy_digest": self.weighting_policy_digest,
            "mechanism_digest": self.mechanism_digest,
            "accountant_digest": self.accountant_digest,
            "normalization": self.normalization,
            "adjacency": self.adjacency,
        }

    @property
    def receipt_digest(self) -> str:
        """Return the digest binding every other receipt field."""
        return _digest("receipt", self._body())

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping including the binding ``receipt_digest``."""
        return {**self._body(), "receipt_digest": self.receipt_digest}

    def to_json(self) -> str:
        """Return byte-stable ASCII JSON with sorted keys and a newline."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class WeightingSensitivityAssessment:
    """Result of comparing a weighting policy with its DP declarations.

    Args:
        compatible: Whether no incompatibility reason was found.
        reason_codes: Ordered incompatibility codes from ``REASON_CODES``.
        receipt: Value-free compatibility receipt.
        effective_sensitivity: Derived per-client L2 sensitivity, or ``None``
            when the declaration does not bound it. Hidden from ``repr``.
        effective_noise_multiplier: Applied noise scale divided by the
            effective sensitivity, or ``None``. Hidden from ``repr``.
    """

    compatible: bool
    reason_codes: tuple[str, ...]
    receipt: WeightingCompatibilityReceipt
    effective_sensitivity: Fraction | None = field(default=None, repr=False)
    effective_noise_multiplier: Fraction | None = field(default=None, repr=False)


def weighted_sum_sensitivity(
    clip_norm: int | float | Fraction,
    max_weight: int | float | Fraction,
    fixed_denominator: int | float | Fraction,
    *,
    adjacency: str = "add_remove",
) -> Fraction:
    """Return the exact L2 sensitivity of a capped, fixed-denominator mean.

    For ``sum_i w_i * x_i / W`` with ``||x_i|| <= C`` and ``w_i <= w_max``,
    adding or removing one client changes the result by at most
    ``C * w_max / W``; replacing one client by at most ``2 * C * w_max / W``.

    Args:
        clip_norm: Positive L2 clipping norm ``C``.
        max_weight: Positive maximum client weight ``w_max``.
        fixed_denominator: Positive data-independent denominator ``W``.
        adjacency: ``add_remove`` or ``replace_one``.

    Returns:
        The exact sensitivity as a :class:`fractions.Fraction`.

    Raises:
        WeightingSensitivityError: With code ``invalid_sensitivity_input``.

    Examples:
        >>> weighted_sum_sensitivity(1, 4, 40)
        Fraction(1, 10)
        >>> weighted_sum_sensitivity(1, 4, 40, adjacency="replace_one")
        Fraction(1, 5)
    """
    code = "invalid_sensitivity_input"
    factor = _ADJACENCY_FACTORS.get(adjacency) if type(adjacency) is str else None
    if factor is None:
        raise WeightingSensitivityError(code, "adjacency")
    clip = _exact(clip_norm, code=code, field_name="clip_norm")
    weight = _exact(max_weight, code=code, field_name="max_weight")
    denominator = _exact(fixed_denominator, code=code, field_name="fixed_denominator")
    return factor * clip * weight / denominator


def derive_effective_sensitivity(
    policy: WeightingPolicyDeclaration, mechanism: UpdateMechanismDeclaration
) -> Fraction | None:
    """Derive per-client sensitivity from clipping, weights and normalization.

    Args:
        policy: Weighting-policy declaration.
        mechanism: Mechanism declaration supplying clip norm and adjacency.

    Returns:
        The exact sensitivity, or ``None`` when realized-sum normalization or
        uncapped count weights leave it unbounded by the declaration.

    Raises:
        WeightingSensitivityError: With code ``invalid_declaration`` when an
            argument has the wrong type.
    """
    _require_types(policy, mechanism)
    return _effective_sensitivity(policy, mechanism)


def check_weighting_sensitivity(
    policy: WeightingPolicyDeclaration,
    mechanism: UpdateMechanismDeclaration,
    accountant: AccountantDeclaration,
    *,
    weight_inputs: Iterable[str] | None = None,
) -> WeightingSensitivityAssessment:
    """Compare a DP round's weighting policy with its mechanism and accountant.

    The check requires capped weights derived from declared bands and a fixed
    denominator, rejects count-proportional weights, exact counts supplied as
    weight inputs, floors above the cap and realized-sum normalization, and
    requires the mechanism calibration and accountant multiplier to cover the
    derived sensitivity.

    Args:
        policy: Weighting-policy declaration.
        mechanism: Update mechanism declaration.
        accountant: Accountant declaration.
        weight_inputs: Optional per-client weight inputs, each a declared band
            code. Numbers are treated as exact counts and rejected. The inputs
            are checked locally and never enter the receipt.

    Returns:
        The assessment with ordered reason codes and a value-free receipt.

    Raises:
        WeightingSensitivityError: With code ``invalid_declaration`` or
            ``invalid_weight_inputs`` for malformed arguments.
    """
    _require_types(policy, mechanism)
    if type(accountant) is not AccountantDeclaration:
        raise WeightingSensitivityError("invalid_declaration", "accountant")
    reasons: set[str] = set()
    if policy.weight_source == "record_count":
        reasons.add("weighting_count_proportional")
    reasons.update(_input_reasons(policy, weight_inputs))
    if policy.weight_cap is None:
        reasons.add("weighting_uncapped")
    else:
        floors = [*policy.protected_slice_floors.values()]
        if policy.weight_floor is not None:
            floors.append(policy.weight_floor)
        if any(floor > policy.weight_cap for floor in floors):
            reasons.add("weighting_floor_exceeds_cap")
    if policy.normalization != "fixed_denominator":
        reasons.add("weighting_data_dependent_normalization")
    if mechanism.mechanism != "gaussian":
        reasons.add("mechanism_unsupported")
    if mechanism.privacy_unit != "client":
        reasons.add("mechanism_privacy_unit_unsupported")
    if accountant.privacy_unit != mechanism.privacy_unit:
        reasons.add("accountant_privacy_unit_mismatch")
    if accountant.adjacency != mechanism.adjacency:
        reasons.add("accountant_adjacency_mismatch")
    sensitivity = _effective_sensitivity(policy, mechanism)
    multiplier = None
    if sensitivity is not None:
        if mechanism.sensitivity < sensitivity:
            reasons.add("mechanism_sensitivity_too_small")
        noise_scale = mechanism.noise_multiplier * mechanism.sensitivity
        multiplier = noise_scale / sensitivity
        if accountant.noise_multiplier > multiplier:
            reasons.add("accountant_noise_multiplier_overstated")
    ordered = tuple(code for code in REASON_CODES if code in reasons)
    receipt = WeightingCompatibilityReceipt(
        outcome="incompatible" if ordered else "compatible",
        reason_codes=ordered,
        weighting_policy_digest=policy.digest(),
        mechanism_digest=mechanism.digest(),
        accountant_digest=accountant.digest(),
        normalization=policy.normalization,
        adjacency=mechanism.adjacency,
    )
    return WeightingSensitivityAssessment(
        compatible=not ordered,
        reason_codes=ordered,
        receipt=receipt,
        effective_sensitivity=sensitivity,
        effective_noise_multiplier=multiplier,
    )


def require_weighting_sensitivity(
    policy: WeightingPolicyDeclaration,
    mechanism: UpdateMechanismDeclaration,
    accountant: AccountantDeclaration,
    *,
    weight_inputs: Iterable[str] | None = None,
) -> WeightingCompatibilityReceipt:
    """Return the receipt for a compatible round, or fail closed.

    Args:
        policy: Weighting-policy declaration.
        mechanism: Update mechanism declaration.
        accountant: Accountant declaration.
        weight_inputs: Optional per-client declared band codes.

    Returns:
        The value-free compatibility receipt.

    Raises:
        WeightingSensitivityError: With the first reason code as ``code`` and
            every reason in ``reason_codes`` when the round is incompatible.
    """
    assessment = check_weighting_sensitivity(
        policy, mechanism, accountant, weight_inputs=weight_inputs
    )
    if not assessment.compatible:
        raise WeightingSensitivityError(
            assessment.reason_codes[0], reason_codes=assessment.reason_codes
        )
    return assessment.receipt


def _require_types(policy: object, mechanism: object) -> None:
    if type(policy) is not WeightingPolicyDeclaration:
        raise WeightingSensitivityError("invalid_declaration", "policy")
    if type(mechanism) is not UpdateMechanismDeclaration:
        raise WeightingSensitivityError("invalid_declaration", "mechanism")


def _effective_sensitivity(
    policy: WeightingPolicyDeclaration, mechanism: UpdateMechanismDeclaration
) -> Fraction | None:
    weight = policy.max_weight()
    if weight is None or policy.fixed_denominator is None:
        return None
    return weighted_sum_sensitivity(
        mechanism.clip_norm,
        weight,
        policy.fixed_denominator,
        adjacency=mechanism.adjacency,
    )


def _input_reasons(
    policy: WeightingPolicyDeclaration, weight_inputs: object
) -> set[str]:
    if weight_inputs is None:
        return set()
    if isinstance(weight_inputs, (str, bytes, Mapping)) or not isinstance(
        weight_inputs, Iterable
    ):
        raise WeightingSensitivityError("invalid_weight_inputs")
    try:
        inputs = tuple(islice(weight_inputs, _MAX_WEIGHT_INPUTS + 1))
    except Exception:
        raise WeightingSensitivityError("invalid_weight_inputs") from None
    if len(inputs) > _MAX_WEIGHT_INPUTS:
        raise WeightingSensitivityError("invalid_weight_inputs")
    reasons: set[str] = set()
    for item in inputs:
        if isinstance(item, numbers.Number):
            reasons.add("weighting_exact_count_input")
        elif type(item) is str:
            if item not in policy.band_weights:
                reasons.add("weighting_undeclared_band")
        else:
            raise WeightingSensitivityError("invalid_weight_inputs")
    return reasons


__all__ = [
    "FEDERATED_WEIGHTING_SENSITIVITY_SCHEMA_VERSION",
    "REASON_CODES",
    "AccountantDeclaration",
    "UpdateMechanismDeclaration",
    "WeightingCompatibilityReceipt",
    "WeightingPolicyDeclaration",
    "WeightingSensitivityAssessment",
    "WeightingSensitivityError",
    "check_weighting_sensitivity",
    "derive_effective_sensitivity",
    "require_weighting_sensitivity",
    "weighted_sum_sensitivity",
]
