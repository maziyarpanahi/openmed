"""Deterministic aggregation of federated LoRA factor updates.

Averaging LoRA ``A`` and ``B`` factors separately does not average the
low-rank updates they encode: ``mean(B) @ mean(A)`` differs from
``mean(B @ A)`` whenever sites diverge. This module validates paired factor
updates against a round-declared aggregation mode and combines them in the
low-rank product space:

* ``frozen_a`` pins one shared ``A`` factor per module by digest and averages
  only ``B``. Because ``A`` is shared, ``mean(B) @ A`` equals the weighted mean
  of ``B @ A`` exactly before the single final rounding to float64.
* ``product_space`` averages the products ``B @ A`` exactly and re-factorizes
  the mean to the declared rank with a deterministic Jacobi eigensolver. The
  re-factorization residual is recorded only as a value-free band.
* ``reject`` refuses every batch.

All sums use exact integer arithmetic over the binary values of the float64
inputs, and updates are processed in canonical digest order, so identical
inputs produce byte-identical aggregates regardless of arrival order. The
module is stdlib-only, operates on small plaintext matrices or an
already-unmasked sum, and never reads or writes files. Masking, fixed-point
encoding, robust strategies and site weighting policies are owned elsewhere;
weights are injected by the caller.

Errors carry one fixed reason code and never echo submitted values, module
names or weights. Aggregation evidence records only the mode, rank, rank
policy, residual band and digests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final

LORA_UPDATE_SCHEMA_VERSION: Final = "openmed.training.federated_lora_update.v1"
LORA_ROUND_SCHEMA_VERSION: Final = "openmed.training.federated_lora_round.v1"
LORA_AGGREGATION_EVIDENCE_SCHEMA_VERSION: Final = (
    "openmed.training.federated_lora_aggregation.v1"
)

LORA_AGGREGATION_MODES: Final = ("frozen_a", "product_space", "reject")
LORA_RANK_POLICIES: Final = ("uniform", "zero_pad")
LORA_RESIDUAL_BANDS: Final = (
    "zero",
    "below_1e-12",
    "below_1e-9",
    "below_1e-6",
    "below_1e-3",
    "below_1e-1",
    "at_least_1e-1",
)
LORA_RESIDUAL_BAND_BOUNDS: Final[Mapping[str, Fraction | None]] = MappingProxyType(
    {
        "zero": Fraction(0),
        "below_1e-12": Fraction(1, 10**12),
        "below_1e-9": Fraction(1, 10**9),
        "below_1e-6": Fraction(1, 10**6),
        "below_1e-3": Fraction(1, 10**3),
        "below_1e-1": Fraction(1, 10),
        "at_least_1e-1": None,
    }
)
LORA_AGGREGATION_REASON_CODES: Final = frozenset(
    {
        "lora_aggregation_rejected",
        "lora_declaration_invalid",
        "lora_duplicate_update",
        "lora_factor_unpaired",
        "lora_frozen_a_mismatch",
        "lora_limit_exceeded",
        "lora_mode_unsupported",
        "lora_module_mismatch",
        "lora_naive_factor_average_rejected",
        "lora_rank_mismatch",
        "lora_residual_exceeds_ceiling",
        "lora_scaling_mismatch",
        "lora_update_invalid",
        "lora_weight_invalid",
    }
)

_MAX_FEATURES: Final = 64
_MAX_RANK: Final = 16
_MAX_MODULES: Final = 8
_MAX_UPDATES: Final = 32
_MAX_ALPHA: Final = 1 << 20
_MAX_ABS_VALUE: Final = float(1 << 20)
_MAX_WEIGHT_PART: Final = 1 << 64
_MAX_SWEEPS: Final = 64
_JACOBI_TOLERANCE: Final = 1e-28
_NAIVE_MODES: Final = frozenset(
    {"per_factor_mean", "per_factor", "naive", "factor_mean", "fedavg_factors"}
)
_BAND_EXPONENTS: Final = (
    ("below_1e-12", 12),
    ("below_1e-9", 9),
    ("below_1e-6", 6),
    ("below_1e-3", 3),
    ("below_1e-1", 1),
)
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NAME: Final = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|[0-9]+))*\Z"
)
_FACTOR_SUFFIXES: Final = (".lora_A.weight", ".lora_B.weight")
_FACTOR_DOMAIN: Final = b"openmed.training.federated_lora_factor.v1\0"
_UPDATE_DOMAIN: Final = b"openmed.training.federated_lora_update.v1\0"
_ROUND_DOMAIN: Final = b"openmed.training.federated_lora_round.v1\0"
_INPUTS_DOMAIN: Final = b"openmed.training.federated_lora_inputs.v1\0"
_AGGREGATE_DOMAIN: Final = b"openmed.training.federated_lora_aggregate.v1\0"

Matrix = tuple[tuple[float, ...], ...]
_IntMatrix = list[list[int]]


class LoraAggregationError(ValueError):
    """Fail-closed refusal carrying one stable reason code.

    The message is exactly the reason code. Submitted values, module names,
    weights and digests are never attached.

    Attributes:
        code: One of ``LORA_AGGREGATION_REASON_CODES``.
    """

    def __init__(self, code: str) -> None:
        if code not in LORA_AGGREGATION_REASON_CODES:
            code = "lora_update_invalid"
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class LoraFactorPair:
    """Paired LoRA factors for one target module.

    The low-rank update is ``factor_b @ factor_a`` before ``alpha / rank``
    scaling. Values are normalized to float64; negative zero becomes zero.
    Factor values are excluded from ``repr``.

    Args:
        target_module: Dotted ASCII module name, at most 256 characters.
        factor_a: ``rank x in_features`` matrix as nested sequences of finite
            numbers, or an array-like object exposing ``tolist()``.
        factor_b: ``out_features x rank`` matrix in the same form.

    Raises:
        LoraAggregationError: ``lora_factor_unpaired`` when the inner
            dimensions disagree, ``lora_limit_exceeded`` past the documented
            size bounds, or ``lora_update_invalid`` for other invalid input.
    """

    target_module: str
    factor_a: Matrix = field(repr=False)
    factor_b: Matrix = field(repr=False)

    def __post_init__(self) -> None:
        _require_name(self.target_module, "lora_update_invalid")
        factor_a = _matrix(self.factor_a, _MAX_RANK, _MAX_FEATURES)
        factor_b = _matrix(self.factor_b, _MAX_FEATURES, _MAX_RANK)
        if len(factor_b[0]) != len(factor_a):
            raise LoraAggregationError("lora_factor_unpaired")
        object.__setattr__(self, "factor_a", factor_a)
        object.__setattr__(self, "factor_b", factor_b)

    @property
    def rank(self) -> int:
        """Return the shared inner dimension of the factor pair."""
        return len(self.factor_a)

    @property
    def in_features(self) -> int:
        """Return the column count of ``factor_a``."""
        return len(self.factor_a[0])

    @property
    def out_features(self) -> int:
        """Return the row count of ``factor_b``."""
        return len(self.factor_b)

    @property
    def factor_a_digest(self) -> str:
        """Return the domain-separated content digest of ``factor_a``."""
        return lora_factor_digest(self.factor_a)

    @property
    def factor_b_digest(self) -> str:
        """Return the domain-separated content digest of ``factor_b``."""
        return lora_factor_digest(self.factor_b)


@dataclass(frozen=True, slots=True, kw_only=True)
class LoraUpdate:
    """A versioned LoRA update descriptor with paired factors per module.

    Modules are stored in canonical name order. The content ``digest`` binds
    the schema version, rank, alpha, module shapes and every factor value; it
    is the key for canonical ordering and for injected weights.

    Args:
        rank: Declared LoRA rank shared by every module, from 1 to 16.
        alpha: Positive integer LoRA alpha; the update scale is
            ``alpha / rank`` as in the adapter training recipe.
        modules: One to eight factor pairs with unique target modules.
        schema_version: Exact supported update schema identifier.

    Raises:
        LoraAggregationError: ``lora_rank_mismatch`` when a pair's shape does
            not match ``rank``; ``lora_update_invalid`` or
            ``lora_limit_exceeded`` for other invalid input.
    """

    rank: int
    alpha: int
    modules: tuple[LoraFactorPair, ...]
    schema_version: str = LORA_UPDATE_SCHEMA_VERSION
    _digest: str = field(init=False, repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        if self.schema_version != LORA_UPDATE_SCHEMA_VERSION:
            raise LoraAggregationError("lora_update_invalid")
        _require_int(self.rank, 1, _MAX_RANK, "lora_update_invalid")
        _require_int(self.alpha, 1, _MAX_ALPHA, "lora_update_invalid")
        modules = self.modules
        if type(modules) is not tuple or not modules:
            raise LoraAggregationError("lora_update_invalid")
        if len(modules) > _MAX_MODULES:
            raise LoraAggregationError("lora_limit_exceeded")
        if any(type(pair) is not LoraFactorPair for pair in modules):
            raise LoraAggregationError("lora_update_invalid")
        if len({pair.target_module for pair in modules}) != len(modules):
            raise LoraAggregationError("lora_update_invalid")
        if any(pair.rank != self.rank for pair in modules):
            raise LoraAggregationError("lora_rank_mismatch")
        ordered = tuple(sorted(modules, key=lambda pair: pair.target_module))
        object.__setattr__(self, "modules", ordered)
        digest = _sha256(_UPDATE_DOMAIN, self.to_bytes())
        object.__setattr__(self, "_digest", digest)

    @property
    def digest(self) -> str:
        """Return the content digest used for canonical ordering."""
        return self._digest

    @classmethod
    def from_named_factors(
        cls, factors: Mapping[str, Any], *, rank: int, alpha: int
    ) -> LoraUpdate:
        """Pair factors named ``<module>.lora_A.weight`` / ``.lora_B.weight``.

        Args:
            factors: Mapping from parameter name to matrix. Every module must
                supply exactly one ``A`` and one ``B`` factor.
            rank: Declared LoRA rank.
            alpha: Declared LoRA alpha.

        Returns:
            A validated update with modules in canonical order.

        Raises:
            LoraAggregationError: ``lora_factor_unpaired`` when a module lacks
                its partner factor; ``lora_update_invalid`` for unrecognized
                parameter names.
        """
        if not isinstance(factors, Mapping) or not factors:
            raise LoraAggregationError("lora_update_invalid")
        if len(factors) > 2 * _MAX_MODULES:
            raise LoraAggregationError("lora_limit_exceeded")
        grouped: dict[str, dict[str, Any]] = {}
        for name, value in factors.items():
            if type(name) is not str:
                raise LoraAggregationError("lora_update_invalid")
            for role, suffix in zip("AB", _FACTOR_SUFFIXES):
                if name.endswith(suffix):
                    module = name[: -len(suffix)]
                    _require_name(module, "lora_update_invalid")
                    grouped.setdefault(module, {})[role] = value
                    break
            else:
                raise LoraAggregationError("lora_update_invalid")
        if any(len(roles) != 2 for roles in grouped.values()):
            raise LoraAggregationError("lora_factor_unpaired")
        pairs = tuple(
            LoraFactorPair(module, factor_a=roles["A"], factor_b=roles["B"])
            for module, roles in grouped.items()
        )
        return cls(rank=rank, alpha=alpha, modules=pairs)

    def descriptor(self) -> dict[str, Any]:
        """Return value-free update metadata with per-factor digests.

        Returns:
            A fresh mapping with the schema version, rank, alpha, update
            digest and, per module, the name, feature dimensions and factor
            digests. No factor values are included.
        """
        return {
            "schema_version": self.schema_version,
            "rank": self.rank,
            "alpha": self.alpha,
            "update_digest": self.digest,
            "modules": [
                {
                    "target_module": pair.target_module,
                    "in_features": pair.in_features,
                    "out_features": pair.out_features,
                    "factor_a_digest": pair.factor_a_digest,
                    "factor_b_digest": pair.factor_b_digest,
                }
                for pair in self.modules
            ],
        }

    def to_bytes(self) -> bytes:
        """Return the canonical encoding of the update.

        The encoding is a 4-byte big-endian header length, a compact
        sorted-key JSON header with the schema version, rank, alpha and
        module shapes, then each module's ``A`` and ``B`` factors as
        row-major little-endian float64 values in canonical module order.
        """
        header = {
            "schema_version": self.schema_version,
            "rank": self.rank,
            "alpha": self.alpha,
            "modules": [
                {
                    "target_module": pair.target_module,
                    "in_features": pair.in_features,
                    "out_features": pair.out_features,
                }
                for pair in self.modules
            ],
        }
        encoded = _canonical_json(header)
        parts = [len(encoded).to_bytes(4, "big"), encoded]
        for pair in self.modules:
            parts.append(_matrix_bytes(pair.factor_a))
            parts.append(_matrix_bytes(pair.factor_b))
        return b"".join(parts)


@dataclass(frozen=True, slots=True)
class LoraModuleSpec:
    """Coordinator-declared shape of one LoRA target module.

    Args:
        target_module: Dotted ASCII module name.
        in_features: Column count of ``A``, from 1 to 64.
        out_features: Row count of ``B``, from 1 to 64.
        frozen_a_digest: Pinned ``lora_factor_digest`` of the shared ``A``
            factor. Required in ``frozen_a`` mode and refused otherwise.
    """

    target_module: str
    in_features: int
    out_features: int
    frozen_a_digest: str | None = None

    def __post_init__(self) -> None:
        _require_name(self.target_module, "lora_declaration_invalid")
        _require_int(self.in_features, 1, _MAX_FEATURES, "lora_declaration_invalid")
        _require_int(self.out_features, 1, _MAX_FEATURES, "lora_declaration_invalid")
        if self.frozen_a_digest is not None and (
            type(self.frozen_a_digest) is not str
            or _DIGEST.fullmatch(self.frozen_a_digest) is None
        ):
            raise LoraAggregationError("lora_declaration_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping of the declared module shape."""
        return {
            "target_module": self.target_module,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "frozen_a_digest": self.frozen_a_digest,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class LoraRoundDeclaration:
    """Round-declared LoRA aggregation contract owned by the coordinator.

    Args:
        mode: ``frozen_a``, ``product_space`` or ``reject``. Naive per-factor
            averaging names are refused with
            ``lora_naive_factor_average_rejected``.
        rank: Aggregate LoRA rank, at most each module's smaller dimension.
        alpha: LoRA alpha every update must declare.
        modules: Complete set of expected target modules.
        rank_policy: ``uniform`` requires every update to use ``rank``.
            ``zero_pad`` (``product_space`` only) admits updates of rank at
            most ``rank``; see the module documentation for its semantics.
        residual_ceiling: Optional residual band; aggregation fails with
            ``lora_residual_exceeds_ceiling`` when the observed band is
            coarser. ``at_least_1e-1`` is not a valid ceiling.
        schema_version: Exact supported round schema identifier.

    Raises:
        LoraAggregationError: With a stable reason code for any invalid
            declaration.
    """

    mode: str
    rank: int
    alpha: int
    modules: tuple[LoraModuleSpec, ...]
    rank_policy: str = "uniform"
    residual_ceiling: str | None = None
    schema_version: str = LORA_ROUND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LORA_ROUND_SCHEMA_VERSION:
            raise LoraAggregationError("lora_declaration_invalid")
        if type(self.mode) is not str:
            raise LoraAggregationError("lora_mode_unsupported")
        if self.mode in _NAIVE_MODES:
            raise LoraAggregationError("lora_naive_factor_average_rejected")
        if self.mode not in LORA_AGGREGATION_MODES:
            raise LoraAggregationError("lora_mode_unsupported")
        _require_int(self.rank, 1, _MAX_RANK, "lora_declaration_invalid")
        _require_int(self.alpha, 1, _MAX_ALPHA, "lora_declaration_invalid")
        modules = self.modules
        if (
            type(modules) is not tuple
            or not 1 <= len(modules) <= _MAX_MODULES
            or any(type(spec) is not LoraModuleSpec for spec in modules)
            or len({spec.target_module for spec in modules}) != len(modules)
        ):
            raise LoraAggregationError("lora_declaration_invalid")
        for spec in modules:
            if self.rank > min(spec.in_features, spec.out_features):
                raise LoraAggregationError("lora_declaration_invalid")
            pinned = spec.frozen_a_digest is not None
            if pinned != (self.mode == "frozen_a"):
                raise LoraAggregationError("lora_declaration_invalid")
        if self.rank_policy not in LORA_RANK_POLICIES or (
            self.rank_policy == "zero_pad" and self.mode != "product_space"
        ):
            raise LoraAggregationError("lora_declaration_invalid")
        if self.residual_ceiling is not None and (
            self.residual_ceiling not in LORA_RESIDUAL_BANDS[:-1]
        ):
            raise LoraAggregationError("lora_declaration_invalid")
        ordered = tuple(sorted(modules, key=lambda spec: spec.target_module))
        object.__setattr__(self, "modules", ordered)

    @property
    def digest(self) -> str:
        """Return the domain-separated digest of the canonical declaration."""
        return _sha256(_ROUND_DOMAIN, _canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh versioned mapping with modules in canonical order."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "rank": self.rank,
            "alpha": self.alpha,
            "rank_policy": self.rank_policy,
            "residual_ceiling": self.residual_ceiling,
            "modules": [spec.to_dict() for spec in self.modules],
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class LoraAggregationEvidence:
    """Value-free record of one LoRA aggregation.

    Args:
        mode: Declared aggregation mode.
        rank: Aggregate LoRA rank.
        rank_policy: Declared rank policy (``uniform`` or ``zero_pad``).
        residual_band: Coarsest per-module relative Frobenius residual band.
        declaration_digest: Digest of the round declaration.
        inputs_digest: Digest over the sorted update digests; individual
            update digests are not recorded.
        aggregate_digest: Digest of the canonical aggregate bytes.
        schema_version: Evidence schema identifier.
    """

    mode: str
    rank: int
    rank_policy: str
    residual_band: str
    declaration_digest: str
    inputs_digest: str
    aggregate_digest: str
    schema_version: str = LORA_AGGREGATION_EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping with only codes, the rank and digests."""
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "rank": self.rank,
            "rank_policy": self.rank_policy,
            "residual_band": self.residual_band,
            "declaration_digest": self.declaration_digest,
            "inputs_digest": self.inputs_digest,
            "aggregate_digest": self.aggregate_digest,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with sorted keys and a trailing newline."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class LoraAggregate:
    """Aggregated LoRA factors at the declared rank plus value-free evidence.

    Args:
        update: Aggregate factors as a ``LoraUpdate`` at the round's rank and
            alpha. Factor values are excluded from ``repr``.
        evidence: Value-free aggregation evidence.
    """

    update: LoraUpdate
    evidence: LoraAggregationEvidence

    @property
    def digest(self) -> str:
        """Return the domain-separated digest of ``to_bytes()``."""
        return self.evidence.aggregate_digest

    def to_bytes(self) -> bytes:
        """Return the canonical aggregate encoding (see ``LoraUpdate``)."""
        return self.update.to_bytes()


def lora_factor_digest(matrix: Any) -> str:
    """Return the domain-separated content digest of one factor matrix.

    Coordinators use this to pin the shared ``A`` factor in ``frozen_a``
    rounds. The digest binds the matrix shape and its float64 values.

    Args:
        matrix: Nested sequences of finite numbers or an array-like object
            exposing ``tolist()``; at most 64 rows and 64 columns.

    Returns:
        A lowercase ``sha256:`` reference.

    Raises:
        LoraAggregationError: For invalid or oversized matrices.
    """
    values = _matrix(matrix, _MAX_FEATURES, _MAX_FEATURES)
    shape = len(values).to_bytes(4, "big") + len(values[0]).to_bytes(4, "big")
    return _sha256(_FACTOR_DOMAIN, shape + _matrix_bytes(values))


def aggregate_lora_updates(
    updates: Sequence[LoraUpdate],
    declaration: LoraRoundDeclaration,
    *,
    weights: Mapping[str, int | float | Fraction] | None = None,
) -> LoraAggregate:
    """Aggregate paired LoRA updates under a round-declared mode.

    The whole batch is validated before any arithmetic. Updates are sorted by
    content digest and combined with exact integer arithmetic; aggregate
    values are rounded to float64 once, so any arrival order yields
    byte-identical results.

    Args:
        updates: One to 32 distinct validated updates.
        declaration: Coordinator-owned round declaration.
        weights: Optional positive weights keyed by ``LoraUpdate.digest``,
            covering exactly the submitted updates. Defaults to uniform.
            Integers and fractions are limited to 64-bit parts.

    Returns:
        Aggregate factors and value-free evidence.

    Raises:
        LoraAggregationError: With a stable reason code; the batch is refused
            as a whole and no partial aggregate is returned.
    """
    if type(declaration) is not LoraRoundDeclaration:
        raise LoraAggregationError("lora_declaration_invalid")
    if declaration.mode == "reject":
        raise LoraAggregationError("lora_aggregation_rejected")
    if (
        not isinstance(updates, Sequence)
        or isinstance(updates, (str, bytes))
        or not updates
        or any(type(update) is not LoraUpdate for update in updates)
    ):
        raise LoraAggregationError("lora_update_invalid")
    if len(updates) > _MAX_UPDATES:
        raise LoraAggregationError("lora_limit_exceeded")
    ordered = sorted(updates, key=lambda update: update.digest)
    digests = [update.digest for update in ordered]
    if len(set(digests)) != len(digests):
        raise LoraAggregationError("lora_duplicate_update")
    for update in ordered:
        _check_update(update, declaration)
    coefficients = _coefficients(ordered, declaration, weights)

    pairs = []
    worst = 0
    for index, spec in enumerate(declaration.modules):
        factors = [update.modules[index] for update in ordered]
        if declaration.mode == "frozen_a":
            pair, band = _aggregate_frozen_a(spec, factors, coefficients)
        else:
            pair, band = _aggregate_product_space(
                spec, factors, coefficients, declaration.rank
            )
        pairs.append(pair)
        worst = max(worst, LORA_RESIDUAL_BANDS.index(band))
    residual_band = LORA_RESIDUAL_BANDS[worst]
    ceiling = declaration.residual_ceiling
    if ceiling is not None and worst > LORA_RESIDUAL_BANDS.index(ceiling):
        raise LoraAggregationError("lora_residual_exceeds_ceiling")

    update = LoraUpdate(
        rank=declaration.rank, alpha=declaration.alpha, modules=tuple(pairs)
    )
    evidence = LoraAggregationEvidence(
        mode=declaration.mode,
        rank=declaration.rank,
        rank_policy=declaration.rank_policy,
        residual_band=residual_band,
        declaration_digest=declaration.digest,
        inputs_digest=_sha256(_INPUTS_DOMAIN, "\n".join(digests).encode("ascii")),
        aggregate_digest=_sha256(_AGGREGATE_DOMAIN, update.to_bytes()),
    )
    return LoraAggregate(update=update, evidence=evidence)


def _check_update(update: LoraUpdate, declaration: LoraRoundDeclaration) -> None:
    if len(update.modules) != len(declaration.modules):
        raise LoraAggregationError("lora_module_mismatch")
    for pair, spec in zip(update.modules, declaration.modules):
        if (
            pair.target_module != spec.target_module
            or pair.in_features != spec.in_features
            or pair.out_features != spec.out_features
        ):
            raise LoraAggregationError("lora_module_mismatch")
    if update.alpha != declaration.alpha:
        raise LoraAggregationError("lora_scaling_mismatch")
    if declaration.rank_policy == "zero_pad":
        if update.rank > declaration.rank:
            raise LoraAggregationError("lora_rank_mismatch")
    elif update.rank != declaration.rank:
        raise LoraAggregationError("lora_rank_mismatch")
    if declaration.mode == "frozen_a":
        for pair, spec in zip(update.modules, declaration.modules):
            pinned = spec.frozen_a_digest or ""
            if not hmac.compare_digest(pair.factor_a_digest, pinned):
                raise LoraAggregationError("lora_frozen_a_mismatch")


def _coefficients(
    ordered: list[LoraUpdate],
    declaration: LoraRoundDeclaration,
    weights: Mapping[str, Any] | None,
) -> list[Fraction]:
    if weights is None:
        raw = [Fraction(1)] * len(ordered)
    else:
        if not isinstance(weights, Mapping) or set(weights) != {
            update.digest for update in ordered
        }:
            raise LoraAggregationError("lora_weight_invalid")
        raw = [_weight(weights[update.digest]) for update in ordered]
    total = sum(raw, Fraction(0))
    # Zero padding preserves each site's alpha / rank scaled delta: a rank r_i
    # product is multiplied by round_rank / r_i before averaging.
    return [
        weight / total * Fraction(declaration.rank, update.rank)
        for weight, update in zip(raw, ordered)
    ]


def _weight(value: object) -> Fraction:
    if type(value) is int or type(value) is Fraction:
        weight = Fraction(value)
        if (
            abs(weight.numerator) >= _MAX_WEIGHT_PART
            or weight.denominator >= _MAX_WEIGHT_PART
        ):
            raise LoraAggregationError("lora_weight_invalid")
    elif type(value) is float and math.isfinite(value):
        weight = Fraction(value)
    else:
        raise LoraAggregationError("lora_weight_invalid")
    if weight <= 0:
        raise LoraAggregationError("lora_weight_invalid")
    return weight


def _aggregate_frozen_a(
    spec: LoraModuleSpec,
    factors: list[LoraFactorPair],
    coefficients: list[Fraction],
) -> tuple[LoraFactorPair, str]:
    shared_a = factors[0].factor_a
    mean_b, mean_b_den = _weighted_sum(
        [pair.factor_b for pair in factors], coefficients
    )
    rounded_b = _round(mean_b, mean_b_den)
    a_ints, a_den = _scaled_ints(shared_a)
    target = _int_matmul(mean_b, a_ints)
    b_ints, b_den = _scaled_ints(rounded_b)
    band = _residual_band(
        target, mean_b_den * a_den, _int_matmul(b_ints, a_ints), b_den * a_den
    )
    pair = LoraFactorPair(spec.target_module, factor_a=shared_a, factor_b=rounded_b)
    return pair, band


def _aggregate_product_space(
    spec: LoraModuleSpec,
    factors: list[LoraFactorPair],
    coefficients: list[Fraction],
    rank: int,
) -> tuple[LoraFactorPair, str]:
    products = []
    for pair in factors:
        b_ints, b_den = _scaled_ints(pair.factor_b)
        a_ints, a_den = _scaled_ints(pair.factor_a)
        products.append((_int_matmul(b_ints, a_ints), b_den * a_den))
    target, target_den = _weighted_int_sum(products, coefficients)
    factor_b, factor_a = _refactorize(target, target_den, rank)
    b_ints, b_den = _scaled_ints(factor_b)
    a_ints, a_den = _scaled_ints(factor_a)
    band = _residual_band(
        target, target_den, _int_matmul(b_ints, a_ints), b_den * a_den
    )
    pair = LoraFactorPair(spec.target_module, factor_a=factor_a, factor_b=factor_b)
    return pair, band


def _refactorize(target: _IntMatrix, den: int, rank: int) -> tuple[Matrix, Matrix]:
    """Return ``(B, A)`` spanning the top-``rank`` singular subspace.

    The Gram matrix of the smaller side is formed exactly and rounded once.
    Eigenvectors come from a cyclic Jacobi sweep in fixed order, are ranked by
    descending eigenvalue with ties broken by index, and are signed so their
    largest-magnitude component (first on ties) is positive. The other factor
    is the exact projection of the target, rounded once.
    """
    rows, cols = len(target), len(target[0])
    if cols <= rows:
        transposed = [list(column) for column in zip(*target)]
        gram = _int_matmul(transposed, target)
        basis = _top_eigenvectors(_round(gram, den * den), rank)
        factor_a = tuple(tuple(vector) for vector in basis)
        basis_ints, basis_den = _scaled_ints(factor_a)
        columns = [list(column) for column in zip(*basis_ints)]
        factor_b = _round(_int_matmul(target, columns), den * basis_den)
        return factor_b, factor_a
    gram = _int_matmul(target, [list(column) for column in zip(*target)])
    basis = _top_eigenvectors(_round(gram, den * den), rank)
    basis_ints, basis_den = _scaled_ints(tuple(tuple(vector) for vector in basis))
    factor_a = _round(_int_matmul(basis_ints, target), den * basis_den)
    factor_b = tuple(tuple(column) for column in zip(*basis))
    return factor_b, factor_a


def _top_eigenvectors(gram: Matrix, rank: int) -> list[list[float]]:
    size = len(gram)
    a = [list(row) for row in gram]
    v = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    scale = math.fsum(value * value for row in a for value in row)
    for _ in range(_MAX_SWEEPS):
        off = math.fsum(
            a[p][q] * a[p][q] for p in range(size) for q in range(p + 1, size)
        )
        if off <= scale * _JACOBI_TOLERANCE:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                apq = a[p][q]
                if apq == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                if abs(theta) > 1e150:
                    t = 0.5 / theta
                else:
                    sign = 1.0 if theta >= 0.0 else -1.0
                    t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(size):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                a[p][q] = a[q][p] = 0.0
                for k in range(size):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    order = sorted(range(size), key=lambda index: (-a[index][index], index))
    basis = []
    for index in order[:rank]:
        vector = [v[k][index] for k in range(size)]
        pivot = max(range(size), key=lambda k: (abs(vector[k]), -k))
        if vector[pivot] < 0.0:
            vector = [-value for value in vector]
        basis.append([value + 0.0 for value in vector])
    return basis


def _weighted_sum(
    matrices: list[Matrix], coefficients: list[Fraction]
) -> tuple[_IntMatrix, int]:
    return _weighted_int_sum(
        [_scaled_ints(matrix) for matrix in matrices], coefficients
    )


def _weighted_int_sum(
    terms: list[tuple[_IntMatrix, int]], coefficients: list[Fraction]
) -> tuple[_IntMatrix, int]:
    """Return ``(N, D)`` with ``N / D == sum(c_i * M_i / d_i)`` exactly."""
    scaled = [coefficient / den for coefficient, (_, den) in zip(coefficients, terms)]
    common = 1
    for value in scaled:
        common = math.lcm(common, value.denominator)
    rows, cols = len(terms[0][0]), len(terms[0][0][0])
    total = [[0] * cols for _ in range(rows)]
    for value, (matrix, _) in zip(scaled, terms):
        factor = value.numerator * (common // value.denominator)
        for total_row, row in zip(total, matrix):
            for index, entry in enumerate(row):
                total_row[index] += factor * entry
    return total, common


def _residual_band(
    target: _IntMatrix, target_den: int, approx: _IntMatrix, approx_den: int
) -> str:
    """Classify ``||T - P||_F / ||T||_F`` exactly into a residual band."""
    numerator = 0
    norm = 0
    for target_row, approx_row in zip(target, approx):
        for t, p in zip(target_row, approx_row):
            difference = t * approx_den - p * target_den
            numerator += difference * difference
            norm += t * t
    if numerator == 0:
        return "zero"
    denominator = norm * approx_den * approx_den
    for band, exponent in _BAND_EXPONENTS:
        if numerator * 10 ** (2 * exponent) < denominator:
            return band
    return "at_least_1e-1"


def _scaled_ints(matrix: Matrix) -> tuple[_IntMatrix, int]:
    """Return ``(N, d)`` with ``N / d`` equal to the float matrix exactly."""
    ratios = [[value.as_integer_ratio() for value in row] for row in matrix]
    den = max(d for row in ratios for _, d in row)
    return [[n * (den // d) for n, d in row] for row in ratios], den


def _int_matmul(left: _IntMatrix, right: _IntMatrix) -> _IntMatrix:
    columns = list(zip(*right))
    return [
        [sum(x * y for x, y in zip(row, column)) for column in columns] for row in left
    ]


def _round(matrix: _IntMatrix, den: int) -> Matrix:
    # CPython's int / int true division is correctly rounded to float64.
    return tuple(tuple(value / den + 0.0 for value in row) for row in matrix)


def _matrix(value: Any, max_rows: int, max_cols: int) -> Matrix:
    if not isinstance(value, (list, tuple)) and hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            raise LoraAggregationError("lora_update_invalid") from None
    if not isinstance(value, (list, tuple)) or not value:
        raise LoraAggregationError("lora_update_invalid")
    if len(value) > max_rows:
        raise LoraAggregationError("lora_limit_exceeded")
    rows = []
    width = None
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise LoraAggregationError("lora_update_invalid")
        if width is None:
            width = len(row)
            if width > max_cols:
                raise LoraAggregationError("lora_limit_exceeded")
        elif len(row) != width:
            raise LoraAggregationError("lora_update_invalid")
        rows.append(tuple(_value(entry) for entry in row))
    return tuple(rows)


def _value(entry: object) -> float:
    if type(entry) is int:
        if abs(entry) > _MAX_ABS_VALUE:
            raise LoraAggregationError("lora_limit_exceeded")
        return float(entry)
    if isinstance(entry, bool) or not isinstance(entry, float):
        raise LoraAggregationError("lora_update_invalid")
    number = float(entry)
    if not math.isfinite(number):
        raise LoraAggregationError("lora_update_invalid")
    if abs(number) > _MAX_ABS_VALUE:
        raise LoraAggregationError("lora_limit_exceeded")
    return number + 0.0


def _matrix_bytes(matrix: Matrix) -> bytes:
    flat = [value for row in matrix for value in row]
    return struct.pack(f"<{len(flat)}d", *flat)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _sha256(domain: bytes, payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(domain + payload).hexdigest()


def _require_name(value: object, code: str) -> None:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 256
        or _NAME.fullmatch(value) is None
    ):
        raise LoraAggregationError(code)


def _require_int(value: object, minimum: int, maximum: int, code: str) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise LoraAggregationError(code)


__all__ = [
    "LORA_AGGREGATION_EVIDENCE_SCHEMA_VERSION",
    "LORA_AGGREGATION_MODES",
    "LORA_AGGREGATION_REASON_CODES",
    "LORA_RANK_POLICIES",
    "LORA_RESIDUAL_BANDS",
    "LORA_RESIDUAL_BAND_BOUNDS",
    "LORA_ROUND_SCHEMA_VERSION",
    "LORA_UPDATE_SCHEMA_VERSION",
    "LoraAggregate",
    "LoraAggregationError",
    "LoraAggregationEvidence",
    "LoraFactorPair",
    "LoraModuleSpec",
    "LoraRoundDeclaration",
    "LoraUpdate",
    "aggregate_lora_updates",
    "lora_factor_digest",
]
