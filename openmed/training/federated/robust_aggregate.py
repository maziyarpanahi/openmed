"""Poisoning-resistant aggregation of compatible dense adapter updates.

Naive averaging lets one extreme or malicious client update move the federated
adapter arbitrarily far. This module aggregates dense adapter deltas with two
deterministic robust strategies and fails closed when the declared bound on
suspected adversarial updates cannot be honored:

* ``coordinate_trimmed_mean`` sorts the ``n`` values of every coordinate,
  drops the ``f`` smallest and ``f`` largest, and averages the rest (Yin, Chen,
  Ramchandran and Bartlett, "Byzantine-Robust Distributed Learning: Towards
  Optimal Statistical Rates", ICML 2018). With at most ``f`` adversarial
  updates every kept value lies between two honest values, so the result stays
  inside the per-coordinate honest range. Requires ``n >= 2f + 1``.
* ``norm_clipped_mean`` scales every update whose whole-update L2 norm exceeds
  the median update norm down to that median, then averages all updates. This
  is the norm-bounding defense of Sun, Kairouz, Suresh and McMahan, "Can You
  Really Backdoor Federated Learning?", arXiv:1911.07963 (2019), with the bound
  taken from the lower median of the submitted norms. With ``n >= 2f + 1`` the
  median is bracketed by honest norms, so no single update can contribute more
  than ``median / n`` in L2 norm to the mean.

Krum and multi-Krum (Blanchard, El Mhamdi, Guerraoui and Stainer, "Machine
Learning with Adversaries: Byzantine Tolerant Gradient Descent", NeurIPS 2017)
are not provided: they need ``n > 2f + 2`` and quadratic pairwise distances.

Updates are compatible only when their validated
:class:`~openmed.training.federated_update_metadata.FederatedUpdateMetadata`
matches the trusted coordinator policy (base-model digest, adapter format,
parameter names, shapes and dtypes) and their values match that layout. The
whole batch is validated before any arithmetic. Arithmetic is pure Python over
float64 with correctly rounded ``math.fsum`` sums of order-independent
multisets, so any arrival order yields byte-identical aggregates. The module is
stdlib-only, performs no I/O, logging or network calls, and accepts nested
lists, tuples or array-like objects exposing ``tolist()``.

Errors carry one stable reason code as their message and never echo submitted
values, parameter names or digests. Diagnostics hold only aggregate counts, the
trimmed fraction, decade bands of interior norm quantiles, the strategy id and
version, and digests; they never identify a client or which update was trimmed
or clipped.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

ROBUST_AGGREGATION_SCHEMA_VERSION: Final = (
    "openmed.training.federated.robust_aggregate.v1"
)
ROBUST_AGGREGATION_STRATEGY_VERSION: Final = 1
ROBUST_AGGREGATION_STRATEGIES: Final = (
    "coordinate_trimmed_mean",
    "norm_clipped_mean",
)
ROBUST_AGGREGATION_REASON_CODES: Final = frozenset(
    {
        "robust_adapter_mismatch",
        "robust_adversary_bound_exceeded",
        "robust_config_invalid",
        "robust_dtype_mismatch",
        "robust_duplicate_update",
        "robust_limit_exceeded",
        "robust_non_finite_value",
        "robust_parameter_mismatch",
        "robust_shape_mismatch",
        "robust_strategy_unsupported",
        "robust_update_invalid",
        "robust_value_out_of_range",
    }
)
ROBUST_NORM_BANDS: Final = (
    "zero",
    "below_1e-6",
    "1e-6_to_1e-5",
    "1e-5_to_1e-4",
    "1e-4_to_1e-3",
    "1e-3_to_1e-2",
    "1e-2_to_1e-1",
    "1e-1_to_1e0",
    "1e0_to_1e1",
    "1e1_to_1e2",
    "1e2_to_1e3",
    "1e3_to_1e4",
    "1e4_to_1e5",
    "1e5_to_1e6",
    "at_least_1e6",
)

MAX_ROBUST_UPDATES: Final = 1024
MAX_ROBUST_ELEMENTS_PER_UPDATE: Final = 1 << 24
MAX_ROBUST_TOTAL_VALUES: Final = 1 << 26
MAX_ROBUST_VALUE: Final = 1.0e12

# Decimal literals parse to the nearest double, so band edges are exact and
# independent of logarithm rounding.
_BAND_EDGES: Final = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1e0,
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
)
_QUANTILES: Final = (("p25", 1), ("p50", 2), ("p75", 3))
_TRIMMED_FRACTION_DIGITS: Final = 4
_PACK_CHUNK: Final = 4096
_UPDATE_DOMAIN: Final = b"openmed.training.federated.robust_aggregate.update.v1\0"
_INPUTS_DOMAIN: Final = b"openmed.training.federated.robust_aggregate.inputs.v1\0"
_CONFIG_DOMAIN: Final = b"openmed.training.federated.robust_aggregate.config.v1\0"
_RESULT_DOMAIN: Final = b"openmed.training.federated.robust_aggregate.result.v1\0"


class RobustAggregationError(ValueError):
    """Fail-closed refusal carrying one stable reason code.

    The message is exactly the reason code. Submitted values, parameter
    names, digests and client details are never attached.

    Attributes:
        code: One of ``ROBUST_AGGREGATION_REASON_CODES``.
    """

    def __init__(self, code: str) -> None:
        if code not in ROBUST_AGGREGATION_REASON_CODES:
            code = "robust_update_invalid"
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class RobustAggregationConfig:
    """Trusted strategy choice and declared suspected-adversary bound.

    Args:
        strategy: One of ``ROBUST_AGGREGATION_STRATEGIES``.
        max_suspected_adversaries: Declared bound ``f`` on the number of
            adversarial updates in one batch. Both strategies require at
            least ``2f + 1`` updates; ``f = 0`` reduces the trimmed mean to the
            plain mean and is intended only for trusted federations.

    Raises:
        RobustAggregationError: ``robust_strategy_unsupported`` for an
            unknown strategy, or ``robust_config_invalid`` for a bound that
            is not an integer between 0 and ``(MAX_ROBUST_UPDATES - 1) // 2``.
    """

    strategy: str
    max_suspected_adversaries: int

    def __post_init__(self) -> None:
        if (
            type(self.strategy) is not str
            or self.strategy not in ROBUST_AGGREGATION_STRATEGIES
        ):
            raise RobustAggregationError("robust_strategy_unsupported")
        bound = self.max_suspected_adversaries
        if type(bound) is not int or not 0 <= bound <= (MAX_ROBUST_UPDATES - 1) // 2:
            raise RobustAggregationError("robust_config_invalid")

    @property
    def minimum_updates(self) -> int:
        """Return the smallest batch size the strategy accepts, ``2f + 1``."""
        return 2 * self.max_suspected_adversaries + 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping of the strategy and its bound."""
        return {
            "strategy": self.strategy,
            "strategy_version": ROBUST_AGGREGATION_STRATEGY_VERSION,
            "max_suspected_adversaries": self.max_suspected_adversaries,
        }


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class RobustAdapterUpdate:
    """One dense adapter update: validated metadata plus its tensor values.

    Values are normalized to float64 nested tuples, negative zero becomes
    zero, and every value must be exactly representable in the declared
    dtype. ``repr`` never shows values or the declared update digest.

    Args:
        metadata: Update metadata already validated against a coordinator
            :class:`FederatedUpdatePolicy`.
        tensors: Mapping from every declared parameter name to nested lists
            or tuples of finite numbers, or array-like objects exposing
            ``tolist()``, matching the declared shape.

    Raises:
        RobustAggregationError: With a stable reason code for invalid
            metadata, names, shapes, dtypes, or values.
    """

    metadata: FederatedUpdateMetadata
    tensors: Mapping[str, Any]
    _values: tuple[float, ...] = field(init=False, compare=False)
    _content_digest: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        metadata = self.metadata
        if type(metadata) is not FederatedUpdateMetadata:
            raise RobustAggregationError("robust_update_invalid")
        if metadata.total_elements > MAX_ROBUST_ELEMENTS_PER_UPDATE:
            raise RobustAggregationError("robust_limit_exceeded")
        tensors = self.tensors
        if not isinstance(tensors, Mapping):
            raise RobustAggregationError("robust_update_invalid")
        names = [parameter.name for parameter in metadata.parameters]
        if len(tensors) != len(names) or any(name not in tensors for name in names):
            raise RobustAggregationError("robust_parameter_mismatch")
        flat: list[float] = []
        nested: dict[str, Any] = {}
        for parameter in metadata.parameters:
            nested[parameter.name] = _shaped(
                tensors[parameter.name], parameter.shape, parameter.dtype, flat
            )
        values = tuple(flat)
        object.__setattr__(self, "tensors", MappingProxyType(nested))
        object.__setattr__(self, "_values", values)
        object.__setattr__(
            self,
            "_content_digest",
            _values_digest(_UPDATE_DOMAIN, metadata.parameters, values),
        )

    def __repr__(self) -> str:
        return (
            f"RobustAdapterUpdate(parameter_count={len(self.metadata.parameters)}, "
            f"element_count={self.metadata.total_elements})"
        )


@dataclass(frozen=True, slots=True)
class RobustAggregationDiagnostics:
    """Aggregate-level robustness diagnostics with no per-client values.

    Args:
        strategy: Strategy identifier from ``ROBUST_AGGREGATION_STRATEGIES``.
        strategy_version: ``ROBUST_AGGREGATION_STRATEGY_VERSION``.
        max_suspected_adversaries: Declared bound ``f``.
        update_count: Number of aggregated updates ``n``.
        parameter_count: Number of adapter tensors.
        element_count: Scalar values per update.
        trimmed_per_side: Values dropped per side of every coordinate.
        trimmed_fraction: ``2 * trimmed_per_side / n`` rounded to four
            decimal places.
        clipped_update_count: Number of updates scaled to the norm bound.
            It never says which updates were scaled.
        norm_quantile_bands: Decade bands of the 25th, 50th and 75th
            percentile update L2 norms. Ranks are clamped to the window
            between the ``f``-th smallest and ``f``-th largest norm, so up to
            ``f`` adversaries cannot set a band alone and, for ``f >= 1``,
            minimum and maximum norms are never reported.
        config_digest: SHA-256 reference of the policy layout and config.
        inputs_digest: SHA-256 reference over sorted update content digests.
        aggregate_digest: SHA-256 reference of the aggregate values.
        schema_version: ``ROBUST_AGGREGATION_SCHEMA_VERSION``.
    """

    strategy: str
    strategy_version: int
    max_suspected_adversaries: int
    update_count: int
    parameter_count: int
    element_count: int
    trimmed_per_side: int
    trimmed_fraction: float
    clipped_update_count: int
    norm_quantile_bands: tuple[str, str, str]
    config_digest: str
    inputs_digest: str
    aggregate_digest: str
    schema_version: str = ROBUST_AGGREGATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != ROBUST_AGGREGATION_SCHEMA_VERSION
            or self.strategy not in ROBUST_AGGREGATION_STRATEGIES
            or self.strategy_version != ROBUST_AGGREGATION_STRATEGY_VERSION
            or type(self.norm_quantile_bands) is not tuple
            or len(self.norm_quantile_bands) != len(_QUANTILES)
            or any(band not in ROBUST_NORM_BANDS for band in self.norm_quantile_bands)
        ):
            raise RobustAggregationError("robust_config_invalid")
        counts = (
            self.max_suspected_adversaries,
            self.update_count,
            self.parameter_count,
            self.element_count,
            self.trimmed_per_side,
            self.clipped_update_count,
        )
        if any(type(count) is not int or count < 0 for count in counts):
            raise RobustAggregationError("robust_config_invalid")
        if type(self.trimmed_fraction) is not float or not (
            0.0 <= self.trimmed_fraction < 1.0
        ):
            raise RobustAggregationError("robust_config_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-ready mapping of the diagnostics."""
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "strategy_version": self.strategy_version,
            "max_suspected_adversaries": self.max_suspected_adversaries,
            "update_count": self.update_count,
            "parameter_count": self.parameter_count,
            "element_count": self.element_count,
            "trimmed_per_side": self.trimmed_per_side,
            "trimmed_fraction": self.trimmed_fraction,
            "clipped_update_count": self.clipped_update_count,
            "norm_quantile_bands": {
                label: band
                for (label, _), band in zip(_QUANTILES, self.norm_quantile_bands)
            },
            "config_digest": self.config_digest,
            "inputs_digest": self.inputs_digest,
            "aggregate_digest": self.aggregate_digest,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with sorted keys and a trailing newline."""
        return (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class RobustAggregate:
    """Robust aggregate tensors plus aggregate-level diagnostics.

    Args:
        parameters: Policy parameter metadata in canonical name order.
        tensors: Read-only mapping from parameter name to nested float tuples.
        diagnostics: Aggregate-level robustness diagnostics.
    """

    parameters: tuple[FederatedParameterMetadata, ...]
    tensors: Mapping[str, Any]
    diagnostics: RobustAggregationDiagnostics

    def tensor(self, name: str) -> Any:
        """Return one aggregate tensor as nested float tuples.

        Args:
            name: A parameter name from the coordinator policy.

        Returns:
            Nested tuples matching the declared shape.

        Raises:
            RobustAggregationError: ``robust_parameter_mismatch`` for an
                unknown name. The message never includes the name.
        """
        if type(name) is not str or name not in self.tensors:
            raise RobustAggregationError("robust_parameter_mismatch")
        return self.tensors[name]

    def to_mapping(self) -> dict[str, Any]:
        """Return fresh nested lists keyed by parameter name in name order."""
        return {
            parameter.name: _as_lists(self.tensors[parameter.name])
            for parameter in self.parameters
        }

    def __repr__(self) -> str:
        return (
            f"RobustAggregate(strategy={self.diagnostics.strategy!r}, "
            f"parameter_count={self.diagnostics.parameter_count}, "
            f"element_count={self.diagnostics.element_count})"
        )


def aggregate_adapter_updates(
    updates: Sequence[RobustAdapterUpdate],
    *,
    policy: FederatedUpdatePolicy,
    config: RobustAggregationConfig,
) -> RobustAggregate:
    """Aggregate compatible adapter updates with a robust strategy.

    The whole batch is validated before any arithmetic: every update must
    match the trusted policy, no update may appear twice (by declared update
    digest or by content), and the batch must hold at least ``2f + 1``
    updates for the declared bound ``f``.

    Args:
        updates: List or tuple of :class:`RobustAdapterUpdate` objects. The
            order does not affect the result.
        policy: Trusted coordinator policy; never derive it from updates.
        config: Trusted strategy and suspected-adversary bound.

    Returns:
        The aggregate tensors and aggregate-level diagnostics.

    Raises:
        RobustAggregationError: With a stable reason code; the batch is
            refused as a whole and no partial aggregate is produced.

    Example:
        >>> from openmed.training.federated_update_metadata import (
        ...     FederatedParameterMetadata,
        ...     FederatedUpdateMetadata,
        ...     FederatedUpdatePolicy,
        ... )
        >>> name = "adapter.lora_B.weight"
        >>> policy = FederatedUpdatePolicy(
        ...     model_digest="sha256:" + "a" * 64,
        ...     parameters=(FederatedParameterMetadata(name, (2,), "float32"),),
        ...     max_total_elements=2,
        ... )
        >>> def update(index, values):
        ...     metadata = FederatedUpdateMetadata(
        ...         model_digest=policy.model_digest,
        ...         adapter_format="dense",
        ...         parameters=policy.parameters,
        ...         total_elements=2,
        ...         update_digest="sha256:" + format(index, "064x"),
        ...         clipped=True,
        ...         policy=policy,
        ...     )
        ...     return RobustAdapterUpdate(metadata=metadata, tensors={name: values})
        >>> batch = [
        ...     update(1, [1.0, -1.0]),
        ...     update(2, [2.0, -2.0]),
        ...     update(3, [3.0, -3.0]),
        ...     update(4, [5000.0, 5000.0]),
        ... ]
        >>> config = RobustAggregationConfig("coordinate_trimmed_mean", 1)
        >>> aggregate_adapter_updates(batch, policy=policy, config=config).tensor(
        ...     name
        ... )
        (2.5, -1.5)
    """
    if type(policy) is not FederatedUpdatePolicy:
        raise RobustAggregationError("robust_config_invalid")
    if type(config) is not RobustAggregationConfig:
        raise RobustAggregationError("robust_config_invalid")
    if type(updates) not in (list, tuple):
        raise RobustAggregationError("robust_update_invalid")
    if len(updates) > MAX_ROBUST_UPDATES:
        raise RobustAggregationError("robust_limit_exceeded")
    declared: set[str] = set()
    contents: set[str] = set()
    for update in updates:
        if type(update) is not RobustAdapterUpdate:
            raise RobustAggregationError("robust_update_invalid")
        _require_compatible(update.metadata, policy)
        if (
            update.metadata.update_digest in declared
            or update._content_digest in contents
        ):
            raise RobustAggregationError("robust_duplicate_update")
        declared.add(update.metadata.update_digest)
        contents.add(update._content_digest)
    count = len(updates)
    if count < config.minimum_updates:
        raise RobustAggregationError("robust_adversary_bound_exceeded")
    elements = sum(parameter.element_count for parameter in policy.parameters)
    if count * elements > MAX_ROBUST_TOTAL_VALUES:
        raise RobustAggregationError("robust_limit_exceeded")

    bound = config.max_suspected_adversaries
    norms = [_l2_norm(update._values) for update in updates]
    ordered_norms = sorted(norms)
    if config.strategy == "coordinate_trimmed_mean":
        flat = _trimmed_mean([update._values for update in updates], bound)
        trimmed_per_side = bound
        clipped = 0
    else:
        threshold = ordered_norms[(count - 1) // 2]
        flat, clipped = _norm_clipped_mean(
            [update._values for update in updates], norms, threshold
        )
        trimmed_per_side = 0
    values = tuple(flat)

    tensors: dict[str, Any] = {}
    offset = 0
    for parameter in policy.parameters:
        tensors[parameter.name] = _nest(values, offset, parameter.shape)
        offset += parameter.element_count

    diagnostics = RobustAggregationDiagnostics(
        strategy=config.strategy,
        strategy_version=ROBUST_AGGREGATION_STRATEGY_VERSION,
        max_suspected_adversaries=bound,
        update_count=count,
        parameter_count=len(policy.parameters),
        element_count=elements,
        trimmed_per_side=trimmed_per_side,
        trimmed_fraction=round(2 * trimmed_per_side / count, _TRIMMED_FRACTION_DIGITS),
        clipped_update_count=clipped,
        norm_quantile_bands=_quantile_bands(ordered_norms, bound),
        config_digest=_config_digest(policy, config),
        inputs_digest=_sha256(
            _INPUTS_DOMAIN, "\n".join(sorted(contents)).encode("ascii")
        ),
        aggregate_digest=_values_digest(_RESULT_DOMAIN, policy.parameters, values),
    )
    return RobustAggregate(
        parameters=policy.parameters,
        tensors=MappingProxyType(tensors),
        diagnostics=diagnostics,
    )


def _require_compatible(
    metadata: FederatedUpdateMetadata, policy: FederatedUpdatePolicy
) -> None:
    if (
        metadata.model_digest != policy.model_digest
        or metadata.adapter_format != policy.adapter_format
    ):
        raise RobustAggregationError("robust_adapter_mismatch")
    if metadata.parameters != policy.parameters:
        raise RobustAggregationError("robust_parameter_mismatch")
    if policy.require_clipped and not metadata.clipped:
        raise RobustAggregationError("robust_update_invalid")


def _trimmed_mean(rows: list[tuple[float, ...]], bound: int) -> list[float]:
    count = len(rows)
    result = []
    for column in zip(*rows):
        kept = sorted(column)[bound : count - bound]
        mean = math.fsum(kept) / len(kept)
        # The exact mean lies within the kept values; clamping only removes
        # the final rounding error so the honest-range guarantee is exact.
        result.append(min(max(mean, kept[0]), kept[-1]) + 0.0)
    return result


def _norm_clipped_mean(
    rows: list[tuple[float, ...]], norms: list[float], threshold: float
) -> tuple[list[float], int]:
    scales = [1.0 if norm <= threshold else threshold / norm for norm in norms]
    clipped = sum(1 for norm in norms if norm > threshold)
    count = len(rows)
    result = [
        math.fsum(value * scale for value, scale in zip(column, scales)) / count + 0.0
        for column in zip(*rows)
    ]
    return result, clipped


def _l2_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def _quantile_bands(ordered_norms: list[float], bound: int) -> tuple[str, str, str]:
    count = len(ordered_norms)
    bands = []
    for _, quarter in _QUANTILES:
        # Nearest-rank percentile, clamped into the robust order-statistic
        # window [f, n - 1 - f] so suspected adversaries cannot set it alone.
        rank = (count * quarter + 3) // 4 - 1
        rank = min(max(rank, bound), count - 1 - bound)
        bands.append(_norm_band(ordered_norms[rank]))
    return bands[0], bands[1], bands[2]


def _norm_band(value: float) -> str:
    if value == 0.0:
        return "zero"
    for index, edge in enumerate(_BAND_EDGES):
        if value < edge:
            return ROBUST_NORM_BANDS[index + 1]
    return ROBUST_NORM_BANDS[-1]


def _shaped(value: Any, shape: tuple[int, ...], dtype: str, flat: list[float]) -> Any:
    value = _materialize(value)
    if not shape:
        number = _number(value)
        if not _representable(number, dtype):
            raise RobustAggregationError("robust_dtype_mismatch")
        flat.append(number)
        return number
    if type(value) not in (list, tuple) or len(value) != shape[0]:
        raise RobustAggregationError("robust_shape_mismatch")
    return tuple(_shaped(item, shape[1:], dtype, flat) for item in value)


def _materialize(value: Any) -> Any:
    if type(value) in (list, tuple, int, float, bool):
        return value
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            raise RobustAggregationError("robust_update_invalid") from None
    return value


def _number(value: Any) -> float:
    if type(value) in (list, tuple):
        raise RobustAggregationError("robust_shape_mismatch")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RobustAggregationError("robust_update_invalid")
    if isinstance(value, int):
        if not -MAX_ROBUST_VALUE <= value <= MAX_ROBUST_VALUE:
            raise RobustAggregationError("robust_value_out_of_range")
        return float(value) + 0.0
    number = float(value)
    if not math.isfinite(number):
        raise RobustAggregationError("robust_non_finite_value")
    if not -MAX_ROBUST_VALUE <= number <= MAX_ROBUST_VALUE:
        raise RobustAggregationError("robust_value_out_of_range")
    return number + 0.0


def _representable(value: float, dtype: str) -> bool:
    if dtype == "float64":
        return True
    try:
        if dtype == "float16":
            return struct.unpack("<e", struct.pack("<e", value))[0] == value
        packed = struct.pack("<f", value)
    except (OverflowError, struct.error):
        return False
    if struct.unpack("<f", packed)[0] != value:
        return False
    if dtype == "bfloat16":
        return struct.unpack("<I", packed)[0] & 0xFFFF == 0
    return True


def _nest(values: tuple[float, ...], offset: int, shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return values[offset : offset + shape[0]]
    stride = math.prod(shape[1:])
    return tuple(
        _nest(values, offset + index * stride, shape[1:]) for index in range(shape[0])
    )


def _as_lists(value: Any) -> Any:
    if type(value) is tuple:
        return [_as_lists(item) for item in value]
    return value


def _layout_bytes(parameters: tuple[FederatedParameterMetadata, ...]) -> bytes:
    return _canonical_json([parameter.to_dict() for parameter in parameters])


def _values_digest(
    domain: bytes,
    parameters: tuple[FederatedParameterMetadata, ...],
    values: tuple[float, ...],
) -> str:
    digest = hashlib.sha256(domain)
    digest.update(_layout_bytes(parameters))
    digest.update(b"\0")
    for start in range(0, len(values), _PACK_CHUNK):
        chunk = values[start : start + _PACK_CHUNK]
        digest.update(struct.pack(f"<{len(chunk)}d", *chunk))
    return "sha256:" + digest.hexdigest()


def _config_digest(
    policy: FederatedUpdatePolicy, config: RobustAggregationConfig
) -> str:
    payload = {
        "schema_version": ROBUST_AGGREGATION_SCHEMA_VERSION,
        "model_digest": policy.model_digest,
        "adapter_format": policy.adapter_format,
        "parameters": [parameter.to_dict() for parameter in policy.parameters],
        **config.to_dict(),
    }
    return _sha256(_CONFIG_DOMAIN, _canonical_json(payload))


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _sha256(domain: bytes, payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(domain + payload).hexdigest()


__all__ = [
    "MAX_ROBUST_ELEMENTS_PER_UPDATE",
    "MAX_ROBUST_TOTAL_VALUES",
    "MAX_ROBUST_UPDATES",
    "MAX_ROBUST_VALUE",
    "ROBUST_AGGREGATION_REASON_CODES",
    "ROBUST_AGGREGATION_SCHEMA_VERSION",
    "ROBUST_AGGREGATION_STRATEGIES",
    "ROBUST_AGGREGATION_STRATEGY_VERSION",
    "ROBUST_NORM_BANDS",
    "RobustAdapterUpdate",
    "RobustAggregate",
    "RobustAggregationConfig",
    "RobustAggregationDiagnostics",
    "RobustAggregationError",
    "aggregate_adapter_updates",
]
