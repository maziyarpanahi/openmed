"""Privacy-safe aggregate metric envelopes for federated training."""

from __future__ import annotations

import json
import math
import re
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any, Final

from .federated_status import DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE

FEDERATED_METRIC_SCHEMA_VERSION = "openmed.training.federated_metric.v1"
DEFAULT_FEDERATED_METRIC_MINIMUM_GROUP_SIZE = DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE

_METRIC_ID = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_MECHANISM_VERSION = re.compile(r"v[1-9][0-9]{0,3}\Z")
_ENVELOPE_BUILDER_TOKEN: Final = object()
_ENVELOPE_FIELDS: Final = frozenset(
    {
        "aggregate_value",
        "clipping_lower_bound",
        "clipping_upper_bound",
        "confidence_level",
        "metric_id",
        "metric_kind",
        "minimum_group_size",
        "participant_count_band",
        "privacy_mechanism",
        "privacy_mechanism_version",
        "schema_version",
        "uncertainty_lower_bound",
        "uncertainty_method",
        "uncertainty_upper_bound",
    }
)


class FederatedMetricKind(str, Enum):
    """Closed kinds supported by aggregate metric envelopes."""

    COUNT = "count"
    RATE = "rate"
    BOUNDED_MEAN = "bounded_mean"


class FederatedPrivacyMechanism(str, Enum):
    """Closed privacy mechanisms reported by an envelope."""

    THRESHOLD_ONLY = "threshold_only"
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


class FederatedParticipantCountBand(str, Enum):
    """Coarse participant bands relative to the minimum group size."""

    SUPPRESSED = "suppressed"
    MINIMUM_TO_UNDER_DOUBLE = "minimum_to_under_double"
    DOUBLE_TO_UNDER_FOURFOLD = "double_to_under_fourfold"
    FOURFOLD_OR_MORE = "fourfold_or_more"


class FederatedUncertaintyMethod(str, Enum):
    """Closed uncertainty descriptions for released metrics."""

    NONE = "none"
    CONFIDENCE_INTERVAL = "confidence_interval"
    SUPPRESSED = "suppressed"


class FederatedMetricError(ValueError):
    """Raised when an aggregate metric envelope is not privacy safe."""


@dataclass(frozen=True)
class FederatedMetricEnvelope:
    """Immutable, versioned output from :func:`build_federated_metric_envelope`."""

    metric_id: str
    metric_kind: FederatedMetricKind
    aggregate_value: int | float | None
    clipping_lower_bound: int | float
    clipping_upper_bound: int | float
    privacy_mechanism: FederatedPrivacyMechanism
    privacy_mechanism_version: str
    minimum_group_size: int
    participant_count_band: FederatedParticipantCountBand
    uncertainty_method: FederatedUncertaintyMethod
    uncertainty_lower_bound: int | float | None = None
    uncertainty_upper_bound: int | float | None = None
    confidence_level: float | None = None
    schema_version: str = FEDERATED_METRIC_SCHEMA_VERSION
    _builder_token: InitVar[object | None] = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _ENVELOPE_BUILDER_TOKEN:
            raise FederatedMetricError(
                "federated metric envelopes must be built from aggregate inputs"
            )
        _validate_envelope(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping with no client-level fields."""

        return {
            "aggregate_value": self.aggregate_value,
            "clipping_lower_bound": self.clipping_lower_bound,
            "clipping_upper_bound": self.clipping_upper_bound,
            "confidence_level": self.confidence_level,
            "metric_id": self.metric_id,
            "metric_kind": self.metric_kind.value,
            "minimum_group_size": self.minimum_group_size,
            "participant_count_band": self.participant_count_band.value,
            "privacy_mechanism": self.privacy_mechanism.value,
            "privacy_mechanism_version": self.privacy_mechanism_version,
            "schema_version": self.schema_version,
            "uncertainty_lower_bound": self.uncertainty_lower_bound,
            "uncertainty_method": self.uncertainty_method.value,
            "uncertainty_upper_bound": self.uncertainty_upper_bound,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FederatedMetricEnvelope:
        """Parse a closed envelope mapping and reject unknown fields."""

        if type(payload) is not dict or set(payload) != _ENVELOPE_FIELDS:
            raise FederatedMetricError("invalid federated metric envelope fields")
        try:
            metric_kind = FederatedMetricKind(payload["metric_kind"])
            privacy_mechanism = FederatedPrivacyMechanism(payload["privacy_mechanism"])
            participant_count_band = FederatedParticipantCountBand(
                payload["participant_count_band"]
            )
            uncertainty_method = FederatedUncertaintyMethod(
                payload["uncertainty_method"]
            )
        except (TypeError, ValueError):
            raise FederatedMetricError(
                "invalid federated metric envelope enum value"
            ) from None
        return cls(
            metric_id=payload["metric_id"],
            metric_kind=metric_kind,
            aggregate_value=payload["aggregate_value"],
            clipping_lower_bound=payload["clipping_lower_bound"],
            clipping_upper_bound=payload["clipping_upper_bound"],
            privacy_mechanism=privacy_mechanism,
            privacy_mechanism_version=payload["privacy_mechanism_version"],
            minimum_group_size=payload["minimum_group_size"],
            participant_count_band=participant_count_band,
            uncertainty_method=uncertainty_method,
            uncertainty_lower_bound=payload["uncertainty_lower_bound"],
            uncertainty_upper_bound=payload["uncertainty_upper_bound"],
            confidence_level=payload["confidence_level"],
            schema_version=payload["schema_version"],
            _builder_token=_ENVELOPE_BUILDER_TOKEN,
        )


def build_federated_metric_envelope(
    *,
    metric_id: str,
    metric_kind: FederatedMetricKind,
    aggregate_value: int | float,
    clipping_lower_bound: int | float,
    clipping_upper_bound: int | float,
    privacy_mechanism: FederatedPrivacyMechanism,
    privacy_mechanism_version: str,
    participant_count: int,
    minimum_group_size: int = DEFAULT_FEDERATED_METRIC_MINIMUM_GROUP_SIZE,
    uncertainty_method: FederatedUncertaintyMethod = FederatedUncertaintyMethod.NONE,
    uncertainty_lower_bound: int | float | None = None,
    uncertainty_upper_bound: int | float | None = None,
    confidence_level: float | None = None,
) -> FederatedMetricEnvelope:
    """Build an envelope while discarding values from sub-threshold groups."""

    _require_minimum_group_size(minimum_group_size)
    _require_non_negative_int(participant_count, "participant count")
    participant_count_band = _participant_count_band(
        participant_count, minimum_group_size
    )
    validated_release = FederatedMetricEnvelope(
        metric_id=metric_id,
        metric_kind=metric_kind,
        aggregate_value=aggregate_value,
        clipping_lower_bound=clipping_lower_bound,
        clipping_upper_bound=clipping_upper_bound,
        privacy_mechanism=privacy_mechanism,
        privacy_mechanism_version=privacy_mechanism_version,
        minimum_group_size=minimum_group_size,
        participant_count_band=(
            FederatedParticipantCountBand.MINIMUM_TO_UNDER_DOUBLE
            if participant_count_band is FederatedParticipantCountBand.SUPPRESSED
            else participant_count_band
        ),
        uncertainty_method=uncertainty_method,
        uncertainty_lower_bound=uncertainty_lower_bound,
        uncertainty_upper_bound=uncertainty_upper_bound,
        confidence_level=confidence_level,
        _builder_token=_ENVELOPE_BUILDER_TOKEN,
    )
    if participant_count_band is FederatedParticipantCountBand.SUPPRESSED:
        return FederatedMetricEnvelope(
            metric_id=metric_id,
            metric_kind=metric_kind,
            aggregate_value=None,
            clipping_lower_bound=clipping_lower_bound,
            clipping_upper_bound=clipping_upper_bound,
            privacy_mechanism=privacy_mechanism,
            privacy_mechanism_version=privacy_mechanism_version,
            minimum_group_size=minimum_group_size,
            participant_count_band=participant_count_band,
            uncertainty_method=FederatedUncertaintyMethod.SUPPRESSED,
            _builder_token=_ENVELOPE_BUILDER_TOKEN,
        )
    return validated_release


def _validate_envelope(envelope: FederatedMetricEnvelope) -> None:
    if (
        type(envelope.schema_version) is not str
        or envelope.schema_version != FEDERATED_METRIC_SCHEMA_VERSION
    ):
        raise FederatedMetricError("unsupported federated metric schema")
    if (
        type(envelope.metric_id) is not str
        or _METRIC_ID.fullmatch(envelope.metric_id) is None
    ):
        raise FederatedMetricError("invalid aggregate metric identifier")
    if not isinstance(envelope.metric_kind, FederatedMetricKind):
        raise FederatedMetricError("invalid aggregate metric kind")
    if not isinstance(envelope.privacy_mechanism, FederatedPrivacyMechanism):
        raise FederatedMetricError("invalid privacy mechanism")
    if type(envelope.privacy_mechanism_version) is not str or (
        _MECHANISM_VERSION.fullmatch(envelope.privacy_mechanism_version) is None
    ):
        raise FederatedMetricError("invalid privacy mechanism version")
    _require_minimum_group_size(envelope.minimum_group_size)
    if not isinstance(envelope.participant_count_band, FederatedParticipantCountBand):
        raise FederatedMetricError("invalid participant count band")
    if not isinstance(envelope.uncertainty_method, FederatedUncertaintyMethod):
        raise FederatedMetricError("invalid uncertainty method")
    _validate_clipping_bounds(envelope)
    if envelope.participant_count_band is FederatedParticipantCountBand.SUPPRESSED:
        _validate_suppressed_envelope(envelope)
        return
    _validate_released_envelope(envelope)


def _validate_clipping_bounds(envelope: FederatedMetricEnvelope) -> None:
    lower = _require_finite_number(
        envelope.clipping_lower_bound, "clipping lower bound"
    )
    upper = _require_finite_number(
        envelope.clipping_upper_bound, "clipping upper bound"
    )
    if lower >= upper:
        raise FederatedMetricError("clipping bounds must be increasing")
    if envelope.metric_kind is FederatedMetricKind.COUNT:
        if not _is_int_number(lower) or not _is_int_number(upper) or lower < 0:
            raise FederatedMetricError(
                "count clipping bounds must be non-negative integers"
            )
    if envelope.metric_kind is FederatedMetricKind.RATE and (lower < 0 or upper > 1):
        raise FederatedMetricError("rate clipping bounds must stay within zero and one")


def _validate_suppressed_envelope(envelope: FederatedMetricEnvelope) -> None:
    if envelope.aggregate_value is not None:
        raise FederatedMetricError(
            "suppressed metric must not retain an aggregate value"
        )
    if envelope.uncertainty_method is not FederatedUncertaintyMethod.SUPPRESSED:
        raise FederatedMetricError("suppressed metric must suppress uncertainty")
    if any(
        value is not None
        for value in (
            envelope.uncertainty_lower_bound,
            envelope.uncertainty_upper_bound,
            envelope.confidence_level,
        )
    ):
        raise FederatedMetricError(
            "suppressed metric must not retain uncertainty values"
        )


def _validate_released_envelope(envelope: FederatedMetricEnvelope) -> None:
    value = _require_finite_number(envelope.aggregate_value, "aggregate value")
    lower = _require_finite_number(
        envelope.clipping_lower_bound, "clipping lower bound"
    )
    upper = _require_finite_number(
        envelope.clipping_upper_bound, "clipping upper bound"
    )
    if not lower <= value <= upper:
        raise FederatedMetricError("aggregate value falls outside clipping bounds")
    if envelope.metric_kind is FederatedMetricKind.COUNT and not _is_int_number(
        envelope.aggregate_value
    ):
        raise FederatedMetricError("count aggregate value must be an integer")
    if envelope.uncertainty_method is FederatedUncertaintyMethod.SUPPRESSED:
        raise FederatedMetricError("released metric cannot use suppressed uncertainty")
    if envelope.uncertainty_method is FederatedUncertaintyMethod.NONE:
        if any(
            item is not None
            for item in (
                envelope.uncertainty_lower_bound,
                envelope.uncertainty_upper_bound,
                envelope.confidence_level,
            )
        ):
            raise FederatedMetricError("uncertainty values require an interval")
        return
    uncertainty_lower = _require_finite_number(
        envelope.uncertainty_lower_bound, "uncertainty lower bound"
    )
    uncertainty_upper = _require_finite_number(
        envelope.uncertainty_upper_bound, "uncertainty upper bound"
    )
    confidence = _require_finite_number(envelope.confidence_level, "confidence level")
    if not lower <= uncertainty_lower <= value <= uncertainty_upper <= upper:
        raise FederatedMetricError(
            "uncertainty interval must contain the aggregate within clipping bounds"
        )
    if not 0 < confidence < 1:
        raise FederatedMetricError("confidence level must be between zero and one")


def _participant_count_band(
    participant_count: int, minimum_group_size: int
) -> FederatedParticipantCountBand:
    if participant_count < minimum_group_size:
        return FederatedParticipantCountBand.SUPPRESSED
    if participant_count < minimum_group_size * 2:
        return FederatedParticipantCountBand.MINIMUM_TO_UNDER_DOUBLE
    if participant_count < minimum_group_size * 4:
        return FederatedParticipantCountBand.DOUBLE_TO_UNDER_FOURFOLD
    return FederatedParticipantCountBand.FOURFOLD_OR_MORE


def _require_finite_number(value: object, field: str) -> int | float:
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise FederatedMetricError(f"{field} must be finite")


def _is_int_number(value: object) -> bool:
    return type(value) is int


def _require_non_negative_int(value: object, field: str) -> None:
    if type(value) is not int or value < 0:
        raise FederatedMetricError(f"{field} must be a non-negative integer")


def _require_minimum_group_size(value: object) -> None:
    if type(value) is not int or value < 2:
        raise FederatedMetricError(
            "minimum group size must be an integer greater than one"
        )
