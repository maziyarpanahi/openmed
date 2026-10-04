"""Value-free telemetry for federated round phases.

Federated rounds already publish metadata-only status summaries.  This module
adds the corresponding local telemetry family on top of
:class:`~openmed.core.no_phi_telemetry.NoPHITelemetryExporter`: phase
transitions, rejection reason counts, banded update counts, and a phase-latency
histogram.

Two rules make the family safe for a coordinator or a site:

* Every dimension value must come from the closed vocabularies declared by the
  exporter.  A client pseudonym, a filesystem path, a digest, or any other
  free-form string is refused instead of being recorded as a new label.
* An aggregate update count is recorded as a coarse band.  An exact count is
  only released when it reaches the minimum group size; smaller counts are
  refused rather than rounded silently.

The recorder is local and pull-oriented like the exporter: ``export()`` and
``render_prometheus()`` format an in-memory snapshot and never contact a
collector.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final

from openmed.core.no_phi_telemetry import (
    FEDERATED_PHASE_LATENCY_NAME,
    FEDERATED_PHASE_VALUES,
    FEDERATED_REASON_CODE_VALUES,
    FEDERATED_UPDATE_BAND_VALUES,
    PIPELINE_STATUS_VALUES,
    CounterName,
    DimensionName,
    NoPHITelemetryExporter,
    TelemetrySchemaError,
    TelemetrySnapshot,
)

FEDERATED_TELEMETRY_MINIMUM_GROUP_SIZE: Final = 5
FEDERATED_PHASE_STATUS_VALUES: Final[tuple[str, ...]] = PIPELINE_STATUS_VALUES


class FederatedTelemetryError(TelemetrySchemaError):
    """Raised when federated telemetry input cannot be recorded safely."""


class UnapprovedFederatedValueError(FederatedTelemetryError):
    """Raised when a value is outside the closed federated vocabulary."""


class FederatedUpdateBand(str, Enum):
    """Coarse update-count bands relative to the minimum group size."""

    SUPPRESSED = "suppressed"
    MINIMUM_TO_UNDER_DOUBLE = "minimum_to_under_double"
    DOUBLE_TO_UNDER_FOURFOLD = "double_to_under_fourfold"
    FOURFOLD_OR_MORE = "fourfold_or_more"

    def __str__(self) -> str:
        return self.value


def band_update_count(
    count: int,
    *,
    minimum_group_size: int = FEDERATED_TELEMETRY_MINIMUM_GROUP_SIZE,
) -> FederatedUpdateBand:
    """Return the coarse band for one aggregate update count.

    The band vocabulary is relative to ``minimum_group_size`` so the same
    thresholds describe every deployment.  Counts below the minimum group size
    collapse into ``suppressed`` and are never represented exactly.
    """

    safe_count = _require_count(count)
    safe_minimum = _require_minimum_group_size(minimum_group_size)
    return _band(safe_count, safe_minimum)


class FederatedRoundTelemetry:
    """Record value-free federated round telemetry through a local exporter.

    The recorder owns no transport, endpoint, or background thread.  It writes
    aggregate samples into a :class:`NoPHITelemetryExporter`, which the caller
    may export explicitly or leave unused.
    """

    def __init__(
        self,
        *,
        exporter: NoPHITelemetryExporter | None = None,
        minimum_group_size: int = FEDERATED_TELEMETRY_MINIMUM_GROUP_SIZE,
    ) -> None:
        self._minimum_group_size = _require_minimum_group_size(minimum_group_size)
        if exporter is None:
            exporter = NoPHITelemetryExporter()
        if not isinstance(exporter, NoPHITelemetryExporter):
            raise FederatedTelemetryError(
                "federated telemetry requires a no-PHI telemetry exporter"
            )
        self._exporter = exporter

    @property
    def exporter(self) -> NoPHITelemetryExporter:
        """Return the underlying local exporter."""

        return self._exporter

    @property
    def minimum_group_size(self) -> int:
        """Return the smallest update count that may be released exactly."""

        return self._minimum_group_size

    def record_phase_transition(
        self,
        *,
        phase: str,
        status: str = "success",
    ) -> None:
        """Count one observed transition into ``phase``.

        Callers should validate lifecycle legality with
        ``openmed.training.federated_round.validate_round_transition`` before
        recording; this recorder only accepts the closed phase vocabulary.
        """

        safe_phase = _require_closed_value(phase, FEDERATED_PHASE_VALUES, field="phase")
        safe_status = _require_closed_value(
            status, FEDERATED_PHASE_STATUS_VALUES, field="status"
        )
        self._exporter.increment(
            CounterName.FEDERATED_PHASE_TRANSITIONS,
            dimensions={
                DimensionName.PHASE.value: safe_phase,
                DimensionName.STATUS.value: safe_status,
            },
        )

    def record_rejection(
        self,
        *,
        reason_code: str,
        phase: str,
    ) -> None:
        """Count one rejected or held round using a closed reason code."""

        safe_reason = _require_closed_value(
            reason_code, FEDERATED_REASON_CODE_VALUES, field="reason_code"
        )
        safe_phase = _require_closed_value(phase, FEDERATED_PHASE_VALUES, field="phase")
        self._exporter.increment(
            CounterName.FEDERATED_PHASE_REJECTIONS,
            dimensions={
                DimensionName.PHASE.value: safe_phase,
                DimensionName.REASON_CODE.value: safe_reason,
            },
        )

    def record_update_count(
        self,
        *,
        count: int,
        phase: str,
        exact: bool = False,
    ) -> None:
        """Count round updates for ``phase`` as a coarse band.

        With ``exact=False`` the counter advances by one and the sample carries
        the ``update_band`` label, so the series counts rounds per band.  With
        ``exact=True`` the counter advances by the released count and the sample
        carries no band, so that series sums updates.  Releasing an exact count
        below the minimum group size is refused with
        :class:`UnapprovedFederatedValueError` instead of being rounded
        silently.  A zero count records nothing.
        """

        safe_count = _require_count(count)
        safe_phase = _require_closed_value(phase, FEDERATED_PHASE_VALUES, field="phase")
        if type(exact) is not bool:
            raise FederatedTelemetryError("federated exact update flag is invalid")
        if safe_count == 0:
            return
        if exact and safe_count < self._minimum_group_size:
            raise UnapprovedFederatedValueError(
                "federated update counts below the minimum group must be suppressed"
            )
        dimensions = {DimensionName.PHASE.value: safe_phase}
        if exact:
            amount = safe_count
        else:
            amount = 1
            band = _band(safe_count, self._minimum_group_size)
            dimensions[DimensionName.UPDATE_BAND.value] = band.value
        self._exporter.increment(
            CounterName.FEDERATED_PHASE_UPDATES,
            amount=amount,
            dimensions=dimensions,
        )

    def observe_phase_latency(
        self,
        *,
        phase: str,
        status: str = "success",
        seconds: float | None = None,
        milliseconds: float | None = None,
    ) -> None:
        """Record one phase latency in exactly one unit."""

        if seconds is not None and milliseconds is not None:
            raise FederatedTelemetryError("federated phase latency has multiple units")
        if seconds is None and milliseconds is None:
            raise FederatedTelemetryError(
                "federated phase latency requires a measurement"
            )
        safe_phase = _require_closed_value(phase, FEDERATED_PHASE_VALUES, field="phase")
        safe_status = _require_closed_value(
            status, FEDERATED_PHASE_STATUS_VALUES, field="status"
        )
        dimensions = {
            DimensionName.PHASE.value: safe_phase,
            DimensionName.STATUS.value: safe_status,
        }
        if milliseconds is not None:
            self._exporter.observe_latency_ms(
                milliseconds,
                dimensions=dimensions,
                name=FEDERATED_PHASE_LATENCY_NAME,
            )
            return
        self._exporter.observe_latency_seconds(
            seconds,
            dimensions=dimensions,
            name=FEDERATED_PHASE_LATENCY_NAME,
        )

    def clear(self) -> None:
        """Remove all in-memory samples from the underlying exporter."""

        self._exporter.clear()

    def snapshot(self) -> TelemetrySnapshot:
        """Return the current aggregate snapshot."""

        return self._exporter.snapshot()

    def export(self) -> dict[str, Any]:
        """Return a fresh deterministic dictionary with no external I/O."""

        return self._exporter.export()

    def export_json(self) -> str:
        """Return the canonical JSON representation of :meth:`export`."""

        return self._exporter.export_json()

    def render_prometheus(self) -> str:
        """Render the snapshot as deterministic Prometheus text.

        This is formatting only.  The recorder never configures or contacts a
        collector.
        """

        return self._exporter.render_prometheus()


def _require_closed_value(
    value: object,
    allowed: tuple[str, ...],
    *,
    field: str,
) -> str:
    """Return a closed-vocabulary value without echoing rejected input."""

    if type(value) is str and value in allowed:
        return value
    raise UnapprovedFederatedValueError(
        f"federated telemetry rejected an unapproved {field}"
    )


def _require_count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise FederatedTelemetryError(
            "federated update count must be a non-negative integer"
        )
    return value


def _require_minimum_group_size(value: object) -> int:
    if type(value) is not int or value < 2:
        raise FederatedTelemetryError(
            "minimum group size must be an integer greater than one"
        )
    return value


def _band(count: int, minimum_group_size: int) -> FederatedUpdateBand:
    if count < minimum_group_size:
        return FederatedUpdateBand.SUPPRESSED
    if count < minimum_group_size * 2:
        return FederatedUpdateBand.MINIMUM_TO_UNDER_DOUBLE
    if count < minimum_group_size * 4:
        return FederatedUpdateBand.DOUBLE_TO_UNDER_FOURFOLD
    return FederatedUpdateBand.FOURFOLD_OR_MORE


__all__ = [
    "FEDERATED_PHASE_STATUS_VALUES",
    "FEDERATED_TELEMETRY_MINIMUM_GROUP_SIZE",
    "FederatedRoundTelemetry",
    "FederatedTelemetryError",
    "FederatedUpdateBand",
    "UnapprovedFederatedValueError",
    "band_update_count",
]
