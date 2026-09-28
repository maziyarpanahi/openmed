"""Deterministic longitudinal resolution for value-free SDOH evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Final

from .sdoh_deduplicate import SDOHSourceReference

SDOH_LONGITUDINAL_SCHEMA_VERSION: Final = 1
SDOH_LONGITUDINAL_ADVISORY: Final = (
    "Longitudinal SDOH status is evidence-backed review metadata, not an "
    "eligibility, diagnosis, or autonomous clinical decision."
)

SAME_TIME_STATUS_CONFLICT: Final = "same_time_status_conflict"
LATEST_EPISODE_UNRESOLVED: Final = "latest_episode_unresolved"
NO_CURRENT_SUPPORT: Final = "no_current_support"

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


@dataclass(frozen=True, slots=True)
class SDOHStatusObservation:
    """One dated, value-free SDOH status observation."""

    observation_id: str
    category: str
    status: str
    effective_at: date | datetime
    source: SDOHSourceReference
    supports_current: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _identifier(self.observation_id))
        object.__setattr__(self, "category", _identifier(self.category))
        object.__setattr__(self, "status", _identifier(self.status))
        object.__setattr__(self, "effective_at", _utc_datetime(self.effective_at))
        if not isinstance(self.source, SDOHSourceReference):
            raise TypeError("source must be an SDOHSourceReference")
        if type(self.supports_current) is not bool:
            raise TypeError("supports_current must be a boolean")


@dataclass(frozen=True, slots=True)
class SDOHStatusEpisode:
    """All observations recorded for one category at one effective time."""

    category: str
    effective_at: datetime
    observations: tuple[SDOHStatusObservation, ...]
    resolved_status: str | None
    conflict_codes: tuple[str, ...]

    @property
    def source_references(self) -> tuple[SDOHSourceReference, ...]:
        """Return every source reference retained by the episode."""

        return tuple(observation.source for observation in self.observations)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic episode representation."""

        return {
            "category": self.category,
            "effective_at": _format_datetime(self.effective_at),
            "resolved_status": self.resolved_status,
            "conflict_codes": list(self.conflict_codes),
            "observation_ids": [
                observation.observation_id for observation in self.observations
            ],
            "source_references": [
                reference.to_dict() for reference in self.source_references
            ],
        }


@dataclass(frozen=True, slots=True)
class SDOHLongitudinalResult:
    """Status episodes and an explicitly supported current status."""

    category: str
    episodes: tuple[SDOHStatusEpisode, ...]
    current_status: str | None
    current_source_references: tuple[SDOHSourceReference, ...]
    conflict_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic longitudinal review report."""

        return {
            "schema_version": SDOH_LONGITUDINAL_SCHEMA_VERSION,
            "category": self.category,
            "current_status": self.current_status,
            "current_source_references": [
                reference.to_dict() for reference in self.current_source_references
            ],
            "conflict_codes": list(self.conflict_codes),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "advisory": SDOH_LONGITUDINAL_ADVISORY,
        }


def resolve_sdoh_longitudinal_status(
    observations: Iterable[SDOHStatusObservation],
) -> tuple[SDOHLongitudinalResult, ...]:
    """Resolve observations into dated episodes without discarding conflict.

    A current status is emitted only when the latest episode is unambiguous and
    at least one observation in that episode explicitly supports current use.
    """

    grouped: dict[tuple[str, datetime], list[SDOHStatusObservation]] = {}
    for observation in observations:
        if not isinstance(observation, SDOHStatusObservation):
            raise TypeError("observations must contain SDOHStatusObservation values")
        effective_at = _utc_datetime(observation.effective_at)
        grouped.setdefault((observation.category, effective_at), []).append(observation)

    categories = sorted({category for category, _ in grouped})
    results: list[SDOHLongitudinalResult] = []
    for category in categories:
        episodes: list[SDOHStatusEpisode] = []
        for (_, effective_at), members in sorted(
            ((key, value) for key, value in grouped.items() if key[0] == category),
            key=lambda item: item[0][1],
        ):
            ordered = tuple(sorted(members, key=_observation_key))
            statuses = {observation.status for observation in ordered}
            conflict_codes = (SAME_TIME_STATUS_CONFLICT,) if len(statuses) > 1 else ()
            episodes.append(
                SDOHStatusEpisode(
                    category=category,
                    effective_at=effective_at,
                    observations=ordered,
                    resolved_status=next(iter(statuses))
                    if len(statuses) == 1
                    else None,
                    conflict_codes=conflict_codes,
                )
            )

        current_status: str | None = None
        current_sources: tuple[SDOHSourceReference, ...] = ()
        result_codes: set[str] = {
            code for episode in episodes for code in episode.conflict_codes
        }
        if episodes:
            latest = episodes[-1]
            if latest.resolved_status is None:
                result_codes.add(LATEST_EPISODE_UNRESOLVED)
            elif any(
                observation.supports_current for observation in latest.observations
            ):
                current_status = latest.resolved_status
                current_sources = tuple(
                    observation.source
                    for observation in latest.observations
                    if observation.supports_current
                )
            else:
                result_codes.add(NO_CURRENT_SUPPORT)
        results.append(
            SDOHLongitudinalResult(
                category=category,
                episodes=tuple(episodes),
                current_status=current_status,
                current_source_references=current_sources,
                conflict_codes=tuple(sorted(result_codes)),
            )
        )
    return tuple(results)


def _identifier(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("identifier must be a string")
    if _SAFE_IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError("identifier must use the safe controlled-value format")
    return value


def _utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, time.min)
    else:
        raise TypeError("effective_at must be a date or datetime")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _format_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _observation_key(
    observation: SDOHStatusObservation,
) -> tuple[str, str, int, int, str, str]:
    source = observation.source
    return (
        source.source_id,
        source.version_id,
        source.start,
        source.end,
        observation.observation_id,
        observation.status,
    )


__all__ = [
    "LATEST_EPISODE_UNRESOLVED",
    "NO_CURRENT_SUPPORT",
    "SAME_TIME_STATUS_CONFLICT",
    "SDOH_LONGITUDINAL_ADVISORY",
    "SDOH_LONGITUDINAL_SCHEMA_VERSION",
    "SDOHLongitudinalResult",
    "SDOHStatusEpisode",
    "SDOHStatusObservation",
    "resolve_sdoh_longitudinal_status",
]
