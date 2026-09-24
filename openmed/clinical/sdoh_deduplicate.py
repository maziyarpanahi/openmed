"""Deterministic, provenance-preserving SDOH evidence deduplication.

Protected observation text is used only to compute one-way fingerprints. It is
never retained in a cluster, serialized, or interpolated into an exception.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Final

SDOH_DEDUPLICATION_SCHEMA_VERSION: Final = 1
SDOH_DEDUPLICATION_ADVISORY: Final = (
    "Duplicate SDOH evidence is a review aid; clusters retain provenance and "
    "must not be treated as independent clinical observations."
)

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SDOHSourceReference:
    """A value-free, document-local reference to source evidence."""

    source_id: str
    version_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _safe_identifier(self.source_id))
        object.__setattr__(self, "version_id", _safe_identifier(self.version_id))
        start, end = _offsets(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic value-free source reference."""

        return {
            "source_id": self.source_id,
            "version_id": self.version_id,
            "source_offsets": {"start": self.start, "end": self.end},
        }


@dataclass(frozen=True, slots=True)
class SDOHEvidenceObservation:
    """One SDOH observation supplied to the deduplicator.

    ``protected_text`` is excluded from representations and is discarded when
    the immutable output clusters are constructed.
    """

    observation_id: str
    category: str
    status: str | None
    source: SDOHSourceReference
    protected_text: str = field(repr=False)
    temporality: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _safe_identifier(self.observation_id)
        )
        object.__setattr__(self, "category", _safe_identifier(self.category))
        object.__setattr__(self, "status", _optional_identifier(self.status))
        object.__setattr__(self, "temporality", _optional_identifier(self.temporality))
        if not isinstance(self.source, SDOHSourceReference):
            raise TypeError("source must be an SDOHSourceReference")
        if not isinstance(self.protected_text, str):
            raise TypeError("protected_text must be a string")
        if not self.protected_text.strip():
            raise ValueError("protected_text must not be empty")


@dataclass(frozen=True, slots=True)
class SDOHEvidenceCluster:
    """A value-free duplicate cluster retaining every evidence reference."""

    cluster_id: str
    category: str
    status: str | None
    temporality: str | None
    duplicate_kind: str
    exact_fingerprints: tuple[str, ...]
    normalized_fingerprint: str
    observation_ids: tuple[str, ...]
    source_references: tuple[SDOHSourceReference, ...]

    @property
    def evidence_count(self) -> int:
        """Return the number of source observations in this cluster."""

        return len(self.observation_ids)

    @property
    def independent_evidence_count(self) -> int:
        """Return the non-inflated contribution of this cluster."""

        return 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible cluster containing no protected text."""

        return {
            "cluster_id": self.cluster_id,
            "category": self.category,
            "status": self.status,
            "temporality": self.temporality,
            "duplicate_kind": self.duplicate_kind,
            "evidence_count": self.evidence_count,
            "independent_evidence_count": self.independent_evidence_count,
            "exact_fingerprints": list(self.exact_fingerprints),
            "normalized_fingerprint": self.normalized_fingerprint,
            "observation_ids": list(self.observation_ids),
            "source_references": [
                reference.to_dict() for reference in self.source_references
            ],
        }


@dataclass(frozen=True, slots=True)
class SDOHDeduplicationResult:
    """Deterministically ordered duplicate clusters and aggregate counts."""

    clusters: tuple[SDOHEvidenceCluster, ...]
    schema_version: int = SDOH_DEDUPLICATION_SCHEMA_VERSION

    @property
    def evidence_count(self) -> int:
        """Return the original observation count."""

        return sum(cluster.evidence_count for cluster in self.clusters)

    @property
    def independent_evidence_count(self) -> int:
        """Return the duplicate-safe evidence count."""

        return len(self.clusters)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, value-free deduplication report."""

        return {
            "schema_version": self.schema_version,
            "evidence_count": self.evidence_count,
            "independent_evidence_count": self.independent_evidence_count,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "advisory": SDOH_DEDUPLICATION_ADVISORY,
        }


def deduplicate_sdoh_evidence(
    observations: Iterable[SDOHEvidenceObservation],
) -> SDOHDeduplicationResult:
    """Cluster exact and normalized copied-forward observations.

    Category, status, and temporality remain part of the identity, so
    semantically conflicting observations are never collapsed together.
    """

    grouped: dict[
        tuple[str, str | None, str | None, str],
        list[tuple[SDOHEvidenceObservation, str]],
    ] = {}
    for observation in observations:
        if not isinstance(observation, SDOHEvidenceObservation):
            raise TypeError("observations must contain SDOHEvidenceObservation values")
        normalized = normalize_sdoh_evidence_text(observation.protected_text)
        key = (
            observation.category,
            observation.status,
            observation.temporality,
            normalized,
        )
        grouped.setdefault(key, []).append(
            (observation, _fingerprint(observation.protected_text))
        )

    clusters: list[SDOHEvidenceCluster] = []
    for key in sorted(grouped, key=_group_sort_key):
        category, status, temporality, normalized = key
        members = sorted(grouped[key], key=lambda item: _observation_sort_key(item[0]))
        exact_fingerprints = tuple(sorted({fingerprint for _, fingerprint in members}))
        if len(members) == 1:
            duplicate_kind = "unique"
        elif len(exact_fingerprints) == 1:
            duplicate_kind = "exact"
        else:
            duplicate_kind = "normalized"
        normalized_fingerprint = _fingerprint(normalized)
        cluster_id = _fingerprint(
            "\x1f".join(
                (
                    category,
                    status or "",
                    temporality or "",
                    normalized_fingerprint,
                )
            )
        )
        clusters.append(
            SDOHEvidenceCluster(
                cluster_id=cluster_id,
                category=category,
                status=status,
                temporality=temporality,
                duplicate_kind=duplicate_kind,
                exact_fingerprints=exact_fingerprints,
                normalized_fingerprint=normalized_fingerprint,
                observation_ids=tuple(item.observation_id for item, _ in members),
                source_references=tuple(item.source for item, _ in members),
            )
        )
    return SDOHDeduplicationResult(clusters=tuple(clusters))


def normalize_sdoh_evidence_text(value: str) -> str:
    """Return the deterministic comparison form used for duplicate matching."""

    if not isinstance(value, str):
        raise TypeError("evidence text must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _TOKEN_RE.sub(" ", normalized)
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("evidence text must contain a comparable token")
    return normalized


def _fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _safe_identifier(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("identifier must be a string")
    if _SAFE_IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError("identifier must use the safe opaque-reference format")
    return value


def _optional_identifier(value: object) -> str | None:
    if value is None:
        return None
    return _safe_identifier(value)


def _offsets(start: object, end: object) -> tuple[int, int]:
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise TypeError("source offsets must be integers")
    if start < 0 or end <= start:
        raise ValueError("source offsets must form a non-empty half-open span")
    return start, end


def _observation_sort_key(
    observation: SDOHEvidenceObservation,
) -> tuple[str, str, int, int, str]:
    source = observation.source
    return (
        source.source_id,
        source.version_id,
        source.start,
        source.end,
        observation.observation_id,
    )


def _group_sort_key(
    key: tuple[str, str | None, str | None, str],
) -> tuple[str, str, str, str]:
    category, status, temporality, normalized = key
    return category, status or "", temporality or "", normalized


__all__ = [
    "SDOH_DEDUPLICATION_ADVISORY",
    "SDOH_DEDUPLICATION_SCHEMA_VERSION",
    "SDOHDeduplicationResult",
    "SDOHEvidenceCluster",
    "SDOHEvidenceObservation",
    "SDOHSourceReference",
    "deduplicate_sdoh_evidence",
    "normalize_sdoh_evidence_text",
]
