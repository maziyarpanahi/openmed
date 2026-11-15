"""Session-local speaker references with explicit acoustic-cluster lineage.

No names, audio, embeddings or inferred clinical roles enter this registry.
Adapters must supply opaque cluster references and explicit revision events;
similarity matching and identity verification remain outside this contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "ALIAS_SCHEMA_VERSION",
    "ClinicalRole",
    "LineageEvent",
    "LineageKind",
    "RoleReview",
    "SpeakerAliasError",
    "SpeakerAliases",
]

ALIAS_SCHEMA_VERSION = "openmed.speech.aliases.v1"
MAX_CLUSTERS = 4096
MAX_ACTIVE_CLUSTERS = 256
MAX_LINEAGE_EVENTS = 4096
_REF = re.compile(r"[0-9a-f]{32}\Z")
_ALIAS = re.compile(r"speaker_[0-9]{4}\Z")


class SpeakerAliasError(ValueError):
    """Value-free failure for invalid or stale session lineage."""


class LineageKind(str, Enum):
    """Explicit changes reported by the local acoustic clustering adapter."""

    RELABEL = "relabel"
    MERGE = "merge"
    SPLIT = "split"


class ClinicalRole(str, Enum):
    """Reviewer-confirmed role vocabulary, distinct from personal identity."""

    PATIENT = "patient"
    CLINICIAN = "clinician"
    CAREGIVER = "caregiver"
    INTERPRETER = "interpreter"
    OTHER = "other"


def _require(condition: bool) -> None:
    if not condition:
        raise SpeakerAliasError("invalid speaker alias transition")


def _ref(value: object) -> None:
    _require(type(value) is str and _REF.fullmatch(value) is not None)


def _revision(value: object) -> None:
    _require(type(value) is int and 0 <= value <= 1_000_000)


def _refs(values: object, maximum: int) -> None:
    _require(type(values) is tuple and len(values) <= maximum)
    assert isinstance(values, tuple)
    for value in values:
        _ref(value)
    _require(len(set(values)) == len(values))


@dataclass(frozen=True, slots=True, kw_only=True)
class LineageEvent:
    """Explicit merge, split or relabel of opaque acoustic cluster references.

    Args:
        kind: One-to-one relabel, many-to-one merge, or one-to-many split.
        parents: Existing, non-retired cluster references.
        children: New cluster references present in the incoming revision.
    """

    kind: LineageKind
    parents: tuple[str, ...]
    children: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(type(self.kind) is LineageKind)
        _refs(self.parents, MAX_ACTIVE_CLUSTERS)
        _refs(self.children, MAX_ACTIVE_CLUSTERS)
        counts = (len(self.parents), len(self.children))
        _require(
            (self.kind is LineageKind.RELABEL and counts == (1, 1))
            or (self.kind is LineageKind.MERGE and counts[0] >= 2 and counts[1] == 1)
            or (self.kind is LineageKind.SPLIT and counts[0] == 1 and counts[1] >= 2)
        )
        _require(not set(self.parents).intersection(self.children))
        object.__setattr__(self, "parents", tuple(sorted(self.parents)))
        object.__setattr__(self, "children", tuple(sorted(self.children)))


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleReview:
    """Explicit reviewer attestation bound to a session and revision.

    Args:
        session_ref: Opaque session reference, never a patient or encounter ID.
        alias: Current session alias being reviewed.
        role: Confirmed role; this does not authenticate the speaker's identity.
        revision: Exact current diarization revision viewed by the reviewer.
        review_ref: Opaque reference to the caller's separately protected approval.
    """

    session_ref: str
    alias: str
    role: ClinicalRole
    revision: int
    review_ref: str

    def __post_init__(self) -> None:
        _ref(self.session_ref)
        _ref(self.review_ref)
        _require(type(self.alias) is str and _ALIAS.fullmatch(self.alias) is not None)
        _require(type(self.role) is ClinicalRole)
        _revision(self.revision)


class SpeakerAliases:
    """Bounded session-scoped alias registry with atomic revision updates.

    Args:
        session_ref: Caller-assigned opaque 128-bit lowercase hexadecimal reference.

    A relabel retains its alias. A merge keeps the earliest allocated parent alias
    and records the retired aliases. A split retires the ambiguous parent alias
    and allocates fresh child aliases. Merge/split events invalidate affected role
    approvals. Historical transcript references are never silently rewritten.

    Callers must serialize access to one instance. This registry does not infer
    lineage, authenticate reviewers or keep biometric embeddings.
    """

    def __init__(self, session_ref: str) -> None:
        _ref(session_ref)
        self._session_ref = session_ref
        self._revision = -1
        self._clusters: dict[str, str] = {}
        self._retired: set[str] = set()
        self._active: tuple[str, ...] = ()
        self._roles: dict[str, RoleReview] = {}
        self._events: list[dict[str, Any]] = []
        self._next_alias = 1
        self._closed = False

    def apply_revision(
        self,
        revision: int,
        active_clusters: tuple[str, ...],
        events: tuple[LineageEvent, ...] = (),
    ) -> None:
        """Apply a complete active-cluster snapshot and explicit lineage atomically.

        Args:
            revision: Strictly increasing revision, starting at zero or later.
            active_clusters: Opaque references observed in this revision.
            events: Disjoint transitions; children cannot be parents in this batch.

        Unseen clusters without lineage are new speakers, not inferred matches.
        Temporarily absent clusters retain aliases and may return unless retired.
        A retired cluster cannot be reused. Invalid transitions change no state.
        """
        _require(not self._closed)
        _revision(revision)
        _require(revision > self._revision)
        _refs(active_clusters, MAX_ACTIVE_CLUSTERS)
        _require(type(events) is tuple and len(events) <= MAX_ACTIVE_CLUSTERS)
        _require(all(type(event) is LineageEvent for event in events))
        _require(len(self._events) + len(events) <= MAX_LINEAGE_EVENTS)
        active = set(active_clusters)
        _require(not active.intersection(self._retired))
        new = active.difference(self._clusters)
        _require(len(self._clusters) + len(new) <= MAX_CLUSTERS)
        parents: set[str] = set()
        children: set[str] = set()
        for event in events:
            _require(not parents.intersection(event.parents))
            _require(not children.intersection(event.children))
            _require(
                all(
                    parent in self._clusters and parent not in self._retired
                    for parent in event.parents
                )
            )
            _require(set(event.children).issubset(new))
            _require(not active.intersection(event.parents))
            parents.update(event.parents)
            children.update(event.children)
        _require(not parents.intersection(children))

        # Prepare a candidate state first; failures never consume aliases or roles.
        clusters = self._clusters.copy()
        retired = self._retired.copy()
        roles = self._roles.copy()
        history = self._events.copy()
        counter = self._next_alias

        def allocate() -> str:
            nonlocal counter
            _require(counter <= MAX_CLUSTERS)
            alias = f"speaker_{counter:04d}"
            counter += 1
            return alias

        ordered = sorted(
            events, key=lambda event: (event.parents, event.children, event.kind)
        )
        for event in ordered:
            parent_aliases = tuple(
                sorted({clusters[parent] for parent in event.parents})
            )
            _require(len(parent_aliases) == len(event.parents))
            if event.kind is LineageKind.RELABEL:
                clusters[event.children[0]] = parent_aliases[0]
            elif event.kind is LineageKind.MERGE:
                clusters[event.children[0]] = parent_aliases[0]
            else:
                for child in event.children:
                    clusters[child] = allocate()
            retired.update(event.parents)
            if event.kind is not LineageKind.RELABEL:
                for alias in parent_aliases:
                    roles.pop(alias, None)
            history.append(
                {
                    "revision": revision,
                    "kind": event.kind.value,
                    "parent_clusters": event.parents,
                    "child_clusters": event.children,
                    "parent_aliases": parent_aliases,
                    "child_aliases": tuple(clusters[child] for child in event.children),
                }
            )
        for cluster in sorted(new.difference(children)):
            clusters[cluster] = allocate()
        self._clusters, self._retired, self._roles = clusters, retired, roles
        self._events, self._next_alias = history, counter
        self._active, self._revision = tuple(sorted(active)), revision

    def alias_for(self, cluster_ref: str) -> str:
        """Return the current alias for an active cluster; reject stale references."""
        _ref(cluster_ref)
        _require(not self._closed and cluster_ref in self._active)
        return self._clusters[cluster_ref]

    def confirm_role(self, review: RoleReview) -> None:
        """Attach an explicit approval only to a current alias in this session.

        The application must authenticate the reviewer and obtain confirmation
        before constructing this attestation. A guessed role is not confirmation.
        """
        _require(not self._closed and type(review) is RoleReview)
        _require(
            review.session_ref == self._session_ref
            and review.revision == self._revision
        )
        _require(review.alias in {self._clusters[cluster] for cluster in self._active})
        self._roles[review.alias] = review

    def to_dict(self) -> dict[str, Any]:
        """Return opaque lineage and confirmed roles, with no identity claims."""
        active = []
        for cluster in self._active:
            alias = self._clusters[cluster]
            role = self._roles.get(alias)
            active.append(
                {
                    "cluster_ref": cluster,
                    "alias": alias,
                    "role": role.role.value if role else None,
                    "role_review_ref": role.review_ref if role else None,
                    "role_review_revision": role.revision if role else None,
                }
            )
        return {
            "schema_version": ALIAS_SCHEMA_VERSION,
            "session_ref": self._session_ref,
            "revision": self._revision,
            "closed": self._closed,
            "active": active,
            "lineage": [event.copy() for event in self._events],
        }

    def to_json(self) -> str:
        """Return deterministic ASCII JSON containing no acoustic or name data."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def close(self) -> None:
        """Discard alias, lineage and role state; further updates are rejected."""
        self._clusters.clear()
        self._retired.clear()
        self._roles.clear()
        self._events.clear()
        self._active = ()
        self._closed = True
