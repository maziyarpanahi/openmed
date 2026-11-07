"""Bounded local acoustic observations, separate from role and identity review."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Protocol

from .speaker_aliases import LineageEvent, SpeakerAliasError, SpeakerAliases

MAX_AUDIO_BYTES = 60 * 16000 * 2
MAX_TURNS = 1024
MAX_CLUSTERS = 32
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REF = re.compile(r"[0-9a-f]{32}\Z")


class DiarizationError(ValueError):
    """Controlled failure with no audio, model path or provider exception text."""


def require(condition: bool) -> None:
    """Reject invalid contract values without interpolating source data."""
    if not condition:
        raise DiarizationError("invalid_diarization")


def digest(value: object) -> str:
    """Hash a bounded canonical structure for request and provenance binding."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class DiarizationRequest:
    """One caller-owned mono PCM16LE window at 16 kHz, up to sixty seconds.

    Args:
        pcm: Protected audio; never included in repr or reports.
        revision: Strictly increasing session revision.
        offset_samples: Window start on the caller's recording timeline.
        request_ref: Random, session-local opaque reference, never a patient ID.
    """

    pcm: bytes = field(repr=False)
    revision: int
    offset_samples: int
    request_ref: str

    def __post_init__(self) -> None:
        require(type(self.pcm) is bytes and 0 < len(self.pcm) <= MAX_AUDIO_BYTES)
        require(len(self.pcm) % 2 == 0)
        require(type(self.revision) is int and 0 <= self.revision <= 1_000_000)
        require(
            type(self.offset_samples) is int
            and 0 <= self.offset_samples <= 16000 * 86400
        )
        require(
            type(self.request_ref) is str and bool(_REF.fullmatch(self.request_ref))
        )

    @property
    def sample_count(self) -> int:
        """Return the number of mono samples."""
        return len(self.pcm) // 2

    @property
    def binding(self) -> str:
        """Bind observations to the exact audio, revision and timeline."""
        return digest(
            [
                hashlib.sha256(self.pcm).hexdigest(),
                self.revision,
                self.offset_samples,
                self.request_ref,
                16000,
            ]
        )


class ClusterEvidence(str, Enum):
    """Acoustic uncertainty only; neither state confirms a clinical role."""

    UNREVIEWED = "unreviewed"
    UNSTABLE = "unstable"


@dataclass(frozen=True, slots=True, kw_only=True)
class AcousticTurn:
    """Half-open window-relative sample interval and opaque cluster reference."""

    start_sample: int
    end_sample: int
    cluster_ref: str
    evidence: ClusterEvidence = ClusterEvidence.UNREVIEWED
    boundary_clipped: bool = False

    def __post_init__(self) -> None:
        require(type(self.start_sample) is int and type(self.end_sample) is int)
        require(0 <= self.start_sample < self.end_sample <= MAX_AUDIO_BYTES // 2)
        require(
            type(self.cluster_ref) is str and bool(_REF.fullmatch(self.cluster_ref))
        )
        require(type(self.evidence) is ClusterEvidence)
        require(type(self.boundary_clipped) is bool)
        require(not self.boundary_clipped or self.evidence is ClusterEvidence.UNSTABLE)


@dataclass(frozen=True, slots=True, kw_only=True)
class DiarizationBatch:
    """Complete observations for one request, including explicit cluster changes.

    Args:
        request_binding: Exact request binding.
        provenance_digest: Hash of pinned provider/artifact/terms metadata.
        turns: Bounded observations; no text, embeddings or role hypotheses.
        lineage: Explicit acoustic lineage only. Never infer it from turn order.
    """

    request_binding: str
    provenance_digest: str
    turns: tuple[AcousticTurn, ...]
    lineage: tuple[LineageEvent, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.request_binding, self.provenance_digest):
            require(type(value) is str and bool(_HEX.fullmatch(value)))
        require(type(self.turns) is tuple and len(self.turns) <= MAX_TURNS)
        require(all(type(turn) is AcousticTurn for turn in self.turns))
        require(len({turn.cluster_ref for turn in self.turns}) <= MAX_CLUSTERS)
        require(type(self.lineage) is tuple and len(self.lineage) <= MAX_CLUSTERS)
        require(all(type(event) is LineageEvent for event in self.lineage))


class DiarizationProvider(Protocol):
    """Synchronous local provider; implementations document execution enforcement.

    The native adapter enforces limits in a disposable process. An injected
    in-process provider remains responsible for honoring cancellation itself.
    """

    def observe(
        self, request: DiarizationRequest, *, cancelled: Event | None = None
    ) -> DiarizationBatch:
        """Return complete observations or a controlled failure, never partial data."""
        ...


class DiarizationSession:
    """Apply observations to private aliases without any role assignment API.

    Args:
        session_ref: Opaque session reference used by the existing alias contract.

    Callers serialize access. No audio is retained after ``apply``. Reports remain
    sensitive metadata and need caller access controls. There is no identity claim,
    role confirmation, or automatic clinical action in this interface.
    """

    def __init__(self, session_ref: str) -> None:
        self._aliases = SpeakerAliases(session_ref)

    def apply(self, request: DiarizationRequest, batch: DiarizationBatch) -> dict:
        """Validate source binding, then apply an atomic acoustic revision.

        Overlap is observed only for intersecting turns from distinct clusters.
        Otherwise its status is unknown, not confirmed absence. Empty observations
        indicate insufficient speech evidence, not successful identification.
        """
        require(type(request) is DiarizationRequest and type(batch) is DiarizationBatch)
        require(batch.request_binding == request.binding)
        require(all(turn.end_sample <= request.sample_count for turn in batch.turns))
        clusters = tuple(sorted({turn.cluster_ref for turn in batch.turns}))
        try:
            self._aliases.apply_revision(request.revision, clusters, batch.lineage)
        except SpeakerAliasError:
            raise DiarizationError("invalid_diarization_transition") from None
        ordered = sorted(
            batch.turns, key=lambda turn: (turn.start_sample, turn.end_sample)
        )
        turns = []
        for turn in ordered:
            overlap = any(
                other.cluster_ref != turn.cluster_ref
                and other.start_sample < turn.end_sample
                and turn.start_sample < other.end_sample
                for other in ordered
            )
            turns.append(
                {
                    "start_sample": request.offset_samples + turn.start_sample,
                    "end_sample": request.offset_samples + turn.end_sample,
                    "cluster_ref": turn.cluster_ref,
                    "alias": self._aliases.alias_for(turn.cluster_ref),
                    "evidence": turn.evidence.value,
                    "boundary_clipped": turn.boundary_clipped,
                    "overlap": "observed" if overlap else "unknown",
                    "confidence": None,
                    "clinical_role": None,
                    "identity": None,
                }
            )
        return {
            "schema_version": "openmed.diarization.v1",
            "request_binding": batch.request_binding,
            "provenance_digest": batch.provenance_digest,
            "revision": request.revision,
            "sample_rate_hz": 16000,
            "status": "unreviewed" if turns else "insufficient_evidence",
            "turns": turns,
            "aliases": self._aliases.to_dict(),
        }

    def close(self) -> None:
        """Discard aliases and reject subsequent revisions."""
        self._aliases.close()
