"""Client-side roster consistency checks before secure-aggregation share release.

Pairwise masking and dropout recovery assume that the coordinator reports the
same enrolled and survivor sets to every client. This module lets an honest
client refuse to contribute or to reveal recovery material when the roster is
padded, inconsistent or presented in more than one version for a round.

The module owns the roster commitment and the client-side checks only.
Signatures, operator attestations and recovery-share storage are injected
ports; no cryptography, network access or key material lives here. Errors and
evidence carry controlled reason codes, counts and digests, never pseudonymous
keys, signatures or share material.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, NoReturn, Protocol

FEDERATED_ROSTER_SCHEMA_VERSION = "openmed.training.federated_roster.v1"
FEDERATED_ROSTER_EVIDENCE_SCHEMA_VERSION = (
    "openmed.training.federated_roster_evidence.v1"
)

_ROSTER_DOMAIN: Final = b"openmed.training.federated_roster.v1\0"
_COSIGN_DOMAIN: Final = b"openmed.training.federated_roster.cosign.v1\0"
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_KEY: Final = re.compile(r"[A-Za-z0-9._:+/=-]{1,256}\Z")
_MAX_SIGNATURE_BYTES: Final = 4096
_MAX_SHARE_BYTES: Final = 65536
_PORT_FAILED: Final = object()


class FederatedRosterReasonCode(str, Enum):
    """Stable, content-free reasons for refusing a roster or reveal."""

    INVALID_ROSTER = "invalid_roster"
    INVALID_REQUEST = "invalid_request"
    ROUND_MISMATCH = "round_mismatch"
    OUT_OF_ORDER = "out_of_order"
    LOCAL_NOT_ENROLLED = "local_not_enrolled"
    ATTESTED_FLOOR_NOT_MET = "attested_identity_floor_not_met"
    ROSTER_EQUIVOCATION = "roster_equivocation"
    SURVIVOR_SET_EQUIVOCATION = "survivor_set_equivocation"
    LOCAL_NOT_SURVIVOR = "local_not_survivor"
    INSUFFICIENT_COSIGNATURES = "insufficient_cosignatures"
    DOUBLE_MASK_REVEAL = "double_mask_reveal"
    MASK_KIND_MISMATCH = "mask_kind_mismatch"
    PORT_FAILURE = "port_failure"
    ROUND_ABORTED = "round_aborted"

    def __str__(self) -> str:
        return self.value


class FederatedRosterGuardState(str, Enum):
    """Per-round progress of one client's roster guard."""

    OPEN = "open"
    SUBMITTED = "submitted"
    COSIGNED = "cosigned"
    RELEASED = "released"
    ABORTED = "aborted"

    def __str__(self) -> str:
        return self.value


class FederatedMaskShareKind(str, Enum):
    """The two kinds of recovery material a client holds for each peer."""

    SELF_MASK = "self_mask"
    PAIRWISE_MASK = "pairwise_mask"

    def __str__(self) -> str:
        return self.value


class FederatedRosterError(ValueError):
    """Raised with a fixed message when a roster check refuses an action.

    Attributes:
        reason: The controlled reason code for the refusal.
    """

    def __init__(self, reason: FederatedRosterReasonCode) -> None:
        self.reason = reason
        super().__init__(f"federated roster refused: {reason.value}")


class RosterSignatureVerifier(Protocol):
    """Injected port that verifies a participant's roster co-signature."""

    def verify(self, signer_key: str, message: bytes, signature: bytes) -> bool:
        """Return ``True`` only for a valid signature by ``signer_key``."""
        ...


class RosterSigner(Protocol):
    """Injected port that signs with the local participant's key."""

    def sign(self, message: bytes) -> bytes:
        """Return a signature over ``message``."""
        ...


class OperatorAttestationRegistry(Protocol):
    """Injected port listing identities an operator has attested out of band."""

    def is_operator_attested(self, participant_key: str) -> bool:
        """Return ``True`` only for an operator-attested participant key."""
        ...


class RecoveryShareSource(Protocol):
    """Injected port that returns the local client's share for one peer."""

    def share_for(self, peer_key: str, kind: FederatedMaskShareKind) -> bytes:
        """Return the locally held recovery share of ``kind`` for ``peer_key``."""
        ...


@dataclass(frozen=True)
class FederatedRoster:
    """The enrolled pseudonymous keys and recovery threshold for one round.

    Keys are canonicalized into sorted order. The threshold must be at least
    two and strictly more than half of the enrolled keys, so two disjoint
    survivor sets can never both collect threshold co-signatures from
    participants that each sign once.

    Attributes:
        round_digest: ``sha256:`` digest identifying the round.
        enrolled_keys: Sorted, unique pseudonymous participant keys.
        threshold: Minimum survivors and co-signatures for recovery.
    """

    round_digest: str
    enrolled_keys: tuple[str, ...] = field(repr=False)
    threshold: int

    def __post_init__(self) -> None:
        _require_digest(self.round_digest)
        keys = _canonical_keys(self.enrolled_keys)
        if not keys:
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
        threshold = self.threshold
        if (
            type(threshold) is not int
            or threshold < 2
            or threshold > len(keys)
            or 2 * threshold <= len(keys)
        ):
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
        object.__setattr__(self, "enrolled_keys", keys)

    def digest(self) -> str:
        """Return the digest of the enrollment roster without a survivor set.

        Returns:
            A ``sha256:`` digest over the canonical enrollment roster.
        """

        return _hash_payload(
            {
                "enrolled_keys": list(self.enrolled_keys),
                "round_digest": self.round_digest,
                "schema_version": FEDERATED_ROSTER_SCHEMA_VERSION,
                "survivor_keys": None,
                "threshold": self.threshold,
            }
        )


@dataclass(frozen=True)
class FederatedRosterCommitment:
    """A roster plus the survivor set that recovery will be computed against.

    Attributes:
        roster: The enrollment roster for the round.
        survivor_keys: Sorted, unique enrolled keys reported as survivors.
    """

    roster: FederatedRoster
    survivor_keys: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.roster) is not FederatedRoster:
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
        survivors = _canonical_keys(self.survivor_keys)
        if len(survivors) < self.roster.threshold or not set(survivors).issubset(
            self.roster.enrolled_keys
        ):
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
        object.__setattr__(self, "survivor_keys", survivors)

    def digest(self) -> str:
        """Return the canonical roster commitment digest.

        The digest covers the sorted enrolled keys, round digest, threshold and
        sorted survivor set under a versioned domain separator.

        Returns:
            A ``sha256:`` digest string.
        """

        return _hash_payload(
            {
                "enrolled_keys": list(self.roster.enrolled_keys),
                "round_digest": self.roster.round_digest,
                "schema_version": FEDERATED_ROSTER_SCHEMA_VERSION,
                "survivor_keys": list(self.survivor_keys),
                "threshold": self.roster.threshold,
            }
        )


@dataclass(frozen=True)
class FederatedRosterCosignature:
    """One participant's signature over a roster commitment.

    Attributes:
        round_digest: The round the signed commitment belongs to.
        roster_digest: The signed commitment digest.
        signer_key: Pseudonymous key of the signer (hidden from ``repr``).
        signature: Opaque signature bytes (hidden from ``repr``).
    """

    round_digest: str
    roster_digest: str
    signer_key: str = field(repr=False)
    signature: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_digest(self.round_digest)
        _require_digest(self.roster_digest)
        _require_key(self.signer_key)
        if (
            type(self.signature) is not bytes
            or not self.signature
            or len(self.signature) > _MAX_SIGNATURE_BYTES
        ):
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)

    def signing_message(self) -> bytes:
        """Return the exact bytes this co-signature claims to sign."""

        return _signing_message(self.round_digest, self.roster_digest)


@dataclass(frozen=True)
class FederatedShareRequest:
    """A coordinator request for one kind of recovery share for one peer.

    Attributes:
        peer_key: Pseudonymous key of the peer (hidden from ``repr``).
        kind: Which mask material is requested.
    """

    peer_key: str = field(repr=False)
    kind: FederatedMaskShareKind

    def __post_init__(self) -> None:
        _require_key(self.peer_key, FederatedRosterReasonCode.INVALID_REQUEST)
        if not isinstance(self.kind, FederatedMaskShareKind):
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_REQUEST)


@dataclass(frozen=True)
class FederatedReleasedShare:
    """One recovery share approved for release to the coordinator.

    Attributes:
        peer_key: Pseudonymous key of the peer (hidden from ``repr``).
        kind: Which mask material the share recovers.
        share: Opaque share bytes (hidden from ``repr``).
    """

    peer_key: str = field(repr=False)
    kind: FederatedMaskShareKind
    share: bytes = field(repr=False)


@dataclass(frozen=True)
class FederatedShareRelease:
    """Shares approved for one recovery request, bound to one commitment.

    Attributes:
        commitment_digest: Digest of the co-signed roster commitment.
        shares: Released shares in request order.
    """

    commitment_digest: str
    shares: tuple[FederatedReleasedShare, ...] = field(repr=False)


@dataclass(frozen=True)
class FederatedRosterEvidence:
    """Metadata-only record of a roster guard's decisions for one round.

    Attributes:
        state: Current guard state.
        reason_code: The abort reason, or the most recent refusal, if any.
        round_digest: The round the guard is bound to.
        roster_digest: Digest of the admitted enrollment roster, if any.
        commitment_digest: Digest of the co-signed commitment, if any.
        released_share_count: Number of shares released so far.
        schema_version: Evidence schema identifier.
    """

    state: FederatedRosterGuardState
    reason_code: FederatedRosterReasonCode | None
    round_digest: str
    roster_digest: str | None
    commitment_digest: str | None
    released_share_count: int
    schema_version: str = FEDERATED_ROSTER_EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping of codes, counts and digests."""

        return {
            "commitment_digest": self.commitment_digest,
            "reason_code": (
                self.reason_code.value if self.reason_code is not None else None
            ),
            "released_share_count": self.released_share_count,
            "roster_digest": self.roster_digest,
            "round_digest": self.round_digest,
            "schema_version": self.schema_version,
            "state": self.state.value,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def roster_signing_message(commitment: FederatedRosterCommitment) -> bytes:
    """Return the domain-separated bytes participants co-sign for a commitment.

    Args:
        commitment: The roster commitment to sign.

    Returns:
        Bytes binding the round digest and the commitment digest.

    Raises:
        FederatedRosterError: If ``commitment`` is not a roster commitment.
    """

    if type(commitment) is not FederatedRosterCommitment:
        raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
    return _signing_message(commitment.roster.round_digest, commitment.digest())


class FederatedRosterGuard:
    """Stateful client-side guard for one secure-aggregation round.

    The guard walks one client through three checks and remembers what it has
    already seen, so the coordinator cannot show it two versions of a round:

    1. :meth:`authorize_submission` admits one enrollment roster before any
       masked input is submitted and enforces the attested-identity floor.
    2. :meth:`cosign` signs exactly one survivor set for the round.
    3. :meth:`release_shares` requires threshold valid co-signatures on that
       same set, aborts on any validly signed conflicting set, and never
       reveals both self-mask and pairwise-mask material for one peer.

    Any refusal that shows the coordinator is inconsistent moves the guard to
    ``aborted`` permanently. The guard is safe to share between threads.

    Args:
        round_digest: ``sha256:`` digest of the round this guard serves.
        local_key: The local participant's pseudonymous key.
        coordinator_keys: Keys controlled by the coordinator. They never count
            toward the attested floor or the co-signature threshold.
        minimum_attested_peers: Minimum number of operator-attested enrolled
            peers, excluding the local key and coordinator keys.
        verifier: Injected co-signature verifier.
        attestations: Injected operator attestation registry.

    Raises:
        FederatedRosterError: If the configuration is invalid.
    """

    def __init__(
        self,
        *,
        round_digest: str,
        local_key: str,
        coordinator_keys: Iterable[str],
        minimum_attested_peers: int,
        verifier: RosterSignatureVerifier,
        attestations: OperatorAttestationRegistry,
    ) -> None:
        _require_digest(round_digest)
        _require_key(local_key)
        if isinstance(coordinator_keys, (str, bytes)):
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
        coordinators = frozenset(_canonical_keys(coordinator_keys, allow_empty=True))
        if local_key in coordinators:
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
        if type(minimum_attested_peers) is not int or minimum_attested_peers < 1:
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
        if not callable(getattr(verifier, "verify", None)) or not callable(
            getattr(attestations, "is_operator_attested", None)
        ):
            raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
        self._round_digest = round_digest
        self._local_key = local_key
        self._coordinators = coordinators
        self._floor = minimum_attested_peers
        self._verifier = verifier
        self._attestations = attestations
        self._lock = threading.Lock()
        self._state = FederatedRosterGuardState.OPEN
        self._reason: FederatedRosterReasonCode | None = None
        self._roster: FederatedRoster | None = None
        self._commitment: FederatedRosterCommitment | None = None
        self._cosignature: FederatedRosterCosignature | None = None
        self._revealed: dict[str, FederatedMaskShareKind] = {}
        self._released_count = 0

    def __repr__(self) -> str:
        return f"FederatedRosterGuard(state={self._state.value!r})"

    @property
    def state(self) -> FederatedRosterGuardState:
        """The guard's current per-round state."""

        return self._state

    def evidence(self) -> FederatedRosterEvidence:
        """Return metadata-only evidence of this guard's decisions.

        Returns:
            Evidence containing codes, counts and digests only.
        """

        with self._lock:
            return FederatedRosterEvidence(
                state=self._state,
                reason_code=self._reason,
                round_digest=self._round_digest,
                roster_digest=(
                    self._roster.digest() if self._roster is not None else None
                ),
                commitment_digest=(
                    self._commitment.digest() if self._commitment is not None else None
                ),
                released_share_count=self._released_count,
            )

    def authorize_submission(self, roster: FederatedRoster) -> None:
        """Admit the enrollment roster before any masked input is submitted.

        Args:
            roster: The enrollment roster the coordinator presents.

        Raises:
            FederatedRosterError: If the round was aborted, the roster is for
                another round, the local key is not enrolled, fewer attested
                peers than the floor are enrolled, or a different roster was
                already admitted for this round.
        """

        with self._lock:
            self._require_live()
            if type(roster) is not FederatedRoster:
                self._refuse(FederatedRosterReasonCode.INVALID_ROSTER)
            if roster.round_digest != self._round_digest:
                self._refuse(FederatedRosterReasonCode.ROUND_MISMATCH)
            if self._roster is not None:
                if roster != self._roster:
                    self._abort(FederatedRosterReasonCode.ROSTER_EQUIVOCATION)
                return
            if self._local_key not in roster.enrolled_keys:
                self._refuse(FederatedRosterReasonCode.LOCAL_NOT_ENROLLED)
            attested = 0
            for key in roster.enrolled_keys:
                if key == self._local_key or key in self._coordinators:
                    continue
                result = _call_port(self._attestations.is_operator_attested, key)
                if result is _PORT_FAILED:
                    self._abort(FederatedRosterReasonCode.PORT_FAILURE)
                if result is True:
                    attested += 1
            if attested < self._floor:
                self._abort(FederatedRosterReasonCode.ATTESTED_FLOOR_NOT_MET)
            self._roster = roster
            self._state = FederatedRosterGuardState.SUBMITTED

    def cosign(
        self, commitment: FederatedRosterCommitment, signer: RosterSigner
    ) -> FederatedRosterCosignature:
        """Co-sign exactly one survivor set for this round.

        Repeating the call with the same commitment returns the cached
        co-signature without signing again.

        Args:
            commitment: The roster commitment the coordinator presents.
            signer: Injected signer for the local key.

        Returns:
            The local co-signature over the commitment.

        Raises:
            FederatedRosterError: If the submission was not authorized, the
                enrollment roster or survivor set differs from what this guard
                already saw, the local key is reported as dropped, or the
                signer fails.
        """

        with self._lock:
            commitment = self._require_commitment(commitment)
            if self._commitment is not None:
                if commitment != self._commitment:
                    self._abort(FederatedRosterReasonCode.SURVIVOR_SET_EQUIVOCATION)
                if self._cosignature is None:
                    self._abort(FederatedRosterReasonCode.OUT_OF_ORDER)
                return self._cosignature
            if self._local_key not in commitment.survivor_keys:
                self._abort(FederatedRosterReasonCode.LOCAL_NOT_SURVIVOR)
            if not callable(getattr(signer, "sign", None)):
                self._refuse(FederatedRosterReasonCode.INVALID_REQUEST)
            digest = commitment.digest()
            signature = _call_port(
                signer.sign, _signing_message(self._round_digest, digest)
            )
            if (
                type(signature) is not bytes
                or not signature
                or len(signature) > _MAX_SIGNATURE_BYTES
            ):
                self._abort(FederatedRosterReasonCode.PORT_FAILURE)
            cosignature = FederatedRosterCosignature(
                round_digest=self._round_digest,
                roster_digest=digest,
                signer_key=self._local_key,
                signature=signature,
            )
            self._commitment = commitment
            self._cosignature = cosignature
            self._state = FederatedRosterGuardState.COSIGNED
            return cosignature

    def release_shares(
        self,
        commitment: FederatedRosterCommitment,
        cosignatures: Iterable[FederatedRosterCosignature],
        requests: Iterable[FederatedShareRequest],
        share_source: RecoveryShareSource,
    ) -> FederatedShareRelease:
        """Release recovery shares only under one consistent, co-signed roster.

        Every check runs before ``share_source`` is called, and nothing is
        returned unless every requested share was read successfully.

        Args:
            commitment: The commitment the recovery request is computed for.
            cosignatures: Co-signatures collected for the round.
            requests: Requested ``(peer, mask kind)`` shares.
            share_source: Injected store of locally held recovery shares.

        Returns:
            The approved shares, bound to the commitment digest.

        Raises:
            FederatedRosterError: If the round was aborted or not co-signed,
                the commitment differs from the co-signed one, any validly
                signed conflicting survivor set is present, fewer than the
                threshold of distinct survivors co-signed, a request would
                reveal both mask kinds for a peer or a kind inconsistent with
                the survivor set, or a port fails.
        """

        with self._lock:
            commitment = self._require_commitment(commitment)
            if self._commitment is None:
                self._refuse(FederatedRosterReasonCode.OUT_OF_ORDER)
            if commitment != self._commitment:
                self._abort(FederatedRosterReasonCode.SURVIVOR_SET_EQUIVOCATION)
            digest = commitment.digest()
            signers = self._valid_cosigners(commitment, digest, cosignatures)
            if len(signers) < commitment.roster.threshold:
                self._refuse(FederatedRosterReasonCode.INSUFFICIENT_COSIGNATURES)
            approved = self._check_requests(commitment, requests)
            if not callable(getattr(share_source, "share_for", None)):
                self._refuse(FederatedRosterReasonCode.INVALID_REQUEST)
            shares: list[FederatedReleasedShare] = []
            for peer, kind in approved:
                share = _call_port(share_source.share_for, peer, kind)
                if (
                    type(share) is not bytes
                    or not share
                    or len(share) > _MAX_SHARE_BYTES
                ):
                    self._abort(FederatedRosterReasonCode.PORT_FAILURE)
                shares.append(FederatedReleasedShare(peer, kind, share))
            for peer, kind in approved:
                self._revealed[peer] = kind
            self._released_count += len(shares)
            self._state = FederatedRosterGuardState.RELEASED
            self._reason = None
            return FederatedShareRelease(commitment_digest=digest, shares=tuple(shares))

    def _valid_cosigners(
        self,
        commitment: FederatedRosterCommitment,
        digest: str,
        cosignatures: Iterable[FederatedRosterCosignature],
    ) -> set[str]:
        if isinstance(cosignatures, (str, bytes)):
            self._refuse(FederatedRosterReasonCode.INVALID_REQUEST)
        try:
            bundle = tuple(cosignatures)
        except Exception:
            bundle = None
        if bundle is None or any(
            type(item) is not FederatedRosterCosignature for item in bundle
        ):
            self._refuse(FederatedRosterReasonCode.INVALID_REQUEST)
        enrolled = frozenset(commitment.roster.enrolled_keys)
        survivors = frozenset(commitment.survivor_keys)
        signers: set[str] = set()
        for item in bundle:
            if item.round_digest != self._round_digest:
                continue
            if item.signer_key not in enrolled or item.signer_key in self._coordinators:
                continue
            valid = _call_port(
                self._verifier.verify,
                item.signer_key,
                item.signing_message(),
                item.signature,
            )
            if valid is _PORT_FAILED:
                self._abort(FederatedRosterReasonCode.PORT_FAILURE)
            if valid is not True:
                continue
            if item.roster_digest != digest:
                self._abort(FederatedRosterReasonCode.SURVIVOR_SET_EQUIVOCATION)
            if item.signer_key in survivors:
                signers.add(item.signer_key)
        return signers

    def _check_requests(
        self,
        commitment: FederatedRosterCommitment,
        requests: Iterable[FederatedShareRequest],
    ) -> tuple[tuple[str, FederatedMaskShareKind], ...]:
        if isinstance(requests, (str, bytes)):
            self._refuse(FederatedRosterReasonCode.INVALID_REQUEST)
        try:
            batch = tuple(requests)
        except Exception:
            batch = None
        if (
            not batch
            or any(type(item) is not FederatedShareRequest for item in batch)
            or any(
                item.peer_key == self._local_key
                or item.peer_key not in commitment.roster.enrolled_keys
                for item in batch
            )
        ):
            self._refuse(FederatedRosterReasonCode.INVALID_REQUEST)
        kinds: dict[str, FederatedMaskShareKind] = {}
        for item in batch:
            earlier = kinds.get(item.peer_key, self._revealed.get(item.peer_key))
            if earlier is not None and earlier is not item.kind:
                self._abort(FederatedRosterReasonCode.DOUBLE_MASK_REVEAL)
            kinds[item.peer_key] = item.kind
        survivors = frozenset(commitment.survivor_keys)
        for peer, kind in kinds.items():
            expected = (
                FederatedMaskShareKind.SELF_MASK
                if peer in survivors
                else FederatedMaskShareKind.PAIRWISE_MASK
            )
            if kind is not expected:
                self._abort(FederatedRosterReasonCode.MASK_KIND_MISMATCH)
        return tuple(kinds.items())

    def _require_commitment(self, commitment: object) -> FederatedRosterCommitment:
        self._require_live()
        if not isinstance(commitment, FederatedRosterCommitment):
            self._refuse(FederatedRosterReasonCode.INVALID_ROSTER)
        if self._roster is None:
            self._refuse(FederatedRosterReasonCode.OUT_OF_ORDER)
        if commitment.roster.round_digest != self._round_digest:
            self._refuse(FederatedRosterReasonCode.ROUND_MISMATCH)
        if commitment.roster != self._roster:
            self._abort(FederatedRosterReasonCode.ROSTER_EQUIVOCATION)
        return commitment

    def _require_live(self) -> None:
        if self._state is FederatedRosterGuardState.ABORTED:
            raise FederatedRosterError(FederatedRosterReasonCode.ROUND_ABORTED)

    def _refuse(self, reason: FederatedRosterReasonCode) -> NoReturn:
        self._reason = reason
        raise FederatedRosterError(reason)

    def _abort(self, reason: FederatedRosterReasonCode) -> NoReturn:
        self._state = FederatedRosterGuardState.ABORTED
        self._reason = reason
        raise FederatedRosterError(reason)


def _call_port(method: Any, *args: Any) -> Any:
    try:
        return method(*args)
    except Exception:
        return _PORT_FAILED


def _signing_message(round_digest: str, roster_digest: str) -> bytes:
    return (
        _COSIGN_DOMAIN
        + round_digest.encode("ascii")
        + b"\0"
        + roster_digest.encode("ascii")
    )


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(_ROSTER_DOMAIN + canonical).hexdigest()


def _require_digest(value: object) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)


def _require_key(
    value: object,
    reason: FederatedRosterReasonCode = FederatedRosterReasonCode.INVALID_ROSTER,
) -> None:
    if type(value) is not str or _KEY.fullmatch(value) is None:
        raise FederatedRosterError(reason)


def _canonical_keys(value: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
    try:
        keys = tuple(value)
    except Exception:
        keys = None
    if keys is None or (not keys and not allow_empty):
        raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
    for key in keys:
        _require_key(key)
    if len(set(keys)) != len(keys):
        raise FederatedRosterError(FederatedRosterReasonCode.INVALID_ROSTER)
    return tuple(sorted(keys))


__all__ = [
    "FEDERATED_ROSTER_EVIDENCE_SCHEMA_VERSION",
    "FEDERATED_ROSTER_SCHEMA_VERSION",
    "FederatedMaskShareKind",
    "FederatedReleasedShare",
    "FederatedRoster",
    "FederatedRosterCommitment",
    "FederatedRosterCosignature",
    "FederatedRosterError",
    "FederatedRosterEvidence",
    "FederatedRosterGuard",
    "FederatedRosterGuardState",
    "FederatedRosterReasonCode",
    "FederatedShareRelease",
    "FederatedShareRequest",
    "OperatorAttestationRegistry",
    "RecoveryShareSource",
    "RosterSignatureVerifier",
    "RosterSigner",
    "roster_signing_message",
]
