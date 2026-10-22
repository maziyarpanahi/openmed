"""Multi-client roster equivocation scenarios with synthetic signers and shares."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from openmed.training.federated_roster import (
    FederatedMaskShareKind,
    FederatedRoster,
    FederatedRosterCommitment,
    FederatedRosterError,
    FederatedRosterGuard,
    FederatedRosterGuardState,
    FederatedRosterReasonCode,
    FederatedShareRequest,
)

pytestmark = pytest.mark.integration

_ROUND = "sha256:" + "3" * 64
CLIENTS = tuple(f"pk-synthetic-client-{index}" for index in range(5))
A, B, C, D, E = CLIENTS
COORDINATOR = "pk-synthetic-coordinator"


class Keyring:
    """HMAC fake for per-client signing keys."""

    def __init__(self) -> None:
        self._secrets = {
            key: hashlib.sha256(b"synthetic:" + key.encode()).digest()
            for key in (*CLIENTS, COORDINATOR)
        }

    def sign_as(self, key: str):
        secret = self._secrets[key]

        class _Signer:
            def sign(self, message: bytes) -> bytes:
                return hmac.new(secret, message, hashlib.sha256).digest()

        return _Signer()

    def verify(self, signer_key: str, message: bytes, signature: bytes) -> bool:
        secret = self._secrets.get(signer_key)
        if secret is None:
            return False
        expected = hmac.new(secret, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)


class Attestations:
    def is_operator_attested(self, participant_key: str) -> bool:
        return participant_key in CLIENTS


class ShareLedger:
    """Collects every share any honest client releases."""

    def __init__(self) -> None:
        self.released: list[tuple[str, str, FederatedMaskShareKind]] = []

    def for_client(self, owner: str):
        ledger = self

        class _Source:
            def share_for(self, peer_key: str, kind: FederatedMaskShareKind) -> bytes:
                ledger.released.append((owner, peer_key, kind))
                return b"synthetic-share"

        return _Source()


def _requests(commitment: FederatedRosterCommitment, local: str) -> list:
    return [
        FederatedShareRequest(
            peer,
            FederatedMaskShareKind.SELF_MASK
            if peer in commitment.survivor_keys
            else FederatedMaskShareKind.PAIRWISE_MASK,
        )
        for peer in commitment.roster.enrolled_keys
        if peer != local
    ]


def _run_equivocating_round(*, broadcast_all: bool):
    keyring = Keyring()
    roster = FederatedRoster(round_digest=_ROUND, enrolled_keys=CLIENTS, threshold=3)
    # The coordinator tells A, B and C that D and E dropped, while telling D and
    # E that only C dropped.
    views = {
        A: FederatedRosterCommitment(roster=roster, survivor_keys=(A, B, C)),
        B: FederatedRosterCommitment(roster=roster, survivor_keys=(A, B, C)),
        C: FederatedRosterCommitment(roster=roster, survivor_keys=(A, B, C)),
        D: FederatedRosterCommitment(roster=roster, survivor_keys=(A, B, D, E)),
        E: FederatedRosterCommitment(roster=roster, survivor_keys=(A, B, D, E)),
    }
    guards = {
        client: FederatedRosterGuard(
            round_digest=_ROUND,
            local_key=client,
            coordinator_keys=(COORDINATOR,),
            minimum_attested_peers=3,
            verifier=keyring,
            attestations=Attestations(),
        )
        for client in CLIENTS
    }
    cosignatures = {}
    for client, guard in guards.items():
        guard.authorize_submission(roster)
        cosignatures[client] = guard.cosign(views[client], keyring.sign_as(client))

    ledger = ShareLedger()
    reasons = {}
    for client, guard in guards.items():
        if broadcast_all:
            bundle = list(cosignatures.values())
        else:
            # A filtering coordinator only forwards matching signatures.
            bundle = [
                cosignatures[peer]
                for peer in CLIENTS
                if views[peer].digest() == views[client].digest()
            ]
        try:
            guard.release_shares(
                views[client],
                bundle,
                _requests(views[client], client),
                ledger.for_client(client),
            )
            reasons[client] = None
        except FederatedRosterError as error:
            reasons[client] = error.reason
    return guards, ledger, reasons


def test_two_survivor_sets_make_every_honest_client_abort() -> None:
    guards, ledger, reasons = _run_equivocating_round(broadcast_all=True)

    assert ledger.released == []
    assert set(reasons.values()) == {
        FederatedRosterReasonCode.SURVIVOR_SET_EQUIVOCATION
    }
    for guard in guards.values():
        assert guard.state is FederatedRosterGuardState.ABORTED
        assert guard.evidence().released_share_count == 0


def test_filtering_coordinator_cannot_unmask_any_single_update() -> None:
    _, ledger, reasons = _run_equivocating_round(broadcast_all=False)

    # The minority view lacks threshold co-signatures and releases nothing.
    assert reasons[D] is FederatedRosterReasonCode.INSUFFICIENT_COSIGNATURES
    assert reasons[E] is FederatedRosterReasonCode.INSUFFICIENT_COSIGNATURES
    assert not [entry for entry in ledger.released if entry[0] in (D, E)]

    # No peer ever has both its self mask and pairwise secret released.
    kinds_by_peer: dict[str, set[FederatedMaskShareKind]] = {}
    for _owner, peer, kind in ledger.released:
        kinds_by_peer.setdefault(peer, set()).add(kind)
    assert kinds_by_peer
    assert all(len(kinds) == 1 for kinds in kinds_by_peer.values())
