"""Client-side roster consistency checks before secure-aggregation share release."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any
from unittest.mock import patch

import pytest

from openmed.training.federated_roster import (
    FEDERATED_ROSTER_SCHEMA_VERSION,
    FederatedMaskShareKind,
    FederatedRoster,
    FederatedRosterCommitment,
    FederatedRosterCosignature,
    FederatedRosterError,
    FederatedRosterGuard,
    FederatedRosterGuardState,
    FederatedRosterReasonCode,
    FederatedShareRequest,
    roster_signing_message,
)

_ROUND = "sha256:" + "1" * 64
_OTHER_ROUND = "sha256:" + "2" * 64
_SENTINEL = "SENTINELPSEUDONYM"
_SHARE_SENTINEL = b"SENTINELSHARE"

A, B, C, D, E = (f"pk-{_SENTINEL}-{name}" for name in "abcde")
COORDINATOR = f"pk-{_SENTINEL}-coordinator"
PAD1, PAD2 = (f"pk-{_SENTINEL}-pad{index}" for index in (1, 2))
ENROLLED = (A, B, C, D, E)
_ATTESTED = frozenset(ENROLLED)


class FakeKeyring:
    """HMAC-based fake signature port; one secret per pseudonymous key."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self._secrets = {
            key: hashlib.sha256(b"synthetic-secret:" + key.encode()).digest()
            for key in keys
        }
        self.verify_calls = 0

    def signer(self, key: str) -> FakeSigner:
        return FakeSigner(self._secrets[key])

    def verify(self, signer_key: str, message: bytes, signature: bytes) -> bool:
        self.verify_calls += 1
        secret = self._secrets.get(signer_key)
        if secret is None:
            return False
        expected = hmac.new(secret, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)


class FakeSigner:
    """Signs with one key; counts calls."""

    def __init__(self, secret: bytes) -> None:
        self._secret = secret
        self.calls = 0

    def sign(self, message: bytes) -> bytes:
        self.calls += 1
        return hmac.new(self._secret, message, hashlib.sha256).digest()


class FakeAttestations:
    """Operator attestation registry with a fixed attested set."""

    def __init__(self, attested: frozenset[str]) -> None:
        self._attested = attested

    def is_operator_attested(self, participant_key: str) -> bool:
        return participant_key in self._attested


class FakeShareSource:
    """Recovery-share store that records every reveal."""

    def __init__(self) -> None:
        self.revealed: list[tuple[str, FederatedMaskShareKind]] = []

    def share_for(self, peer_key: str, kind: FederatedMaskShareKind) -> bytes:
        self.revealed.append((peer_key, kind))
        return _SHARE_SENTINEL + b":" + kind.value.encode()


KEYRING = FakeKeyring((*ENROLLED, COORDINATOR, PAD1, PAD2))


def _roster(
    enrolled: tuple[str, ...] = ENROLLED,
    *,
    threshold: int = 3,
    round_digest: str = _ROUND,
) -> FederatedRoster:
    return FederatedRoster(
        round_digest=round_digest, enrolled_keys=enrolled, threshold=threshold
    )


def _commitment(
    survivors: tuple[str, ...], roster: FederatedRoster | None = None
) -> FederatedRosterCommitment:
    return FederatedRosterCommitment(
        roster=roster or _roster(), survivor_keys=survivors
    )


def _guard(
    local: str = A,
    *,
    floor: int = 2,
    attested: frozenset[str] = _ATTESTED,
    keyring: Any = KEYRING,
) -> FederatedRosterGuard:
    return FederatedRosterGuard(
        round_digest=_ROUND,
        local_key=local,
        coordinator_keys=(COORDINATOR,),
        minimum_attested_peers=floor,
        verifier=keyring,
        attestations=FakeAttestations(attested),
    )


def _cosign(commitment: FederatedRosterCommitment, *signers: str) -> list[Any]:
    message = roster_signing_message(commitment)
    return [
        FederatedRosterCosignature(
            round_digest=commitment.roster.round_digest,
            roster_digest=commitment.digest(),
            signer_key=key,
            signature=KEYRING.signer(key).sign(message),
        )
        for key in signers
    ]


def _cosigned_guard(
    commitment: FederatedRosterCommitment, local: str = A
) -> FederatedRosterGuard:
    guard = _guard(local)
    guard.authorize_submission(commitment.roster)
    guard.cosign(commitment, KEYRING.signer(local))
    return guard


def _assert_no_sensitive(*values: object) -> None:
    for value in values:
        text = value if isinstance(value, str) else repr(value)
        assert _SENTINEL not in text
        assert "SENTINELSHARE" not in text
        assert "synthetic-secret" not in text


def _assert_safe_error(error: FederatedRosterError) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_no_sensitive(str(error), repr(error), error.args)


# -- roster commitment -------------------------------------------------------


def test_roster_commitment_digest_is_canonical_and_domain_separated() -> None:
    first = _commitment((C, A, B))
    second = FederatedRosterCommitment(
        roster=_roster(tuple(reversed(ENROLLED))), survivor_keys=(B, C, A)
    )

    assert first.digest() == second.digest()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first.digest())
    assert first.roster.enrolled_keys == tuple(sorted(ENROLLED))
    assert first.survivor_keys == tuple(sorted((A, B, C)))

    real_sha256 = hashlib.sha256
    with patch(
        "openmed.training.federated_roster.hashlib.sha256", wraps=real_sha256
    ) as digest:
        first.digest()
    hashed = digest.call_args.args[0]
    assert hashed.startswith(b"openmed.training.federated_roster.v1\0")
    payload = json.loads(hashed.split(b"\0", 1)[1])
    assert payload == {
        "enrolled_keys": sorted(ENROLLED),
        "round_digest": _ROUND,
        "schema_version": FEDERATED_ROSTER_SCHEMA_VERSION,
        "survivor_keys": sorted((A, B, C)),
        "threshold": 3,
    }


@pytest.mark.parametrize(
    "changed",
    [
        lambda: _commitment((A, B, D)),
        lambda: _commitment((A, B, C, D), _roster(threshold=4)),
        lambda: _commitment((A, B, C), _roster(round_digest=_OTHER_ROUND)),
        lambda: _commitment((A, B, C), _roster((A, B, C, D, PAD1))),
    ],
)
def test_every_committed_field_changes_the_digest(changed: Any) -> None:
    baseline = _commitment((A, B, C)).digest()
    assert changed().digest() != baseline
    # Threshold alone changes the digest for an identical survivor set.
    assert (
        _commitment((A, B, C, D), _roster(threshold=4)).digest()
        != _commitment((A, B, C, D)).digest()
    )


def test_signing_message_binds_round_and_commitment() -> None:
    commitment = _commitment((A, B, C))
    message = roster_signing_message(commitment)

    assert message.startswith(b"openmed.training.federated_roster.cosign.v1\0")
    assert _ROUND.encode() in message
    assert commitment.digest().encode() in message
    assert _SENTINEL.encode() not in message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enrolled_keys": (A, A, B, C)},
        {"enrolled_keys": ()},
        {"enrolled_keys": A},
        {"enrolled_keys": (A, "bad key with spaces", C)},
        {"enrolled_keys": (A, B, 7)},
        {"threshold": 1},
        {"threshold": 6},
        {"threshold": True},
        # Two disjoint survivor sets must not both reach the threshold.
        {"threshold": 2},
        {"round_digest": "sha256:short"},
    ],
)
def test_invalid_rosters_are_refused_without_echo(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "round_digest": _ROUND,
        "enrolled_keys": ENROLLED,
        "threshold": 3,
    }
    values.update(kwargs)
    with pytest.raises(FederatedRosterError) as caught:
        FederatedRoster(**values)
    assert caught.value.reason is FederatedRosterReasonCode.INVALID_ROSTER
    _assert_safe_error(caught.value)


@pytest.mark.parametrize("survivors", [(A, B), (A, B, PAD1), (A, A, B, C)])
def test_invalid_survivor_sets_are_refused(survivors: tuple[str, ...]) -> None:
    with pytest.raises(FederatedRosterError) as caught:
        _commitment(survivors)
    assert caught.value.reason is FederatedRosterReasonCode.INVALID_ROSTER
    _assert_safe_error(caught.value)


def test_reprs_hide_pseudonyms_and_signatures() -> None:
    commitment = _commitment((A, B, C))
    cosignature = _cosign(commitment, A)[0]
    request = FederatedShareRequest(D, FederatedMaskShareKind.PAIRWISE_MASK)
    guard = _guard()

    _assert_no_sensitive(commitment, commitment.roster, cosignature, request, guard)
    assert cosignature.signature not in repr(cosignature).encode()


# -- attested identity floor -------------------------------------------------


def test_consistent_roster_admits_submission() -> None:
    guard = _guard()
    guard.authorize_submission(_roster())

    assert guard.state is FederatedRosterGuardState.SUBMITTED
    evidence = guard.evidence()
    assert evidence.reason_code is None
    assert evidence.roster_digest == _roster().digest()


def test_padded_roster_below_attested_floor_is_refused_before_submission() -> None:
    padded = _roster((A, B, COORDINATOR, PAD1, PAD2))
    guard = _guard(floor=2, attested=frozenset({A, B, COORDINATOR}))
    shares = FakeShareSource()

    with pytest.raises(FederatedRosterError) as caught:
        guard.authorize_submission(padded)

    assert caught.value.reason is FederatedRosterReasonCode.ATTESTED_FLOOR_NOT_MET
    assert guard.state is FederatedRosterGuardState.ABORTED
    _assert_safe_error(caught.value)
    commitment = _commitment((A, B, COORDINATOR), padded)
    with pytest.raises(FederatedRosterError) as later:
        guard.cosign(commitment, KEYRING.signer(A))
    assert later.value.reason is FederatedRosterReasonCode.ROUND_ABORTED
    assert shares.revealed == []


def test_coordinator_identity_never_counts_toward_attested_floor() -> None:
    # Coordinator is attested by an operator but is not an independent peer.
    roster = _roster((A, B, COORDINATOR, PAD1, PAD2))
    guard = _guard(floor=2, attested=frozenset({A, B, COORDINATOR}))
    with pytest.raises(FederatedRosterError) as caught:
        guard.authorize_submission(roster)
    assert caught.value.reason is FederatedRosterReasonCode.ATTESTED_FLOOR_NOT_MET

    # The local key also never counts: only attested peers isolate the update.
    guard = _guard(floor=1, attested=frozenset({A}))
    with pytest.raises(FederatedRosterError) as caught:
        guard.authorize_submission(_roster())
    assert caught.value.reason is FederatedRosterReasonCode.ATTESTED_FLOOR_NOT_MET


def test_attested_floor_exactly_met_passes() -> None:
    guard = _guard(floor=2, attested=frozenset({A, B, C}))
    guard.authorize_submission(_roster())
    assert guard.state is FederatedRosterGuardState.SUBMITTED


def test_local_key_must_be_enrolled() -> None:
    guard = _guard(local=PAD1)
    with pytest.raises(FederatedRosterError) as caught:
        guard.authorize_submission(_roster())
    assert caught.value.reason is FederatedRosterReasonCode.LOCAL_NOT_ENROLLED
    _assert_safe_error(caught.value)


def test_roster_for_another_round_is_refused() -> None:
    guard = _guard()
    with pytest.raises(FederatedRosterError) as caught:
        guard.authorize_submission(_roster(round_digest=_OTHER_ROUND))
    assert caught.value.reason is FederatedRosterReasonCode.ROUND_MISMATCH


def test_second_enrollment_roster_for_round_aborts() -> None:
    guard = _guard()
    guard.authorize_submission(_roster())
    guard.authorize_submission(_roster())  # identical roster is idempotent

    with pytest.raises(FederatedRosterError) as caught:
        guard.authorize_submission(_roster((A, B, C, D, PAD1)))
    assert caught.value.reason is FederatedRosterReasonCode.ROSTER_EQUIVOCATION
    assert guard.state is FederatedRosterGuardState.ABORTED


def test_attestation_port_failure_fails_closed_without_context() -> None:
    class BrokenAttestations:
        def is_operator_attested(self, participant_key: str) -> bool:
            raise RuntimeError(f"lookup failed for {participant_key}")

    guard = FederatedRosterGuard(
        round_digest=_ROUND,
        local_key=A,
        coordinator_keys=(COORDINATOR,),
        minimum_attested_peers=2,
        verifier=KEYRING,
        attestations=BrokenAttestations(),
    )
    with pytest.raises(FederatedRosterError) as caught:
        guard.authorize_submission(_roster())
    assert caught.value.reason is FederatedRosterReasonCode.PORT_FAILURE
    assert guard.state is FederatedRosterGuardState.ABORTED
    _assert_safe_error(caught.value)


def test_truthy_non_bool_attestation_is_not_accepted() -> None:
    class TruthyAttestations:
        def is_operator_attested(self, participant_key: str) -> object:
            return "yes"

    guard = FederatedRosterGuard(
        round_digest=_ROUND,
        local_key=A,
        coordinator_keys=(),
        minimum_attested_peers=1,
        verifier=KEYRING,
        attestations=TruthyAttestations(),
    )
    with pytest.raises(FederatedRosterError) as caught:
        guard.authorize_submission(_roster())
    assert caught.value.reason is FederatedRosterReasonCode.ATTESTED_FLOOR_NOT_MET


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_attested_peers": 0},
        {"minimum_attested_peers": True},
        {"round_digest": "nope"},
        {"local_key": ""},
        {"coordinator_keys": A},
        {"coordinator_keys": (A,)},
    ],
)
def test_invalid_guard_configuration_is_refused(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "round_digest": _ROUND,
        "local_key": A,
        "coordinator_keys": (COORDINATOR,),
        "minimum_attested_peers": 2,
        "verifier": KEYRING,
        "attestations": FakeAttestations(_ATTESTED),
    }
    values.update(kwargs)
    with pytest.raises(FederatedRosterError) as caught:
        FederatedRosterGuard(**values)
    _assert_safe_error(caught.value)


# -- co-signing and equivocation ---------------------------------------------


def test_cosign_requires_prior_submission() -> None:
    guard = _guard()
    with pytest.raises(FederatedRosterError) as caught:
        guard.cosign(_commitment((A, B, C)), KEYRING.signer(A))
    assert caught.value.reason is FederatedRosterReasonCode.OUT_OF_ORDER


def test_cosign_signs_the_canonical_message_once() -> None:
    commitment = _commitment((A, B, C))
    guard = _guard()
    guard.authorize_submission(commitment.roster)
    signer = KEYRING.signer(A)

    first = guard.cosign(commitment, signer)
    again = guard.cosign(_commitment((C, B, A)), signer)

    assert first == again
    assert signer.calls == 1
    assert KEYRING.verify(A, roster_signing_message(commitment), first.signature)
    assert guard.state is FederatedRosterGuardState.COSIGNED
    assert guard.evidence().commitment_digest == commitment.digest()


def test_cosigning_a_second_survivor_set_aborts() -> None:
    guard = _cosigned_guard(_commitment((A, B, C)))
    signer = KEYRING.signer(A)

    with pytest.raises(FederatedRosterError) as caught:
        guard.cosign(_commitment((A, B, C, D)), signer)

    assert caught.value.reason is FederatedRosterReasonCode.SURVIVOR_SET_EQUIVOCATION
    assert signer.calls == 0
    assert guard.state is FederatedRosterGuardState.ABORTED
    _assert_safe_error(caught.value)


def test_cosign_with_changed_enrollment_or_threshold_aborts() -> None:
    guard = _guard()
    guard.authorize_submission(_roster())
    with pytest.raises(FederatedRosterError) as caught:
        guard.cosign(_commitment((A, B, C, D), _roster(threshold=4)), KEYRING.signer(A))
    assert caught.value.reason is FederatedRosterReasonCode.ROSTER_EQUIVOCATION
    assert guard.state is FederatedRosterGuardState.ABORTED


def test_local_client_reported_as_dropped_aborts_without_signing() -> None:
    guard = _guard(local=D)
    guard.authorize_submission(_roster())
    signer = KEYRING.signer(D)
    with pytest.raises(FederatedRosterError) as caught:
        guard.cosign(_commitment((A, B, C)), signer)
    assert caught.value.reason is FederatedRosterReasonCode.LOCAL_NOT_SURVIVOR
    assert signer.calls == 0
    assert guard.state is FederatedRosterGuardState.ABORTED


def test_signer_failure_fails_closed() -> None:
    class BrokenSigner:
        def sign(self, message: bytes) -> bytes:
            raise RuntimeError("hsm offline " + A)

    guard = _guard()
    guard.authorize_submission(_roster())
    with pytest.raises(FederatedRosterError) as caught:
        guard.cosign(_commitment((A, B, C)), BrokenSigner())
    assert caught.value.reason is FederatedRosterReasonCode.PORT_FAILURE
    _assert_safe_error(caught.value)


# -- share release -----------------------------------------------------------


def _requests_for(commitment: FederatedRosterCommitment, local: str) -> list[Any]:
    requests = []
    for peer in commitment.roster.enrolled_keys:
        if peer == local:
            continue
        kind = (
            FederatedMaskShareKind.SELF_MASK
            if peer in commitment.survivor_keys
            else FederatedMaskShareKind.PAIRWISE_MASK
        )
        requests.append(FederatedShareRequest(peer, kind))
    return requests


def test_consistent_roster_with_threshold_cosignatures_releases_shares() -> None:
    commitment = _commitment((A, B, C))
    guard = _cosigned_guard(commitment)
    shares = FakeShareSource()

    release = guard.release_shares(
        commitment,
        _cosign(commitment, A, B, C),
        _requests_for(commitment, A),
        shares,
    )

    assert sorted(shares.revealed) == sorted(
        [
            (B, FederatedMaskShareKind.SELF_MASK),
            (C, FederatedMaskShareKind.SELF_MASK),
            (D, FederatedMaskShareKind.PAIRWISE_MASK),
            (E, FederatedMaskShareKind.PAIRWISE_MASK),
        ]
    )
    assert len(release.shares) == 4
    assert guard.state is FederatedRosterGuardState.RELEASED
    evidence = guard.evidence()
    assert evidence.released_share_count == 4
    assert evidence.reason_code is None
    _assert_no_sensitive(release, evidence, evidence.to_json(), guard)


def test_release_below_threshold_cosignatures_is_refused() -> None:
    commitment = _commitment((A, B, C))
    guard = _cosigned_guard(commitment)
    shares = FakeShareSource()

    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            _cosign(commitment, A, B),
            _requests_for(commitment, A),
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.INSUFFICIENT_COSIGNATURES
    assert shares.revealed == []
    # Not proof of malice: a later, complete bundle may still be accepted.
    assert guard.state is FederatedRosterGuardState.COSIGNED


@pytest.mark.parametrize(
    "bundle",
    [
        # Duplicate signatures from one survivor count once.
        lambda c: _cosign(c, A, B) + _cosign(c, B),
        # Signatures from dropped peers do not count.
        lambda c: _cosign(c, A, B, D),
        # Coordinator signatures never count.
        lambda c: _cosign(c, A, B, COORDINATOR),
        # Forged signatures do not count.
        lambda c: (
            _cosign(c, A, B)
            + [
                FederatedRosterCosignature(
                    round_digest=_ROUND,
                    roster_digest=c.digest(),
                    signer_key=C,
                    signature=b"\x00" * 32,
                )
            ]
        ),
        # Signatures over another round's roster do not count.
        lambda c: (
            _cosign(c, A, B)
            + _cosign(_commitment((A, B, C), _roster(round_digest=_OTHER_ROUND)), C)
        ),
    ],
)
def test_only_valid_distinct_survivor_cosignatures_count(bundle: Any) -> None:
    commitment = _commitment((A, B, C))
    guard = _cosigned_guard(commitment)
    shares = FakeShareSource()
    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment, bundle(commitment), _requests_for(commitment, A), shares
        )
    assert caught.value.reason is FederatedRosterReasonCode.INSUFFICIENT_COSIGNATURES
    assert shares.revealed == []


def test_release_requires_local_cosignature_first() -> None:
    commitment = _commitment((A, B, C))
    guard = _guard()
    guard.authorize_submission(commitment.roster)
    shares = FakeShareSource()
    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            _cosign(commitment, A, B, C),
            _requests_for(commitment, A),
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.OUT_OF_ORDER
    assert shares.revealed == []


def test_release_for_a_different_survivor_set_than_cosigned_aborts() -> None:
    cosigned = _commitment((A, B, C))
    presented = _commitment((A, B, C, D))
    guard = _cosigned_guard(cosigned)
    shares = FakeShareSource()

    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            presented,
            _cosign(presented, A, B, C, D),
            _requests_for(presented, A),
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.SURVIVOR_SET_EQUIVOCATION
    assert guard.state is FederatedRosterGuardState.ABORTED
    assert shares.revealed == []
    _assert_safe_error(caught.value)


def test_conflicting_signed_survivor_set_in_bundle_aborts() -> None:
    commitment = _commitment((A, B, C))
    conflicting = _commitment((A, B, C, D))
    guard = _cosigned_guard(commitment)
    shares = FakeShareSource()

    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            _cosign(commitment, A, B, C) + _cosign(conflicting, D),
            _requests_for(commitment, A),
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.SURVIVOR_SET_EQUIVOCATION
    assert guard.state is FederatedRosterGuardState.ABORTED
    assert shares.revealed == []


def test_unverifiable_conflicting_cosignature_does_not_abort() -> None:
    commitment = _commitment((A, B, C))
    conflicting = _commitment((A, B, C, D))
    guard = _cosigned_guard(commitment)
    forged = FederatedRosterCosignature(
        round_digest=_ROUND,
        roster_digest=conflicting.digest(),
        signer_key=D,
        signature=b"\x01" * 32,
    )
    shares = FakeShareSource()
    guard.release_shares(
        commitment,
        [*_cosign(commitment, A, B, C), forged],
        _requests_for(commitment, A),
        shares,
    )
    assert guard.state is FederatedRosterGuardState.RELEASED


def test_request_revealing_both_mask_types_for_one_peer_is_refused() -> None:
    commitment = _commitment((A, B, C))
    guard = _cosigned_guard(commitment)
    shares = FakeShareSource()

    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            _cosign(commitment, A, B, C),
            [
                FederatedShareRequest(D, FederatedMaskShareKind.PAIRWISE_MASK),
                FederatedShareRequest(D, FederatedMaskShareKind.SELF_MASK),
            ],
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.DOUBLE_MASK_REVEAL
    assert shares.revealed == []
    assert guard.state is FederatedRosterGuardState.ABORTED
    _assert_safe_error(caught.value)


def test_double_reveal_across_batches_is_refused() -> None:
    commitment = _commitment((A, B, C))
    guard = _cosigned_guard(commitment)
    shares = FakeShareSource()
    cosignatures = _cosign(commitment, A, B, C)
    guard.release_shares(
        commitment,
        cosignatures,
        [FederatedShareRequest(D, FederatedMaskShareKind.PAIRWISE_MASK)],
        shares,
    )
    # Repeating an approved reveal is idempotent.
    guard.release_shares(
        commitment,
        cosignatures,
        [FederatedShareRequest(D, FederatedMaskShareKind.PAIRWISE_MASK)],
        shares,
    )
    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            cosignatures,
            [FederatedShareRequest(D, FederatedMaskShareKind.SELF_MASK)],
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.DOUBLE_MASK_REVEAL
    assert shares.revealed == [(D, FederatedMaskShareKind.PAIRWISE_MASK)] * 2


@pytest.mark.parametrize(
    ("peer", "kind"),
    [
        (B, FederatedMaskShareKind.PAIRWISE_MASK),  # survivor pairwise secret
        (D, FederatedMaskShareKind.SELF_MASK),  # dropped peer self mask
    ],
)
def test_mask_kind_must_match_committed_survivor_status(
    peer: str, kind: FederatedMaskShareKind
) -> None:
    commitment = _commitment((A, B, C))
    guard = _cosigned_guard(commitment)
    shares = FakeShareSource()
    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            _cosign(commitment, A, B, C),
            [FederatedShareRequest(peer, kind)],
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.MASK_KIND_MISMATCH
    assert shares.revealed == []
    assert guard.state is FederatedRosterGuardState.ABORTED


@pytest.mark.parametrize("peer", [A, PAD1])
def test_requests_for_self_or_unenrolled_peers_are_refused(peer: str) -> None:
    commitment = _commitment((A, B, C))
    guard = _cosigned_guard(commitment)
    shares = FakeShareSource()
    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            _cosign(commitment, A, B, C),
            [FederatedShareRequest(peer, FederatedMaskShareKind.SELF_MASK)],
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.INVALID_REQUEST
    assert shares.revealed == []


def test_share_port_failure_returns_nothing_and_aborts() -> None:
    class FlakyShares(FakeShareSource):
        def share_for(self, peer_key: str, kind: FederatedMaskShareKind) -> bytes:
            if self.revealed:
                raise RuntimeError("store failed: " + peer_key)
            return super().share_for(peer_key, kind)

    commitment = _commitment((A, B, C))
    guard = _cosigned_guard(commitment)
    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            _cosign(commitment, A, B, C),
            _requests_for(commitment, A),
            FlakyShares(),
        )
    assert caught.value.reason is FederatedRosterReasonCode.PORT_FAILURE
    assert guard.state is FederatedRosterGuardState.ABORTED
    assert guard.evidence().released_share_count == 0
    _assert_safe_error(caught.value)


def test_verifier_failure_fails_closed() -> None:
    class BrokenVerifier:
        def verify(self, signer_key: str, message: bytes, signature: bytes) -> bool:
            raise RuntimeError("verifier down " + signer_key)

    commitment = _commitment((A, B, C))
    guard = FederatedRosterGuard(
        round_digest=_ROUND,
        local_key=A,
        coordinator_keys=(COORDINATOR,),
        minimum_attested_peers=2,
        verifier=BrokenVerifier(),
        attestations=FakeAttestations(_ATTESTED),
    )
    guard.authorize_submission(commitment.roster)
    guard.cosign(commitment, KEYRING.signer(A))
    shares = FakeShareSource()
    with pytest.raises(FederatedRosterError) as caught:
        guard.release_shares(
            commitment,
            _cosign(commitment, A, B, C),
            _requests_for(commitment, A),
            shares,
        )
    assert caught.value.reason is FederatedRosterReasonCode.PORT_FAILURE
    assert shares.revealed == []
    _assert_safe_error(caught.value)


# -- evidence ----------------------------------------------------------------


def test_evidence_is_metadata_only_and_stable() -> None:
    guard = _guard()
    guard.authorize_submission(_roster())
    with pytest.raises(FederatedRosterError):
        guard.authorize_submission(_roster((A, B, C, D, PAD1)))

    evidence = guard.evidence()
    payload = evidence.to_dict()
    assert set(payload) == {
        "commitment_digest",
        "reason_code",
        "released_share_count",
        "roster_digest",
        "round_digest",
        "schema_version",
        "state",
    }
    assert payload["state"] == "aborted"
    assert payload["reason_code"] == "roster_equivocation"
    assert evidence.to_json() == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _assert_no_sensitive(evidence, evidence.to_json())


def test_reason_codes_are_controlled_tokens() -> None:
    for code in FederatedRosterReasonCode:
        assert re.fullmatch(r"[a-z_]+", code.value)
        assert str(code) == code.value
