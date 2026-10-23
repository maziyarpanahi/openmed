"""Synthetic, offline contracts for compromised site key holds."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.compliance.audit_chain import HashChainAuditLog
from openmed.training.federated_key_compromise import (
    FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION,
    AcceptedUpdateCommitment,
    HoldLiftGrant,
    IncidentStatus,
    KeyCompromiseErrorCode,
    KeyCompromiseHoldError,
    KeyCompromiseHoldRegistry,
    KeyCompromiseIncident,
    LineageNode,
    RemediationClass,
    RoundCommitmentSet,
    SignedKeyCompromiseIncident,
    promote_round,
)
from openmed.training.federated_round import (
    FederatedRoundLifecycle,
    FederatedRoundState,
)

INCIDENT_KEY = b"synthetic-incident-signing-key"
OPERATOR_KEY = b"synthetic-operator-signing-key"
SITE_SENTINEL = "SITE-SENTINEL-ST-MARY"
COMPROMISED_KEY = "key:" + SITE_SENTINEL + ".signing.v1"
OTHER_KEY = "key:other-site.signing.v1"
OPERATOR_SENTINEL = "operator:OPERATOR-SENTINEL-77"


def _digest(char: str) -> str:
    return "sha256:" + char * 64


U_EARLY = _digest("1")  # accepted before the exposure window
U_IN = _digest("2")  # accepted inside the exposure window into round-3
U_LATE = _digest("3")  # accepted after revocation took effect
U_IN_R5 = _digest("4")  # accepted inside a second incident window
CLEAN_A = _digest("a")
CLEAN_B = _digest("b")
CLEAN_C = _digest("c")
UPDATE_SENTINELS = (U_EARLY, U_IN, U_LATE, U_IN_R5, CLEAN_A, CLEAN_B, CLEAN_C)


def _sign(key: bytes, message: bytes) -> str:
    return hmac.new(key, message, hashlib.sha256).hexdigest()


class HmacVerifier:
    """Synthetic signature verifier with a constant-time comparison."""

    def __init__(self, key: bytes) -> None:
        self._key = key
        self.calls = 0

    def verify(self, message: bytes, signature: str) -> bool:
        self.calls += 1
        return hmac.compare_digest(_sign(self._key, message), signature)


class FakeRevocation:
    def __init__(self, revoked: dict[str, int | None]) -> None:
        self.revoked = revoked

    def revocation_time(self, key_reference: str) -> int | None:
        return self.revoked[key_reference]


class FakeCommitments:
    def __init__(self) -> None:
        self.by_key = {
            COMPROMISED_KEY: [
                AcceptedUpdateCommitment(U_EARLY, "round-1", 50),
                AcceptedUpdateCommitment(U_IN, "round-3", 150),
                AcceptedUpdateCommitment(U_LATE, "round-5", 300),
            ],
            OTHER_KEY: [AcceptedUpdateCommitment(CLEAN_A, "round-2", 120)],
        }
        self.rounds = {
            "round-3": RoundCommitmentSet(
                retained=frozenset({U_IN, CLEAN_A, CLEAN_B}), quorum=2
            ),
            "round-5": RoundCommitmentSet(
                retained=frozenset({U_LATE, CLEAN_C}), quorum=1
            ),
        }
        self.fail = False

    def accepted_updates(self, key_reference: str):
        if self.fail:
            raise RuntimeError(f"backend down for {key_reference}")
        return list(self.by_key.get(key_reference, ()))

    def round_commitments(self, round_ref: str):
        return self.rounds.get(round_ref)


class FakeLineage:
    """c0 -> c1 -> c2 -> c3 -> c4 -> c6, c2 -> c3b -> c5, and c6 merges c5."""

    def __init__(self) -> None:
        self.nodes = [
            LineageNode("ckpt-0", "round-0", ()),
            LineageNode("ckpt-1", "round-1", ("ckpt-0",)),
            LineageNode("ckpt-2", "round-2", ("ckpt-1",)),
            LineageNode("ckpt-3", "round-3", ("ckpt-2",)),
            LineageNode("ckpt-4", "round-4", ("ckpt-3",)),
            LineageNode("ckpt-3b", "round-3b", ("ckpt-2",)),
            LineageNode("ckpt-5", "round-5", ("ckpt-3b",)),
            LineageNode("ckpt-6", "round-6", ("ckpt-4", "ckpt-5")),
        ]
        self.fail = False
        self.calls = 0

    def nodes_snapshot(self):
        self.calls += 1
        if self.fail:
            raise OSError("/private/lineage/store unavailable")
        return list(self.nodes)


def _incident(
    *,
    incident_id: str = "incident:" + "d" * 32,
    key_reference: str = COMPROMISED_KEY,
    exposure_start: int = 100,
    discovered_at: int = 200,
    key: bytes = INCIDENT_KEY,
) -> SignedKeyCompromiseIncident:
    incident = KeyCompromiseIncident(
        incident_id=incident_id,
        key_reference=key_reference,
        exposure_start=exposure_start,
        discovered_at=discovered_at,
    )
    return SignedKeyCompromiseIncident(
        incident=incident, signature=_sign(key, incident.signing_bytes())
    )


def _grant(
    incident_digest: str,
    *,
    key: bytes = OPERATOR_KEY,
    operator_reference: str = OPERATOR_SENTINEL,
) -> HoldLiftGrant:
    unsigned = HoldLiftGrant(
        incident_digest=incident_digest,
        operator_reference=operator_reference,
        issued_at=400,
        signature="0" * 64,
    )
    return replace(unsigned, signature=_sign(key, unsigned.signing_bytes()))


@pytest.fixture
def ports():
    return {
        "revocation": FakeRevocation({COMPROMISED_KEY: 250, OTHER_KEY: None}),
        "commitments": FakeCommitments(),
        "lineage": FakeLineage(),
    }


@pytest.fixture
def audit():
    return HashChainAuditLog()


def _registry(ports, audit=None, state=None, incident_verifier=None):
    return KeyCompromiseHoldRegistry(
        incident_verifier=incident_verifier or HmacVerifier(INCIDENT_KEY),
        authority_verifier=HmacVerifier(OPERATOR_KEY),
        revocation=ports["revocation"],
        commitments=ports["commitments"],
        lineage=ports["lineage"],
        audit_sink=audit,
        state=state,
    )


ALL_CHECKPOINTS = {f"ckpt-{n}" for n in ("0", "1", "2", "3", "4", "3b", "5", "6")}


# Acceptance: holds exactly the affected rounds and descendants -------------


def test_one_compromised_key_holds_exactly_window_rounds_and_descendants(ports, audit):
    registry = _registry(ports, audit)
    report = registry.apply_incident(_incident())

    assert report.status is IncidentStatus.HELD
    assert set(registry.held_checkpoints()) == {"ckpt-3", "ckpt-4", "ckpt-6"}
    assert set(registry.held_rounds()) == {"round-3", "round-4", "round-6"}
    assert report.held_checkpoint_count == 3
    assert report.held_round_count == 3
    assert report.window_closed is True
    for clean in ALL_CHECKPOINTS - {"ckpt-3", "ckpt-4", "ckpt-6"}:
        assert registry.is_checkpoint_held(clean) is False
    # Negative controls: round-1 (before the window) and round-5 (after the
    # revocation) accepted the same key but stay promotable.
    assert registry.is_round_held("round-1") is False
    assert registry.is_round_held("round-5") is False
    assert registry.is_round_held("round-2") is False


def test_window_without_revocation_stays_open_and_fails_closed(ports):
    ports["revocation"].revoked[COMPROMISED_KEY] = None
    registry = _registry(ports)
    report = registry.apply_incident(_incident())

    assert report.window_closed is False
    assert set(registry.held_rounds()) == {
        "round-3",
        "round-4",
        "round-5",
        "round-6",
    }
    assert registry.is_round_held("round-1") is False


def test_revocation_before_discovery_keeps_window_until_discovery(ports):
    ports["revocation"].revoked[COMPROMISED_KEY] = 120
    report = _registry(ports).apply_incident(_incident())
    assert report.window_closed is True
    assert report.held_round_count == 3


def test_window_boundaries_are_inclusive(ports):
    ports["commitments"].by_key[COMPROMISED_KEY] = [
        AcceptedUpdateCommitment(U_EARLY, "round-1", 100),
        AcceptedUpdateCommitment(U_LATE, "round-5", 250),
    ]
    registry = _registry(ports)
    registry.apply_incident(_incident())
    assert registry.is_round_held("round-1")
    assert registry.is_round_held("round-5")


def test_incident_without_affected_updates_holds_nothing(ports, audit):
    registry = _registry(ports, audit)
    report = registry.apply_incident(_incident(exposure_start=310, discovered_at=320))
    assert report.status is IncidentStatus.NO_AFFECTED_UPDATES
    assert report.remediation is RemediationClass.NONE_REQUIRED
    assert registry.held_checkpoints() == ()
    assert len(audit.records) == 1


# Acceptance: promotion and lifting --------------------------------------------


def test_promotion_of_held_descendants_fails(ports):
    registry = _registry(ports)
    registry.apply_incident(_incident())
    evaluating = FederatedRoundLifecycle(state=FederatedRoundState.EVALUATING)

    for checkpoint, round_ref in (
        ("ckpt-3", "round-3"),
        ("ckpt-4", "round-4"),
        ("ckpt-6", "round-6"),
    ):
        with pytest.raises(KeyCompromiseHoldError) as caught:
            promote_round(
                evaluating,
                checkpoint_ref=checkpoint,
                round_ref=round_ref,
                registry=registry,
            )
        assert caught.value.code is KeyCompromiseErrorCode.CHECKPOINT_HELD
        assert checkpoint not in str(caught.value)

    promoted = promote_round(
        evaluating, checkpoint_ref="ckpt-5", round_ref="round-5", registry=registry
    )
    assert promoted.state is FederatedRoundState.PROMOTED


def test_new_descendants_created_after_the_hold_are_not_promotable(ports):
    registry = _registry(ports)
    registry.apply_incident(_incident())
    ports["lineage"].nodes.append(LineageNode("ckpt-7", "round-7", ("ckpt-6",)))
    ports["lineage"].nodes.append(LineageNode("ckpt-8", "round-8", ("ckpt-5",)))

    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.require_promotable(checkpoint_ref="ckpt-7", round_ref="round-7")
    assert caught.value.code is KeyCompromiseErrorCode.CHECKPOINT_HELD
    registry.require_promotable(checkpoint_ref="ckpt-8", round_ref="round-8")


def test_held_round_refuses_promotion_even_with_unknown_checkpoint(ports):
    ports["commitments"].by_key[COMPROMISED_KEY].append(
        AcceptedUpdateCommitment(CLEAN_C, "round-orphan", 160)
    )
    registry = _registry(ports)
    report = registry.apply_incident(_incident())
    assert registry.is_round_held("round-orphan")
    assert report.remediation is RemediationClass.UNSUPPORTED
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.require_promotable(checkpoint_ref="ckpt-x", round_ref="round-orphan")
    assert caught.value.code is KeyCompromiseErrorCode.ROUND_HELD


def test_unknown_checkpoint_fails_closed_while_holds_are_active(ports):
    registry = _registry(ports)
    registry.require_promotable(checkpoint_ref="ckpt-unknown")
    registry.apply_incident(_incident())
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.require_promotable(checkpoint_ref="ckpt-unknown")
    assert caught.value.code is KeyCompromiseErrorCode.LINEAGE_UNRESOLVED


def test_promotion_check_fails_closed_when_lineage_is_unavailable(ports):
    registry = _registry(ports)
    registry.apply_incident(_incident())
    ports["lineage"].fail = True
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.require_promotable(checkpoint_ref="ckpt-5")
    assert caught.value.code is KeyCompromiseErrorCode.LINEAGE_UNAVAILABLE
    assert caught.value.__cause__ is None
    assert "private" not in str(caught.value)


def test_lifting_a_hold_without_operator_authority_fails(ports, audit):
    registry = _registry(ports, audit)
    report = registry.apply_incident(_incident())

    with pytest.raises(KeyCompromiseHoldError) as missing:
        registry.lift_hold(report.incident_digest, None)
    assert missing.value.code is KeyCompromiseErrorCode.AUTHORITY_REQUIRED

    forged = _grant(report.incident_digest, key=b"attacker-key")
    with pytest.raises(KeyCompromiseHoldError) as invalid:
        registry.lift_hold(report.incident_digest, forged)
    assert invalid.value.code is KeyCompromiseErrorCode.AUTHORITY_INVALID

    other = _incident(incident_id="incident:" + "e" * 32).incident.digest
    replayed = _grant(other)
    with pytest.raises(KeyCompromiseHoldError) as mismatch:
        registry.lift_hold(report.incident_digest, replayed)
    assert mismatch.value.code is KeyCompromiseErrorCode.AUTHORITY_INVALID

    assert registry.is_checkpoint_held("ckpt-3")
    assert len(audit.records) == 1


def test_lifting_with_operator_authority_releases_the_hold(ports, audit):
    registry = _registry(ports, audit)
    report = registry.apply_incident(_incident())
    lifted = registry.lift_hold(report.incident_digest, _grant(report.incident_digest))

    assert lifted.status is IncidentStatus.LIFTED
    assert registry.held_checkpoints() == ()
    registry.require_promotable(checkpoint_ref="ckpt-6", round_ref="round-6")
    assert [record.event_type for record in audit.records] == [
        "training.federated.key_compromise.recorded",
        "training.federated.key_compromise.lifted",
    ]
    assert audit.verify()


def test_lifting_an_unknown_incident_fails(ports):
    registry = _registry(ports)
    digest = _digest("f")
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.lift_hold(digest, _grant(digest))
    assert caught.value.code is KeyCompromiseErrorCode.UNKNOWN_INCIDENT


def test_lift_is_not_applied_when_audit_sink_fails(ports):
    class BrokenSink:
        def __init__(self) -> None:
            self.fail = False
            self.log = HashChainAuditLog()

        def append(self, event_type, payload):
            if self.fail:
                raise OSError("disk full at /private/audit")
            return self.log.append(event_type, payload)

    sink = BrokenSink()
    registry = _registry(ports, sink)
    report = registry.apply_incident(_incident())
    sink.fail = True
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.lift_hold(report.incident_digest, _grant(report.incident_digest))
    assert caught.value.code is KeyCompromiseErrorCode.AUDIT_UNAVAILABLE
    assert registry.is_checkpoint_held("ckpt-3")


def test_hold_is_kept_and_audit_retried_when_sink_fails_on_record(ports):
    class FlakySink:
        def __init__(self) -> None:
            self.fail = True
            self.log = HashChainAuditLog()

        def append(self, event_type, payload):
            if self.fail:
                raise OSError("disk full")
            return self.log.append(event_type, payload)

    sink = FlakySink()
    registry = _registry(ports, sink)
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.apply_incident(_incident())
    assert caught.value.code is KeyCompromiseErrorCode.AUDIT_UNAVAILABLE
    assert registry.is_checkpoint_held("ckpt-3")

    sink.fail = False
    registry.apply_incident(_incident())
    registry.apply_incident(_incident())
    assert len(sink.log.records) == 1


# Acceptance: overlapping incidents and restarts are idempotent ----------------


def test_reapplying_the_same_incident_is_idempotent(ports, audit):
    registry = _registry(ports, audit)
    first = registry.apply_incident(_incident())
    second = registry.apply_incident(_incident())
    assert first == second
    assert len(audit.records) == 1


def test_resigned_incident_with_same_payload_is_the_same_incident(ports, audit):
    registry = _registry(ports, audit)
    signed = _incident()
    registry.apply_incident(signed)
    # A verifier that accepts the original signature encoding in upper case.
    upper = replace(signed, signature=signed.signature.upper())

    class CaseInsensitiveVerifier(HmacVerifier):
        def verify(self, message, signature):
            return super().verify(message, signature.lower())

    restarted = _registry(
        ports,
        audit,
        state=registry.to_dict(),
        incident_verifier=CaseInsensitiveVerifier(INCIDENT_KEY),
    )
    assert restarted.apply_incident(upper).incident_digest == signed.incident.digest
    assert len(audit.records) == 1


def test_conflicting_payload_for_known_incident_id_is_refused(ports):
    registry = _registry(ports)
    registry.apply_incident(_incident())
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.apply_incident(_incident(exposure_start=90))
    assert caught.value.code is KeyCompromiseErrorCode.INCIDENT_CONFLICT


def test_restart_from_persisted_state_is_idempotent(ports, audit):
    registry = _registry(ports, audit)
    report = registry.apply_incident(_incident())
    state = json.loads(registry.to_json())

    restarted = _registry(ports, audit, state=state)
    assert restarted.held_checkpoints() == registry.held_checkpoints()
    assert restarted.apply_incident(_incident()) == report
    assert len(audit.records) == 1
    assert restarted.to_json() == registry.to_json()
    with pytest.raises(KeyCompromiseHoldError):
        restarted.require_promotable(checkpoint_ref="ckpt-4")


def test_restart_after_lift_does_not_silently_reapply_the_hold(ports, audit):
    registry = _registry(ports, audit)
    report = registry.apply_incident(_incident())
    registry.lift_hold(report.incident_digest, _grant(report.incident_digest))
    restarted = _registry(ports, audit, state=registry.to_dict())

    replayed = restarted.apply_incident(_incident())
    assert replayed.status is IncidentStatus.LIFTED
    assert restarted.held_checkpoints() == ()
    again = restarted.lift_hold(report.incident_digest, _grant(report.incident_digest))
    assert again.status is IncidentStatus.LIFTED
    assert len(audit.records) == 2


def test_overlapping_incidents_hold_until_every_incident_is_lifted(ports, audit):
    registry = _registry(ports, audit)
    first = registry.apply_incident(_incident())
    second = registry.apply_incident(
        _incident(
            incident_id="incident:" + "e" * 32, exposure_start=140, discovered_at=160
        )
    )
    assert first.incident_digest != second.incident_digest
    assert set(registry.held_checkpoints()) == {"ckpt-3", "ckpt-4", "ckpt-6"}

    registry.lift_hold(first.incident_digest, _grant(first.incident_digest))
    assert set(registry.held_checkpoints()) == {"ckpt-3", "ckpt-4", "ckpt-6"}
    with pytest.raises(KeyCompromiseHoldError):
        registry.require_promotable(checkpoint_ref="ckpt-3")

    registry.lift_hold(second.incident_digest, _grant(second.incident_digest))
    assert registry.held_checkpoints() == ()
    assert len(audit.records) == 4


def test_application_order_of_disjoint_incidents_gives_the_same_holds(ports):
    second_incident = _incident(
        incident_id="incident:" + "e" * 32,
        key_reference=OTHER_KEY,
        exposure_start=100,
        discovered_at=130,
    )
    left = _registry(ports)
    left.apply_incident(_incident())
    left.apply_incident(second_incident)
    right = _registry(ports)
    right.apply_incident(second_incident)
    right.apply_incident(_incident())
    assert left.held_checkpoints() == right.held_checkpoints()
    assert set(left.held_rounds()) == {
        "round-2",
        "round-3",
        "round-3b",
        "round-4",
        "round-5",
        "round-6",
    }


def test_malformed_state_is_refused_without_echo(ports):
    with pytest.raises(KeyCompromiseHoldError) as caught:
        _registry(ports, state={"schema_version": "x", "incidents": SITE_SENTINEL})
    assert caught.value.code is KeyCompromiseErrorCode.INVALID_STATE
    assert SITE_SENTINEL not in str(caught.value)


# Remediation classification -------------------------------------------------


def test_remediation_reaggregates_without_affected_updates_when_quorum_holds(ports):
    report = _registry(ports).apply_incident(_incident())
    assert report.remediation is RemediationClass.REAGGREGATE_WITHOUT_AFFECTED


def test_remediation_retrains_from_clean_ancestor_without_quorum(ports):
    ports["commitments"].rounds["round-3"] = RoundCommitmentSet(
        retained=frozenset({U_IN, CLEAN_A}), quorum=2
    )
    report = _registry(ports).apply_incident(_incident())
    assert report.remediation is RemediationClass.RETRAIN_FROM_CLEAN_ANCESTOR


def test_remediation_retrains_when_round_commitments_are_unavailable(ports):
    del ports["commitments"].rounds["round-3"]
    report = _registry(ports).apply_incident(_incident())
    assert report.remediation is RemediationClass.RETRAIN_FROM_CLEAN_ANCESTOR


def test_remediation_walks_past_parents_held_by_another_incident(ports):
    registry = _registry(ports)
    registry.apply_incident(
        _incident(
            incident_id="incident:" + "e" * 32,
            key_reference=OTHER_KEY,
            exposure_start=100,
            discovered_at=130,
        )
    )
    report = registry.apply_incident(_incident())
    # round-3's parent ckpt-2 is held by the first incident; ckpt-1 is clean.
    assert report.remediation is RemediationClass.RETRAIN_FROM_CLEAN_ANCESTOR


def test_remediation_is_unsupported_without_a_clean_ancestor(ports):
    ports["commitments"].by_key[COMPROMISED_KEY] = [
        AcceptedUpdateCommitment(U_IN, "round-0", 150)
    ]
    report = _registry(ports).apply_incident(_incident())
    assert report.remediation is RemediationClass.UNSUPPORTED
    assert report.held_checkpoint_count == len(ALL_CHECKPOINTS)


def test_remediation_takes_the_most_severe_root_class(ports):
    ports["commitments"].by_key[COMPROMISED_KEY].append(
        AcceptedUpdateCommitment(U_IN_R5, "round-3b", 160)
    )
    report = _registry(ports).apply_incident(_incident())
    # round-3 can be re-aggregated; round-3b has no round commitments.
    assert report.remediation is RemediationClass.RETRAIN_FROM_CLEAN_ANCESTOR


# Signed incident records and port failures -----------------------------------


def test_unsigned_or_forged_incident_is_refused_and_holds_nothing(ports, audit):
    registry = _registry(ports, audit)
    forged = _incident(key=b"attacker-key")
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.apply_incident(forged)
    assert caught.value.code is KeyCompromiseErrorCode.SIGNATURE_INVALID
    assert registry.held_checkpoints() == ()
    assert audit.records == []


def test_verifier_exceptions_and_truthy_values_do_not_admit(ports):
    class Truthy:
        def verify(self, message, signature):
            return 1

    class Raising:
        def verify(self, message, signature):
            raise RuntimeError(SITE_SENTINEL)

    for verifier in (Truthy(), Raising()):
        registry = _registry(ports, incident_verifier=verifier)
        with pytest.raises(KeyCompromiseHoldError) as caught:
            registry.apply_incident(_incident())
        assert caught.value.code is KeyCompromiseErrorCode.SIGNATURE_INVALID
        assert caught.value.__cause__ is None
        assert SITE_SENTINEL not in str(caught.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"incident_id": "incident:short"},
        {"incident_id": "not-an-incident"},
        {"key_reference": ""},
        {"key_reference": "key with spaces"},
        {"exposure_start": -1},
        {"exposure_start": True},
        {"exposure_start": 300, "discovered_at": 200},
        {"discovered_at": 1.5},
    ],
)
def test_invalid_incident_fields_are_refused_without_echo(kwargs):
    with pytest.raises(KeyCompromiseHoldError) as caught:
        _incident(**kwargs)
    assert caught.value.code is KeyCompromiseErrorCode.INVALID_INCIDENT
    for value in kwargs.values():
        if isinstance(value, str) and value:
            assert value not in str(caught.value)


def test_signed_incident_round_trips_through_strict_dict(ports):
    signed = _incident()
    payload = signed.to_dict()
    assert SignedKeyCompromiseIncident.from_dict(payload) == signed
    payload["incident"]["extra"] = SITE_SENTINEL
    with pytest.raises(KeyCompromiseHoldError) as caught:
        SignedKeyCompromiseIncident.from_dict(payload)
    assert caught.value.code is KeyCompromiseErrorCode.INVALID_INCIDENT


def test_identity_port_failure_refuses_without_leaking(ports):
    registry = _registry(ports)
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.apply_incident(_incident(key_reference="key:unknown.site"))
    assert caught.value.code is KeyCompromiseErrorCode.IDENTITY_UNAVAILABLE
    assert caught.value.__cause__ is None
    assert "unknown.site" not in str(caught.value)


def test_commitment_port_failure_refuses_without_leaking(ports):
    ports["commitments"].fail = True
    with pytest.raises(KeyCompromiseHoldError) as caught:
        _registry(ports).apply_incident(_incident())
    assert caught.value.code is KeyCompromiseErrorCode.COMMITMENT_UNAVAILABLE
    assert caught.value.__cause__ is None
    assert SITE_SENTINEL not in str(caught.value)


@pytest.mark.parametrize(
    "bad",
    [
        [AcceptedUpdateCommitment.__new__(AcceptedUpdateCommitment)],
        [object()],
        [
            AcceptedUpdateCommitment(U_IN, "round-3", 150),
            AcceptedUpdateCommitment(U_IN, "round-4", 160),
        ],
    ],
)
def test_malformed_commitment_port_output_is_refused(ports, bad):
    ports["commitments"].by_key[COMPROMISED_KEY] = bad
    with pytest.raises(KeyCompromiseHoldError) as caught:
        _registry(ports).apply_incident(_incident())
    assert caught.value.code is KeyCompromiseErrorCode.COMMITMENT_INVALID


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "unknown_parent", "cycle", "duplicate_round"],
)
def test_malformed_lineage_is_refused(ports, mutation):
    nodes = ports["lineage"].nodes
    if mutation == "duplicate":
        nodes.append(LineageNode("ckpt-1", "round-9", ("ckpt-0",)))
    elif mutation == "unknown_parent":
        nodes.append(LineageNode("ckpt-9", "round-9", ("ckpt-missing",)))
    elif mutation == "cycle":
        nodes[0] = LineageNode("ckpt-0", "round-0", ("ckpt-6",))
    else:
        nodes.append(LineageNode("ckpt-9", "round-3", ("ckpt-2",)))
    with pytest.raises(KeyCompromiseHoldError) as caught:
        _registry(ports).apply_incident(_incident())
    assert caught.value.code is KeyCompromiseErrorCode.LINEAGE_INVALID


def test_lineage_port_failure_refuses_without_leaking(ports):
    ports["lineage"].fail = True
    with pytest.raises(KeyCompromiseHoldError) as caught:
        _registry(ports).apply_incident(_incident())
    assert caught.value.code is KeyCompromiseErrorCode.LINEAGE_UNAVAILABLE
    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AcceptedUpdateCommitment("not-a-digest", "round-1", 1),
        lambda: AcceptedUpdateCommitment(U_IN, "bad round", 1),
        lambda: AcceptedUpdateCommitment(U_IN, "round-1", -5),
        lambda: RoundCommitmentSet(retained=frozenset({U_IN}), quorum=0),
        lambda: RoundCommitmentSet(retained=frozenset({"x"}), quorum=1),
        lambda: LineageNode("ckpt-1", "round-1", ("ckpt-1",)),
        lambda: LineageNode("bad ref", None, ()),
    ],
)
def test_port_value_objects_validate_inputs(factory):
    with pytest.raises(KeyCompromiseHoldError) as caught:
        factory()
    assert caught.value.code is KeyCompromiseErrorCode.INVALID_REFERENCE


# Acceptance: reports contain no site identity, key material or update values -


def _forbidden_values(signed: SignedKeyCompromiseIncident) -> tuple[str, ...]:
    return (
        SITE_SENTINEL,
        COMPROMISED_KEY,
        OPERATOR_SENTINEL,
        signed.signature,
        INCIDENT_KEY.decode(),
        OPERATOR_KEY.decode(),
        *UPDATE_SENTINELS,
        *(value[len("sha256:") :] for value in UPDATE_SENTINELS),
    )


def test_reports_audit_and_state_contain_no_identity_keys_or_update_values(
    ports, audit
):
    registry = _registry(ports, audit)
    signed = _incident()
    report = registry.apply_incident(signed)
    lifted = registry.lift_hold(report.incident_digest, _grant(report.incident_digest))

    rendered = "\n".join(
        [
            report.to_json(),
            lifted.to_json(),
            repr(report),
            repr(signed),
            repr(signed.incident),
            repr(_grant(report.incident_digest)),
            json.dumps(audit.to_payload(), sort_keys=True),
            registry.to_json(),
            repr(registry),
        ]
    )
    for forbidden in _forbidden_values(signed):
        assert forbidden not in rendered


def test_report_shape_is_closed_and_versioned(ports):
    report = _registry(ports).apply_incident(_incident())
    payload = report.to_dict()
    assert set(payload) == {
        "schema_version",
        "incident_digest",
        "status",
        "window_closed",
        "held_round_count",
        "held_checkpoint_count",
        "held_set_digest",
        "affected_commitment_digest",
        "remediation",
    }
    assert payload["schema_version"] == FEDERATED_KEY_COMPROMISE_SCHEMA_VERSION
    assert payload["incident_digest"].startswith("sha256:")
    assert payload["held_set_digest"].startswith("sha256:")
    assert report.to_json().endswith("\n")
    with pytest.raises(FrozenInstanceError):
        report.held_round_count = 0


def test_affected_commitment_digest_is_order_independent(ports):
    first = _registry(ports).apply_incident(_incident())
    ports["commitments"].by_key[COMPROMISED_KEY].reverse()
    second = _registry(ports).apply_incident(_incident())
    assert first == second


@pytest.mark.integration
def test_audited_hold_and_lift_flow_across_round_lifecycle_and_audit_chain(
    ports, audit
):
    registry = _registry(ports, audit)
    report = registry.apply_incident(_incident())
    evaluating = FederatedRoundLifecycle(state=FederatedRoundState.EVALUATING)
    with pytest.raises(KeyCompromiseHoldError):
        promote_round(
            evaluating, checkpoint_ref="ckpt-4", round_ref="round-4", registry=registry
        )
    registry.lift_hold(report.incident_digest, _grant(report.incident_digest))
    promoted = promote_round(
        evaluating, checkpoint_ref="ckpt-4", round_ref="round-4", registry=registry
    )
    assert promoted.state is FederatedRoundState.PROMOTED
    assert audit.verify()
    assert audit.records[0].payload == report.to_dict()


def test_report_lookup_returns_current_status(ports):
    registry = _registry(ports)
    report = registry.apply_incident(_incident())
    assert registry.report(report.incident_digest) == report
    registry.lift_hold(report.incident_digest, _grant(report.incident_digest))
    assert registry.report(report.incident_digest).status is IncidentStatus.LIFTED
    with pytest.raises(KeyCompromiseHoldError) as caught:
        registry.report(_digest("f"))
    assert caught.value.code is KeyCompromiseErrorCode.UNKNOWN_INCIDENT


def test_malformed_lift_grant_is_refused_without_echo():
    with pytest.raises(KeyCompromiseHoldError) as caught:
        HoldLiftGrant(
            incident_digest=_digest("d"),
            operator_reference="operator with spaces " + OPERATOR_SENTINEL,
            issued_at=1,
            signature="0" * 64,
        )
    assert caught.value.code is KeyCompromiseErrorCode.AUTHORITY_INVALID
    assert OPERATOR_SENTINEL not in str(caught.value)


def test_tampered_state_counts_are_refused(ports):
    registry = _registry(ports)
    registry.apply_incident(_incident())
    state = registry.to_dict()
    state["incidents"][0]["held_checkpoints"].pop()
    with pytest.raises(KeyCompromiseHoldError) as caught:
        _registry(ports, state=state)
    assert caught.value.code is KeyCompromiseErrorCode.INVALID_STATE


def test_malformed_revocation_time_fails_closed(ports):
    ports["revocation"].revoked[COMPROMISED_KEY] = "yesterday"
    with pytest.raises(KeyCompromiseHoldError) as caught:
        _registry(ports).apply_incident(_incident())
    assert caught.value.code is KeyCompromiseErrorCode.IDENTITY_UNAVAILABLE


def test_external_base_checkpoint_counts_as_a_clean_ancestor(ports):
    ports["lineage"].nodes[0] = LineageNode("ckpt-0", None, ())
    ports["commitments"].by_key[COMPROMISED_KEY] = [
        AcceptedUpdateCommitment(U_IN, "round-1", 150)
    ]
    report = _registry(ports).apply_incident(_incident())
    assert report.remediation is RemediationClass.RETRAIN_FROM_CLEAN_ANCESTOR
    assert report.held_checkpoint_count == len(ALL_CHECKPOINTS) - 1
