"""Synthetic controls for the privacy-safe federated contribution ledger."""

from __future__ import annotations

import copy
import json
import re
import socket

import pytest

from openmed.compliance.governance.contribution_ledger import (
    CONTRIBUTION_LEDGER_SCHEMA,
    RECEIPT_FIELDS,
    ContributionLedger,
    ContributionLedgerError,
    ContributionReceipt,
    ContributionRole,
    DisclosureProof,
    DisclosureReasonCode,
    LedgerCheckpoint,
    LedgerReasonCode,
    PublicContributionLedger,
    participant_handle,
    verify_disclosure,
    verify_ledger,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

# Synthetic sentinels: none of these may surface in public or holder output.
SENTINEL_INSTITUTION = "Sentinel Synthetic Regional Hospital"
SENTINEL_PEER = "Sentinel Synthetic Children Clinic"
SENTINEL_PATIENT_COUNT = "48213"
SENTINEL_METRIC = "0.913725"

KEY = b"synthetic-federation-key-0001-abcdefghijklmnop"
OTHER_KEY = b"synthetic-federation-key-0002-qrstuvwxyzABCDEF"
LEDGER_ID = "fed-synthetic-01"

ROUND_A = "sha256:" + "a1" * 32
ROUND_B = "sha256:" + "b2" * 32
UPDATE_A = "sha256:" + "c3" * 32
UPDATE_B = "sha256:" + "d4" * 32


def _ledger(key: bytes = KEY, ledger_id: str = LEDGER_ID) -> ContributionLedger:
    return ContributionLedger(ledger_id, key)


def _populated(
    key: bytes = KEY,
) -> tuple[ContributionLedger, list[ContributionReceipt]]:
    ledger = _ledger(key)
    site = ledger.participant_handle(SENTINEL_INSTITUTION)
    peer = ledger.participant_handle(SENTINEL_PEER)
    receipts = [
        ledger.append(
            participant_handle=site,
            round_ref=ROUND_A,
            round_role=ContributionRole.PARTICIPANT,
            policy_version=3,
            update_commitment=UPDATE_A,
        ),
        ledger.append(
            participant_handle=peer,
            round_ref=ROUND_A,
            round_role="reviewer",
            policy_version=3,
        ),
        ledger.append(
            participant_handle=site,
            round_ref=ROUND_B,
            round_role=ContributionRole.PARTICIPANT,
            policy_version=4,
            update_commitment=UPDATE_B,
        ),
    ]
    return ledger, receipts


def _document(ledger: ContributionLedger) -> dict:
    return copy.deepcopy(ledger.to_public_dict())


# ---------------------------------------------------------------------------
# Append and continuity verification
# ---------------------------------------------------------------------------


def test_append_and_verify_round_trip() -> None:
    ledger, receipts = _populated()

    report = verify_ledger(ledger.to_public_dict())

    assert report.valid is True
    assert report.reason is LedgerReasonCode.OK
    assert report.entry_count == 3
    assert report.failed_sequence is None
    assert report.head_digest == receipts[-1].entry_digest
    assert [receipt.sequence for receipt in receipts] == [0, 1, 2]
    assert ledger.verify().valid is True

    public = PublicContributionLedger.from_json(ledger.to_json())
    assert public.to_json() == ledger.to_json()
    assert verify_ledger(public).valid is True
    assert verify_ledger(ledger.to_json()).valid is True
    assert public.schema == CONTRIBUTION_LEDGER_SCHEMA
    assert len(public.entries) == 3
    for entry in public.entries:
        assert tuple(sorted(entry.commitments)) == tuple(sorted(RECEIPT_FIELDS))


def test_empty_ledger_verifies_against_genesis_checkpoint() -> None:
    ledger = _ledger()
    checkpoint = ledger.checkpoint()

    assert checkpoint.length == 0
    assert checkpoint.head_digest == ledger.to_public_dict()["genesis_digest"]
    assert verify_ledger(ledger.to_public_dict(), checkpoint=checkpoint).valid


def test_verification_with_matching_checkpoint_and_extension() -> None:
    ledger, _ = _populated()
    checkpoint = ledger.checkpoint()
    assert checkpoint == LedgerCheckpoint(3, ledger.verify().head_digest)

    assert verify_ledger(ledger.to_public_dict(), checkpoint=checkpoint).valid
    ledger.append(
        participant_handle=ledger.participant_handle(SENTINEL_PEER),
        round_ref=ROUND_B,
        round_role="coordinator",
        policy_version=4,
    )
    extended = verify_ledger(ledger.to_public_dict(), checkpoint=checkpoint)
    assert extended.valid is True
    assert extended.entry_count == 4


def test_tampered_commitment_is_detected_at_its_sequence() -> None:
    ledger, _ = _populated()
    document = _document(ledger)
    assert verify_ledger(document).valid  # negative control

    document["entries"][1]["commitments"]["round_role"] = "sha256:" + "0" * 64
    report = verify_ledger(document)

    assert report.valid is False
    assert report.reason is LedgerReasonCode.ENTRY_TAMPERED
    assert report.failed_sequence == 1
    assert report.head_digest is None


def test_rehashed_tamper_breaks_the_next_link() -> None:
    from openmed.compliance.audit_chain import AuditRecord

    ledger, _ = _populated()
    document = _document(ledger)
    entry = document["entries"][1]
    entry["commitments"]["policy_version"] = "sha256:" + "1" * 64
    entry["entry_digest"] = AuditRecord(
        sequence=1,
        event_type="contribution_receipt",
        payload={
            "commitments": entry["commitments"],
            "ledger_id": LEDGER_ID,
            "schema": CONTRIBUTION_LEDGER_SCHEMA,
        },
        previous_hash=entry["previous_digest"],
    ).compute_hash()

    report = verify_ledger(document)

    assert report.reason is LedgerReasonCode.LINK_MISMATCH
    assert report.failed_sequence == 2


def test_rewritten_tail_is_detected_by_checkpoint_as_fork() -> None:
    ledger, _ = _populated()
    checkpoint = ledger.checkpoint()
    branch = _ledger()
    site = branch.participant_handle(SENTINEL_INSTITUTION)
    peer = branch.participant_handle(SENTINEL_PEER)
    branch.append(
        participant_handle=site,
        round_ref=ROUND_A,
        round_role="participant",
        policy_version=3,
        update_commitment=UPDATE_A,
    )
    branch.append(
        participant_handle=peer,
        round_ref=ROUND_A,
        round_role="reviewer",
        policy_version=3,
    )
    branch.append(
        participant_handle=site,
        round_ref=ROUND_B,
        round_role="participant",
        policy_version=4,
        update_commitment=UPDATE_A,  # diverges from the published history
    )

    assert verify_ledger(branch.to_public_dict()).valid  # internally consistent
    report = verify_ledger(branch.to_public_dict(), checkpoint=checkpoint)

    assert report.valid is False
    assert report.reason is LedgerReasonCode.FORK_DETECTED


def test_reordered_entries_are_detected() -> None:
    ledger, _ = _populated()
    document = _document(ledger)
    entries = document["entries"]
    entries[0], entries[1] = entries[1], entries[0]

    report = verify_ledger(document)

    assert report.reason is LedgerReasonCode.SEQUENCE_REORDER
    assert report.valid is False


def test_missing_entry_is_detected_as_gap() -> None:
    ledger, _ = _populated()
    document = _document(ledger)
    del document["entries"][1]

    assert verify_ledger(document).reason is LedgerReasonCode.SEQUENCE_GAP

    head_dropped = _document(ledger)
    del head_dropped["entries"][0]
    assert verify_ledger(head_dropped).reason is LedgerReasonCode.SEQUENCE_GAP


def test_duplicate_sequence_from_another_branch_is_a_fork() -> None:
    ledger, _ = _populated()
    branch, _ = _populated(OTHER_KEY)
    document = _document(ledger)
    # Same ledger id, different commitments at sequence 2.
    foreign = copy.deepcopy(branch.to_public_dict()["entries"][2])
    document["entries"].append(foreign)

    assert verify_ledger(document).reason is LedgerReasonCode.FORK_DETECTED

    shared_parent = _document(ledger)
    shared_parent["entries"][2]["previous_digest"] = shared_parent["entries"][1][
        "previous_digest"
    ]
    assert verify_ledger(shared_parent).reason is LedgerReasonCode.FORK_DETECTED


def test_truncation_is_detected_with_checkpoint() -> None:
    ledger, _ = _populated()
    checkpoint = ledger.checkpoint()
    document = _document(ledger)
    del document["entries"][-1]

    # A truncated prefix is internally consistent; only an anchor detects it.
    assert verify_ledger(document).valid is True
    report = verify_ledger(document, checkpoint=checkpoint)
    assert report.valid is False
    assert report.reason is LedgerReasonCode.TRUNCATED

    document["entries"] = []
    assert (
        verify_ledger(document, checkpoint=checkpoint).reason
        is LedgerReasonCode.TRUNCATED
    )


def test_genesis_and_ledger_identity_are_bound() -> None:
    ledger, _ = _populated()

    renamed = _document(ledger)
    renamed["ledger_id"] = "fed-synthetic-02"
    assert verify_ledger(renamed).reason is LedgerReasonCode.GENESIS_MISMATCH

    regenesis = _document(ledger)
    regenesis["genesis_digest"] = "sha256:" + "2" * 64
    assert verify_ledger(regenesis).reason is LedgerReasonCode.GENESIS_MISMATCH

    relinked = _document(ledger)
    relinked["entries"][0]["previous_digest"] = "sha256:" + "3" * 64
    assert verify_ledger(relinked).reason in {
        LedgerReasonCode.ENTRY_TAMPERED,
        LedgerReasonCode.LINK_MISMATCH,
    }


def test_malformed_documents_return_stable_code() -> None:
    ledger, _ = _populated()
    bad_inputs: list[object] = [
        "{not json",
        [],
        {},
        {**_document(ledger), "schema": "openmed.other.v1"},
        {**_document(ledger), "extra": 1},
    ]
    entry_extra = _document(ledger)
    entry_extra["entries"][0]["note"] = "x"
    bad_inputs.append(entry_extra)
    bad_digest = _document(ledger)
    bad_digest["entries"][0]["commitments"]["participant_handle"] = "not-a-digest"
    bad_inputs.append(bad_digest)
    missing_field = _document(ledger)
    del missing_field["entries"][0]["commitments"]["round_ref"]
    bad_inputs.append(missing_field)
    bool_sequence = _document(ledger)
    bool_sequence["entries"][0]["sequence"] = False
    bad_inputs.append(bool_sequence)

    for bad in bad_inputs:
        report = verify_ledger(bad)
        assert report.valid is False
        assert report.reason is LedgerReasonCode.MALFORMED

    with pytest.raises(ContributionLedgerError) as caught:
        PublicContributionLedger.from_json("{not json")
    assert caught.value.code == "contribution_ledger_invalid_document"
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_reason_code_vocabulary_is_stable() -> None:
    assert [code.value for code in LedgerReasonCode] == [
        "ok",
        "ledger_malformed",
        "genesis_mismatch",
        "fork_detected",
        "sequence_gap",
        "sequence_reorder",
        "entry_tampered",
        "link_mismatch",
        "truncated",
    ]
    assert [code.value for code in DisclosureReasonCode] == [
        "ok",
        "proof_malformed",
        "ledger_invalid",
        "ledger_mismatch",
        "entry_not_found",
        "entry_mismatch",
        "opening_invalid",
    ]
    assert [role.value for role in ContributionRole] == [
        "participant",
        "coordinator",
        "reviewer",
        "incident_responder",
    ]
    assert RECEIPT_FIELDS == (
        "participant_handle",
        "round_ref",
        "round_role",
        "policy_version",
        "update_commitment",
    )


# ---------------------------------------------------------------------------
# Selective disclosure
# ---------------------------------------------------------------------------


def test_selective_disclosure_verifies_only_chosen_fields() -> None:
    ledger, receipts = _populated()
    receipt = receipts[0]

    proof = receipt.disclose(["round_role", "policy_version"])
    result = verify_disclosure(ledger.to_public_dict(), proof)

    assert result.valid is True
    assert result.reason is DisclosureReasonCode.OK
    assert result.sequence == 0
    assert dict(result.disclosed) == {"round_role": "participant", "policy_version": 3}

    proof_bytes = proof.to_json()
    hidden_handle = ledger.participant_handle(SENTINEL_INSTITUTION)
    for hidden in (hidden_handle, ROUND_A, UPDATE_A):
        assert hidden not in proof_bytes
    assert hidden_handle.split(":", 1)[1] not in proof_bytes
    decoded = json.loads(proof_bytes)
    assert sorted(decoded["openings"]) == ["policy_version", "round_role"]
    for name in ("participant_handle", "round_ref", "update_commitment"):
        assert name not in decoded["openings"]
        assert name not in proof_bytes

    # Round trip through serialized bytes still verifies.
    reparsed = DisclosureProof.from_json(proof_bytes)
    assert verify_disclosure(ledger.to_json(), reparsed).valid is True
    assert verify_disclosure(ledger.to_public_dict(), proof_bytes).valid is True


def test_each_field_can_be_disclosed_alone() -> None:
    ledger, receipts = _populated()
    public = ledger.public_ledger()
    expected = {
        "participant_handle": ledger.participant_handle(SENTINEL_INSTITUTION),
        "round_ref": ROUND_B,
        "round_role": "participant",
        "policy_version": 4,
        "update_commitment": UPDATE_B,
    }
    for name in RECEIPT_FIELDS:
        result = verify_disclosure(public, receipts[2].disclose([name]))
        assert result.valid is True
        assert dict(result.disclosed) == {name: expected[name]}

    reviewer = verify_disclosure(public, receipts[1].disclose(["update_commitment"]))
    assert reviewer.valid is True
    assert dict(reviewer.disclosed) == {"update_commitment": None}


def test_public_ledger_does_not_link_repeat_participants() -> None:
    ledger, receipts = _populated()
    entries = ledger.to_public_dict()["entries"]
    first, third = entries[0]["commitments"], entries[2]["commitments"]

    assert first["participant_handle"] != third["participant_handle"]
    # Openings still show the same pseudonym to an authorized verifier.
    a = verify_disclosure(
        ledger.public_ledger(), receipts[0].disclose(["participant_handle"])
    )
    b = verify_disclosure(
        ledger.public_ledger(), receipts[2].disclose(["participant_handle"])
    )
    assert a.disclosed == b.disclosed


def test_wrong_openings_are_rejected() -> None:
    ledger, receipts = _populated()
    public = ledger.to_public_dict()
    good = receipts[0].disclose(["round_role", "round_ref"]).to_dict()
    assert verify_disclosure(public, good).valid is True  # negative control

    wrong_salt = copy.deepcopy(good)
    wrong_salt["openings"]["round_role"]["salt"] = "f" * 64
    wrong_value = copy.deepcopy(good)
    wrong_value["openings"]["round_role"]["value"] = "coordinator"
    swapped_field = copy.deepcopy(good)
    swapped_field["openings"]["round_ref"]["value"] = UPDATE_A
    for bad in (wrong_salt, wrong_value, swapped_field):
        result = verify_disclosure(public, bad)
        assert result.valid is False
        assert result.reason is DisclosureReasonCode.OPENING_INVALID
        assert dict(result.disclosed) == {}

    moved = copy.deepcopy(good)
    moved["sequence"] = 2
    assert (
        verify_disclosure(public, moved).reason is DisclosureReasonCode.ENTRY_MISMATCH
    )

    missing = copy.deepcopy(good)
    missing["sequence"] = 9
    assert (
        verify_disclosure(public, missing).reason
        is DisclosureReasonCode.ENTRY_NOT_FOUND
    )

    foreign_ledger = copy.deepcopy(good)
    foreign_ledger["ledger_id"] = "fed-synthetic-02"
    assert (
        verify_disclosure(public, foreign_ledger).reason
        is DisclosureReasonCode.LEDGER_MISMATCH
    )


def test_opening_from_other_key_ledger_is_rejected() -> None:
    ledger, _ = _populated()
    _, other_receipts = _populated(OTHER_KEY)

    result = verify_disclosure(
        ledger.to_public_dict(), other_receipts[0].disclose(["round_role"])
    )

    assert result.valid is False
    assert result.reason is DisclosureReasonCode.ENTRY_MISMATCH


def test_disclosure_against_broken_ledger_is_rejected() -> None:
    ledger, receipts = _populated()
    document = _document(ledger)
    del document["entries"][1]

    result = verify_disclosure(document, receipts[0].disclose(["round_role"]))

    assert result.valid is False
    assert result.reason is DisclosureReasonCode.LEDGER_INVALID

    checkpoint = ledger.checkpoint()
    truncated = _document(ledger)
    del truncated["entries"][-1]
    assert (
        verify_disclosure(
            truncated, receipts[0].disclose(["round_role"]), checkpoint=checkpoint
        ).reason
        is DisclosureReasonCode.LEDGER_INVALID
    )


def test_malformed_proofs_and_disclosure_requests() -> None:
    ledger, receipts = _populated()
    public = ledger.public_ledger()
    good = receipts[0].disclose(["round_role"]).to_dict()

    unknown = copy.deepcopy(good)
    unknown["openings"]["patient_count"] = {"salt": "a" * 64, "value": 5}
    metrics = {**copy.deepcopy(good), "metrics": {"auc": 0.9}}
    bad_salt = copy.deepcopy(good)
    bad_salt["openings"]["round_role"]["salt"] = "xyz"
    empty = {**copy.deepcopy(good), "openings": {}}
    for bad in (unknown, metrics, bad_salt, empty, "{not json", 7):
        result = verify_disclosure(public, bad)
        assert result.valid is False
        assert result.reason is DisclosureReasonCode.PROOF_MALFORMED

    with pytest.raises(ContributionLedgerError) as caught:
        receipts[0].disclose(["local_f1"])
    assert caught.value.code == "contribution_ledger_unknown_field"
    with pytest.raises(ContributionLedgerError) as caught:
        receipts[0].disclose([])
    assert caught.value.code == "contribution_ledger_empty_disclosure"
    with pytest.raises(ContributionLedgerError) as caught:
        receipts[0].disclose("round_role")
    assert caught.value.code == "contribution_ledger_unknown_field"


def test_receipt_round_trips_for_holder_storage() -> None:
    ledger, receipts = _populated()
    stored = json.dumps(receipts[1].to_dict(), sort_keys=True)

    restored = ContributionReceipt.from_dict(json.loads(stored))

    assert restored == receipts[1]
    assert verify_disclosure(
        ledger.public_ledger(), restored.disclose(RECEIPT_FIELDS)
    ).valid
    with pytest.raises(ContributionLedgerError) as caught:
        ContributionReceipt.from_dict({**receipts[1].to_dict(), "site_name": "x"})
    assert caught.value.code == "contribution_ledger_invalid_receipt"


# ---------------------------------------------------------------------------
# Pseudonyms and keys
# ---------------------------------------------------------------------------


def test_pseudonyms_are_stable_per_key_and_unlinkable_across_keys() -> None:
    first = participant_handle(KEY, SENTINEL_INSTITUTION)
    again = participant_handle(KEY, SENTINEL_INSTITUTION)
    peer = participant_handle(KEY, SENTINEL_PEER)
    other = participant_handle(OTHER_KEY, SENTINEL_INSTITUTION)

    assert first == again == _ledger().participant_handle(SENTINEL_INSTITUTION)
    assert first != peer
    assert first != other
    for handle in (first, peer, other):
        assert handle.startswith("pseud:")
        assert len(handle) == len("pseud:") + 64
        assert SENTINEL_INSTITUTION not in handle
        assert SENTINEL_INSTITUTION.lower().replace(" ", "") not in handle
    # Unlinkable: no shared prefix structure beyond the fixed tag.
    assert first[6:14] != other[6:14]
    # Same key, different ledger ids keep the same per-federation pseudonym.
    assert (
        _ledger(ledger_id="fed-synthetic-99").participant_handle(SENTINEL_INSTITUTION)
        == first
    )


def test_invalid_keys_and_labels_are_rejected_without_echo() -> None:
    for bad_key in (b"short", "synthetic-federation-key-as-text-0000000", None, b""):
        with pytest.raises(ContributionLedgerError) as caught:
            ContributionLedger(LEDGER_ID, bad_key)  # type: ignore[arg-type]
        assert caught.value.code == "contribution_ledger_invalid_key"
        assert str(caught.value) == "contribution_ledger_invalid_key"
    for bad_label in ("", 42, None, "x" * 513):
        with pytest.raises(ContributionLedgerError) as caught:
            participant_handle(KEY, bad_label)  # type: ignore[arg-type]
        assert caught.value.code == "contribution_ledger_invalid_label"
    for bad_id in ("", "Has Spaces", SENTINEL_INSTITUTION, "x" * 65, 3):
        with pytest.raises(ContributionLedgerError) as caught:
            ContributionLedger(bad_id, KEY)  # type: ignore[arg-type]
        assert caught.value.code == "contribution_ledger_invalid_ledger_id"


def test_append_rejects_raw_identifiers_and_bad_fields() -> None:
    ledger = _ledger()
    handle = ledger.participant_handle(SENTINEL_INSTITUTION)
    valid = dict(
        participant_handle=handle,
        round_ref=ROUND_A,
        round_role="participant",
        policy_version=1,
        update_commitment=UPDATE_A,
    )
    cases = [
        ({"participant_handle": SENTINEL_INSTITUTION}, "invalid_handle"),
        ({"participant_handle": "pseud:" + "A" * 64}, "invalid_handle"),
        ({"round_ref": "round-7"}, "invalid_round_ref"),
        ({"round_role": "aggregator"}, "invalid_role"),
        ({"round_role": SENTINEL_INSTITUTION}, "invalid_role"),
        ({"policy_version": 0}, "invalid_policy_version"),
        ({"policy_version": True}, "invalid_policy_version"),
        ({"policy_version": "3"}, "invalid_policy_version"),
        ({"update_commitment": SENTINEL_METRIC}, "invalid_update_commitment"),
        ({"update_commitment": None}, "invalid_update_commitment"),
    ]
    for change, code in cases:
        with pytest.raises(ContributionLedgerError) as caught:
            ledger.append(**{**valid, **change})
        assert caught.value.code == f"contribution_ledger_{code}"
        assert SENTINEL_INSTITUTION not in str(caught.value)
        assert SENTINEL_METRIC not in str(caught.value)
    with pytest.raises(ContributionLedgerError) as caught:
        ledger.append(**{**valid, "round_role": "reviewer"})
    assert caught.value.code == "contribution_ledger_invalid_update_commitment"
    assert len(ledger) == 0  # nothing appended on failure


def test_metric_like_fields_are_refused() -> None:
    ledger = _ledger()
    handle = ledger.participant_handle(SENTINEL_INSTITUTION)
    base = dict(
        participant_handle=handle,
        round_ref=ROUND_A,
        round_role="participant",
        policy_version=1,
        update_commitment=UPDATE_A,
    )
    for refused in (
        {"patient_count": int(SENTINEL_PATIENT_COUNT)},
        {"local_f1": float(SENTINEL_METRIC)},
        {"institution": SENTINEL_INSTITUTION},
        {"performance_slices": {"age_65_plus": float(SENTINEL_METRIC)}},
    ):
        with pytest.raises(ContributionLedgerError) as caught:
            ledger.append(**base, **refused)
        assert caught.value.code == "contribution_ledger_field_refused"
        message = str(caught.value)
        for sentinel in (SENTINEL_INSTITUTION, SENTINEL_PATIENT_COUNT, SENTINEL_METRIC):
            assert sentinel not in message
    assert len(ledger) == 0


def test_update_metadata_supplies_commitment_digest() -> None:
    parameter = FederatedParameterMetadata("lora.a", (2, 2), "float32")
    policy = FederatedUpdatePolicy(
        model_digest="sha256:" + "e5" * 32, parameters=(parameter,)
    )
    metadata = FederatedUpdateMetadata(
        model_digest="sha256:" + "e5" * 32,
        adapter_format="dense",
        parameters=(parameter,),
        total_elements=4,
        update_digest=UPDATE_B,
        clipped=True,
        policy=policy,
    )
    ledger = _ledger()
    receipt = ledger.append(
        participant_handle=ledger.participant_handle(SENTINEL_PEER),
        round_ref=ROUND_A,
        round_role="participant",
        policy_version=2,
        update_commitment=metadata,
    )

    result = verify_disclosure(
        ledger.public_ledger(), receipt.disclose(["update_commitment"])
    )
    assert dict(result.disclosed) == {"update_commitment": UPDATE_B}


# ---------------------------------------------------------------------------
# Leakage, determinism, resume and offline behavior
# ---------------------------------------------------------------------------


def test_sentinels_never_appear_in_public_holder_or_report_output() -> None:
    ledger, receipts = _populated()
    proof = receipts[0].disclose(RECEIPT_FIELDS)
    surfaces = [
        ledger.to_json(),
        json.dumps(ledger.to_public_dict()),
        repr(ledger),
        repr(ledger.public_ledger()),
        repr(ledger.checkpoint()),
        repr(ledger.verify()),
        json.dumps(ledger.verify().to_dict()),
        proof.to_json(),
        repr(proof),
        repr(verify_disclosure(ledger.public_ledger(), proof)),
        json.dumps(verify_disclosure(ledger.public_ledger(), proof).to_dict()),
    ]
    for receipt in receipts:
        surfaces.append(repr(receipt))
        surfaces.append(json.dumps(receipt.to_dict()))
    joined = "\n".join(surfaces)
    # Digests are hex, so match the count only as a standalone number.
    assert (
        re.search(rf"(?<![0-9a-f]){SENTINEL_PATIENT_COUNT}(?![0-9a-f])", joined) is None
    )
    for sentinel in (
        SENTINEL_INSTITUTION,
        SENTINEL_PEER,
        SENTINEL_METRIC,
        KEY.decode("ascii"),
        OTHER_KEY.decode("ascii"),
        "patient",
        "metric",
    ):
        assert sentinel not in joined
    # Holder-facing reprs do not show pseudonyms or salts either.
    handle = ledger.participant_handle(SENTINEL_INSTITUTION)
    for receipt in receipts:
        assert handle not in repr(receipt)
        assert UPDATE_A not in repr(receipt)
    assert handle not in repr(proof)
    assert handle not in repr(verify_disclosure(ledger.public_ledger(), proof))
    assert KEY.decode("ascii") not in repr(ledger)


def test_public_ledger_omits_all_plaintext_receipt_values() -> None:
    ledger, _ = _populated()
    public_bytes = ledger.to_json()
    for plain in (
        ledger.participant_handle(SENTINEL_INSTITUTION),
        ledger.participant_handle(SENTINEL_PEER),
        ROUND_A,
        ROUND_B,
        UPDATE_A,
        UPDATE_B,
        '"participant"',
        '"reviewer"',
    ):
        assert plain not in public_bytes


def test_ledger_output_is_deterministic() -> None:
    first, first_receipts = _populated()
    second, second_receipts = _populated()

    assert first.to_json() == second.to_json()
    assert first_receipts == second_receipts
    assert (
        first_receipts[0].disclose(["round_ref"]).to_json()
        == second_receipts[0].disclose(["round_ref"]).to_json()
    )
    other, _ = _populated(OTHER_KEY)
    assert other.to_json() != first.to_json()


def test_resume_extends_a_verified_ledger_only() -> None:
    ledger, _ = _populated()
    resumed = ContributionLedger.resume(ledger.to_public_dict(), KEY)
    assert resumed.to_json() == ledger.to_json()

    handle = ledger.participant_handle(SENTINEL_PEER)
    fields = dict(
        participant_handle=handle,
        round_ref=ROUND_B,
        round_role="reviewer",
        policy_version=4,
    )
    assert resumed.append(**fields) == ledger.append(**fields)
    assert resumed.to_json() == ledger.to_json()

    broken = _document(ledger)
    del broken["entries"][1]
    with pytest.raises(ContributionLedgerError) as caught:
        ContributionLedger.resume(broken, KEY)
    assert caught.value.code == "contribution_ledger_unverified"
    with pytest.raises(ContributionLedgerError) as caught:
        ContributionLedger.resume(
            ledger.to_public_dict(),
            KEY,
            checkpoint=LedgerCheckpoint(9, ledger.checkpoint().head_digest),
        )
    assert caught.value.code == "contribution_ledger_unverified"


def test_checkpoint_validation() -> None:
    for length, head in ((-1, "sha256:" + "0" * 64), (1, "nope"), (True, None)):
        with pytest.raises(ContributionLedgerError) as caught:
            LedgerCheckpoint(length, head)  # type: ignore[arg-type]
        assert caught.value.code == "contribution_ledger_invalid_checkpoint"


def test_full_workflow_runs_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)

    ledger, receipts = _populated()
    assert verify_ledger(ledger.to_json(), checkpoint=ledger.checkpoint()).valid
    assert verify_disclosure(
        ledger.to_json(), receipts[2].disclose(["round_ref"])
    ).valid


def test_ledger_with_derived_keys_cannot_be_pickled() -> None:
    import pickle

    with pytest.raises(TypeError):
        pickle.dumps(_ledger())
