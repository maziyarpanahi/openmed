"""Offline multi-site flow: commit splits, train on train only, cite holdouts."""

from __future__ import annotations

import json

import pytest

from openmed.training.federated_holdout_split import (
    FederatedHoldoutLedger,
    FederatedHoldoutSplitError,
    HoldoutRotationReason,
    build_site_holdout_split,
)

_FEDERATION = "fed-synthetic-flow"


def _site_notes(site: str, patients: int = 40) -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{site}-NOTE-{index:03d}-{note}", f"{site}-PATIENT-{index:03d}")
        for index in range(patients)
        for note in range(3)
    )


def _patient(note: tuple[str, str]) -> str:
    return note[1]


@pytest.mark.integration
def test_sites_commit_before_training_and_rotation_invalidates_results() -> None:
    secrets = {
        "site-a": b"synthetic-flow-secret-site-a-000000",
        "site-b": b"synthetic-flow-secret-site-b-000000",
    }
    notes = {site: _site_notes(site.upper()) for site in secrets}
    ledger = FederatedHoldoutLedger(_FEDERATION)

    # Training cannot start for any site before its holdout is committed.
    for site in secrets:
        with pytest.raises(FederatedHoldoutSplitError) as excinfo:
            ledger.require_active_commitment(site)
        assert excinfo.value.code.value == "holdout_not_committed"

    splits = {}
    transfers = []
    for site, secret in secrets.items():
        split = build_site_holdout_split(
            [_patient(note) for note in notes[site]],
            site_secret=secret,
            federation_id=_FEDERATION,
            holdout_fraction=0.25,
        )
        splits[site] = split
        wire = split.export_evidence().to_json()
        transfers.append(wire)
        ledger.register(site, json.loads(wire))

    evaluation_results = {}
    for round_index in range(3):
        for site, split in splits.items():
            active = ledger.require_active_commitment(site)
            train, holdout = split.partition_examples(
                notes[site], patient_key_of=_patient
            )
            admitted = split.admit_training_examples(train, patient_key_of=_patient)
            assert admitted == train
            with pytest.raises(FederatedHoldoutSplitError):
                split.admit_training_examples(
                    (*train, holdout[-1]), patient_key_of=_patient
                )
            evaluation_results[(site, round_index)] = {
                "holdout_commitment_digest": active,
                "round_index": round_index,
            }
            # The split is fixed across rounds.
            assert split.commitment_digest == active

    for (site, _), result in evaluation_results.items():
        assert ledger.verify_evaluation_evidence(site, result).valid

    # A re-split between rounds without a recorded rotation fails closed.
    resplit = build_site_holdout_split(
        [_patient(note) for note in notes["site-a"]],
        site_secret=b"synthetic-flow-secret-site-a-rotated",
        federation_id=_FEDERATION,
        holdout_fraction=0.25,
        rotation_epoch=1,
    )
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        ledger.register("site-a", resplit.export_evidence())
    assert excinfo.value.code.value == "recommitment_without_rotation"

    # A recorded rotation accepts the new split and invalidates old results.
    ledger.record_rotation("site-a", HoldoutRotationReason.SECRET_COMPROMISED)
    ledger.register("site-a", resplit.export_evidence())
    for (site, _), result in evaluation_results.items():
        verification = ledger.verify_evaluation_evidence(site, result)
        if site == "site-a":
            assert verification.reason_codes == ("holdout_commitment_rotated",)
        else:
            assert verification.valid

    exported = "\n".join((*transfers, ledger.to_json()))
    for sentinel in ("PATIENT", "NOTE", "synthetic-flow-secret"):
        assert sentinel not in exported
