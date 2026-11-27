"""Patient-level site holdout splits committed before federated training."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import pytest

from openmed.eval.governance.holdout_commitment import verify_holdout_commitment
from openmed.training.federated_holdout_split import (
    FEDERATED_HOLDOUT_EVIDENCE_SCHEMA_VERSION,
    FederatedHoldoutLedger,
    FederatedHoldoutSplitError,
    HoldoutPatientCountBand,
    HoldoutRotationReason,
    HoldoutSplitPartition,
    HoldoutSplitReasonCode,
    SiteHoldoutEvidence,
    build_site_holdout_split,
)

_SECRET = b"synthetic-site-secret-0123456789abcdef"
_OTHER_SECRET = b"synthetic-site-secret-fedcba9876543210"
_FEDERATION = "fed-synthetic-2026"
_SITE = "site-a"


@dataclass(frozen=True)
class _Note:
    note_id: str
    patient_key: str


def _patients(count: int = 60) -> tuple[str, ...]:
    return tuple(f"SYNTH-PATIENT-{index:04d}" for index in range(count))


def _notes(count: int = 60, notes_per_patient: int = 4) -> tuple[_Note, ...]:
    return tuple(
        _Note(f"SYNTH-NOTE-{index:04d}-{note:02d}", patient)
        for index, patient in enumerate(_patients(count))
        for note in range(notes_per_patient)
    )


def _split(**overrides):
    arguments = {
        "patient_keys": [note.patient_key for note in _notes()],
        "site_secret": _SECRET,
        "federation_id": _FEDERATION,
        "holdout_fraction": 0.25,
    }
    arguments.update(overrides)
    return build_site_holdout_split(**arguments)


def _code(excinfo: pytest.ExceptionInfo[FederatedHoldoutSplitError]) -> str:
    return excinfo.value.code.value


# Acceptance: patients with several notes land entirely in one partition.


def test_every_patient_lands_entirely_in_train_or_holdout() -> None:
    split = _split()
    train, holdout = split.partition_examples(
        _notes(), patient_key_of=lambda note: note.patient_key
    )

    assert train and holdout
    train_patients = {note.patient_key for note in train}
    holdout_patients = {note.patient_key for note in holdout}
    assert train_patients.isdisjoint(holdout_patients)
    assert train_patients | holdout_patients == set(_patients())
    for patient in _patients():
        partitions = {split.partition_of(patient)}
        assert len(partitions) == 1
        expected = len([n for n in _notes() if n.patient_key == patient])
        bucket = holdout if partitions == {HoldoutSplitPartition.HOLDOUT} else train
        assert len([n for n in bucket if n.patient_key == patient]) == expected


def test_split_is_deterministic_under_a_fixed_secret_and_order_independent() -> None:
    first = _split()
    second = _split(patient_keys=list(reversed(_patients())))

    assert first.commitment_digest == second.commitment_digest
    assert all(
        first.partition_of(patient) is second.partition_of(patient)
        for patient in _patients()
    )


def test_a_different_secret_changes_assignment_and_commitment() -> None:
    first = _split()
    second = _split(site_secret=_OTHER_SECRET)

    assert first.commitment_digest != second.commitment_digest
    assert any(
        first.partition_of(patient) is not second.partition_of(patient)
        for patient in _patients()
    )


def test_a_different_federation_or_epoch_changes_the_commitment() -> None:
    base = _split().commitment_digest

    assert _split(federation_id="fed-synthetic-other").commitment_digest != base
    assert _split(rotation_epoch=1).commitment_digest != base


def test_holdout_fraction_controls_the_split_roughly() -> None:
    split = _split(patient_keys=_patients(400), holdout_fraction=0.2)
    holdout = [
        patient
        for patient in _patients(400)
        if split.partition_of(patient) is HoldoutSplitPartition.HOLDOUT
    ]
    assert 40 <= len(holdout) <= 120


def test_note_level_assignment_negative_control_would_mix_patients() -> None:
    # Negative control: hashing note identifiers instead of patient keys
    # places the same patient in both partitions, which the builder prevents.
    note_split = _split(patient_keys=[note.note_id for note in _notes()])
    mixed = {
        patient
        for patient in _patients()
        if len(
            {
                note_split.partition_of(note.note_id)
                for note in _notes()
                if note.patient_key == patient
            }
        )
        > 1
    }
    assert mixed


def test_site_local_commitment_reuses_the_holdout_commitment_primitive() -> None:
    split = _split()
    local = split._commitment

    assert verify_holdout_commitment(local).valid
    assert local.commitment_digest == split.commitment_digest
    holdout_count = sum(
        split.partition_of(patient) is HoldoutSplitPartition.HOLDOUT
        for patient in _patients()
    )
    assert local.manifest_item_counts["case"] == holdout_count


# Acceptance: training input that contains a holdout member is refused.


def test_training_port_admits_only_train_partition_examples() -> None:
    split = _split()
    train, _ = split.partition_examples(
        _notes(), patient_key_of=lambda note: note.patient_key
    )

    admitted = split.admit_training_examples(
        train, patient_key_of=lambda note: note.patient_key
    )
    assert admitted == tuple(train)


def test_training_port_refuses_any_committed_holdout_member() -> None:
    split = _split()
    train, holdout = split.partition_examples(
        _notes(), patient_key_of=lambda note: note.patient_key
    )
    leaked = (*train, holdout[0])

    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        split.admit_training_examples(
            leaked, patient_key_of=lambda note: note.patient_key
        )
    assert _code(excinfo) == "holdout_member_in_training"
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        split.require_training_patient(holdout[0].patient_key)
    assert excinfo.value.code is HoldoutSplitReasonCode.HOLDOUT_MEMBER_IN_TRAINING


def test_training_port_refuses_the_whole_batch_from_a_stream() -> None:
    split = _split()
    train, holdout = split.partition_examples(
        _notes(), patient_key_of=lambda note: note.patient_key
    )
    consumed: list[_Note] = []

    def stream():
        for note in (holdout[0], *train):
            consumed.append(note)
            yield note

    admitted = None
    with pytest.raises(FederatedHoldoutSplitError):
        admitted = split.admit_training_examples(
            stream(), patient_key_of=lambda note: note.patient_key
        )
    assert admitted is None
    assert consumed


def test_training_port_refuses_patients_outside_the_committed_split() -> None:
    split = _split()

    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        split.require_training_patient("SYNTH-PATIENT-NEW-9999")
    assert _code(excinfo) == "unassigned_patient"
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        split.partition_of("SYNTH-PATIENT-NEW-9999")
    assert _code(excinfo) == "unassigned_patient"


@pytest.mark.parametrize("bad_key", ["", b"", None, 7, ("SYNTH",)])
def test_invalid_patient_keys_fail_closed(bad_key) -> None:
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        _split(patient_keys=[*_patients(), bad_key])
    assert _code(excinfo) == "invalid_patient_key"
    split = _split()
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        split.require_training_patient(bad_key)
    assert _code(excinfo) == "invalid_patient_key"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"site_secret": b"short"}, "invalid_site_secret"),
        (
            {"site_secret": "synthetic-site-secret-0123456789abcdef"},
            "invalid_site_secret",
        ),
        ({"federation_id": "Bad Federation!"}, "invalid_federation_id"),
        ({"federation_id": ""}, "invalid_federation_id"),
        ({"holdout_fraction": 0.0}, "invalid_holdout_fraction"),
        ({"holdout_fraction": 1.0}, "invalid_holdout_fraction"),
        ({"holdout_fraction": float("nan")}, "invalid_holdout_fraction"),
        ({"holdout_fraction": True}, "invalid_holdout_fraction"),
        ({"rotation_epoch": -1}, "invalid_rotation_epoch"),
        ({"rotation_epoch": True}, "invalid_rotation_epoch"),
        ({"minimum_group_size": 1}, "invalid_minimum_group_size"),
        ({"patient_keys": "SYNTH-PATIENT-0001"}, "invalid_patient_key"),
        ({"patient_keys": ["SYNTH-ONLY-PATIENT"]}, "empty_partition"),
    ],
)
def test_builder_rejects_unsafe_configuration(overrides, code) -> None:
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        _split(**overrides)
    assert _code(excinfo) == code
    assert str(excinfo.value) == code


# Acceptance: re-splitting without a recorded rotation fails closed.


def test_ledger_accepts_first_commitment_and_identical_resubmission() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    evidence = _split().export_evidence()

    ledger.register(_SITE, evidence)
    ledger.register(_SITE, evidence.to_dict())

    assert ledger.require_active_commitment(_SITE) == evidence.commitment_digest


def test_resplitting_without_a_recorded_rotation_fails_closed() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    ledger.register(_SITE, _split().export_evidence())

    for resplit in (
        _split(site_secret=_OTHER_SECRET),
        _split(holdout_fraction=0.4),
        _split(patient_keys=_patients(59)),
        _split(rotation_epoch=1),
    ):
        with pytest.raises(FederatedHoldoutSplitError) as excinfo:
            ledger.register(_SITE, resplit.export_evidence())
        assert _code(excinfo) == "recommitment_without_rotation"
    assert ledger.require_active_commitment(_SITE) == _split().commitment_digest


def test_training_cannot_start_before_a_commitment_exists() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)

    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        ledger.require_active_commitment(_SITE)
    assert _code(excinfo) == "holdout_not_committed"
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        ledger.record_rotation(_SITE, HoldoutRotationReason.COHORT_REFRESH)
    assert _code(excinfo) == "holdout_not_committed"


def test_recorded_rotation_allows_new_commitment_and_invalidates_old_results() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    original = _split().export_evidence()
    ledger.register(_SITE, original)
    old_result = {"holdout_commitment_digest": original.commitment_digest}
    assert ledger.verify_evaluation_evidence(_SITE, old_result).valid

    ledger.record_rotation(_SITE, HoldoutRotationReason.SECRET_COMPROMISED)
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        ledger.require_active_commitment(_SITE)
    assert _code(excinfo) == "holdout_not_committed"

    rotated = _split(site_secret=_OTHER_SECRET, rotation_epoch=1).export_evidence()
    ledger.register(_SITE, rotated)
    assert ledger.require_active_commitment(_SITE) == rotated.commitment_digest

    stale = ledger.verify_evaluation_evidence(_SITE, old_result)
    assert not stale.valid
    assert stale.reason_codes == ("holdout_commitment_rotated",)
    fresh = ledger.verify_evaluation_evidence(
        _SITE, {"holdout_commitment_digest": rotated.commitment_digest}
    )
    assert fresh.valid
    assert fresh.reason_codes == ("valid",)


def test_retired_commitments_cannot_be_reregistered() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    original = _split().export_evidence()
    ledger.register(_SITE, original)
    ledger.record_rotation(_SITE, HoldoutRotationReason.POLICY_CHANGE)

    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        ledger.register(_SITE, original)
    assert _code(excinfo) == "retired_commitment_reused"


def test_rotation_reason_must_be_a_controlled_code() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    ledger.register(_SITE, _split().export_evidence())

    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        ledger.record_rotation(_SITE, "because")  # type: ignore[arg-type]
    assert _code(excinfo) == "invalid_rotation_reason"
    assert ledger.require_active_commitment(_SITE) == _split().commitment_digest


def test_sites_are_independent_within_one_federation() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    ledger.register("site-a", _split().export_evidence())
    ledger.register("site-b", _split(site_secret=_OTHER_SECRET).export_evidence())

    assert ledger.require_active_commitment("site-a") != (
        ledger.require_active_commitment("site-b")
    )
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        ledger.register("Site A!", _split().export_evidence())
    assert _code(excinfo) == "invalid_site_ref"


def test_ledger_state_round_trips_without_losing_rotation_history() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    original = _split().export_evidence()
    ledger.register(_SITE, original)
    ledger.record_rotation(_SITE, HoldoutRotationReason.COHORT_REFRESH)
    rotated = _split(rotation_epoch=1).export_evidence()
    ledger.register(_SITE, rotated)

    restored = FederatedHoldoutLedger.from_dict(json.loads(ledger.to_json()))

    assert restored.to_json() == ledger.to_json()
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        restored.register(_SITE, original)
    assert _code(excinfo) == "retired_commitment_reused"
    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        restored.register(_SITE, _split(site_secret=_OTHER_SECRET).export_evidence())
    assert _code(excinfo) == "recommitment_without_rotation"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.update(federation_id="Bad Federation!"),
        lambda state: state.update(schema_version="other"),
        lambda state: state.update(extra=1),
        lambda state: state["sites"]["site-a"].update(active_commitment_digest="x"),
        lambda state: state["sites"]["site-a"]["rotations"][0].update(reason="why"),
        lambda state: state["sites"]["site-a"]["rotations"].append(
            dict(state["sites"]["site-a"]["rotations"][0])
        ),
    ],
)
def test_ledger_restore_fails_closed_on_tampered_state(mutate) -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    ledger.register(_SITE, _split().export_evidence())
    ledger.record_rotation(_SITE, HoldoutRotationReason.COHORT_REFRESH)
    ledger.register(_SITE, _split(rotation_epoch=1).export_evidence())
    state = json.loads(ledger.to_json())
    mutate(state)

    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        FederatedHoldoutLedger.from_dict(state)
    assert _code(excinfo) == "invalid_ledger_state"


# Acceptance: evaluation evidence citing a different commitment fails.


def test_evaluation_evidence_citing_a_different_commitment_fails() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    ledger.register(_SITE, _split().export_evidence())
    other = _split(site_secret=_OTHER_SECRET).commitment_digest

    result = ledger.verify_evaluation_evidence(
        _SITE, {"holdout_commitment_digest": other}
    )

    assert not result.valid
    assert result.reason_codes == ("evaluation_commitment_mismatch",)
    assert other not in json.dumps(result.to_dict())


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"holdout_commitment_digest": None},
        {"holdout_commitment_digest": "sha256:not-a-digest"},
        "sha256:" + "a" * 64,
        None,
    ],
)
def test_evaluation_evidence_without_a_valid_citation_fails(evidence) -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    ledger.register(_SITE, _split().export_evidence())

    result = ledger.verify_evaluation_evidence(_SITE, evidence)

    assert not result.valid
    assert result.reason_codes == ("evaluation_commitment_missing",)


def test_evaluation_evidence_for_an_uncommitted_site_fails() -> None:
    ledger = FederatedHoldoutLedger(_FEDERATION)
    digest = _split().commitment_digest

    result = ledger.verify_evaluation_evidence(
        "site-z", {"holdout_commitment_digest": digest}
    )

    assert not result.valid
    assert result.reason_codes == ("holdout_not_committed",)


# Acceptance: only the root digest and count bands leave the site.


def test_exported_evidence_contains_only_digest_and_count_bands() -> None:
    split = _split()
    evidence = split.export_evidence()
    document = evidence.to_dict()

    assert set(document) == {
        "commitment_digest",
        "holdout_patient_count_band",
        "schema_version",
        "train_patient_count_band",
    }
    assert document["schema_version"] == FEDERATED_HOLDOUT_EVIDENCE_SCHEMA_VERSION
    assert document["commitment_digest"] == split.commitment_digest
    assert document["holdout_patient_count_band"] in {
        band.value for band in HoldoutPatientCountBand
    }
    assert json.loads(evidence.to_json()) == document
    assert SiteHoldoutEvidence.from_dict(document) == evidence


def test_count_bands_are_coarse_and_suppress_small_partitions() -> None:
    small = _split(patient_keys=_patients(8), holdout_fraction=0.3)
    evidence = small.export_evidence()
    holdout = sum(
        small.partition_of(patient) is HoldoutSplitPartition.HOLDOUT
        for patient in _patients(8)
    )

    assert holdout < 5
    assert evidence.holdout_patient_count_band is HoldoutPatientCountBand.SUPPRESSED
    large = _split(patient_keys=_patients(400)).export_evidence()
    assert large.train_patient_count_band is HoldoutPatientCountBand.HUNDREDS


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(patient_keys=["SYNTH-PATIENT-0001"]),
        lambda doc: doc.update(commitment_digest="sha256:short"),
        lambda doc: doc.update(holdout_patient_count_band="exactly_17"),
        lambda doc: doc.update(schema_version="other"),
        lambda doc: doc.pop("train_patient_count_band"),
    ],
)
def test_evidence_parsing_fails_closed(mutate) -> None:
    document = _split().export_evidence().to_dict()
    mutate(document)

    with pytest.raises(FederatedHoldoutSplitError) as excinfo:
        SiteHoldoutEvidence.from_dict(document)
    assert _code(excinfo) == "invalid_evidence"
    ledger = FederatedHoldoutLedger(_FEDERATION)
    with pytest.raises(FederatedHoldoutSplitError):
        ledger.register(_SITE, document)


def test_secrets_patient_keys_and_note_ids_never_leave_the_site() -> None:
    split = _split()
    ledger = FederatedHoldoutLedger(_FEDERATION)
    evidence = split.export_evidence()
    ledger.register(_SITE, evidence)
    train, holdout = split.partition_examples(
        _notes(), patient_key_of=lambda note: note.patient_key
    )
    errors: list[str] = []
    for action in (
        lambda: split.admit_training_examples(
            (*train, *holdout), patient_key_of=lambda note: note.patient_key
        ),
        lambda: split.require_training_patient(holdout[0].patient_key),
        lambda: split.partition_of("SYNTH-PATIENT-NEW-9999"),
        lambda: ledger.register(
            _SITE, _split(site_secret=_OTHER_SECRET).export_evidence()
        ),
    ):
        with pytest.raises(FederatedHoldoutSplitError) as excinfo:
            action()
        assert excinfo.value.__cause__ is None
        assert excinfo.value.__context__ is None
        errors.extend((str(excinfo.value), repr(excinfo.value)))

    exported = "\n".join(
        (
            evidence.to_json(),
            repr(evidence),
            repr(split),
            str(split),
            ledger.to_json(),
            repr(ledger),
            json.dumps(
                ledger.verify_evaluation_evidence(
                    _SITE, {"holdout_commitment_digest": evidence.commitment_digest}
                ).to_dict()
            ),
            *errors,
        )
    )
    sentinels = (
        "SYNTH-PATIENT",
        "SYNTH-NOTE",
        "synthetic-site-secret",
        _SECRET.hex(),
    )
    assert not [sentinel for sentinel in sentinels if sentinel in exported]
    assert "SYNTH" not in repr(split)
    assert set(split.__slots__) >= {"_site_secret"}


def test_split_refuses_pickling_so_secrets_cannot_be_serialized() -> None:
    with pytest.raises(TypeError):
        pickle.dumps(_split())
