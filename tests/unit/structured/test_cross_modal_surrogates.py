"""Cross-modal surrogate consistency tests using synthetic offline data."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from openmed.core.pii import PIIEntity, deidentify
from openmed.core.surrogate_vault import (
    HMAC_SCHEME,
    SubjectResolutionError,
    SurrogateVault,
)
from openmed.interop._pii import resolve_subject_surrogate
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.risk.reid import cross_modal_linkage_risk_report
from openmed.structured.consistency import (
    assert_cross_modal_consistency,
    deidentify_subject_column,
)
from openmed.structured.relational import (
    ColumnRef,
    KeySpace,
    RelationalSchema,
    deidentify_linked_tables,
)

VAULT_SECRET = "synthetic-cross-modal-vault-secret"
SYNTHETIC_NAME = "Avery Synthetic"
SYNTHETIC_MRN = "SYNTH-MRN-000001"


def _prediction_result(
    text: str,
    entities: Sequence[tuple[str, str]],
) -> PredictionResult:
    predictions: list[EntityPrediction] = []
    for surface, label in entities:
        start = text.index(surface)
        predictions.append(
            EntityPrediction(
                text=surface,
                label=label,
                start=start,
                end=start + len(surface),
                confidence=0.99,
            )
        )
    return PredictionResult(
        text=text,
        entities=predictions,
        model_name="synthetic-pii",
        timestamp="now",
    )


def _note_identifiers() -> list[dict[str, str]]:
    return [
        {"text": SYNTHETIC_NAME, "label": "NAME"},
        {"text": SYNTHETIC_MRN, "label": "MRN"},
    ]


def test_note_and_table_share_one_subject_surrogate_end_to_end(monkeypatch):
    note = f"Patient {SYNTHETIC_NAME} has MRN {SYNTHETIC_MRN}."

    def fake_extract(text: str, *args, **kwargs) -> PredictionResult:
        del args, kwargs
        return _prediction_result(
            text,
            [(SYNTHETIC_NAME, "NAME"), (SYNTHETIC_MRN, "MRN")],
        )

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract)
    vault = SurrogateVault.in_memory(VAULT_SECRET)
    table = deidentify_subject_column(
        [{"mrn": SYNTHETIC_MRN, "cohort": "synthetic"}],
        subject_column="mrn",
        vault=vault,
        note_identifiers_by_subject={SYNTHETIC_MRN: _note_identifiers()},
    )
    deidentified_note = deidentify(
        note,
        method="replace",
        surrogate_vault=vault,
        use_safety_sweep=False,
    )

    note_surrogates = {entity.surrogate for entity in deidentified_note.pii_entities}
    table_surrogate = table.records[0]["mrn"]
    assert note_surrogates == {table_surrogate}
    assert_cross_modal_consistency(
        deidentified_note.pii_entities[0].surrogate or "",
        table_surrogate,
    )
    assert SYNTHETIC_NAME not in deidentified_note.deidentified_text
    assert SYNTHETIC_MRN not in deidentified_note.deidentified_text


def test_large_synthetic_cohort_has_no_subject_surrogate_collisions():
    vault = SurrogateVault.in_memory(VAULT_SECRET)

    surrogates = {
        vault.resolve_subject(f"SYNTH-SUBJECT-{index:06d}") for index in range(4096)
    }

    assert len(surrogates) == 4096
    assert all(value.startswith("PATIENT_") for value in surrogates)


def test_linked_table_engine_reuses_prebound_note_subject_surrogate():
    vault = SurrogateVault.in_memory(VAULT_SECRET)
    expected = resolve_subject_surrogate(
        [
            PIIEntity(
                text=SYNTHETIC_NAME,
                label="NAME",
                start=0,
                end=len(SYNTHETIC_NAME),
                confidence=0.99,
            )
        ],
        structured_identifier=SYNTHETIC_MRN,
        vault=vault,
    )
    schema = RelationalSchema(
        key_spaces=(KeySpace("PATIENT_ID", (ColumnRef("patients", "mrn"),)),),
        subject_key_space="PATIENT_ID",
    )

    released = deidentify_linked_tables(
        {"patients": [{"mrn": SYNTHETIC_MRN}]},
        schema,
        vault=vault,
        date_shift_secret="synthetic-date-secret",
    )

    assert released.tables["patients"][0]["mrn"] == expected


def test_vault_manifest_logs_and_audit_evidence_contain_no_raw_identifiers(
    tmp_path,
    caplog,
):
    vault_path = tmp_path / "cross-modal-vault.json"
    vault = SurrogateVault.from_file(
        vault_path,
        hmac_secret=VAULT_SECRET,
    )
    table = deidentify_subject_column(
        [{"mrn": SYNTHETIC_MRN}],
        subject_column="mrn",
        vault=vault,
        note_identifiers_by_subject={SYNTHETIC_MRN: _note_identifiers()},
    )
    consistency = assert_cross_modal_consistency(
        table.records[0]["mrn"],
        table.records[0]["mrn"],
    )

    audit_outputs = json.dumps(
        {
            "manifest": table.manifest.to_dict(),
            "consistency": consistency.to_dict(),
        },
        sort_keys=True,
    )
    persisted = vault_path.read_text(encoding="utf-8")
    for raw_identifier in (SYNTHETIC_NAME, SYNTHETIC_MRN):
        assert raw_identifier not in persisted
        assert raw_identifier not in audit_outputs
        assert raw_identifier not in caplog.text
    assert table.manifest.source_hashes
    assert all(
        value.startswith(f"{HMAC_SCHEME}:") for value in table.manifest.source_hashes
    )


def test_subject_assignment_is_deterministic_after_vault_reload(tmp_path):
    vault_path = tmp_path / "cross-modal-vault.json"
    first_vault = SurrogateVault.from_file(
        vault_path,
        hmac_secret=VAULT_SECRET,
    )
    first = deidentify_subject_column(
        [{"mrn": SYNTHETIC_MRN}],
        subject_column="mrn",
        vault=first_vault,
        note_identifiers_by_subject={SYNTHETIC_MRN: _note_identifiers()},
    )

    reloaded = SurrogateVault.from_file(
        vault_path,
        hmac_secret=VAULT_SECRET,
    )
    second = deidentify_subject_column(
        [{"mrn": SYNTHETIC_MRN}],
        subject_column="mrn",
        vault=reloaded,
        note_identifiers_by_subject={
            SYNTHETIC_MRN: list(reversed(_note_identifiers()))
        },
    )

    assert second.records == first.records
    assert second.manifest.source_hashes == first.manifest.source_hashes


def test_conflicting_subject_alias_is_rejected_without_raw_value_in_error():
    vault = SurrogateVault.in_memory(VAULT_SECRET)
    vault.resolve_subject(
        "SYNTH-SUBJECT-000001",
        aliases=((SYNTHETIC_NAME, "NAME"),),
    )

    with pytest.raises(SubjectResolutionError) as exc_info:
        vault.resolve_subject(
            "SYNTH-SUBJECT-000002",
            aliases=((SYNTHETIC_NAME, "NAME"),),
        )

    assert SYNTHETIC_NAME not in str(exc_info.value)


def test_cross_modal_linkage_leakage_gate_passes_for_pseudonymous_outputs():
    vault = SurrogateVault.in_memory(VAULT_SECRET)
    first = vault.resolve_subject(SYNTHETIC_MRN)
    second = vault.resolve_subject("SYNTH-MRN-000002")

    report = cross_modal_linkage_risk_report(
        f"Patient {first} completed follow-up.",
        [{"mrn": first}, {"mrn": second}],
        source_identifier_groups=(
            (SYNTHETIC_NAME, SYNTHETIC_MRN),
            ("Blake Synthetic", "SYNTH-MRN-000002"),
        ),
    )

    assert report["combined_attack_rate"] == 0.0
    assert report["single_modality_risk_bound"] == 0.0
    assert report["passed"] is True


def test_cross_modal_linkage_gate_rejects_complementary_raw_leaks():
    report = cross_modal_linkage_risk_report(
        "Avery Synthetic remains in this released note.",
        [{"mrn": "SYNTH-MRN-000002"}],
        source_identifier_groups=(
            (SYNTHETIC_NAME, SYNTHETIC_MRN),
            ("Blake Synthetic", "SYNTH-MRN-000002"),
        ),
    )

    assert report["note_attack_rate"] == 0.5
    assert report["table_attack_rate"] == 0.5
    assert report["combined_attack_rate"] == 1.0
    assert report["passed"] is False
