import json
from datetime import datetime, timedelta

import pytest

from openmed.core.date_shift import stable_offset_for
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult

HMAC_TEST_MATERIAL = bytes(range(1, 33))


def _prediction(text: str, *entities: EntityPrediction) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=list(entities),
        model_name="unit-test",
        timestamp=datetime.now().isoformat(),
    )


def _date_entity(
    text: str,
    surface: str,
    *,
    label: str = "DATE",
) -> EntityPrediction:
    start = text.index(surface)
    return EntityPrediction(
        text=surface,
        label=label,
        start=start,
        end=start + len(surface),
        confidence=0.99,
    )


def _deidentify_with_entities(
    text: str,
    *entities: EntityPrediction,
    **run_kwargs,
):
    def model_detector(model_text: str, **kwargs):
        assert model_text == text
        return _prediction(model_text, *entities)

    return (
        Pipeline(model_detector=model_detector, use_safety_sweep=False)
        .run(
            text,
            method="shift_dates",
            **run_kwargs,
        )
        .deidentification_result
    )


def _shift_mmddyyyy(value: str, offset: int) -> str:
    shifted = datetime.strptime(value, "%m/%d/%Y") + timedelta(days=offset)
    return shifted.strftime("%m/%d/%Y")


def test_stable_offset_for_is_repeatable_bounded_and_nonzero():
    first = stable_offset_for("patient-a", max_days=30, secret=HMAC_TEST_MATERIAL)
    second = stable_offset_for("patient-a", max_days=30, secret=HMAC_TEST_MATERIAL)

    assert first == second
    assert -30 <= first <= 30
    assert first != 0


def test_stable_offset_for_rejects_empty_patient_key_without_echoing_value():
    with pytest.raises(ValueError, match="patient_key must be non-empty"):
        stable_offset_for("", max_days=30, secret=HMAC_TEST_MATERIAL)


def test_stable_offset_for_rejects_empty_secret():
    with pytest.raises(ValueError, match="secret must be non-empty"):
        stable_offset_for("patient-a", max_days=30, secret="")


def test_patient_keyed_shift_dates_are_stable_across_documents():
    first_text = "Admit 01/10/2020 discharge 01/20/2020"
    second_text = "DOB 03/05/1980 follow-up 04/04/2020"
    subject_id = "patient-stable-430"
    offset = stable_offset_for(
        subject_id,
        max_days=30,
        secret=HMAC_TEST_MATERIAL,
    )

    first = _deidentify_with_entities(
        first_text,
        _date_entity(first_text, "01/10/2020"),
        _date_entity(first_text, "01/20/2020"),
        patient_key=subject_id,
        date_shift_max_days=30,
        date_shift_secret=HMAC_TEST_MATERIAL,
    )
    second = _deidentify_with_entities(
        second_text,
        _date_entity(second_text, "03/05/1980", label="DATE_OF_BIRTH"),
        _date_entity(second_text, "04/04/2020"),
        patient_key=subject_id,
        date_shift_max_days=30,
        date_shift_secret=HMAC_TEST_MATERIAL,
    )

    first_start = _shift_mmddyyyy("01/10/2020", offset)
    first_end = _shift_mmddyyyy("01/20/2020", offset)
    second_dob = _shift_mmddyyyy("03/05/1980", offset)
    second_followup = _shift_mmddyyyy("04/04/2020", offset)

    assert first.deidentified_text == f"Admit {first_start} discharge {first_end}"
    assert second.deidentified_text == f"DOB {second_dob} follow-up {second_followup}"

    shifted_start = datetime.strptime(first_start, "%m/%d/%Y")
    shifted_end = datetime.strptime(first_end, "%m/%d/%Y")
    assert shifted_end - shifted_start == timedelta(days=10)


def test_different_patient_keys_can_produce_different_offsets():
    first_offset = stable_offset_for(
        "patient-a",
        max_days=365,
        secret=HMAC_TEST_MATERIAL,
    )
    second_key = next(
        key
        for key in (f"patient-b-{index}" for index in range(100))
        if stable_offset_for(key, max_days=365, secret=HMAC_TEST_MATERIAL)
        != first_offset
    )

    assert (
        stable_offset_for(second_key, max_days=365, secret=HMAC_TEST_MATERIAL)
        != first_offset
    )


def test_patient_key_is_not_returned_in_output_mapping_or_logs(caplog):
    subject_id = "MRN-raw-patient-key-430"
    text = "Visit 01/10/2020"

    result = _deidentify_with_entities(
        text,
        _date_entity(text, "01/10/2020"),
        patient_key=subject_id,
        date_shift_max_days=30,
        date_shift_secret=HMAC_TEST_MATERIAL,
        keep_mapping=True,
    )

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert subject_id not in result.deidentified_text
    assert subject_id not in json.dumps(result.mapping, sort_keys=True)
    assert subject_id not in serialized
    assert subject_id not in caplog.text


def test_omitting_patient_key_preserves_explicit_offset_behavior(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("stable offsets require patient_key")

    monkeypatch.setattr("openmed.core.pii.stable_offset_for", fail_if_called)
    text = "Visit 01/10/2020"

    result = _deidentify_with_entities(
        text,
        _date_entity(text, "01/10/2020"),
        date_shift_days=30,
    )

    assert result.deidentified_text == "Visit 02/09/2020"


def test_patient_key_requires_shift_dates_method():
    subject_id = "patient-a"
    with pytest.raises(ValueError, match="patient_key requires"):
        Pipeline(use_safety_sweep=False).run(
            "Visit 01/10/2020",
            method="mask",
            patient_key=subject_id,
            date_shift_secret=HMAC_TEST_MATERIAL,
        )


def test_patient_key_requires_date_shift_secret():
    subject_id = "patient-a"

    with pytest.raises(ValueError, match="patient_key requires date_shift_secret"):
        Pipeline(use_safety_sweep=False).run(
            "Visit 01/10/2020",
            method="shift_dates",
            patient_key=subject_id,
        )
