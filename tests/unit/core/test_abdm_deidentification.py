"""India ABDM/ABHA de-identification acceptance coverage."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from faker import Faker

from openmed.core import pii
from openmed.core.anonymizer.providers.clinical_ids import (
    register_clinical_providers,
    validate_abha_number,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.audit import AuditReport, AuditSpan
from openmed.core.custom_recognizer import ABDMRecognizer, abdm_mode_enabled
from openmed.core.labels import (
    DIRECT_IDENTIFIER,
    ID_NUM,
    normalize_label,
    policy_label_for,
)
from openmed.core.pii_i18n import validate_aadhaar
from openmed.core.policy import load_policy
from openmed.processing.outputs import PredictionResult

_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "abdm_deidentification.json"
_EXPECTED_LABELS = {
    "ABHA_NUMBER",
    "ABHA_ADDRESS",
    "AADHAAR",
    "PAN",
    "ABDM_HPR_ID",
    "ABDM_HFR_ID",
}


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _empty_prediction(
    text: str,
    *args: object,
    **kwargs: object,
) -> PredictionResult:
    model_name = kwargs.get("model_name") or (args[0] if args else "stub")
    return PredictionResult(
        text=text,
        entities=[],
        model_name=str(model_name),
        timestamp=datetime.now().isoformat(),
    )


def _disable_model_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pii, "extract_pii", _empty_prediction)


def test_abdm_recognizer_detects_every_identifier_as_direct_identifier() -> None:
    fixture = _fixture()
    text = str(fixture["note"])

    entities = ABDMRecognizer().detect_entities(text)

    assert {entity.label for entity in entities} == _EXPECTED_LABELS
    assert all(normalize_label(entity.label) == ID_NUM for entity in entities)
    assert all(
        policy_label_for(entity.label) == DIRECT_IDENTIFIER for entity in entities
    )
    for entity in entities:
        assert entity.metadata["custom_recognizer"]["text_hash"].startswith(
            "hmac-sha256:"
        )
        assert entity.text not in repr(entity.metadata)


def test_abdm_recognizer_accepts_structural_abha_without_checksum_assumption() -> None:
    entities = ABDMRecognizer().detect_entities("ABHA 11328941304001")

    assert [entity.label for entity in entities] == ["ABHA_NUMBER"]


def test_abdm_recognizer_rejects_malformed_numbers() -> None:
    text = "ABHA 00000000000000 and Aadhaar 655227804686"

    assert ABDMRecognizer().detect_entities(text) == []


@pytest.mark.parametrize(
    "alias",
    ("in", "hi", "te", "en_IN", "hi_IN"),
)
def test_abdm_registry_surrogates_round_trip_for_all_india_aliases(alias: str) -> None:
    faker = Faker("en_IN")
    register_clinical_providers(faker)
    faker.seed_instance(672)

    for id_type in (
        "abha_number",
        "abha_address",
        "aadhaar",
        "pan",
        "abdm_hpr_id",
        "abdm_hfr_id",
    ):
        spec = get_national_id(alias, id_type)
        assert spec is not None
        surrogate = getattr(faker, spec.faker_method)()
        assert spec.validate(surrogate), f"invalid {alias}/{id_type}: {surrogate}"


def test_generated_abha_and_aadhaar_pass_registered_validators() -> None:
    faker = Faker("en_IN")
    register_clinical_providers(faker)
    faker.seed_instance(672)

    for _ in range(40):
        assert validate_abha_number(faker.abha_number())
        assert validate_aadhaar(faker.aadhaar())


def test_abdm_deidentify_replaces_all_fixture_identifiers_without_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_model_detection(monkeypatch)
    fixture = _fixture()
    text = str(fixture["note"])
    identifiers = dict(fixture["identifiers"])

    result = pii.deidentify(
        text,
        method="replace",
        lang="en",
        locale="en_US",
        abdm=True,
        use_safety_sweep=False,
        consistent=True,
        seed=672,
    )

    assert {entity.label for entity in result.pii_entities} == _EXPECTED_LABELS
    assert all(entity.action == "replace" for entity in result.pii_entities)
    assert all(value not in result.deidentified_text for value in identifiers.values())

    by_label = {entity.label: entity.surrogate for entity in result.pii_entities}
    assert validate_abha_number(by_label["ABHA_NUMBER"])
    assert validate_aadhaar(by_label["AADHAAR"])


def test_india_locale_auto_enables_abdm_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_model_detection(monkeypatch)
    fixture = _fixture()

    result = pii.deidentify(
        str(fixture["note"]),
        method="mask",
        lang="en",
        locale="en_IN",
        use_safety_sweep=False,
    )

    assert {entity.label for entity in result.pii_entities} == _EXPECTED_LABELS


@pytest.mark.parametrize(
    "kwargs",
    (
        {"lang": "hi", "locale": "en_US"},
        {"lang": "te", "locale": "en_US"},
        {"lang": "en", "locale": "hi_IN"},
        {"policy": "india_dpdp_act", "lang": "en", "locale": "en_US"},
    ),
)
def test_abdm_mode_resolver_covers_policy_language_and_locale(
    kwargs: dict[str, str],
) -> None:
    assert abdm_mode_enabled(None, **kwargs)


def test_explicit_abdm_false_overrides_india_auto_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_model_detection(monkeypatch)
    text = str(_fixture()["note"])

    result = pii.deidentify(
        text,
        method="mask",
        lang="hi",
        locale="en_IN",
        abdm=False,
        use_safety_sweep=False,
    )

    assert result.deidentified_text == text
    assert result.pii_entities == []


def test_abdm_bundle_preserves_custom_allow_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_model_detection(monkeypatch)
    abha = "11328941304001"
    pan = "GUYJE2143V"
    text = f"ABHA {abha}; PAN {pan}."

    result = pii.deidentify(
        text,
        method="mask",
        abdm=True,
        custom_recognizer={"allow_terms": [abha]},
        use_safety_sweep=False,
    )

    assert abha in result.deidentified_text
    assert pan not in result.deidentified_text
    assert [entity.label for entity in result.pii_entities] == ["PAN"]


def test_abdm_default_does_not_affect_non_india_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_model_detection(monkeypatch)
    text = str(_fixture()["note"])

    result = pii.deidentify(
        text,
        method="mask",
        lang="en",
        locale="en_US",
        use_safety_sweep=False,
    )

    assert result.deidentified_text == text
    assert result.pii_entities == []


def test_abdm_audit_outputs_never_contain_raw_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_model_detection(monkeypatch)
    fixture = _fixture()
    identifiers = dict(fixture["identifiers"])

    report = pii.deidentify(
        str(fixture["note"]),
        method="replace",
        locale="en_IN",
        abdm=True,
        use_safety_sweep=False,
        consistent=True,
        seed=672,
        audit=True,
    )

    assert isinstance(report, AuditReport)
    serialized = report.to_json() + report.export_review_bundle_json()
    assert all(value not in serialized for value in identifiers.values())
    assert all(span.text_hash.startswith("sha256:") for span in report.spans)


def test_audit_sink_hashes_injected_raw_abdm_evidence() -> None:
    identifiers = dict(_fixture()["identifiers"])
    abha = identifiers["ABHA_NUMBER"]
    pan = identifiers["PAN"]
    span = AuditSpan(
        start=0,
        end=len(abha),
        label="ABHA_NUMBER",
        canonical_label="ID_NUM",
        sources=["custom:deny"],
        confidence=1.0,
        threshold=1.0,
        action="replace",
        surrogate="90000000000001",
        text_hash="sha256:" + "0" * 64,
        evidence={"value": abha, "note": f"PAN {pan}"},
    )

    serialized = json.dumps(span.to_dict(), sort_keys=True)
    assert abha not in serialized
    assert pan not in serialized
    assert "sha256:" in serialized
    assert "redacted:" in serialized


def test_india_dpdp_policy_contract_replaces_every_abdm_identifier_label() -> None:
    profile = load_policy("india_dpdp_act")

    assert abdm_mode_enabled(None, policy=profile)
    assert all(profile.action_for(label) == "replace" for label in _EXPECTED_LABELS)
