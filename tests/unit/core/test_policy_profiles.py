from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from openmed.core.arbitration import MODE_HIGH_RECALL_UNION
from openmed.core.cascade import R3_ACCURATE, CascadeRouter
from openmed.core.labels import (
    CANONICAL_LABELS,
    CLINICAL_CONCEPT,
    DIRECT_IDENTIFIER,
    LABEL_TO_POPIA,
    NDPA_SENSITIVE_CLASS_LABELS,
    NDPA_SENSITIVE_DATA_CLASSES,
    POPIA_IDENTIFIER_CLASSES,
    QUASI_IDENTIFIER,
    SENSITIVE_ATTRIBUTE,
    id_subtype_for,
    ndpa_classes_for,
    normalize_label,
    policy_label_for,
)
from openmed.core.pii import deidentify
from openmed.core.pipeline import Pipeline
from openmed.core.policy import (
    CANONICAL_POLICY_NAMES,
    CURRENT_POLICY_SCHEMA_VERSION,
    PolicyName,
    canonical_policy_name,
    compile_policy,
    lint_policy,
    list_policies,
    load_policy,
)
from openmed.core.policy_lint import lint_policy as lint_policy_report
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash
from openmed.processing.outputs import EntityPrediction, PredictionResult

_KE_DPA_FIXTURE = Path(__file__).with_name("fixtures") / "ke_dpa_synthetic.json"
_KE_DPA_SENSITIVE_CLASSES = {
    "Race",
    "Health status",
    "Ethnic social origin",
    "Conscience",
    "Belief",
    "Genetic data",
    "Biometric data",
    "Property details",
    "Marital status",
    "Family details",
    "Sex",
    "Sexual orientation",
}


def _ke_dpa_fixture() -> dict[str, object]:
    return json.loads(_KE_DPA_FIXTURE.read_text(encoding="utf-8"))


def _prediction(
    text: str,
    entities: list[EntityPrediction] | None = None,
) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=entities or [],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )


def _entity(text: str, surface: str, label: str, score: float) -> EntityPrediction:
    start = text.index(surface)
    return EntityPrediction(
        text=surface,
        label=label,
        start=start,
        end=start + len(surface),
        confidence=score,
    )


def _patch_extract(monkeypatch, entity: EntityPrediction) -> None:
    from openmed.core import pii

    def fake_extract(text: str, *args: object, **kwargs: object) -> PredictionResult:
        return _prediction(text, [entity])

    monkeypatch.setattr(pii, "extract_pii", fake_extract)


def _patch_extract_many(
    monkeypatch,
    entities: list[EntityPrediction],
) -> None:
    from openmed.core import pii

    def fake_extract(text: str, *args: object, **kwargs: object) -> PredictionResult:
        return _prediction(text, entities)

    monkeypatch.setattr(pii, "extract_pii", fake_extract)


def _span(label: str = "PERSON", score: float = 0.95) -> OpenMedSpan:
    return OpenMedSpan(
        doc_id="doc-1",
        start=0,
        end=8,
        text_hash=hmac_text_hash(f"{label}:{score}", "test-secret"),
        entity_type=label,
        canonical_label=label,
        score=score,
        detector="model:tiny",
    )


def _popia_checklist_rows() -> dict[str, tuple[str, ...]]:
    checklist_path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "compliance"
        / "za-popia-identifier-checklist.md"
    )
    checklist = checklist_path.read_text(encoding="utf-8")
    table = checklist.split("<!-- popia-identifier-table:start -->", 1)[1].split(
        "<!-- popia-identifier-table:end -->",
        1,
    )[0]
    rows: dict[str, tuple[str, ...]] = {}
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        class_match = re.fullmatch(r"`([^`]+)`", cells[0])
        assert class_match is not None
        rows[class_match.group(1)] = tuple(re.findall(r"`([A-Z][A-Z0-9_]*)`", cells[2]))
    return rows


def _ng_ndpa_checklist_rows() -> dict[str, tuple[str, ...]]:
    checklist_path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "compliance"
        / "ng-ndpa-identifier-checklist.md"
    )
    checklist = checklist_path.read_text(encoding="utf-8")
    table = checklist.split("<!-- ndpa-sensitive-data-table:start -->", 1)[1].split(
        "<!-- ndpa-sensitive-data-table:end -->",
        1,
    )[0]
    rows: dict[str, tuple[str, ...]] = {}
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        class_match = re.fullmatch(r"`([^`]+)`", cells[0])
        assert class_match is not None
        rows[class_match.group(1)] = tuple(re.findall(r"`([A-Z][A-Z0-9_]*)`", cells[2]))
    return rows


def _ng_ndpa_fixture_cases() -> list[dict[str, object]]:
    fixture_path = Path(__file__).parent / "fixtures" / "ng_ndpa_synthetic.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert payload["synthetic"] is True
    cases = payload["cases"]
    assert isinstance(cases, list)
    assert all(isinstance(case, dict) for case in cases)
    return cases


def test_all_policy_literals_load_and_gdpr_alias_resolves():
    for name in CANONICAL_POLICY_NAMES:
        profile = load_policy(name)

        assert profile.name == name
        assert set(profile.actions) == set(CANONICAL_LABELS)
        assert lint_policy(name) == ()

    assert load_policy("gdpr").name == "gdpr_pseudonymization"


def test_all_policy_profiles_run_through_deidentify(monkeypatch):
    text = "Patient John Doe"
    _patch_extract(monkeypatch, _entity(text, "John Doe", "NAME", 0.95))

    for name in CANONICAL_POLICY_NAMES:
        result = deidentify(text, policy=name, use_safety_sweep=False)

        assert result.metadata["policy"]["name"] == name
        assert result.pii_entities


def test_gdpr_art9_health_profile_and_alias_load():
    profile = load_policy("gdpr_art9_health")
    gdpr = load_policy("gdpr_pseudonymization")

    assert profile.name == "gdpr_art9_health"
    assert load_policy("gdpr_health").name == "gdpr_art9_health"
    assert "gdpr_art9_health" in list_policies()
    assert profile.safety_sweep_mandatory is True
    assert "R3" in profile.forced_cascade_tiers
    assert profile.action_for("LOCATION") == "mask"
    assert profile.action_for("CONDITION") == "mask"
    assert gdpr.action_for("CONDITION") == "keep"
    assert lint_policy("gdpr_art9_health") == ()


def test_gdpr_art9_health_masks_clinical_fixture_that_gdpr_keeps(monkeypatch):
    text = "Patient has diabetes"
    _patch_extract(monkeypatch, _entity(text, "diabetes", "CONDITION", 0.99))

    gdpr = deidentify(
        text,
        policy="gdpr_pseudonymization",
        use_safety_sweep=False,
    )
    article9 = deidentify(
        text,
        policy="gdpr_art9_health",
        use_safety_sweep=False,
    )

    assert gdpr.deidentified_text == text
    assert article9.deidentified_text == "Patient has [CONDITION]"
    assert article9.pii_entities[0].action == "mask"
    assert article9.pii_entities[0].metadata["policy_action"]["action"] == "mask"


def test_canada_pipeda_profile_and_alias_load():
    profile = load_policy("canada_pipeda")

    assert profile.name == "canada_pipeda"
    assert load_policy("pipeda").name == "canada_pipeda"
    assert "canada_pipeda" in list_policies()
    assert profile.action_for("ID_NUM") == "mask"
    assert profile.action_for("SSN") == "mask"
    assert profile.action_for("PERSON") == "replace"
    assert profile.keep_mapping is True
    assert profile.reversible_id is True


def test_australia_privacy_act_profile_alias_and_lint_load():
    profile = load_policy("australia_privacy_act")

    assert profile.name == "australia_privacy_act"
    assert load_policy("au_privacy").name == "australia_privacy_act"
    assert "australia_privacy_act" in list_policies()
    assert profile.action_for("ID_NUM") == "mask"
    assert profile.action_for("SSN") == "mask"
    assert profile.action_for("PERSON") == "replace"
    assert profile.action_for("LOCATION") == "replace"
    assert profile.keep_mapping is False
    assert profile.reversible_id is False
    assert lint_policy("australia_privacy_act") == ()


_NORTH_AFRICA_SENSITIVE_CLASS_LABELS = {
    "eg_pdpl": {
        "identity": "ID_NUM",
        "person_name": "PERSON",
        "health": "CONDITION",
        "genetic": "GENE_SYMBOL",
        "biometric_or_other_sensitive": "OTHER",
        "financial": "ACCOUNT_NUMBER",
    },
    "ma_law_09_08": {
        "identity": "ID_NUM",
        "person_name": "PERSON",
        "health": "CONDITION",
        "genetic": "GENE_SYMBOL",
        "physical_or_psychological_identity": "OTHER",
        "financial_identity": "ACCOUNT_NUMBER",
    },
}


@pytest.mark.parametrize(
    ("policy_name", "sensitive_classes"),
    _NORTH_AFRICA_SENSITIVE_CLASS_LABELS.items(),
)
def test_north_africa_profiles_mask_statutory_sensitive_classes(
    policy_name,
    sensitive_classes,
):
    profile = load_policy(policy_name)

    assert profile.default_action == "mask"
    assert profile.safety_sweep_mandatory is True
    assert profile.keep_mapping is False
    assert profile.reversible_id is False
    assert set(profile.actions) == set(CANONICAL_LABELS)
    assert set(sensitive_classes.values()) <= set(CANONICAL_LABELS)
    assert all(
        profile.action_for(label) != "keep" for label in sensitive_classes.values()
    )
    assert lint_policy(policy_name) == ()

    compiled = compile_policy(policy_name)
    assert compiled.proof.verified is True
    assert compiled.proof.coverage_percent == 100.0


@pytest.mark.parametrize(
    ("policy_name", "locale", "text", "identifiers"),
    [
        (
            "eg_pdpl",
            "ar_EG",
            "اسم المريض: أحمد علي\n"
            "اسم المريض: Ahmed Hassan\n"
            "العنوان: ١٢ شارع النيل، القاهرة\n"
            "الرقم القومي: 29801011234567\n"
            "الهاتف: +20 10 1234 5678\n"
            "التاريخ الميلادي: 15/03/2026 م\n"
            "التاريخ الهجري: 15/09/1447 هـ",
            (
                "أحمد علي",
                "Ahmed Hassan",
                "١٢ شارع النيل، القاهرة",
                "29801011234567",
                "+20 10 1234 5678",
                "15/03/2026",
                "15/09/1447",
            ),
        ),
        (
            "ma_law_09_08",
            "ar_MA",
            "اسم المريض: فاطمة الزهراء\n"
            "اسم المريض: Fatima Zahra\n"
            "العنوان: ١٥ زنقة الأطلس، الدار البيضاء\n"
            "رقم البطاقة الوطنية: AB123456\n"
            "الهاتف: +212 6 12 34 56 78\n"
            "التاريخ الميلادي: 15/03/2026 م\n"
            "التاريخ الهجري: 15/09/1447 هـ",
            (
                "فاطمة الزهراء",
                "Fatima Zahra",
                "١٥ زنقة الأطلس، الدار البيضاء",
                "AB123456",
                "+212 6 12 34 56 78",
                "15/03/2026",
                "15/09/1447",
            ),
        ),
    ],
)
def test_north_africa_profiles_leave_zero_synthetic_identifier_leakage(
    monkeypatch,
    policy_name,
    locale,
    text,
    identifiers,
):
    _patch_extract_many(monkeypatch, [])

    result = deidentify(
        text,
        lang="ar",
        locale=locale,
        policy=policy_name,
        use_safety_sweep=False,
    )

    assert all(identifier not in result.deidentified_text for identifier in identifiers)
    assert result.mapping is None
    assert result.metadata["policy"]["name"] == policy_name
    assert result.metadata["safety_sweep"]["source"] == "safety_sweep"
    assert result.pii_entities
    assert all(entity.action == "mask" for entity in result.pii_entities)
    swept_names = {
        entity.text
        for entity in result.pii_entities
        if entity.canonical_label == "PERSON"
        and (entity.metadata or {}).get("source") == "safety_sweep"
    }
    assert all(
        any(name in swept_name for swept_name in swept_names)
        for name in identifiers[:2]
    )


def test_china_pipl_profile_covers_sensitive_personal_information():
    profile = load_policy("china_pipl")

    assert profile.name == "china_pipl"
    assert "china_pipl" in list_policies()
    assert set(profile.actions) == set(CANONICAL_LABELS)
    assert profile.policy_label_actions == {
        DIRECT_IDENTIFIER: "replace",
        QUASI_IDENTIFIER: "mask",
        SENSITIVE_ATTRIBUTE: "mask",
        CLINICAL_CONCEPT: "mask",
    }
    assert all(
        profile.action_for(label) == "replace"
        for label in CANONICAL_LABELS
        if policy_label_for(label) == DIRECT_IDENTIFIER
    )
    assert profile.action_for("ID_NUM") == "replace"
    assert profile.action_for("PHONE") == "replace"
    assert profile.action_for("CREDIT_CARD") == "replace"
    assert profile.action_for("PERSON") == "replace"
    assert profile.action_for("CONDITION") == "mask"
    assert profile.keep_mapping is True
    assert profile.reversible_id is True
    assert "de-identification does not remove" in profile.metadata["scope_note"]
    assert profile.metadata["legal_basis"] == {
        "sensitive_personal_information": "PIPL Article 28",
        "anonymization_and_deidentification": "PIPL Articles 4 and 73",
        "official_source": (
            "https://www.miit.gov.cn/jgsj/zfs/fl/art/2022/"
            "art_515a4b20c12f430eab54bb4f56d89f56.html"
        ),
    }
    assert profile.metadata["legal_disclaimer"] == (
        "This technical profile is not legal advice."
    )
    assert lint_policy("china_pipl") == ()


def test_india_dpdp_act_profile_is_complete_and_assist_only():
    profile = load_policy("india_dpdp_act")

    assert PolicyName.INDIA_DPDP_ACT.value == "india_dpdp_act"
    assert profile.name == PolicyName.INDIA_DPDP_ACT.value
    assert profile.schema_version == CURRENT_POLICY_SCHEMA_VERSION == 1
    assert profile.safety_sweep_mandatory is True
    assert profile.default_action == "replace"
    assert set(profile.actions) == set(CANONICAL_LABELS)
    assert all(profile.action_for(label) for label in CANONICAL_LABELS)
    assert all(
        profile.action_for(label) == "replace"
        for label in CANONICAL_LABELS
        if policy_label_for(label) == DIRECT_IDENTIFIER
    )
    assert all(
        profile.action_for(label) == "mask"
        for label in CANONICAL_LABELS
        if policy_label_for(label) == QUASI_IDENTIFIER
    )
    assert profile.action_for("ID_NUM") == "replace"
    assert profile.action_for("SSN") == "replace"
    assert profile.action_for("PERSON") == "replace"
    assert profile.action_for("PHONE") == "replace"
    assert profile.action_for("EMAIL") == "replace"
    assert profile.action_for("STREET_ADDRESS") == "replace"
    assert profile.action_for("AGE") == "mask"
    assert profile.action_for("DATE") == "mask"
    assert profile.action_for("ZIPCODE") == "mask"
    assert profile.action_for("GENDER") == "mask"
    assert "india_dpdp_act" in list_policies()
    assert lint_policy("india_dpdp_act") == ()

    provenance = profile.metadata["provenance"]
    disclaimer = profile.metadata["disclaimer"]
    assert "Digital Personal Data Protection Act, 2023" in provenance["basis"]
    assert "Digital Personal Data Protection Rules, 2025" in provenance["basis"]
    assert set(provenance["official_sources"]) == {"act", "rules"}
    assert all(
        source.startswith("https://www.meity.gov.in/")
        for source in provenance["official_sources"].values()
    )
    assert disclaimer["assist_only"] is True
    assert disclaimer["legal_advice"] is False
    assert disclaimer["autonomous_determination"] is False


@pytest.mark.parametrize(
    "source_label",
    [
        "aadhaar",
        "Aadhaar Number",
        "ABHA",
        "ABHA Number",
        "PAN",
        "Permanent Account Number",
    ],
)
def test_indian_national_identifier_aliases_normalize_to_id_num(
    source_label: str,
):
    assert normalize_label(source_label) == "ID_NUM"
    assert id_subtype_for(source_label) == "national_id"


def test_za_popia_profile_loads_with_special_information_defaults():
    profile = load_policy("za_popia")

    assert profile.name == "za_popia"
    assert "za_popia" in list_policies()
    assert profile.default_action == "replace"
    assert profile.policy_label_actions == {
        "DIRECT_IDENTIFIER": "replace",
        "QUASI_IDENTIFIER": "replace",
        "SENSITIVE_ATTRIBUTE": "mask",
        "CLINICAL_CONCEPT": "mask",
    }
    assert profile.safety_sweep_mandatory is True
    assert profile.keep_mapping is False
    assert profile.reversible_id is False
    assert profile.strict_no_leak is True
    assert set(profile.actions) == set(CANONICAL_LABELS)

    special_category_actions = {
        label: profile.action_for(label)
        for label in (
            "ID_NUM",
            "SSN",
            "GENDER",
            "EYE_COLOR",
            "HEIGHT",
            "CONDITION",
            "GENE_SYMBOL",
            "OTHER",
        )
    }
    assert special_category_actions == {
        "ID_NUM": "mask",
        "SSN": "mask",
        "GENDER": "mask",
        "EYE_COLOR": "mask",
        "HEIGHT": "mask",
        "CONDITION": "mask",
        "GENE_SYMBOL": "mask",
        "OTHER": "mask",
    }
    assert lint_policy("za_popia") == ()
    lint_report = lint_policy_report("za_popia")
    assert lint_report["valid"] is True
    assert lint_report["error_count"] == 0
    assert lint_report["warning_count"] == 0


def test_za_popia_checklist_classes_have_non_keep_canonical_coverage():
    profile = load_policy("za_popia")
    checklist_rows = _popia_checklist_rows()

    assert set(checklist_rows) == set(POPIA_IDENTIFIER_CLASSES)
    assert set(LABEL_TO_POPIA) == set(CANONICAL_LABELS)
    assert {label for labels in checklist_rows.values() for label in labels} == set(
        CANONICAL_LABELS
    )

    for popia_class, labels in checklist_rows.items():
        assert labels
        assert all(LABEL_TO_POPIA[label] == popia_class for label in labels)
        assert all(profile.action_for(label) != "keep" for label in labels)


@pytest.mark.parametrize(
    ("text", "detections"),
    [
        (
            "Clinic record for ID 8001015009087.",
            [("8001015009087", "ID_NUM")],
        ),
        (
            "Call the patient on +27 82 123 4567.",
            [("+27 82 123 4567", "PHONE")],
        ),
        (
            "Home address: 12 Vilakazi Street, Soweto.",
            [("12 Vilakazi Street, Soweto", "STREET_ADDRESS")],
        ),
        (
            "Patient Nkosinathi Dlamini attended the clinic.",
            [("Nkosinathi Dlamini", "PERSON")],
        ),
        (
            "Patient Annelie van der Merwe attended the clinic.",
            [("Annelie van der Merwe", "PERSON")],
        ),
    ],
    ids=("sa-id", "za-phone", "sa-address", "isizulu-name", "afrikaans-name"),
)
def test_za_popia_synthetic_clinical_fixtures_leave_zero_direct_identifiers(
    monkeypatch,
    text: str,
    detections: list[tuple[str, str]],
):
    entities = [_entity(text, surface, label, 0.99) for surface, label in detections]
    _patch_extract_many(monkeypatch, entities)

    result = deidentify(
        text,
        policy="za_popia",
        use_safety_sweep=False,
        seed=42,
    )
    audit = deidentify(
        text,
        policy="za_popia",
        use_safety_sweep=False,
        seed=42,
        audit=True,
    )

    assert result.mapping is None
    assert all(surface not in result.deidentified_text for surface, _ in detections)
    assert result.metadata["safety_sweep"]["source"] == "safety_sweep"
    assert audit.policy == "za_popia"
    assert audit.safety_sweep["enabled"] is True
    assert audit.residual_risk["risk_report"]["leakage_rate"] == 0.0
    assert all(span.action != "keep" for span in audit.spans)


def test_ng_ndpa_profile_loads_with_strict_sensitive_data_defaults():
    profile = load_policy("ng_ndpa")

    assert profile.name == "ng_ndpa"
    assert "ng_ndpa" in list_policies()
    assert profile.default_action == "replace"
    assert profile.policy_label_actions == {
        "DIRECT_IDENTIFIER": "replace",
        "QUASI_IDENTIFIER": "mask",
        "SENSITIVE_ATTRIBUTE": "mask",
        "CLINICAL_CONCEPT": "mask",
    }
    assert profile.safety_sweep_mandatory is True
    assert profile.keep_mapping is False
    assert profile.reversible_id is False
    assert profile.strict_no_leak is True
    assert set(profile.actions) == set(CANONICAL_LABELS)

    assert {
        label: profile.action_for(label)
        for label in (
            "ID_NUM",
            "ACCOUNT_NUMBER",
            "EYE_COLOR",
            "HEIGHT",
            "CONDITION",
            "GENE_SYMBOL",
            "ORGANIZATION",
            "OTHER",
        )
    } == {
        "ID_NUM": "mask",
        "ACCOUNT_NUMBER": "mask",
        "EYE_COLOR": "mask",
        "HEIGHT": "mask",
        "CONDITION": "mask",
        "GENE_SYMBOL": "mask",
        "ORGANIZATION": "mask",
        "OTHER": "mask",
    }
    assert lint_policy("ng_ndpa") == ()
    lint_report = lint_policy_report("ng_ndpa")
    assert lint_report["valid"] is True
    assert lint_report["error_count"] == 0
    assert lint_report["warning_count"] == 0


def test_ng_ndpa_checklist_classes_have_non_keep_canonical_coverage():
    profile = load_policy("ng_ndpa")
    checklist_rows = _ng_ndpa_checklist_rows()

    assert set(checklist_rows) == set(NDPA_SENSITIVE_DATA_CLASSES)
    assert set(NDPA_SENSITIVE_CLASS_LABELS) == set(NDPA_SENSITIVE_DATA_CLASSES)

    for ndpa_class, labels in checklist_rows.items():
        assert labels
        assert frozenset(labels) == NDPA_SENSITIVE_CLASS_LABELS[ndpa_class]
        assert all(ndpa_class in ndpa_classes_for(label) for label in labels)
        assert all(profile.action_for(label) != "keep" for label in labels)


@pytest.mark.parametrize(
    "case",
    _ng_ndpa_fixture_cases(),
    ids=lambda case: str(case["id"]),
)
def test_ng_ndpa_synthetic_fixtures_leave_zero_direct_identifiers(
    monkeypatch,
    case: dict[str, object],
):
    text = case["text"]
    raw_detections = case["detections"]
    assert isinstance(text, str)
    assert isinstance(raw_detections, list)

    detections: list[tuple[str, str]] = []
    for detection in raw_detections:
        assert isinstance(detection, dict)
        surface = detection["surface"]
        label = detection["label"]
        assert isinstance(surface, str)
        assert isinstance(label, str)
        detections.append((surface, label))

    entities = [_entity(text, surface, label, 0.99) for surface, label in detections]
    _patch_extract_many(monkeypatch, entities)

    result = deidentify(
        text,
        policy="ng_ndpa",
        use_safety_sweep=False,
        seed=42,
    )
    audit = deidentify(
        text,
        policy="ng_ndpa",
        use_safety_sweep=False,
        seed=42,
        audit=True,
    )

    assert result.mapping is None
    assert all(surface not in result.deidentified_text for surface, _ in detections)
    assert result.metadata["safety_sweep"]["source"] == "safety_sweep"
    assert audit.policy == "ng_ndpa"
    assert audit.safety_sweep["enabled"] is True
    assert audit.residual_risk["risk_report"]["leakage_rate"] == 0.0
    assert all(span.action != "keep" for span in audit.spans)


def test_ke_dpa_profile_alias_posture_and_lint_load():
    profile = load_policy("ke_dpa")

    assert profile.name == "ke_dpa"
    assert load_policy("kenya_dpa").name == "ke_dpa"
    assert canonical_policy_name("kenya-dpa") == "ke_dpa"
    assert "ke_dpa" in list_policies()
    assert profile.strict_no_leak is True
    assert profile.safety_sweep_mandatory is True
    assert profile.keep_mapping is False
    assert profile.reversible_id is False
    assert profile.action_for("ID_NUM") == "mask"
    assert profile.action_for("LOCATION") == "mask"
    assert profile.action_for("CONDITION") == "mask"
    assert profile.action_for("OTHER") == "mask"
    assert lint_policy("ke_dpa") == ()


def test_ke_dpa_section_2_sensitive_classes_have_non_keep_actions():
    fixture = _ke_dpa_fixture()
    sensitive_classes = fixture["sensitive_classes"]
    assert isinstance(sensitive_classes, list)
    assert {item["dpa_class"] for item in sensitive_classes} == (
        _KE_DPA_SENSITIVE_CLASSES
    )

    profile = load_policy("ke_dpa")
    for item in sensitive_classes:
        labels = item["canonical_labels"]
        assert labels, item["dpa_class"]
        for label in labels:
            assert label in CANONICAL_LABELS, (item["dpa_class"], label)
            assert profile.action_for(label) != "keep", (
                item["dpa_class"],
                label,
            )


def test_ke_dpa_synthetic_clinical_fixtures_have_zero_identifier_leakage(
    monkeypatch,
):
    fixture = _ke_dpa_fixture()
    records = fixture["records"]
    assert isinstance(records, list)

    for record in records:
        text = record["text"]
        entities = [
            _entity(text, item["surface"], item["label"], 0.99)
            for item in record["entities"]
        ]
        _patch_extract_many(monkeypatch, entities)

        result = deidentify(
            text,
            policy="ke_dpa",
            use_smart_merging=False,
            use_safety_sweep=False,
        )

        for surface in record["direct_identifiers"]:
            assert surface not in result.deidentified_text, (
                record["fixture_id"],
                surface,
            )
        for item in record["entities"]:
            assert item["surface"] not in result.deidentified_text, (
                record["fixture_id"],
                item["surface"],
            )

        audit = deidentify(
            text,
            policy="ke_dpa",
            use_smart_merging=False,
            use_safety_sweep=False,
            audit=True,
        )

        assert audit.policy == "ke_dpa"
        assert audit.safety_sweep["enabled"] is True
        assert audit.resolved_profile["use_safety_sweep"] is True
        assert audit.residual_risk["risk_report"]["leakage_rate"] == 0.0


def test_canada_pipeda_masks_canadian_identifier_entities(monkeypatch):
    text = "SIN 123-456-782 health card ABCD123456"
    _patch_extract_many(
        monkeypatch,
        [
            _entity(text, "123-456-782", "ID_NUM", 0.99),
            _entity(text, "ABCD123456", "ID_NUM", 0.99),
        ],
    )

    result = deidentify(
        text,
        policy="canada_pipeda",
        use_safety_sweep=False,
    )

    assert result.deidentified_text == "SIN [ID_NUM] health card [ID_NUM_2]"
    assert result.mapping == {
        "[ID_NUM]": "123-456-782",
        "[ID_NUM_2]": "ABCD123456",
    }
    assert [entity.action for entity in result.pii_entities] == ["mask", "mask"]
    assert all(entity.reversible_id for entity in result.pii_entities)


def test_uk_ico_profile_and_alias_load():
    profile = load_policy("uk_ico_anonymisation")

    assert profile.name == "uk_ico_anonymisation"
    assert load_policy("uk_ico").name == "uk_ico_anonymisation"
    assert canonical_policy_name("uk-ico") == "uk_ico_anonymisation"
    assert "uk_ico_anonymisation" in list_policies()
    assert profile.action_for("NHS_NUMBER") == "mask"
    assert profile.action_for("ID_NUM") == "mask"
    assert profile.action_for("PERSON") == "mask"
    assert profile.action_for("LOCATION") == "replace"
    assert profile.action_for("CONDITION") == "keep"
    assert profile.keep_mapping is True
    assert profile.reversible_id is True


def test_uk_ico_masks_nhs_number_and_pseudonymizes_quasi_identifier(monkeypatch):
    text = "NHS number NHS-A4857773456 for patient in Cardiff"
    _patch_extract_many(
        monkeypatch,
        [
            _entity(text, "NHS-A4857773456", "NHS_NUMBER", 0.99),
            _entity(text, "Cardiff", "LOCATION", 0.99),
        ],
    )

    result = deidentify(
        text,
        policy="uk_ico_anonymisation",
        use_smart_merging=False,
        use_safety_sweep=False,
        seed=42,
    )

    assert result.deidentified_text != text
    assert result.deidentified_text.startswith("NHS number [NHS_NUMBER]")
    assert "Cardiff" not in result.deidentified_text
    assert result.mapping is not None
    assert result.mapping["[NHS_NUMBER]"] == "NHS-A4857773456"
    assert result.pii_entities[0].canonical_label == "ID_NUM"
    assert [entity.action for entity in result.pii_entities] == ["mask", "replace"]
    assert all(entity.reversible_id for entity in result.pii_entities)


def test_australia_privacy_act_masks_medicare_and_tfn_entities(monkeypatch):
    text = "Medicare 2123 45670 1 TFN 123 456 782"
    _patch_extract_many(
        monkeypatch,
        [
            _entity(text, "2123 45670 1", "ID_NUM", 0.99),
            _entity(text, "123 456 782", "SSN", 0.99),
        ],
    )

    result = deidentify(
        text,
        policy="australia_privacy_act",
        use_safety_sweep=False,
    )

    assert result.deidentified_text == "Medicare [ID_NUM] TFN [SSN]"
    assert result.mapping is None
    assert [entity.action for entity in result.pii_entities] == ["mask", "mask"]
    assert [
        entity.metadata["policy_action"]["policy"] for entity in result.pii_entities
    ] == [
        "australia_privacy_act",
        "australia_privacy_act",
    ]


def test_deidentify_without_policy_preserves_default_output(monkeypatch):
    text = "Patient John Doe"
    _patch_extract(monkeypatch, _entity(text, "John Doe", "NAME", 0.95))

    result = deidentify(text, method="mask", use_safety_sweep=False)

    assert result.deidentified_text == "Patient [NAME]"
    assert result.mapping is None
    assert "reversible_id" not in result.to_dict()["pii_entities"][0]


def test_unknown_policy_raises_before_detection(monkeypatch):
    from openmed.core import pii

    def fail_extract(text: str, *args: object, **kwargs: object) -> PredictionResult:
        raise AssertionError("detection should not run for an unknown policy")

    monkeypatch.setattr(pii, "extract_pii", fail_extract)

    with pytest.raises(ValueError) as exc_info:
        deidentify("Patient John Doe", policy="not_a_profile")

    message = str(exc_info.value)
    assert "not_a_profile" in message
    for name in CANONICAL_POLICY_NAMES:
        assert name in message


@pytest.mark.parametrize(
    "bad_policy", [123, {"posture": "strict"}, ["hipaa_safe_harbor"]]
)
def test_malformed_policy_type_raises_type_error_explaining_accepted_forms(bad_policy):
    with pytest.raises(
        TypeError, match="policy must be a profile name string or a PolicyProfile"
    ):
        load_policy(bad_policy)


def test_strict_no_leak_forces_union_sweep_and_accurate_cascade(monkeypatch):
    text = "Visited Paris"
    _patch_extract(monkeypatch, _entity(text, "Paris", "LOCATION", 0.2))

    result = deidentify(text, policy="strict_no_leak", use_safety_sweep=False)

    assert result.deidentified_text == "Visited [LOCATION]"

    profile = load_policy("strict_no_leak")
    assert "R3" in profile.forced_cascade_tiers

    pipeline_result = Pipeline(
        model_detector=lambda text, **kwargs: _prediction(text),
        policy="strict_no_leak",
        use_safety_sweep=False,
    ).run("No identifiers")
    assert pipeline_result.stage("safety_sweep").metadata["enabled"] is True
    assert (
        pipeline_result.audit_record["policy"]["arbitration_mode"]
        == MODE_HIGH_RECALL_UNION
    )

    router = CascadeRouter(
        tiny_detector=lambda text, **kwargs: [_span(score=0.2)],
        accurate_detector=lambda text, **kwargs: [],
    )
    cascade_result = Pipeline(
        cascade_router=router,
        policy="strict_no_leak",
        use_safety_sweep=False,
    ).run("Patient record")
    routes = [
        route["route"]
        for route in cascade_result.stage("deterministic_detectors").metadata["routes"]
    ]
    assert R3_ACCURATE in routes


def test_clinical_minimal_redaction_keeps_quasi_identifier_that_strict_masks(
    monkeypatch,
):
    text = "Follow up in Paris"
    entity = _entity(text, "Paris", "LOCATION", 0.95)
    _patch_extract(monkeypatch, entity)

    clinical = deidentify(
        text,
        policy="clinical_minimal_redaction",
        use_safety_sweep=False,
    )
    strict = deidentify(text, policy="strict_no_leak", use_safety_sweep=False)

    assert clinical.deidentified_text == text
    assert strict.deidentified_text == "Follow up in [LOCATION]"


def test_gdpr_pseudonymization_retains_mapping_and_reversible_id(monkeypatch):
    text = "Patient John Doe"
    _patch_extract(monkeypatch, _entity(text, "John Doe", "NAME", 0.95))

    result = deidentify(
        text,
        policy="gdpr_pseudonymization",
        use_safety_sweep=False,
        seed=42,
    )

    assert result.mapping is not None
    assert "John Doe" in result.mapping.values()
    assert result.pii_entities[0].reversible_id is not None
    assert result.pii_entities[0].reversible_id.startswith("rev_")
    assert (
        result.to_dict()["pii_entities"][0]["reversible_id"]
        == result.pii_entities[0].reversible_id
    )
    assert result.metadata["policy"]["name"] == "gdpr_pseudonymization"


def test_hipaa_safe_harbor_applies_safe_harbor_action_map(monkeypatch):
    text = "Visit on someday"
    _patch_extract(monkeypatch, _entity(text, "someday", "DATE", 0.95))

    result = deidentify(
        text,
        policy="hipaa_safe_harbor",
        use_safety_sweep=False,
    )

    assert load_policy("hipaa_safe_harbor").action_for("DATE") == "mask"
    assert result.deidentified_text == "Visit on [DATE]"
    assert result.pii_entities[0].metadata["policy_action"]["action"] == "mask"
