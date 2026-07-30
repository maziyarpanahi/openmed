from __future__ import annotations

from dataclasses import replace

import pytest

from openmed.core.labels import (
    ACCOUNT_NUMBER,
    API_KEY,
    ID_NUM,
    ID_SUBTYPE_MRN,
    ID_SUBTYPE_NATIONAL_ID,
    ID_SUBTYPE_NPI,
    OTHER,
    PASSWORD,
    SSN,
)
from openmed.training import (
    DIRECTID_CONTRACT_REF,
    DIRECTID_GATE_CODES,
    DIRECTID_REQUIRED_ID_SUBTYPES,
    DIRECTID_TINY_HEAD_CONTRACT,
    DirectIDContractError,
    gate_requirements_by_code,
    load_preset,
    validate_directid_contract,
    validate_directid_preset,
)


def test_directid_contract_validates_required_labels_and_gate_evidence():
    contract = validate_directid_contract()
    gate_requirements = gate_requirements_by_code(contract)

    assert contract.family == "OpenMed-PII-DirectID"
    assert contract.tier == "tiny"
    assert set(contract.labels) >= {
        ID_NUM,
        SSN,
        ACCOUNT_NUMBER,
        "CREDIT_CARD",
        "IBAN",
        "EMAIL",
        "PHONE",
        API_KEY,
    }
    assert contract.id_subtypes == DIRECTID_REQUIRED_ID_SUBTYPES
    assert set(contract.id_subtypes) == {
        ID_SUBTYPE_MRN,
        ID_SUBTYPE_NPI,
        ID_SUBTYPE_NATIONAL_ID,
    }
    assert contract.safety_sweep_required is True
    assert contract.safety_sweep_source == "safety_sweep"
    assert {"source", "patterns_version", "start", "end", "text_hash"} <= set(
        contract.safety_sweep_provenance_fields
    )
    assert contract.gate_codes() == DIRECTID_GATE_CODES
    assert set(gate_requirements) == {"G1b", "G3", "G4", "G5"}
    assert "structured_id_recall" in gate_requirements["G1b"].required_fields
    assert "critical_leakage_count" in gate_requirements["G3"].required_fields
    assert "quant_recall_delta" in gate_requirements["G4"].required_fields
    assert "ram_mb" in gate_requirements["G5"].required_fields


def test_directid_contract_rejects_missing_required_label():
    contract = replace(
        DIRECTID_TINY_HEAD_CONTRACT,
        labels=tuple(
            label for label in DIRECTID_TINY_HEAD_CONTRACT.labels if label != SSN
        ),
    )

    with pytest.raises(DirectIDContractError, match="SSN"):
        validate_directid_contract(contract)


def test_directid_contract_rejects_non_direct_identifier_label():
    contract = replace(
        DIRECTID_TINY_HEAD_CONTRACT,
        labels=DIRECTID_TINY_HEAD_CONTRACT.labels + (OTHER,),
    )

    with pytest.raises(DirectIDContractError, match="non-direct identifier"):
        validate_directid_contract(contract)


def test_directid_contract_rejects_non_directid_extra_label():
    contract = replace(
        DIRECTID_TINY_HEAD_CONTRACT,
        labels=DIRECTID_TINY_HEAD_CONTRACT.labels + (PASSWORD,),
    )

    with pytest.raises(DirectIDContractError, match="PASSWORD"):
        validate_directid_contract(contract)


def test_directid_contract_rejects_missing_required_id_subtype():
    contract = replace(
        DIRECTID_TINY_HEAD_CONTRACT,
        id_subtypes=tuple(
            subtype
            for subtype in DIRECTID_TINY_HEAD_CONTRACT.id_subtypes
            if subtype != ID_SUBTYPE_NPI
        ),
    )

    with pytest.raises(DirectIDContractError, match=ID_SUBTYPE_NPI):
        validate_directid_contract(contract)


def test_tiny_distill_preset_declares_directid_contract_compatibility():
    config = load_preset("tiny_distill")
    validation = validate_directid_preset(config)

    assert config.head_contract == DIRECTID_CONTRACT_REF
    assert validation.preset_name == "tiny_distill"
    assert validation.mode == "A"
    assert validation.tier == "tiny"
    assert validation.quantization_default == "int8"
    assert validation.id_subtypes == DIRECTID_REQUIRED_ID_SUBTYPES
    assert config.loss.critical_labels == DIRECTID_TINY_HEAD_CONTRACT.critical_labels


def test_directid_preset_validation_rejects_incompatible_preset():
    config = replace(load_preset("tiny_distill"), head_contract=None)

    with pytest.raises(DirectIDContractError, match="head_contract"):
        validate_directid_preset(config)


def test_directid_preset_validation_rejects_extra_critical_labels():
    config = load_preset("tiny_distill")
    config = replace(
        config,
        loss=replace(
            config.loss,
            critical_labels=config.loss.critical_labels + (PASSWORD,),
        ),
    )

    with pytest.raises(DirectIDContractError, match="PASSWORD"):
        validate_directid_preset(config)
