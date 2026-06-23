"""DirectID tiny-head contract for the P0 identifier detector family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from openmed.core.labels import (
    ACCOUNT_NUMBER,
    API_KEY,
    CANONICAL_LABELS,
    CREDIT_CARD,
    DIRECT_IDENTIFIER,
    EMAIL,
    IBAN,
    ID_NUM,
    PHONE,
    SSN,
    normalize_label,
    policy_label_for,
)
from openmed.training.recipe import TrainingRecipeConfig, load_preset

DIRECTID_CONTRACT_VERSION = "openmed.training.directid.v1"
DIRECTID_FAMILY = "OpenMed-PII-DirectID"
DIRECTID_TIER = "tiny"
DIRECTID_CONTRACT_REF = "openmed.training.directid:DIRECTID_TINY_HEAD_CONTRACT@v1"
DIRECTID_LABEL_SET_REF = "openmed.core.labels:CANONICAL_LABELS@policy-spine-v1"

DIRECTID_REQUIRED_LABELS = (
    ID_NUM,
    SSN,
    ACCOUNT_NUMBER,
    CREDIT_CARD,
    IBAN,
    EMAIL,
    PHONE,
    API_KEY,
)
DIRECTID_STRUCTURED_ID_LABELS = (
    ID_NUM,
    SSN,
    ACCOUNT_NUMBER,
    CREDIT_CARD,
    IBAN,
    API_KEY,
)
DIRECTID_GATE_CODES = ("G1b", "G3", "G4", "G5")


class DirectIDContractError(ValueError):
    """Raised when the DirectID head contract or preset is incompatible."""


@dataclass(frozen=True)
class GateEvidenceRequirement:
    """Release-gate evidence required from the DirectID training family."""

    code: str
    target: str
    required_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "required_fields": list(self.required_fields),
            "target": self.target,
        }


@dataclass(frozen=True)
class DirectIDHeadContract:
    """Versioned contract shared by DirectID data, training, and gates."""

    schema_version: str
    family: str
    tier: str
    contract_ref: str
    label_set_ref: str
    labels: tuple[str, ...]
    critical_labels: tuple[str, ...]
    structured_id_labels: tuple[str, ...]
    safety_sweep_required: bool
    safety_sweep_source: str
    safety_sweep_provenance_fields: tuple[str, ...]
    quantization_default: str
    optional_quantization_formats: tuple[str, ...]
    gate_requirements: tuple[GateEvidenceRequirement, ...]

    def gate_codes(self) -> tuple[str, ...]:
        return tuple(requirement.code for requirement in self.gate_requirements)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_ref": self.contract_ref,
            "critical_labels": list(self.critical_labels),
            "family": self.family,
            "gate_requirements": [
                requirement.to_dict() for requirement in self.gate_requirements
            ],
            "label_set_ref": self.label_set_ref,
            "labels": list(self.labels),
            "optional_quantization_formats": list(self.optional_quantization_formats),
            "quantization_default": self.quantization_default,
            "safety_sweep_provenance_fields": list(self.safety_sweep_provenance_fields),
            "safety_sweep_required": self.safety_sweep_required,
            "safety_sweep_source": self.safety_sweep_source,
            "schema_version": self.schema_version,
            "structured_id_labels": list(self.structured_id_labels),
            "tier": self.tier,
        }


@dataclass(frozen=True)
class DirectIDPresetValidation:
    """Validated compatibility summary for a DirectID training preset."""

    preset_name: str
    mode: str
    contract_ref: str
    family: str
    tier: str
    labels: tuple[str, ...]
    critical_labels: tuple[str, ...]
    gate_codes: tuple[str, ...]
    quantization_default: str

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_ref": self.contract_ref,
            "critical_labels": list(self.critical_labels),
            "family": self.family,
            "gate_codes": list(self.gate_codes),
            "labels": list(self.labels),
            "mode": self.mode,
            "preset_name": self.preset_name,
            "quantization_default": self.quantization_default,
            "tier": self.tier,
        }


DIRECTID_TINY_HEAD_CONTRACT = DirectIDHeadContract(
    schema_version=DIRECTID_CONTRACT_VERSION,
    family=DIRECTID_FAMILY,
    tier=DIRECTID_TIER,
    contract_ref=DIRECTID_CONTRACT_REF,
    label_set_ref=DIRECTID_LABEL_SET_REF,
    labels=DIRECTID_REQUIRED_LABELS,
    critical_labels=DIRECTID_REQUIRED_LABELS,
    structured_id_labels=DIRECTID_STRUCTURED_ID_LABELS,
    safety_sweep_required=True,
    safety_sweep_source="safety_sweep",
    safety_sweep_provenance_fields=(
        "source",
        "patterns_version",
        "start",
        "end",
        "text_hash",
    ),
    quantization_default="int8",
    optional_quantization_formats=("int4",),
    gate_requirements=(
        GateEvidenceRequirement(
            code="G1b",
            target="structured-id recall >= 0.995",
            required_fields=(
                "per_label_recall",
                "structured_id_recall",
                "eval_set_hash",
            ),
        ),
        GateEvidenceRequirement(
            code="G3",
            target="critical leakage count == 0",
            required_fields=(
                "critical_leakage_count",
                "residual_leakage_rate",
                "leakage_fixture_hash",
            ),
        ),
        GateEvidenceRequirement(
            code="G4",
            target="INT8 recall delta < 0.005; INT4 recall delta < 0.010",
            required_fields=(
                "format",
                "quant_recall_delta",
                "fp_parent_per_label_recall",
            ),
        ),
        GateEvidenceRequirement(
            code="G5",
            target="Tiny-tier RAM and latency fit",
            required_fields=("param_count", "p50_ms", "p95_ms", "ram_mb"),
        ),
    ),
)


def validate_directid_contract(
    contract: DirectIDHeadContract = DIRECTID_TINY_HEAD_CONTRACT,
) -> DirectIDHeadContract:
    """Validate the DirectID contract and return it when coherent."""

    errors = _contract_errors(contract)
    if errors:
        raise DirectIDContractError("; ".join(errors))
    return contract


def validate_directid_preset(
    config: TrainingRecipeConfig | None = None,
    *,
    contract: DirectIDHeadContract = DIRECTID_TINY_HEAD_CONTRACT,
) -> DirectIDPresetValidation:
    """Validate that a recipe preset is compatible with DirectID tiny training."""

    validate_directid_contract(contract)
    recipe = config if config is not None else load_preset("tiny_distill")
    errors = _preset_errors(recipe, contract)
    if errors:
        raise DirectIDContractError("; ".join(errors))
    return DirectIDPresetValidation(
        preset_name=recipe.preset_name,
        mode=recipe.mode,
        contract_ref=contract.contract_ref,
        family=contract.family,
        tier=contract.tier,
        labels=contract.labels,
        critical_labels=contract.critical_labels,
        gate_codes=contract.gate_codes(),
        quantization_default=recipe.quantization.default,
    )


def gate_requirements_by_code(
    contract: DirectIDHeadContract = DIRECTID_TINY_HEAD_CONTRACT,
) -> Mapping[str, GateEvidenceRequirement]:
    """Return DirectID gate requirements keyed by release-gate code."""

    validate_directid_contract(contract)
    return {requirement.code: requirement for requirement in contract.gate_requirements}


def _contract_errors(contract: DirectIDHeadContract) -> list[str]:
    errors: list[str] = []
    labels = tuple(normalize_label(label) for label in contract.labels)
    critical_labels = tuple(
        normalize_label(label) for label in contract.critical_labels
    )
    structured_id_labels = tuple(
        normalize_label(label) for label in contract.structured_id_labels
    )

    if contract.schema_version != DIRECTID_CONTRACT_VERSION:
        errors.append(
            "schema_version must be "
            f"{DIRECTID_CONTRACT_VERSION!r}, got {contract.schema_version!r}"
        )
    if contract.family != DIRECTID_FAMILY:
        errors.append(f"family must be {DIRECTID_FAMILY!r}")
    if contract.tier != DIRECTID_TIER:
        errors.append(f"tier must be {DIRECTID_TIER!r}")
    if contract.contract_ref != DIRECTID_CONTRACT_REF:
        errors.append(f"contract_ref must be {DIRECTID_CONTRACT_REF!r}")
    if contract.label_set_ref != DIRECTID_LABEL_SET_REF:
        errors.append(f"label_set_ref must be {DIRECTID_LABEL_SET_REF!r}")

    _append_label_errors("labels", labels, errors)
    _append_label_errors("critical_labels", critical_labels, errors)
    _append_label_errors("structured_id_labels", structured_id_labels, errors)

    required_missing = sorted(set(DIRECTID_REQUIRED_LABELS) - set(labels))
    if required_missing:
        errors.append(
            "labels missing required DirectID label(s): " + ", ".join(required_missing)
        )
    critical_missing = sorted(set(DIRECTID_REQUIRED_LABELS) - set(critical_labels))
    if critical_missing:
        errors.append(
            "critical_labels missing required DirectID label(s): "
            + ", ".join(critical_missing)
        )
    structured_missing = sorted(
        set(DIRECTID_STRUCTURED_ID_LABELS) - set(structured_id_labels)
    )
    if structured_missing:
        errors.append(
            "structured_id_labels missing required label(s): "
            + ", ".join(structured_missing)
        )
    non_critical = sorted(set(critical_labels) - set(labels))
    if non_critical:
        errors.append(
            "critical_labels must be a subset of labels; extra label(s): "
            + ", ".join(non_critical)
        )
    non_structured = sorted(set(structured_id_labels) - set(labels))
    if non_structured:
        errors.append(
            "structured_id_labels must be a subset of labels; extra label(s): "
            + ", ".join(non_structured)
        )

    if contract.safety_sweep_required is not True:
        errors.append("safety_sweep_required must be true")
    if contract.safety_sweep_source != "safety_sweep":
        errors.append("safety_sweep_source must be 'safety_sweep'")
    required_provenance = {"source", "patterns_version", "start", "end", "text_hash"}
    missing_provenance = sorted(
        required_provenance - set(contract.safety_sweep_provenance_fields)
    )
    if missing_provenance:
        errors.append(
            "safety_sweep_provenance_fields missing field(s): "
            + ", ".join(missing_provenance)
        )
    if contract.quantization_default != "int8":
        errors.append("quantization_default must be 'int8'")
    if "int4" not in contract.optional_quantization_formats:
        errors.append("optional_quantization_formats must include 'int4'")
    if contract.gate_codes() != DIRECTID_GATE_CODES:
        errors.append("gate_requirements must declare G1b, G3, G4, and G5 in order")
    for requirement in contract.gate_requirements:
        if not requirement.required_fields:
            errors.append(f"{requirement.code} must declare required_fields")
        if not requirement.target:
            errors.append(f"{requirement.code} must declare a target")
    return errors


def _preset_errors(
    recipe: TrainingRecipeConfig,
    contract: DirectIDHeadContract,
) -> list[str]:
    errors: list[str] = []
    if recipe.head_contract != contract.contract_ref:
        errors.append("head_contract must reference the DirectID tiny head contract")
    if recipe.mode != "A":
        errors.append("DirectID tiny training must use Mode A")
    if recipe.output_tier != contract.tier:
        errors.append(f"output_tier must be {contract.tier!r}")
    if recipe.label_set_ref != contract.label_set_ref:
        errors.append(f"label_set_ref must be {contract.label_set_ref!r}")
    if recipe.quantization.default != contract.quantization_default:
        errors.append(f"quantization.default must be {contract.quantization_default!r}")
    if recipe.hard_negatives_required is not True:
        errors.append("hard_negatives_required must be true")

    missing_critical = sorted(
        set(contract.critical_labels) - set(recipe.loss.critical_labels)
    )
    if missing_critical:
        errors.append(
            "loss.critical_labels missing DirectID label(s): "
            + ", ".join(missing_critical)
        )
    return errors


def _append_label_errors(
    field_name: str,
    labels: tuple[str, ...],
    errors: list[str],
) -> None:
    if not labels:
        errors.append(f"{field_name} must not be empty")
        return
    unknown_labels = sorted(set(labels) - CANONICAL_LABELS)
    if unknown_labels:
        errors.append(
            f"{field_name} contains unknown label(s): " + ", ".join(unknown_labels)
        )
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        errors.append(
            f"{field_name} contains duplicate label(s): " + ", ".join(duplicate_labels)
        )
    non_direct = sorted(
        label
        for label in labels
        if label in CANONICAL_LABELS and policy_label_for(label) != DIRECT_IDENTIFIER
    )
    if non_direct:
        errors.append(
            f"{field_name} contains non-direct identifier label(s): "
            + ", ".join(non_direct)
        )


__all__ = [
    "DIRECTID_CONTRACT_REF",
    "DIRECTID_CONTRACT_VERSION",
    "DIRECTID_FAMILY",
    "DIRECTID_GATE_CODES",
    "DIRECTID_LABEL_SET_REF",
    "DIRECTID_REQUIRED_LABELS",
    "DIRECTID_STRUCTURED_ID_LABELS",
    "DIRECTID_TIER",
    "DIRECTID_TINY_HEAD_CONTRACT",
    "DirectIDContractError",
    "DirectIDHeadContract",
    "DirectIDPresetValidation",
    "GateEvidenceRequirement",
    "gate_requirements_by_code",
    "validate_directid_contract",
    "validate_directid_preset",
]
