"""Declarative openEHR archetype bindings for grounded clinical entities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

__all__ = [
    "OpenEHRBinding",
    "DEFAULT_OPENEHR_BINDINGS",
    "binding_for_entity_kind",
]


@dataclass(frozen=True)
class OpenEHRBinding:
    """Map one OpenMed entity kind family to openEHR flat paths.

    Paths are relative to the WebTemplate root id. ``{index}`` is replaced with
    the zero-based occurrence index for the binding family so repeated entities
    remain EHRbase-flat-compatible.
    """

    kind: str
    aliases: tuple[str, ...]
    archetype_id: str
    archetype_label: str
    text_path: str
    code_path: str | None = None
    quantity_path: str | None = None
    preferred_code_systems: tuple[str, ...] = ()

    def flat_text_path(self, template_id: str, index: int) -> str:
        """Return the indexed flat path for the entity text element."""

        return _join_template_path(template_id, self.text_path, index)

    def flat_code_path(self, template_id: str, index: int) -> str | None:
        """Return the indexed flat path for the optional coded element."""

        if self.code_path is None:
            return None
        return _join_template_path(template_id, self.code_path, index)

    def flat_quantity_path(self, template_id: str, index: int) -> str | None:
        """Return the indexed flat path for the optional quantity element."""

        if self.quantity_path is None:
            return None
        return _join_template_path(template_id, self.quantity_path, index)


DEFAULT_OPENEHR_BINDINGS: Final[tuple[OpenEHRBinding, ...]] = (
    OpenEHRBinding(
        kind="problem",
        aliases=("condition", "diagnosis", "problem", "problem_diagnosis"),
        archetype_id="openEHR-EHR-EVALUATION.problem_diagnosis.v1",
        archetype_label="Problem/Diagnosis",
        text_path="problems/problem_diagnosis:{index}/problem_name",
        code_path="problems/problem_diagnosis:{index}/problem_code",
        preferred_code_systems=("SNOMED", "SNOMED-CT", "ICD10CM"),
    ),
    OpenEHRBinding(
        kind="medication",
        aliases=("medication", "drug", "medication_order", "medicine"),
        archetype_id="openEHR-EHR-INSTRUCTION.medication_order.v1",
        archetype_label="Medication order",
        text_path="medications/medication_order:{index}/medication_name",
        code_path="medications/medication_order:{index}/medication_code",
        preferred_code_systems=("RXNORM", "SNOMED", "SNOMED-CT"),
    ),
    OpenEHRBinding(
        kind="laboratory",
        aliases=(
            "lab",
            "lab_result",
            "laboratory",
            "laboratory_result",
            "measurement",
            "test_result",
        ),
        archetype_id="openEHR-EHR-OBSERVATION.laboratory_test_result.v1",
        archetype_label="Laboratory test result",
        text_path=(
            "laboratory/laboratory_test_result:{index}/"
            "laboratory_analyte_result/analyte_name"
        ),
        code_path=(
            "laboratory/laboratory_test_result:{index}/"
            "laboratory_analyte_result/analyte_code"
        ),
        quantity_path=(
            "laboratory/laboratory_test_result:{index}/"
            "laboratory_analyte_result/analyte_value"
        ),
        preferred_code_systems=("LOINC", "SNOMED", "SNOMED-CT"),
    ),
    OpenEHRBinding(
        kind="vital_sign",
        aliases=("vital", "vital_sign", "vital_signs"),
        archetype_id="openEHR-EHR-OBSERVATION.vital_signs.v1",
        archetype_label="Vital signs",
        text_path="vitals/vital_sign:{index}/vital_sign_name",
        code_path="vitals/vital_sign:{index}/vital_sign_code",
        quantity_path="vitals/vital_sign:{index}/value",
        preferred_code_systems=("LOINC", "SNOMED", "SNOMED-CT"),
    ),
)


def binding_for_entity_kind(
    entity_kind: str,
    bindings: Sequence[OpenEHRBinding] = DEFAULT_OPENEHR_BINDINGS,
) -> OpenEHRBinding:
    """Return the configured openEHR binding for a canonical entity label."""

    normalized = _normalize_kind(entity_kind)
    for binding in bindings:
        if normalized == binding.kind or normalized in binding.aliases:
            return binding
    known = sorted({alias for binding in bindings for alias in binding.aliases})
    raise ValueError(
        f"No openEHR binding for entity kind {entity_kind!r}. "
        f"Known kinds: {', '.join(known)}."
    )


def _join_template_path(template_id: str, relative_path: str, index: int) -> str:
    return f"{template_id}/{relative_path.format(index=index)}"


def _normalize_kind(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
