"""Clinical PHI flagship dataset assembly manifest.

This module records source references, labels, and release-gate coverage for
the OpenMed-ClinicalPrivacy-tier0 program without bundling any corpus rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from openmed.core.labels import (
    ACCOUNT_NUMBER,
    AGE,
    API_KEY,
    BUILDING_NUMBER,
    CANONICAL_LABELS,
    CREDIT_CARD,
    DATE,
    DATE_OF_BIRTH,
    EMAIL,
    FIRST_NAME,
    GPS_COORDINATES,
    IBAN,
    ID_NUM,
    LAST_NAME,
    LOCATION,
    MIDDLE_NAME,
    ORGANIZATION,
    PASSWORD,
    PERSON,
    PHONE,
    SSN,
    STREET_ADDRESS,
    URL,
    ZIPCODE,
)
from openmed.eval.datasets.dua_stubs import load_dua_corpus
from openmed.eval.datasets.public import PUBLIC_LABEL_MAPS, load_public_dataset
from openmed.eval.suites.shield import (
    PUBLIC_SAMPLE_NOTES_CONFIG,
    PUBLIC_SAMPLE_REPOSITORY,
    PUBLIC_SAMPLE_SPANS_CONFIG,
    SHIELD,
)

CLINICAL_PHI_MANIFEST_ID = "openmed-clinicalprivacy-tier0"
CLINICAL_PRIVACY_MODEL_ID = "OpenMed/OpenMed-ClinicalPrivacy-tier0"
CLINICAL_PHI_MANIFEST_REF = (
    f"openmed.eval.datasets:load_clinical_phi_manifest@{CLINICAL_PHI_MANIFEST_ID}"
)


def _unique_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(labels))


_CLINICAL_LABEL_GROUPS: Mapping[str, tuple[str, ...]] = {
    "names": (PERSON, FIRST_NAME, LAST_NAME, MIDDLE_NAME),
    "dates": (DATE, DATE_OF_BIRTH),
    "ages_over_89": (AGE,),
    "addresses": (
        LOCATION,
        STREET_ADDRESS,
        BUILDING_NUMBER,
        ZIPCODE,
        GPS_COORDINATES,
    ),
    "ids": (ID_NUM, SSN, ACCOUNT_NUMBER),
    "providers": (PERSON, ORGANIZATION),
    "facilities": (ORGANIZATION,),
    "contacts": (EMAIL, PHONE, URL),
}

_G1A_LABELS = _unique_labels(
    _CLINICAL_LABEL_GROUPS["names"]
    + _CLINICAL_LABEL_GROUPS["dates"]
    + _CLINICAL_LABEL_GROUPS["ages_over_89"]
    + _CLINICAL_LABEL_GROUPS["addresses"]
    + _CLINICAL_LABEL_GROUPS["ids"]
    + _CLINICAL_LABEL_GROUPS["contacts"]
)
_G2_LABELS = _unique_labels(
    _CLINICAL_LABEL_GROUPS["names"]
    + _CLINICAL_LABEL_GROUPS["dates"]
    + _CLINICAL_LABEL_GROUPS["addresses"]
)
_G3_LABELS = (
    ID_NUM,
    SSN,
    ACCOUNT_NUMBER,
    API_KEY,
    CREDIT_CARD,
    IBAN,
    PASSWORD,
)


@dataclass(frozen=True)
class ClinicalPHISource:
    """One corpus reference in the clinical PHI dataset assembly plan."""

    source_id: str
    dataset: str
    role: str
    access: str
    loader_ref: str
    license_id: str
    redistribution: str
    labels: tuple[str, ...]
    source_url: str = ""
    split: str = ""
    label_map: Mapping[str, str] = field(default_factory=dict)
    requires_credentials: bool = False
    eval_only: bool = False
    synthetic: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "access": self.access,
            "dataset": self.dataset,
            "eval_only": self.eval_only,
            "label_map": dict(sorted(self.label_map.items())),
            "labels": list(self.labels),
            "license_id": self.license_id,
            "loader_ref": self.loader_ref,
            "notes": self.notes,
            "redistribution": self.redistribution,
            "requires_credentials": self.requires_credentials,
            "role": self.role,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "split": self.split,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class GateFamilyRequirement:
    """Release-gate coverage expected from the assembled datasets."""

    gate: str
    metric: str
    comparator: str
    threshold: float
    labels: tuple[str, ...]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparator": self.comparator,
            "evidence": list(self.evidence),
            "gate": self.gate,
            "labels": list(self.labels),
            "metric": self.metric,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class ClinicalPHIDatasetManifest:
    """Dataset assembly contract for the clinical PHI flagship."""

    manifest_id: str
    model_id: str
    tier: str
    recipe_mode: str
    sources: tuple[ClinicalPHISource, ...]
    gate_families: tuple[GateFamilyRequirement, ...]
    label_groups: Mapping[str, tuple[str, ...]]
    benchmark_suite: str = SHIELD
    schema_version: str = "openmed.eval.datasets.clinical_phi.v1"

    def source(self, source_id: str) -> ClinicalPHISource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"unknown clinical PHI source: {source_id}")

    def gate(self, gate: str) -> GateFamilyRequirement:
        for requirement in self.gate_families:
            if requirement.gate == gate:
                return requirement
        raise KeyError(f"unknown clinical PHI gate: {gate}")

    def required_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        for values in self.label_groups.values():
            labels.extend(values)
        return _unique_labels(tuple(labels))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_suite": self.benchmark_suite,
            "gate_families": [
                requirement.to_dict() for requirement in self.gate_families
            ],
            "label_groups": {
                name: list(values) for name, values in sorted(self.label_groups.items())
            },
            "manifest_id": self.manifest_id,
            "model_id": self.model_id,
            "recipe_mode": self.recipe_mode,
            "schema_version": self.schema_version,
            "sources": [source.to_dict() for source in self.sources],
            "tier": self.tier,
        }


def load_clinical_phi_manifest() -> ClinicalPHIDatasetManifest:
    """Return the committed clinical PHI dataset assembly manifest."""

    manifest = ClinicalPHIDatasetManifest(
        manifest_id=CLINICAL_PHI_MANIFEST_ID,
        model_id=CLINICAL_PRIVACY_MODEL_ID,
        tier="tier0",
        recipe_mode="C",
        sources=(
            ClinicalPHISource(
                source_id="shield_public_sample",
                dataset=SHIELD,
                role="public_comparison",
                access="public_reference",
                loader_ref="openmed.eval.suites.shield:load_shield_fixtures",
                license_id="access-controlled-full; public-sample",
                redistribution="reference-only",
                labels=tuple(PUBLIC_LABEL_MAPS[SHIELD].values()),
                source_url=f"https://huggingface.co/datasets/{PUBLIC_SAMPLE_REPOSITORY}",
                split="train",
                label_map=PUBLIC_LABEL_MAPS[SHIELD],
                notes=(
                    "Public sample supports comparison reporting; full corpus "
                    "requires approved access and remains outside the repository."
                ),
            ),
            ClinicalPHISource(
                source_id="synthetic_golden_deid",
                dataset="golden",
                role="synthetic_training_and_leakage_fixtures",
                access="committed_synthetic",
                loader_ref="openmed.eval.golden:load_benchmark_fixtures",
                license_id="apache-2.0",
                redistribution="committed synthetic fixtures only",
                labels=_unique_labels(_G1A_LABELS + _G3_LABELS),
                source_url="openmed/eval/golden/fixtures",
                synthetic=True,
                notes="Synthetic-only fixtures; no real PHI and no DUA content.",
            ),
            _dua_source("i2b2_eval_only", "i2b2"),
            _dua_source("n2c2_eval_only", "n2c2"),
        ),
        gate_families=(
            GateFamilyRequirement(
                gate="G1a",
                metric="recall",
                comparator=">=",
                threshold=0.990,
                labels=_G1A_LABELS,
                evidence=("shield_public_sample", "synthetic_golden_deid"),
            ),
            GateFamilyRequirement(
                gate="G2",
                metric="name_address_date_recall",
                comparator=">=",
                threshold=0.980,
                labels=_G2_LABELS,
                evidence=("shield_public_sample", "synthetic_golden_deid"),
            ),
            GateFamilyRequirement(
                gate="G3",
                metric="critical_leakage_count",
                comparator="==",
                threshold=0.0,
                labels=_G3_LABELS,
                evidence=("synthetic_golden_deid", "i2b2_eval_only", "n2c2_eval_only"),
            ),
        ),
        label_groups=_CLINICAL_LABEL_GROUPS,
    )
    validate_clinical_phi_manifest(manifest)
    return manifest


def clinical_phi_manifest_hash(
    manifest: ClinicalPHIDatasetManifest | None = None,
) -> str:
    """Return a deterministic hash over the manifest contract."""

    payload = (manifest or load_clinical_phi_manifest()).to_dict()
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def validate_clinical_phi_manifest(manifest: ClinicalPHIDatasetManifest) -> None:
    """Validate source references, label coverage, and gate evidence."""

    source_ids = [source.source_id for source in manifest.sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("clinical PHI source ids must be unique")

    if manifest.model_id != CLINICAL_PRIVACY_MODEL_ID:
        raise ValueError("clinical PHI manifest must target the named flagship model")

    for label in manifest.required_labels():
        _require_canonical_label(label)

    source_by_id = {source.source_id: source for source in manifest.sources}
    shield = source_by_id.get("shield_public_sample")
    if shield is None or shield.dataset != SHIELD:
        raise ValueError("clinical PHI manifest must include SHIELD public evidence")
    if shield.label_map != PUBLIC_LABEL_MAPS[SHIELD]:
        raise ValueError("SHIELD label map must match the public adapter map")
    if shield.redistribution != "reference-only":
        raise ValueError("SHIELD must be referenced, not redistributed")

    for source in manifest.sources:
        for label in source.labels:
            _require_canonical_label(label)
        if source.access == "credentialed_eval_only":
            if not source.requires_credentials or not source.eval_only:
                raise ValueError(
                    "DUA sources must be credentialed eval-only references"
                )
            if "committed" in source.redistribution:
                raise ValueError("DUA sources must not be committed")

    gate_names = {gate.gate for gate in manifest.gate_families}
    if gate_names != {"G1a", "G2", "G3"}:
        raise ValueError("clinical PHI manifest must cover G1a, G2, and G3")

    for requirement in manifest.gate_families:
        for label in requirement.labels:
            _require_canonical_label(label)
        missing_sources = [
            evidence
            for evidence in requirement.evidence
            if evidence not in source_by_id
        ]
        if missing_sources:
            raise ValueError(
                f"gate {requirement.gate} references unknown source(s): "
                + ", ".join(missing_sources)
            )


def resolve_clinical_phi_source(
    source_id: str,
    *,
    public_paths: Mapping[str, str | Path] | None = None,
    credentialed_paths: Mapping[str, str | Path] | None = None,
    manifest: ClinicalPHIDatasetManifest | None = None,
) -> Any:
    """Resolve one source through its adapter without downloading corpus rows."""

    active_manifest = manifest or load_clinical_phi_manifest()
    source = active_manifest.source(source_id)
    if source.dataset == SHIELD:
        path = (public_paths or {}).get(source.source_id)
        return load_public_dataset(SHIELD, path)
    if source.synthetic:
        from openmed.eval.golden import load_benchmark_fixtures

        return tuple(load_benchmark_fixtures())
    if source.access == "credentialed_eval_only":
        path = (credentialed_paths or {}).get(source.source_id)
        return load_dua_corpus(source.dataset, path)
    raise ValueError(f"unknown clinical PHI source access mode: {source.access}")


def _dua_source(source_id: str, dataset: str) -> ClinicalPHISource:
    return ClinicalPHISource(
        source_id=source_id,
        dataset=dataset,
        role="gated_heldout_eval",
        access="credentialed_eval_only",
        loader_ref="openmed.eval.datasets:load_dua_corpus",
        license_id="local-dua-required",
        redistribution="not redistributed; local credentialed path only",
        labels=_unique_labels(_G1A_LABELS + _G2_LABELS + _G3_LABELS),
        requires_credentials=True,
        eval_only=True,
        notes="Declared for held-out evaluation; rows are never bundled.",
    )


def _require_canonical_label(label: str) -> None:
    if label not in CANONICAL_LABELS:
        raise ValueError(f"unknown canonical label in clinical PHI manifest: {label}")


__all__ = [
    "CLINICAL_PHI_MANIFEST_ID",
    "CLINICAL_PHI_MANIFEST_REF",
    "CLINICAL_PRIVACY_MODEL_ID",
    "ClinicalPHIDatasetManifest",
    "ClinicalPHISource",
    "GateFamilyRequirement",
    "clinical_phi_manifest_hash",
    "load_clinical_phi_manifest",
    "resolve_clinical_phi_source",
    "validate_clinical_phi_manifest",
    "PUBLIC_SAMPLE_NOTES_CONFIG",
    "PUBLIC_SAMPLE_SPANS_CONFIG",
]
