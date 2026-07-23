"""Clinical PHI flagship dataset assembly manifest.

This module records source references, labels, and release-gate coverage for
the OpenMed-ClinicalPrivacy-tier0 program without bundling any corpus rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

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

if TYPE_CHECKING:
    from openmed.eval.harness import BenchmarkFixture
    from openmed.eval.metrics import EvalSpan

CLINICAL_PHI_MANIFEST_ID = "openmed-clinicalprivacy-tier0"
CLINICAL_PRIVACY_MODEL_ID = "OpenMed/OpenMed-ClinicalPrivacy-tier0"
INDIA_CLINICAL_PHI_CORPUS_ID = "openmed-india-clinical-deid-synthetic-v1"
CLINICAL_PHI_MANIFEST_REF = (
    f"openmed.eval.datasets:load_clinical_phi_manifest@{CLINICAL_PHI_MANIFEST_ID}"
)
INDIA_CLINICAL_PHI_SCHEMA_VERSION = "openmed.eval.datasets.india_clinical_phi.v1"

_INDIA_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "golden" / "fixtures" / "i18n"
)
_INDIA_MANIFEST_PATH = _INDIA_FIXTURE_DIR / "india_clinical_manifest.json"
_REQUIRED_INDIA_SCRIPTS = frozenset({"Latin", "Devanagari", "Tamil"})
_REQUIRED_INDIA_IDENTIFIER_TYPES = frozenset(
    {
        "aadhaar",
        "abha",
        "indian_phone",
        "pan",
        "person_name",
        "pin_code",
        "street_address",
    }
)
_SCRIPT_RANGES = {
    "Latin": ((0x0041, 0x005A), (0x0061, 0x007A)),
    "Devanagari": ((0x0900, 0x097F),),
    "Tamil": ((0x0B80, 0x0BFF),),
}


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


@dataclass(frozen=True)
class IndiaClinicalPHIIdentityAlias:
    """One declared script-specific alias for a synthetic person."""

    document_id: str
    text: str
    script: str


@dataclass(frozen=True)
class IndiaClinicalPHIIdentity:
    """Cross-document identity declaration for surrogate-consistency tests."""

    group_id: str
    aliases: tuple[IndiaClinicalPHIIdentityAlias, ...]


@dataclass(frozen=True)
class IndiaClinicalPHICorpusManifest:
    """Safety and coverage contract for the synthetic India corpus."""

    corpus_id: str
    schema_version: str
    fixture_file: str
    synthetic_only: bool
    contains_real_phi: bool
    contains_dua_data: bool
    license_id: str
    provenance: str
    disclaimer: str
    minimum_documents: int
    required_scripts: tuple[str, ...]
    required_identifier_types: tuple[str, ...]
    required_clinical_terms: tuple[str, ...]
    cross_document_identities: tuple[IndiaClinicalPHIIdentity, ...]


@dataclass(frozen=True)
class IndiaClinicalPHIRecord:
    """One normalized synthetic code-mixed India clinical note."""

    fixture_id: str
    document_id: str
    language: str
    languages: tuple[str, ...]
    scripts: tuple[str, ...]
    text: str
    gold_spans: tuple[EvalSpan, ...]
    clinical_terms: tuple[str, ...]
    metadata: Mapping[str, Any]

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        """Return a standard eval-harness view of this record."""

        from openmed.eval.harness import BenchmarkFixture

        metadata = dict(self.metadata)
        metadata.update(
            {
                "clinical_terms": list(self.clinical_terms),
                "document_id": self.document_id,
                "languages": list(self.languages),
                "scripts": list(self.scripts),
            }
        )
        return BenchmarkFixture(
            fixture_id=self.fixture_id,
            text=self.text,
            gold_spans=self.gold_spans,
            language=self.language,
            metadata=metadata,
        )


@dataclass(frozen=True)
class IndiaClinicalPHICorpus:
    """Validated manifest plus deterministic normalized India records."""

    manifest: IndiaClinicalPHICorpusManifest
    records: tuple[IndiaClinicalPHIRecord, ...]


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
                notes=(
                    "Synthetic-only fixtures, including mined hard negatives; "
                    "no real PHI and no DUA content."
                ),
            ),
            ClinicalPHISource(
                source_id="india_synthetic_clinical_deid",
                dataset=INDIA_CLINICAL_PHI_CORPUS_ID,
                role="synthetic_india_deid_evaluation",
                access="committed_synthetic",
                loader_ref=("openmed.eval.datasets:load_india_clinical_phi_corpus"),
                license_id="apache-2.0",
                redistribution="committed synthetic fixtures only",
                labels=(PERSON, ID_NUM, PHONE, STREET_ADDRESS, ZIPCODE),
                source_url=(
                    "openmed/eval/golden/fixtures/i18n/india_clinical_manifest.json"
                ),
                synthetic=True,
                notes=(
                    "Synthetic-only code-mixed India clinical fixtures; no real "
                    "PHI, production data, restricted corpus, or DUA content."
                ),
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


def load_india_clinical_phi_corpus(
    manifest_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
) -> IndiaClinicalPHICorpus:
    """Load and validate the committed synthetic India clinical corpus.

    Args:
        manifest_path: Optional path to a manifest with the committed schema.
        fixture_path: Optional JSONL path overriding ``manifest.fixture_file``.

    Returns:
        A validated manifest and records sorted by document and fixture id.
    """

    active_manifest_path = (
        Path(manifest_path) if manifest_path is not None else _INDIA_MANIFEST_PATH
    )
    payload = json.loads(active_manifest_path.read_text(encoding="utf-8"))
    manifest = _india_manifest_from_mapping(payload)
    validate_india_clinical_phi_manifest(manifest)

    active_fixture_path = (
        Path(fixture_path)
        if fixture_path is not None
        else active_manifest_path.parent / manifest.fixture_file
    )
    rows = [
        json.loads(line)
        for line in active_fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = tuple(
        sorted(
            (_india_record_from_mapping(row, manifest) for row in rows),
            key=lambda record: (record.document_id, record.fixture_id),
        )
    )
    corpus = IndiaClinicalPHICorpus(manifest=manifest, records=records)
    validate_india_clinical_phi_corpus(corpus)
    return corpus


def validate_india_clinical_phi_manifest(
    manifest: IndiaClinicalPHICorpusManifest,
) -> None:
    """Fail closed when the India corpus safety declaration is incomplete."""

    if manifest.corpus_id != INDIA_CLINICAL_PHI_CORPUS_ID:
        raise ValueError("India clinical corpus id is not recognized")
    if manifest.schema_version != INDIA_CLINICAL_PHI_SCHEMA_VERSION:
        raise ValueError("India clinical corpus schema version is not supported")
    if Path(manifest.fixture_file).name != manifest.fixture_file:
        raise ValueError("India clinical fixture_file must be a local filename")
    if manifest.synthetic_only is not True:
        raise ValueError("India clinical corpus must be declared synthetic-only")
    if manifest.contains_real_phi is not False:
        raise ValueError("India clinical corpus must declare contains_real_phi=false")
    if manifest.contains_dua_data is not False:
        raise ValueError("India clinical corpus must declare contains_dua_data=false")
    if not manifest.license_id or not manifest.provenance:
        raise ValueError("India clinical corpus license and provenance are required")

    disclaimer = manifest.disclaimer.casefold()
    if "assist-only" not in disclaimer or "non-decisional" not in disclaimer:
        raise ValueError(
            "India clinical corpus disclaimer must say assist-only and non-decisional"
        )
    if manifest.minimum_documents < 3:
        raise ValueError("India clinical corpus must require at least three documents")
    if not _REQUIRED_INDIA_SCRIPTS <= set(manifest.required_scripts):
        raise ValueError(
            "India clinical corpus must require Latin, Devanagari, and Tamil"
        )
    if not _REQUIRED_INDIA_IDENTIFIER_TYPES <= set(manifest.required_identifier_types):
        raise ValueError("India clinical corpus is missing required identifier types")
    if not manifest.required_clinical_terms:
        raise ValueError("India clinical corpus must declare AYUSH clinical terms")

    group_ids = [identity.group_id for identity in manifest.cross_document_identities]
    if not group_ids or len(group_ids) != len(set(group_ids)):
        raise ValueError("India clinical cross-document identity ids must be unique")
    for identity in manifest.cross_document_identities:
        if len(identity.aliases) < 3:
            raise ValueError(
                "India clinical cross-document identities require three aliases"
            )
        alias_keys = [
            (alias.document_id, alias.text, alias.script) for alias in identity.aliases
        ]
        if len(alias_keys) != len(set(alias_keys)):
            raise ValueError("India clinical identity aliases must be unique")
        if {alias.script for alias in identity.aliases} != _REQUIRED_INDIA_SCRIPTS:
            raise ValueError(
                "India clinical identity aliases must cover all required scripts"
            )


def validate_india_clinical_phi_corpus(corpus: IndiaClinicalPHICorpus) -> None:
    """Validate record uniqueness, coverage, and cross-document identities."""

    manifest = corpus.manifest
    validate_india_clinical_phi_manifest(manifest)
    if len(corpus.records) < manifest.minimum_documents:
        raise ValueError("India clinical corpus has too few documents")

    fixture_ids = [record.fixture_id for record in corpus.records]
    document_ids = [record.document_id for record in corpus.records]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("India clinical fixture ids must be unique")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("India clinical document ids must be unique")

    covered_scripts = {script for record in corpus.records for script in record.scripts}
    if not set(manifest.required_scripts) <= covered_scripts:
        raise ValueError("India clinical corpus does not cover every required script")

    covered_identifier_types = {
        str(span.metadata["identifier_type"])
        for record in corpus.records
        for span in record.gold_spans
    }
    if not set(manifest.required_identifier_types) <= covered_identifier_types:
        raise ValueError(
            "India clinical corpus does not cover every required identifier type"
        )

    covered_terms = {
        term for record in corpus.records for term in record.clinical_terms
    }
    if not set(manifest.required_clinical_terms) <= covered_terms:
        raise ValueError("India clinical corpus does not cover every AYUSH term")

    records_by_document = {record.document_id: record for record in corpus.records}
    declared_groups = {
        identity.group_id: identity for identity in manifest.cross_document_identities
    }
    actual_aliases: dict[str, set[tuple[str, str, str]]] = {
        group_id: set() for group_id in declared_groups
    }
    for record in corpus.records:
        for span in record.gold_spans:
            group_id = str(span.metadata.get("identity_group") or "")
            if not group_id:
                continue
            if span.label != PERSON:
                raise ValueError("only PERSON spans may declare an identity_group")
            if group_id not in declared_groups:
                raise ValueError(
                    "India clinical span references an unknown identity group"
                )
            alias_script = str(span.metadata.get("alias_script") or "")
            actual_aliases[group_id].add((record.document_id, span.text, alias_script))

    for group_id, identity in declared_groups.items():
        expected_aliases = {
            (alias.document_id, alias.text, alias.script) for alias in identity.aliases
        }
        missing_documents = {
            alias.document_id
            for alias in identity.aliases
            if alias.document_id not in records_by_document
        }
        if missing_documents:
            raise ValueError(
                "India clinical identity references unknown document(s): "
                + ", ".join(sorted(missing_documents))
            )
        if actual_aliases[group_id] != expected_aliases:
            raise ValueError(
                f"India clinical identity aliases do not match manifest: {group_id}"
            )


def _india_manifest_from_mapping(
    payload: Mapping[str, Any],
) -> IndiaClinicalPHICorpusManifest:
    if not isinstance(payload, Mapping):
        raise ValueError("India clinical corpus manifest must be a mapping")
    raw_identities = payload.get("cross_document_identities")
    if not isinstance(raw_identities, list):
        raise ValueError("India clinical manifest identities must be a list")

    identities: list[IndiaClinicalPHIIdentity] = []
    for raw_identity in raw_identities:
        if not isinstance(raw_identity, Mapping):
            raise ValueError("India clinical identity must be a mapping")
        raw_aliases = raw_identity.get("aliases")
        if not isinstance(raw_aliases, list):
            raise ValueError("India clinical identity aliases must be a list")
        aliases = tuple(
            IndiaClinicalPHIIdentityAlias(
                document_id=_required_text(alias, "document_id"),
                text=_required_text(alias, "text"),
                script=_required_text(alias, "script"),
            )
            for alias in raw_aliases
            if isinstance(alias, Mapping)
        )
        if len(aliases) != len(raw_aliases):
            raise ValueError("India clinical identity alias must be a mapping")
        identities.append(
            IndiaClinicalPHIIdentity(
                group_id=_required_text(raw_identity, "group_id"),
                aliases=aliases,
            )
        )

    return IndiaClinicalPHICorpusManifest(
        corpus_id=_required_text(payload, "corpus_id"),
        schema_version=_required_text(payload, "schema_version"),
        fixture_file=_required_text(payload, "fixture_file"),
        synthetic_only=_required_bool(payload, "synthetic_only"),
        contains_real_phi=_required_bool(payload, "contains_real_phi"),
        contains_dua_data=_required_bool(payload, "contains_dua_data"),
        license_id=_required_text(payload, "license_id"),
        provenance=_required_text(payload, "provenance"),
        disclaimer=_required_text(payload, "disclaimer"),
        minimum_documents=_required_int(payload, "minimum_documents"),
        required_scripts=_required_text_tuple(payload, "required_scripts"),
        required_identifier_types=_required_text_tuple(
            payload, "required_identifier_types"
        ),
        required_clinical_terms=_required_text_tuple(
            payload, "required_clinical_terms"
        ),
        cross_document_identities=tuple(identities),
    )


def _india_record_from_mapping(
    payload: Mapping[str, Any],
    manifest: IndiaClinicalPHICorpusManifest,
) -> IndiaClinicalPHIRecord:
    from openmed.eval.metrics import normalize_eval_spans

    if not isinstance(payload, Mapping):
        raise ValueError("India clinical fixture must be a mapping")
    if payload.get("schema_version") != manifest.schema_version:
        raise ValueError("India clinical fixture schema version does not match")

    fixture_id = _required_text(payload, "id")
    document_id = _required_text(payload, "document_id")
    language = _required_text(payload, "language")
    languages = _required_text_tuple(payload, "languages")
    scripts = _required_text_tuple(payload, "scripts")
    text = _required_text(payload, "text")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("India clinical fixture metadata must be a mapping")
    metadata = dict(metadata)
    if metadata.get("synthetic") is not True or metadata.get("source") != "generated":
        raise ValueError("India clinical fixture must be synthetic and generated")
    if metadata.get("code_mixed") is not True or len(languages) < 2:
        raise ValueError("India clinical fixtures must be declared code-mixed")

    clinical_terms = _required_text_tuple(metadata, "clinical_terms")
    for term in clinical_terms:
        if term not in text:
            raise ValueError("India clinical term must occur in fixture text")
    if len(scripts) != len(set(scripts)):
        raise ValueError("India clinical fixture scripts must be unique")
    for script in scripts:
        if script not in manifest.required_scripts:
            raise ValueError(f"India clinical fixture uses unknown script: {script}")
        if not _contains_script(text, script):
            raise ValueError(
                f"India clinical fixture does not contain declared script: {script}"
            )

    raw_spans = payload.get("gold_spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("India clinical fixture gold_spans must be a non-empty list")
    normalized_rows: list[dict[str, Any]] = []
    span_ids: set[str] = set()
    address_groups: dict[str, set[str]] = {}
    for raw_span in raw_spans:
        if not isinstance(raw_span, Mapping):
            raise ValueError("India clinical gold span must be a mapping")
        span_id = _required_text(raw_span, "id")
        if span_id in span_ids:
            raise ValueError("India clinical span ids must be unique per document")
        span_ids.add(span_id)

        label = _required_text(raw_span, "label")
        if label not in CANONICAL_LABELS:
            raise ValueError(f"India clinical span label must be canonical: {label}")
        span_metadata = raw_span.get("metadata")
        if not isinstance(span_metadata, Mapping):
            raise ValueError("India clinical span metadata must be a mapping")
        span_metadata = dict(span_metadata)
        if span_metadata.get("synthetic") is not True:
            raise ValueError("India clinical span must be marked synthetic")
        identifier_type = _required_text(span_metadata, "identifier_type")
        if identifier_type not in manifest.required_identifier_types:
            raise ValueError(
                f"India clinical span identifier type is not declared: {identifier_type}"
            )
        if not isinstance(span_metadata.get("direct_identifier"), bool):
            raise ValueError("India clinical span direct_identifier must be boolean")
        span_metadata["span_id"] = span_id
        normalized_rows.append({**raw_span, "metadata": span_metadata})

        address_group = str(span_metadata.get("address_group") or "")
        if identifier_type in {"street_address", "pin_code"}:
            if not address_group:
                raise ValueError("India address spans require an address_group")
            address_groups.setdefault(address_group, set()).add(identifier_type)

    spans = tuple(
        normalize_eval_spans(
            normalized_rows,
            default_language=language,
            source_text=text,
        )
    )
    for span in spans:
        if text[span.start : span.end] != span.text:
            raise ValueError(
                f"India clinical span text does not match offsets: {span.metadata['span_id']}"
            )
        _validate_india_synthetic_value(
            str(span.metadata["identifier_type"]), span.text
        )
    for address_group, identifier_types in address_groups.items():
        if identifier_types != {"street_address", "pin_code"}:
            raise ValueError(
                f"India address group must pair street and PIN spans: {address_group}"
            )

    return IndiaClinicalPHIRecord(
        fixture_id=fixture_id,
        document_id=document_id,
        language=language,
        languages=languages,
        scripts=scripts,
        text=text,
        gold_spans=spans,
        clinical_terms=clinical_terms,
        metadata=metadata,
    )


def _validate_india_synthetic_value(identifier_type: str, value: str) -> None:
    compact = re.sub(r"[\s-]", "", value)
    if identifier_type == "aadhaar":
        from openmed.core.pii_i18n import validate_aadhaar

        if not validate_aadhaar(value):
            raise ValueError("synthetic Aadhaar must pass the Verhoeff checksum")
    elif identifier_type == "abha" and not re.fullmatch(r"\d{14}", compact):
        raise ValueError("synthetic ABHA must contain 14 digits")
    elif identifier_type == "pan" and not re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", compact):
        raise ValueError("synthetic PAN must match the official field shape")
    elif identifier_type == "indian_phone" and not re.fullmatch(
        r"\+91[6-9]\d{9}", compact
    ):
        raise ValueError("synthetic India phone must use +91 and ten mobile digits")
    elif identifier_type == "pin_code" and not re.fullmatch(r"[1-9]\d{5}", compact):
        raise ValueError("synthetic India PIN must contain six digits")


def _contains_script(text: str, script: str) -> bool:
    ranges = _SCRIPT_RANGES.get(script)
    if ranges is None:
        return False
    return any(
        lower <= ord(character) <= upper
        for character in text
        for lower, upper in ranges
    )


def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _required_text_tuple(
    payload: Mapping[str, Any], field_name: str
) -> tuple[str, ...]:
    values = payload.get(field_name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field_name} values must be non-empty strings")
    return tuple(value.strip() for value in values)


def _required_int(payload: Mapping[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _required_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


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
    if source.dataset == INDIA_CLINICAL_PHI_CORPUS_ID:
        return load_india_clinical_phi_corpus()
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
    "INDIA_CLINICAL_PHI_CORPUS_ID",
    "INDIA_CLINICAL_PHI_SCHEMA_VERSION",
    "ClinicalPHIDatasetManifest",
    "ClinicalPHISource",
    "GateFamilyRequirement",
    "IndiaClinicalPHICorpus",
    "IndiaClinicalPHICorpusManifest",
    "IndiaClinicalPHIIdentity",
    "IndiaClinicalPHIIdentityAlias",
    "IndiaClinicalPHIRecord",
    "clinical_phi_manifest_hash",
    "load_clinical_phi_manifest",
    "load_india_clinical_phi_corpus",
    "resolve_clinical_phi_source",
    "validate_clinical_phi_manifest",
    "validate_india_clinical_phi_corpus",
    "validate_india_clinical_phi_manifest",
    "PUBLIC_SAMPLE_NOTES_CONFIG",
    "PUBLIC_SAMPLE_SPANS_CONFIG",
]
