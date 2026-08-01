"""Offline-first dataset contract for the DirectID tiny training family.

The module commits source references, coverage requirements, and hash-only
evidence. It deliberately does not bundle or persist corpus rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import (
    ACCOUNT_NUMBER,
    EMAIL,
    ID_NUM,
    ID_SUBTYPE_MRN,
    PHONE,
    SSN,
    normalize_label,
)
from openmed.training.directid import (
    DIRECTID_FAMILY,
    DIRECTID_TIER,
    DIRECTID_TINY_HEAD_CONTRACT,
    DirectIDHeadContract,
    validate_directid_contract,
)
from openmed.training.hard_negatives import (
    HARD_NEGATIVE_CATEGORIES,
    HardNegativeGenerator,
    count_hard_negatives,
    sample_hard_negatives,
)
from openmed.training.recipe import load_preset

DIRECTID_DATASET_SCHEMA_VERSION = "openmed.training.directid_dataset.v1"
DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION = (
    "openmed.training.directid_dataset_evidence.v1"
)
DIRECTID_DATASET_MANIFEST_ID = "openmed-directid-tiny-dataset-v1"
DIRECTID_DATASET_MANIFEST_REF = (
    "openmed.training.directid_dataset:load_directid_dataset_manifest"
    f"@{DIRECTID_DATASET_MANIFEST_ID}"
)

PUBLIC_PERMISSIVE = "public_permissive"
SYNTHETIC = "synthetic"
USER_SUPPLIED_RESTRICTED = "user_supplied_restricted"
DIRECTID_SOURCE_CLASSES = (
    PUBLIC_PERMISSIVE,
    SYNTHETIC,
    USER_SUPPLIED_RESTRICTED,
)
DIRECTID_SPLITS = ("train", "validation", "test")

NEMOTRON_PII_SOURCE_ID = "nemotron_pii_cc_by_4_0"
SYNTHETIC_SOURCE_ID = "openmed_directid_synthetic"
RESTRICTED_SOURCE_ID = "user_supplied_restricted_directid"

_NEMOTRON_PII_REVISION = "5d3d58ac206e8b286080c45f640db88d51886498"
_NEMOTRON_PII_LABELS = (
    ID_NUM,
    SSN,
    ACCOUNT_NUMBER,
    EMAIL,
    PHONE,
)
_NEMOTRON_PII_ID_SUBTYPES = (ID_SUBTYPE_MRN,)
_PERMISSIVE_LICENSES = frozenset({"Apache-2.0", "CC-BY-4.0", "CC0-1.0"})
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class DirectIDDatasetError(ValueError):
    """Raised when a DirectID dataset manifest or split fails closed."""


@dataclass(frozen=True)
class DirectIDDatasetSource:
    """One reference-only source in the DirectID dataset plan."""

    source_id: str
    dataset: str
    source_class: str
    role: str
    license_id: str
    source_url: str
    revision: str
    redistribution: str
    provenance: str
    labels: tuple[str, ...]
    id_subtypes: tuple[str, ...]
    splits: tuple[str, ...]
    required: bool
    synthetic: bool
    local_only: bool = True
    bundled_payload: bool = False
    content_hash_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return stable source metadata without corpus rows or local paths."""

        return {
            "bundled_payload": self.bundled_payload,
            "content_hash_required": self.content_hash_required,
            "dataset": self.dataset,
            "id_subtypes": list(self.id_subtypes),
            "labels": list(self.labels),
            "license_id": self.license_id,
            "local_only": self.local_only,
            "provenance": self.provenance,
            "redistribution": self.redistribution,
            "required": self.required,
            "revision": self.revision,
            "role": self.role,
            "source_class": self.source_class,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "splits": list(self.splits),
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class DirectIDSyntheticSettings:
    """Pinned generation settings for positive and hard-negative fixtures."""

    schema_version: str
    seed: int
    records_per_label: int
    records_per_id_subtype: int
    positive_generator_refs: tuple[str, ...]
    hard_negative_generator_ref: str
    hard_negative_categories: tuple[str, ...]
    contains_real_phi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contains_real_phi": self.contains_real_phi,
            "hard_negative_categories": list(self.hard_negative_categories),
            "hard_negative_generator_ref": self.hard_negative_generator_ref,
            "positive_generator_refs": list(self.positive_generator_refs),
            "records_per_id_subtype": self.records_per_id_subtype,
            "records_per_label": self.records_per_label,
            "schema_version": self.schema_version,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class DirectIDSplitManifest:
    """Coverage and hard-negative requirements for one dataset split."""

    name: str
    source_ids: tuple[str, ...]
    optional_source_ids: tuple[str, ...]
    required_labels: tuple[str, ...]
    required_id_subtypes: tuple[str, ...]
    hard_negatives_required: bool
    minimum_hard_negatives_per_batch: int
    hard_negative_categories: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_negative_categories": list(self.hard_negative_categories),
            "hard_negatives_required": self.hard_negatives_required,
            "minimum_hard_negatives_per_batch": (self.minimum_hard_negatives_per_batch),
            "name": self.name,
            "optional_source_ids": list(self.optional_source_ids),
            "required_id_subtypes": list(self.required_id_subtypes),
            "required_labels": list(self.required_labels),
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True)
class DirectIDDatasetManifest:
    """Versioned data assembly contract consumed by DirectID training."""

    manifest_id: str
    schema_version: str
    contract_ref: str
    family: str
    tier: str
    sources: tuple[DirectIDDatasetSource, ...]
    splits: tuple[DirectIDSplitManifest, ...]
    synthetic_settings: DirectIDSyntheticSettings

    def source(self, source_id: str) -> DirectIDDatasetSource:
        """Return a source by stable identifier."""

        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"unknown DirectID dataset source: {source_id}")

    def split(self, name: str) -> DirectIDSplitManifest:
        """Return a split by name."""

        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(f"unknown DirectID dataset split: {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_ref": self.contract_ref,
            "family": self.family,
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "sources": [source.to_dict() for source in self.sources],
            "splits": [split.to_dict() for split in self.splits],
            "synthetic_settings": self.synthetic_settings.to_dict(),
            "tier": self.tier,
        }


@dataclass(frozen=True)
class DirectIDSplitEvidence:
    """Hash-only coverage evidence for one validated split."""

    split: str
    dataset_hash: str
    record_count: int
    positive_record_count: int
    hard_negative_count: int
    label_counts: Mapping[str, int]
    id_subtype_counts: Mapping[str, int]
    hard_negative_category_counts: Mapping[str, int]
    source_ids: tuple[str, ...]
    source_record_counts: Mapping[str, int]
    source_hashes: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_hash": self.dataset_hash,
            "hard_negative_category_counts": dict(
                sorted(self.hard_negative_category_counts.items())
            ),
            "hard_negative_count": self.hard_negative_count,
            "id_subtype_counts": dict(sorted(self.id_subtype_counts.items())),
            "label_counts": dict(sorted(self.label_counts.items())),
            "positive_record_count": self.positive_record_count,
            "record_count": self.record_count,
            "source_ids": list(self.source_ids),
            "source_record_counts": dict(sorted(self.source_record_counts.items())),
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "split": self.split,
        }


@dataclass(frozen=True)
class DirectIDBatchEvidence:
    """Non-sensitive proof that one batch contains required negatives."""

    split: str
    record_count: int
    hard_negative_count: int
    hard_negative_categories: tuple[str, ...]


def load_directid_dataset_manifest() -> DirectIDDatasetManifest:
    """Return the offline-first DirectID dataset manifest."""

    contract = validate_directid_contract()
    all_splits = DIRECTID_SPLITS
    manifest = DirectIDDatasetManifest(
        manifest_id=DIRECTID_DATASET_MANIFEST_ID,
        schema_version=DIRECTID_DATASET_SCHEMA_VERSION,
        contract_ref=contract.contract_ref,
        family=contract.family,
        tier=contract.tier,
        sources=(
            DirectIDDatasetSource(
                source_id=NEMOTRON_PII_SOURCE_ID,
                dataset="nvidia/Nemotron-PII",
                source_class=PUBLIC_PERMISSIVE,
                role="public_permissive_training_and_test_reference",
                license_id="CC-BY-4.0",
                source_url=(
                    "https://huggingface.co/datasets/nvidia/Nemotron-PII/tree/"
                    + _NEMOTRON_PII_REVISION
                ),
                revision=_NEMOTRON_PII_REVISION,
                redistribution="reference-only; no corpus rows committed",
                provenance=(
                    "Pinned upstream synthetic dataset revision; callers provide "
                    "a locally cached copy and its SHA-256 digest."
                ),
                labels=_NEMOTRON_PII_LABELS,
                id_subtypes=_NEMOTRON_PII_ID_SUBTYPES,
                splits=("train", "test"),
                required=True,
                synthetic=True,
            ),
            DirectIDDatasetSource(
                source_id=SYNTHETIC_SOURCE_ID,
                dataset="openmed-directid-generated-v1",
                source_class=SYNTHETIC,
                role="offline_positive_and_hard_negative_generation",
                license_id="Apache-2.0",
                source_url="openmed/training/directid_dataset.py",
                revision=DIRECTID_DATASET_SCHEMA_VERSION,
                redistribution="generated synthetic records; no rows committed",
                provenance=(
                    "Deterministic local generators pinned by the synthetic "
                    "settings hash."
                ),
                labels=contract.labels,
                id_subtypes=contract.id_subtypes,
                splits=all_splits,
                required=True,
                synthetic=True,
            ),
            DirectIDDatasetSource(
                source_id=RESTRICTED_SOURCE_ID,
                dataset="user-supplied-restricted-directid",
                source_class=USER_SUPPLIED_RESTRICTED,
                role="optional_local_training_or_evaluation",
                license_id="user-supplied",
                source_url="",
                revision="caller-provided",
                redistribution="never bundled; user-supplied reference only",
                provenance=(
                    "Caller-controlled local records; only a caller-provided "
                    "SHA-256 digest and aggregate coverage may enter evidence."
                ),
                labels=contract.labels,
                id_subtypes=contract.id_subtypes,
                splits=all_splits,
                required=False,
                synthetic=False,
            ),
        ),
        splits=tuple(
            DirectIDSplitManifest(
                name=split_name,
                source_ids=(
                    (NEMOTRON_PII_SOURCE_ID, SYNTHETIC_SOURCE_ID)
                    if split_name in {"train", "test"}
                    else (SYNTHETIC_SOURCE_ID,)
                ),
                optional_source_ids=(RESTRICTED_SOURCE_ID,),
                required_labels=contract.labels,
                required_id_subtypes=contract.id_subtypes,
                hard_negatives_required=True,
                minimum_hard_negatives_per_batch=1,
                hard_negative_categories=HARD_NEGATIVE_CATEGORIES,
            )
            for split_name in all_splits
        ),
        synthetic_settings=DirectIDSyntheticSettings(
            schema_version="openmed.training.directid_synthetic.v1",
            seed=3801,
            records_per_label=256,
            records_per_id_subtype=256,
            positive_generator_refs=(
                "openmed.core.anonymizer.providers.clinical_ids",
                "openmed.training.synthetic.locale_phi",
            ),
            hard_negative_generator_ref=(
                "openmed.training.hard_negatives:HardNegativeGenerator"
            ),
            hard_negative_categories=HARD_NEGATIVE_CATEGORIES,
        ),
    )
    validate_directid_dataset_manifest(manifest, contract=contract)
    return manifest


def validate_directid_dataset_manifest(
    manifest: DirectIDDatasetManifest,
    *,
    contract: DirectIDHeadContract = DIRECTID_TINY_HEAD_CONTRACT,
) -> None:
    """Fail closed on source policy, contract coverage, or split drift.

    Args:
        manifest: Dataset assembly contract to validate.
        contract: DirectID head contract that the dataset must cover.

    Raises:
        DirectIDDatasetError: If source policy or coverage is incomplete.
    """

    validate_directid_contract(contract)
    errors: list[str] = []
    if manifest.manifest_id != DIRECTID_DATASET_MANIFEST_ID:
        errors.append("manifest_id must name the DirectID tiny dataset")
    if manifest.schema_version != DIRECTID_DATASET_SCHEMA_VERSION:
        errors.append("schema_version does not match the DirectID dataset schema")
    if manifest.contract_ref != contract.contract_ref:
        errors.append("contract_ref must match the DirectID head contract")
    if manifest.family != DIRECTID_FAMILY or manifest.tier != DIRECTID_TIER:
        errors.append("manifest family and tier must target DirectID tiny")

    source_ids = [source.source_id for source in manifest.sources]
    if len(source_ids) != len(set(source_ids)):
        errors.append("source ids must be unique")
    source_by_id = {source.source_id: source for source in manifest.sources}
    source_classes = {source.source_class for source in manifest.sources}
    missing_classes = sorted(set(DIRECTID_SOURCE_CLASSES) - source_classes)
    if missing_classes:
        errors.append("missing source class(es): " + ", ".join(missing_classes))

    for source in manifest.sources:
        _validate_source(source, contract, errors)

    split_names = [split.name for split in manifest.splits]
    if len(split_names) != len(set(split_names)):
        errors.append("split names must be unique")
    if set(split_names) != set(DIRECTID_SPLITS):
        errors.append("splits must declare train, validation, and test")
    for split in manifest.splits:
        _validate_split(split, source_by_id, contract, errors)

    settings = manifest.synthetic_settings
    if settings.contains_real_phi:
        errors.append("synthetic settings must declare contains_real_phi=false")
    if settings.seed < 0:
        errors.append("synthetic seed must be non-negative")
    if settings.records_per_label <= 0 or settings.records_per_id_subtype <= 0:
        errors.append("synthetic coverage counts must be positive")
    if set(settings.hard_negative_categories) != set(HARD_NEGATIVE_CATEGORIES):
        errors.append("synthetic settings must include every hard-negative category")
    if not settings.positive_generator_refs:
        errors.append("synthetic settings must pin positive generators")
    if not settings.hard_negative_generator_ref:
        errors.append("synthetic settings must pin the hard-negative generator")

    if errors:
        raise DirectIDDatasetError("; ".join(errors))


def directid_dataset_manifest_hash(
    manifest: DirectIDDatasetManifest | None = None,
) -> str:
    """Return a deterministic SHA-256 digest of manifest-only metadata.

    Args:
        manifest: Optional manifest override.

    Returns:
        A ``sha256:``-prefixed digest.
    """

    active = manifest or load_directid_dataset_manifest()
    validate_directid_dataset_manifest(active)
    return _payload_hash(active.to_dict())


def directid_synthetic_settings_hash(
    manifest: DirectIDDatasetManifest | None = None,
) -> str:
    """Return the pinned generation-settings digest.

    Args:
        manifest: Optional manifest override.

    Returns:
        A ``sha256:``-prefixed digest.
    """

    active = manifest or load_directid_dataset_manifest()
    validate_directid_dataset_manifest(active)
    return _payload_hash(active.synthetic_settings.to_dict())


def directid_source_manifest_hash(source: DirectIDDatasetSource) -> str:
    """Return a reference hash for one source without reading corpus rows.

    Args:
        source: Reference-only source metadata.

    Returns:
        A ``sha256:``-prefixed digest.
    """

    return _payload_hash(source.to_dict())


def directid_records_hash(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash local records without returning, logging, or persisting their text.

    Args:
        records: JSON-serializable local rows.

    Returns:
        A ``sha256:``-prefixed digest.

    Raises:
        DirectIDDatasetError: If a record cannot be hashed deterministically.
    """

    rows = [dict(record) for record in records]
    try:
        return _payload_hash(rows)
    except (TypeError, ValueError) as exc:
        raise DirectIDDatasetError(
            "DirectID records must be JSON-serializable for deterministic hashing"
        ) from exc


def generate_directid_hard_negatives(
    split_name: str,
    *,
    manifest: DirectIDDatasetManifest | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generate one deterministic fixture for every required negative category.

    Args:
        split_name: One of ``train``, ``validation``, or ``test``.
        manifest: Optional manifest override.

    Returns:
        Synthetic training items emitted by the hard-negative harness.
    """

    active = manifest or load_directid_dataset_manifest()
    validate_directid_dataset_manifest(active)
    active.split(split_name)
    split_seed = active.synthetic_settings.seed + DIRECTID_SPLITS.index(split_name)
    generator = HardNegativeGenerator(seed=split_seed)
    settings_hash = directid_synthetic_settings_hash(active)
    rows: list[dict[str, Any]] = []
    for example in generator.generate_all_categories():
        row = example.to_training_item()
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "contains_real_phi": False,
                "generation_settings_hash": settings_hash,
                "synthetic": True,
            }
        )
        row.update(
            {
                "metadata": metadata,
                "source_id": SYNTHETIC_SOURCE_ID,
                "split": split_name,
            }
        )
        rows.append(row)
    return tuple(rows)


def validate_directid_split_records(
    split_name: str,
    records: Sequence[Mapping[str, Any]],
    *,
    manifest: DirectIDDatasetManifest | None = None,
) -> DirectIDSplitEvidence:
    """Validate label, subtype, and negative coverage without emitting text.

    Args:
        split_name: Dataset split being validated.
        records: Local records using labels and ``source_id`` metadata.
        manifest: Optional manifest override.

    Returns:
        Aggregate counts and hashes with no raw record content.

    Raises:
        DirectIDDatasetError: If any required coverage or provenance is absent.
    """

    active = manifest or load_directid_dataset_manifest()
    validate_directid_dataset_manifest(active)
    split = active.split(split_name)
    allowed_sources = set(split.source_ids) | set(split.optional_source_ids)
    label_counts: dict[str, int] = {}
    subtype_counts: dict[str, int] = {}
    hard_negative_counts: dict[str, int] = {}
    records_by_source: dict[str, list[Mapping[str, Any]]] = {}
    positive_record_count = 0

    for record in records:
        if not isinstance(record, Mapping):
            raise DirectIDDatasetError(f"{split_name} records must be mappings")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise DirectIDDatasetError(
                f"{split_name} records must declare source_id for provenance"
            )
        if source_id not in allowed_sources:
            raise DirectIDDatasetError(
                f"{split_name} record references unknown source_id"
            )
        records_by_source.setdefault(source_id, []).append(record)

        if bool(record.get("is_hard_negative")):
            category = record.get("hard_negative_category")
            if (
                not isinstance(category, str)
                or category not in HARD_NEGATIVE_CATEGORIES
            ):
                raise DirectIDDatasetError(
                    f"{split_name} hard negative has an invalid category"
                )
            hard_negative_counts[category] = hard_negative_counts.get(category, 0) + 1
            continue

        positive_record_count += 1
        for label in _labels_from_record(record):
            canonical = normalize_label(label)
            if canonical not in split.required_labels:
                raise DirectIDDatasetError(
                    f"{split_name} contains non-DirectID label {canonical}"
                )
            label_counts[canonical] = label_counts.get(canonical, 0) + 1
        for subtype in _id_subtypes_from_record(record):
            if subtype not in split.required_id_subtypes:
                raise DirectIDDatasetError(
                    f"{split_name} contains unsupported ID subtype {subtype}"
                )
            subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

    missing_labels = sorted(set(split.required_labels) - set(label_counts))
    missing_subtypes = sorted(set(split.required_id_subtypes) - set(subtype_counts))
    missing_negative_categories = sorted(
        set(split.hard_negative_categories) - set(hard_negative_counts)
    )
    missing_required_sources = sorted(set(split.source_ids) - set(records_by_source))
    errors: list[str] = []
    if missing_labels:
        errors.append("missing DirectID label(s): " + ", ".join(missing_labels))
    if missing_subtypes:
        errors.append(
            "missing critical identifier subtype(s): " + ", ".join(missing_subtypes)
        )
    if missing_negative_categories:
        errors.append(
            "missing hard-negative category(s): "
            + ", ".join(missing_negative_categories)
        )
    if missing_required_sources:
        errors.append(
            "missing required source(s): " + ", ".join(missing_required_sources)
        )
    if errors:
        raise DirectIDDatasetError(f"{split_name}: " + "; ".join(errors))

    dataset_hash = directid_records_hash(records)
    return DirectIDSplitEvidence(
        split=split_name,
        dataset_hash=dataset_hash,
        record_count=len(records),
        positive_record_count=positive_record_count,
        hard_negative_count=sum(hard_negative_counts.values()),
        label_counts=label_counts,
        id_subtype_counts=subtype_counts,
        hard_negative_category_counts=hard_negative_counts,
        source_ids=tuple(sorted(records_by_source)),
        source_record_counts={
            source_id: len(source_records)
            for source_id, source_records in records_by_source.items()
        },
        source_hashes={
            source_id: directid_records_hash(source_records)
            for source_id, source_records in records_by_source.items()
        },
    )


def validate_directid_batch(
    split_name: str,
    batch: Sequence[Mapping[str, Any]],
    *,
    manifest: DirectIDDatasetManifest | None = None,
) -> DirectIDBatchEvidence:
    """Fail closed when a DirectID batch omits required hard negatives.

    Args:
        split_name: Dataset split associated with the batch.
        batch: Candidate training batch.
        manifest: Optional manifest override.

    Returns:
        Aggregate hard-negative batch evidence.

    Raises:
        DirectIDDatasetError: If the hard-negative minimum is not satisfied.
    """

    active = manifest or load_directid_dataset_manifest()
    validate_directid_dataset_manifest(active)
    split = active.split(split_name)
    allowed_sources = set(split.source_ids) | set(split.optional_source_ids)
    hard_negative_count = count_hard_negatives(batch)
    if hard_negative_count < split.minimum_hard_negatives_per_batch:
        raise DirectIDDatasetError(
            f"{split_name} batch requires at least "
            f"{split.minimum_hard_negatives_per_batch} hard negative(s)"
        )
    categories: set[str] = set()
    for record in batch:
        if not bool(record.get("is_hard_negative")):
            continue
        category = record.get("hard_negative_category")
        if not isinstance(category, str) or category not in HARD_NEGATIVE_CATEGORIES:
            raise DirectIDDatasetError(
                f"{split_name} batch contains an invalid hard-negative category"
            )
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or source_id not in allowed_sources:
            raise DirectIDDatasetError(
                f"{split_name} hard negative must declare a known source_id"
            )
        categories.add(category)
    return DirectIDBatchEvidence(
        split=split_name,
        record_count=len(batch),
        hard_negative_count=hard_negative_count,
        hard_negative_categories=tuple(sorted(categories)),
    )


def prepare_directid_batch(
    split_name: str,
    batch: Sequence[Mapping[str, Any]],
    *,
    manifest: DirectIDDatasetManifest | None = None,
) -> tuple[dict[str, Any], ...]:
    """Apply the existing hard-negative sampler and validate its output.

    Args:
        split_name: Dataset split associated with the batch.
        batch: Candidate batch before hard-negative sampling.
        manifest: Optional manifest override.

    Returns:
        A copied batch satisfying the split's hard-negative minimum.

    Raises:
        DirectIDDatasetError: If the preset or sampled batch violates the manifest.
    """

    active = manifest or load_directid_dataset_manifest()
    validate_directid_dataset_manifest(active)
    split = active.split(split_name)
    recipe = load_preset("tiny_distill")
    split_seed = active.synthetic_settings.seed + DIRECTID_SPLITS.index(split_name)
    sampled = sample_hard_negatives(
        batch,
        recipe_config=recipe,
        seed=split_seed,
        min_hard_negatives_per_batch=split.minimum_hard_negatives_per_batch,
    )
    settings_hash = directid_synthetic_settings_hash(active)
    prepared: list[dict[str, Any]] = []
    for record in sampled:
        item = dict(record)
        if bool(item.get("is_hard_negative")) and not item.get("source_id"):
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "contains_real_phi": False,
                    "generation_settings_hash": settings_hash,
                    "synthetic": True,
                }
            )
            item.update(
                {
                    "metadata": metadata,
                    "source_id": SYNTHETIC_SOURCE_ID,
                    "split": split_name,
                }
            )
        prepared.append(item)
    validate_directid_batch(split_name, prepared, manifest=active)
    return tuple(prepared)


def build_directid_dataset_evidence(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    manifest: DirectIDDatasetManifest | None = None,
) -> dict[str, Any]:
    """Build downstream gate evidence containing hashes and aggregates only.

    Args:
        records_by_split: Local records keyed by every required split.
        manifest: Optional manifest override.

    Returns:
        Provenance, generation settings, coverage counts, and dataset hashes.

    Raises:
        DirectIDDatasetError: If splits or required coverage are incomplete.
    """

    active = manifest or load_directid_dataset_manifest()
    validate_directid_dataset_manifest(active)
    missing_splits = sorted(set(DIRECTID_SPLITS) - set(records_by_split))
    extra_splits = sorted(set(records_by_split) - set(DIRECTID_SPLITS))
    if missing_splits or extra_splits:
        details: list[str] = []
        if missing_splits:
            details.append("missing split(s): " + ", ".join(missing_splits))
        if extra_splits:
            details.append("unknown split(s): " + ", ".join(extra_splits))
        raise DirectIDDatasetError("; ".join(details))

    split_evidence = {
        split_name: validate_directid_split_records(
            split_name,
            records_by_split[split_name],
            manifest=active,
        ).to_dict()
        for split_name in DIRECTID_SPLITS
    }
    source_provenance = [
        {
            "content_hash_required": source.content_hash_required,
            "license_id": source.license_id,
            "provenance": source.provenance,
            "required": source.required,
            "revision": source.revision,
            "source_class": source.source_class,
            "source_id": source.source_id,
            "source_manifest_hash": directid_source_manifest_hash(source),
        }
        for source in active.sources
    ]
    return {
        "contract_ref": active.contract_ref,
        "family": active.family,
        "manifest_hash": directid_dataset_manifest_hash(active),
        "manifest_id": active.manifest_id,
        "raw_records_persisted": False,
        "schema_version": DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION,
        "source_provenance": source_provenance,
        "splits": split_evidence,
        "synthetic_generation": {
            "settings": active.synthetic_settings.to_dict(),
            "settings_hash": directid_synthetic_settings_hash(active),
        },
        "tier": active.tier,
    }


def _validate_source(
    source: DirectIDDatasetSource,
    contract: DirectIDHeadContract,
    errors: list[str],
) -> None:
    if source.source_class not in DIRECTID_SOURCE_CLASSES:
        errors.append(f"{source.source_id} has an unknown source class")
    if source.bundled_payload:
        errors.append(f"{source.source_id} must not bundle corpus payloads")
    if not source.local_only:
        errors.append(f"{source.source_id} must be local-only at assembly time")
    if not source.content_hash_required:
        errors.append(f"{source.source_id} must require a content hash")
    if not source.splits or not set(source.splits) <= set(DIRECTID_SPLITS):
        errors.append(f"{source.source_id} declares invalid splits")

    labels = tuple(normalize_label(label) for label in source.labels)
    unknown_labels = sorted(set(labels) - set(contract.labels))
    if unknown_labels:
        errors.append(
            f"{source.source_id} contains non-DirectID label(s): "
            + ", ".join(unknown_labels)
        )
    unknown_subtypes = sorted(set(source.id_subtypes) - set(contract.id_subtypes))
    if unknown_subtypes:
        errors.append(
            f"{source.source_id} contains unsupported subtype(s): "
            + ", ".join(unknown_subtypes)
        )
    if source.source_class == PUBLIC_PERMISSIVE:
        if source.license_id not in _PERMISSIVE_LICENSES:
            errors.append(f"{source.source_id} must use a permissive license")
        if not source.source_url or not source.revision:
            errors.append(f"{source.source_id} must pin source URL and revision")
    if source.source_class == SYNTHETIC and not source.synthetic:
        errors.append(f"{source.source_id} must be marked synthetic")
    if source.source_class == USER_SUPPLIED_RESTRICTED:
        if source.required:
            errors.append(f"{source.source_id} restricted data must remain optional")
        if "never bundled" not in source.redistribution:
            errors.append(f"{source.source_id} must prohibit restricted bundling")


def _validate_split(
    split: DirectIDSplitManifest,
    source_by_id: Mapping[str, DirectIDDatasetSource],
    contract: DirectIDHeadContract,
    errors: list[str],
) -> None:
    referenced_ids = split.source_ids + split.optional_source_ids
    unknown_sources = sorted(set(referenced_ids) - set(source_by_id))
    if unknown_sources:
        errors.append(
            f"{split.name} references unknown source(s): " + ", ".join(unknown_sources)
        )
        return
    if len(referenced_ids) != len(set(referenced_ids)):
        errors.append(f"{split.name} source references must be unique")
    for source_id in referenced_ids:
        if split.name not in source_by_id[source_id].splits:
            errors.append(f"{source_id} does not declare split {split.name}")

    if set(split.required_labels) != set(contract.labels):
        errors.append(f"{split.name} must require every DirectID contract label")
    if set(split.required_id_subtypes) != set(contract.id_subtypes):
        errors.append(f"{split.name} must require every critical ID subtype")
    covered_labels = {
        normalize_label(label)
        for source_id in split.source_ids
        for label in source_by_id[source_id].labels
    }
    missing_labels = sorted(set(contract.labels) - covered_labels)
    if missing_labels:
        errors.append(
            f"{split.name} source plan missing DirectID label(s): "
            + ", ".join(missing_labels)
        )
    covered_subtypes = {
        subtype
        for source_id in split.source_ids
        for subtype in source_by_id[source_id].id_subtypes
    }
    missing_subtypes = sorted(set(contract.id_subtypes) - covered_subtypes)
    if missing_subtypes:
        errors.append(
            f"{split.name} source plan missing critical identifier subtype(s): "
            + ", ".join(missing_subtypes)
        )
    if split.hard_negatives_required is not True:
        errors.append(f"{split.name} must require hard negatives")
    if split.minimum_hard_negatives_per_batch <= 0:
        errors.append(f"{split.name} must require a positive hard-negative count")
    if set(split.hard_negative_categories) != set(HARD_NEGATIVE_CATEGORIES):
        errors.append(f"{split.name} must require every hard-negative category")


def _labels_from_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    raw_labels = record.get("labels")
    if isinstance(raw_labels, str):
        values.append(raw_labels)
    elif isinstance(raw_labels, Sequence):
        for item in raw_labels:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("label"), str):
                values.append(item["label"])
    for field_name in ("spans", "entities"):
        raw_spans = record.get(field_name)
        if isinstance(raw_spans, Sequence) and not isinstance(raw_spans, str):
            for span in raw_spans:
                if isinstance(span, Mapping) and isinstance(span.get("label"), str):
                    values.append(span["label"])
    return tuple(values)


def _id_subtypes_from_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    _append_subtype_value(values, record.get("id_subtype"))
    _append_subtype_value(values, record.get("id_subtypes"))
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        _append_subtype_value(values, metadata.get("id_subtype"))
        _append_subtype_value(values, metadata.get("id_subtypes"))
    for field_name in ("labels", "spans", "entities"):
        items = record.get(field_name)
        if isinstance(items, Sequence) and not isinstance(items, str):
            for item in items:
                if isinstance(item, Mapping):
                    _append_subtype_value(values, item.get("id_subtype"))
    return tuple(values)


def _append_subtype_value(values: list[str], raw: Any) -> None:
    if isinstance(raw, str):
        values.append(raw)
    elif isinstance(raw, Sequence):
        values.extend(item for item in raw if isinstance(item, str))


def _payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    if _SHA256_RE.fullmatch(digest) is None:  # pragma: no cover - defensive
        raise RuntimeError("invalid SHA-256 digest")
    return digest


__all__ = [
    "DIRECTID_DATASET_EVIDENCE_SCHEMA_VERSION",
    "DIRECTID_DATASET_MANIFEST_ID",
    "DIRECTID_DATASET_MANIFEST_REF",
    "DIRECTID_DATASET_SCHEMA_VERSION",
    "DIRECTID_SOURCE_CLASSES",
    "DIRECTID_SPLITS",
    "NEMOTRON_PII_SOURCE_ID",
    "PUBLIC_PERMISSIVE",
    "RESTRICTED_SOURCE_ID",
    "SYNTHETIC",
    "SYNTHETIC_SOURCE_ID",
    "USER_SUPPLIED_RESTRICTED",
    "DirectIDBatchEvidence",
    "DirectIDDatasetError",
    "DirectIDDatasetManifest",
    "DirectIDDatasetSource",
    "DirectIDSplitEvidence",
    "DirectIDSplitManifest",
    "DirectIDSyntheticSettings",
    "build_directid_dataset_evidence",
    "directid_dataset_manifest_hash",
    "directid_records_hash",
    "directid_source_manifest_hash",
    "directid_synthetic_settings_hash",
    "generate_directid_hard_negatives",
    "load_directid_dataset_manifest",
    "prepare_directid_batch",
    "validate_directid_batch",
    "validate_directid_dataset_manifest",
    "validate_directid_split_records",
]
