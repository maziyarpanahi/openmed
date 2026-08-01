from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from openmed.core.labels import ID_NUM, ID_SUBTYPE_NPI, SSN
from openmed.training import (
    DIRECTID_DATASET_MANIFEST_REF,
    DIRECTID_TINY_HEAD_CONTRACT,
    DirectIDDatasetError,
    build_directid_dataset_evidence,
    directid_dataset_manifest_hash,
    directid_synthetic_settings_hash,
    generate_directid_hard_negatives,
    load_directid_dataset_manifest,
    load_preset,
    prepare_directid_batch,
    validate_directid_batch,
    validate_directid_dataset_manifest,
    validate_directid_split_records,
)
from openmed.training.directid_dataset import (
    DIRECTID_SOURCE_CLASSES,
    DIRECTID_SPLITS,
    NEMOTRON_PII_SOURCE_ID,
    PUBLIC_PERMISSIVE,
    USER_SUPPLIED_RESTRICTED,
)
from openmed.training.hard_negatives import HARD_NEGATIVE_CATEGORIES


def _synthetic_split_records(split_name: str) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = [
        {
            "fixture_id": f"synthetic-{split_name}-{label.lower()}",
            "labels": [label],
            "source_id": "openmed_directid_synthetic",
            "split": split_name,
            "synthetic": True,
        }
        for label in DIRECTID_TINY_HEAD_CONTRACT.labels
    ]
    records.extend(
        {
            "fixture_id": f"synthetic-{split_name}-{subtype}",
            "id_subtype": subtype,
            "labels": [ID_NUM],
            "source_id": "openmed_directid_synthetic",
            "split": split_name,
            "synthetic": True,
        }
        for subtype in DIRECTID_TINY_HEAD_CONTRACT.id_subtypes
    )
    records.extend(generate_directid_hard_negatives(split_name))
    if split_name in {"train", "test"}:
        records.append(
            {
                "fixture_id": f"synthetic-{split_name}-public-source-reference",
                "labels": [ID_NUM],
                "source_id": NEMOTRON_PII_SOURCE_ID,
                "split": split_name,
                "synthetic": True,
                "test_fixture": True,
            }
        )
    return tuple(records)


def test_manifest_covers_source_classes_without_bundled_restricted_data() -> None:
    manifest = load_directid_dataset_manifest()

    assert {source.source_class for source in manifest.sources} == set(
        DIRECTID_SOURCE_CLASSES
    )
    assert all(source.local_only for source in manifest.sources)
    assert all(not source.bundled_payload for source in manifest.sources)

    public = next(
        source
        for source in manifest.sources
        if source.source_class == PUBLIC_PERMISSIVE
    )
    assert public.license_id == "CC-BY-4.0"
    assert len(public.revision) == 40
    assert public.synthetic is True

    restricted = next(
        source
        for source in manifest.sources
        if source.source_class == USER_SUPPLIED_RESTRICTED
    )
    assert restricted.required is False
    assert restricted.source_url == ""
    assert "never bundled" in restricted.redistribution
    assert restricted.content_hash_required is True


def test_every_split_covers_contract_labels_subtypes_and_negative_categories() -> None:
    manifest = load_directid_dataset_manifest()

    assert tuple(split.name for split in manifest.splits) == DIRECTID_SPLITS
    for split in manifest.splits:
        assert set(split.required_labels) == set(DIRECTID_TINY_HEAD_CONTRACT.labels)
        assert set(split.required_id_subtypes) == set(
            DIRECTID_TINY_HEAD_CONTRACT.id_subtypes
        )
        assert split.hard_negatives_required is True
        assert split.minimum_hard_negatives_per_batch > 0
        assert set(split.hard_negative_categories) == set(HARD_NEGATIVE_CATEGORIES)


def test_manifest_fails_closed_when_a_critical_identifier_class_is_absent() -> None:
    manifest = load_directid_dataset_manifest()
    synthetic = manifest.source("openmed_directid_synthetic")
    without_npi = replace(
        synthetic,
        id_subtypes=tuple(
            subtype for subtype in synthetic.id_subtypes if subtype != ID_SUBTYPE_NPI
        ),
    )
    broken = replace(
        manifest,
        sources=tuple(
            without_npi if source.source_id == without_npi.source_id else source
            for source in manifest.sources
        ),
    )

    with pytest.raises(DirectIDDatasetError, match="npi"):
        validate_directid_dataset_manifest(broken)


def test_split_coverage_fails_closed_for_missing_label_or_subtype() -> None:
    records = _synthetic_split_records("train")
    without_ssn = tuple(record for record in records if record.get("labels") != [SSN])
    without_npi = tuple(
        record for record in records if record.get("id_subtype") != ID_SUBTYPE_NPI
    )
    without_one_negative_category = tuple(
        record
        for record in records
        if record.get("hard_negative_category") != HARD_NEGATIVE_CATEGORIES[-1]
    )

    with pytest.raises(DirectIDDatasetError, match="SSN"):
        validate_directid_split_records("train", without_ssn)
    with pytest.raises(DirectIDDatasetError, match="npi"):
        validate_directid_split_records("train", without_npi)
    with pytest.raises(DirectIDDatasetError, match="hard-negative category"):
        validate_directid_split_records("train", without_one_negative_category)


@pytest.mark.parametrize("split_name", DIRECTID_SPLITS)
def test_existing_sampler_supplies_and_validates_each_split_batch(
    split_name: str,
) -> None:
    clean_batch = ({"fixture_id": "synthetic-clean", "labels": [ID_NUM]},)

    with pytest.raises(DirectIDDatasetError, match="hard negative"):
        validate_directid_batch(split_name, clean_batch)

    prepared = prepare_directid_batch(split_name, clean_batch)
    evidence = validate_directid_batch(split_name, prepared)

    assert len(prepared) == 2
    assert evidence.hard_negative_count == 1
    assert set(evidence.hard_negative_categories) <= set(HARD_NEGATIVE_CATEGORIES)
    assert prepared[-1]["source_id"] == "openmed_directid_synthetic"
    assert prepared[-1]["metadata"]["contains_real_phi"] is False


def test_split_evidence_records_hashes_and_aggregate_coverage_only() -> None:
    records_by_split = {
        split_name: _synthetic_split_records(split_name)
        for split_name in DIRECTID_SPLITS
    }

    evidence = build_directid_dataset_evidence(records_by_split)
    serialized = json.dumps(evidence, sort_keys=True)

    assert evidence["manifest_hash"] == directid_dataset_manifest_hash()
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        evidence["synthetic_generation"]["settings_hash"],
    )
    assert (
        directid_synthetic_settings_hash()
        == evidence["synthetic_generation"]["settings_hash"]
    )
    assert evidence["raw_records_persisted"] is False
    assert "Synthetic checksum fixture" not in serialized
    for split_name in DIRECTID_SPLITS:
        split_evidence = evidence["splits"][split_name]
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", split_evidence["dataset_hash"])
        assert set(split_evidence["label_counts"]) == set(
            DIRECTID_TINY_HEAD_CONTRACT.labels
        )
        assert set(split_evidence["id_subtype_counts"]) == set(
            DIRECTID_TINY_HEAD_CONTRACT.id_subtypes
        )
        assert set(split_evidence["hard_negative_category_counts"]) == set(
            HARD_NEGATIVE_CATEGORIES
        )
        assert set(split_evidence["source_hashes"]) == set(split_evidence["source_ids"])
        assert all(
            re.fullmatch(r"sha256:[0-9a-f]{64}", source_hash)
            for source_hash in split_evidence["source_hashes"].values()
        )


def test_manifest_hash_is_stable_and_tiny_preset_points_to_manifest() -> None:
    manifest_hash = directid_dataset_manifest_hash()
    recipe = load_preset("tiny_distill")

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_hash)
    assert directid_dataset_manifest_hash() == manifest_hash
    assert recipe.dapt.corpus_ref == DIRECTID_DATASET_MANIFEST_REF
