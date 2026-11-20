"""Offline tests for exporting redistributable crosswalks as FHIR ConceptMaps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical.grounding.crosswalk import (
    CrosswalkEntry,
    CrosswalkLicenseError,
    CrosswalkResource,
    UMLSCrosswalk,
    crosswalk_resource_to_conceptmap,
    load_default_crosswalks,
)

_SYNTHETIC_HASH = f"sha256:{'0' * 64}"
_RXNORM_URI = "http://www.nlm.nih.gov/research/umls/rxnorm"


def _entry(
    source_code: str,
    target_code: str,
    *,
    source_system: str = "ICD-10-CN",
    target_system: str = "ICD10",
    target_display: str = "Synthetic display",
) -> CrosswalkEntry:
    return CrosswalkEntry(
        source_system=source_system,
        source_code=source_code,
        locale="zh-CN",
        aliases=(f"alias {source_code} {target_code}",),
        target_system=target_system,
        target_code=target_code,
        target_display=target_display,
    )


def _resource(
    entries: tuple[CrosswalkEntry, ...],
    *,
    license_id: str = "CC0-1.0",
    redistributable: bool = True,
) -> CrosswalkResource:
    return CrosswalkResource(
        name="synthetic-crosswalk",
        version="2026.01.0",
        license_id=license_id,
        redistributable=redistributable,
        entries=entries,
        content_hash=_SYNTHETIC_HASH,
    )


def _bundled_icd10cn() -> CrosswalkResource:
    resources = load_default_crosswalks()
    return next(
        resource
        for resource in resources
        if resource.name == "openmed-icd10cn-icd10-starter"
    )


def test_bundled_icd10cn_exports_one_element_per_distinct_source_code() -> None:
    resource = _bundled_icd10cn()

    concept_map = crosswalk_resource_to_conceptmap(resource)

    assert concept_map["resourceType"] == "ConceptMap"
    assert concept_map["status"] == "active"
    assert concept_map["name"] == resource.name
    assert concept_map["version"] == resource.version
    assert concept_map["copyright"] == resource.license_id
    assert concept_map["identifier"] == {"value": resource.content_hash}
    distinct_source_codes = {entry.source_code for entry in resource.entries}
    elements = [
        element for group in concept_map["group"] for element in group["element"]
    ]
    assert len(elements) == len(distinct_source_codes)
    assert {element["code"] for element in elements} == distinct_source_codes
    targets = [target for element in elements for target in element["target"]]
    assert targets
    assert all(target["equivalence"] == "relatedto" for target in targets)


def test_group_systems_use_registry_uris_and_keep_unregistered_labels() -> None:
    resource = _resource(
        (
            _entry(
                "RX-SYN-1",
                "HP:0000001",
                source_system="RXNORM",
                target_system="HPO",
                target_display="Synthetic root",
            ),
            _entry("E11.900", "E11.9", target_display="Synthetic diabetes"),
        )
    )

    concept_map = crosswalk_resource_to_conceptmap(resource)

    pairs = {(group["source"], group["target"]) for group in concept_map["group"]}
    assert pairs == {
        # The RxNorm label resolves to its registry URI; it is not refused
        # even though that URI contains "umls".
        (_RXNORM_URI, "http://human-phenotype-ontology.org"),
        # ICD-10-CN has no registry URI and keeps its label; WHO "ICD10"
        # is kept verbatim so it does not become the ICD-10-CM URI.
        ("ICD-10-CN", "ICD10"),
    }


def test_rxnorm_system_uri_is_exported_unchanged_and_not_refused() -> None:
    resource = _resource((_entry("RX-SYN-2", "E11.9", source_system=_RXNORM_URI),))

    concept_map = crosswalk_resource_to_conceptmap(resource)

    assert [group["source"] for group in concept_map["group"]] == [_RXNORM_URI]


def test_groups_elements_and_targets_are_sorted_by_system_pair_and_code() -> None:
    resource = _resource(
        (
            _entry("I10.x00", "I10", target_display="Synthetic hypertension"),
            _entry(
                "HP-XX",
                "HP:0001250",
                source_system="CHPO-SYN",
                target_system="HPO",
                target_display="Synthetic seizure",
            ),
            _entry("E11.900", "E11.9", target_display="Synthetic diabetes"),
            _entry("E11.900", "E11.8", target_display="Synthetic complication"),
        )
    )

    concept_map = crosswalk_resource_to_conceptmap(resource)

    assert [(group["source"], group["target"]) for group in concept_map["group"]] == [
        ("CHPO-SYN", "http://human-phenotype-ontology.org"),
        ("ICD-10-CN", "ICD10"),
    ]
    icd_group = concept_map["group"][1]
    assert [element["code"] for element in icd_group["element"]] == [
        "E11.900",
        "I10.x00",
    ]
    assert [target["code"] for target in icd_group["element"][0]["target"]] == [
        "E11.8",
        "E11.9",
    ]


def test_two_runs_produce_byte_identical_json() -> None:
    first = json.dumps(crosswalk_resource_to_conceptmap(_bundled_icd10cn())).encode(
        "utf-8"
    )
    second = json.dumps(crosswalk_resource_to_conceptmap(_bundled_icd10cn())).encode(
        "utf-8"
    )

    assert first == second


def test_a_umls_crosswalk_cannot_be_exported(tmp_path: Path) -> None:
    (tmp_path / "MRCONSO.RRF").write_text(
        "CUI|LAT|SAB|CODE|STR\nC-SYN-ICD|ENG|ICD10CM|E11.9|Synthetic diabetes\n",
        encoding="utf-8",
    )
    (tmp_path / "MRMAP.RRF").write_text(
        "source_system|source_code|target_system|target_code\n"
        "ICD10CM|E11.9|SNOMEDCT_US|44054006\n",
        encoding="utf-8",
    )
    engine = UMLSCrosswalk(tmp_path)

    with pytest.raises(CrosswalkLicenseError, match="UMLS"):
        crosswalk_resource_to_conceptmap(engine)


@pytest.mark.parametrize(
    "source_system",
    [
        "UMLS",
        "SNOMED CT",
        "cpt4",
        "SNOMEDCT_US",
        "SNOMEDCT_UK",
        "SNOMEDCT_VET",
        "http://snomed.info/sct",
        "http://snomed.info/sct/900000000000207008",
        "http://terminology.hl7.org/CodeSystem/umls",
        "http://www.ama-assn.org/go/cpt",
        "urn:openmed:restricted:cpt",
    ],
)
def test_restricted_labels_uris_and_source_abbreviations_cannot_be_exported(
    source_system: str,
) -> None:
    resource = _resource((_entry("C0011849", "E11.9", source_system=source_system),))

    with pytest.raises(CrosswalkLicenseError, match="restricted terminology"):
        crosswalk_resource_to_conceptmap(resource)


def test_a_restricted_uri_target_cannot_be_exported() -> None:
    # Entry validation restricts targets to HPO/ICD10, so flip the field
    # afterwards to prove the exporter re-checks target systems itself.
    entry = _entry("E11.900", "E11.9")
    object.__setattr__(entry, "target_system", "http://snomed.info/sct")
    resource = _resource((entry,))

    with pytest.raises(CrosswalkLicenseError, match="restricted terminology"):
        crosswalk_resource_to_conceptmap(resource)


def test_an_unredistributable_resource_cannot_be_exported() -> None:
    with pytest.raises(CrosswalkLicenseError, match="not declared redistributable"):
        _resource((_entry("E11.900", "E11.9"),), redistributable=False)

    flipped = _resource((_entry("E11.900", "E11.9"),))
    object.__setattr__(flipped, "redistributable", False)
    with pytest.raises(CrosswalkLicenseError, match="not declared redistributable"):
        crosswalk_resource_to_conceptmap(flipped)
