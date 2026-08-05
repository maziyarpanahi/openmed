"""Run the offline facility-EMR to national-HMIS synthetic reference demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fixture_server import openmrs_fixture_server
from synthetic_data import (
    DEFAULT_PATIENT_COUNT,
    DEFAULT_SEED,
    DISTRICT_ORG_UNIT,
    SyntheticDataset,
    generate_dataset,
)

from openmed.clinical.exporters import DHIS2ExportConfig, export_dhis2
from openmed.interop import assert_redacted
from openmed.interop.openmrs import (
    DeidentifiedResource,
    OpenMRSAdapter,
    OpenMRSClient,
    OpenMRSConfig,
    manifest_paths,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
ORG_UNITS_FIXTURE = EXAMPLE_DIR / "fixtures" / "organisation_units.json"
ARTIFACT_NAMES = (
    "01-raw-openmrs.json",
    "02-deidentified-fhir-bundle.json",
    "03-openmrs-deidentification-manifest.json",
    "04-dhis2-aggregate.json",
    "05-dhis2-tracker.json",
    "06-dhis2-export-manifest.json",
    "07-demo-manifest.json",
)


@dataclass(frozen=True)
class _RedactionResult:
    deidentified_text: str


class ExactSyntheticRedactor:
    """Replace the complete, known synthetic identifier inventory."""

    def __init__(self, mapping: dict[str, str]) -> None:
        replacements: dict[str, str] = {}
        for replacement, original in mapping.items():
            replacements.setdefault(original, replacement)
        self._replacements = tuple(
            sorted(
                replacements.items(),
                key=lambda item: (-len(item[0]), item[0]),
            )
        )

    def __call__(self, text: str, **_: Any) -> _RedactionResult:
        transformed = text
        for original, replacement in self._replacements:
            transformed = transformed.replace(original, replacement)
        return _RedactionResult(deidentified_text=transformed)


def build_parser() -> argparse.ArgumentParser:
    """Build the documented fixture-mode command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("africa-reference-output"),
        help="directory for inspectable stage artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="fixed synthetic-data seed",
    )
    parser.add_argument(
        "--patient-count",
        type=int,
        default=DEFAULT_PATIENT_COUNT,
        help="number of fictional patients",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=17,
        help="fixture-server FHIR search page size",
    )
    return parser


def run_demo(
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    patient_count: int = DEFAULT_PATIENT_COUNT,
    page_size: int = 17,
) -> dict[str, Path]:
    """Execute every fixture-backed stage and return the artifact paths."""

    if page_size < 1:
        raise ValueError("page_size must be greater than zero")

    dataset = generate_dataset(seed=seed, patient_count=patient_count)
    phi_mapping = dataset.phi_mapping()
    redactor = ExactSyntheticRedactor(phi_mapping)
    fhir_fixture_resources = dataset.fhir_resources()
    rest_fixture_resources = dataset.rest_resources()

    with openmrs_fixture_server(
        fhir_fixture_resources,
        rest_fixture_resources,
    ) as base_url:
        with OpenMRSClient(
            OpenMRSConfig(
                base_url=base_url,
                page_size=page_size,
                max_retries=0,
            )
        ) as client:
            raw_resources = {
                "fhir2": {
                    resource_type: list(client.pull_fhir(resource_type))
                    for resource_type in ("Patient", "Encounter", "Observation")
                },
                "rest": {
                    resource_type: list(client.pull_rest(resource_type))
                    for resource_type in ("patient", "encounter", "obs")
                },
            }
            adapter = OpenMRSAdapter(client, deidentifier=redactor)
            deidentified_fhir_records = tuple(
                record
                for resource_type in ("Patient", "Encounter", "Observation")
                for record in adapter.pull_fhir(resource_type)
            )
            deidentified_rest_records = tuple(
                record
                for resource_type in ("encounter", "obs")
                for record in adapter.pull_rest(resource_type)
            )
            deidentified_records = deidentified_fhir_records + deidentified_rest_records
            fhir_bundle = adapter.export_bundle(
                deidentified_fhir_records,
                doc_id=f"africa-reference-{seed}",
            )

    for record in deidentified_records:
        assert_redacted(_canonical_json(record.resource), phi_mapping)
    _validate_transaction_bundle(fhir_bundle, expected_entries=patient_count * 3)
    openmrs_manifest = _openmrs_manifest(deidentified_records, dataset)
    aggregate_source, tracker_source = _build_dhis2_sources(
        dataset,
        deidentified_records,
    )
    dhis2_result = export_dhis2(
        aggregate_source,
        tracker_source,
        ORG_UNITS_FIXTURE,
        config=DHIS2ExportConfig(
            generalization_level=3,
            small_cell_threshold=5,
            date_mode="coarsen",
            period_granularity="month",
        ),
        text_redactor=redactor,
    )
    _assert_district_only(dhis2_result.combined_payload, dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        ARTIFACT_NAMES[0]: output_dir / ARTIFACT_NAMES[0],
        ARTIFACT_NAMES[1]: output_dir / ARTIFACT_NAMES[1],
        ARTIFACT_NAMES[2]: output_dir / ARTIFACT_NAMES[2],
        ARTIFACT_NAMES[3]: output_dir / ARTIFACT_NAMES[3],
        ARTIFACT_NAMES[4]: output_dir / ARTIFACT_NAMES[4],
        ARTIFACT_NAMES[5]: output_dir / ARTIFACT_NAMES[5],
        ARTIFACT_NAMES[6]: output_dir / ARTIFACT_NAMES[6],
    }
    _write_json(
        artifacts[ARTIFACT_NAMES[0]],
        {
            "fixture": "fictional-mtoni-district",
            "seed": seed,
            "resources": raw_resources,
        },
    )
    _write_json(artifacts[ARTIFACT_NAMES[1]], fhir_bundle)
    _write_json(artifacts[ARTIFACT_NAMES[2]], openmrs_manifest)
    _write_json(artifacts[ARTIFACT_NAMES[3]], dhis2_result.aggregate_payload)
    _write_json(artifacts[ARTIFACT_NAMES[4]], dhis2_result.tracker_payload)
    _write_json(artifacts[ARTIFACT_NAMES[5]], dhis2_result.manifest)

    boundary_paths = tuple(
        artifacts[name]
        for name in (
            ARTIFACT_NAMES[1],
            ARTIFACT_NAMES[2],
            ARTIFACT_NAMES[3],
            ARTIFACT_NAMES[4],
            ARTIFACT_NAMES[5],
        )
    )
    for path in boundary_paths:
        assert_redacted(path.read_text(encoding="utf-8"), phi_mapping)

    demo_manifest = {
        "schema_version": 1,
        "mode": "fixture",
        "seed": seed,
        "patient_count": patient_count,
        "district_org_unit": dataset.district_uid,
        "privacy_boundary": "de-identification-before-egress",
        "artifacts": [
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifacts.values()
            if path.name != ARTIFACT_NAMES[6]
        ],
    }
    _write_json(artifacts[ARTIFACT_NAMES[6]], demo_manifest)
    assert_redacted(
        artifacts[ARTIFACT_NAMES[6]].read_text(encoding="utf-8"),
        phi_mapping,
    )

    console_output = (
        "Fixture demo complete: "
        f"{patient_count} synthetic patients, "
        f"{len(fhir_bundle['entry'])} de-identified FHIR resources, "
        f"district org unit {DISTRICT_ORG_UNIT}, "
        f"{len(artifacts)} stage artifacts."
    )
    assert_redacted(console_output, phi_mapping)
    print(console_output)
    return artifacts


def _openmrs_manifest(
    records: tuple[DeidentifiedResource, ...],
    dataset: SyntheticDataset,
) -> dict[str, Any]:
    counts = Counter(f"{record.api}:{record.resource_name}" for record in records)
    transformed_paths = sorted(
        {path for record in records for path in manifest_paths(record.manifest)}
    )
    return {
        "schema_version": 1,
        "adapter": "openmrs-fhir2",
        "fixture_seed": dataset.seed,
        "resource_counts": dict(sorted(counts.items())),
        "transformed_resource_count": sum(
            bool(manifest_paths(record.manifest)) for record in records
        ),
        "transformed_paths": transformed_paths,
    }


def _build_dhis2_sources(
    dataset: SyntheticDataset,
    records: tuple[DeidentifiedResource, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    deidentified_patients = {
        record.resource["id"]: record.resource
        for record in records
        if record.resource_name == "Patient"
    }
    counts = Counter(
        (
            patient.facility_uid,
            patient.encounter_date[:7].replace("-", ""),
            patient.diagnosis_code,
        )
        for patient in dataset.patients
    )
    data_value_sets: list[dict[str, Any]] = []
    for facility in dataset.facilities:
        periods = sorted(
            period for facility_uid, period, _ in counts if facility_uid == facility.uid
        )
        for period in dict.fromkeys(periods):
            data_values = [
                {
                    "dataElement": f"icd11-{diagnosis_code}",
                    "value": str(count),
                }
                for (
                    facility_uid,
                    value_period,
                    diagnosis_code,
                ), count in sorted(counts.items())
                if facility_uid == facility.uid and value_period == period
            ]
            data_value_sets.append(
                {
                    "dataSet": "dsMtoniIcd11Monthly",
                    "period": period,
                    "orgUnit": facility.uid,
                    "completeDate": f"{period[:4]}-{period[4:]}-28",
                    "dataValues": data_values,
                }
            )

    tracked_entities: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for patient in dataset.patients:
        fhir_patient = deidentified_patients[patient.patient_id]
        display_name = fhir_patient["name"][0]["text"]
        national_id = fhir_patient["identifier"][0]["value"]
        tracked_entity = f"tei-{patient.index:03d}"
        tracked_entities.append(
            {
                "trackedEntity": tracked_entity,
                "trackedEntityType": "tetSyntheticPatient",
                "orgUnit": patient.facility_uid,
                "geometry": {
                    "type": "Point",
                    "coordinates": [patient.longitude, patient.latitude],
                },
                "latitude": patient.latitude,
                "longitude": patient.longitude,
                "attributes": [
                    {
                        "attribute": "attrDisplayName",
                        "value": display_name,
                    },
                    {
                        "attribute": "attrNationalIdentifier",
                        "value": national_id,
                    },
                ],
            }
        )
        events.append(
            {
                "event": f"event-{patient.index:03d}",
                "program": "programIcd11Surveillance",
                "programStage": "stageDiagnosis",
                "trackedEntity": tracked_entity,
                "orgUnit": patient.facility_uid,
                "occurredAt": f"{patient.encounter_date}T08:15:00Z",
                "geometry": {
                    "type": "Point",
                    "coordinates": [patient.longitude, patient.latitude],
                },
                "dataValues": [
                    {
                        "dataElement": "deIcd11Diagnosis",
                        "value": patient.diagnosis_code,
                    }
                ],
                "notes": [
                    {"value": (f"Facility-coded synthetic case for {display_name}")}
                ],
            }
        )

    return (
        {"dataValueSets": data_value_sets},
        {"trackedEntities": tracked_entities, "events": events},
    )


def _validate_transaction_bundle(bundle: Any, *, expected_entries: int) -> None:
    if not isinstance(bundle, dict):
        raise ValueError("FHIR export must be a JSON object")
    if bundle.get("resourceType") != "Bundle" or bundle.get("type") != "transaction":
        raise ValueError("FHIR export must be a transaction Bundle")
    entries = bundle.get("entry")
    if not isinstance(entries, list) or len(entries) != expected_entries:
        raise ValueError("FHIR transaction Bundle has an unexpected entry count")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("resource"), dict):
            raise ValueError("FHIR transaction entry lacks a resource")
        request = entry.get("request")
        if not isinstance(request, dict) or request.get("method") != "POST":
            raise ValueError("FHIR transaction entry lacks a POST request")


def _assert_district_only(payload: dict[str, Any], dataset: SyntheticDataset) -> None:
    serialized = _canonical_json(payload)
    for facility in dataset.facilities:
        if facility.uid in serialized:
            raise ValueError("facility org unit survived district generalization")
    for key in ('"geometry"', '"latitude"', '"longitude"'):
        if key in serialized:
            raise ValueError("precise geography survived DHIS2 export")
    org_units = _collect_key(payload, "orgUnit")
    if not org_units or set(org_units) != {dataset.district_uid}:
        raise ValueError("DHIS2 export must reference only the district org unit")


def _collect_key(value: Any, target: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == target:
                found.append(child)
            else:
                found.extend(_collect_key(child, target))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_key(child, target))
    return found


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8")


def main() -> None:
    """Run the documented fixture-backed command."""

    args = build_parser().parse_args()
    run_demo(
        args.output_dir,
        seed=args.seed,
        patient_count=args.patient_count,
        page_size=args.page_size,
    )


if __name__ == "__main__":
    main()
