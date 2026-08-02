"""Pull OpenMRS FHIR2 resources and prepare a local de-identified handoff."""

from __future__ import annotations

import os
from pathlib import Path

from openmed.interop.openmrs import OpenMRSAdapter, OpenMRSClient, OpenMRSConfig


def _config_from_environment() -> OpenMRSConfig:
    base_url = os.environ["OPENMRS_BASE_URL"]
    destination_url = os.environ.get("OPENMRS_DESTINATION_URL")
    session_token = os.environ.get("OPENMRS_SESSION_TOKEN")
    username = os.environ.get("OPENMRS_USERNAME")
    password = os.environ.get("OPENMRS_PASSWORD")

    if session_token:
        return OpenMRSConfig(
            base_url=base_url,
            destination_url=destination_url,
            session_token=session_token,
        )
    if not username or not password:
        raise RuntimeError(
            "set OPENMRS_SESSION_TOKEN or both OPENMRS_USERNAME and OPENMRS_PASSWORD"
        )
    return OpenMRSConfig(
        base_url=base_url,
        destination_url=destination_url,
        username=username,
        password=password,
    )


def main() -> None:
    """Run a FHIR2 handoff without printing clinical payload content."""

    patient_uuid = os.environ["OPENMRS_PATIENT_UUID"]
    output_path = Path(
        os.environ.get("OPENMRS_NDJSON_PATH", "openmrs-deidentified.ndjson")
    )
    allow_write = os.environ.get("OPENMRS_ALLOW_WRITE") == "1"

    with OpenMRSClient(_config_from_environment()) as client:
        adapter = OpenMRSAdapter(client)
        records = (
            adapter.pull_fhir("Patient", params={"_id": patient_uuid})
            + adapter.pull_fhir("Encounter", params={"patient": patient_uuid})
            + adapter.pull_fhir("Observation", params={"patient": patient_uuid})
        )

        summary = adapter.export_ndjson(records, output_path)
        writes = adapter.write_back(records, dry_run=not allow_write)

    transformed_path_count = sum(len(record.transformed_paths) for record in records)
    print(f"resources prepared: {len(records)}")
    print(f"transformed paths: {transformed_path_count}")
    print(f"NDJSON resources: {summary.resources_deidentified}")
    print(f"write requests completed: {sum(not result.dry_run for result in writes)}")


if __name__ == "__main__":
    main()
