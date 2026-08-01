"""End-to-end safety coverage for the African OpenMRS-to-DHIS2 reference."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from openmed.interop import assert_redacted
from tests.unit.clinical.fixtures.dhis2.schema_checker import (
    validate_aggregate_payload,
    validate_tracker_payload,
)

ROOT = Path(__file__).parents[3]
EXAMPLE = ROOT / "examples" / "africa-openmrs-dhis2"
RUN_DEMO = EXAMPLE / "run_demo.py"
DOCS = ROOT / "docs" / "deploy" / "africa-reference-deployment.md"
COMPOSE = EXAMPLE / "docker-compose.yml"
DISTRICT_UID = "ouDistrictMtoni"
ARTIFACT_NAMES = (
    "01-raw-openmrs.json",
    "02-deidentified-fhir-bundle.json",
    "03-openmrs-deidentification-manifest.json",
    "04-dhis2-aggregate.json",
    "05-dhis2-tracker.json",
    "06-dhis2-export-manifest.json",
    "07-demo-manifest.json",
)


def _run_demo(output_dir: Path, *, page_size: int = 17) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(RUN_DEMO),
            "--output-dir",
            str(output_dir),
            "--seed",
            "875",
            "--patient-count",
            "50",
            "--page-size",
            str(page_size),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _load(output_dir: Path, name: str) -> Any:
    return json.loads((output_dir / name).read_text(encoding="utf-8"))


def _seeded_phi(raw: dict[str, Any]) -> dict[str, str]:
    originals: list[str] = []
    for patient in raw["resources"]["fhir2"]["Patient"]:
        name = patient["name"][0]
        originals.extend(
            [
                name["text"],
                name["family"],
                *name["given"],
                patient["identifier"][0]["value"],
                patient["telecom"][0]["value"],
                patient["extension"][0]["valueString"],
            ]
        )
        originals.extend(
            part.strip() for part in patient["extension"][0]["valueString"].split(",")
        )
    return {
        f"[SEEDED_IDENTIFIER_{index:04d}]": original
        for index, original in enumerate(dict.fromkeys(originals), start=1)
    }


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


def test_fixture_demo_round_trip_is_valid_private_and_district_only(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    process = _run_demo(output_dir)
    assert process.stderr == ""
    assert "50 synthetic patients" in process.stdout
    assert all((output_dir / name).is_file() for name in ARTIFACT_NAMES)

    raw = _load(output_dir, ARTIFACT_NAMES[0])
    assert raw["seed"] == 875
    assert {
        resource_type: len(resources)
        for resource_type, resources in raw["resources"]["fhir2"].items()
    } == {"Encounter": 50, "Observation": 50, "Patient": 50}
    assert {
        resource_type: len(resources)
        for resource_type, resources in raw["resources"]["rest"].items()
    } == {"encounter": 50, "obs": 50, "patient": 50}
    phi_mapping = _seeded_phi(raw)
    assert len(phi_mapping) >= 250

    bundle = _load(output_dir, ARTIFACT_NAMES[1])
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    assert len(bundle["entry"]) == 150
    assert Counter(entry["resource"]["resourceType"] for entry in bundle["entry"]) == {
        "Patient": 50,
        "Encounter": 50,
        "Observation": 50,
    }
    assert all(entry["request"]["method"] == "POST" for entry in bundle["entry"])
    assert all(entry["fullUrl"].startswith("urn:uuid:") for entry in bundle["entry"])

    openmrs_manifest = _load(output_dir, ARTIFACT_NAMES[2])
    assert openmrs_manifest["resource_counts"] == {
        "fhir2:Encounter": 50,
        "fhir2:Observation": 50,
        "fhir2:Patient": 50,
        "rest:encounter": 50,
        "rest:obs": 50,
    }

    aggregate = _load(output_dir, ARTIFACT_NAMES[3])
    tracker = _load(output_dir, ARTIFACT_NAMES[4])
    validate_aggregate_payload(aggregate)
    validate_tracker_payload(tracker)
    combined = {"aggregate": aggregate, "tracker": tracker}
    assert set(_collect_key(combined, "orgUnit")) == {DISTRICT_UID}

    serialized_dhis2 = json.dumps(combined, ensure_ascii=False, sort_keys=True)
    org_units = json.loads(
        (EXAMPLE / "fixtures" / "organisation_units.json").read_text(encoding="utf-8")
    )["organisationUnits"]
    facility_uids = {org_unit["id"] for org_unit in org_units if org_unit["level"] == 4}
    assert not any(facility_uid in serialized_dhis2 for facility_uid in facility_uids)
    assert '"geometry"' not in serialized_dhis2
    assert '"latitude"' not in serialized_dhis2
    assert '"longitude"' not in serialized_dhis2

    for name in ARTIFACT_NAMES[1:]:
        assert_redacted(
            (output_dir / name).read_text(encoding="utf-8"),
            phi_mapping,
        )
    assert_redacted(process.stdout, phi_mapping)


def test_same_seed_produces_byte_identical_stage_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_demo(first, page_size=7)
    _run_demo(second, page_size=19)

    assert {name: (first / name).read_bytes() for name in ARTIFACT_NAMES} == {
        name: (second / name).read_bytes() for name in ARTIFACT_NAMES
    }


def test_documented_walkthrough_matches_the_cli_surface() -> None:
    help_result = subprocess.run(
        [sys.executable, str(RUN_DEMO), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    docs = DOCS.read_text(encoding="utf-8")
    for flag in ("--output-dir", "--seed", "--patient-count", "--page-size"):
        assert flag in help_result.stdout
        assert flag in docs
    assert ".venv/bin/python examples/africa-openmrs-dhis2/run_demo.py" in docs
    assert "tests/unit/examples/test_africa_reference_demo.py" in docs


def test_optional_live_compose_profile_is_explicit_and_loopback_only() -> None:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert compose["services"]
    for service in compose["services"].values():
        assert service["profiles"] == ["live"]
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:")

    docs = DOCS.read_text(encoding="utf-8")
    assert "--profile live" in docs
    assert "CI never starts this profile" in docs
