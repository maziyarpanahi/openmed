"""Deterministic synthetic data for the African OpenMRS-to-DHIS2 demo.

Every person, identifier, phone number, and coordinate emitted here is
fictional. The generator intentionally includes values that must be removed at
the facility boundary so the demo can prove its privacy gates are effective.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

DEFAULT_SEED = 875
DEFAULT_PATIENT_COUNT = 50
DISTRICT_ORG_UNIT = "ouDistrictMtoni"
ICD11_SYSTEM = "http://id.who.int/icd/release/11/mms"

_GIVEN_NAMES = (
    "Amina",
    "Amara",
    "Chidi",
    "Fatuma",
    "Imani",
    "Kato",
    "Lerato",
    "Musa",
    "Nia",
    "Odhiambo",
    "Sade",
    "Thabo",
)
_DIAGNOSES = (
    ("1A00", "Cholera"),
    ("5A11", "Type 2 diabetes mellitus"),
    ("BA00", "Essential hypertension"),
)


@dataclass(frozen=True)
class Facility:
    """One fictional facility inside the synthetic district."""

    uid: str
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class SyntheticPatient:
    """One fictional patient and the local identifiers seeded for redaction."""

    index: int
    patient_id: str
    encounter_id: str
    observation_id: str
    given_name: str
    family_name: str
    msisdn: str
    national_id: str
    latitude: float
    longitude: float
    facility_uid: str
    diagnosis_code: str
    diagnosis_display: str
    encounter_date: str

    @property
    def full_name(self) -> str:
        """Return the deliberately identifying display name."""

        return f"{self.given_name} {self.family_name}"

    @property
    def gps(self) -> str:
        """Return the deliberately identifying coordinate pair."""

        return f"{self.latitude:.5f}, {self.longitude:.5f}"


@dataclass(frozen=True)
class SyntheticDataset:
    """A reproducible fictional district health system."""

    seed: int
    district_uid: str
    facilities: tuple[Facility, ...]
    patients: tuple[SyntheticPatient, ...]

    def phi_mapping(self) -> dict[str, str]:
        """Return replacement-token to original-value mappings for leak gates."""

        mapping: dict[str, str] = {}
        for patient in self.patients:
            suffix = f"{patient.index:03d}"
            mapping[f"[FULL_NAME_{suffix}]"] = patient.full_name
            mapping[f"[GIVEN_NAME_{suffix}]"] = patient.given_name
            mapping[f"[FAMILY_NAME_{suffix}]"] = patient.family_name
            mapping[f"[MSISDN_{suffix}]"] = patient.msisdn
            mapping[f"[NATIONAL_ID_{suffix}]"] = patient.national_id
            mapping[f"[GPS_{suffix}]"] = patient.gps
            mapping[f"[LATITUDE_{suffix}]"] = f"{patient.latitude:.5f}"
            mapping[f"[LONGITUDE_{suffix}]"] = f"{patient.longitude:.5f}"
        return mapping

    def fhir_resources(self) -> dict[str, list[dict[str, Any]]]:
        """Build the OpenMRS FHIR2 fixture recording for this dataset."""

        resources: dict[str, list[dict[str, Any]]] = {
            "Patient": [],
            "Encounter": [],
            "Observation": [],
        }
        for patient in self.patients:
            resources["Patient"].append(_patient_resource(patient))
            resources["Encounter"].append(_encounter_resource(patient))
            resources["Observation"].append(_observation_resource(patient))
        return resources

    def rest_resources(self) -> dict[str, list[dict[str, Any]]]:
        """Build the matching OpenMRS legacy REST fixture recording."""

        resources: dict[str, list[dict[str, Any]]] = {
            "patient": [],
            "encounter": [],
            "obs": [],
        }
        for patient in self.patients:
            resources["patient"].append(_rest_patient_resource(patient))
            resources["encounter"].append(_rest_encounter_resource(patient))
            resources["obs"].append(_rest_observation_resource(patient))
        return resources

    def recording(self) -> dict[str, Any]:
        """Return a JSON-serializable recorded-response fixture."""

        return {
            "fixture": "fictional-mtoni-district",
            "seed": self.seed,
            "districtOrgUnit": self.district_uid,
            "fhir2": self.fhir_resources(),
            "rest": self.rest_resources(),
        }


def generate_dataset(
    *,
    seed: int = DEFAULT_SEED,
    patient_count: int = DEFAULT_PATIENT_COUNT,
) -> SyntheticDataset:
    """Generate one district, three facilities, and synthetic patient records."""

    if patient_count < 1:
        raise ValueError("patient_count must be greater than zero")

    facilities = (
        Facility("ouFacilityKijani", "Kijani Health Centre", -0.31230, 32.58120),
        Facility("ouFacilityMawingu", "Mawingu District Clinic", -0.28710, 32.60640),
        Facility("ouFacilityTumaini", "Tumaini Community Hospital", -0.33580, 32.62510),
    )
    rng = random.Random(seed)
    first_day = date(2026, 1, 5)
    patients: list[SyntheticPatient] = []
    for offset in range(patient_count):
        index = offset + 1
        facility = facilities[offset % len(facilities)]
        diagnosis_code, diagnosis_display = _DIAGNOSES[rng.randrange(len(_DIAGNOSES))]
        patients.append(
            SyntheticPatient(
                index=index,
                patient_id=f"patient-{index:03d}",
                encounter_id=f"encounter-{index:03d}",
                observation_id=f"observation-{index:03d}",
                given_name=_GIVEN_NAMES[offset % len(_GIVEN_NAMES)],
                family_name=f"Fiction{index:03d}",
                msisdn=f"+256 700 {index // 1000:03d} {index % 1000:03d}",
                national_id=f"CF-{seed:04d}-{index:06d}",
                latitude=facility.latitude + rng.uniform(-0.008, 0.008),
                longitude=facility.longitude + rng.uniform(-0.008, 0.008),
                facility_uid=facility.uid,
                diagnosis_code=diagnosis_code,
                diagnosis_display=diagnosis_display,
                encounter_date=(first_day + timedelta(days=offset)).isoformat(),
            )
        )

    return SyntheticDataset(
        seed=seed,
        district_uid=DISTRICT_ORG_UNIT,
        facilities=facilities,
        patients=tuple(patients),
    )


def _patient_resource(patient: SyntheticPatient) -> dict[str, Any]:
    return {
        "resourceType": "Patient",
        "id": patient.patient_id,
        "active": True,
        "identifier": [
            {
                "system": "https://fictional.example.org/national-id",
                "value": patient.national_id,
            }
        ],
        "name": [
            {
                "use": "official",
                "text": patient.full_name,
                "family": patient.family_name,
                "given": [patient.given_name],
            }
        ],
        "telecom": [{"system": "phone", "value": patient.msisdn, "use": "mobile"}],
        "address": [{"district": "Mtoni", "country": "Fictional Republic"}],
        "extension": [
            {
                "url": "https://fictional.example.org/fhir/StructureDefinition/gps",
                "valueString": patient.gps,
            }
        ],
    }


def _encounter_resource(patient: SyntheticPatient) -> dict[str, Any]:
    return {
        "resourceType": "Encounter",
        "id": patient.encounter_id,
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "AMB",
            "display": "ambulatory",
        },
        "subject": {"reference": f"Patient/{patient.patient_id}"},
        "serviceProvider": {"reference": f"Organization/{patient.facility_uid}"},
        "period": {
            "start": f"{patient.encounter_date}T08:00:00Z",
            "end": f"{patient.encounter_date}T08:30:00Z",
        },
        "reasonCode": [
            {
                "coding": [
                    {
                        "system": ICD11_SYSTEM,
                        "code": patient.diagnosis_code,
                        "display": patient.diagnosis_display,
                    }
                ]
            }
        ],
    }


def _observation_resource(patient: SyntheticPatient) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": patient.observation_id,
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": ICD11_SYSTEM,
                    "code": patient.diagnosis_code,
                    "display": patient.diagnosis_display,
                }
            ]
        },
        "subject": {"reference": f"Patient/{patient.patient_id}"},
        "encounter": {"reference": f"Encounter/{patient.encounter_id}"},
        "effectiveDateTime": f"{patient.encounter_date}T08:15:00Z",
        "valueString": (
            f"{patient.full_name}; phone {patient.msisdn}; national ID "
            f"{patient.national_id}; home GPS {patient.gps}."
        ),
    }


def _rest_patient_resource(patient: SyntheticPatient) -> dict[str, Any]:
    return {
        "uuid": patient.patient_id,
        "display": patient.full_name,
        "identifiers": [
            {
                "identifier": patient.national_id,
                "identifierType": {"display": "Fictional national identifier"},
            }
        ],
        "person": {
            "display": patient.full_name,
            "names": [
                {
                    "givenName": patient.given_name,
                    "familyName": patient.family_name,
                }
            ],
        },
    }


def _rest_encounter_resource(patient: SyntheticPatient) -> dict[str, Any]:
    return {
        "uuid": patient.encounter_id,
        "patient": {"uuid": patient.patient_id},
        "encounterDatetime": f"{patient.encounter_date}T08:00:00.000+0000",
        "encounterType": {"uuid": "encounter-type-outpatient"},
        "encounterNotes": [
            f"{patient.full_name} attended the fictional outpatient clinic.",
            f"Local contact {patient.msisdn}; home GPS {patient.gps}.",
        ],
    }


def _rest_observation_resource(patient: SyntheticPatient) -> dict[str, Any]:
    return {
        "uuid": patient.observation_id,
        "person": {"uuid": patient.patient_id},
        "encounter": {"uuid": patient.encounter_id},
        "obsDatetime": f"{patient.encounter_date}T08:15:00.000+0000",
        "concept": {
            "uuid": f"icd11-{patient.diagnosis_code}",
            "display": patient.diagnosis_display,
        },
        "value": (
            f"ICD-11 {patient.diagnosis_code} recorded for "
            f"{patient.full_name} at {patient.gps}."
        ),
        "comment": (f"Contact {patient.msisdn}; national ID {patient.national_id}."),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--patient-count",
        type=int,
        default=DEFAULT_PATIENT_COUNT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Write a deterministic synthetic OpenMRS fixture recording."""

    args = _parser().parse_args()
    dataset = generate_dataset(seed=args.seed, patient_count=args.patient_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dataset.recording(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
