"""Synthetic, versioned reference-server compatibility expectations."""

from __future__ import annotations

from openmed.interop.fhir.conformance import CASES, MATRIX_VERSION, SERVERS

# These are protocol expectations, not claims that a product release has been
# certified. The opt-in profile binds each run to its observed software.version.
MATRIX = {
    server: {
        "schema_version": MATRIX_VERSION,
        "fhir_version": "4.0.1",
        "cases": CASES,
    }
    for server in SERVERS
}


def synthetic_patient(resource_id: str) -> dict[str, str]:
    """Create an identifier-free synthetic resource for isolated containers."""

    return {"resourceType": "Patient", "id": resource_id, "active": True}
