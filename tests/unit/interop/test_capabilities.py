"""Offline coherence tests for the integration capability matrix."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.interop.capabilities import (
    CAPABILITY_MATRIX,
    INTEGRATION_CAPABILITIES,
    CapabilityMatrix,
    IntegrationCapability,
    capability,
    validate_capability_matrix,
)

ROOT = Path(__file__).resolve().parents[3]


def test_matrix_is_sorted_and_validates_against_local_repository() -> None:
    validate_capability_matrix(CAPABILITY_MATRIX, repository_root=ROOT)

    names = [entry.name for entry in INTEGRATION_CAPABILITIES]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert all(
        entry.test_guarantee.startswith("offline-") for entry in CAPABILITY_MATRIX
    )


def test_matrix_covers_documented_optional_and_data_surfaces() -> None:
    names = {entry.name for entry in CAPABILITY_MATRIX}
    assert {
        "cda",
        "columnar",
        "dask",
        "duckdb",
        "fhir",
        "hl7v2",
        "langchain",
        "llamaindex",
        "openmrs",
        "pandas",
        "spark",
        "spacy",
    } <= names

    assert capability("Lang-Chain").name == "langchain"
    assert capability("openmrs").policy == "configured-network"
    assert capability("quickumls").policy == "user-supplied-resource"


def test_matrix_serialization_is_deterministic_and_phi_free() -> None:
    first = CAPABILITY_MATRIX.to_json()
    second = CAPABILITY_MATRIX.to_json()

    assert first == second
    assert first.endswith("\n") is False
    assert '"schema_version": 1' in first
    for sensitive_marker in (
        "Jane Roe",
        "555-0100",
        "patient-meridian-canary",
        "jane.roe@example.com",
    ):
        assert sensitive_marker not in first

    markdown = CAPABILITY_MATRIX.to_markdown()
    assert "docs/integrations/columnar-redactor.md" in markdown
    assert "tests/unit/interop/test_capabilities.py" not in markdown


def test_validation_does_not_open_network_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("capability validation must remain offline")

    monkeypatch.setattr(socket, "socket", fail_socket)
    validate_capability_matrix(repository_root=ROOT)


def test_matrix_import_is_lazy_for_optional_adapters() -> None:
    code = """
import sys
from openmed.interop.capabilities import CAPABILITY_MATRIX

assert len(CAPABILITY_MATRIX) > 0
blocked = {
    name
    for name in sys.modules
    if name in {"apache_beam", "dask", "haystack", "langchain_core", "pandas", "polars"}
}
assert not blocked, blocked
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_invalid_matrix_reports_structural_errors() -> None:
    entry = IntegrationCapability(
        name="duplicate",
        surface="Synthetic adapter",
        module="openmed.interop.synthetic",
        extra="missing-extra",
        optional_dependencies=("not/a-requirement",),
        documentation=("docs/missing.md",),
        tests=("tests/unit/missing.py",),
        policy="remote-anywhere",
        test_guarantee="online",
        description="PHI-free synthetic test record.",
    )
    matrix = CapabilityMatrix((entry, entry))

    with pytest.raises(ValueError, match="duplicate capability name"):
        validate_capability_matrix(matrix, repository_root=ROOT)
