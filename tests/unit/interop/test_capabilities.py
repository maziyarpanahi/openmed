"""Offline coherence tests for the integration capability matrix."""

from __future__ import annotations

import socket
import subprocess
import sys
import traceback
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from openmed.interop.capabilities import (
    CAPABILITY_MATRIX,
    INTEGRATION_CAPABILITIES,
    CapabilityMatrix,
    IntegrationCapability,
    capabilities,
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
        "airflow",
        "arrow_flight",
        "cda",
        "columnar",
        "dagster",
        "dask",
        "dataflow",
        "distributed_sql",
        "duckdb",
        "executable_udf",
        "fhir",
        "hl7v2",
        "langchain",
        "llamaindex",
        "openmrs",
        "pandas",
        "pandas_on_spark",
        "postgres",
        "ray_map_batches",
        "remote_function",
        "search_ingest",
        "spark",
        "spacy",
        "sqlalchemy",
        "stream_processor",
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


def test_unknown_capability_errors_do_not_echo_caller_input() -> None:
    secret = "synthetic-patient-482901"

    with pytest.raises(KeyError) as exc_info:
        CAPABILITY_MATRIX.get(secret)

    rendered = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert secret not in rendered


def test_capability_sequences_are_bounded_without_string_hooks() -> None:
    entry = CAPABILITY_MATRIX[0]

    class StringFailure(BaseException):
        pass

    class HostileValue:
        def __str__(self) -> str:
            raise StringFailure("synthetic-sensitive-string-hook")

    with pytest.raises(TypeError, match="optional_dependencies item must be a string"):
        replace(
            entry,
            optional_dependencies=(HostileValue(),),  # type: ignore[arg-type]
        )

    class EndlessStrings(Iterator[str]):
        def __iter__(self) -> EndlessStrings:
            return self

        def __next__(self) -> str:
            return "bounded-value"

    with pytest.raises(ValueError, match="optional_dependencies exceed"):
        replace(
            entry,
            optional_dependencies=EndlessStrings(),  # type: ignore[arg-type]
        )


def test_matrix_iterables_are_bounded_and_fatal_failures_are_sanitized() -> None:
    entry = CAPABILITY_MATRIX[0]

    class EndlessCapabilities(Iterator[IntegrationCapability]):
        def __iter__(self) -> EndlessCapabilities:
            return self

        def __next__(self) -> IntegrationCapability:
            return entry

    with pytest.raises(ValueError, match="capabilities exceed"):
        CapabilityMatrix(EndlessCapabilities())  # type: ignore[arg-type]

    secret = "synthetic-sensitive-iterator-hook"

    class IteratorFailure(BaseException):
        pass

    class BrokenCapabilities:
        def __iter__(self) -> Iterator[Any]:
            raise IteratorFailure(secret)

    with pytest.raises(TypeError) as exc_info:
        validate_capability_matrix(BrokenCapabilities())  # type: ignore[arg-type]

    assert secret not in str(exc_info.value)


def test_report_surfaces_revalidate_tampered_metadata_and_indent() -> None:
    secret = "synthetic-patient-482901"
    entry = replace(CAPABILITY_MATRIX[0])
    object.__setattr__(entry, "name", secret)

    with pytest.raises(ValueError, match="serialized safely") as exc_info:
        entry.to_dict()

    assert secret not in str(exc_info.value)
    with pytest.raises(ValueError, match="between 0 and 8"):
        CAPABILITY_MATRIX.to_json(indent=9)
    with pytest.raises(ValueError, match="between 0 and 8"):
        CAPABILITY_MATRIX.to_json(indent=True)  # type: ignore[arg-type]


def test_capability_properties_revalidate_tampered_requirements() -> None:
    secret = "synthetic-patient-482901"
    entry = replace(CAPABILITY_MATRIX[0])
    object.__setattr__(entry, "optional_dependencies", (f"safe>=1`{secret}",))

    for accessor in (
        lambda: entry.dependency_names,
        lambda: entry.supported_versions,
        lambda: entry.to_dict(),
    ):
        with pytest.raises(ValueError) as exc_info:
            accessor()
        assert secret not in str(exc_info.value)


def test_matrix_report_surfaces_reject_corrupted_state() -> None:
    secret = "synthetic-patient-482901"
    matrix = CapabilityMatrix((CAPABILITY_MATRIX[0],))
    object.__setattr__(matrix, "schema_version", 2)

    for accessor in (
        lambda: matrix.to_dict(),
        lambda: matrix.to_markdown(),
        lambda: tuple(matrix),
    ):
        with pytest.raises(ValueError) as exc_info:
            accessor()
        assert str(exc_info.value) == "capability matrix cannot be reported safely"
        assert secret not in str(exc_info.value)


def test_capabilities_returns_detached_canonical_records() -> None:
    secret = "synthetic-patient-482901"
    returned = capabilities()
    object.__setattr__(returned[0], "name", secret)

    assert capabilities()[0].name == "airflow"
    assert capability("airflow").name == "airflow"


def test_validation_rejects_report_injection_without_repository_checks() -> None:
    entry = replace(
        CAPABILITY_MATRIX[0],
        optional_dependencies=("safe>=1`unsafe",),
        documentation=("docs/integrations/matrix.md)unsafe",),
    )

    with pytest.raises(ValueError, match="unsafe report metadata"):
        validate_capability_matrix(CapabilityMatrix((entry,)), repository_root=None)


def test_schema_version_is_bounded_before_serialization() -> None:
    with pytest.raises(ValueError, match="outside the supported integer range"):
        CapabilityMatrix((CAPABILITY_MATRIX[0],), schema_version=1 << 80)
