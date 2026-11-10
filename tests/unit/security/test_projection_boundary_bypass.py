"""Fail-closed cross-namespace and policy-bypass tests."""

from __future__ import annotations

from pathlib import Path

from openmed.compliance.projections import (
    ProjectionBoundary,
    ProjectionNamespace,
    ProjectionOperation,
    ProjectionPolicyRequest,
)
from openmed.structured.store import StoreState

T0 = "2026-01-02T03:04:05Z"


def _request(
    namespace: ProjectionNamespace,
    operation: ProjectionOperation,
    *,
    role: str = "clinician",
    attributes: tuple[str, ...] = (),
) -> ProjectionPolicyRequest:
    return ProjectionPolicyRequest(
        operation=operation,
        namespace=namespace,
        purpose="care",
        role=role,
        attributes=attributes,
    )


def test_deidentified_api_rejects_identified_namespace_request(tmp_path: Path) -> None:
    boundary = ProjectionBoundary(tmp_path / "projections")
    api = boundary.deidentified()

    result = api.write(
        "projection_aaaaaaaaaaaaaaaa",
        b"synthetic boundary canary",
        _request(
            ProjectionNamespace.IDENTIFIED,
            ProjectionOperation.WRITE,
            attributes=("identified_access",),
        ),
        consent_scope="care",
        consent_revision="v1",
        occurred_at=T0,
    )

    assert result.state is StoreState.DENIED
    assert result.code == "namespace_mismatch"
    identified_objects = tmp_path / "projections" / "identified" / "objects"
    assert not tuple(path for path in identified_objects.rglob("*") if path.is_file())
    boundary.close()


def test_role_and_operation_mismatch_cannot_bypass_policy(tmp_path: Path) -> None:
    boundary = ProjectionBoundary(tmp_path / "projections")
    api = boundary.deidentified()

    role_denied = api.write(
        "projection_aaaaaaaaaaaaaaaa",
        b"synthetic boundary canary",
        _request(
            ProjectionNamespace.DEIDENTIFIED,
            ProjectionOperation.WRITE,
            role="guest",
        ),
        consent_scope="care",
        consent_revision="v1",
        occurred_at=T0,
    )
    operation_denied = api.write(
        "projection_bbbbbbbbbbbbbbbb",
        b"synthetic boundary canary",
        _request(
            ProjectionNamespace.DEIDENTIFIED,
            ProjectionOperation.READ,
        ),
        consent_scope="care",
        consent_revision="v1",
        occurred_at=T0,
    )

    assert role_denied.state is StoreState.DENIED
    assert role_denied.code == "role_denied"
    assert operation_denied.state is StoreState.DENIED
    assert operation_denied.code == "operation_mismatch"
    assert boundary.integrity_check().ok
    boundary.close()


def test_namespace_symlink_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "projections"
    external = tmp_path / "external"
    external.mkdir()
    root.mkdir()
    (root / "identified").symlink_to(external, target_is_directory=True)

    result = ProjectionBoundary.open(root)

    assert result.state is StoreState.FAILURE
    assert result.code == "projection_open_failed"
