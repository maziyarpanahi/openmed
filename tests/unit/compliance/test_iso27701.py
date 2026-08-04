from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from openmed.compliance.iso27701 import (
    CAPABILITIES,
    MANIFEST_FILENAME,
    MARKDOWN_FILENAME,
    build_control_evidence_pack,
    generate_control_evidence_pack,
    load_control_evidence_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pack_enumerates_statuses_capabilities_and_evidence_pointers() -> None:
    pack = build_control_evidence_pack()
    manifest = pack.to_dict()

    statuses = {control["status"] for control in manifest["controls"]}
    capabilities = {
        capability
        for control in manifest["controls"]
        for capability in control["capabilities"]
    }
    evidence_types = {
        pointer["type"]
        for control in manifest["controls"]
        for pointer in control["evidence"]
    }

    assert statuses == {"covered", "partial", "out-of-scope"}
    assert capabilities == set(CAPABILITIES)
    assert {"config-flag", "guard-test", "report-artifact"} <= evidence_types
    assert manifest["summary"]["total"] == len(manifest["controls"])
    assert sum(
        manifest["summary"][status] for status in ("covered", "partial", "out-of-scope")
    ) == len(manifest["controls"])


def test_out_of_scope_controls_have_rationale_without_claimed_evidence() -> None:
    controls = build_control_evidence_pack().to_dict()["controls"]
    out_of_scope = [
        control for control in controls if control["status"] == "out-of-scope"
    ]

    assert out_of_scope
    assert all(control["rationale"].strip() for control in out_of_scope)
    assert all(control["capabilities"] == [] for control in out_of_scope)
    assert all(control["evidence"] == [] for control in out_of_scope)


def test_local_evidence_references_point_to_committed_sources() -> None:
    controls = build_control_evidence_pack().to_dict()["controls"]
    local_pointer_types = {"documentation", "guard-test", "implementation"}

    for control in controls:
        for pointer in control["evidence"]:
            if pointer["type"] not in local_pointer_types:
                continue
            path_text = pointer["reference"].split("::", maxsplit=1)[0]
            assert (REPO_ROOT / path_text).is_file(), pointer["reference"]


def test_generated_manifest_validates_against_committed_schema(tmp_path: Path) -> None:
    schema = load_control_evidence_schema()
    Draft202012Validator.check_schema(schema)
    result = generate_control_evidence_pack(tmp_path)

    Draft202012Validator(schema).validate(result.manifest)
    assert json.loads(result.manifest_path.read_text(encoding="utf-8")) == (
        result.manifest
    )


def test_schema_rejects_out_of_scope_control_without_rationale() -> None:
    manifest = copy.deepcopy(build_control_evidence_pack().to_dict())
    out_of_scope = next(
        control
        for control in manifest["controls"]
        if control["status"] == "out-of-scope"
    )
    out_of_scope["rationale"] = ""

    with pytest.raises(ValidationError):
        Draft202012Validator(load_control_evidence_schema()).validate(manifest)


def test_generation_writes_matching_deterministic_json_and_markdown(
    tmp_path: Path,
) -> None:
    first = generate_control_evidence_pack(tmp_path / "first")
    second = generate_control_evidence_pack(tmp_path / "second")

    assert first.manifest_path.name == MANIFEST_FILENAME
    assert first.markdown_path.name == MARKDOWN_FILENAME
    assert first.manifest == second.manifest
    assert first.markdown == second.markdown
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.markdown_path.read_bytes() == second.markdown_path.read_bytes()
    assert "## Control mapping" in first.markdown
    assert "out-of-scope" in first.markdown
    for control in first.manifest["controls"]:
        heading = (
            f"### {control['framework']} {control['control_id']} — {control['title']}"
        )
        assert heading in first.markdown
        assert f"- Status: **{control['status']}**" in first.markdown


def test_generation_runs_with_all_socket_connections_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("control-evidence generation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    result = generate_control_evidence_pack(tmp_path)

    assert result.manifest_path.is_file()
    assert result.markdown_path.is_file()
