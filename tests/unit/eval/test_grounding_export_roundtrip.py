"""Focused synthetic grounding export and MedMentions metric contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.clinical.grounding import Candidate
from openmed.eval.golden.loader import list_fixture_paths
from openmed.eval.medmentions_linking import (
    MEDMENTIONS_TOP1_FLOOR,
    MEDMENTIONS_TOP1_TARGET,
    evaluate_medmentions_st21pv,
)
from openmed.eval.suites import grounding_export as grounding_export_suite
from openmed.eval.suites.grounding_export import (
    run_grounding_export_suite,
    validate_fhir_r4_shape,
)


def test_grounding_export_fixture_is_not_generic_deidentification_gold() -> None:
    assert all(path.name != "grounding_export.jsonl" for path in list_fixture_paths())


def test_synthetic_grounding_export_roundtrip_passes_offline_smoke() -> None:
    report = run_grounding_export_suite()

    assert report.fixture_count == 4
    assert report.metrics["passed"] is True
    assert report.metrics["fhir"]["errors"] == 0
    assert report.metrics["fhir"]["official_validator_executed"] is False
    assert report.metrics["omop"]["achilles_smoke_passed"] is True
    assert report.metadata["synthetic"] is True


def test_structural_fhir_check_rejects_deliberately_broken_resource() -> None:
    broken = {
        "resourceType": "Observation",
        "status": "final",
        "subject": {"reference": "Patient/synthetic"},
    }

    assert validate_fhir_r4_shape(broken) == ("Observation.code is missing",)


def test_official_validator_allows_only_openmed_extension_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator_jar = tmp_path / "validator.jar"
    validator_jar.write_bytes(b"synthetic-validator")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        commands.append(command)
        output = Path(command[command.index("-output") + 1])
        output.write_text(
            json.dumps({"resourceType": "OperationOutcome", "issue": []}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(grounding_export_suite.subprocess, "run", fake_run)

    result = grounding_export_suite.validate_with_hl7_validator(
        {"resourceType": "Bundle", "type": "collection", "entry": []},
        validator_jar=validator_jar,
    )

    assert result.errors == 0
    assert commands
    extension_index = commands[0].index("-extension")
    assert commands[0][extension_index + 1] == (
        "https://openmed.ai/fhir/StructureDefinition/"
    )


def test_medmentions_top1_report_enforces_floor_without_bundling_corpus(
    tmp_path: Path,
) -> None:
    path = tmp_path / "caller_projection.jsonl"
    rows = [
        {"mention": f"synthetic mention {index}", "cui": f"C{index}"}
        for index in range(4)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    def provider(mention: str, top_k: int):
        index = int(mention.rsplit(" ", 1)[1])
        code = f"C{index}" if index < 3 else "WRONG"
        return [Candidate("UMLS", code, "synthetic concept", 1.0)][:top_k]

    report = evaluate_medmentions_st21pv(path, provider=provider)

    assert report.metrics["top1_accuracy"] == 0.75
    assert report.metrics["floor"] == MEDMENTIONS_TOP1_FLOOR == 0.55
    assert report.metrics["target"] == MEDMENTIONS_TOP1_TARGET == 0.70
    assert report.metrics["passed"] is True
    assert report.metadata["corpus_bundled"] is False
    assert report.metadata["restricted_vocabulary_bundled"] is False
