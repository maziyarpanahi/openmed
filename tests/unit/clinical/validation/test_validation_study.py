"""Tests for the clinical-validation study runner over a synthetic sample.

These tests run entirely offline with deterministic synthetic runners. They
assert the report has the expected fields, deterministic provenance and repro
hashes, a verifiable signature, no raw PHI, and correct acceptance behaviour.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.clinical.validation import (
    CLINICAL_VALIDATION_DISCLAIMER,
    VALIDATION_PROTOCOL_ID,
    VALIDATION_REPORT_SCHEMA_VERSION,
    StudyConfig,
    ValidationReport,
    load_study_dataset,
    run_validation_study,
)
from openmed.clinical.validation.study import _DEFAULT_SIGNING_KEY

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "tests" / "fixtures" / "clinical" / "validation_sample.jsonl"

# Raw PHI tokens that must never appear anywhere in a serialized report.
PHI_TOKENS = (
    "Alice",
    "Morgan",
    "Robert",
    "Chen",
    "Dana",
    "Ruiz",
    "Priya",
    "Nair",
    "Helen",
    "Park",
    "Carmen",
    "Ortiz",
    "Diego",
    "Salas",
    "Manuel",
    "Vega",
    "4471203",
    "8830156",
    "5590231",
    "555-0142",
    "555-7788",
    "555-3311",
    "2021-03-14",
    "1948-11-02",
)

FORBIDDEN_DATA_MARKERS = ("mimic", "i2b2", "n2c2", "dua", "umls", "snomed", "cpt")


def _perfect_runner(fixture, model_name, device):
    return [
        {
            "start": span.start,
            "end": span.end,
            "label": span.label,
            "text": fixture.text[span.start : span.end],
        }
        for span in fixture.gold_spans
    ]


def _leaky_runner(fixture, model_name, device):
    """Miss every PHONE span and leak all pediatric PHI."""

    group = fixture.metadata.get("group")
    out = []
    for span in fixture.gold_spans:
        if span.label == "PHONE":
            continue
        if group == "pediatric":
            continue
        out.append(
            {
                "start": span.start,
                "end": span.end,
                "label": span.label,
                "text": fixture.text[span.start : span.end],
            }
        )
    return out


def _config(**overrides):
    base = {
        "dataset_path": SAMPLE,
        "model_name": "synthetic-model",
        "dataset_id": "synthetic-clinical-validation-sample",
        "data_revision": "git:testrev",
        "study_id": "unit-study",
    }
    base.update(overrides)
    return StudyConfig(**base)


def test_sample_dataset_is_synthetic_and_phi_offsets_are_exact() -> None:
    rows = [
        json.loads(line)
        for line in SAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = rows[0]
    assert meta["kind"] == "meta"
    assert meta["synthetic"] is True
    text_blob = SAMPLE.read_text(encoding="utf-8").lower()
    for marker in FORBIDDEN_DATA_MARKERS:
        assert marker not in text_blob
    for row in rows[1:]:
        assert row.get("synthetic") is True
        text = row["text"]
        for span in row["gold_spans"]:
            snippet = text[span["start"] : span["end"]]
            assert snippet == snippet.strip()
            assert snippet != ""


def test_load_study_dataset_skips_meta_row() -> None:
    fixtures = load_study_dataset(SAMPLE)
    assert len(fixtures) == 8
    assert all(fixture.fixture_id.startswith("val-") for fixture in fixtures)


def test_perfect_runner_meets_all_acceptance_criteria() -> None:
    report = run_validation_study(_config(), runner=_perfect_runner)
    payload = report.to_dict()

    assert payload["schema_version"] == VALIDATION_REPORT_SCHEMA_VERSION
    assert payload["protocol_id"] == VALIDATION_PROTOCOL_ID
    assert payload["fixture_count"] == 8
    assert payload["accepted"] is True
    assert payload["disclaimer"] == CLINICAL_VALIDATION_DISCLAIMER

    overall = payload["metrics"]["overall"]
    assert overall["recall"] == 1.0
    assert overall["precision"] == 1.0
    assert overall["f1"] == 1.0
    assert overall["leakage_rate"] == 0.0

    metrics_seen = {result["metric"] for result in payload["acceptance"]}
    assert {"recall", "precision", "f1", "leakage_rate"} <= metrics_seen
    assert all(result["passed"] for result in payload["acceptance"])


def test_report_has_expected_top_level_fields_and_subgroups() -> None:
    report = run_validation_study(_config(), runner=_perfect_runner)
    payload = report.to_dict()

    for field in (
        "acceptance",
        "accepted",
        "device",
        "disclaimer",
        "fixture_count",
        "metrics",
        "model_name",
        "protocol_id",
        "provenance",
        "repro_hash",
        "schema_version",
        "signature",
        "study_id",
        "subgroups",
    ):
        assert field in payload

    assert set(payload["subgroups"]) == {"group", "language"}
    assert set(payload["subgroups"]["group"]["per_group"]) == {
        "adult",
        "geriatric",
        "pediatric",
    }
    assert set(payload["subgroups"]["language"]["per_group"]) == {"en", "es"}

    provenance = payload["provenance"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", provenance["dataset_manifest_hash"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", provenance["eval_code_hash"])
    assert provenance["dataset_source"] == "user-supplied"
    assert provenance["data_revision"] == "git:testrev"


def test_report_contains_no_raw_phi() -> None:
    report = run_validation_study(_config(), runner=_perfect_runner)
    serialized = report.to_json() + "\n" + report.to_markdown()
    for token in PHI_TOKENS:
        assert token not in serialized


def test_repro_and_provenance_hashes_are_deterministic() -> None:
    first = run_validation_study(_config(), runner=_perfect_runner)
    second = run_validation_study(_config(), runner=_perfect_runner)

    assert first.repro_hash == second.repro_hash
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first.repro_hash)
    assert (
        first.provenance["dataset_manifest_hash"]
        == second.provenance["dataset_manifest_hash"]
    )
    assert first.signature.value == second.signature.value


def test_report_signature_verifies_and_detects_tampering() -> None:
    report = run_validation_study(_config(), runner=_perfect_runner)
    assert report.signature is not None
    assert report.signature.algorithm == "HMAC-SHA256"
    assert report.verify(_DEFAULT_SIGNING_KEY) is True
    assert report.verify("wrong-key") is False

    tampered = ValidationReport(
        study_id=report.study_id,
        model_name=report.model_name,
        device=report.device,
        protocol_id=report.protocol_id,
        protocol_schema_version=report.protocol_schema_version,
        fixture_count=report.fixture_count,
        metrics={"overall": {**report.metrics["overall"], "leakage_rate": 0.9}},
        subgroups=report.subgroups,
        acceptance=report.acceptance,
        provenance=report.provenance,
        repro_hash=report.repro_hash,
        signature=report.signature,
    )
    assert tampered.verify(_DEFAULT_SIGNING_KEY) is False


def test_leaky_runner_fails_leakage_and_recall_and_flags_disparity() -> None:
    report = run_validation_study(_config(model_name="leaky"), runner=_leaky_runner)
    payload = report.to_dict()

    assert payload["accepted"] is False

    results = {result["metric"]: result for result in payload["acceptance"]}
    assert results["leakage_rate"]["passed"] is False
    assert results["recall"]["passed"] is False
    assert results["subgroup_leakage_disparity"]["passed"] is False

    group_report = payload["subgroups"]["group"]
    assert group_report["worst_group"] == "pediatric"
    assert group_report["per_group"]["pediatric"]["leakage_rate"] == 1.0


def test_written_json_and_markdown_round_trip(tmp_path: Path) -> None:
    report = run_validation_study(_config(), runner=_perfect_runner)
    json_path = report.write_json(tmp_path / "report.json")
    md_path = report.write_markdown(tmp_path / "report.md")

    assert json_path.exists()
    assert md_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["repro_hash"] == report.repro_hash
    assert loaded["accepted"] is True

    markdown = md_path.read_text(encoding="utf-8")
    assert "# Clinical Validation Report" in markdown
    assert CLINICAL_VALIDATION_DISCLAIMER in markdown
    assert "PASS" in markdown
    for token in PHI_TOKENS:
        assert token not in markdown


def test_threshold_overrides_change_acceptance() -> None:
    report = run_validation_study(
        _config(threshold_overrides={"recall": {"threshold": 1.01}}),
        runner=_perfect_runner,
    )
    results = {result["metric"]: result for result in report.to_dict()["acceptance"]}
    assert results["recall"]["threshold"] == 1.01
    assert results["recall"]["passed"] is False
    assert report.accepted is False


def test_study_config_from_mapping_requires_fields() -> None:
    with pytest.raises(ValueError):
        StudyConfig.from_mapping({"model_name": "m"})

    config = StudyConfig.from_mapping(
        {
            "dataset_path": str(SAMPLE),
            "model_name": "m",
            "dataset_id": "d",
            "data_revision": "r",
        }
    )
    assert config.dataset_path == str(SAMPLE)
    assert config.device == "cpu"


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text(
        json.dumps({"kind": "meta", "synthetic": True}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        run_validation_study(_config(dataset_path=empty), runner=_perfect_runner)
