"""Unit tests for eval dataset cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pytest

from openmed.eval import (
    DATA_PROVENANCE_ASSERTION,
    DATASET_CARD_SUITES,
    ModelCardArtifact,
    ModelCardProvenanceError,
    build_all_dataset_cards,
    build_dataset_card,
    build_eval_model_card,
)
from openmed.eval.datasets import license_for
from openmed.eval.golden import GOLDEN_CATEGORIES, load_golden_fixtures
from openmed.eval.suites import DRUGPROT, GOLDEN, MASAKHANER, NAAMAPADAM, SHIELD
from openmed.eval.suites.shield import (
    PUBLIC_SAMPLE_NOTES_CONFIG,
    PUBLIC_SAMPLE_SPANS_CONFIG,
)

DRUGPROT_FIXTURE_DIR = (
    Path(__file__).parents[2] / "fixtures" / "drugprot_synthetic" / "training"
)


def test_build_all_dataset_cards_is_offline_and_covers_concrete_suites() -> None:
    cards = build_all_dataset_cards()

    assert tuple(card.dataset for card in cards) == DATASET_CARD_SUITES
    assert tuple(card.dataset for card in cards) == (
        GOLDEN,
        SHIELD,
        DRUGPROT,
        MASAKHANER,
        NAAMAPADAM,
    )
    for card in cards:
        dataset_license = license_for(card.dataset)
        assert card.license_id == dataset_license.license_id
        assert card.source_url == dataset_license.source_url
        assert card.redistribution == dataset_license.redistribution

    external_counts = {
        card.dataset: card.record_count
        for card in cards
        if card.dataset in {SHIELD, DRUGPROT, MASAKHANER}
    }
    assert external_counts == {SHIELD: 0, DRUGPROT: 0, MASAKHANER: 0}


def test_golden_card_counts_committed_fixtures_without_text() -> None:
    fixtures = load_golden_fixtures()
    card = build_dataset_card(GOLDEN)

    assert card.record_count == len(fixtures)
    assert card.splits == tuple(sorted(GOLDEN_CATEGORIES))
    assert "SSN" in card.labels
    assert "en" in card.languages

    rendered = card.to_json() + card.to_markdown()
    for fixture in fixtures:
        assert fixture.text not in rendered


def test_dataset_card_markdown_and_json_are_byte_stable() -> None:
    first = build_dataset_card(GOLDEN)
    second = build_dataset_card(GOLDEN)

    assert first.to_markdown() == second.to_markdown()
    assert first.to_json() == second.to_json()


def test_shield_card_counts_explicit_rows_and_uses_license_registry() -> None:
    text = "Jordan Smith 555-0100"
    spans = [
        _shield_span(text, "patient", "Jordan Smith"),
        _shield_span(text, "phone", "555-0100"),
    ]

    def rows_loader(
        repository: str,
        config: str,
        split: str,
    ) -> list[Mapping[str, object]]:
        del repository, split
        if config == PUBLIC_SAMPLE_NOTES_CONFIG:
            return [
                {
                    "note_id": "shield-card-1",
                    "note_text": text,
                    "note_type": "synthetic_unit",
                }
            ]
        if config == PUBLIC_SAMPLE_SPANS_CONFIG:
            return spans
        raise AssertionError(f"unexpected config: {config}")

    card = build_dataset_card(SHIELD, rows_loader=rows_loader)
    dataset_license = license_for(SHIELD)

    assert card.record_count == 1
    assert card.license_id == dataset_license.license_id
    assert card.source_url == dataset_license.source_url
    assert card.labels == (
        "AGE",
        "DATE",
        "ID_NUM",
        "LOCATION",
        "ORGANIZATION",
        "PERSON",
        "PHONE",
        "URL",
    )
    assert card.languages == ("en",)
    assert card.splits == ("train",)
    assert text not in card.to_json()
    assert text not in card.to_markdown()


def test_drugprot_card_counts_explicit_fixture_path_without_text() -> None:
    card = build_dataset_card(DRUGPROT, path=DRUGPROT_FIXTURE_DIR)
    dataset_license = license_for(DRUGPROT)

    assert card.record_count == 1
    assert card.license_id == dataset_license.license_id
    assert card.source_url == dataset_license.source_url
    assert card.labels == ("OTHER",)
    assert card.languages == ("en",)
    assert card.splits == ("training",)

    rendered = card.to_json() + card.to_markdown()
    assert "Aspirin inhibits TP53" not in rendered
    assert "Metformin activates EGFR" not in rendered


def test_eval_model_card_claims_are_complete_against_provenance(
    tmp_path: Path,
) -> None:
    card = build_eval_model_card(
        _eval_model_card_artifacts(tmp_path),
        data_sources=[_synthetic_source()],
    )
    datasheet = json.loads(card.to_json())
    manifest = datasheet["provenance_manifest"]["artifacts"]

    assert datasheet["data_provenance"]["assertion"] == DATA_PROVENANCE_ASSERTION
    assert datasheet["data_provenance"]["permissive_or_synthetic_only"] is True
    assert datasheet["data_provenance"]["restricted_source_ids"] == []
    assert set(manifest) == {
        "calibration_report",
        "coverage_report",
        "fairness_report",
        "gate_report",
        "transfer_matrix",
    }
    assert datasheet["quantitative_claims"]
    for claim in datasheet["quantitative_claims"]:
        artifact = manifest[claim["artifact_id"]]
        assert claim["artifact_hash"] == artifact["sha256"]

    claim_ids = {claim["claim_id"] for claim in datasheet["quantitative_claims"]}
    assert "gate.per_language_leakage.en" in claim_ids
    assert "fairness.leakage_disparity" in claim_ids
    assert "calibration.groups.OpenMed/pii-tiny.PERSON.en.chosen_threshold" in claim_ids
    assert "coverage.fixture_count" in claim_ids
    assert "transfer.en.fr" in claim_ids

    markdown = card.to_markdown()
    assert "| Leakage rate for en | 0 | `gate_report` |" in markdown
    assert "| Fairness leakage disparity | 0.02 | `fairness_report` |" in markdown


def test_eval_model_card_verification_refuses_tampered_source(
    tmp_path: Path,
) -> None:
    artifacts = _eval_model_card_artifacts(tmp_path)
    card = build_eval_model_card(artifacts, data_sources=[_synthetic_source()])

    gate_path = tmp_path / "gate-report.json"
    tampered_gate = _gate_report_payload()
    tampered_gate["residual_leakage_rate"] = 0.2
    _write_json(gate_path, tampered_gate)

    with pytest.raises(ModelCardProvenanceError, match="gate_report hash mismatch"):
        build_eval_model_card(
            artifacts,
            data_sources=[_synthetic_source()],
            verify=True,
            provenance_manifest=card.provenance_manifest,
        )


def test_eval_model_card_markdown_and_datasheet_are_byte_stable(
    tmp_path: Path,
) -> None:
    first = build_eval_model_card(
        _eval_model_card_artifacts(tmp_path),
        data_sources=[_synthetic_source()],
    )
    second = build_eval_model_card(
        _eval_model_card_artifacts(tmp_path),
        data_sources=[_synthetic_source()],
    )

    assert first.to_markdown() == second.to_markdown()
    assert first.to_json() == second.to_json()


def test_eval_model_datasheet_rejects_dua_source_ids(tmp_path: Path) -> None:
    with pytest.raises(ModelCardProvenanceError, match="i2b2"):
        build_eval_model_card(
            _eval_model_card_artifacts(tmp_path),
            data_sources=[
                {
                    "license_id": "local-dua-required",
                    "source_id": "i2b2_eval_only",
                    "synthetic": False,
                }
            ],
        )


def test_eval_model_card_artifact_hashes_are_content_stable(
    tmp_path: Path,
) -> None:
    first_path = _write_json(tmp_path / "first-gate.json", _gate_report_payload())
    second_path = _write_json(tmp_path / "second-gate.json", _gate_report_payload())

    first = build_eval_model_card(
        [ModelCardArtifact.from_path("gate_report", "gate_report", first_path)],
        data_sources=[_synthetic_source()],
    )
    second = build_eval_model_card(
        [ModelCardArtifact.from_path("gate_report", "gate_report", second_path)],
        data_sources=[_synthetic_source()],
    )

    first_hash = first.provenance_manifest["artifacts"]["gate_report"]["sha256"]
    second_hash = second.provenance_manifest["artifacts"]["gate_report"]["sha256"]
    assert first_hash == second_hash


def _shield_span(text: str, label: str, value: str) -> dict[str, object]:
    start = text.index(value)
    return {
        "note_id": "shield-card-1",
        "span_end": start + len(value),
        "span_label": label,
        "span_start": start,
    }


def _eval_model_card_artifacts(tmp_path: Path) -> list[ModelCardArtifact]:
    return [
        ModelCardArtifact.from_path(
            "gate_report",
            "gate_report",
            _write_json(tmp_path / "gate-report.json", _gate_report_payload()),
        ),
        ModelCardArtifact.from_path(
            "fairness_report",
            "fairness_report",
            _write_json(tmp_path / "fairness-report.json", _fairness_report_payload()),
        ),
        ModelCardArtifact.from_path(
            "calibration_report",
            "calibration_report",
            _write_json(
                tmp_path / "calibration-report.json",
                _calibration_report_payload(),
            ),
        ),
        ModelCardArtifact.from_path(
            "coverage_report",
            "coverage_report",
            _write_json(tmp_path / "coverage-report.json", _coverage_report_payload()),
        ),
        ModelCardArtifact.from_path(
            "transfer_matrix",
            "transfer_matrix",
            _write_json(tmp_path / "transfer-matrix.json", _transfer_matrix_payload()),
        ),
    ]


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _synthetic_source() -> dict[str, object]:
    return {
        "license_id": "Apache-2.0",
        "source_id": "synthetic_golden_deid",
        "synthetic": True,
    }


def _gate_report_payload() -> dict[str, object]:
    return {
        "critical_leakage_count": 0,
        "decision": "RELEASABLE",
        "metrics": {
            "leakage": {
                "leaked_chars_by_language": {"en": 0, "fr": 1},
                "total_chars_by_language": {"en": 100, "fr": 100},
            }
        },
        "p50_ms": 42.0,
        "p95_ms": 99.0,
        "per_label_precision": {"DATE": 0.98, "PERSON": 0.99},
        "per_label_recall": {"DATE": 0.991, "PERSON": 0.992},
        "policy": "hipaa_safe_harbor",
        "repo_id": "OpenMed/pii-tiny",
        "residual_leakage_rate": 0.01,
        "target_leakage_rate": 0.02,
        "threshold_profile": "strict",
    }


def _fairness_report_payload() -> dict[str, object]:
    return {
        "fixture_count": 8,
        "leakage_disparity": 0.02,
        "model_name": "OpenMed/pii-tiny",
        "per_group": {
            "adult": {
                "covered_chars": 100,
                "leakage_rate": 0.0,
                "leaked_chars": 0,
                "recall": 1.0,
                "span_count": 4,
                "total_chars": 100,
            },
            "older_adult": {
                "covered_chars": 98,
                "leakage_rate": 0.02,
                "leaked_chars": 2,
                "recall": 0.98,
                "span_count": 4,
                "total_chars": 100,
            },
        },
        "suite": "golden",
        "worst_group": "older_adult",
        "worst_group_leakage": 0.02,
    }


def _calibration_report_payload() -> dict[str, object]:
    return {
        "groups": [
            {
                "chosen_threshold": 0.91,
                "label": "PERSON",
                "language": "en",
                "model_id": "OpenMed/pii-tiny",
                "negative_weight": 12.0,
                "over_redaction": 0.01,
                "positive_weight": 20.0,
                "precision": 0.99,
                "recall": 0.995,
                "resulting_leakage": 0.0,
                "target_leakage": 0.0,
            },
            {
                "chosen_threshold": 0.88,
                "label": "DATE",
                "language": "fr",
                "model_id": "OpenMed/pii-tiny",
                "negative_weight": 8.0,
                "over_redaction": 0.02,
                "positive_weight": 10.0,
                "precision": 0.98,
                "recall": 0.992,
                "resulting_leakage": 0.0,
                "target_leakage": 0.0,
            },
        ],
        "min_recall": 0.99,
        "model_id": "OpenMed/pii-tiny",
        "suite": "golden",
        "target_leakage": 0.0,
    }


def _coverage_report_payload() -> dict[str, object]:
    return {
        "categories": {"covered": ["multilingual"], "missing": []},
        "category_counts": {"multilingual": 2},
        "fixture_count": 12,
        "labels": {"covered": ["DATE", "PERSON"], "missing": []},
        "languages": {"covered": ["en", "fr"], "missing": []},
    }


def _transfer_matrix_payload() -> dict[str, object]:
    return {
        "matrix": {
            "en": {"fr": 0.97},
            "fr": {"en": 0.96},
        },
        "source_languages": ["en", "fr"],
        "target_languages": ["en", "fr"],
    }
