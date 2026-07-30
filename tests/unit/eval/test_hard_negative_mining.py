"""Tests for corpus-scale synthetic hard-negative mining."""

from __future__ import annotations

import json

import pytest

from openmed.eval.error_analysis import hard_negative_over_redaction_report
from openmed.eval.golden import (
    HARD_NEGATIVE_CATEGORY,
    GoldenFixture,
    HardNegativeMiningConfig,
    MemoryBudgetExceeded,
    build_hard_negative_fixture_pack,
    mine_hard_negative_candidates,
)
from openmed.eval.harness import BenchmarkFixture


def test_streaming_miner_reproducible_dedup_and_memory_budget(tmp_path) -> None:
    shard_a = tmp_path / "shard-a.jsonl"
    shard_b = tmp_path / "shard-b.jsonl"
    shard_a.write_text(
        "\n".join(json.dumps(row) for row in _synthetic_records()[:2]) + "\n",
        encoding="utf-8",
    )
    shard_b.write_text(
        json.dumps(_synthetic_records()[2]) + "\n",
        encoding="utf-8",
    )
    config = HardNegativeMiningConfig(
        seed=27,
        min_difficulty_score=0.1,
        per_label_language_limit=1,
        peak_memory_budget_bytes=64 * 1024 * 1024,
    )

    first = mine_hard_negative_candidates(
        [shard_a, shard_b],
        config=config,
        runner=_metadata_runner,
        reference_surfaces={
            "DATE": ["2026-01-01", "2026/01/02"],
            "EMAIL": ["patient@example.test"],
            "ID_NUM": ["ZX-12345"],
            "PERSON": ["Alex Mercer"],
        },
    )
    second = mine_hard_negative_candidates(
        [shard_a, shard_b],
        config=config,
        runner=_metadata_runner,
        reference_surfaces={
            "DATE": ["2026-01-01", "2026/01/02"],
            "EMAIL": ["patient@example.test"],
            "ID_NUM": ["ZX-12345"],
            "PERSON": ["Alex Mercer"],
        },
    )

    assert first.scanned_records == 3
    assert first.peak_memory_bytes <= config.peak_memory_budget_bytes
    assert first.ranking_to_json() == second.ranking_to_json()
    assert first.difficulty_report.near_duplicate_retention_rate <= 0.02
    assert first.difficulty_report.duplicate_candidates_seen >= 1
    assert any(
        candidate.model_false_positive_score > 0
        for candidate in first.selected_candidates
    )

    selected_id_negatives = [
        candidate
        for candidate in first.selected_candidates
        if candidate.label == "ID_NUM" and candidate.language == "en"
    ]
    assert len(selected_id_negatives) == 1


def test_mining_memory_guard_fails_when_configured_budget_is_exceeded() -> None:
    config = HardNegativeMiningConfig(
        seed=1,
        min_difficulty_score=0.1,
        peak_memory_budget_bytes=1,
    )

    with pytest.raises(MemoryBudgetExceeded):
        mine_hard_negative_candidates(
            [_synthetic_records()[:1]],
            config=config,
            runner=_metadata_runner,
            reference_surfaces={"ID_NUM": ["ZX-12345"]},
        )


def test_selected_candidates_build_synthetic_zero_span_fixture_pack() -> None:
    result = mine_hard_negative_candidates(
        [_synthetic_records()],
        config=HardNegativeMiningConfig(
            seed=11,
            min_difficulty_score=0.1,
            per_label_language_limit=1,
        ),
        runner=_metadata_runner,
        reference_surfaces={"DATE": ["2026-01-01"], "ID_NUM": ["ZX-12345"]},
    )

    pack = build_hard_negative_fixture_pack(result.selected_candidates)
    fixtures = [GoldenFixture.from_mapping(row) for row in pack["fixtures"]]

    assert pack["synthetic"] is True
    assert "i2b2" not in json.dumps(pack).lower()
    assert "n2c2" not in json.dumps(pack).lower()
    assert "mimic" not in json.dumps(pack).lower()
    assert fixtures
    for fixture in fixtures:
        assert fixture.category == HARD_NEGATIVE_CATEGORY
        assert fixture.gold_spans == ()
        assert fixture.metadata["synthetic"] is True
        for candidate in fixture.metadata["hard_negative_candidates"]:
            assert candidate["synthetic"] is True
            assert (
                fixture.text[candidate["start"] : candidate["end"]]
                == (candidate["text"])
            )


def test_hard_negatives_raise_over_redaction_rate_on_heldout_check() -> None:
    easy = BenchmarkFixture.from_mapping(
        {
            "gold_spans": [],
            "id": "easy-negative",
            "language": "en",
            "metadata": {
                "category": HARD_NEGATIVE_CATEGORY,
                "hard_negative_candidates": [
                    {
                        "difficulty_score": 0.1,
                        "end": 16,
                        "label": "PERSON",
                        "start": 0,
                        "synthetic": True,
                        "text": "routine followup",
                    }
                ],
                "synthetic": True,
            },
            "text": "routine followup stayed visible.",
        }
    )
    result = mine_hard_negative_candidates(
        [_synthetic_records()],
        config=HardNegativeMiningConfig(
            seed=9,
            min_difficulty_score=0.1,
            per_label_language_limit=1,
        ),
        runner=_metadata_runner,
        reference_surfaces={"DATE": ["2026-01-01"], "ID_NUM": ["ZX-12345"]},
    )
    hard_pack = build_hard_negative_fixture_pack(result.selected_candidates)
    hard_fixtures = [
        BenchmarkFixture.from_mapping(row) for row in hard_pack["fixtures"]
    ]

    baseline = hard_negative_over_redaction_report(
        "heldout-test-model",
        [easy],
        runner=_difficulty_threshold_runner,
    )
    enriched = hard_negative_over_redaction_report(
        "heldout-test-model",
        hard_fixtures,
        runner=_difficulty_threshold_runner,
    )

    assert baseline.over_redaction_rate == 0.0
    assert enriched.over_redaction_rate > baseline.over_redaction_rate
    assert enriched.over_redacted_candidates > 0


def _synthetic_records() -> list[dict[str, object]]:
    first_text = (
        "Patient Alex Mercer had ID ZX-12345. Training bin MRN-12345 "
        "stayed visible. Date 2026-01-01 was gold; code A1-2026 stayed "
        "visible. demo-alert@example.invalid was example only."
    )
    second_text = (
        "Synthetic duplicate row kept MRN 12345 visible while 2026/01/02 "
        "remained a date reference."
    )
    third_text = (
        "Nora Patel signed a fictional roster; Riverside Training Ward was "
        "a course name."
    )
    return [
        _record(
            fixture_id="synthetic-shard-a-1",
            text=first_text,
            gold_values=[
                ("Alex Mercer", "PERSON"),
                ("ZX-12345", "ID_NUM"),
                ("2026-01-01", "DATE"),
            ],
            candidates=[
                ("MRN-12345", "ID_NUM"),
                ("A1-2026", "DATE"),
                ("demo-alert@example.invalid", "EMAIL"),
            ],
        ),
        _record(
            fixture_id="synthetic-shard-a-2",
            text=second_text,
            gold_values=[("2026/01/02", "DATE")],
            candidates=[("MRN 12345", "ID_NUM")],
        ),
        _record(
            fixture_id="synthetic-shard-b-1",
            text=third_text,
            gold_values=[],
            candidates=[
                ("Nora Patel", "PERSON"),
                ("Riverside Training Ward", "ORGANIZATION"),
            ],
        ),
    ]


def _record(
    *,
    fixture_id: str,
    text: str,
    gold_values: list[tuple[str, str]],
    candidates: list[tuple[str, str]],
) -> dict[str, object]:
    candidate_rows = [
        {
            **_span(text, value, label),
            "reason": "annotated_non_phi",
            "synthetic": True,
        }
        for value, label in candidates
    ]
    return {
        "gold_spans": [_span(text, value, label) for value, label in gold_values],
        "id": fixture_id,
        "language": "en",
        "metadata": {
            "category": "synthetic_mining_shard",
            "hard_negative_candidates": candidate_rows,
            "predicted_spans": [
                {
                    **_span(text, value, label),
                    "metadata": {"confidence": 0.92},
                }
                for value, label in candidates
            ],
            "source": "synthetic_corpus",
            "synthetic": True,
        },
        "text": text,
    }


def _span(text: str, value: str, label: str) -> dict[str, object]:
    start = text.index(value)
    return {
        "end": start + len(value),
        "label": label,
        "start": start,
        "text": value,
    }


def _metadata_runner(fixture, model_name, device):
    assert model_name == "hard-negative-miner"
    assert device == "cpu"
    return fixture.metadata.get("predicted_spans", [])


def _difficulty_threshold_runner(fixture, model_name, device):
    assert model_name == "heldout-test-model"
    assert device == "cpu"
    predictions = []
    for candidate in fixture.metadata.get("hard_negative_candidates", []):
        if candidate.get("difficulty_score", 0.0) < 0.75:
            continue
        predictions.append(
            {
                "end": candidate["end"],
                "label": candidate["label"],
                "metadata": {"confidence": 0.9},
                "start": candidate["start"],
            }
        )
    return predictions
