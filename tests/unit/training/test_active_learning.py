from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openmed.cli import main_module
from openmed.training.active_learning import ActiveLearningQueue
from openmed.training.adjudication import (
    AdjudicationCandidate,
    make_adjudication_item,
)


def test_gate_failure_ingestion_prioritizes_critical_labels(tmp_path: Path) -> None:
    queue = ActiveLearningQueue(tmp_path / "queue.jsonl")

    queue.ingest_gate_report(
        {
            "repo_id": "OpenMed/unit-model",
            "decision": "QUARANTINED",
            "per_label_recall": {"PERSON": 0.98, "SSN": 0.995},
            "critical_leakage_count": 1,
            "failure_spans": [
                {
                    "start": 8,
                    "end": 20,
                    "label": "PERSON",
                    "text": "Jordan Smith",
                },
                {
                    "start": 42,
                    "end": 53,
                    "label": "SSN",
                    "text": "123-45-6789",
                },
            ],
        }
    )

    batch = queue.next_batch_dicts(2)

    assert [row["label"] for row in batch] == ["SSN", "PERSON"]
    assert batch[0]["reason"] == "critical_leakage"
    assert batch[1]["reason"] == "low_recall"


def test_labeled_items_are_not_requeued_across_runs(tmp_path: Path) -> None:
    event_log = tmp_path / "queue.jsonl"
    item = make_adjudication_item(
        text="Patient Jordan Smith was discharged.",
        candidates=[
            AdjudicationCandidate(
                start=8,
                end=20,
                label="PERSON",
                text="Jordan Smith",
                score=0.52,
                sources=("teacher_a",),
            ),
            AdjudicationCandidate(
                start=8,
                end=20,
                label="LAST_NAME",
                text="Jordan Smith",
                score=0.48,
                sources=("teacher_b",),
            ),
        ],
        reason="inter_model_disagreement",
        record_id="note-1",
    )

    queue = ActiveLearningQueue(event_log)
    queue.ingest_adjudication(item)
    candidate = queue.next_batch(1)[0]
    labeled_hash = queue.mark_labeled(candidate)

    reopened = ActiveLearningQueue(event_log)
    reopened.ingest_adjudication(item)

    assert labeled_hash in reopened.labeled_hashes
    assert reopened.next_batch(10) == ()


def test_priority_bands_keep_gate_critical_and_adjudication_first(
    tmp_path: Path,
) -> None:
    queue = ActiveLearningQueue(tmp_path / "queue.jsonl")
    queue.ingest_gate_report(
        {
            "repo_id": "OpenMed/unit-model",
            "per_label_recall": {"PERSON": 0.0, "SSN": 0.99},
            "critical_leakage_count": 1,
            "failure_spans": [
                {"start": 8, "end": 20, "label": "PERSON"},
                {"start": 42, "end": 53, "label": "SSN"},
            ],
        }
    )
    queue.ingest_adjudication(
        make_adjudication_item(
            text="Call 555-111-2222.",
            candidates=[
                AdjudicationCandidate(
                    start=5,
                    end=17,
                    label="PHONE",
                    score=0.7,
                    sources=("teacher_a",),
                )
            ],
            reason="inter_model_disagreement",
            record_id="note-2",
        )
    )

    batch = queue.next_batch_dicts(3)

    assert [row["label"] for row in batch] == ["SSN", "PHONE", "PERSON"]


def test_batch_jsonl_is_bounded_and_never_carries_raw_text(tmp_path: Path) -> None:
    event_log = tmp_path / "queue.jsonl"
    queue = ActiveLearningQueue(event_log)
    queue.ingest_gate_report(
        {
            "repo_id": "OpenMed/unit-model",
            "per_label_recall": {"PERSON": 0.97, "ID_NUM": 0.995},
            "critical_leakage_count": 1,
            "failure_spans": [
                {"start": 8, "end": 20, "label": "PERSON", "text": "Jordan Smith"},
                {"start": 42, "end": 53, "label": "ID_NUM", "text": "MRN-12345"},
            ],
        }
    )

    payload = queue.next_batch_jsonl(1)
    rows = _jsonl_rows(payload)

    assert len(rows) == 1
    assert "Jordan Smith" not in payload
    assert "MRN-12345" not in payload
    assert "Jordan Smith" not in event_log.read_text(encoding="utf-8")
    assert "MRN-12345" not in event_log.read_text(encoding="utf-8")
    assert not _contains_key(rows[0], "text")
    assert not _contains_key(rows[0], "source_text")


def test_active_learning_next_batch_cli_emits_jsonl(
    tmp_path: Path,
    capsys,
) -> None:
    event_log = tmp_path / "queue.jsonl"
    queue = ActiveLearningQueue(event_log)
    queue.ingest_gate_report(
        {
            "repo_id": "OpenMed/unit-model",
            "per_label_recall": {"PERSON": 0.98},
            "failure_spans": [{"start": 8, "end": 20, "label": "PERSON"}],
        }
    )

    exit_code = main_module.main(
        [
            "active-learning",
            "next-batch",
            "--state",
            str(event_log),
            "--size",
            "1",
        ]
    )

    assert exit_code == 0
    rows = _jsonl_rows(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["source"] == "release_gate"
    assert rows[0]["label"] == "PERSON"


def _jsonl_rows(payload: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _contains_key(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False
