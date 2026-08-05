from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

dagster = pytest.importorskip("dagster")

from openmed.integrations import dagster_assets  # noqa: E402, I001


_SOURCE_ROWS = [
    {
        "record_id": "synthetic-001",
        "note": "Synthetic Patient John Doe called 555-0101.",
        "status": "ready",
    },
    {
        "record_id": "synthetic-002",
        "note": "Synthetic Patient Jane Roe emailed jane@example.test.",
        "status": "ready",
    },
]


def test_partitioned_asset_materializes_synthetic_rows_and_emits_phi_free_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dagster_assets, "process_batch", _fake_process_batch)

    result = dagster.materialize(
        [dagster_assets.redacted_dataset],
        resources={"source_dataset": _SOURCE_ROWS},
        partition_key="2026-01-01",
        run_config={
            "ops": {
                "redacted_dataset": {
                    "config": {
                        "policy_profile": "strict_no_leak",
                        "text_columns": ["note"],
                    }
                }
            }
        },
    )

    assert result.success
    assert result.output_for_node("redacted_dataset") == [
        {
            "record_id": "synthetic-001",
            "note": "Synthetic Patient [PERSON] called [PHONE].",
            "status": "ready",
        },
        {
            "record_id": "synthetic-002",
            "note": "Synthetic Patient [PERSON] emailed [EMAIL].",
            "status": "ready",
        },
    ]

    materialization = next(
        event.event_specific_data.materialization
        for event in result.all_events
        if event.is_step_materialization
    )
    metadata = {name: value.value for name, value in materialization.metadata.items()}
    assert metadata["row_count"] == 2
    assert metadata["redacted_rows"] == 2
    assert metadata["redacted_cells"] == 2
    assert metadata["redacted_spans"] == 4
    assert metadata["per_label_counts"] == {
        "EMAIL": 1,
        "PERSON": 2,
        "PHONE": 1,
    }
    assert metadata["raw_text_included"] is False

    rendered_metadata = json.dumps(metadata, sort_keys=True)
    for token in (
        "John Doe",
        "Jane Roe",
        "555-0101",
        "jane@example.test",
    ):
        assert token not in rendered_metadata


def test_op_redacts_input_rows_and_forwards_config(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def recording_process_batch(texts: list[str], **kwargs: Any) -> Any:
        calls.append(kwargs)
        return _fake_process_batch(texts, **kwargs)

    monkeypatch.setattr(dagster_assets, "process_batch", recording_process_batch)

    @dagster.job
    def deidentify_job(source_dataset):
        dagster_assets.deidentify_dataset_op(source_dataset)

    result = deidentify_job.execute_in_process(
        input_values={"source_dataset": _SOURCE_ROWS},
        run_config={
            "ops": {
                "deidentify_dataset_op": {
                    "config": {
                        "policy": "hipaa_safe_harbor",
                        "text_columns": ["note"],
                        "method": "replace",
                        "confidence_threshold": 0.9,
                    }
                }
            }
        },
    )

    assert result.success
    assert result.output_for_node("deidentify_dataset_op")[0]["note"] == (
        "Synthetic Patient [PERSON] called [PHONE]."
    )
    assert calls == [
        {
            "operation": "deidentify",
            "method": "replace",
            "model_name": dagster_assets.DEFAULT_MODEL_NAME,
            "policy": "hipaa_safe_harbor",
            "confidence_threshold": 0.9,
        }
    ]


def test_asset_config_rejects_unknown_policy_profile() -> None:
    with pytest.raises(dagster.DagsterInvalidConfigError, match="not in enum"):
        dagster.materialize(
            [dagster_assets.redacted_dataset],
            resources={"source_dataset": _SOURCE_ROWS},
            partition_key="2026-01-01",
            run_config={
                "ops": {
                    "redacted_dataset": {
                        "config": {
                            "policy_profile": "not-a-policy-profile",
                            "text_columns": ["note"],
                        }
                    }
                }
            },
        )


def _fake_process_batch(texts: list[str], **kwargs: Any) -> Any:
    replacements = {
        "John Doe": ("[PERSON]", "PERSON"),
        "Jane Roe": ("[PERSON]", "PERSON"),
        "555-0101": ("[PHONE]", "PHONE"),
        "jane@example.test": ("[EMAIL]", "EMAIL"),
    }
    items = []
    for text in texts:
        redacted = text
        entities = []
        for surface, (replacement, label) in replacements.items():
            if surface not in text:
                continue
            redacted = redacted.replace(surface, replacement)
            entities.append(SimpleNamespace(label=label, entity_type=label))
        items.append(
            SimpleNamespace(
                success=True,
                result=SimpleNamespace(
                    deidentified_text=redacted,
                    pii_entities=entities,
                ),
            )
        )
    return SimpleNamespace(items=items)
