"""Tests for scheduled model release batch dispatch."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.release.dispatch_batch import (
    BatchDispatchError,
    QueueItem,
    build_matrix,
    load_queue,
    run_batch,
    select_items,
    validate_queue,
)


ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "recipes" / "queue.yaml"
CONVERT_WORKFLOW = ROOT / ".github" / "workflows" / "convert-models.yml"
QUEUE_DOC = ROOT / "docs" / "model-release-queue.md"


def test_release_queue_selects_multiple_monday_models():
    items = load_queue(QUEUE)
    monday_items = select_items(items, weekday="monday")
    matrix = build_matrix(monday_items)

    assert len({item.model_id for item in monday_items}) > 1
    assert len(matrix["include"]) == len(monday_items)
    assert {entry["weekday"] for entry in matrix["include"]} == {"monday"}


def test_quantized_edge_items_trail_green_parent_rows():
    items = load_queue(QUEUE)
    by_id = {item.id: item for item in items}

    edge_items = [
        item for item in items if set(item.formats) & {"mlx-8bit", "mlx-4bit", "coreml"}
    ]
    assert edge_items
    for item in edge_items:
        assert item.weekday == "wednesday"
        assert item.depends_on_green_parent
        for parent_id in item.depends_on_green_parent:
            assert by_id[parent_id].weekday in {"monday", "tuesday"}


def test_quantized_edge_item_without_parent_is_rejected():
    items = [
        QueueItem(
            id="edge",
            model_id="OpenMed/example",
            weekday="wednesday",
            theme="quantized-edge",
            formats=("mlx-8bit",),
        )
    ]

    with pytest.raises(BatchDispatchError, match="depends_on_green_parent"):
        validate_queue(items)


def test_batch_runner_continues_and_surfaces_per_model_failure(tmp_path):
    items = [
        QueueItem(
            id="ok",
            model_id="OpenMed/ok",
            weekday="monday",
            theme="language-pack",
            formats=("mlx-fp",),
            publish=False,
        ),
        QueueItem(
            id="bad",
            model_id="OpenMed/bad",
            weekday="monday",
            theme="language-pack",
            formats=("mlx-fp",),
            publish=False,
        ),
    ]
    calls: list[list[str]] = []

    def runner(command, env):
        calls.append(command)
        if "--model" in command and command[command.index("--model") + 1] == "OpenMed/bad":
            raise subprocess.CalledProcessError(1, command)

    results = run_batch(items, output_root=tmp_path, runner=runner)

    assert [result.item_id for result in results] == ["ok", "bad"]
    assert [result.ok for result in results] == [True, False]
    assert len(calls) == 2
    assert "OpenMed/bad" in str(results[1].error)


def test_workflow_has_scheduled_batch_matrix_and_manual_dispatch():
    workflow = CONVERT_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "cron:" in workflow
    assert "scripts/release/dispatch_batch.py plan" in workflow
    assert "fromJson(needs.plan-batch.outputs.matrix)" in workflow
    assert "fail-fast: false" in workflow
    assert "OPENMED_BATCH_ITEM: ${{ toJson(matrix) }}" in workflow


def test_queue_documentation_describes_format_and_weekly_ordering():
    text = QUEUE_DOC.read_text(encoding="utf-8")
    compact = " ".join(text.split())

    assert "recipes/queue.yaml" in text
    assert "depends_on_green_parent" in text
    assert "Monday and Tuesday rows publish parent artifacts first" in compact
    assert "Wednesday rows are reserved for edge artifacts" in compact
