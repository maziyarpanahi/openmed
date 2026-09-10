"""Tests for local model release batch dispatch."""

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
    parse_item_json,
    run_batch,
    run_item,
    select_items,
    validate_queue,
)

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "recipes" / "queue.yaml"
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


def test_edge_parent_must_be_earlier_published_non_edge_for_same_model():
    parent = QueueItem(
        id="parent",
        model_id="OpenMed/example",
        weekday="monday",
        theme="language-pack",
        formats=("mlx-fp",),
    )
    edge = QueueItem(
        id="edge",
        model_id="OpenMed/example",
        weekday="wednesday",
        theme="quantized-edge",
        formats=("mlx-8bit",),
        depends_on_green_parent=("parent",),
    )

    validate_queue([parent, edge])

    with pytest.raises(BatchDispatchError, match="same model_id"):
        validate_queue(
            [
                parent,
                QueueItem(
                    **{
                        **edge.__dict__,
                        "model_id": "OpenMed/different",
                    }
                ),
            ]
        )

    with pytest.raises(BatchDispatchError, match="unpublished parent"):
        validate_queue(
            [
                QueueItem(**{**parent.__dict__, "publish": False}),
                edge,
            ]
        )


def test_queue_parser_rejects_unsupported_format_and_non_boolean_publish(tmp_path):
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        """
version: 1
weekly_themes:
  monday: language-pack
  tuesday: clinical-ner
  wednesday: quantized-edge
  thursday: benchmark-refresh
  friday: sdk-release
items:
  - id: invalid
    weekday: monday
    theme: language-pack
    model_id: OpenMed/example
    formats: [gguf]
    publish: "false"
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(BatchDispatchError, match="unsupported release format"):
        load_queue(queue)

    queue.write_text(
        queue.read_text(encoding="utf-8").replace("[gguf]", "[mlx-fp]"),
        encoding="utf-8",
    )
    with pytest.raises(BatchDispatchError, match="must be a boolean"):
        load_queue(queue)


def test_queue_parser_rejects_theme_drift(tmp_path):
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        """
version: 1
weekly_themes:
  monday: language-pack
  tuesday: clinical-ner
  wednesday: quantized-edge
  thursday: benchmark-refresh
  friday: sdk-release
items:
  - id: drifted-theme
    weekday: monday
    theme: clinical-ner
    model_id: OpenMed/example
    formats: [mlx-fp]
    publish: true
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(BatchDispatchError, match="does not match"):
        load_queue(queue)


def test_matrix_item_json_must_be_an_object():
    with pytest.raises(BatchDispatchError, match="must be an object"):
        parse_item_json("[]")


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
        if (
            "--model" in command
            and command[command.index("--model") + 1] == "OpenMed/bad"
        ):
            raise subprocess.CalledProcessError(1, command)

    results = run_batch(items, output_root=tmp_path, runner=runner)

    assert [result.item_id for result in results] == ["ok", "bad"]
    assert [result.ok for result in results] == [True, False]
    assert len(calls) == 2
    assert "OpenMed/bad" in str(results[1].error)


def test_item_runs_convert_gate_then_publish_without_exposing_token(
    monkeypatch,
    tmp_path,
):
    item = QueueItem(
        id="gated",
        model_id="OpenMed/example",
        weekday="monday",
        theme="language-pack",
        formats=("mlx-fp",),
        gate_command=("python", "synthetic_gate.py"),
    )
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setenv("HF_WRITE_TOKEN", "synthetic-test-token")

    def runner(command, env):
        calls.append((command, "HF_WRITE_TOKEN" in env))

    result = run_item(item, output_root=tmp_path, runner=runner)

    assert result.ok
    assert [command[1] for command, _ in calls] == [
        "-m",
        "synthetic_gate.py",
        "-m",
    ]
    assert [has_token for _, has_token in calls] == [False, False, True]
    assert calls[2][0][2] == "openmed.core.hf_publish"


def test_queue_documentation_describes_format_and_weekly_ordering():
    text = QUEUE_DOC.read_text(encoding="utf-8")
    compact = " ".join(text.split())

    assert "recipes/queue.yaml" in text
    assert "depends_on_green_parent" in text
    assert "Monday and Tuesday rows list parent artifacts first" in compact
    assert "Wednesday rows are reserved for edge artifacts" in compact
    assert "does not run this queue automatically" in compact
