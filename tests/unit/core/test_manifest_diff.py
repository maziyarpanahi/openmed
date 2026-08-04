"""Tests for manifest release diffs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.manifest_diff import diff_manifests

BASE_ROW: dict[str, object] = {
    "repo_id": "OpenMed/base",
    "family": "PII",
    "task": "token-classification",
    "languages": ["en"],
    "tier": "Small",
    "param_count": 44_000_000,
    "architecture": "bert",
    "base_model": "OpenMed/source",
    "formats": ["mlx-fp", "pytorch"],
    "canonical_labels": ["PERSON"],
    "benchmark": {
        "dataset": "synthetic",
        "scores": [
            {"metric": "recall", "value": 0.99},
            {"metric": "micro_f1", "value": 0.98},
        ],
    },
    "arxiv": None,
    "license": "apache-2.0",
    "reproducibility_hash": "sha256:test",
    "released": "2026-06-24",
}


def test_diff_manifests_reports_repo_sets_and_field_changes(
    tmp_path: Path,
) -> None:
    old_manifest = _write_manifest(
        tmp_path / "old.jsonl",
        [
            _row("OpenMed/removed"),
            _row("OpenMed/reordered"),
            _row("OpenMed/changed"),
        ],
    )
    new_manifest = _write_manifest(
        tmp_path / "new.jsonl",
        [
            _row("OpenMed/added"),
            _row(
                "OpenMed/reordered",
                formats=["pytorch", "mlx-fp"],
                benchmark={
                    "scores": [
                        {"value": 0.98, "metric": "micro_f1"},
                        {"value": 0.99, "metric": "recall"},
                    ],
                    "dataset": "synthetic",
                },
            ),
            _row(
                "OpenMed/changed",
                tier="Medium",
                param_count=88_000_000,
                formats=["pytorch", "coreml-int8"],
                license="other",
                benchmark={
                    "dataset": "synthetic",
                    "scores": [
                        {"metric": "recall", "value": 0.94},
                        {"metric": "micro_f1", "value": 0.93},
                    ],
                },
            ),
        ],
    )

    diff = diff_manifests(old_manifest, new_manifest)

    assert diff.added == ("OpenMed/added",)
    assert diff.removed == ("OpenMed/removed",)
    assert [change.repo_id for change in diff.changed] == ["OpenMed/changed"]
    changed_fields = diff.changed[0].changes
    assert set(changed_fields) == {
        "tier",
        "param_count",
        "formats",
        "license",
        "benchmark",
    }
    assert changed_fields["tier"].before == "Small"
    assert changed_fields["tier"].after == "Medium"
    assert changed_fields["formats"].before == ["mlx-fp", "pytorch"]
    assert changed_fields["formats"].after == ["coreml-int8", "pytorch"]


def test_models_diff_json_emits_machine_readable_diff(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old_manifest = _write_manifest(
        tmp_path / "old.jsonl",
        [_row("OpenMed/removed"), _row("OpenMed/changed")],
    )
    new_manifest = _write_manifest(
        tmp_path / "new.jsonl",
        [_row("OpenMed/added"), _row("OpenMed/changed", tier="Large")],
    )

    result = main_module.main(
        ["models", "diff", str(old_manifest), str(new_manifest), "--json"]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    envelope = json.loads(captured.out)
    assert envelope["ok"] is True
    assert envelope["command"] == "models diff"
    payload = envelope["data"]
    assert payload["added"] == ["OpenMed/added"]
    assert payload["removed"] == ["OpenMed/removed"]
    assert payload["changed"] == [
        {
            "repo_id": "OpenMed/changed",
            "changes": {"tier": {"before": "Small", "after": "Large"}},
        }
    ]


def test_models_diff_human_summary_lists_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old_manifest = _write_manifest(
        tmp_path / "old.jsonl",
        [_row("OpenMed/removed"), _row("OpenMed/changed")],
    )
    new_manifest = _write_manifest(
        tmp_path / "new.jsonl",
        [_row("OpenMed/added"), _row("OpenMed/changed", license="other")],
    )

    result = main_module.main(["models", "diff", str(old_manifest), str(new_manifest)])
    captured = capsys.readouterr()

    assert result == 0
    assert "Added: 1" in captured.out
    assert "Removed: 1" in captured.out
    assert "Changed: 1" in captured.out
    assert "OpenMed/added" in captured.out
    assert "OpenMed/removed" in captured.out
    assert "license: apache-2.0 -> other" in captured.out


def test_models_diff_fail_on_removed_only_fails_when_repo_disappears(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old_manifest = _write_manifest(tmp_path / "old.jsonl", [_row("OpenMed/stable")])
    no_removed_manifest = _write_manifest(
        tmp_path / "new-no-removed.jsonl",
        [_row("OpenMed/stable", tier="Medium"), _row("OpenMed/added")],
    )
    removed_manifest = _write_manifest(tmp_path / "new-removed.jsonl", [])

    no_removed = main_module.main(
        [
            "models",
            "diff",
            str(old_manifest),
            str(no_removed_manifest),
            "--fail-on-removed",
        ]
    )
    capsys.readouterr()
    removed = main_module.main(
        [
            "models",
            "diff",
            str(old_manifest),
            str(removed_manifest),
            "--fail-on-removed",
        ]
    )

    assert no_removed == 0
    assert removed == 1


def test_models_diff_missing_manifest_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    new_manifest = _write_manifest(tmp_path / "new.jsonl", [_row("OpenMed/stable")])
    missing_manifest = tmp_path / "missing.jsonl"

    result = main_module.main(
        ["models", "diff", str(missing_manifest), str(new_manifest)]
    )
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "Failed to diff manifests:" in captured.err
    assert str(missing_manifest) in captured.err


def _row(repo_id: str, **overrides: object) -> dict[str, object]:
    row = deepcopy(BASE_ROW)
    row["repo_id"] = repo_id
    row.update(overrides)
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
