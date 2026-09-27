"""Focused contract tests for golden Journey drift reporting."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from openmed.eval.golden_journey import render_semantic_diff, semantic_diff

ROOT = Path(__file__).resolve().parents[3]


def test_semantic_diff_is_path_sorted_and_machine_readable() -> None:
    expected = {"a": [1, {"x": "old"}], "removed": True}
    actual = {"a": [1, {"x": "new"}, 3], "added": False}

    differences = semantic_diff(expected, actual)

    assert [item["path"] for item in differences] == [
        "/a/1/x",
        "/a/2",
        "/added",
        "/removed",
    ]
    assert render_semantic_diff(expected, actual).splitlines()[0] == (
        '{"actual":"new","expected":"old","path":"/a/1/x"}'
    )


def test_semantic_diff_escapes_json_pointer_tokens() -> None:
    assert semantic_diff({"a/b~c": 1}, {"a/b~c": 2}) == [
        {"actual": 2, "expected": 1, "path": "/a~1b~0c"}
    ]


def test_regeneration_defaults_to_read_only_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts" / "regenerate_v3_golden_journey.py"
    golden = ROOT / "tests" / "fixtures" / "journey" / "v3" / "golden.json"
    before = golden.read_bytes()
    monkeypatch.setattr(sys, "argv", [str(script)])

    with pytest.raises(SystemExit) as captured:
        runpy.run_path(str(script), run_name="__main__")

    assert captured.value.code == 0
    assert golden.read_bytes() == before
