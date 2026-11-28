"""Tests for the web package's generated task pack module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.task_packs import (
    TASK_PACKS_DATA_FILE,
    TaskPackError,
    available_packs,
    resolve,
)
from scripts.android.build_android_catalog import load_manifest_rows
from scripts.web.build_web_catalog import build_web_catalog_rows
from scripts.web.build_web_task_packs import (
    DEFAULT_MODELS,
    DEFAULT_OUTPUT,
    REGENERATE_COMMAND,
    main,
    render_module,
    web_task_packs,
)


@pytest.fixture(scope="module")
def catalog_rows() -> list[dict[str, object]]:
    return build_web_catalog_rows(load_manifest_rows(DEFAULT_MODELS))


def _document() -> dict[str, object]:
    return json.loads(TASK_PACKS_DATA_FILE.read_text(encoding="utf-8"))


def test_committed_module_is_current(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--check"]) == 0
    assert "matches" in capsys.readouterr().out


def test_web_view_matches_the_python_resolver(
    catalog_rows: list[dict[str, object]],
) -> None:
    packs = {
        pack["pack_id"]: pack for pack in web_task_packs(_document(), catalog_rows)
    }

    assert sorted(packs) == list(available_packs())
    for pack_id in available_packs():
        resolution = resolve(pack_id, "web")
        pack = packs[pack_id]
        assert pack["pack_version"] == resolution.pack_version
        assert pack["task"] == resolution.task
        assert pack["runtime"] == resolution.runtime == "web"
        assert pack["format"] == resolution.format
        assert pack["status"] == resolution.status
        assert pack["canonical_labels"] == list(resolution.canonical_labels)
        assert pack["notice_key"] == resolution.notice_key
        assert [
            (item["rank"], item["repo_id"], item["param_count"], item["license"])
            for item in pack["candidates"]
        ] == [
            (item.rank, item.repo_id, item.param_count, item.license)
            for item in resolution.candidates
        ]


def test_candidates_are_joined_with_their_catalog_rows(
    catalog_rows: list[dict[str, object]],
) -> None:
    catalog = {row["repo_id"]: row for row in catalog_rows}
    for pack in web_task_packs(_document(), catalog_rows):
        for item in pack["candidates"]:
            assert (
                item["reproducibility_hash"]
                == (catalog[item["repo_id"]]["reproducibility_hash"])
            )


def test_a_candidate_outside_the_catalog_fails_closed(
    catalog_rows: list[dict[str, object]],
) -> None:
    resolution = resolve("pii", "web")
    first = resolution.candidates[0].repo_id
    missing = [row for row in catalog_rows if row["repo_id"] != first]
    relicensed = [
        {**row, "license": "other"} if row["repo_id"] == first else row
        for row in catalog_rows
    ]
    resized = [
        {**row, "param_count": 1} if row["repo_id"] == first else row
        for row in catalog_rows
    ]
    for rows in (missing, relicensed, resized):
        with pytest.raises(TaskPackError) as raised:
            web_task_packs(_document(), rows)
        assert raised.value.reason == "catalog_mismatch"


def test_check_reports_a_stale_module_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / DEFAULT_OUTPUT.name
    output.write_text("// stale\n", encoding="utf-8")

    assert main(["--check", "--output", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == "// stale\n"
    assert REGENERATE_COMMAND in capsys.readouterr().err

    assert main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == DEFAULT_OUTPUT.read_text(
        encoding="utf-8"
    )
    assert main(["--check", "--output", str(output)]) == 0


def test_unreadable_or_invalid_sources_fail_without_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    catalog_rows: list[dict[str, object]],
) -> None:
    with pytest.raises(TaskPackError) as raised:
        render_module(b"{not json", catalog_rows)
    assert raised.value.reason == "manifest_unreadable"

    invalid = tmp_path / "task_packs.json"
    invalid.write_text(json.dumps({"schema": "openmed.task_packs"}), encoding="utf-8")
    output = tmp_path / "out.ts"
    assert main(["--source", str(invalid), "--output", str(output)]) == 1
    assert main(["--source", str(tmp_path / "absent.json"), "--output", str(output)])
    error = capsys.readouterr().err
    assert "schema_invalid" in error
    assert "source_unreadable" in error
    assert str(tmp_path) not in error
    assert not output.exists()


def test_generated_module_names_its_source_and_command() -> None:
    text = DEFAULT_OUTPUT.read_text(encoding="utf-8")

    assert text.startswith("// GENERATED FILE. Do not edit by hand.")
    assert f"// Regenerate: {REGENERATE_COMMAND}" in text
    assert "openmed/core/data/task_packs.json" in text
    # The module carries pack data only, never the catalog rows themselves.
    assert "MODEL_CATALOG" not in text
