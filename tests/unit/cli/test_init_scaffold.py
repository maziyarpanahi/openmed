"""Tests for the deterministic ``openmed init`` project scaffold."""

from __future__ import annotations

import json
import runpy
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from openmed.cli import main_module
from openmed.cli.scaffold import (
    MANAGED_FILES,
    PERSONA_PRESETS,
    ScaffoldConflictError,
    ScaffoldError,
    render_project_scaffold,
    scaffold_project,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[3]
CONFIG_SCHEMA = ROOT / "openmed" / "core" / "config.schema.json"

EXPECTED_POLICIES = {
    "researcher": "research_limited_dataset",
    "app-developer": "strict_no_leak",
    "data-engineer": "strict_no_leak",
}
EXPECTED_SYNTHETIC_RECORDS = {
    "researcher": 1,
    "app-developer": 1,
    "data-engineer": 2,
}


def _project_contents(project: Path) -> dict[str, bytes]:
    return {name: (project / name).read_bytes() for name in MANAGED_FILES}


def _deny_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("project scaffold attempted network access")


@pytest.mark.parametrize("preset", PERSONA_PRESETS)
def test_each_persona_is_schema_valid_synthetic_and_runnable_offline(
    preset: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / preset
    monkeypatch.setenv("HF_TOKEN", "unit-secret-that-must-not-be-rendered")
    monkeypatch.setattr(socket, "create_connection", _deny_network)

    result = scaffold_project(project, preset=preset)

    assert result.created == MANAGED_FILES
    assert result.overwritten == ()
    assert result.unchanged == ()
    assert {path.name for path in project.iterdir()} == set(MANAGED_FILES)

    schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    config = tomllib.loads((project / "openmed.toml").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(config)
    assert config["local_only"] is True
    assert "hf_token" not in config

    combined = "\n".join(
        (project / name).read_text(encoding="utf-8") for name in MANAGED_FILES
    )
    assert "unit-secret-that-must-not-be-rendered" not in combined
    assert "SYNTHETIC_" in combined
    assert "example.invalid" in combined or preset != "app-developer"
    assert "BEGIN PRIVATE KEY" not in combined

    pipeline = project / "pipeline.py"
    compile(pipeline.read_text(encoding="utf-8"), str(pipeline), "exec")
    namespace = runpy.run_path(str(pipeline), run_name=f"test_{preset}")
    assert namespace["main"](["--check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "local_only": True,
        "policy": EXPECTED_POLICIES[preset],
        "preset": preset,
        "synthetic_records": EXPECTED_SYNTHETIC_RECORDS[preset],
    }

    globals_dict = namespace["main"].__globals__
    if preset == "data-engineer":

        class FakeBatchProcessor:
            def __init__(self, **kwargs: object) -> None:
                assert kwargs["policy"] == EXPECTED_POLICIES[preset]

            def process_texts(self, notes: tuple[str, ...]) -> SimpleNamespace:
                assert all("SYNTHETIC_" in note for note in notes)
                items = [
                    SimpleNamespace(
                        result=SimpleNamespace(deidentified_text="[synthetic-redacted]")
                    )
                    for _note in notes
                ]
                return SimpleNamespace(
                    failed_items=0,
                    get_successful_results=lambda: items,
                    successful_items=len(items),
                    total_items=len(items),
                )

        globals_dict["BatchProcessor"] = FakeBatchProcessor
    else:

        def fake_deidentify(text: str, **kwargs: object) -> SimpleNamespace:
            assert "SYNTHETIC_" in text
            assert kwargs["policy"] == EXPECTED_POLICIES[preset]
            return SimpleNamespace(
                deidentified_text="[synthetic-redacted]",
                pii_entities=[object()],
            )

        globals_dict["deidentify"] = fake_deidentify

    assert namespace["main"]([]) == 0
    run_output = capsys.readouterr().out
    assert "[synthetic-redacted]" in run_output


def test_rendering_and_identical_reruns_are_deterministic(tmp_path: Path) -> None:
    project = tmp_path / "project"
    rendered_once = render_project_scaffold("researcher")
    rendered_twice = render_project_scaffold("researcher")

    assert rendered_once == rendered_twice
    assert tuple(rendered_once) == MANAGED_FILES

    first = scaffold_project(project, preset="researcher")
    contents = _project_contents(project)
    mtimes = {name: (project / name).stat().st_mtime_ns for name in MANAGED_FILES}
    second = scaffold_project(project, preset="researcher")

    assert first.created == MANAGED_FILES
    assert second.created == ()
    assert second.overwritten == ()
    assert second.unchanged == MANAGED_FILES
    assert _project_contents(project) == contents
    assert {
        name: (project / name).stat().st_mtime_ns for name in MANAGED_FILES
    } == mtimes


def test_conflict_preflight_writes_nothing_and_force_is_scoped(
    tmp_path: Path,
) -> None:
    project = tmp_path / "existing"
    scaffold_project(project, preset="researcher")
    unrelated = project / "keep-me.txt"
    unrelated.write_text("unrelated user content", encoding="utf-8")
    pipeline = project / "pipeline.py"
    pipeline.write_text("private-canary-content", encoding="utf-8")
    before = _project_contents(project)

    with pytest.raises(ScaffoldConflictError) as exc_info:
        scaffold_project(project, preset="app-developer")

    assert "private-canary-content" not in str(exc_info.value)
    assert _project_contents(project) == before
    assert unrelated.read_text(encoding="utf-8") == "unrelated user content"

    result = scaffold_project(project, preset="app-developer", force=True)

    assert set(result.overwritten) == {"openmed.toml", "pipeline.py", "README.md"}
    assert result.unchanged == (".env.example", ".gitignore")
    assert result.created == ()
    assert "private-canary-content" not in pipeline.read_text(encoding="utf-8")
    assert unrelated.read_text(encoding="utf-8") == "unrelated user content"


def test_nonempty_directory_without_managed_collisions_is_preserved(
    tmp_path: Path,
) -> None:
    project = tmp_path / "existing"
    project.mkdir()
    unrelated = project / "notes.txt"
    unrelated.write_text("caller-owned", encoding="utf-8")

    result = scaffold_project(project, preset="data-engineer")

    assert result.created == MANAGED_FILES
    assert unrelated.read_text(encoding="utf-8") == "caller-owned"


def test_symbolic_link_managed_path_is_never_replaced(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside-canary", encoding="utf-8")
    (project / "pipeline.py").symlink_to(outside)

    with pytest.raises(ScaffoldError, match="symbolic-link scaffold path"):
        scaffold_project(project, force=True)

    assert outside.read_text(encoding="utf-8") == "outside-canary"
    assert set(project.iterdir()) == {project / "pipeline.py"}


def test_destination_file_is_rejected_without_modification(tmp_path: Path) -> None:
    destination = tmp_path / "project"
    destination.write_text("caller-owned", encoding="utf-8")

    with pytest.raises(ScaffoldError, match="not a directory"):
        scaffold_project(destination, force=True)

    assert destination.read_text(encoding="utf-8") == "caller-owned"


def test_cli_registers_persona_alias_and_default() -> None:
    parser = main_module.build_parser()

    default_args = parser.parse_args(["init", "demo"])
    alias_args = parser.parse_args(["init", "demo", "--persona", "data-engineer"])

    assert default_args.directory == Path("demo")
    assert default_args.preset == "researcher"
    assert default_args.force is False
    assert alias_args.preset == "data-engineer"


def test_cli_json_output_and_safe_conflict_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "cli-project"
    monkeypatch.setattr(socket, "create_connection", _deny_network)

    return_code = main_module.main(
        ["init", str(project), "--preset", "researcher", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["ok"] is True
    assert payload["command"] == "init"
    assert payload["data"] == {
        "destination": str(project),
        "preset": "researcher",
        "created": list(MANAGED_FILES),
        "overwritten": [],
        "unchanged": [],
    }

    (project / "pipeline.py").write_text("confidential-canary", encoding="utf-8")
    return_code = main_module.main(["init", str(project), "--json"])
    output = capsys.readouterr().out
    error = json.loads(output)

    assert return_code == 1
    assert error["ok"] is False
    assert error["error"]["code"] == "scaffold_conflict"
    assert "pipeline.py" in error["error"]["message"]
    assert "confidential-canary" not in output


def test_unknown_programmatic_preset_lists_stable_choices() -> None:
    with pytest.raises(ScaffoldError, match="researcher, app-developer, data-engineer"):
        render_project_scaffold("unknown")
