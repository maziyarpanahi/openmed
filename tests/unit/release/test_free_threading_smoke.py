"""Tests for the free-threaded CPython smoke script and its CI job."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import warnings
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "free_threading_smoke.py"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

_spec = importlib.util.spec_from_file_location("free_threading_smoke", SCRIPT)
assert _spec is not None and _spec.loader is not None
smoke = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = smoke
_spec.loader.exec_module(smoke)

_FAST = ["--threads", "4", "--iterations", "3"]


def _note_fragments() -> list[str]:
    """Return every synthetic note and its identifier-looking tokens."""

    fragments = []
    for _, text in smoke.SYNTHETIC_NOTES:
        fragments.append(text)
        fragments.extend(
            token.strip(".,:")
            for token in text.split()
            if any(character.isdigit() for character in token)
            and len(token.strip(".,:")) >= 4
        )
    return fragments


def _without_digest(report_text: str) -> str:
    # A hex digest could contain a short digit run by chance; it carries no
    # input text, so it is checked for shape instead.
    report = json.loads(report_text)
    digest = report["sweep"].pop("baseline_digest")
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    return json.dumps(report, sort_keys=True)


def _assert_text_free(*outputs: str) -> None:
    for output in outputs:
        for fragment in _note_fragments():
            assert fragment not in output, fragment


def test_real_sweep_matches_its_single_thread_baseline(tmp_path, capsys) -> None:
    report_path = tmp_path / "report.json"

    assert smoke.main([*_FAST, "--report", str(report_path)]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == smoke.REPORT_SCHEMA_VERSION
    assert report["sweep"]["status"] == "match"
    assert report["sweep"]["mismatched_passes"] == 0
    assert report["sweep"]["error_types"] == []
    assert report["sweep"]["baseline_spans"] > 0
    assert report["free_threaded_build"] is smoke.free_threaded_build()
    _assert_text_free(
        _without_digest(report_path.read_text(encoding="utf-8")),
        capsys.readouterr().out,
    )


def test_injected_thread_mismatch_exits_non_zero(tmp_path, capsys) -> None:
    """Workers see different spans than the main-thread baseline."""

    from openmed.core.safety_sweep import safety_sweep

    def drifting_sweep(text, spans, **kwargs):
        result = safety_sweep(text, spans, **kwargs)
        if threading.current_thread() is threading.main_thread():
            return result
        return result[1:]

    report_path = tmp_path / "report.json"
    status = smoke.main(
        [*_FAST, "--report", str(report_path)],
        sweep=drifting_sweep,
    )

    assert status == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["sweep"]["status"] == "mismatch"
    assert report["sweep"]["mismatched_passes"] == 4 * 3
    output = capsys.readouterr().out
    assert "sweep=mismatch" in output
    _assert_text_free(_without_digest(report_path.read_text(encoding="utf-8")), output)


def test_a_raising_sweep_is_reported_by_type_only(tmp_path, capsys) -> None:
    def failing_sweep(text, spans, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return []
        raise RuntimeError(f"synthetic failure while sweeping {text}")

    report_path = tmp_path / "report.json"
    status = smoke.main([*_FAST, "--report", str(report_path)], sweep=failing_sweep)

    assert status == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["sweep"]["status"] == "crash"
    assert report["sweep"]["error_types"] == ["RuntimeError"]
    text = (
        _without_digest(report_path.read_text(encoding="utf-8"))
        + capsys.readouterr().out
    )
    assert "synthetic failure" not in text
    _assert_text_free(text)


def test_report_contains_only_states_counts_and_digests(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    smoke.main([*_FAST, "--report", str(report_path)])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {
        "schema_version",
        "python",
        "free_threaded_build",
        "gil_enabled_before_import",
        "gil_enabled_after_import",
        "gil_reenabled_by",
        "sweep",
    }
    assert set(report["sweep"]) == {
        "status",
        "threads",
        "iterations",
        "notes",
        "baseline_spans",
        "baseline_digest",
        "mismatched_passes",
        "error_types",
    }
    assert len(report["sweep"]["baseline_digest"]) == 64


def test_gil_reenablement_is_a_warning_not_a_failure(monkeypatch, capsys) -> None:
    from openmed.core.safety_sweep import safety_sweep

    monkeypatch.setattr(
        smoke,
        "import_openmed",
        lambda: (safety_sweep, ["synthetic_ext._speedups"]),
    )

    assert smoke.main(_FAST) == 0
    output = capsys.readouterr().out
    assert (
        "::warning title=GIL re-enabled::importing openmed re-enabled the GIL "
        "via module synthetic_ext._speedups"
    ) in output


def test_gil_warning_parser_names_the_first_module() -> None:
    message = (
        "The global interpreter lock (GIL) has been enabled to load module "
        "'synthetic_ext._speedups', which has not declared that it can run "
        "safely without the GIL. To override this behavior and keep the GIL "
        "disabled (at your own risk), run with PYTHON_GIL=0 or -Xgil=0."
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn(message, RuntimeWarning, stacklevel=1)
        warnings.warn("unrelated 'synthetic' notice", UserWarning, stacklevel=1)

    assert smoke.gil_reenabling_modules(caught) == ["synthetic_ext._speedups"]


def test_require_free_threaded_refuses_a_gil_build(monkeypatch, capsys) -> None:
    monkeypatch.setattr(smoke, "free_threaded_build", lambda: False)

    assert smoke.main([*_FAST, "--require-free-threaded"]) == 2
    assert "not a free-threaded build" in capsys.readouterr().out


def test_arguments_are_validated() -> None:
    with pytest.raises(SystemExit):
        smoke.main(["--threads", "1"])
    with pytest.raises(SystemExit):
        smoke.main(["--iterations", "0"])


def test_ci_runs_the_smoke_on_free_threaded_linux_with_the_core_install() -> None:
    workflow = yaml.load(
        CI_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    job = workflow["jobs"]["free-threading-smoke"]

    assert job["runs-on"] == "ubuntu-latest"
    assert job["needs"] == "repo-policy"
    assert "continue-on-error" not in job
    assert int(job["timeout-minutes"]) <= 20
    assert all(value == "1" for value in job["env"].values())
    setup_python = next(
        step for step in job["steps"] if "actions/setup-python" in step.get("uses", "")
    )
    assert setup_python["with"]["python-version"] == "3.14t"
    commands = [step.get("run", "") for step in job["steps"]]
    install = next(command for command in commands if "uv sync" in command)
    assert "--frozen" in install and "--extra" not in install
    run = next(command for command in commands if "free_threading_smoke.py" in command)
    assert "--require-free-threaded" in run and "--report" in run


def test_docs_state_free_threading_is_experimental_and_core_only() -> None:
    docs = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")

    assert "Free-threaded CPython 3.14t: experimental, core only" in docs
    assert "free-threading-smoke" in docs
    assert "free_threading_smoke.py" in docs
