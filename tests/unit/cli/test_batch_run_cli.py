"""Tests for the ``openmed batch-run`` start/resume/report commands.

Every note is generated from an index, and the PHI vocabulary is invented. The
de-identification call is replaced with a deterministic stand-in so the suite
exercises the run lifecycle rather than a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import openmed
from openmed.cli import main_module

# Invented PHI. Its presence in a manifest, report or log line is a leak.
PHI_NAME = "Evelyn Quantum"
PHI_MRN = "ZQ-7391"
PHI_SUBSTRINGS = (PHI_NAME, PHI_MRN, "042-66-9001")

FAILING_MARKER = "TRIGGER-SHARD-FAILURE"


class _FakeDeidResult:
    def __init__(self, text: str) -> None:
        self.deidentified_text = text


def _fake_deidentify(text: str, *args, **kwargs):
    if FAILING_MARKER in text:
        raise RuntimeError(f"handler refused to process {PHI_NAME} ({PHI_MRN})")
    redacted = text
    for needle in PHI_SUBSTRINGS:
        redacted = redacted.replace(needle, "[REDACTED]")
    return _FakeDeidResult(redacted)


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    """Write synthetic notes, each carrying invented PHI."""

    directory = tmp_path / "notes"
    directory.mkdir()
    for index in range(12):
        (directory / f"note-{index:05d}.txt").write_text(
            f"Note {index:05d}. Patient {PHI_NAME}, MRN {PHI_MRN}, "
            f"SSN 042-66-9001. Visit {index}.",
            encoding="utf-8",
        )
    return directory


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openmed, "deidentify", _fake_deidentify)


def _run(args: list[str], capsys) -> tuple[int, str]:
    code = main_module.main(args)
    return code, capsys.readouterr().out


def _start(run_dir: Path, notes_dir: Path, *, shards: int = 4) -> list[str]:
    return [
        "batch-run",
        "start",
        "--run-dir",
        str(run_dir),
        "--input-dir",
        str(notes_dir),
        "--run-id",
        "run-0001",
        "--shards",
        str(shards),
        "--json",
    ]


def test_start_completes_and_reports_counts_and_fingerprints(
    tmp_path: Path, notes: Path, capsys
) -> None:
    run_dir = tmp_path / "run"
    code, out = _run(_start(run_dir, notes), capsys)
    payload = json.loads(out)

    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "batch-run start"

    data = payload["data"]
    assert data["run_state"] == "complete"
    assert data["shard_count"] == 4
    assert data["document_count"] == 12
    assert data["status_counts"]["completed"] == 4
    assert data["outputs"]["all_valid"] is True
    assert len(data["plan_fingerprint"]) == 64


def test_start_output_and_manifest_contain_no_phi(
    tmp_path: Path, notes: Path, capsys, caplog
) -> None:
    run_dir = tmp_path / "run"
    report_path = tmp_path / "report.json"

    with caplog.at_level(logging.DEBUG):
        code, out = _run(
            _start(run_dir, notes) + ["--report", str(report_path)], capsys
        )
    assert code == 0

    manifest_bytes = (run_dir / "manifest.json").read_text(encoding="utf-8")
    report_bytes = report_path.read_text(encoding="utf-8")
    logs = "\n".join(
        f"{record.name} {record.levelname} {record.getMessage()} {record.args!r}"
        for record in caplog.records
    )

    for surface, name in (
        (out, "cli stdout"),
        (manifest_bytes, "manifest"),
        (report_bytes, "report file"),
        (logs, "logs"),
    ):
        leaked = [needle for needle in PHI_SUBSTRINGS if needle in surface]
        assert leaked == [], f"PHI leaked into {name}: {leaked!r}"

    # The guard must be able to notice; prove the assertion above is not vacuous.
    planted = f"openmed.test INFO processing {PHI_NAME}"
    assert [n for n in PHI_SUBSTRINGS if n in planted] == [PHI_NAME]


def test_resume_recomputes_only_the_failed_shards(
    tmp_path: Path, notes: Path, capsys
) -> None:
    run_dir = tmp_path / "run"
    # Poison one note so its shard fails on the first pass.
    poisoned = notes / "note-00003.txt"
    original = poisoned.read_text(encoding="utf-8")
    poisoned.write_text(f"{original} {FAILING_MARKER}", encoding="utf-8")

    code, out = _run(_start(run_dir, notes), capsys)
    assert code == 1, out
    first = json.loads(out)
    assert first["data"]["status_counts"]["failed"] >= 1
    assert first["data"]["run_state"] == "in_progress"

    # Repair the corpus and resume; only the failed shard should re-run.
    poisoned.write_text(original, encoding="utf-8")
    code, out = _run(
        [
            "batch-run",
            "resume",
            "--run-dir",
            str(run_dir),
            "--input-dir",
            str(notes),
            "--json",
        ],
        capsys,
    )
    payload = json.loads(out)

    assert code == 0, out
    data = payload["data"]
    assert data["run_state"] == "complete"
    assert data["status_counts"]["completed"] == 4
    assert data["status_counts"]["failed"] == 0
    assert data["outputs"]["all_valid"] is True
    # Only the repaired shard consumed a second attempt.
    assert sorted(row["attempts"] for row in data["shards"]) == [1, 1, 1, 2]


def test_a_failing_run_still_emits_exactly_one_json_document(
    tmp_path: Path, notes: Path, capsys
) -> None:
    """A gate-negative run reports its findings and exits 1, in one envelope.

    Emitting the report and then raising would put two documents on stdout and
    break single-document parsing for the agents this envelope exists to serve.
    """

    run_dir = tmp_path / "run"
    poisoned = notes / "note-00003.txt"
    poisoned.write_text(
        f"{poisoned.read_text(encoding='utf-8')} {FAILING_MARKER}", encoding="utf-8"
    )

    code, out = _run(_start(run_dir, notes), capsys)

    assert code == 1
    payload = json.loads(out)  # raises "Extra data" if more than one document
    assert payload["ok"] is True
    assert payload["command"] == "batch-run start"
    assert payload["data"]["run_state"] == "in_progress"
    assert out.count('"schema_version"') >= 1


def test_report_is_a_pure_query_over_an_existing_run(
    tmp_path: Path, notes: Path, capsys
) -> None:
    run_dir = tmp_path / "run"
    _run(_start(run_dir, notes), capsys)
    before = (run_dir / "manifest.json").read_bytes()

    code, out = _run(
        ["batch-run", "report", "--run-dir", str(run_dir), "--json"], capsys
    )
    payload = json.loads(out)

    assert code == 0
    assert payload["command"] == "batch-run report"
    assert payload["data"]["run_state"] == "complete"
    assert (run_dir / "manifest.json").read_bytes() == before


def test_markdown_report_renders_without_phi(
    tmp_path: Path, notes: Path, capsys
) -> None:
    run_dir = tmp_path / "run"
    _run(_start(run_dir, notes), capsys)

    code = main_module.main(
        [
            "batch-run",
            "report",
            "--run-dir",
            str(run_dir),
            "--format",
            "markdown",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert out.startswith("# Batch Run Report: run-0001")
    assert "| Shards | 4 |" in out
    assert all(needle not in out for needle in PHI_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_missing_run_reports_a_stable_error_envelope(tmp_path: Path, capsys) -> None:
    code, out = _run(
        ["batch-run", "report", "--run-dir", str(tmp_path / "absent"), "--json"],
        capsys,
    )
    payload = json.loads(out)

    assert code == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "run_not_found"


def test_invalid_shard_count_is_a_usage_error(
    tmp_path: Path, notes: Path, capsys
) -> None:
    code, out = _run(
        _start(tmp_path / "run", notes, shards=0),
        capsys,
    )
    payload = json.loads(out)

    assert code == 2
    assert payload["error"]["code"] == "invalid_shards"


def test_starting_over_an_existing_run_is_refused(
    tmp_path: Path, notes: Path, capsys
) -> None:
    run_dir = tmp_path / "run"
    _run(_start(run_dir, notes), capsys)

    code, out = _run(_start(run_dir, notes), capsys)
    payload = json.loads(out)

    assert code == 2
    assert payload["error"]["code"] == "run_exists"


@pytest.mark.parametrize(
    "run_id",
    [
        "nightly run 2026-08-02",  # a space: the first thing an operator types
        "run#42",
        "Patient Evelyn Quantum MRN ZQ-7391",
        "x" * 129,
    ],
)
def test_unpublishable_run_id_is_refused_before_anything_is_written(
    tmp_path: Path, notes: Path, capsys, run_id: str
) -> None:
    """The check must precede the run, not follow it.

    A run id only the manifest accepts is written durably, the whole corpus is
    processed, and then every later ``report`` and ``resume`` re-reads it and
    fails identically -- an unrecoverable run created by a single space.
    """

    run_dir = tmp_path / "run"
    code, out = _run(
        [
            "batch-run",
            "start",
            "--run-dir",
            str(run_dir),
            "--input-dir",
            str(notes),
            "--run-id",
            run_id,
            "--shards",
            "4",
            "--json",
        ],
        capsys,
    )
    payload = json.loads(out)

    assert code == 2
    assert payload["error"]["code"] == "invalid_run_id"
    assert "--run-id" in payload["error"]["message"]
    assert all(needle not in out for needle in PHI_SUBSTRINGS)
    # Nothing durable was created, so the run remains startable.
    assert not (run_dir / "manifest.json").exists()
    assert list(run_dir.glob("shard-*.jsonl")) == []


def test_a_rejected_run_id_leaves_the_run_startable(
    tmp_path: Path, notes: Path, capsys
) -> None:
    run_dir = tmp_path / "run"
    _run(
        [
            "batch-run",
            "start",
            "--run-dir",
            str(run_dir),
            "--input-dir",
            str(notes),
            "--run-id",
            "bad id",
            "--shards",
            "4",
            "--json",
        ],
        capsys,
    )
    code, out = _run(_start(run_dir, notes), capsys)

    assert code == 0, out
    assert json.loads(out)["data"]["run_state"] == "complete"


def test_shard_output_carries_no_per_record_identifier(
    tmp_path: Path, notes: Path, capsys
) -> None:
    """An unsalted per-record digest is a reversible pseudonym, not protection."""

    from openmed.processing.distributed import stable_document_hash

    run_dir = tmp_path / "run"
    _run(_start(run_dir, notes), capsys)

    outputs = sorted(run_dir.glob("shard-*.jsonl"))
    assert outputs
    for path in outputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert set(record) == {"text"}
    blob = "\n".join(path.read_text(encoding="utf-8") for path in outputs)
    assert stable_document_hash("note-00000") not in blob
    assert "note-00000" not in blob


def test_max_attempts_makes_a_run_reachably_exhausted(
    tmp_path: Path, notes: Path, capsys
) -> None:
    """Without a budget a retry loop never terminates; with one it does."""

    run_dir = tmp_path / "run"
    poisoned = notes / "note-00003.txt"
    poisoned.write_text(
        f"{poisoned.read_text(encoding='utf-8')} {FAILING_MARKER}", encoding="utf-8"
    )

    code, out = _run(_start(run_dir, notes), capsys)
    assert code == 1
    assert json.loads(out)["data"]["run_state"] == "in_progress"

    resume = [
        "batch-run",
        "resume",
        "--run-dir",
        str(run_dir),
        "--input-dir",
        str(notes),
        "--max-attempts",
        "2",
        "--json",
    ]
    code, out = _run(resume, capsys)
    payload = json.loads(out)

    assert code == 1
    assert payload["data"]["run_state"] == "exhausted"
    assert payload["data"]["resume"]["is_exhausted"] is True


def test_duplicate_file_stems_are_diagnosable(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "notes"
    (corpus / "a").mkdir(parents=True)
    (corpus / "b").mkdir(parents=True)
    (corpus / "a" / "note.txt").write_text("first", encoding="utf-8")
    (corpus / "b" / "note.txt").write_text("second", encoding="utf-8")

    code, out = _run(
        [
            "batch-run",
            "start",
            "--run-dir",
            str(tmp_path / "run"),
            "--input-dir",
            str(corpus),
            "--recursive",
            "--run-id",
            "run-dupe",
            "--shards",
            "2",
            "--json",
        ],
        capsys,
    )
    payload = json.loads(out)

    assert code == 2
    assert payload["error"]["code"] == "duplicate_document_id"
    # The positions must be named, and only the positions.
    assert "index 1" in payload["error"]["message"]
    assert "note" not in payload["error"]["message"].replace("note files", "")


def test_unusable_pattern_is_a_usage_error(tmp_path: Path, notes: Path, capsys) -> None:
    code, out = _run(
        [
            "batch-run",
            "start",
            "--run-dir",
            str(tmp_path / "run"),
            "--input-dir",
            str(notes),
            "--pattern",
            "",
            "--run-id",
            "run-pattern",
            "--shards",
            "2",
            "--json",
        ],
        capsys,
    )
    payload = json.loads(out)

    assert code == 2
    assert payload["error"]["code"] in {"invalid_pattern", "no_input"}


def test_report_path_pointing_at_a_directory_is_refused_up_front(
    tmp_path: Path, notes: Path, capsys
) -> None:
    run_dir = tmp_path / "run"
    target = tmp_path / "already-a-directory"
    target.mkdir()

    code, out = _run(_start(run_dir, notes) + ["--report", str(target)], capsys)
    payload = json.loads(out)

    assert code == 2
    assert payload["error"]["code"] == "invalid_report_path"
    # Refused before the corpus was consumed.
    assert not (run_dir / "manifest.json").exists()


def test_handler_exception_text_never_reaches_the_error_envelope(
    tmp_path: Path, notes: Path, capsys
) -> None:
    """A worker failure is reported by type; its message carried PHI."""

    run_dir = tmp_path / "run"
    for path in notes.glob("*.txt"):
        path.write_text(
            f"{path.read_text(encoding='utf-8')} {FAILING_MARKER}", encoding="utf-8"
        )

    code, out = _run(_start(run_dir, notes), capsys)

    assert code == 1
    assert all(needle not in out for needle in PHI_SUBSTRINGS)
    data = json.loads(out)["data"]
    assert {row["error_type"] for row in data["failures"]} == {"RuntimeError"}
