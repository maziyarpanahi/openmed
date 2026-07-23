"""Crash-safety tests for durable batch checkpoints and atomic output."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.pii import DeidentificationResult
from openmed.processing.batch import (
    BatchCheckpointError,
    BatchItem,
    BatchItemResult,
    BatchProcessor,
    _atomic_write_bytes,
)
from openmed.processing.outputs import PredictionResult


@dataclass
class _SyntheticDeidentification:
    original_text: str
    deidentified_text: str
    pii_entities: list[dict[str, str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "deidentified_text": self.deidentified_text,
            "pii_entities": self.pii_entities,
        }


class _SyntheticProcessor(BatchProcessor):
    def __init__(self, *, event_log: Path | None = None, delay: float = 0.0) -> None:
        super().__init__(
            model_name="synthetic-model",
            operation="deidentify",
            batch_size=1,
            checkpoint_interval=2,
            method="mask",
        )
        self.event_log = event_log
        self.delay = delay

    def _process_batch_chunk(
        self,
        items: list[BatchItem],
    ) -> list[BatchItemResult]:
        results = []
        for item in items:
            if self.event_log is not None:
                with self.event_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"{item.id}\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            if self.delay:
                time.sleep(self.delay)
            results.append(
                BatchItemResult(
                    id=item.id,
                    result=_SyntheticDeidentification(
                        original_text=item.text,
                        deidentified_text=f"redacted::{item.id}",
                        pii_entities=[{"label": "NAME"}],
                    ),
                    processing_time=0.0,
                    source=item.source,
                )
            )
        return results


def _run_checkpoint_worker(
    input_dir: str,
    output_dir: str,
    checkpoint_path: str,
    event_log: str,
) -> None:
    files = sorted(Path(input_dir).glob("*.txt"))
    _SyntheticProcessor(
        event_log=Path(event_log), delay=0.2
    ).process_files_to_directory(
        files,
        input_root=input_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )


def _wait_for_uncommitted_interval(
    checkpoint_path: Path,
    event_log: Path,
    *,
    timeout: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            checkpoint = {}
        events = (
            event_log.read_text(encoding="utf-8").splitlines()
            if event_log.exists()
            else []
        )
        if checkpoint.get("committed_count") == 2 and len(events) >= 4:
            return
        time.sleep(0.01)
    raise AssertionError("worker did not reach the expected checkpoint boundary")


@pytest.mark.skipif(
    not hasattr(signal, "SIGKILL")
    or "fork" not in multiprocessing.get_all_start_methods(),
    reason="requires POSIX SIGKILL and fork",
)
def test_sigkill_resume_is_byte_identical_and_rework_is_bounded(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    interrupted_output = tmp_path / "interrupted"
    uninterrupted_output = tmp_path / "uninterrupted"
    input_dir.mkdir()
    raw_inputs = [
        "Patient Alice Example called 555-0100.",
        "Patient Bob Example called 555-0101.",
        "Patient Carol Example called 555-0102.",
        "Patient David Example called 555-0103.",
        "Patient Erin Example called 555-0104.",
        "Patient Frank Example called 555-0105.",
    ]
    for index, text in enumerate(raw_inputs):
        (input_dir / f"note-{index}.txt").write_text(text, encoding="utf-8")

    files = sorted(input_dir.glob("*.txt"))
    _SyntheticProcessor().process_files_to_directory(
        files,
        input_root=input_dir,
        output_dir=uninterrupted_output,
        checkpoint_path=uninterrupted_output / "checkpoint.json",
    )

    interrupted_output.mkdir()
    checkpoint_path = interrupted_output / "checkpoint.json"
    event_log = tmp_path / "processed.log"
    context = multiprocessing.get_context("fork")
    worker = context.Process(
        target=_run_checkpoint_worker,
        args=(
            str(input_dir),
            str(interrupted_output),
            str(checkpoint_path),
            str(event_log),
        ),
    )
    worker.start()
    _wait_for_uncommitted_interval(checkpoint_path, event_log)
    os.kill(worker.pid, signal.SIGKILL)
    worker.join(timeout=5)
    assert worker.exitcode == -signal.SIGKILL

    _SyntheticProcessor(event_log=event_log).process_files_to_directory(
        files,
        input_root=input_dir,
        output_dir=interrupted_output,
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=True,
    )

    for input_file in files:
        relative_path = input_file.relative_to(input_dir)
        assert (interrupted_output / relative_path).read_bytes() == (
            uninterrupted_output / relative_path
        ).read_bytes()

    process_events = event_log.read_text(encoding="utf-8").splitlines()
    assert len(process_events) - len(files) <= 2
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "complete"
    checkpoint_text = checkpoint_path.read_text(encoding="utf-8") + Path(
        f"{checkpoint_path}.part"
    ).read_text(encoding="utf-8")
    for raw_input in raw_inputs:
        assert raw_input not in checkpoint_text
        for phi_value in raw_input.removesuffix(".").split()[1:]:
            assert phi_value not in checkpoint_text


def test_resume_refuses_mismatched_committed_output_artifact(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = input_dir / "note.txt"
    source.write_text("Patient Alice Example called 555-0100.", encoding="utf-8")
    checkpoint_path = output_dir / "checkpoint.json"
    processor = _SyntheticProcessor()
    processor.process_files_to_directory(
        [source],
        input_root=input_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )

    (output_dir / "note.txt").write_text("tampered", encoding="utf-8")

    with pytest.raises(
        BatchCheckpointError,
        match="output artifact does not match the checkpoint hash",
    ):
        processor.process_files_to_directory(
            [source],
            input_root=input_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
        )


def test_resume_refuses_mismatched_final_output(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    checkpoint_path = tmp_path / "results.checkpoint.json"
    processor = _SyntheticProcessor()
    processor.process_texts(
        ["Patient Alice Example"],
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        BatchCheckpointError,
        match="Final batch output does not match the checkpoint hash",
    ):
        processor.process_texts(
            ["Patient Alice Example"],
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
        )


def test_resume_refuses_mismatched_committed_journal(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    checkpoint_path = tmp_path / "results.checkpoint.json"
    processor = _SyntheticProcessor()
    processor.process_texts(
        ["Patient Alice Example"],
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    journal_path = Path(f"{checkpoint_path}.part")
    journal_path.write_bytes(b"tampered\n" + journal_path.read_bytes())

    with pytest.raises(
        BatchCheckpointError,
        match="Committed batch output does not match the checkpoint hash",
    ):
        processor.process_texts(
            ["Patient Alice Example"],
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
        )


def test_resume_refuses_mismatched_configuration(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    checkpoint_path = tmp_path / "results.checkpoint.json"
    processor = _SyntheticProcessor()
    processor.config = {"profile": "first"}
    processor.process_texts(
        ["Patient Alice Example"],
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )

    resumed_processor = _SyntheticProcessor()
    resumed_processor.config = {"profile": "second"}
    with pytest.raises(
        BatchCheckpointError,
        match="Checkpoint configuration does not match this batch",
    ):
        resumed_processor.process_texts(
            ["Patient Alice Example"],
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema_version", {}),
        ("total_items", "1"),
        ("committed_count", 1.0),
        ("checkpoint_interval", False),
    ],
)
def test_resume_rejects_malformed_checkpoint_integers(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    output_path = tmp_path / "results.json"
    checkpoint_path = tmp_path / "results.checkpoint.json"
    processor = _SyntheticProcessor()
    processor.process_texts(
        ["Patient Alice Example"],
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint[field] = invalid_value
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(BatchCheckpointError, match="Checkpoint .* is invalid"):
        processor.process_texts(
            ["Patient Alice Example"],
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
        )


def test_resume_refuses_mismatched_item_output_offset(tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"
    checkpoint_path = tmp_path / "results.checkpoint.json"
    processor = _SyntheticProcessor()
    processor.process_texts(
        ["Patient Alice Example"],
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["items"][0]["committed_output_offset"] += 1
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(
        BatchCheckpointError,
        match="Checkpoint item output offset does not match the journal",
    ):
        processor.process_texts(
            ["Patient Alice Example"],
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
        )


def test_resume_requires_hash_for_every_successful_file(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = input_dir / "note.txt"
    source.write_text("Patient Alice Example", encoding="utf-8")
    checkpoint_path = output_dir / "checkpoint.json"
    processor = _SyntheticProcessor()
    processor.process_files_to_directory(
        [source],
        input_root=input_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    del checkpoint["items"][0]["artifact_size"]
    del checkpoint["items"][0]["artifact_sha256"]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(
        BatchCheckpointError,
        match="Checkpoint artifact metadata is incomplete",
    ):
        processor.process_files_to_directory(
            [source],
            input_root=input_dir,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
        )


@pytest.mark.parametrize("target_name", ["checkpoint.json", "checkpoint.json.part"])
@pytest.mark.parametrize(
    "crash_phase",
    [
        "after_write",
        "after_fsync",
        "before_replace",
        "after_replace",
        "after_directory_fsync",
    ],
)
def test_atomic_checkpoint_and_output_survive_every_crash_point(
    tmp_path: Path,
    target_name: str,
    crash_phase: str,
) -> None:
    target = tmp_path / target_name
    previous = b'{"generation":1}\n'
    candidate = b'{"generation":2}\n'
    target.write_bytes(previous)

    def crash_hook(phase: str, path: Path) -> None:
        assert path == target
        if phase == crash_phase:
            raise RuntimeError("simulated power loss")

    with pytest.raises(RuntimeError, match="simulated power loss"):
        _atomic_write_bytes(target, candidate, hook=crash_hook)

    persisted = target.read_bytes()
    assert persisted in {previous, candidate}
    assert json.loads(persisted)["generation"] in {1, 2}
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_cli_parses_resume_and_checkpoint_interval_flags() -> None:
    parser = main_module.build_parser()
    batch_args = parser.parse_args(
        [
            "batch",
            "--texts",
            "synthetic",
            "--output",
            "results.json",
            "--resume",
            "--checkpoint-interval",
            "3",
        ]
    )
    pii_args = parser.parse_args(
        [
            "pii",
            "batch",
            "--input-dir",
            "input",
            "--output-dir",
            "output",
            "--resume",
            "--checkpoint-interval",
            "4",
        ]
    )

    assert batch_args.resume is True
    assert batch_args.checkpoint_interval == 3
    assert pii_args.resume is True
    assert pii_args.checkpoint_interval == 4


def test_batch_cli_writes_and_resumes_atomic_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def analyze(text: str, **kwargs: object) -> PredictionResult:
        del kwargs
        calls.append(text)
        return PredictionResult(
            text=text,
            entities=[],
            model_name="synthetic-model",
            timestamp="2026-01-01T00:00:00",
            processing_time=0.0,
        )

    monkeypatch.setattr(BatchProcessor, "_get_analyze_text", lambda self: analyze)
    output_path = tmp_path / "results.json"
    arguments = [
        "batch",
        "--model",
        "synthetic-model",
        "--texts",
        "Patient Alice Example",
        "Patient Bob Example",
        "--output",
        str(output_path),
        "--output-format",
        "json",
        "--quiet",
        "--checkpoint-interval",
        "1",
    ]

    assert main_module.main(arguments) == 0
    assert calls == ["Patient Alice Example", "Patient Bob Example"]
    assert json.loads(output_path.read_text(encoding="utf-8"))["total_items"] == 2
    calls.clear()
    capsys.readouterr()

    assert main_module.main([*arguments, "--resume"]) == 0
    assert calls == []
    assert "Results written to" in capsys.readouterr().out


def test_pii_batch_cli_checkpoint_is_phi_free_and_resume_skips_completed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    raw_text = "Patient Alice Example called 555-0100."
    (input_dir / "note.txt").write_text(raw_text, encoding="utf-8")
    calls: list[str] = []

    def deidentify_batch(
        texts: list[str],
        **kwargs: object,
    ) -> list[DeidentificationResult]:
        calls.extend(texts)
        return [
            DeidentificationResult(
                original_text=text,
                deidentified_text="Patient [NAME] called [PHONE].",
                pii_entities=[],
                method=str(kwargs["method"]),
                timestamp=datetime(2026, 1, 1),
            )
            for text in texts
        ]

    monkeypatch.setattr("openmed.core.pii._deidentify_batch", deidentify_batch)
    monkeypatch.setattr(BatchProcessor, "_get_shared_loader", lambda self: None)
    arguments = [
        "pii",
        "batch",
        "--model",
        "synthetic-model",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--checkpoint-interval",
        "1",
    ]

    assert main_module.main(arguments) == 0
    assert calls == [raw_text]
    assert (output_dir / "note.txt").read_text(encoding="utf-8") == (
        "Patient [NAME] called [PHONE]."
    )
    checkpoint_path = output_dir / ".openmed-batch.checkpoint.json"
    checkpoint_material = checkpoint_path.read_text(encoding="utf-8") + Path(
        f"{checkpoint_path}.part"
    ).read_text(encoding="utf-8")
    assert raw_text not in checkpoint_material
    assert "Alice Example" not in checkpoint_material
    assert "555-0100" not in checkpoint_material
    calls.clear()
    capsys.readouterr()

    assert main_module.main([*arguments, "--resume"]) == 0
    assert calls == []
    assert "Processed 1 files, 0 failed" in capsys.readouterr().out


def test_pii_batch_cli_json_mode_emits_only_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "note.txt").write_text(
        "Patient Alice Example called 555-0100.",
        encoding="utf-8",
    )

    def deidentify_batch(
        texts: list[str],
        **kwargs: object,
    ) -> list[DeidentificationResult]:
        return [
            DeidentificationResult(
                original_text=text,
                deidentified_text="Patient [NAME] called [PHONE].",
                pii_entities=[],
                method=str(kwargs["method"]),
                timestamp=datetime(2026, 1, 1),
            )
            for text in texts
        ]

    monkeypatch.setattr("openmed.core.pii._deidentify_batch", deidentify_batch)
    monkeypatch.setattr(BatchProcessor, "_get_shared_loader", lambda self: None)

    assert (
        main_module.main(
            [
                "pii",
                "batch",
                "--model",
                "synthetic-model",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope["ok"] is True
    assert envelope["command"] == "pii batch"
    assert envelope["data"]["successful_items"] == 1
    assert envelope["data"]["output_dir"] == str(output_dir)
    assert captured.err == ""
