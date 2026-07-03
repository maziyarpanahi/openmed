from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from uuid import uuid4

import pytest

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing import BatchProgress, deidentify_bucket
from openmed.processing.object_storage import _import_fsspec


def test_deidentify_bucket_memory_store_preserves_layout_and_redacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsspec = pytest.importorskip("fsspec")
    namespace = f"om225-{uuid4().hex}"
    fs = fsspec.filesystem("memory")
    fs.makedirs(f"/{namespace}/input/nested", exist_ok=True)
    fs.pipe(
        f"/{namespace}/input/root.txt",
        b"Patient John Doe called 555-0101.",
    )
    fs.pipe(
        f"/{namespace}/input/nested/visit.txt",
        b"Patient Jane Roe emailed jane@example.test.",
    )
    fs.pipe(f"/{namespace}/input/ignored.csv", b"Patient John Doe")
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    monkeypatch.setattr("openmed.deidentify", _fake_deidentify)
    records: list[BatchProgress] = []
    legacy_calls: list[tuple[int, int, str, bool]] = []

    def progress_callback(completed, total, item_result) -> None:
        legacy_calls.append(
            (completed, total, item_result.relative_path, item_result.success)
        )

    try:
        result = deidentify_bucket(
            f"memory://{namespace}/input",
            f"memory://{namespace}/output",
            policy="strict_no_leak",
            concurrency=2,
            progress_callback=progress_callback,
            on_progress=records.append,
        )

        assert result.total_objects == 2
        assert result.successful_objects == 2
        assert result.failed_objects == 0
        assert result.total_spans == 4
        assert [item.relative_path for item in result.items] == [
            "nested/visit.txt",
            "root.txt",
        ]
        assert fs.cat(f"/{namespace}/output/root.txt").decode() == (
            "Patient [PERSON] called [PHONE]."
        )
        assert fs.cat(f"/{namespace}/output/nested/visit.txt").decode() == (
            "Patient [PERSON] emailed [EMAIL]."
        )
        assert not fs.exists(f"/{namespace}/output/ignored.csv")

        rendered_outputs = "\n".join(
            fs.cat(path).decode()
            for path in sorted(fs.glob(f"/{namespace}/output/**/*.txt"))
        )
        assert "John Doe" not in rendered_outputs
        assert "Jane Roe" not in rendered_outputs
        assert "555-0101" not in rendered_outputs
        assert "jane@example.test" not in rendered_outputs

        assert [record.completed for record in records] == [1, 2]
        assert [record.total for record in records] == [2, 2]
        assert len(legacy_calls) == 2
        assert {call[2] for call in legacy_calls} == {"nested/visit.txt", "root.txt"}
        assert all(call[3] for call in legacy_calls)
    finally:
        if fs.exists(f"/{namespace}"):
            fs.rm(f"/{namespace}", recursive=True)


def test_deidentify_bucket_rejects_overlapping_output_root() -> None:
    with pytest.raises(ValueError, match="uri_out"):
        deidentify_bucket(
            "memory://om225-overlap/input",
            "memory://om225-overlap/input/redacted",
        )


def test_importing_processing_does_not_require_fsspec_or_cloud_backends() -> None:
    script = """
import builtins

blocked_roots = {"adlfs", "fsspec", "gcsfs", "s3fs"}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked_roots:
        raise ImportError(name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import openmed
import openmed.processing
assert hasattr(openmed.processing, "deidentify_bucket")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_deidentify_bucket_reports_missing_fsspec_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_fsspec(name: str):
        if name == "fsspec":
            raise ImportError(name)
        return __import__(name)

    monkeypatch.setattr(
        "openmed.processing.object_storage.import_module",
        missing_fsspec,
    )

    with pytest.raises(ImportError, match=r"openmed\[cloud\]"):
        _import_fsspec()


def _fake_deidentify(text: str, **kwargs) -> DeidentificationResult:
    assert kwargs["policy"] == "strict_no_leak"
    replacements = {
        "John Doe": ("[PERSON]", "PERSON"),
        "Jane Roe": ("[PERSON]", "PERSON"),
        "555-0101": ("[PHONE]", "PHONE"),
        "jane@example.test": ("[EMAIL]", "EMAIL"),
    }
    redacted = text
    entities: list[PIIEntity] = []
    for surface, (replacement, label) in replacements.items():
        start = text.find(surface)
        if start == -1:
            continue
        entities.append(
            PIIEntity(
                text=surface,
                label=label,
                confidence=0.99,
                start=start,
                end=start + len(surface),
                entity_type=label,
                original_text=surface,
                redacted_text=replacement,
            )
        )
        redacted = redacted.replace(surface, replacement)

    return DeidentificationResult(
        original_text=text,
        deidentified_text=redacted,
        pii_entities=entities,
        method=kwargs.get("method", "mask"),
        timestamp=datetime.now(),
    )
