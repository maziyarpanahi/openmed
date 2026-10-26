import pytest

from openmed.processing.batch import BatchProcessor


def inputs(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    bad, good = root / "bad.txt", root / "good.txt"
    bad.write_bytes(b"\xff")
    good.write_text("synthetic note", encoding="utf-8")
    return root, bad, good


def processor(monkeypatch, *, operation="analyze_text", continue_on_error=True):
    instance = BatchProcessor(
        model_name="synthetic",
        operation=operation,
        loader=object(),
        continue_on_error=continue_on_error,
    )
    monkeypatch.setattr(
        instance,
        "_get_analyze_text",
        lambda: lambda text, **kwargs: {"text": text, "entities": []},
    )
    if operation == "deidentify":
        import openmed.core.pii as pii

        monkeypatch.setattr(
            pii,
            "_deidentify_batch",
            lambda texts, **kwargs: [
                {"deidentified_text": "synthetic output"} for text in texts
            ],
        )
    return instance


def test_bad_encoding_becomes_failed_item_and_later_files_run(tmp_path, monkeypatch):
    _, bad, good = inputs(tmp_path)
    progress = []
    result = processor(monkeypatch).process_files(
        [bad, good], on_progress=progress.append
    )
    assert [item.id for item in result.items] == ["bad.txt", "good.txt"]
    assert [item.success for item in result.items] == [False, True]
    assert result.items[0].error
    assert len(progress) == 2
    assert result.items[1].result["text"] == "synthetic note"


@pytest.mark.parametrize("operation", ["analyze_text", "deidentify"])
def test_strict_mode_still_raises_decode_error(tmp_path, monkeypatch, operation):
    root, bad, good = inputs(tmp_path)
    instance = processor(monkeypatch, operation=operation, continue_on_error=False)
    with pytest.raises(UnicodeDecodeError):
        if operation == "deidentify":
            instance.process_files_to_directory(
                [bad, good], input_root=root, output_dir=tmp_path / "out"
            )
        else:
            instance.process_files([bad, good])


def test_directory_output_skips_bad_file_and_writes_good_file(tmp_path, monkeypatch):
    root, bad, good = inputs(tmp_path)
    out = tmp_path / "out"
    result = processor(monkeypatch, operation="deidentify").process_files_to_directory(
        [bad, good], input_root=root, output_dir=out
    )
    assert [item.success for item in result.items] == [False, True]
    assert not (out / "bad.txt").exists()
    assert (out / "good.txt").read_text(encoding="utf-8") == "synthetic output"


def test_missing_file_handling_is_unchanged(tmp_path, monkeypatch):
    root, _, good = inputs(tmp_path)
    result = processor(monkeypatch).process_files([root / "missing.txt", good])
    assert [item.success for item in result.items] == [False, True]


@pytest.mark.parametrize("reader", ["files", "directory"])
def test_decode_reports_omit_native_input_details(
    tmp_path, monkeypatch, caplog, reader
):
    from pathlib import Path

    root, bad, good = inputs(tmp_path)
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == bad:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "SYNTHETIC-CONTENT-MARKER")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    instance = processor(
        monkeypatch, operation="deidentify" if reader == "directory" else "analyze_text"
    )
    if reader == "directory":
        result = instance.process_files_to_directory(
            [bad, good], input_root=root, output_dir=tmp_path / "out"
        )
    else:
        result = instance.process_files([bad, good])
    assert [item.success for item in result.items] == [False, True]
    assert result.items[0].error == "UnicodeDecodeError"
    assert "SYNTHETIC-CONTENT-MARKER" not in str(result.items[0])
    assert "SYNTHETIC-CONTENT-MARKER" not in caplog.text
