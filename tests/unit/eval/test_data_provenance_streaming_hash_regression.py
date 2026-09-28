"""Regression checks for bounded dataset hashing and stable digests."""

import hashlib
import json
from pathlib import Path

import pytest

from openmed.eval.data_provenance import compute_dataset_content_hash


def _digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "data",
    [b"", b"synthetic", bytes(range(256)) * 8193],
    ids=["empty", "small", "multi-chunk"],
)
def test_file_digest_remains_compatible(tmp_path, data):
    path = tmp_path / "synthetic.bin"
    path.write_bytes(data)
    assert compute_dataset_content_hash(path) == _digest(data)


def test_directory_digest_remains_compatible(tmp_path):
    (tmp_path / "nested").mkdir()
    files = {"z.bin": b"last", "nested/\u00e9.bin": b"first", "empty": b""}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    expected = json.dumps(
        {"files": {name: _digest(data) for name, data in files.items()}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert compute_dataset_content_hash(tmp_path) == _digest(expected)


@pytest.mark.parametrize("directory", [False, True])
def test_file_reads_are_bounded(tmp_path, monkeypatch, directory):
    data = bytes(range(256)) * 8193
    path = tmp_path / "synthetic.bin"
    path.write_bytes(data)
    original_open = Path.open
    sizes = []

    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, size=-1):
            sizes.append(size)
            assert 0 < size <= 1024 * 1024, "unbounded dataset read"
            return self.handle.read(size)

    def watched_open(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return BoundedReader(handle) if self == path and mode == "rb" else handle

    monkeypatch.setattr(Path, "open", watched_open)
    result = compute_dataset_content_hash(tmp_path if directory else path)
    assert sizes and len(sizes) >= 3
    if not directory:
        assert result == _digest(data)


def test_missing_source_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_dataset_content_hash(tmp_path / "missing")
