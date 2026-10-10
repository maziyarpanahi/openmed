"""Synthetic ZIP attacks, decoder exclusion, and bounded allocation evidence."""

from __future__ import annotations

import io
import struct
import traceback
import tracemalloc
import zipfile
import zlib
from pathlib import Path

import pytest

from openmed.interop.archive_safety import ArchiveSafetyPolicy
from openmed.multimodal import documents_docx, epub, odt, pptx, xlsx
from openmed.multimodal.archive import admitted_zip
from openmed.multimodal.exceptions import ArchiveDocumentError, UnsupportedDocumentError

FORMATS = ("docx", "xlsx", "pptx", "epub", "odt")
POLICY = ArchiveSafetyPolicy(
    max_entries=4,
    max_member_uncompressed_bytes=4096,
    max_total_uncompressed_bytes=6144,
)


def ingest(kind, path, policy=POLICY):
    if kind == "xlsx":
        return xlsx.redact_xlsx(
            path,
            path.with_suffix(".redacted.xlsx"),
            archive_policy=policy,
            cell_deidentifier=lambda text: text,
        )
    module = documents_docx if kind == "docx" else globals()[kind]
    return getattr(module, f"extract_{kind}")(path, archive_policy=policy)


def patch_member(path, *, declared=None, flags=None, method=None, crc=None):
    raw = bytearray(path.read_bytes())
    central = raw.index(b"PK\x01\x02")
    local = struct.unpack_from("<I", raw, central + 42)[0]
    for value, fmt, central_offset, local_offset in (
        (declared, "<I", 24, 22),
        (flags, "<H", 8, 6),
        (method, "<H", 10, 8),
        (crc, "<I", 16, 14),
    ):
        if value is not None:
            struct.pack_into(fmt, raw, central + central_offset, value)
            struct.pack_into(fmt, raw, local + local_offset, value)
    path.write_bytes(raw)


def attack(path, kind):
    with zipfile.ZipFile(path, "w") as archive:
        name = "Synthetic_Patient_Secret.xml"
        if kind == "ratio":
            archive.writestr(name, b"A" * 4096, compress_type=zipfile.ZIP_DEFLATED)
        elif kind == "member":
            archive.writestr(name, b"A" * 4097)
        elif kind == "entries":
            for index in range(5):
                archive.writestr(f"part{index}", b"")
        elif kind == "total":
            archive.writestr(name, b"A" * 4096)
            archive.writestr("part2", b"B" * 4096)
        elif kind == "path":
            archive.writestr("../Synthetic_Patient_Secret", b"safe")
        elif kind == "lie":
            archive.writestr(
                name, b"A" * (1024 * 1024), compress_type=zipfile.ZIP_DEFLATED
            )
        else:
            archive.writestr(name, b"safe")
    if kind == "lie":
        # Forge BOTH headers and the CRC of the prefix: ZipExtFile.read would
        # silently accept this prefix instead of exposing the remaining output.
        patch_member(path, declared=1024, crc=zlib.crc32(b"A" * 1024))
    elif kind == "encrypted":
        patch_member(path, flags=1)
    elif kind == "compression":
        patch_member(path, method=99)
    elif kind == "version":
        raw = bytearray(path.read_bytes())
        central = raw.index(b"PK\x01\x02")
        struct.pack_into("<H", raw, central + 6, 255)
        path.write_bytes(raw)
    return path


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize(
    ("kind", "code"),
    [
        ("ratio", "expansion_ratio"),
        ("member", "member_size_limit"),
        ("entries", "entry_limit"),
        ("total", "total_size_limit"),
        ("lie", "archive_size_mismatch"),
        ("path", "path_traversal"),
        ("encrypted", "archive_encrypted"),
        ("compression", "archive_compression"),
        ("version", "archive_invalid"),
    ],
)
def test_all_formats_reject_before_decoder_and_without_phi(
    tmp_path,
    monkeypatch,
    format,
    kind,
    code,
):
    path = attack(tmp_path / f"Synthetic_Patient_Secret.{format}", kind)

    def forbidden(*args, **kwargs):
        pytest.fail("Package reached optional decoder before admission")

    monkeypatch.setattr(documents_docx, "_import_docx", forbidden)
    monkeypatch.setattr(pptx, "_import_pptx", forbidden)
    monkeypatch.setattr(xlsx, "_load_workbook_function", lambda: forbidden)
    with pytest.raises(ArchiveDocumentError) as caught:
        ingest(format, path)
    assert isinstance(caught.value, UnsupportedDocumentError)
    assert caught.value.reason_code == code
    assert caught.value.count == 1
    diagnostic = "".join(traceback.format_exception_only(caught.type, caught.value))
    assert "Synthetic_Patient_Secret" not in diagnostic
    assert str(tmp_path) not in diagnostic
    assert caught.value.args == (f"{code}: 1",)


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("kind", ["ratio", "member", "entries", "lie"])
def test_rejected_fixtures_peak_allocations_below_4_mib(tmp_path, format, kind):
    path = attack(tmp_path / f"fixture.{format}", kind)
    tracemalloc.start()
    try:
        with pytest.raises(ArchiveDocumentError):
            ingest(format, path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024


@pytest.mark.parametrize("format", FORMATS)
def test_forged_size_never_inflates_more_than_declared_plus_one(
    tmp_path,
    monkeypatch,
    format,
):
    path = attack(tmp_path / f"fixture.{format}", "lie")
    original = zlib.decompressobj
    output_count = 0

    class MeasuredInflater:
        def __init__(self, *args):
            self.inner = original(*args)

        def decompress(self, data, max_length):
            nonlocal output_count
            assert 0 < max_length <= 1025
            result = self.inner.decompress(data, max_length)
            output_count += len(result)
            return result

        def __getattr__(self, name):
            return getattr(self.inner, name)

    monkeypatch.setattr(zlib, "decompressobj", MeasuredInflater)
    with pytest.raises(ArchiveDocumentError, match="archive_size_mismatch"):
        ingest(format, path)
    assert output_count == 1025


@pytest.mark.parametrize("kind", ["ratio", "member", "entries", "total"])
def test_metadata_denials_never_create_decompressor(tmp_path, monkeypatch, kind):
    path = attack(tmp_path / "fixture.zip", kind)
    monkeypatch.setattr(zlib, "decompressobj", lambda *a: pytest.fail("decompressed"))
    with pytest.raises(ArchiveDocumentError):
        with admitted_zip(path, POLICY):
            pytest.fail("admitted")


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_bounded_reader_accepts_exact_limits_empty_and_streams(compression):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        archive.writestr("part", bytes(range(256)) * 16)
        archive.writestr("empty", b"")
    with admitted_zip(stream, POLICY) as archive:
        assert archive.read("part") == bytes(range(256)) * 16
        assert archive.read("empty") == b""
    assert not stream.closed


def test_false_entry_count_is_rejected_before_zipfile_allocation(tmp_path):
    path = attack(tmp_path / "entries.zip", "entries")
    raw = bytearray(path.read_bytes())
    end = raw.index(b"PK\x05\x06")
    struct.pack_into("<2H", raw, end + 8, 1, 1)
    path.write_bytes(raw)
    with pytest.raises(ArchiveDocumentError, match="entry_limit"):
        with admitted_zip(path, POLICY):
            pytest.fail("admitted")


def test_100_mib_high_ratio_default_policy_under_4_mib(tmp_path):
    path = tmp_path / "large.odt"
    # Construct outside tracemalloc without ever allocating the expanded body.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("content.xml", "w") as member:
            for _ in range(1600):
                member.write(b"A" * 65536)
    tracemalloc.start()
    try:
        with pytest.raises(ArchiveDocumentError, match="expansion_ratio"):
            odt.extract_odt(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024


@pytest.mark.parametrize("size", [65536, 65537, 131072, 524288])
def test_ordinary_deflate_crosses_chunk_boundaries(size):
    import random

    # Repeated pseudorandom blocks compress at ~4x, below the admission ratio.
    block = random.Random(0).randbytes(32768)
    payload = (block * (size // len(block) + 1))[:size]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("part", payload)
    with admitted_zip(stream) as archive:
        assert archive.read("part") == payload


def test_zip64_local_header_with_bounded_sizes_is_accepted():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("part", "w", force_zip64=True) as member:
            member.write(b"Synthetic document")
    with admitted_zip(stream) as archive:
        assert archive.read("part") == b"Synthetic document"


def test_streaming_zip_data_descriptor_is_accepted():
    class NonSeekable(io.BytesIO):
        def seek(self, *args):
            raise io.UnsupportedOperation()

    destination = NonSeekable()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("part", b"Synthetic document")
    with admitted_zip(io.BytesIO(destination.getvalue())) as archive:
        assert archive.read("part") == b"Synthetic document"


def test_prefix_crc_forgery_is_accepted_by_unchecked_zipfile(tmp_path):
    path = attack(tmp_path / "lying.zip", "lie")
    with zipfile.ZipFile(path) as archive:
        assert archive.read(archive.namelist()[0]) == b"A" * 1024
    with pytest.raises(ArchiveDocumentError, match="archive_size_mismatch"):
        with admitted_zip(path):
            pytest.fail("admitted forged output")
