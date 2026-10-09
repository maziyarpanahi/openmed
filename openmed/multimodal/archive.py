"""Bounded, local ZIP admission shared by document adapters."""

from __future__ import annotations

import os
import stat
import struct
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import BinaryIO

from openmed.interop.archive_safety import (
    ArchiveMember,
    ArchiveSafetyPolicy,
    inspect_archive_members,
)

from .exceptions import ArchiveDocumentError

_CHUNK = 64 * 1024
_MAX_DIRECTORY_BYTES = 16 * 1024 * 1024


def _fail(code: str, count: int = 1) -> None:
    raise ArchiveDocumentError(code, count) from None


def _check_directory(source: BinaryIO, policy: ArchiveSafetyPolicy) -> None:
    """Bound metadata allocation before ZipFile loads the central directory."""
    source.seek(0, 2)
    size = source.tell()
    source.seek(max(0, size - 65557))
    tail = source.read(65557)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or len(tail) - offset < 22:
        _fail("archive_invalid")
    _, disk, directory_disk, disk_count, count, length, _, comment = struct.unpack(
        "<4s4H2IH", tail[offset : offset + 22]
    )
    if disk or directory_disk or disk_count != count:
        _fail("archive_invalid")
    if count == 65535 or length == 0xFFFFFFFF:
        _fail("archive_zip64_directory")
    if count > policy.max_entries:
        _fail("entry_limit")
    if length > _MAX_DIRECTORY_BYTES:
        _fail("archive_metadata_limit")
    if offset + 22 + comment != len(tail):
        _fail("archive_invalid")
    end = size - len(tail) + offset
    start = end - length
    if start < 0:
        _fail("archive_invalid")
    source.seek(start)
    observed = 0
    while source.tell() < end:
        header = source.read(46)
        if len(header) != 46 or header[:4] != b"PK\x01\x02":
            _fail("archive_invalid")
        observed += 1
        if observed > policy.max_entries:
            _fail("entry_limit")
        name, extra, comment = struct.unpack_from("<3H", header, 28)
        source.seek(name + extra + comment, 1)
    if source.tell() != end or observed != count:
        _fail("archive_invalid")
    source.seek(0)


class AdmittedZip(zipfile.ZipFile):
    """A ZIP whose metadata and actual output are checked before extraction.

    Reads never trust ZipExtFile's declared-size truncation: raw DEFLATE output
    is limited to the declared size plus one byte, in at most 64 KiB chunks.
    """

    def __init__(self, source: BinaryIO, policy: ArchiveSafetyPolicy) -> None:
        _check_directory(source, policy)
        super().__init__(source)
        infos = self.infolist()
        report = inspect_archive_members(
            (
                ArchiveMember(
                    path=info.orig_filename,
                    compressed_size=info.compress_size,
                    uncompressed_size=info.file_size,
                    kind=(
                        "symlink"
                        if stat.S_ISLNK(info.external_attr >> 16)
                        else "directory"
                        if info.is_dir()
                        else "file"
                    ),
                )
                for info in infos
            ),
            policy,
        )
        if not report.allowed:
            code = next(iter(report.reason_counts))
            _fail(code, report.reason_counts[code])
        for info in infos:
            if info.flag_bits & 0x2041:
                _fail("archive_encrypted")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                _fail("archive_compression")
            if info.flag_bits & 0x20:
                _fail("archive_invalid")
        # Validate even unused parts before any parser/library sees the package.
        for info in infos:
            for _ in self._chunks(info):
                pass

    def _chunks(self, info: zipfile.ZipInfo) -> Iterator[bytes]:
        source = self.fp
        if source is None:
            _fail("archive_invalid")
        source.seek(info.header_offset)
        header = source.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            _fail("archive_invalid")
        _, _, flags, method, _, _, crc, compressed, declared, name, extra = (
            struct.unpack("<4s5H3I2H", header)
        )
        if flags != info.flag_bits or method != info.compress_type:
            _fail("archive_invalid")
        raw_name = source.read(name)
        local_name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        if local_name != info.orig_filename:
            _fail("archive_invalid")
        if not flags & 8 and (
            crc != info.CRC
            or compressed not in {info.compress_size, 0xFFFFFFFF}
            or declared not in {info.file_size, 0xFFFFFFFF}
        ):
            _fail("archive_size_mismatch")
        source.seek(extra, 1)
        if source.tell() + info.compress_size > self.start_dir:
            _fail("archive_invalid")
        remaining = info.compress_size
        produced = 0
        checksum = 0
        inflater = zlib.decompressobj(-15) if method == zipfile.ZIP_DEFLATED else None
        while remaining:
            data = source.read(min(_CHUNK, remaining))
            if not data:
                _fail("archive_invalid")
            remaining -= len(data)
            while data:
                budget = min(_CHUNK, info.file_size + 1 - produced)
                if inflater is None:
                    output, data = data[:budget], data[budget:]
                else:
                    output = inflater.decompress(data, budget)
                    data = inflater.unconsumed_tail
                produced += len(output)
                if produced > info.file_size:
                    _fail("archive_size_mismatch")
                checksum = zlib.crc32(output, checksum)
                yield output
                if inflater is not None and inflater.eof:
                    if inflater.unused_data or data or remaining:
                        _fail("archive_invalid")
                    break
        if produced != info.file_size:
            _fail("archive_size_mismatch")
        if inflater is not None and not inflater.eof:
            _fail("archive_invalid")
        if checksum != info.CRC:
            _fail("archive_crc")

    def read(self, name: str | zipfile.ZipInfo, pwd: bytes | None = None) -> bytes:
        """Read one admitted member with a strict actual-output bound."""
        if pwd is not None:
            _fail("archive_encrypted")
        info = name if isinstance(name, zipfile.ZipInfo) else self.getinfo(name)
        try:
            return b"".join(self._chunks(info))
        except (OSError, UnicodeError, zlib.error, struct.error):
            _fail("archive_invalid")


@contextmanager
def admitted_zip(
    source: str | os.PathLike[str] | BinaryIO,
    policy: ArchiveSafetyPolicy | None = None,
) -> Iterator[AdmittedZip]:
    """Admit an entire package without retaining decompressed parts.

    Args:
        source: Local path or seekable caller-owned binary stream.
        policy: Existing archive policy, optionally tightened by the caller.

    Yields:
        An admitted archive with bounded member reads. Its ``fp`` can be
        rewound and passed to an optional OOXML library within this context.

    Raises:
        ArchiveDocumentError: With a controlled code and count only.
    """
    owned = isinstance(source, (str, os.PathLike))
    stream = None
    try:
        stream = open(source, "rb") if owned else source
        archive = AdmittedZip(stream, policy or ArchiveSafetyPolicy())
    except (
        OSError,
        ValueError,
        UnicodeError,
        zipfile.BadZipFile,
        zlib.error,
        struct.error,
    ) as error:
        if owned and stream is not None:
            stream.close()
        if isinstance(error, ArchiveDocumentError):
            raise
        _fail("archive_invalid")
    except BaseException:
        if owned and stream is not None:
            stream.close()
        raise
    try:
        with archive:
            yield archive
    finally:
        if owned and stream is not None:
            stream.close()
