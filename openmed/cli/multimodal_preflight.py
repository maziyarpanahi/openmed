"""Offline, bounded local-file admission with content-free CLI diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO, Never, cast

from openmed.multimodal.asset_limits import DESKTOP_V1, MOBILE_V1, LimitProfile
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.batch_preflight import (
    BatchPreflightAsset,
    BatchPreflightLimits,
    preflight_asset_batch,
)
from openmed.multimodal.digest import digest_asset
from openmed.multimodal.image_header import read_image_header
from openmed.multimodal.media_type import detect_media_type
from openmed.multimodal.pdf_geometry import read_pdf_geometry
from openmed.multimodal.tiff_metadata import read_tiff_metadata
from openmed.multimodal.wav_metadata import read_wav_metadata

_PROFILES = {"desktop": DESKTOP_V1, "mobile": MOBILE_V1}
_HEADER_BYTES = 64 * 1024
_PDF_BYTES = 8 * 1024**2
_MAX_FILES = 256
_MAX_ENTRIES = 10000
_MAX_DEPTH = 32


class _Refused(ValueError):
    """Internal closed-code failure; never include underlying exceptions."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise _Refused("invalid_arguments")


@dataclass
class _Node:
    fd: int = field(repr=False)
    path: Path = field(repr=False)


@dataclass
class _BoundedFile:
    source: BinaryIO = field(repr=False)
    size: int
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = 5 * (self.size + 1)

    def read(self, size: int) -> bytes:
        if type(size) is not int or size < 0:
            raise _Refused("read_limit")
        if self.remaining <= 0:
            raise _Refused("read_limit")
        request = min(size, max(0, self.size + 1 - self.tell()), self.remaining)
        data = self.source.read(request)
        self.remaining -= len(data)
        return data

    def tell(self) -> int:
        return self.source.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.source.seek(offset, whence)

    def seekable(self) -> bool:
        return True

    def fileno(self) -> int:
        return self.source.fileno()


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _windows_open(path: Path) -> int:
    # Held ancestors deny write/delete sharing while child paths are opened.
    # OPEN_REPARSE_POINT opens the link itself so it can be rejected, not followed.
    if sys.platform != "win32":
        raise _Refused("secure_open_unavailable")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    info = kernel.GetFileInformationByHandleEx
    info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    info.restype = wintypes.BOOL
    kind = kernel.GetFileType
    kind.argtypes = [wintypes.HANDLE]
    kind.restype = wintypes.DWORD

    class AttributeTag(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]

    handle = create(str(path), 0x80000000, 1, None, 3, 0x02200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise _Refused("source_unavailable")
    try:
        attributes = AttributeTag()
        if not info(handle, 9, ctypes.byref(attributes), ctypes.sizeof(attributes)):
            raise _Refused("source_unavailable")
        if attributes.attributes & 0x400:
            raise _Refused("symlink_refused")
        if kind(handle) != 1:
            raise _Refused("not_regular_file")
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise
    return fd


@contextmanager
def _open_node(path: Path, parent: _Node | None = None) -> Iterator[_Node]:
    if sys.platform == "win32":
        fd = _windows_open(path)
    elif hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd:
        fd = os.open(
            path if parent is None else path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=None if parent is None else parent.fd,
        )
    else:
        raise _Refused("secure_open_unavailable")
    try:
        yield _Node(fd, path)
    finally:
        os.close(fd)


@contextmanager
def _open_root(value: str) -> Iterator[_Node]:
    path = Path(value)
    if ".." in path.parts or len(path.parts) > 128 or "\x00" in value:
        raise _Refused("unsafe_path")
    path = path.absolute()
    if os.name == "nt" and (
        len(path.drive) != 2
        or path.drive[1] != ":"
        or any(":" in part or part.endswith((".", " ")) for part in path.parts[1:])
    ):
        # No UNC, device namespaces or alternate data streams.
        raise _Refused("unsafe_path")
    with ExitStack() as stack:
        node = stack.enter_context(_open_node(Path(path.anchor)))
        for part in path.parts[1:]:
            if not stat.S_ISDIR(os.fstat(node.fd).st_mode):
                raise _Refused("source_unavailable")
            node = stack.enter_context(_open_node(node.path / part, node))
        yield node


def _header(stream: BinaryIO, media: str, profile: LimitProfile) -> tuple[dict, dict]:
    fields: dict[str, Any] = {}
    header: dict[str, Any] = {"status": "readable"}
    if media in {"image/png", "image/jpeg"}:
        image = read_image_header(
            stream, max_header_bytes=_HEADER_BYTES, max_pixels=profile.max_pixels
        )
        fields = {"width": image.width, "height": image.height}
    elif media == "image/tiff":
        tiff = read_tiff_metadata(
            stream, max_header_bytes=_HEADER_BYTES, max_pixels=profile.max_pixels
        )
        fields = {"width": tiff.width, "height": tiff.height}
        header["scope"] = "first_ifd_only"
    elif media == "audio/wav":
        wav = read_wav_metadata(stream, max_header_bytes=_HEADER_BYTES)
        fields = {"duration_seconds": wav.duration_seconds}
        header.update(channels=wav.channels, sample_rate_hz=wav.sample_rate_hz)
    elif media == "application/pdf":
        pdf = read_pdf_geometry(
            stream,
            max_bytes=_PDF_BYTES,
            max_pages=profile.max_pages,
            max_objects=10000,
            max_decompressed_bytes=_PDF_BYTES,
        )
        header = pdf.to_dict()
        if pdf.page_count:
            fields = {"pages": pdf.page_count}
    elif media == "application/dicom":
        header = {"status": "not_evaluated", "reason": "prefix_only"}
    else:
        raise _Refused("unsupported_header")
    return fields, header


@dataclass
class _Collection:
    args: argparse.Namespace = field(repr=False)
    profile: LimitProfile
    handles: ExitStack = field(repr=False)
    rows: list[dict[str, Any]] = field(default_factory=list)
    assets: list[BatchPreflightAsset] = field(default_factory=list, repr=False)
    bindings: list[tuple[int, BinaryIO, tuple[int, ...]]] = field(
        default_factory=list, repr=False
    )
    findings: set[str] = field(default_factory=set)
    entries: int = 0
    admitted_bytes: int = 0

    def failure(self, code: str) -> None:
        if len(self.rows) >= self.args.max_files:
            self.findings.add("file_count_limit")
            return
        self.rows.append(
            {
                "asset_id": f"asset-{len(self.rows) + 1:06d}",
                "status": "error",
                "reason": code,
            }
        )

    def visit(self, node: _Node, depth: int) -> None:
        metadata = os.fstat(node.fd)
        if stat.S_ISDIR(metadata.st_mode):
            if not self.args.directory:
                raise _Refused("directory_opt_in_required")
            if depth > self.args.max_depth:
                self.failure("depth_limit")
                return
            before = _identity(metadata)
            with os.scandir(node.path if os.name == "nt" else node.fd) as entries:
                names = [
                    entry.name
                    for entry in islice(
                        entries, self.args.max_entries - self.entries + 1
                    )
                ]
            self.entries += len(names)
            if self.entries > self.args.max_entries:
                self.findings.add("directory_entry_limit")
                return
            for name in sorted(names):
                if self.entries > self.args.max_entries:
                    break
                if len(self.rows) >= self.args.max_files:
                    self.findings.add("file_count_limit")
                    break
                try:
                    with _open_node(node.path / name, node) as child:
                        self.visit(child, depth + 1)
                except _Refused as exc:
                    self.failure(str(exc))
                except (OSError, ValueError):
                    self.failure("source_unavailable")
            if _identity(os.fstat(node.fd)) != before:
                self.findings.add("directory_changed")
            return
        if not stat.S_ISREG(metadata.st_mode):
            self.failure("not_regular_file")
            return
        if metadata.st_size <= 0:
            self.failure("empty_file")
            return
        if metadata.st_size > self.profile.max_byte_size:
            self.failure("file_byte_limit")
            return
        if self.admitted_bytes + metadata.st_size > self.args.max_total_bytes:
            self.failure("aggregate_byte_limit")
            return
        # Reserve before prefix/header/hash work, including malformed inputs.
        self.admitted_bytes += metadata.st_size
        row: dict[str, Any] = {
            "asset_id": f"asset-{len(self.rows) + 1:06d}",
            "status": "error",
        }
        self.rows.append(row)
        try:
            source = self.handles.enter_context(
                os.fdopen(os.dup(node.fd), "rb", buffering=0)
            )
            stream = cast(BinaryIO, _BoundedFile(source, metadata.st_size))
            media = detect_media_type(stream.read(132))
            stream.seek(0)
            if media is None:
                raise _Refused("unknown_media")
            row["media_type"] = media
            fields, header = _header(stream, media, self.profile)
            row["header"] = header
            digest = digest_asset(stream, max_bytes=metadata.st_size)
            if digest.byte_count != metadata.st_size or _identity(
                os.fstat(stream.fileno())
            ) != _identity(metadata):
                raise _Refused("source_changed")
            manifest = AssetManifest(
                row["asset_id"], media, digest.sha256, digest.byte_count, **fields
            )
            self.assets.append(BatchPreflightAsset(manifest, stream))
            self.bindings.append((len(self.rows) - 1, stream, _identity(metadata)))
        except _Refused as exc:
            row["reason"] = str(exc)
        except (OSError, ValueError, RuntimeError):
            row["reason"] = "header_or_source_invalid"

    def finish(self) -> dict[str, Any]:
        limits = BatchPreflightLimits(
            bytes=self.args.max_total_bytes,
            pages=self.args.max_total_pages,
            pixels=self.args.max_total_pixels,
            frames=self.args.max_total_frames,
            duration=self.args.max_total_duration,
            rejections=0,
        )
        batch = (
            preflight_asset_batch(
                self.assets,
                limits=limits,
                policy="continue",
                asset_limit_profile=self.profile,
            )
            if self.assets
            else None
        )
        for result, (position, stream, identity) in zip(
            batch.assets if batch else (), self.bindings
        ):
            row = self.rows[position]
            row["status"] = result.status
            row["preflight"] = result.preflight.to_dict() if result.preflight else None
            if _identity(os.fstat(stream.fileno())) != identity:
                row.update(status="error", reason="source_changed")
            if row["header"]["status"] not in {"readable", "not_evaluated"}:
                row.update(status="abstain", reason="header_requires_review")
        if not self.rows:
            self.findings.add("no_assets")
        outcome = (
            "accept"
            if batch is not None
            and batch.outcome == "accept"
            and not self.findings
            and all(row["status"] == "accept" for row in self.rows)
            else "hold"
        )
        batch_report = batch.to_dict() if batch else None
        # Position indexes in the shared batch refer to admitted manifests only.
        # Retain an explicit map instead of hiding filesystem failures or inventing manifests.
        return {
            "schema_version": "1.0",
            "command": "multimodal preflight",
            "outcome": outcome,
            "profile": self.profile.to_dict(),
            "findings": sorted(self.findings),
            "assets": self.rows,
            "batch": None
            if batch_report is None
            else {key: value for key, value in batch_report.items() if key != "assets"},
            "batch_asset_ids": [asset.manifest.asset_id for asset in self.assets],
            "collection": {
                "entries_examined": self.entries,
                "admitted_bytes": self.admitted_bytes,
            },
        }


def _options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input", help="Local regular file; directory requires --directory."
    )
    parser.add_argument(
        "--directory",
        action="store_true",
        help="Opt in to recursive bounded directory traversal.",
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--profile", choices=tuple(_PROFILES), default="desktop")
    parser.add_argument("--max-files", type=int, default=128)
    parser.add_argument("--max-entries", type=int, default=4096)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-file-bytes", type=int)
    parser.add_argument("--max-total-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--max-total-pages", type=int, default=100)
    parser.add_argument("--max-total-pixels", type=int, default=100000000)
    parser.add_argument("--max-total-frames", type=int, default=1024)
    parser.add_argument("--max-total-duration", type=int, default=1800)


def add_multimodal_preflight_command(subparsers: Any) -> None:
    """Register discoverable preflight help on the installed argparse CLI."""
    group = subparsers.add_parser("multimodal", help="Offline multimodal admission.")
    commands = group.add_subparsers(dest="multimodal_command")
    command = commands.add_parser(
        "preflight", help="Bounded local headers and admission."
    )
    _options(command)


def run_multimodal_preflight_cli(argv: Sequence[str]) -> int:
    """Run a guarded command with opaque IDs and fixed errors.

    Args:
        argv: Arguments after ``openmed multimodal``.

    Returns:
        Zero for admission, one for a held batch, two for invalid invocation.
        Help succeeds without accessing inputs. Reports never contain paths.
    """
    parser = _Parser(prog="openmed multimodal", allow_abbrev=False)
    commands = parser.add_subparsers(dest="operation", required=True)
    command = commands.add_parser("preflight", allow_abbrev=False)
    _options(command)
    output_format = "json"
    try:
        if (
            isinstance(argv, (str, bytes))
            or len(argv) > 40
            or any(type(arg) is not str or len(arg) > 4096 for arg in argv)
        ):
            raise _Refused("invalid_arguments")
        args = parser.parse_args(list(argv))
        output_format = args.format
        if (
            not 1 <= args.max_files <= _MAX_FILES
            or not 1 <= args.max_entries <= _MAX_ENTRIES
            or not 0 <= args.max_depth <= _MAX_DEPTH
        ):
            raise _Refused("invalid_limits")
        profile = _PROFILES[args.profile]
        if args.max_file_bytes is not None:
            if not 1 <= args.max_file_bytes <= profile.max_byte_size:
                raise _Refused("invalid_limits")
            profile = profile.with_limits(max_byte_size=args.max_file_bytes)
        # Caller may tighten resource ceilings; hard caps prevent accidental unbounded work.
        for value, maximum in (
            (args.max_total_bytes, DESKTOP_V1.max_byte_size),
            (args.max_total_pages, 10000),
            (args.max_total_pixels, DESKTOP_V1.max_total_pixels),
            (args.max_total_frames, 10000),
            (args.max_total_duration, 86400),
        ):
            if not 1 <= value <= maximum:
                raise _Refused("invalid_limits")
        with ExitStack() as handles:
            collection = _Collection(args, profile, handles)
            try:
                with _open_root(args.input) as root:
                    collection.visit(root, 0)
            except _Refused as exc:
                collection.failure(str(exc))
            except (OSError, ValueError):
                collection.failure("source_unavailable")
            report = collection.finish()
        code = 0 if report["outcome"] == "accept" else 1
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    except Exception:
        # Neither exception messages, types, parser values nor tracebacks escape.
        report = {
            "schema_version": "1.0",
            "command": "multimodal preflight",
            "outcome": "error",
            "reason": "invalid_arguments",
        }
        code = 2
    if output_format == "text":
        lines = [f"Multimodal preflight: {report['outcome']}"]
        lines.extend(
            f"{row['asset_id']}: {row['status']}" for row in report.get("assets", ())
        )
        sys.stdout.write("\n".join(lines) + "\n")
    else:
        sys.stdout.write(
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
    return code


def add_multimodal_preflight_typer_command(app: Any, typer_module: Any) -> None:
    """Forward raw Typer arguments to the same privacy-safe parser."""
    group = typer_module.Typer(help="Offline multimodal admission.")

    def preflight(context):
        code = run_multimodal_preflight_cli(["preflight", *context.args])
        if code:
            raise typer_module.Exit(code=code)

    preflight.__annotations__ = {"context": typer_module.Context}
    group.command(
        "preflight",
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )(preflight)
    app.add_typer(group, name="multimodal")
