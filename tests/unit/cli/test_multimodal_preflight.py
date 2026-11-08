"""Synthetic offline admission, filesystem boundaries and diagnostic privacy."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import wave
import zlib
from contextlib import contextmanager

import pytest

from openmed.cli import main
from openmed.cli import multimodal_preflight as cli

SECRET = "Synthetic-Jane-Roe-MRN-00991-token-secret"


def png(width=7, height=5):
    payload = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    chunk = b"IHDR" + payload
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(payload))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk))
    )


def jpeg():
    frame = bytes([8]) + struct.pack(">HH", 5, 7) + bytes([1, 1, 0x11, 0])
    return b"\xff\xd8\xff\xc0" + struct.pack(">H", len(frame) + 2) + frame


def tiff():
    tags = [(256, 7), (257, 5), (258, 8), (262, 1)]
    return (
        b"II*\x00\x08\x00\x00\x00"
        + struct.pack("<H", len(tags))
        + b"".join(struct.pack("<HHIHH", tag, 3, 1, value, 0) for tag, value in tags)
        + b"\x00" * 4
    )


def wav():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * 80)
    return stream.getvalue()


def pdf():
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] >>",
    ]
    data = b"%PDF-1.7\n"
    offsets = []
    for number, value in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + value + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 4\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return (
        data
        + f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )


def invoke(capsys, path, *options):
    code = main(["multimodal", "preflight", str(path), *options])
    captured = capsys.readouterr()
    assert captured.err == ""
    assert SECRET not in captured.out
    assert str(path) not in captured.out
    return code, json.loads(captured.out)


@pytest.mark.parametrize(
    "payload,media",
    [
        (png(), "image/png"),
        (jpeg(), "image/jpeg"),
        (tiff(), "image/tiff"),
        (wav(), "audio/wav"),
    ],
)
def test_real_file_headers_digests_profiles_and_default_privacy(
    tmp_path, capsys, payload, media
):
    path = tmp_path / (SECRET + ".wrong-extension")
    payload += SECRET.encode()
    path.write_bytes(payload)
    before = path.read_bytes()
    code, report = invoke(capsys, path)
    assert code == 0
    row = report["assets"][0]
    assert row["media_type"] == media
    assert row["asset_id"] == "asset-000001"
    assert row["status"] == "accept"
    assert row["preflight"]["manifest"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["profile"]["name"] == "desktop"
    assert path.read_bytes() == before
    assert invoke(capsys, path)[1] == report


@pytest.mark.parametrize(
    "payload,media,header_status",
    [
        (pdf(), "application/pdf", "readable"),
        (b"\x00" * 128 + b"DICM", "application/dicom", "not_evaluated"),
    ],
)
def test_partial_metadata_is_reported_without_inventing_safe_admission(
    tmp_path, capsys, payload, media, header_status
):
    path = tmp_path / SECRET
    path.write_bytes(payload)
    code, report = invoke(capsys, path)
    assert code == 1
    row = report["assets"][0]
    assert row["media_type"] == media
    assert row["header"]["status"] == header_status
    assert row["status"] == "abstain"
    assert row["preflight"]["findings"]
    if media == "application/pdf":
        assert row["header"]["page_count"] == 1
        assert row["header"]["pages"][0]["width"] == 72


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        png()[:12],
        jpeg()[:5],
        tiff()[:9],
        wav()[:20],
        b"%PDF-1.7\n",
        b"\x00" * 128 + b"DIC",
        b"GIF89a",
        SECRET.encode(),
    ],
)
def test_malformed_or_unsupported_inputs_hold_offline(tmp_path, capsys, payload):
    path = tmp_path / SECRET
    path.write_bytes(payload)
    code, report = invoke(capsys, path)
    assert code == 1
    assert report["outcome"] == "hold"
    assert report["assets"][0]["status"] != "accept"


def test_directory_requires_opt_in_then_stably_continues_across_failures(
    tmp_path, capsys
):
    (tmp_path / "z-good").write_bytes(png())
    (tmp_path / "a-bad").write_bytes(SECRET.encode())
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b-wav").write_bytes(wav())
    code, report = invoke(capsys, tmp_path)
    assert code == 1
    assert report["assets"][0]["reason"] == "directory_opt_in_required"
    code, report = invoke(capsys, tmp_path, "--directory")
    assert code == 1
    assert [row["status"] for row in report["assets"]] == ["error", "accept", "accept"]
    assert report["batch_asset_ids"] == ["asset-000002", "asset-000003"]
    assert invoke(capsys, tmp_path, "--directory")[1] == report


def test_entry_file_depth_and_empty_directory_limits(tmp_path, capsys):
    code, report = invoke(capsys, tmp_path, "--directory")
    assert code == 1 and "no_assets" in report["findings"]
    for index in range(3):
        (tmp_path / str(index)).write_bytes(png(index + 1))
    code, report = invoke(capsys, tmp_path, "--directory", "--max-entries", "2")
    assert code == 1 and "directory_entry_limit" in report["findings"]
    assert report["collection"]["admitted_bytes"] == 0
    code, report = invoke(capsys, tmp_path, "--directory", "--max-files", "1")
    assert code == 1 and "file_count_limit" in report["findings"]
    assert len(report["assets"]) == 1
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "child").write_bytes(wav())
    code, report = invoke(capsys, tmp_path, "--directory", "--max-depth", "0")
    assert code == 1
    assert any(row.get("reason") == "depth_limit" for row in report["assets"])


def test_budget_admission_happens_before_header_and_hash_reads(
    tmp_path, capsys, monkeypatch
):
    path = tmp_path / SECRET
    path.write_bytes(png())
    monkeypatch.setattr(
        cli, "_header", lambda *args: pytest.fail("must not inspect over-limit source")
    )
    monkeypatch.setattr(
        cli,
        "digest_asset",
        lambda *args, **kwargs: pytest.fail("must not hash over-limit source"),
    )
    for option, reason in [
        ("--max-file-bytes", "file_byte_limit"),
        ("--max-total-bytes", "aggregate_byte_limit"),
    ]:
        code, report = invoke(capsys, path, option, "1")
        assert code == 1 and report["assets"][0]["reason"] == reason


def test_aggregate_budget_counts_bad_inputs_and_duplicates(tmp_path, capsys):
    (tmp_path / "a").write_bytes(b"x" * len(png()))
    (tmp_path / "b").write_bytes(png())
    code, report = invoke(
        capsys, tmp_path, "--directory", "--max-total-bytes", str(len(png()))
    )
    assert code == 1
    assert report["assets"][1]["reason"] == "aggregate_byte_limit"
    (tmp_path / "a").write_bytes(png())
    code, report = invoke(capsys, tmp_path, "--directory")
    assert code == 1
    assert any(
        row["reason"] == "duplicate_content" for row in report["batch"]["findings"]
    )


def test_aggregate_geometry_and_mobile_profile_are_applied(tmp_path, capsys):
    path = tmp_path / SECRET
    path.write_bytes(png())
    code, report = invoke(
        capsys, path, "--profile", "mobile", "--max-total-pixels", "34"
    )
    assert code == 1 and report["profile"]["name"] == "mobile"
    assert report["batch"]["totals"]["pixels"] == 35
    assert any(row["resource"] == "pixels" for row in report["batch"]["findings"])


def test_source_changes_during_hashing_cannot_be_accepted(
    tmp_path, capsys, monkeypatch
):
    if os.name == "nt":
        pytest.skip("Windows prevents writes while the protected handle is held")
    path = tmp_path / SECRET
    path.write_bytes(png())
    original = cli.digest_asset

    def change(stream, **kwargs):
        result = original(stream, **kwargs)
        path.write_bytes(png(8))
        return result

    monkeypatch.setattr(cli, "digest_asset", change)
    code, report = invoke(capsys, path)
    assert code == 1 and report["assets"][0]["reason"] == "source_changed"


def test_symlink_input_ancestor_and_nested_links_never_follow(tmp_path, capsys):
    target = tmp_path / "target"
    target.mkdir()
    (target / "good").write_bytes(png())
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    for path in (link, link / "good"):
        code, report = invoke(capsys, path, "--directory")
        assert code == 1 and not report["batch_asset_ids"]
    code, report = invoke(capsys, tmp_path, "--directory")
    assert code == 1
    assert sum(row["status"] == "accept" for row in report["assets"]) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-relative race control")
def test_swapped_child_link_is_rejected_before_read(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "folder"
    folder.mkdir()
    victim = folder / "victim"
    victim.write_bytes(png())
    outside = tmp_path / "outside"
    outside.write_bytes(wav())
    original = cli._open_node

    @contextmanager
    def swap(path, parent=None):
        if path == victim:
            victim.unlink()
            victim.symlink_to(outside)
        with original(path, parent) as node:
            yield node

    monkeypatch.setattr(cli, "_open_node", swap)
    code, report = invoke(capsys, folder, "--directory")
    assert code == 1
    assert report["collection"]["admitted_bytes"] == 0
    assert not report["batch_asset_ids"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_fifo_refuses_without_blocking(tmp_path, capsys):
    path = tmp_path / SECRET
    os.mkfifo(path)
    code, report = invoke(capsys, path)
    assert code == 1 and report["assets"][0]["reason"] == "not_regular_file"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [SECRET],
        ["preflight"],
        ["preflight", SECRET, "--unknown", SECRET],
        ["preflight", SECRET, "--max-files", SECRET],
        ["preflight", SECRET, "--profile", SECRET],
        ["preflight", SECRET, "--max-depth", "999"],
        ["preflight", SECRET, "--max-total-bytes", "-1"],
    ],
)
def test_usage_failures_never_echo_arguments(capsys, arguments):
    assert main(["multimodal", *arguments]) == 2
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert json.loads(captured.out)["outcome"] == "error"


def test_ambient_config_is_refused_without_loading_it(capsys):
    assert main(["--config-path", SECRET, "multimodal", "preflight", SECRET]) == 2
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err


def test_text_help_and_no_decoder_or_network_dependencies(
    tmp_path, capsys, monkeypatch
):
    path = tmp_path / SECRET
    path.write_bytes(png())
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "PIL",
            "pydicom",
            "torch",
            "transformers",
            "cv2",
            "requests",
            "httpx",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert main(["multimodal", "preflight", str(path), "--format", "text"]) == 0
    output = capsys.readouterr().out
    assert "asset-000001: accept" in output and SECRET not in output
    assert main(["multimodal", "preflight", "--help"]) == 0
    assert "--directory" in capsys.readouterr().out


def test_typer_uses_identical_guarded_boundary(tmp_path, capsys):
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from openmed.cli.typer_app import build_app

    path = tmp_path / SECRET
    path.write_bytes(png())
    code, expected = invoke(capsys, path)
    runner = CliRunner()
    result = runner.invoke(build_app(), ["multimodal", "preflight", str(path)])
    assert result.exit_code == code
    assert json.loads(result.stdout) == expected
    result = runner.invoke(
        build_app(), ["multimodal", "preflight", str(path), "--max-files", SECRET]
    )
    assert result.exit_code == 2
    assert SECRET not in result.output


def test_growing_source_cannot_expand_actual_read_budget(tmp_path, capsys, monkeypatch):
    if os.name == "nt":
        pytest.skip("Windows holds a handle that denies concurrent writes")
    path = tmp_path / SECRET
    path.write_bytes(png())
    original = cli._header
    observed = []

    def growing(stream, media, profile):
        with path.open("ab") as output:
            output.write(b"x" * 10000)
        observed.append(len(stream.read(10000)))
        stream.seek(0)
        return original(stream, media, profile)

    monkeypatch.setattr(cli, "_header", growing)
    code, report = invoke(capsys, path)
    assert code == 1
    assert observed == [len(png()) + 1]
    assert report["collection"]["admitted_bytes"] == len(png())


@pytest.mark.skipif(os.name != "nt", reason="Windows handle sharing semantics")
def test_windows_held_handles_prevent_source_write_and_parent_rename(tmp_path):
    parent = tmp_path / "folder"
    parent.mkdir()
    path = parent / "file"
    path.write_bytes(png())
    with cli._open_root(str(path)):
        with pytest.raises(OSError):
            path.write_bytes(b"replacement")
        with pytest.raises(OSError):
            parent.rename(tmp_path / "moved")
    assert path.read_bytes() == png()


def test_filesystem_error_messages_are_not_echoed(tmp_path, capsys, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError(SECRET)

    monkeypatch.setattr(cli, "_open_root", fail)
    code, report = invoke(capsys, tmp_path / SECRET)
    assert code == 1
    assert report["assets"][0]["reason"] == "source_unavailable"


def test_fresh_console_import_does_not_load_decoders_or_make_network_calls(tmp_path):
    path = tmp_path / SECRET
    path.write_bytes(png())
    script = """
import builtins, socket, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'PIL', 'pydicom', 'torch', 'transformers', 'cv2'}:
        raise AssertionError('decoder import')
    return original(name, *args, **kwargs)
def network(*args, **kwargs):
    raise AssertionError('network access')
builtins.__import__ = guarded
socket.create_connection = network
socket.socket.connect = network
from openmed.cli import main
raise SystemExit(main(['multimodal', 'preflight', sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert SECRET not in result.stdout + result.stderr
    assert json.loads(result.stdout)["outcome"] == "accept"


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-relative ancestor race")
def test_replaced_ancestor_cannot_redirect_child_reads(tmp_path, capsys, monkeypatch):
    folder, outside, moved = [
        tmp_path / name for name in ("folder", "outside", "moved")
    ]
    folder.mkdir()
    outside.mkdir()
    (folder / "file").write_bytes(png())
    (outside / "file").write_bytes(wav())
    original = cli._open_node

    @contextmanager
    def swap(path, parent=None):
        with original(path, parent) as node:
            if path == folder:
                folder.rename(moved)
                folder.symlink_to(outside, target_is_directory=True)
            yield node

    monkeypatch.setattr(cli, "_open_node", swap)
    code, report = invoke(capsys, folder, "--directory")
    assert code == 0
    assert (
        report["assets"][0]["preflight"]["manifest"]["sha256"]
        == hashlib.sha256(png()).hexdigest()
    )
