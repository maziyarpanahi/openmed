"""Focused tests for metadata-only local trace-store discovery."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import openmed.traces.discovery as discovery
from openmed.traces.discovery import (
    TRACE_DISCOVERY_ENV_VAR,
    TRACE_ROOTS_ENV_VAR,
    TraceStore,
    discover_trace_stores,
)


def test_discovery_counts_files_and_bytes_without_opening_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "synthetic-traces"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "first.jsonl").write_bytes(b"synthetic-one")
    (nested / "second.jsonl").write_bytes(b"synthetic-two")

    def fail_open(*args, **kwargs):
        raise AssertionError("trace payload was opened")

    monkeypatch.setattr(builtins, "open", fail_open)
    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr(Path, "read_bytes", fail_open)
    monkeypatch.setattr(Path, "read_text", fail_open)

    assert discover_trace_stores({"synthetic": root}, environ={}) == (
        TraceStore("synthetic", 2, 26),
    )


def test_discovery_is_deterministic_and_ignores_symlinked_payloads(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "payload").write_bytes(b"one")
    (second / "payload").write_bytes(b"two")

    try:
        (first / "linked").symlink_to(second / "payload")
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")

    roots = {
        "zeta": first,
        "alpha": second,
    }
    expected = (
        TraceStore("alpha", 1, 3),
        TraceStore("zeta", 1, 3),
    )
    assert discover_trace_stores(roots, environ={}) == expected
    assert discover_trace_stores(roots, environ={}) == expected


def test_missing_and_unreadable_roots_are_skipped_without_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    denied = tmp_path / "denied"
    denied.mkdir()
    real_scandir = discovery.os.scandir

    def guarded_scandir(path):
        if Path(path) == denied:
            raise PermissionError("synthetic permission failure")
        return real_scandir(path)

    monkeypatch.setattr(discovery.os, "scandir", guarded_scandir)

    assert (
        discover_trace_stores({"missing": missing, "denied": denied}, environ={}) == ()
    )


def test_platform_rules_and_extra_roots_are_explicit(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".codex" / "sessions").mkdir(parents=True)
    (home / "Library" / "Application Support" / "Claude" / "projects").mkdir(
        parents=True
    )
    (
        home / "Library" / "Application Support" / "Claude" / "projects" / "run"
    ).write_bytes(b"trace")
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "event").write_bytes(b"more")

    results = discover_trace_stores(
        platform_name="Darwin",
        home=home,
        environ={TRACE_ROOTS_ENV_VAR: f"synthetic={extra}"},
    )

    assert results == (
        TraceStore("claude", 1, 5),
        TraceStore("codex", 0, 0),
        TraceStore("synthetic", 1, 4),
    )


def test_discovery_can_be_opted_out(tmp_path: Path) -> None:
    root = tmp_path / "traces"
    root.mkdir()
    (root / "payload").write_bytes(b"secret-free synthetic data")

    assert (
        discover_trace_stores(
            {"synthetic": root},
            environ={TRACE_DISCOVERY_ENV_VAR: "off"},
        )
        == ()
    )
    assert discover_trace_stores({"synthetic": root}, enabled=False, environ={}) == ()


def test_trace_store_serialization_has_no_root_or_payload() -> None:
    summary = TraceStore("synthetic", 4, 12)

    assert summary.size_bytes == 12
    assert summary.to_dict() == {
        "store_type": "synthetic",
        "file_count": 4,
        "byte_size": 12,
    }
