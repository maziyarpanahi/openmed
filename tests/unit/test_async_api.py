"""Tests for the lazy asynchronous public API."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import openmed
import openmed.aio as aio

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("async_helper", "sync_helper"),
    [
        (aio.aextract_pii, aio._extract_pii),
        (aio.adeidentify, aio._deidentify),
        (aio.aanalyze_text, aio._analyze_text),
        (aio.abatch, aio._process_batch),
    ],
)
def test_async_helpers_preserve_sync_signatures(async_helper, sync_helper) -> None:
    assert inspect.iscoroutinefunction(async_helper)
    assert inspect.signature(async_helper) == inspect.signature(sync_helper)


@pytest.mark.parametrize(
    ("async_name", "sync_name"),
    [
        ("aextract_pii", "_extract_pii"),
        ("adeidentify", "_deidentify"),
        ("aanalyze_text", "_analyze_text"),
        ("abatch", "_process_batch"),
    ],
)
def test_async_helpers_offload_and_return_sync_result(
    monkeypatch: pytest.MonkeyPatch,
    async_name: str,
    sync_name: str,
) -> None:
    caller_thread = threading.get_ident()
    observed: dict[str, Any] = {}
    expected = object()

    def fake_sync(*args: Any, **kwargs: Any) -> object:
        observed.update(
            args=args,
            kwargs=kwargs,
            worker_thread=threading.get_ident(),
        )
        return expected

    monkeypatch.setattr(aio, sync_name, fake_sync)
    result = asyncio.run(getattr(aio, async_name)("synthetic", marker=7))

    assert result is expected
    assert observed == {
        "args": ("synthetic",),
        "kwargs": {"marker": 7},
        "worker_thread": observed["worker_thread"],
    }
    assert observed["worker_thread"] != caller_thread


def test_async_helper_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError("synthetic failure")

    monkeypatch.setattr(aio, "_extract_pii", fail)

    with pytest.raises(ValueError, match="synthetic failure"):
        asyncio.run(aio.aextract_pii("synthetic"))


def test_top_level_async_helpers_are_lazy_exports() -> None:
    for name in ("aextract_pii", "adeidentify", "aanalyze_text", "abatch"):
        helper = getattr(openmed, name)
        assert helper is getattr(aio, name)
        assert inspect.iscoroutinefunction(helper)


def test_import_openmed_does_not_import_asyncio() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "assert 'asyncio' not in sys.modules; "
                "import openmed; "
                "assert 'asyncio' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert check.returncode == 0, check.stderr
