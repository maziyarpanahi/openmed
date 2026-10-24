"""Reproduce the committed GLiNER2.5 language-decision summary offline.

Downloads the pinned ``fastino/GLiNER2.5-multi-Decide`` revision once (1.15 GB
of Apache-2.0 safetensors), then reruns the synthetic evaluation from the
local cache with every socket blocked and compares the result with
``docs/benchmarks/gliner2-language-decisions.json``. Set
``OPENMED_RUN_GLINER2_CHECKPOINT_TESTS=1`` with ``openmed[gliner2]`` installed
to run it.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from openmed.eval import language_decisions as ld

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_RUN_ENV = "OPENMED_RUN_GLINER2_CHECKPOINT_TESTS"
_SUMMARY = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "benchmarks"
    / "gliner2-language-decisions.json"
)


def _require_runtime() -> None:
    if os.environ.get(_RUN_ENV, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(f"set {_RUN_ENV}=1 to download and run GLiNER2.5 checkpoints")
    pytest.importorskip("gliner2", reason="install openmed[gliner2]")
    pytest.importorskip("torch", reason="install openmed[gliner2]")
    pytest.importorskip("transformers", reason="install openmed[gliner2]")


@contextmanager
def _no_network() -> Iterator[None]:
    originals = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.create_connection,
        socket.getaddrinfo,
    )

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the evaluation attempted network access")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.create_connection,
            socket.getaddrinfo,
        ) = originals


def test_pinned_checkpoint_reproduces_the_committed_summary_offline() -> None:
    _require_runtime()
    from openmed.ner.families.gliner2 import clear_gliner2_cache, load_gliner2_handle

    # The first load may download the pinned snapshot; the rerun is cache-only.
    load_gliner2_handle(
        ld.GLINER2_LANGUAGE_DECISION_MODEL,
        revision=ld.GLINER2_LANGUAGE_DECISION_REVISION,
        device="cpu",
    )
    clear_gliner2_cache()
    with _no_network():
        report = ld.run_gliner2_language_decision_eval(local_files_only=True)

    committed = json.loads(_SUMMARY.read_text(encoding="utf-8"))
    assert ld.report_differences(committed, report) == []
    for backend in report["backends"].values():
        assert backend["revision"] == ld.GLINER2_LANGUAGE_DECISION_REVISION
        assert backend["license_id"] == "Apache-2.0"

    text = ld.format_report_json(report)
    for sample in ld.build_language_decision_samples():
        assert sample.text not in text
