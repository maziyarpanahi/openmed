"""Tests for the lazy async wrappers around OpenMed's sync APIs."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import threading
from unittest.mock import Mock, patch

import pytest

import openmed
import openmed.aio as aio_module
from openmed.processing.outputs import PredictionResult


def test_import_openmed_does_not_load_asyncio_or_aio():
    """The sync package import remains free of the async module."""
    probe = (
        "import sys; import openmed; "
        "assert 'asyncio' not in sys.modules; "
        "assert 'openmed.aio' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_async_wrappers_match_sync_signatures():
    """Async wrappers accept the same arguments and defaults as sync APIs."""
    assert inspect.signature(openmed.aextract_pii) == inspect.signature(
        openmed.extract_pii
    )
    assert inspect.signature(openmed.adeidentify) == inspect.signature(
        openmed.deidentify
    )
    assert inspect.signature(openmed.aanalyze_text) == inspect.signature(
        openmed.analyze_text
    )


def test_aextract_pii_returns_sync_result_type_without_blocking_call_site():
    """The PII wrapper delegates in a worker and returns the sync result."""
    expected = PredictionResult(
        text="Synthetic note",
        entities=[],
        model_name="fixture-pii-model",
        timestamp="2026-01-01T00:00:00",
    )
    sync_extract = Mock(return_value=expected)

    with patch.object(openmed, "extract_pii", sync_extract):
        result = asyncio.run(
            openmed.aextract_pii("Synthetic note", model_name="fixture-pii-model")
        )

    assert isinstance(result, PredictionResult)
    assert result is expected
    sync_extract.assert_called_once()
    assert sync_extract.call_args.args[0] == "Synthetic note"
    assert sync_extract.call_args.args[1] == "fixture-pii-model"


def test_lazy_sync_resolution_also_runs_in_the_worker_thread():
    """Resolving a lazy sync export must not block the event-loop thread."""
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []
    expected = object()

    def resolve(_name: str):
        worker_threads.append(threading.get_ident())
        return lambda *_args, **_kwargs: expected

    with patch.object(aio_module, "_resolve_sync_export", resolve):
        result = asyncio.run(openmed.aextract_pii("Synthetic note"))

    assert result is expected
    assert worker_threads
    assert all(thread != caller_thread for thread in worker_threads)


def test_other_async_wrappers_delegate_to_their_sync_exports():
    """De-identification and analysis wrappers preserve delegated results."""
    expected_deidentify = object()
    expected_analyze = object()

    with (
        patch.object(openmed, "deidentify", return_value=expected_deidentify) as deid,
        patch.object(openmed, "analyze_text", return_value=expected_analyze) as analyze,
    ):
        deidentified = asyncio.run(openmed.adeidentify("Synthetic note", method="mask"))
        analyzed = asyncio.run(openmed.aanalyze_text("Synthetic note"))

    assert deidentified is expected_deidentify
    assert analyzed is expected_analyze
    deid.assert_called_once()
    analyze.assert_called_once()


def test_abatch_preserves_order_and_supports_async_operations():
    """Batch execution returns results in input order."""

    async def operation(value: int, offset: int = 0) -> int:
        await asyncio.sleep(0)
        return value + offset

    result = asyncio.run(
        openmed.abatch(operation, [2, 1, 3], offset=10, max_concurrency=2)
    )

    assert result == [12, 11, 13]


def test_abatch_concurrency_limit_bounds_scheduled_tasks():
    """A bounded batch does not enqueue one event-loop task per input."""
    peak_tasks = 0

    async def operation(value: int) -> int:
        nonlocal peak_tasks
        peak_tasks = max(peak_tasks, len(asyncio.all_tasks()))
        await asyncio.sleep(0)
        return value

    values = list(range(50))
    result = asyncio.run(openmed.abatch(operation, values, max_concurrency=3))

    assert result == values
    assert peak_tasks <= 4


def test_abatch_materializes_items_off_the_event_loop_thread():
    """Reading a synchronous iterable must not block the event-loop thread."""
    caller_thread = threading.get_ident()
    iterator_threads: list[int] = []

    def items():
        iterator_threads.append(threading.get_ident())
        yield 1
        iterator_threads.append(threading.get_ident())
        yield 2

    result = asyncio.run(openmed.abatch(lambda value: value, items()))

    assert result == [1, 2]
    assert iterator_threads
    assert all(thread != caller_thread for thread in iterator_threads)


@pytest.mark.parametrize("limit", [False, 0, -1, 1.5, "2"])
def test_abatch_rejects_invalid_concurrency_limits(limit: object):
    """Only positive, non-boolean integer limits are accepted."""
    with pytest.raises(ValueError, match="must be positive"):
        asyncio.run(openmed.abatch(lambda value: value, [1], max_concurrency=limit))


def test_abatch_hides_values_from_iterator_failures():
    """Input iteration failures do not echo potentially sensitive values."""
    secret_marker = "synthetic-sensitive-value"

    def failing_items():
        yield 1
        raise RuntimeError(secret_marker)

    with pytest.raises(ValueError) as captured:
        asyncio.run(openmed.abatch(lambda value: value, failing_items()))

    assert secret_marker not in str(captured.value)
