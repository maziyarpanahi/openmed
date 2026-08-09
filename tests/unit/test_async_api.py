"""Tests for the lazy async wrappers around OpenMed's sync APIs."""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
from unittest.mock import Mock, patch

import openmed
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
