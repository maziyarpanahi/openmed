"""Asynchronous wrappers for OpenMed's synchronous public helpers.

The wrappers in this module keep inference behavior in the established
synchronous implementations and move the blocking call to a worker thread.
Importing :mod:`openmed` does not import this module or :mod:`asyncio`; the
top-level async helpers are resolved lazily on first access.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from . import analyze_text as _analyze_text
from .core.pii import deidentify as _deidentify
from .core.pii import extract_pii as _extract_pii
from .processing import process_batch as _process_batch

if TYPE_CHECKING:
    from .core.audit import AuditReport
    from .core.pii import DeidentificationResult
    from .core.results import AnalyzeResult
    from .processing import BatchResult
    from .processing.outputs import PredictionResult

_ResultT = TypeVar("_ResultT")


async def _run_in_thread(
    function: Callable[..., _ResultT],
    /,
    *args: Any,
    **kwargs: Any,
) -> _ResultT:
    """Run ``function`` in asyncio's worker pool with context propagation."""

    return await asyncio.to_thread(function, *args, **kwargs)


async def aextract_pii(*args: Any, **kwargs: Any) -> "PredictionResult":
    """Asynchronously call :func:`openmed.extract_pii` in a worker thread.

    Args:
        *args: Positional arguments accepted by :func:`openmed.extract_pii`.
        **kwargs: Keyword arguments accepted by :func:`openmed.extract_pii`.

    Returns:
        The same prediction result returned by :func:`openmed.extract_pii`.
    """

    return await _run_in_thread(_extract_pii, *args, **kwargs)


async def adeidentify(
    *args: Any,
    **kwargs: Any,
) -> "DeidentificationResult | AuditReport":
    """Asynchronously call :func:`openmed.deidentify` in a worker thread.

    Args:
        *args: Positional arguments accepted by :func:`openmed.deidentify`.
        **kwargs: Keyword arguments accepted by :func:`openmed.deidentify`.

    Returns:
        The same de-identification or audit result returned by
        :func:`openmed.deidentify`.
    """

    return await _run_in_thread(_deidentify, *args, **kwargs)


async def aanalyze_text(
    *args: Any,
    **kwargs: Any,
) -> "AnalyzeResult | str | list[dict[str, Any]]":
    """Asynchronously call :func:`openmed.analyze_text` in a worker thread.

    Args:
        *args: Positional arguments accepted by :func:`openmed.analyze_text`.
        **kwargs: Keyword arguments accepted by :func:`openmed.analyze_text`.

    Returns:
        The same structured or rendered result returned by
        :func:`openmed.analyze_text`.
    """

    return await _run_in_thread(_analyze_text, *args, **kwargs)


async def abatch(*args: Any, **kwargs: Any) -> "BatchResult":
    """Asynchronously call :func:`openmed.process_batch` in a worker thread.

    Args:
        *args: Positional arguments accepted by :func:`openmed.process_batch`.
        **kwargs: Keyword arguments accepted by :func:`openmed.process_batch`.

    Returns:
        The same batch result returned by :func:`openmed.process_batch`.
    """

    return await _run_in_thread(_process_batch, *args, **kwargs)


# ``inspect.signature`` follows ``__wrapped__`` while the async helpers retain
# their distinct names and coroutine identity. This keeps runtime signature
# discovery aligned with the synchronous public API without duplicating its
# long, evolving parameter lists.
setattr(aextract_pii, "__wrapped__", _extract_pii)
setattr(adeidentify, "__wrapped__", _deidentify)
setattr(aanalyze_text, "__wrapped__", _analyze_text)
setattr(abatch, "__wrapped__", _process_batch)


__all__ = ["aanalyze_text", "abatch", "adeidentify", "aextract_pii"]
