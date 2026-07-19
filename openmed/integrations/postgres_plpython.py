"""PostgreSQL PL/Python bodies for in-database de-identification.

The accompanying ``sql/postgres_deidentify.sql`` migration exposes two SQL
functions:

* ``openmed_deidentify(text) -> text``
* ``openmed_deidentify_batch(text[]) -> SETOF text``

PostgreSQL passes PL/Python's per-session ``GD`` dictionary to the functions
below.  The cached :class:`~openmed.core.models.ModelLoader` owns the warmed
model pipeline, so repeated scalar calls and batch calls in the same database
session do not reload it.  This module deliberately contains no logging calls:
raw clinical text must never be written to PostgreSQL logs.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL_NAME = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
DEFAULT_POLICY = "hipaa_safe_harbor"

_SESSION_CACHE_KEY = "openmed.postgres_plpython.pipeline.v1"

ProcessBatch = Callable[..., Any]
LoaderFactory = Callable[[], Any]


@dataclass(frozen=True)
class _SessionState:
    """Objects retained in PL/Python's per-session global dictionary."""

    process_batch: ProcessBatch
    pipeline_loader: Any


def deidentify_text(
    text: str | None,
    session_globals: MutableMapping[str, Any],
    *,
    process_batch_fn: ProcessBatch | None = None,
    loader_factory: LoaderFactory | None = None,
) -> str | None:
    """Return one de-identified PostgreSQL text value.

    Args:
        text: PostgreSQL ``text`` value. ``None`` is returned unchanged for
            direct Python callers; the SQL function is also declared
            ``STRICT`` so PostgreSQL normally handles SQL ``NULL`` itself.
        session_globals: PL/Python's per-session ``GD`` dictionary.
        process_batch_fn: Optional test seam replacing ``process_batch``.
        loader_factory: Optional test seam replacing ``ModelLoader``.

    Returns:
        The de-identified text, or ``None`` for a null input.
    """

    if text is None or text == "":
        return text

    return deidentify_batch(
        [text],
        session_globals,
        process_batch_fn=process_batch_fn,
        loader_factory=loader_factory,
    )[0]


def deidentify_batch(
    texts: Sequence[str | None] | None,
    session_globals: MutableMapping[str, Any],
    *,
    process_batch_fn: ProcessBatch | None = None,
    loader_factory: LoaderFactory | None = None,
) -> list[str | None]:
    """Return de-identified PostgreSQL rows in input order.

    One ``process_batch`` call handles all non-empty values. SQL ``NULL`` and
    empty-string array elements are passed through without model inference.
    The loader cached in ``session_globals`` retains its warmed pipeline across
    calls for the lifetime of the PostgreSQL session.

    Args:
        texts: PostgreSQL ``text[]`` values, including optional null elements.
        session_globals: PL/Python's per-session ``GD`` dictionary.
        process_batch_fn: Optional test seam replacing ``process_batch``.
        loader_factory: Optional test seam replacing ``ModelLoader``.

    Returns:
        One output value for every input array element, in the same order.

    Raises:
        TypeError: If an array element is not text or null.
        RuntimeError: If OpenMed cannot de-identify every non-empty value. The
            message intentionally omits source text and underlying exceptions.
    """

    if texts is None:
        return []

    values = list(texts)
    if any(value is not None and not isinstance(value, str) for value in values):
        raise TypeError("PostgreSQL text[] values must be strings or NULL")

    output = list(values)
    positions = [
        position
        for position, value in enumerate(values)
        if isinstance(value, str) and value != ""
    ]
    if not positions:
        return output

    state = _get_session_state(
        session_globals,
        process_batch_fn=process_batch_fn,
        loader_factory=loader_factory,
    )
    inputs = [values[position] for position in positions]

    try:
        batch_result = state.process_batch(
            inputs,
            model_name=DEFAULT_MODEL_NAME,
            operation="deidentify",
            method="mask",
            policy=DEFAULT_POLICY,
            loader=state.pipeline_loader,
            batch_size=max(1, len(inputs)),
            continue_on_error=True,
        )
    except Exception:
        raise RuntimeError("OpenMed de-identification failed") from None

    redacted = _deidentified_texts(batch_result, expected=len(inputs))
    for position, replacement in zip(positions, redacted):
        output[position] = replacement
    return output


def _get_session_state(
    session_globals: MutableMapping[str, Any],
    *,
    process_batch_fn: ProcessBatch | None,
    loader_factory: LoaderFactory | None,
) -> _SessionState:
    cached = session_globals.get(_SESSION_CACHE_KEY)
    if cached is not None:
        if not isinstance(cached, _SessionState):
            raise RuntimeError("OpenMed PL/Python session cache is invalid")
        return cached

    process_batch = process_batch_fn or _default_process_batch()
    create_loader = loader_factory or _default_loader_factory
    state = _SessionState(
        process_batch=process_batch,
        pipeline_loader=create_loader(),
    )
    session_globals[_SESSION_CACHE_KEY] = state
    return state


def _deidentified_texts(batch_result: Any, *, expected: int) -> list[str]:
    items = getattr(batch_result, "items", batch_result)
    try:
        item_count = len(items)
    except TypeError:
        raise RuntimeError(
            "OpenMed de-identification returned invalid results"
        ) from None

    if item_count != expected:
        raise RuntimeError("OpenMed de-identification returned invalid results")

    redacted: list[str] = []
    for item in items:
        if hasattr(item, "success") and not item.success:
            raise RuntimeError("OpenMed de-identification failed for a batch item")

        result = getattr(item, "result", item)
        if isinstance(result, str):
            redacted.append(result)
            continue

        replacement = getattr(result, "deidentified_text", None)
        if not isinstance(replacement, str):
            raise RuntimeError("OpenMed de-identification returned invalid results")
        redacted.append(replacement)

    return redacted


def _default_process_batch() -> ProcessBatch:
    from openmed.processing import process_batch

    return process_batch


def _default_loader_factory() -> Any:
    from openmed.core import ModelLoader

    return ModelLoader()


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_POLICY",
    "deidentify_batch",
    "deidentify_text",
]
