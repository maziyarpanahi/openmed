"""Async wrappers for OpenMed's synchronous text-processing helpers.

The model and PII pipelines remain synchronous. These helpers run those
pipelines in worker threads so async applications can await them without
blocking the event loop.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

if TYPE_CHECKING:
    from .core import ModelLoader, OpenMedConfig
    from .core.audit import AuditReport
    from .core.budget import RequestBudget
    from .core.lang_id_codemix import TokenLIDHook
    from .core.pii import DeidentificationMethod, DeidentificationResult
    from .core.results import AnalyzeResult
    from .core.surrogate_vault import SurrogateVault
    from .processing.outputs import PredictionResult


_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_ResultT = TypeVar("_ResultT")


def _resolve_sync_export(name: str) -> Callable[..., Any]:
    """Resolve a sync public export only when an async call is executed."""
    import openmed

    return getattr(openmed, name)


def _call_sync(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Resolve and invoke a synchronous export entirely in a worker thread."""

    return _resolve_sync_export(name)(*args, **kwargs)


async def _run_sync(name: str, *args: Any, **kwargs: Any) -> Any:
    """Run a top-level synchronous export in a worker thread."""
    return await asyncio.to_thread(_call_sync, name, args, kwargs)


async def aextract_pii(
    text: str | bytes | bytearray | memoryview,
    model_name: str = _DEFAULT_PII_MODEL,
    confidence_threshold: float = 0.5,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    cache_results: bool = False,
    max_cache_entries: int = 128,
    normalize_accents: Optional[bool] = None,
    *,
    preserve_whitespace: bool = False,
    locale: Optional[str] = None,
    loader: Optional["ModelLoader"] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    custom_recognizer: Any = None,
    abdm: Optional[bool] = None,
    code_mixed: bool = False,
    token_language_tags: Optional[Sequence[Any]] = None,
    lid_model: Optional["TokenLIDHook"] = None,
    transliterated_name_config: Any = None,
    budget: Optional[RequestBudget] = None,
) -> PredictionResult:
    """Extract PII without blocking the current event loop."""
    return await _run_sync(
        "extract_pii",
        text,
        model_name,
        confidence_threshold,
        config,
        use_smart_merging,
        lang,
        cache_results,
        max_cache_entries,
        normalize_accents,
        preserve_whitespace=preserve_whitespace,
        locale=locale,
        loader=loader,
        batch_size=batch_size,
        num_workers=num_workers,
        custom_recognizer=custom_recognizer,
        abdm=abdm,
        code_mixed=code_mixed,
        token_language_tags=token_language_tags,
        lid_model=lid_model,
        transliterated_name_config=transliterated_name_config,
        budget=budget,
    )


async def adeidentify(
    text: str | bytes | bytearray | memoryview,
    method: DeidentificationMethod = "mask",
    model_name: str = _DEFAULT_PII_MODEL,
    confidence_threshold: float = 0.7,  # Higher threshold for safety
    keep_year: bool = False,
    shift_dates: Optional[bool] = None,
    date_shift_days: Optional[int] = None,
    patient_key: Optional[str | bytes] = None,
    date_shift_max_days: Optional[int] = None,
    date_shift_secret: Optional[str | bytes] = None,
    keep_mapping: bool = False,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    use_safety_sweep: bool = True,
    *,
    consistent: bool = False,
    seed: Optional[int] = None,
    locale: Optional[str] = None,
    surrogate_vault: Optional["SurrogateVault"] = None,
    loader: Optional["ModelLoader"] = None,
    policy: Optional[str] = None,
    calibration_thresholds_path: Optional[str | Path] = None,
    custom_recognizer: Any = None,
    abdm: Optional[bool] = None,
    code_mixed: bool = False,
    token_language_tags: Optional[Sequence[Any]] = None,
    lid_model: Optional["TokenLIDHook"] = None,
    transliterated_name_config: Any = None,
    audit: bool = False,
    cache_results: bool = False,
    max_cache_entries: int = 128,
    budget: Optional[RequestBudget] = None,
) -> DeidentificationResult | "AuditReport":
    """De-identify text without blocking the current event loop."""
    return await _run_sync(
        "deidentify",
        text,
        method,
        model_name,
        confidence_threshold,
        keep_year,
        shift_dates,
        date_shift_days,
        patient_key,
        date_shift_max_days,
        date_shift_secret,
        keep_mapping,
        config,
        use_smart_merging,
        lang,
        normalize_accents,
        use_safety_sweep,
        consistent=consistent,
        seed=seed,
        locale=locale,
        surrogate_vault=surrogate_vault,
        loader=loader,
        policy=policy,
        calibration_thresholds_path=calibration_thresholds_path,
        custom_recognizer=custom_recognizer,
        abdm=abdm,
        code_mixed=code_mixed,
        token_language_tags=token_language_tags,
        lid_model=lid_model,
        transliterated_name_config=transliterated_name_config,
        audit=audit,
        cache_results=cache_results,
        max_cache_entries=max_cache_entries,
        budget=budget,
    )


async def aanalyze_text(
    text: str,
    model_name: str = "disease_detection_superclinical",
    *,
    model_id: Optional[str] = None,
    config: Optional[OpenMedConfig] = None,
    loader: Optional[ModelLoader] = None,
    aggregation_strategy: Optional[str] = "simple",
    output_format: str = "dict",
    include_confidence: bool = True,
    confidence_threshold: Optional[float] = 0.0,
    group_entities: bool = False,
    formatter_kwargs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_fast_tokenizer: bool = True,
    sentence_detection: bool = True,
    sentence_language: str = "en",
    sentence_clean: bool = False,
    sentence_segmenter: Optional[Any] = None,
    sentence_backend: Literal["auto", "yasbd"] = "auto",
    assert_context: bool = False,
    cache_results: bool = False,
    max_cache_entries: int = 128,
    **pipeline_kwargs: Any,
) -> Union[AnalyzeResult, str, List[Dict[str, Any]]]:
    """Analyze text without blocking the current event loop."""
    return await _run_sync(
        "analyze_text",
        text,
        model_name,
        model_id=model_id,
        config=config,
        loader=loader,
        aggregation_strategy=aggregation_strategy,
        output_format=output_format,
        include_confidence=include_confidence,
        confidence_threshold=confidence_threshold,
        group_entities=group_entities,
        formatter_kwargs=formatter_kwargs,
        metadata=metadata,
        use_fast_tokenizer=use_fast_tokenizer,
        sentence_detection=sentence_detection,
        sentence_language=sentence_language,
        sentence_clean=sentence_clean,
        sentence_segmenter=sentence_segmenter,
        sentence_backend=sentence_backend,
        assert_context=assert_context,
        cache_results=cache_results,
        max_cache_entries=max_cache_entries,
        **pipeline_kwargs,
    )


async def abatch(
    operation: Callable[..., _ResultT | Awaitable[_ResultT]],
    items: Iterable[Any],
    *,
    max_concurrency: Optional[int] = None,
    **kwargs: Any,
) -> list[_ResultT]:
    """Apply an OpenMed operation to items concurrently and preserve order.

    Async operations such as :func:`aextract_pii` are awaited directly. A
    synchronous operation is run through :func:`asyncio.to_thread`, which also
    makes this helper useful with the original sync APIs. ``kwargs`` are
    passed to every item, and ``max_concurrency`` can bound simultaneous work.
    """
    if not callable(operation):
        raise TypeError("operation must be callable")
    if max_concurrency is not None and (
        not isinstance(max_concurrency, int)
        or isinstance(max_concurrency, bool)
        or max_concurrency < 1
    ):
        raise ValueError("max_concurrency must be positive")

    try:
        values = await asyncio.to_thread(tuple, items)
    except Exception:
        raise ValueError("items could not be read") from None

    async def run_item(item: Any) -> _ResultT:
        if asyncio.iscoroutinefunction(operation):
            return await operation(item, **kwargs)
        result = await asyncio.to_thread(operation, item, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def gather_tasks(tasks: list[asyncio.Task[Any]]) -> list[Any]:
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    if not values:
        return []
    if max_concurrency is None or max_concurrency >= len(values):
        tasks = [asyncio.create_task(run_item(item)) for item in values]
        return cast(list[_ResultT], await gather_tasks(tasks))

    missing = object()
    results: list[_ResultT | object] = [missing] * len(values)
    next_index = 0

    async def worker() -> None:
        nonlocal next_index
        while next_index < len(values):
            index = next_index
            next_index += 1
            results[index] = await run_item(values[index])

    workers = [
        asyncio.create_task(worker()) for _ in range(min(max_concurrency, len(values)))
    ]
    await gather_tasks(workers)
    if any(result is missing for result in results):
        raise RuntimeError("async batch did not produce every result")
    return cast(list[_ResultT], results)


__all__ = ["aextract_pii", "adeidentify", "aanalyze_text", "abatch"]
