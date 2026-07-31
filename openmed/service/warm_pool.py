"""Model warm-pool support for the OpenMed REST service."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from openmed.utils.validation import validate_model_name

from .keep_alive import parse_keep_alive

DEFAULT_WARM_PIPELINE_TASK = "token-classification"
DEFAULT_WARM_AGGREGATION_STRATEGY = "simple"
DEFAULT_WARM_USE_FAST_TOKENIZER = True
DEFAULT_MODEL_FOOTPRINT_BYTES = 512 * 1024 * 1024
DEFAULT_MEMORY_ADMISSION_WAIT_SECONDS = 0.05

_BYTE_SIZE_PATTERN = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>b|kb|kib|mb|mib|gb|gib)?\s*$",
    re.IGNORECASE,
)
_BYTE_SIZE_UNITS = {
    None: 1,
    "b": 1,
    "kb": 1_000,
    "kib": 1024,
    "mb": 1_000_000,
    "mib": 1024 * 1024,
    "gb": 1_000_000_000,
    "gib": 1024 * 1024 * 1024,
}


def parse_max_resident_models(raw_value: Any) -> Optional[int]:
    """Parse the optional service warm-pool resident model limit."""
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError("max resident models must be an integer")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return None
        try:
            value = int(stripped)
        except ValueError as exc:
            raise ValueError("max resident models must be an integer") from exc
    else:
        raise ValueError("max resident models must be an integer")

    if value < 1:
        raise ValueError("max resident models must be greater than or equal to 1")
    return value


def parse_model_memory_budget_bytes(raw_value: Any) -> Optional[int]:
    """Parse the optional service warm-pool memory budget in bytes."""
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and not raw_value.strip():
        return None
    return _parse_positive_byte_size(
        raw_value,
        setting_name="model memory budget",
    )


def parse_default_model_footprint_bytes(raw_value: Any) -> int:
    """Parse the fallback per-model memory footprint used for admission."""
    if raw_value is None:
        return DEFAULT_MODEL_FOOTPRINT_BYTES
    if isinstance(raw_value, str) and not raw_value.strip():
        return DEFAULT_MODEL_FOOTPRINT_BYTES
    return _parse_positive_byte_size(
        raw_value,
        setting_name="default model footprint",
    )


def parse_memory_admission_wait_seconds(raw_value: Any) -> float:
    """Parse the bounded wait used when the model memory budget is saturated."""
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return DEFAULT_MEMORY_ADMISSION_WAIT_SECONDS
    if isinstance(raw_value, bool):
        raise ValueError("memory admission wait must be a non-negative number")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("memory admission wait must be a non-negative number") from exc
    if value < 0:
        raise ValueError("memory admission wait must be greater than or equal to 0")
    return value


def _parse_positive_byte_size(raw_value: Any, *, setting_name: str) -> int:
    if isinstance(raw_value, bool):
        raise ValueError(f"{setting_name} must be a positive byte size")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, float):
        if not raw_value.is_integer():
            raise ValueError(f"{setting_name} must be a whole number of bytes")
        value = int(raw_value)
    elif isinstance(raw_value, str):
        match = _BYTE_SIZE_PATTERN.match(raw_value)
        if match is None:
            raise ValueError(
                f"{setting_name} must be a byte size like '512MiB' or '2GB'"
            )
        value = int(
            float(match.group("value"))
            * _BYTE_SIZE_UNITS[
                match.group("unit").lower() if match.group("unit") else None
            ]
        )
    else:
        raise ValueError(f"{setting_name} must be a positive byte size")

    if value <= 0:
        raise ValueError(f"{setting_name} must be greater than 0")
    return value


class WarmPoolBackpressureError(RuntimeError):
    """Raised when the warm-pool cannot admit a model within its wait bound."""

    def __init__(
        self,
        *,
        model_name: str,
        required_bytes: int,
        budget_bytes: Optional[int],
        wait_seconds: float,
    ) -> None:
        self.model_name = model_name
        self.required_bytes = int(required_bytes)
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.wait_seconds = float(wait_seconds)
        super().__init__(
            "Model memory budget is saturated; retry later"
            if budget_bytes is not None
            else "Model admission is saturated; retry later"
        )


@dataclass
class WarmPoolEntry:
    """Resident model bookkeeping tracked by :class:`WarmPool`."""

    model_name: str
    last_used: float
    active_requests: int = 0
    keep_alive_deadline: Optional[float] = None
    handles: dict[Tuple[Any, ...], Any] = field(default_factory=dict)
    footprint_bytes: int = 0
    pending_footprint_bytes: int = 0
    loading_keys: set[Tuple[Any, ...]] = field(default_factory=set)

    @property
    def resident(self) -> bool:
        """Return whether this entry owns any cached model handles."""
        return bool(self.handles)

    @property
    def loading(self) -> bool:
        """Return whether a cold load is currently in progress."""
        return bool(self.loading_keys)


@dataclass
class WarmPool:
    """Bounded model warm-pool that proxies a shared model loader.

    The pool caches exact pipeline handles by model and pipeline options while
    applying model-level keep-alive and least-recently-used eviction policy.
    """

    loader_provider: Callable[[], Any]
    warm_models: Tuple[str, ...] = ()
    max_resident_models: Optional[int] = None
    memory_budget_bytes: Optional[int] = None
    default_model_footprint_bytes: int = DEFAULT_MODEL_FOOTPRINT_BYTES
    memory_admission_wait_seconds: float = DEFAULT_MEMORY_ADMISSION_WAIT_SECONDS
    footprint_provider: Optional[Callable[[str], int]] = None
    default_keep_alive_seconds: Optional[float] = None
    metrics: Optional[Any] = None
    clock: Callable[[], float] = time.monotonic
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _condition: threading.Condition = field(init=False, repr=False)
    _load_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _entries: dict[str, WarmPoolEntry] = field(default_factory=dict, repr=False)
    _timers: dict[str, threading.Timer] = field(default_factory=dict, repr=False)
    _local: threading.local = field(default_factory=threading.local, repr=False)

    def __post_init__(self) -> None:
        self.max_resident_models = parse_max_resident_models(self.max_resident_models)
        self.memory_budget_bytes = parse_model_memory_budget_bytes(
            self.memory_budget_bytes
        )
        self.default_model_footprint_bytes = parse_default_model_footprint_bytes(
            self.default_model_footprint_bytes
        )
        self.memory_admission_wait_seconds = parse_memory_admission_wait_seconds(
            self.memory_admission_wait_seconds
        )
        self.warm_models = tuple(self.warm_models)
        self._condition = threading.Condition(self._lock)

    @property
    def loader(self) -> Any:
        """Return the wrapped model loader."""
        return self.loader_provider()

    def preload(self) -> None:
        """Load the configured warm set with the default service pipeline shape."""
        for model_name in self.warm_models:
            self.create_pipeline(
                model_name,
                task=DEFAULT_WARM_PIPELINE_TASK,
                aggregation_strategy=DEFAULT_WARM_AGGREGATION_STRATEGY,
                use_fast_tokenizer=DEFAULT_WARM_USE_FAST_TOKENIZER,
            )

    def begin_request(self, model_name: str) -> str:
        """Mark a model as active for one service request."""
        model_key = self.resolve_model_name(model_name)
        with self._lock:
            now = self.clock()
            self._drop_expired_locked(now)
            entry = self._entry_for_model_locked(model_key, now)
            self._cancel_timer_locked(model_key)
            entry.keep_alive_deadline = None
            entry.active_requests += 1
            entry.last_used = now
            self._push_request_model(model_key)
        return model_key

    def finish_request(self, model_key: str, keep_alive: Any) -> None:
        """Mark a request complete and apply keep-alive/eviction policy."""
        model_keys = self._pop_request_models(model_key)
        keep_alive_seconds = self._resolve_keep_alive_seconds(keep_alive)
        with self._lock:
            now = self.clock()
            for active_model_key in model_keys:
                self._finish_model_request_locked(
                    active_model_key,
                    keep_alive_seconds,
                    now,
                )

            self._drop_empty_idle_entries_locked()
            self._evict_over_capacity_locked()
            self._evict_over_budget_locked()
            self._sync_residency_metrics_locked()
            self._condition.notify_all()

    def _finish_model_request_locked(
        self,
        model_key: str,
        keep_alive_seconds: Optional[float],
        now: float,
    ) -> None:
        entry = self._entries.get(model_key)
        if entry is None:
            return

        entry.active_requests = max(entry.active_requests - 1, 0)
        entry.last_used = now
        if entry.active_requests:
            return

        if keep_alive_seconds is None:
            entry.keep_alive_deadline = None
        elif keep_alive_seconds <= 0:
            if entry.resident:
                self._unload_entry_locked(model_key)
            else:
                self._entries.pop(model_key, None)
        else:
            self._schedule_idle_unload_locked(
                model_key,
                keep_alive_seconds,
                now,
            )

    def create_pipeline(
        self,
        model_name: str,
        task: str = DEFAULT_WARM_PIPELINE_TASK,
        aggregation_strategy: Optional[str] = None,
        use_fast_tokenizer: bool = DEFAULT_WARM_USE_FAST_TOKENIZER,
        **kwargs: Any,
    ) -> Any:
        """Create or return a cached pipeline handle for a model."""
        model_key = self.resolve_model_name(model_name)
        cache_key = self._pipeline_cache_key(
            model_key,
            task=task,
            aggregation_strategy=aggregation_strategy,
            use_fast_tokenizer=use_fast_tokenizer,
            kwargs=kwargs,
        )

        def load_handle() -> Any:
            return self.loader.create_pipeline(
                model_key,
                task=task,
                aggregation_strategy=aggregation_strategy,
                use_fast_tokenizer=use_fast_tokenizer,
                **kwargs,
            )

        return self._get_or_load_handle(model_key, cache_key, load_handle)

    def load_model(
        self,
        model_name: str,
        force_reload: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Load or return a cached model component handle."""
        model_key = self.resolve_model_name(model_name)
        cache_key = (
            "load_model",
            self._freeze_cache_value(kwargs),
        )

        def load_handle() -> Any:
            return self.loader.load_model(
                model_key,
                force_reload=force_reload,
                **kwargs,
            )

        return self._get_or_load_handle(
            model_key,
            cache_key,
            load_handle,
            reuse_cached=not force_reload,
        )

    def get_max_sequence_length(
        self,
        model_name: str,
        *,
        tokenizer: Optional[Any] = None,
    ) -> Optional[int]:
        """Proxy sequence length inference to the wrapped loader."""
        return self.loader.get_max_sequence_length(model_name, tokenizer=tokenizer)

    def resolve_model_name(self, model_name: str) -> str:
        """Resolve a registry alias, local path, or full model id."""
        validated = validate_model_name(model_name)
        resolver = getattr(self.loader, "resolve_model_name", None)
        if callable(resolver):
            return resolver(validated)
        return validated

    def _get_or_load_handle(
        self,
        model_key: str,
        cache_key: Tuple[Any, ...],
        load_handle: Callable[[], Any],
        *,
        reuse_cached: bool = True,
    ) -> Any:
        reserved_bytes = 0
        while True:
            with self._lock:
                now = self.clock()
                self._drop_expired_locked(now)
                entry = self._entry_for_model_locked(model_key, now)
                self._activate_current_request_model_locked(model_key, now)

                if reuse_cached and cache_key in entry.handles:
                    entry.last_used = now
                    return entry.handles[cache_key]

                if cache_key in entry.loading_keys or (
                    not entry.resident and entry.loading
                ):
                    self._condition.wait()
                    continue

                reserved_bytes = self._reserve_load_budget_locked(entry)
                entry.loading_keys.add(cache_key)
                entry.last_used = now
                self._sync_residency_metrics_locked()
                break

        started_at = time.perf_counter()
        try:
            with self._load_lock:
                handle = load_handle()
        except Exception:
            with self._lock:
                self._finish_failed_load_locked(model_key, cache_key, reserved_bytes)
            raise

        load_latency_seconds = time.perf_counter() - started_at
        with self._lock:
            now = self.clock()
            entry = self._entry_for_model_locked(model_key, now)
            entry.handles[cache_key] = handle
            entry.last_used = now
            entry.loading_keys.discard(cache_key)
            if reserved_bytes:
                entry.pending_footprint_bytes = max(
                    entry.pending_footprint_bytes - reserved_bytes,
                    0,
                )
                entry.footprint_bytes = self._measured_model_footprint_bytes_locked(
                    model_key,
                    fallback_bytes=reserved_bytes,
                )
            elif entry.resident and entry.footprint_bytes <= 0:
                entry.footprint_bytes = self._measured_model_footprint_bytes_locked(
                    model_key,
                    fallback_bytes=self.default_model_footprint_bytes,
                )

            self._record_model_load_locked()
            self._record_model_load_latency_locked(load_latency_seconds)
            self._evict_over_capacity_locked()
            self._evict_over_budget_locked()
            self._sync_residency_metrics_locked()
            self._condition.notify_all()
            return handle

    def loaded_models(self) -> dict[str, Any]:
        """Return model cache, warm-pool, and keep-alive status."""
        with self._lock:
            now = self.clock()
            self._drop_expired_locked(now)
            loaded = self._loader_loaded_models_locked()
            models: dict[str, dict[str, Any]] = {}
            for model_name, cache_state in loaded.items():
                entry = self._entries.get(model_name)
                deadline = None if entry is None else entry.keep_alive_deadline
                remaining = None if deadline is None else max(deadline - now, 0.0)
                models[model_name] = {
                    **cache_state,
                    "active_requests": 0 if entry is None else entry.active_requests,
                    "keep_alive_seconds_remaining": remaining,
                    "resident": bool(entry is not None and entry.resident),
                    "loading": bool(entry is not None and entry.loading),
                    "footprint_bytes": 0 if entry is None else entry.footprint_bytes,
                    "pending_footprint_bytes": (
                        0 if entry is None else entry.pending_footprint_bytes
                    ),
                }

            for model_name, entry in self._entries.items():
                if model_name in models:
                    continue
                if (
                    not entry.resident
                    and entry.active_requests <= 0
                    and not entry.loading
                ):
                    continue
                deadline = entry.keep_alive_deadline
                remaining = None if deadline is None else max(deadline - now, 0.0)
                models[model_name] = {
                    "models": 0,
                    "tokenizers": 0,
                    "pipelines": 0,
                    "active_requests": entry.active_requests,
                    "keep_alive_seconds_remaining": remaining,
                    "resident": entry.resident,
                    "loading": entry.loading,
                    "footprint_bytes": entry.footprint_bytes,
                    "pending_footprint_bytes": entry.pending_footprint_bytes,
                }

            return {
                "default_keep_alive_seconds": self.default_keep_alive_seconds,
                "max_resident_models": self.max_resident_models,
                "memory_budget_bytes": self.memory_budget_bytes,
                "resident_memory_bytes": self._resident_memory_bytes_locked(),
                "pending_memory_bytes": self._pending_memory_bytes_locked(),
                "memory_admission_wait_seconds": self.memory_admission_wait_seconds,
                "warm_models": list(self.warm_models),
                "models": models,
            }

    def unload_model(self, model_name: str) -> dict[str, Any]:
        """Unload one inactive model from the pool and wrapped loader."""
        model_key = self.resolve_model_name(model_name)
        with self._lock:
            entry = self._entries.get(model_key)
            active_requests = 0 if entry is None else entry.active_requests
            loading = bool(entry is not None and entry.loading)
            if active_requests or loading:
                return {
                    "unloaded": False,
                    "model_name": model_key,
                    "active_requests": active_requests,
                    "loading": loading,
                    "released": self._zero_release(),
                }

            released = self._unload_entry_locked(model_key)
            self._sync_residency_metrics_locked()
            self._condition.notify_all()
            return {
                "unloaded": any(
                    released[name] for name in ("models", "tokenizers", "pipelines")
                ),
                "model_name": released.get("model_name", model_key),
                "active_requests": 0,
                "loading": False,
                "released": {
                    "models": released.get("models", 0),
                    "tokenizers": released.get("tokenizers", 0),
                    "pipelines": released.get("pipelines", 0),
                },
            }

    def unload_all_models(self) -> dict[str, Any]:
        """Unload every inactive model from the pool and wrapped loader."""
        with self._lock:
            active_models = {
                model_name: entry.active_requests
                for model_name, entry in self._entries.items()
                if entry.active_requests > 0 or entry.loading
            }
            if not active_models:
                for model_name in list(self._timers):
                    self._cancel_timer_locked(model_name)
                resident_count = sum(
                    1 for entry in self._entries.values() if entry.resident
                )
                self._entries.clear()
                unload_all = getattr(self.loader, "unload_all_models", None)
                if callable(unload_all):
                    with self._load_lock:
                        released = unload_all()
                    self._record_model_eviction_locked(
                        resident_count or int(any(released.values()))
                    )
                else:
                    released = self._unload_all_loaded()
                    if resident_count and not any(released.values()):
                        self._record_model_eviction_locked(resident_count)
                self._sync_residency_metrics_locked()
                self._condition.notify_all()
                return {
                    "unloaded": any(released.values()),
                    "released": released,
                    "active_models": {},
                }

            released = self._zero_release()
            loaded_model_names = set(self._loader_loaded_models_locked())
            loaded_model_names.update(
                model_name
                for model_name, entry in self._entries.items()
                if entry.resident
            )
            for model_name in sorted(loaded_model_names):
                if model_name in active_models:
                    continue
                model_released = self._unload_entry_locked(model_name)
                for key in released:
                    released[key] += model_released.get(key, 0)

            self._sync_residency_metrics_locked()
            self._condition.notify_all()
            return {
                "unloaded": any(released.values()),
                "released": released,
                "active_models": active_models,
            }

    def drop_expired(self) -> list[str]:
        """Drop idle entries whose keep-alive deadlines have expired."""
        with self._lock:
            return self._drop_expired_locked(self.clock())

    def resident_model_names(self) -> Tuple[str, ...]:
        """Return currently resident model names in sorted order."""
        with self._lock:
            return tuple(
                sorted(
                    model_name
                    for model_name, entry in self._entries.items()
                    if entry.resident
                )
            )

    def __getattr__(self, name: str) -> Any:
        """Proxy less common loader APIs to the wrapped loader."""
        return getattr(self.loader, name)

    def _resolve_keep_alive_seconds(self, keep_alive: Any) -> Optional[float]:
        if keep_alive is None:
            return self.default_keep_alive_seconds
        return parse_keep_alive(keep_alive)

    def _entry_for_model_locked(
        self,
        model_name: str,
        now: float,
    ) -> WarmPoolEntry:
        entry = self._entries.get(model_name)
        if entry is None:
            entry = WarmPoolEntry(model_name=model_name, last_used=now)
            self._entries[model_name] = entry
        return entry

    def _reserve_load_budget_locked(self, entry: WarmPoolEntry) -> int:
        if entry.resident:
            return 0

        estimate = self._estimated_model_footprint_bytes_locked(entry.model_name)
        if self.memory_budget_bytes is None:
            entry.pending_footprint_bytes += estimate
            return estimate

        if estimate > self.memory_budget_bytes:
            self._record_model_rejection_locked()
            raise WarmPoolBackpressureError(
                model_name=entry.model_name,
                required_bytes=estimate,
                budget_bytes=self.memory_budget_bytes,
                wait_seconds=self.memory_admission_wait_seconds,
            )

        deadline = self.clock() + self.memory_admission_wait_seconds
        while (
            self._accounted_memory_bytes_locked() + estimate > self.memory_budget_bytes
        ):
            self._evict_for_budget_locked(estimate)
            if (
                self._accounted_memory_bytes_locked() + estimate
                <= self.memory_budget_bytes
            ):
                break

            remaining = deadline - self.clock()
            if remaining <= 0:
                self._record_model_rejection_locked()
                raise WarmPoolBackpressureError(
                    model_name=entry.model_name,
                    required_bytes=estimate,
                    budget_bytes=self.memory_budget_bytes,
                    wait_seconds=self.memory_admission_wait_seconds,
                )
            self._condition.wait(timeout=remaining)
            self._drop_expired_locked(self.clock())

        entry.pending_footprint_bytes += estimate
        self._sync_residency_metrics_locked()
        return estimate

    def _finish_failed_load_locked(
        self,
        model_name: str,
        cache_key: Tuple[Any, ...],
        reserved_bytes: int,
    ) -> None:
        entry = self._entries.get(model_name)
        if entry is None:
            self._condition.notify_all()
            return
        entry.loading_keys.discard(cache_key)
        if reserved_bytes:
            entry.pending_footprint_bytes = max(
                entry.pending_footprint_bytes - reserved_bytes,
                0,
            )
        if not entry.resident and entry.active_requests <= 0 and not entry.loading:
            self._entries.pop(model_name, None)
        self._sync_residency_metrics_locked()
        self._condition.notify_all()

    def _estimated_model_footprint_bytes_locked(self, model_name: str) -> int:
        provided = self._provided_model_footprint_bytes(model_name)
        if provided is not None:
            return provided

        loader_estimate = self._loader_model_footprint_bytes(model_name)
        if loader_estimate is not None:
            return loader_estimate

        info = self._loader_model_info(model_name)
        metadata_estimate = self._model_info_footprint_bytes(info)
        if metadata_estimate is not None:
            return metadata_estimate

        return self.default_model_footprint_bytes

    def _measured_model_footprint_bytes_locked(
        self,
        model_name: str,
        *,
        fallback_bytes: int,
    ) -> int:
        provided = self._provided_model_footprint_bytes(model_name)
        if provided is not None:
            return provided

        loader_estimate = self._loader_model_footprint_bytes(model_name)
        if loader_estimate is not None:
            return loader_estimate

        info = self._loader_model_info(model_name)
        metadata_estimate = self._model_info_footprint_bytes(info)
        if metadata_estimate is not None:
            return metadata_estimate

        return max(int(fallback_bytes), 1)

    def _provided_model_footprint_bytes(self, model_name: str) -> Optional[int]:
        if self.footprint_provider is None:
            return None
        value = self.footprint_provider(model_name)
        return self._normalize_footprint_bytes(value)

    def _loader_model_footprint_bytes(self, model_name: str) -> Optional[int]:
        for method_name in (
            "model_memory_footprint_bytes",
            "measure_model_memory_bytes",
            "estimate_model_memory_bytes",
        ):
            method = getattr(self.loader, method_name, None)
            if not callable(method):
                continue
            value = method(model_name)
            normalized = self._normalize_footprint_bytes(value)
            if normalized is not None:
                return normalized
        return None

    def _loader_model_info(self, model_name: str) -> Any:
        get_info = getattr(self.loader, "get_model_info", None)
        if not callable(get_info):
            return None
        return get_info(model_name)

    def _model_info_footprint_bytes(self, info: Any) -> Optional[int]:
        peak_ram_mb = getattr(info, "peak_ram_mb", None)
        if isinstance(peak_ram_mb, Mapping):
            values = [
                float(value)
                for value in peak_ram_mb.values()
                if isinstance(value, (int, float)) and value > 0
            ]
            if values:
                return max(int(max(values) * 1024 * 1024), 1)
        elif isinstance(peak_ram_mb, (int, float)) and peak_ram_mb > 0:
            return max(int(float(peak_ram_mb) * 1024 * 1024), 1)

        param_count = getattr(info, "param_count", None)
        if isinstance(param_count, int) and param_count > 0:
            return max(param_count * 4, 1)
        return None

    def _normalize_footprint_bytes(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("model footprint must be a positive integer")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("model footprint must be a positive integer") from exc
        if normalized <= 0:
            raise ValueError("model footprint must be greater than 0")
        return normalized

    def _push_request_model(self, model_name: str) -> None:
        stack = getattr(self._local, "request_models", None)
        if stack is None:
            stack = []
            self._local.request_models = stack
        stack.append([model_name])

    def _pop_request_models(self, fallback_model_name: str) -> list[str]:
        stack = getattr(self._local, "request_models", None)
        if not stack:
            return [fallback_model_name]
        model_names = stack.pop()
        if not stack:
            del self._local.request_models
        return model_names or [fallback_model_name]

    def _current_request_models(self) -> Optional[list[str]]:
        stack = getattr(self._local, "request_models", None)
        if not stack:
            return None
        return stack[-1]

    def _activate_current_request_model_locked(
        self,
        model_name: str,
        now: float,
    ) -> None:
        current_models = self._current_request_models()
        if current_models is None or model_name in current_models:
            return

        entry = self._entry_for_model_locked(model_name, now)
        self._cancel_timer_locked(model_name)
        entry.keep_alive_deadline = None
        entry.active_requests += 1
        entry.last_used = now
        current_models.append(model_name)

    def _pipeline_cache_key(
        self,
        model_name: str,
        *,
        task: str,
        aggregation_strategy: Optional[str],
        use_fast_tokenizer: bool,
        kwargs: Mapping[str, Any],
    ) -> Tuple[Any, ...]:
        return (
            "pipeline",
            model_name,
            task,
            aggregation_strategy,
            use_fast_tokenizer,
            self._freeze_cache_value(kwargs),
        )

    def _freeze_cache_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (str(key), self._freeze_cache_value(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_cache_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze_cache_value(item) for item in value))
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return repr(value)

    def _schedule_idle_unload_locked(
        self,
        model_name: str,
        keep_alive_seconds: float,
        now: float,
    ) -> None:
        self._cancel_timer_locked(model_name)
        entry = self._entries.get(model_name)
        if entry is None:
            return
        entry.keep_alive_deadline = now + keep_alive_seconds
        timer = threading.Timer(
            keep_alive_seconds,
            self._unload_if_expired,
            args=(model_name,),
        )
        timer.daemon = True
        self._timers[model_name] = timer
        timer.start()

    def _unload_if_expired(self, model_name: str) -> None:
        with self._lock:
            now = self.clock()
            entry = self._entries.get(model_name)
            if entry is None or entry.active_requests:
                return
            deadline = entry.keep_alive_deadline
            if deadline is None or deadline > now:
                return
            self._unload_entry_locked(model_name)
            self._evict_over_capacity_locked()

    def _drop_expired_locked(self, now: float) -> list[str]:
        expired = [
            model_name
            for model_name, entry in self._entries.items()
            if (
                entry.keep_alive_deadline is not None
                and entry.keep_alive_deadline <= now
                and entry.active_requests <= 0
            )
        ]
        for model_name in expired:
            self._unload_entry_locked(model_name)
        if expired:
            self._sync_residency_metrics_locked()
            self._condition.notify_all()
        return expired

    def _evict_over_capacity_locked(self) -> list[str]:
        if self.max_resident_models is None:
            return []

        evicted: list[str] = []
        while self._resident_count_locked() > self.max_resident_models:
            candidates = [
                entry
                for entry in self._entries.values()
                if (entry.resident and entry.active_requests <= 0 and not entry.loading)
            ]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda entry: (entry.last_used, -entry.footprint_bytes),
            )
            evicted.append(victim.model_name)
            self._unload_entry_locked(victim.model_name)
        if evicted:
            self._sync_residency_metrics_locked()
            self._condition.notify_all()
        return evicted

    def _evict_over_budget_locked(self) -> list[str]:
        if self.memory_budget_bytes is None:
            return []
        evicted: list[str] = []
        while self._accounted_memory_bytes_locked() > self.memory_budget_bytes:
            victim = self._budget_victim_locked()
            if victim is None:
                break
            evicted.append(victim.model_name)
            self._unload_entry_locked(victim.model_name)
        if evicted:
            self._sync_residency_metrics_locked()
            self._condition.notify_all()
        return evicted

    def _evict_for_budget_locked(self, required_bytes: int) -> list[str]:
        if self.memory_budget_bytes is None:
            return []
        evicted: list[str] = []
        while (
            self._accounted_memory_bytes_locked() + required_bytes
            > self.memory_budget_bytes
        ):
            victim = self._budget_victim_locked()
            if victim is None:
                break
            evicted.append(victim.model_name)
            self._unload_entry_locked(victim.model_name)
        if evicted:
            self._sync_residency_metrics_locked()
            self._condition.notify_all()
        return evicted

    def _budget_victim_locked(self) -> Optional[WarmPoolEntry]:
        candidates = [
            entry
            for entry in self._entries.values()
            if entry.resident and entry.active_requests <= 0 and not entry.loading
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda entry: (entry.last_used, -entry.footprint_bytes),
        )

    def _resident_count_locked(self) -> int:
        return sum(1 for entry in self._entries.values() if entry.resident)

    def _accounted_memory_bytes_locked(self) -> int:
        return sum(
            (entry.footprint_bytes if entry.resident else 0)
            + entry.pending_footprint_bytes
            for entry in self._entries.values()
        )

    def _resident_memory_bytes_locked(self) -> int:
        return sum(
            entry.footprint_bytes for entry in self._entries.values() if entry.resident
        )

    def _pending_memory_bytes_locked(self) -> int:
        return sum(entry.pending_footprint_bytes for entry in self._entries.values())

    def _drop_empty_idle_entries_locked(self) -> None:
        empty = [
            model_name
            for model_name, entry in self._entries.items()
            if (
                not entry.resident
                and entry.active_requests <= 0
                and entry.keep_alive_deadline is None
                and not entry.loading
            )
        ]
        for model_name in empty:
            self._entries.pop(model_name, None)

    def _unload_entry_locked(self, model_name: str) -> dict[str, Any]:
        self._cancel_timer_locked(model_name)
        entry = self._entries.pop(model_name, None)
        was_resident = bool(entry is not None and entry.resident)
        unload = getattr(self.loader, "unload_model", None)
        if not callable(unload):
            released = self._zero_release()
            released["model_name"] = model_name
            if was_resident:
                self._record_model_eviction_locked()
            return released
        with self._load_lock:
            released = unload(model_name)
        if was_resident or any(released.get(name, 0) for name in self._zero_release()):
            self._record_model_eviction_locked()
        return released

    def _unload_all_loaded(self) -> dict[str, int]:
        released = self._zero_release()
        for model_name in sorted(self._loader_loaded_models_locked()):
            model_released = self._unload_entry_locked(model_name)
            for key in released:
                released[key] += model_released.get(key, 0)
        return released

    def _cancel_timer_locked(self, model_name: str) -> None:
        timer = self._timers.pop(model_name, None)
        if timer is not None:
            timer.cancel()

    def _loader_loaded_models_locked(self) -> dict[str, dict[str, int]]:
        loaded_models = getattr(self.loader, "loaded_models", None)
        if not callable(loaded_models):
            return {}
        with self._load_lock:
            loaded = loaded_models()
        return dict(loaded) if isinstance(loaded, Mapping) else {}

    def _zero_release(self) -> dict[str, int]:
        return {"models": 0, "tokenizers": 0, "pipelines": 0}

    def _record_model_load_locked(self) -> None:
        record = getattr(self.metrics, "record_model_load", None)
        if callable(record):
            record()

    def _record_model_eviction_locked(self, count: int = 1) -> None:
        record = getattr(self.metrics, "record_model_eviction", None)
        if callable(record):
            record(count)

    def _record_model_load_latency_locked(self, seconds: float) -> None:
        record = getattr(self.metrics, "record_model_load_latency", None)
        if callable(record):
            record(seconds)

    def _record_model_rejection_locked(self, count: int = 1) -> None:
        record = getattr(self.metrics, "record_model_rejection", None)
        if callable(record):
            record(count)

    def _sync_residency_metrics_locked(self) -> None:
        record = getattr(self.metrics, "record_model_residency", None)
        if callable(record):
            record(
                resident_count=self._resident_count_locked(),
                resident_bytes=self._resident_memory_bytes_locked(),
                pending_bytes=self._pending_memory_bytes_locked(),
            )
