"""Model warm-pool support for the OpenMed REST service."""

from __future__ import annotations

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


@dataclass
class WarmPoolEntry:
    """Resident model bookkeeping tracked by :class:`WarmPool`."""

    model_name: str
    last_used: float
    active_requests: int = 0
    keep_alive_deadline: Optional[float] = None
    handles: dict[Tuple[Any, ...], Any] = field(default_factory=dict)

    @property
    def resident(self) -> bool:
        """Return whether this entry owns any cached model handles."""
        return bool(self.handles)


@dataclass
class WarmPool:
    """Bounded model warm-pool that proxies a shared model loader.

    The pool caches exact pipeline handles by model and pipeline options while
    applying model-level keep-alive and least-recently-used eviction policy.
    """

    loader_provider: Callable[[], Any]
    warm_models: Tuple[str, ...] = ()
    max_resident_models: Optional[int] = None
    default_keep_alive_seconds: Optional[float] = None
    clock: Callable[[], float] = time.monotonic
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _entries: dict[str, WarmPoolEntry] = field(default_factory=dict, repr=False)
    _timers: dict[str, threading.Timer] = field(default_factory=dict, repr=False)
    _local: threading.local = field(default_factory=threading.local, repr=False)

    def __post_init__(self) -> None:
        self.max_resident_models = parse_max_resident_models(self.max_resident_models)
        self.warm_models = tuple(self.warm_models)

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

        with self._lock:
            now = self.clock()
            self._drop_expired_locked(now)
            entry = self._entry_for_model_locked(model_key, now)
            self._activate_current_request_model_locked(model_key, now)
            if cache_key in entry.handles:
                entry.last_used = now
                return entry.handles[cache_key]

            handle = self.loader.create_pipeline(
                model_key,
                task=task,
                aggregation_strategy=aggregation_strategy,
                use_fast_tokenizer=use_fast_tokenizer,
                **kwargs,
            )
            entry.handles[cache_key] = handle
            entry.last_used = now
            self._evict_over_capacity_locked()
            return handle

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

        with self._lock:
            now = self.clock()
            self._drop_expired_locked(now)
            entry = self._entry_for_model_locked(model_key, now)
            self._activate_current_request_model_locked(model_key, now)
            if not force_reload and cache_key in entry.handles:
                entry.last_used = now
                return entry.handles[cache_key]

            handle = self.loader.load_model(
                model_key,
                force_reload=force_reload,
                **kwargs,
            )
            entry.handles[cache_key] = handle
            entry.last_used = now
            self._evict_over_capacity_locked()
            return handle

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
                }

            for model_name, entry in self._entries.items():
                if model_name in models:
                    continue
                if not entry.resident and entry.active_requests <= 0:
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
                }

            return {
                "default_keep_alive_seconds": self.default_keep_alive_seconds,
                "max_resident_models": self.max_resident_models,
                "warm_models": list(self.warm_models),
                "models": models,
            }

    def unload_model(self, model_name: str) -> dict[str, Any]:
        """Unload one inactive model from the pool and wrapped loader."""
        model_key = self.resolve_model_name(model_name)
        with self._lock:
            entry = self._entries.get(model_key)
            active_requests = 0 if entry is None else entry.active_requests
            if active_requests:
                return {
                    "unloaded": False,
                    "model_name": model_key,
                    "active_requests": active_requests,
                    "released": self._zero_release(),
                }

            released = self._unload_entry_locked(model_key)
            return {
                "unloaded": any(
                    released[name] for name in ("models", "tokenizers", "pipelines")
                ),
                "model_name": released.get("model_name", model_key),
                "active_requests": 0,
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
                if entry.active_requests > 0
            }
            if not active_models:
                for model_name in list(self._timers):
                    self._cancel_timer_locked(model_name)
                self._entries.clear()
                unload_all = getattr(self.loader, "unload_all_models", None)
                released = (
                    unload_all() if callable(unload_all) else self._unload_all_loaded()
                )
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
        return expired

    def _evict_over_capacity_locked(self) -> list[str]:
        if self.max_resident_models is None:
            return []

        evicted: list[str] = []
        while self._resident_count_locked() > self.max_resident_models:
            candidates = [
                entry
                for entry in self._entries.values()
                if entry.resident and entry.active_requests <= 0
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda entry: entry.last_used)
            evicted.append(victim.model_name)
            self._unload_entry_locked(victim.model_name)
        return evicted

    def _resident_count_locked(self) -> int:
        return sum(1 for entry in self._entries.values() if entry.resident)

    def _drop_empty_idle_entries_locked(self) -> None:
        empty = [
            model_name
            for model_name, entry in self._entries.items()
            if (
                not entry.resident
                and entry.active_requests <= 0
                and entry.keep_alive_deadline is None
            )
        ]
        for model_name in empty:
            self._entries.pop(model_name, None)

    def _unload_entry_locked(self, model_name: str) -> dict[str, Any]:
        self._cancel_timer_locked(model_name)
        self._entries.pop(model_name, None)
        unload = getattr(self.loader, "unload_model", None)
        if not callable(unload):
            released = self._zero_release()
            released["model_name"] = model_name
            return released
        return unload(model_name)

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
        loaded = loaded_models()
        return dict(loaded) if isinstance(loaded, Mapping) else {}

    def _zero_release(self) -> dict[str, int]:
        return {"models": 0, "tokenizers": 0, "pipelines": 0}
