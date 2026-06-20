"""Runtime helpers for the OpenMed REST service."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from openmed.core.config import PROFILE_ENV_VAR, OpenMedConfig
from openmed.core.models import ModelLoader
from openmed.utils.validation import validate_model_name

from .keep_alive import parse_keep_alive

SERVICE_PRELOAD_ENV_VAR = "OPENMED_SERVICE_PRELOAD_MODELS"
SERVICE_KEEP_ALIVE_ENV_VAR = "OPENMED_SERVICE_KEEP_ALIVE"


def parse_preload_models(raw_value: Optional[str]) -> Tuple[str, ...]:
    """Parse and validate the preload-model environment variable."""
    if raw_value is None:
        return ()

    models = []
    seen = set()
    for item in raw_value.split(","):
        model_name = item.strip()
        if not model_name:
            continue

        validated = validate_model_name(model_name)
        if validated in seen:
            continue

        models.append(validated)
        seen.add(validated)

    return tuple(models)


@dataclass
class ServiceRuntime:
    """Shared runtime state for the REST service."""

    profile: str
    config: OpenMedConfig
    preload_models: Tuple[str, ...] = ()
    default_keep_alive_seconds: Optional[float] = None
    _loader_factory: Optional[Callable[[OpenMedConfig], ModelLoader]] = None
    _loader: Optional[ModelLoader] = None
    _loader_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _active_models: Dict[str, int] = field(default_factory=dict, repr=False)
    _keep_alive_timers: Dict[str, threading.Timer] = field(
        default_factory=dict, repr=False
    )
    _keep_alive_deadlines: Dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> "ServiceRuntime":
        """Create a runtime using the current process environment."""
        profile = os.getenv(PROFILE_ENV_VAR, "prod")
        config = OpenMedConfig.from_profile(profile)
        preload_models = parse_preload_models(os.getenv(SERVICE_PRELOAD_ENV_VAR))
        keep_alive = parse_keep_alive(os.getenv(SERVICE_KEEP_ALIVE_ENV_VAR))
        return cls(
            profile=profile,
            config=config,
            preload_models=preload_models,
            default_keep_alive_seconds=keep_alive,
            _loader_factory=ModelLoader,
        )

    def get_loader(self) -> ModelLoader:
        """Return the shared loader, creating it on first use."""
        if self._loader is None:
            with self._loader_lock:
                if self._loader is None:
                    factory = self._loader_factory or ModelLoader
                    self._loader = factory(self.config)
        return self._loader

    def preload(self) -> None:
        """Warm configured model pipelines during service startup."""
        if not self.preload_models:
            return

        loader = self.get_loader()
        for model_name in self.preload_models:
            loader.create_pipeline(
                model_name,
                task="token-classification",
                aggregation_strategy="simple",
                use_fast_tokenizer=True,
            )

    def run_model_request(
        self,
        model_name: str,
        keep_alive: Any,
        operation: Callable[[], Any],
    ) -> Any:
        """Run one model-backed operation and update idle-unload bookkeeping."""
        model_key = self.begin_model_request(model_name)
        try:
            return operation()
        finally:
            self.finish_model_request(model_key, keep_alive)

    def begin_model_request(self, model_name: str) -> str:
        """Mark a resolved model as active and cancel pending idle unload."""
        model_key = self._resolve_model_name(model_name)
        with self._loader_lock:
            timer = self._keep_alive_timers.pop(model_key, None)
            if timer is not None:
                timer.cancel()
            self._keep_alive_deadlines.pop(model_key, None)
            self._active_models[model_key] = self._active_models.get(model_key, 0) + 1
        return model_key

    def finish_model_request(self, model_key: str, keep_alive: Any) -> None:
        """Mark a model request as complete and schedule idle unloading."""
        keep_alive_seconds = self._resolve_keep_alive_seconds(keep_alive)
        should_schedule = False
        with self._loader_lock:
            active_requests = max(self._active_models.get(model_key, 0) - 1, 0)
            if active_requests:
                self._active_models[model_key] = active_requests
            else:
                self._active_models.pop(model_key, None)
                should_schedule = keep_alive_seconds is not None

        if should_schedule:
            self._schedule_idle_unload(model_key, keep_alive_seconds or 0.0)

    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload one inactive model from the shared loader cache."""
        model_key = self._resolve_model_name(model_name)
        with self._loader_lock:
            active_requests = self._active_models.get(model_key, 0)
            if active_requests:
                return {
                    "unloaded": False,
                    "model_name": model_key,
                    "active_requests": active_requests,
                    "released": {"models": 0, "tokenizers": 0, "pipelines": 0},
                }
            self._cancel_idle_timer_locked(model_key)
            released = self._unload_model_cache(model_key)
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

    def unload_all_models(self) -> Dict[str, Any]:
        """Unload every inactive model from the shared loader cache."""
        with self._loader_lock:
            active_models = {
                model_name: count
                for model_name, count in self._active_models.items()
                if count > 0
            }
            for model_name in list(self._keep_alive_timers):
                self._cancel_idle_timer_locked(model_name)

            loader = self._loader
            loaded = (
                loader.loaded_models()
                if loader is not None and hasattr(loader, "loaded_models")
                else {}
            )

            if loader is None:
                released = {"models": 0, "tokenizers": 0, "pipelines": 0}
            elif not active_models:
                released = loader.unload_all_models()
            else:
                released = {"models": 0, "tokenizers": 0, "pipelines": 0}
                for model_name in loaded:
                    if model_name in active_models:
                        continue
                    model_released = self._unload_model_cache(model_name)
                    for key in released:
                        released[key] += model_released.get(key, 0)

            return {
                "unloaded": any(released.values()),
                "released": released,
                "active_models": active_models,
            }

    def loaded_models(self) -> Dict[str, Any]:
        """Return cache and keep-alive status for the service runtime."""
        with self._loader_lock:
            loader = self._loader
            loaded = (
                loader.loaded_models()
                if loader is not None and hasattr(loader, "loaded_models")
                else {}
            )
            now = time.monotonic()
            models = {}
            for model_name, cache_state in loaded.items():
                deadline = self._keep_alive_deadlines.get(model_name)
                remaining = None if deadline is None else max(deadline - now, 0.0)
                models[model_name] = {
                    **cache_state,
                    "active_requests": self._active_models.get(model_name, 0),
                    "keep_alive_seconds_remaining": remaining,
                }

            active_only = set(self._active_models) - set(models)
            for model_name in active_only:
                models[model_name] = {
                    "models": 0,
                    "tokenizers": 0,
                    "pipelines": 0,
                    "active_requests": self._active_models[model_name],
                    "keep_alive_seconds_remaining": None,
                }

            return {
                "default_keep_alive_seconds": self.default_keep_alive_seconds,
                "models": models,
            }

    def _resolve_keep_alive_seconds(self, keep_alive: Any) -> Optional[float]:
        if keep_alive is None:
            return self.default_keep_alive_seconds
        return parse_keep_alive(keep_alive)

    def _resolve_model_name(self, model_name: str) -> str:
        validated = validate_model_name(model_name)
        loader = self.get_loader()
        resolver = getattr(loader, "resolve_model_name", None)
        if callable(resolver):
            return resolver(validated)
        return validated

    def _schedule_idle_unload(self, model_key: str, keep_alive_seconds: float) -> None:
        if keep_alive_seconds <= 0:
            self._unload_model_if_idle(model_key)
            return

        timer = threading.Timer(
            keep_alive_seconds,
            self._unload_model_if_idle,
            args=(model_key,),
        )
        timer.daemon = True
        with self._loader_lock:
            self._cancel_idle_timer_locked(model_key)
            self._keep_alive_timers[model_key] = timer
            self._keep_alive_deadlines[model_key] = (
                time.monotonic() + keep_alive_seconds
            )
        timer.start()

    def _unload_model_if_idle(self, model_key: str) -> None:
        with self._loader_lock:
            self._keep_alive_timers.pop(model_key, None)
            self._keep_alive_deadlines.pop(model_key, None)
            if self._active_models.get(model_key, 0):
                return
        self._unload_model_cache(model_key)

    def _unload_model_cache(self, model_key: str) -> Dict[str, Any]:
        loader = self.get_loader()
        unload = getattr(loader, "unload_model", None)
        if not callable(unload):
            return {
                "model_name": model_key,
                "models": 0,
                "tokenizers": 0,
                "pipelines": 0,
            }
        return unload(model_key)

    def _cancel_idle_timer_locked(self, model_key: str) -> None:
        timer = self._keep_alive_timers.pop(model_key, None)
        if timer is not None:
            timer.cancel()
        self._keep_alive_deadlines.pop(model_key, None)
