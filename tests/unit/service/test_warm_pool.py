"""Unit tests for service model warm-pool behavior."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from openmed.core.config import OpenMedConfig
from openmed.service import runtime as service_runtime
from openmed.service.warm_pool import (
    WarmPool,
    WarmPoolBackpressureError,
    parse_default_model_footprint_bytes,
    parse_max_resident_models,
    parse_memory_admission_wait_seconds,
    parse_model_memory_budget_bytes,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CountingLoader:
    """Loader double that records cold pipeline creations and unloads."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.pipelines: dict[tuple[Any, ...], object] = {}
        self.models: dict[str, object] = {}
        self.create_pipeline_calls: list[tuple[str, dict[str, Any]]] = []
        self.pipeline_creations = 0
        self.unloaded_models: list[str] = []
        self.lock = threading.Lock()

    def create_pipeline(self, model_name: str, **kwargs: Any) -> object:
        with self.lock:
            self.create_pipeline_calls.append((model_name, dict(kwargs)))
        key = (
            model_name,
            kwargs.get("task"),
            kwargs.get("aggregation_strategy"),
            kwargs.get("use_fast_tokenizer"),
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
        )
        with self.lock:
            if key not in self.pipelines:
                self.pipeline_creations += 1
                self.pipelines[key] = object()
            return self.pipelines[key]

    def load_model(
        self,
        model_name: str,
        force_reload: bool = False,
        **_: Any,
    ) -> object:
        with self.lock:
            if force_reload or model_name not in self.models:
                self.models[model_name] = object()
            return self.models[model_name]

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def get_max_sequence_length(self, *_: Any, **__: Any) -> int:
        return 512

    def loaded_models(self) -> dict[str, dict[str, int]]:
        with self.lock:
            model_names = {key[0] for key in self.pipelines}
            model_names.update(self.models)
            return {
                model_name: {
                    "models": int(model_name in self.models),
                    "tokenizers": int(model_name in self.models),
                    "pipelines": sum(
                        1 for key in self.pipelines if key[0] == model_name
                    ),
                }
                for model_name in sorted(model_names)
            }

    def unload_model(self, model_name: str) -> dict[str, Any]:
        with self.lock:
            pipeline_keys = [key for key in self.pipelines if key[0] == model_name]
            for key in pipeline_keys:
                self.pipelines.pop(key, None)
            removed_model = int(self.models.pop(model_name, None) is not None)
            self.unloaded_models.append(model_name)
        return {
            "model_name": model_name,
            "models": removed_model,
            "tokenizers": removed_model,
            "pipelines": len(pipeline_keys),
        }

    def unload_all_models(self) -> dict[str, int]:
        with self.lock:
            released = {
                "models": len(self.models),
                "tokenizers": len(self.models),
                "pipelines": len(self.pipelines),
            }
            self.models.clear()
            self.pipelines.clear()
            return released


class TrackingMetrics:
    def __init__(self) -> None:
        self.loads = 0
        self.evictions = 0
        self.rejections = 0
        self.load_latencies: list[float] = []
        self.max_accounted_bytes = 0
        self.residency_samples: list[tuple[int, int, int]] = []
        self.lock = threading.Lock()

    def record_model_load(self, count: int = 1) -> None:
        with self.lock:
            self.loads += count

    def record_model_eviction(self, count: int = 1) -> None:
        with self.lock:
            self.evictions += count

    def record_model_rejection(self, count: int = 1) -> None:
        with self.lock:
            self.rejections += count

    def record_model_load_latency(self, seconds: float) -> None:
        with self.lock:
            self.load_latencies.append(seconds)

    def record_model_residency(
        self,
        *,
        resident_count: int,
        resident_bytes: int,
        pending_bytes: int,
    ) -> None:
        with self.lock:
            self.residency_samples.append(
                (resident_count, resident_bytes, pending_bytes)
            )
            self.max_accounted_bytes = max(
                self.max_accounted_bytes,
                resident_bytes + pending_bytes,
            )


def test_parse_max_resident_models_accepts_empty_unbounded_config() -> None:
    assert parse_max_resident_models(None) is None
    assert parse_max_resident_models("") is None
    assert parse_max_resident_models(" 2 ") == 2
    assert parse_max_resident_models(3) == 3


def test_parse_memory_budget_config_accepts_units_and_wait_bounds() -> None:
    assert parse_model_memory_budget_bytes(None) is None
    assert parse_model_memory_budget_bytes("") is None
    assert parse_model_memory_budget_bytes("64MiB") == 64 * 1024 * 1024
    assert parse_model_memory_budget_bytes("1.5GB") == 1_500_000_000
    assert parse_default_model_footprint_bytes("128kb") == 128_000
    assert parse_memory_admission_wait_seconds("0.25") == 0.25

    with pytest.raises(ValueError, match="greater than 0"):
        parse_model_memory_budget_bytes("0")
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        parse_memory_admission_wait_seconds("-1")


def test_preload_and_hot_request_reuse_resident_handle_without_reload() -> None:
    loader = CountingLoader()
    clock = FakeClock()
    pool = WarmPool(
        lambda: loader,
        warm_models=("model-a",),
        max_resident_models=2,
        clock=clock,
    )

    pool.preload()
    preloaded_handle = next(iter(loader.pipelines.values()))

    model_key = pool.begin_request("model-a")
    handle = pool.create_pipeline("model-a", aggregation_strategy="simple")
    pool.finish_request(model_key, "forever")

    assert handle is preloaded_handle
    assert loader.pipeline_creations == 1
    assert len(loader.create_pipeline_calls) == 1
    assert pool.resident_model_names() == ("model-a",)


def test_exceeding_max_resident_count_evicts_lru_model() -> None:
    loader = CountingLoader()
    clock = FakeClock()
    pool = WarmPool(lambda: loader, max_resident_models=2, clock=clock)

    pool.create_pipeline("model-a", aggregation_strategy="simple")
    clock.advance(1.0)
    pool.create_pipeline("model-b", aggregation_strategy="simple")
    clock.advance(1.0)
    pool.create_pipeline("model-a", aggregation_strategy="simple")
    clock.advance(1.0)
    pool.create_pipeline("model-c", aggregation_strategy="simple")

    assert loader.unloaded_models == ["model-b"]
    assert pool.resident_model_names() == ("model-a", "model-c")


def test_expired_keep_alive_entry_is_dropped() -> None:
    loader = CountingLoader()
    clock = FakeClock()
    pool = WarmPool(lambda: loader, max_resident_models=2, clock=clock)

    model_key = pool.begin_request("model-a")
    pool.create_pipeline("model-a", aggregation_strategy="simple")
    pool.finish_request(model_key, "5s")
    assert pool.resident_model_names() == ("model-a",)

    clock.advance(5.1)
    expired = pool.drop_expired()

    assert expired == ["model-a"]
    assert loader.unloaded_models == ["model-a"]
    assert pool.resident_model_names() == ()


def test_keep_alive_applies_to_model_loaded_inside_request() -> None:
    loader = CountingLoader()
    pool = WarmPool(lambda: loader, max_resident_models=2)

    model_key = pool.begin_request("request-model")
    pool.create_pipeline("effective-model", aggregation_strategy="simple")
    pool.finish_request(model_key, 0)

    assert loader.unloaded_models == ["effective-model"]
    assert pool.resident_model_names() == ()


def test_memory_budget_stress_keeps_peak_accounting_under_ceiling() -> None:
    loader = CountingLoader()
    metrics = TrackingMetrics()
    pool = WarmPool(
        lambda: loader,
        memory_budget_bytes=120,
        default_model_footprint_bytes=60,
        memory_admission_wait_seconds=1.0,
        metrics=metrics,
    )
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(model_name: str) -> None:
        model_key = pool.begin_request(model_name)
        try:
            barrier.wait(timeout=1)
            pool.create_pipeline(model_name, aggregation_strategy="simple")
            time.sleep(0.02)
        except BaseException as exc:
            with errors_lock:
                errors.append(exc)
        finally:
            pool.finish_request(model_key, "forever")

    threads = [
        threading.Thread(target=worker, args=(model_name,))
        for model_name in ("model-a", "model-b", "model-c", "model-d")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert [thread.is_alive() for thread in threads] == [False] * len(threads)
    assert errors == []
    assert metrics.max_accounted_bytes <= 120
    assert metrics.loads == 4
    assert metrics.evictions >= 2
    assert metrics.rejections == 0
    assert pool.loaded_models()["resident_memory_bytes"] <= 120


def test_active_model_is_not_evicted_and_saturated_admission_is_bounded() -> None:
    loader = CountingLoader()
    metrics = TrackingMetrics()
    pool = WarmPool(
        lambda: loader,
        memory_budget_bytes=60,
        default_model_footprint_bytes=60,
        memory_admission_wait_seconds=0.02,
        metrics=metrics,
    )

    active_key = pool.begin_request("model-a")
    pool.create_pipeline("model-a", aggregation_strategy="simple")

    blocked_key = pool.begin_request("model-b")
    started_at = time.monotonic()
    with pytest.raises(WarmPoolBackpressureError):
        try:
            pool.create_pipeline("model-b", aggregation_strategy="simple")
        finally:
            pool.finish_request(blocked_key, "forever")
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    assert loader.unloaded_models == []
    assert metrics.rejections == 1

    pool.finish_request(active_key, "forever")
    pool.create_pipeline("model-b", aggregation_strategy="simple")

    assert loader.unloaded_models == ["model-a"]
    assert pool.resident_model_names() == ("model-b",)


def test_hot_resident_request_does_not_wait_for_background_cold_load() -> None:
    loader = CountingLoader()
    load_started = threading.Event()
    release_load = threading.Event()
    original_create_pipeline = loader.create_pipeline

    def blocking_create_pipeline(model_name: str, **kwargs: Any) -> object:
        if model_name == "model-cold":
            load_started.set()
            assert release_load.wait(timeout=1)
        return original_create_pipeline(model_name, **kwargs)

    loader.create_pipeline = blocking_create_pipeline  # type: ignore[method-assign]
    pool = WarmPool(
        lambda: loader,
        memory_budget_bytes=120,
        default_model_footprint_bytes=60,
        memory_admission_wait_seconds=1.0,
    )

    hot_handle = pool.create_pipeline("model-hot", aggregation_strategy="simple")
    cold_errors: list[BaseException] = []

    def load_cold_model() -> None:
        try:
            pool.create_pipeline("model-cold", aggregation_strategy="simple")
        except BaseException as exc:
            cold_errors.append(exc)

    thread = threading.Thread(target=load_cold_model)
    thread.start()
    assert load_started.wait(timeout=1)

    started_at = time.monotonic()
    returned_handle = pool.create_pipeline("model-hot", aggregation_strategy="simple")
    elapsed = time.monotonic() - started_at

    release_load.set()
    thread.join(timeout=1)

    assert returned_handle is hot_handle
    assert elapsed < 0.05
    assert cold_errors == []
    assert not thread.is_alive()


def test_memory_scheduler_metrics_include_residency_latency_and_rejections() -> None:
    loader = CountingLoader()
    metrics = TrackingMetrics()
    pool = WarmPool(
        lambda: loader,
        memory_budget_bytes=60,
        default_model_footprint_bytes=60,
        memory_admission_wait_seconds=0,
        metrics=metrics,
    )

    active_key = pool.begin_request("model-a")
    pool.create_pipeline("model-a", aggregation_strategy="simple")
    rejected_key = pool.begin_request("model-b")
    with pytest.raises(WarmPoolBackpressureError):
        try:
            pool.create_pipeline("model-b", aggregation_strategy="simple")
        finally:
            pool.finish_request(rejected_key, "forever")
    pool.finish_request(active_key, "forever")

    assert metrics.loads == 1
    assert metrics.rejections == 1
    assert len(metrics.load_latencies) == 1
    assert metrics.residency_samples
    assert metrics.max_accounted_bytes == 60


def test_runtime_reads_warm_pool_config_from_env(monkeypatch) -> None:
    monkeypatch.setattr(service_runtime, "ModelLoader", CountingLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_PRELOAD_MODELS", "model-a,model-b")
    monkeypatch.setenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", "2")
    monkeypatch.setenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", "1MiB")
    monkeypatch.setenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", "256KiB")
    monkeypatch.setenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", "0.2")
    monkeypatch.setenv("OPENMED_SERVICE_KEEP_ALIVE", "10s")
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)

    runtime = service_runtime.ServiceRuntime.from_env()
    pool = runtime.get_loader()

    assert runtime.preload_models == ("model-a", "model-b")
    assert runtime.max_resident_models == 2
    assert runtime.model_memory_budget_bytes == 1024 * 1024
    assert runtime.default_model_footprint_bytes == 256 * 1024
    assert runtime.model_admission_wait_seconds == 0.2
    assert runtime.default_keep_alive_seconds == 10.0
    assert pool.warm_models == ("model-a", "model-b")
    assert pool.max_resident_models == 2
    assert pool.memory_budget_bytes == 1024 * 1024
    assert pool.default_model_footprint_bytes == 256 * 1024
    assert pool.memory_admission_wait_seconds == 0.2


def test_runtime_service_config_initializes_warm_pool_limit() -> None:
    runtime = service_runtime.ServiceRuntime(
        profile="test",
        config=OpenMedConfig.from_profile("test"),
        preload_models=("model-a",),
        max_resident_models=1,
        model_memory_budget_bytes=1024,
        default_model_footprint_bytes=512,
        _loader_factory=CountingLoader,
    )

    runtime.preload()

    pool = runtime.get_loader()
    assert pool.max_resident_models == 1
    assert pool.memory_budget_bytes == 1024
    assert pool.default_model_footprint_bytes == 512
    assert runtime.loaded_models()["max_resident_models"] == 1
    assert runtime.loaded_models()["memory_budget_bytes"] == 1024
    assert runtime.get_model_loader().pipeline_creations == 1
