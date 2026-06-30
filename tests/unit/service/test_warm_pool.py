"""Unit tests for service model warm-pool behavior."""

from __future__ import annotations

from typing import Any

from openmed.core.config import OpenMedConfig
from openmed.service import runtime as service_runtime
from openmed.service.warm_pool import WarmPool, parse_max_resident_models


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

    def create_pipeline(self, model_name: str, **kwargs: Any) -> object:
        self.create_pipeline_calls.append((model_name, dict(kwargs)))
        key = (
            model_name,
            kwargs.get("task"),
            kwargs.get("aggregation_strategy"),
            kwargs.get("use_fast_tokenizer"),
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
        )
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
        if force_reload or model_name not in self.models:
            self.models[model_name] = object()
        return self.models[model_name]

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def get_max_sequence_length(self, *_: Any, **__: Any) -> int:
        return 512

    def loaded_models(self) -> dict[str, dict[str, int]]:
        model_names = {key[0] for key in self.pipelines}
        model_names.update(self.models)
        return {
            model_name: {
                "models": int(model_name in self.models),
                "tokenizers": int(model_name in self.models),
                "pipelines": sum(1 for key in self.pipelines if key[0] == model_name),
            }
            for model_name in sorted(model_names)
        }

    def unload_model(self, model_name: str) -> dict[str, Any]:
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
        released = {
            "models": len(self.models),
            "tokenizers": len(self.models),
            "pipelines": len(self.pipelines),
        }
        self.models.clear()
        self.pipelines.clear()
        return released


def test_parse_max_resident_models_accepts_empty_unbounded_config() -> None:
    assert parse_max_resident_models(None) is None
    assert parse_max_resident_models("") is None
    assert parse_max_resident_models(" 2 ") == 2
    assert parse_max_resident_models(3) == 3


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


def test_runtime_reads_warm_pool_config_from_env(monkeypatch) -> None:
    monkeypatch.setattr(service_runtime, "ModelLoader", CountingLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_PRELOAD_MODELS", "model-a,model-b")
    monkeypatch.setenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", "2")
    monkeypatch.setenv("OPENMED_SERVICE_KEEP_ALIVE", "10s")
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)

    runtime = service_runtime.ServiceRuntime.from_env()
    pool = runtime.get_loader()

    assert runtime.preload_models == ("model-a", "model-b")
    assert runtime.max_resident_models == 2
    assert runtime.default_keep_alive_seconds == 10.0
    assert pool.warm_models == ("model-a", "model-b")
    assert pool.max_resident_models == 2


def test_runtime_service_config_initializes_warm_pool_limit() -> None:
    runtime = service_runtime.ServiceRuntime(
        profile="test",
        config=OpenMedConfig.from_profile("test"),
        preload_models=("model-a",),
        max_resident_models=1,
        _loader_factory=CountingLoader,
    )

    runtime.preload()

    pool = runtime.get_loader()
    assert pool.max_resident_models == 1
    assert runtime.loaded_models()["max_resident_models"] == 1
    assert runtime.get_model_loader().pipeline_creations == 1
