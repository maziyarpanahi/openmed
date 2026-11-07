"""Fair admission shared by REST and GraphQL on one service runtime."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from typing import Any

import httpx
import pytest

import openmed
from openmed.processing.outputs import PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.admission import AdmissionStats
from openmed.service.app import create_app
from openmed.service.auth import hash_api_key

pytestmark = pytest.mark.integration

LOOPBACK_BASE_URL = "http://127.0.0.1"
# Bounds waits on worker threads and queue transitions; passing runs return
# as soon as the awaited state is reached.
_WAIT_SECONDS = 5
_API_KEYS = {"noisy": "synthetic-noisy-key", "quiet": "synthetic-quiet-key"}
_PRINCIPALS = {
    "noisy": "SYNTHETIC_NOISY_PRINCIPAL",
    "quiet": "SYNTHETIC_QUIET_PRINCIPAL",
}
_ANALYZE_QUERY_FIELD = "f{index}: analyze(input: $i{index}) {{ modelName }}"
_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES",
    "OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES",
    "OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
    "OPENMED_SERVICE_RETRY_MAX_ATTEMPTS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_AUTH_JWKS_FILE",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_MTLS_ENABLED",
    "OPENMED_SERVICE_ADMISSION_MAX_CONCURRENCY",
    "OPENMED_SERVICE_ADMISSION_MAX_QUEUE",
    "OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_CONCURRENCY",
    "OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_QUEUE",
    "OPENMED_SERVICE_ADMISSION_NAMESPACE_MAX_CONCURRENCY",
    "OPENMED_SERVICE_ADMISSION_NAMESPACE_MAX_QUEUE",
    "OPENMED_SERVICE_ADMISSION_NAMESPACE_CLAIM",
    "OPENMED_SERVICE_ADMISSION_QUEUE_TIMEOUT_SECONDS",
    "OPENMED_SERVICE_ADMISSION_RETRY_AFTER_SECONDS",
)


class FakeLoader:
    """Model loader double; the analyze function below does the work."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, Any]:
        return {}


class GatedModel:
    """Deterministic fake work that blocks selected synthetic inputs."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self._gates: dict[str, threading.Event] = {}
        self._started: dict[str, threading.Event] = {}

    def hold(self, *texts: str) -> None:
        for text in texts:
            self._gates[text] = threading.Event()
            self._started[text] = threading.Event()

    async def started(self, text: str) -> None:
        assert await asyncio.to_thread(self._started[text].wait, _WAIT_SECONDS)

    def release(self, text: str) -> None:
        self._gates[text].set()

    def release_all(self) -> None:
        for gate in self._gates.values():
            gate.set()

    def __call__(self, text: str, **_: Any) -> PredictionResult:
        with self._lock:
            self.calls.append(text)
        if text in self._started:
            self._started[text].set()
            assert self._gates[text].wait(_WAIT_SECONDS)
        return PredictionResult(
            text=text,
            entities=[],
            model_name="disease_detection_superclinical",
            timestamp=datetime.now().isoformat(),
            processing_time=0.01,
        )


@pytest.fixture(autouse=True)
def service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for name in _SERVICE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "key_id": name,
                    "key_hash": hash_api_key(_API_KEYS[name]),
                    "principal": _PRINCIPALS[name],
                    "scopes": ["analyze:*"],
                }
                for name in ("noisy", "quiet")
            ]
        ),
    )


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> GatedModel:
    gated = GatedModel()
    monkeypatch.setattr(openmed, "analyze_text", gated)
    return gated


def _headers(who: str) -> dict[str, str]:
    return {"X-API-Key": _API_KEYS[who]}


async def _rest(client: httpx.AsyncClient, who: str, text: str) -> httpx.Response:
    return await client.post("/analyze", json={"text": text}, headers=_headers(who))


async def _graphql(client: httpx.AsyncClient, who: str, *texts: str) -> httpx.Response:
    variables = ", ".join(f"$i{index}: AnalyzeInput!" for index in range(len(texts)))
    fields = " ".join(
        _ANALYZE_QUERY_FIELD.format(index=index) for index in range(len(texts))
    )
    return await client.post(
        "/graphql",
        json={
            "query": f"query({variables}) {{ {fields} }}",
            "variables": {
                f"i{index}": {"text": text} for index, text in enumerate(texts)
            },
        },
        headers=_headers(who),
    )


def _assert_rest_rejection(
    response: httpx.Response, status: int, code: str, reason: str
) -> None:
    assert response.status_code == status
    assert response.headers["Retry-After"] == "1"
    error = response.json()["error"]
    assert error["code"] == code
    assert error["details"] == {"reason": reason, "retry_after_seconds": 1.0}


def _assert_graphql_rejection(
    response: httpx.Response, status: int, reason: str
) -> None:
    assert response.status_code == status
    assert response.headers["Retry-After"] == "1"
    assert response.json() == {
        "data": None,
        "errors": [
            {
                "message": "Request admission limit reached; retry later.",
                "extensions": {
                    "code": "OPENMED_ADMISSION_REJECTED",
                    "reason": reason,
                },
            }
        ],
    }


async def _until(predicate: Any) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), _WAIT_SECONDS)


def _run(app: Any, scenario: Any) -> None:
    async def main() -> None:
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                await scenario(client, app.state.runtime.admission_controller)

    asyncio.run(main())


def test_admission_is_off_unless_configured(model: GatedModel) -> None:
    app = create_app()

    async def scenario(client: httpx.AsyncClient, controller: Any) -> None:
        assert controller.enabled is False
        assert (await _rest(client, "noisy", "plain")).status_code == 200
        assert controller.snapshot() == AdmissionStats(0, 0, 0)

    _run(app, scenario)
    assert model.calls == ["plain"]


def test_rest_and_graphql_draw_on_one_principal_budget(
    monkeypatch: pytest.MonkeyPatch, model: GatedModel
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_CONCURRENCY", "1")
    model.hold("held-rest")
    app = create_app()

    async def scenario(client: httpx.AsyncClient, controller: Any) -> None:
        held = asyncio.create_task(_rest(client, "noisy", "held-rest"))
        try:
            await model.started("held-rest")

            _assert_graphql_rejection(
                await _graphql(client, "noisy", "noisy-graphql"),
                429,
                "principal_limit",
            )
            _assert_rest_rejection(
                await _rest(client, "noisy", "noisy-rest"),
                429,
                "admission_limited",
                "principal_limit",
            )
            quiet_graphql = await _graphql(client, "quiet", "quiet-graphql")
            assert quiet_graphql.status_code == 200
            assert "errors" not in quiet_graphql.json()
            assert (await _rest(client, "quiet", "quiet-rest")).status_code == 200
            catalog = await client.post(
                "/graphql",
                json={"query": "{ entityTypes { label } }"},
                headers=_headers("noisy"),
            )
            assert catalog.status_code == 200
            assert "errors" not in catalog.json()
        finally:
            model.release_all()
        assert (await held).status_code == 200
        await _until(lambda: controller.snapshot() == AdmissionStats(0, 0, 0))

        recovered = await _graphql(client, "noisy", "noisy-after")
        assert recovered.status_code == 200
        assert "errors" not in recovered.json()

        metrics = (await client.get("/metrics")).text
        assert (
            "openmed_service_principal_admission_rejected_total"
            '{reason="principal_limit",transport="graphql"} 1'
        ) in metrics
        assert (
            "openmed_service_principal_admission_rejected_total"
            '{reason="principal_limit",transport="rest"} 1'
        ) in metrics
        assert "SYNTHETIC" not in metrics

    _run(app, scenario)
    assert model.calls == ["held-rest", "quiet-graphql", "quiet-rest", "noisy-after"]


def test_each_graphql_field_takes_a_unit_of_the_rest_budget(
    monkeypatch: pytest.MonkeyPatch, model: GatedModel
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_CONCURRENCY", "2")
    model.hold("held-a", "held-b")
    app = create_app()

    async def scenario(client: httpx.AsyncClient, controller: Any) -> None:
        held = asyncio.create_task(_graphql(client, "noisy", "held-a", "held-b"))
        try:
            await model.started("held-a")
            await model.started("held-b")
            assert controller.snapshot().in_flight == 2
            _assert_rest_rejection(
                await _rest(client, "noisy", "over-share"),
                429,
                "admission_limited",
                "principal_limit",
            )
            assert (await _rest(client, "quiet", "quiet-rest")).status_code == 200
        finally:
            model.release_all()
        response = await held
        assert response.status_code == 200
        assert set(response.json()["data"]) == {"f0", "f1"}
        await _until(lambda: controller.snapshot() == AdmissionStats(0, 0, 0))

        # Batching more model fields than the share into one document is
        # rejected as a whole instead of slipping past the ceiling.
        _assert_graphql_rejection(
            await _graphql(client, "noisy", "batch-0", "batch-1", "batch-2"),
            429,
            "principal_limit",
        )
        await _until(lambda: controller.snapshot() == AdmissionStats(0, 0, 0))

    _run(app, scenario)
    assert "over-share" not in model.calls
    assert "batch-2" not in model.calls


def test_global_cap_returns_503_on_every_transport(
    monkeypatch: pytest.MonkeyPatch, model: GatedModel
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_ADMISSION_MAX_CONCURRENCY", "1")
    model.hold("held")
    app = create_app()

    async def scenario(client: httpx.AsyncClient, controller: Any) -> None:
        held = asyncio.create_task(_rest(client, "noisy", "held"))
        try:
            await model.started("held")
            _assert_rest_rejection(
                await _rest(client, "quiet", "quiet-rest"),
                503,
                "admission_saturated",
                "global_limit",
            )
            _assert_graphql_rejection(
                await _graphql(client, "quiet", "quiet-graphql"),
                503,
                "global_limit",
            )
        finally:
            model.release_all()
        assert (await held).status_code == 200
        await _until(lambda: controller.snapshot() == AdmissionStats(0, 0, 0))

    _run(app, scenario)
    assert model.calls == ["held"]


def test_fair_queue_serves_a_quiet_principal_before_a_noisy_backlog(
    monkeypatch: pytest.MonkeyPatch, model: GatedModel
) -> None:
    for name, value in (
        ("OPENMED_SERVICE_ADMISSION_MAX_CONCURRENCY", "1"),
        ("OPENMED_SERVICE_ADMISSION_MAX_QUEUE", "4"),
        ("OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_CONCURRENCY", "1"),
        ("OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_QUEUE", "3"),
        ("OPENMED_SERVICE_ADMISSION_QUEUE_TIMEOUT_SECONDS", "30"),
    ):
        monkeypatch.setenv(name, value)
    model.hold("noisy-1", "noisy-2", "noisy-3", "quiet-1")
    app = create_app()

    async def scenario(client: httpx.AsyncClient, controller: Any) -> None:
        requests = [asyncio.create_task(_rest(client, "noisy", "noisy-1"))]
        try:
            await model.started("noisy-1")
            # Arrivals across both transports: two noisy requests queue before
            # the quiet one.
            for queued, request in enumerate(
                (
                    _rest(client, "noisy", "noisy-2"),
                    _graphql(client, "noisy", "noisy-3"),
                    _graphql(client, "quiet", "quiet-1"),
                ),
                start=1,
            ):
                requests.append(asyncio.create_task(request))
                await _until(lambda q=queued: controller.snapshot().queued == q)

            for current, following in (
                ("noisy-1", "noisy-2"),
                ("noisy-2", "quiet-1"),
                ("quiet-1", "noisy-3"),
            ):
                model.release(current)
                await model.started(following)
        finally:
            model.release_all()
        responses = await asyncio.gather(*requests)
        assert [response.status_code for response in responses] == [200] * 4
        await _until(lambda: controller.snapshot() == AdmissionStats(0, 0, 0))

    _run(app, scenario)
    # FIFO would run noisy-3 before quiet-1.
    assert model.calls == ["noisy-1", "noisy-2", "quiet-1", "noisy-3"]


def test_cancelled_requests_release_permits_once_work_settles(
    monkeypatch: pytest.MonkeyPatch, model: GatedModel
) -> None:
    for name, value in (
        ("OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_CONCURRENCY", "1"),
        ("OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_QUEUE", "1"),
        ("OPENMED_SERVICE_ADMISSION_QUEUE_TIMEOUT_SECONDS", "30"),
    ):
        monkeypatch.setenv(name, value)
    model.hold("held")
    app = create_app()

    async def scenario(client: httpx.AsyncClient, controller: Any) -> None:
        held = asyncio.create_task(_rest(client, "noisy", "held"))
        try:
            await model.started("held")
            queued = asyncio.create_task(_rest(client, "noisy", "queued"))
            await _until(lambda: controller.snapshot().queued == 1)

            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            await _until(lambda: controller.snapshot() == AdmissionStats(1, 0, 1))

            # Cancelling the HTTP wait leaves the model running; its unit stays
            # taken until that work settles.
            held.cancel()
            with pytest.raises(asyncio.CancelledError):
                await held
            assert controller.snapshot() == AdmissionStats(1, 0, 1)
        finally:
            model.release_all()
        await _until(lambda: controller.snapshot() == AdmissionStats(0, 0, 0))
        assert (await _rest(client, "noisy", "after")).status_code == 200
        await _until(lambda: controller.snapshot() == AdmissionStats(0, 0, 0))

    _run(app, scenario)
    assert model.calls == ["held", "after"]
