"""Exercise opt-in local comparisons through the regular REST entry point."""

import asyncio
from threading import Event

import pytest
from fastapi.testclient import TestClient
from test_api import FakeLoader

import openmed
from openmed.service import runtime as rt
from openmed.service.app import create_app

ACTIVE = "synthetic/active"
CANDIDATE = "synthetic/candidate"
NOTE = "SYNTHETIC-SHADOW-INPUT"


def configure(monkeypatch, rate):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    monkeypatch.setenv("OPENMED_SERVICE_SERVED_MODELS", f"{ACTIVE},{CANDIDATE}")
    monkeypatch.setenv("OPENMED_SERVICE_SHADOW_MODEL", CANDIDATE)
    monkeypatch.setenv("OPENMED_SERVICE_SHADOW_SAMPLE_RATE", str(rate))
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.setattr(rt, "ModelLoader", FakeLoader)


@pytest.mark.parametrize("enabled,fails", [(False, False), (True, False), (True, True)])
def test_response_returns_while_candidate_is_blocked(
    monkeypatch, enabled, fails, caplog
):
    configure(monkeypatch, int(enabled))
    started, release = Event(), Event()
    calls = []
    active = {
        "deidentified_text": "[NAME]",
        "pii_entities": [{"label": "NAME", "start": 0, "end": 4}],
    }

    def predict(text, *, model_name, **kwargs):
        calls.append(model_name)
        if model_name == ACTIVE:
            return active
        assert text == NOTE
        started.set()
        if not release.wait(10):
            raise AssertionError("response waited for shadow")
        if fails:
            raise RuntimeError(NOTE)
        return {"deidentified_text": "candidate private output", "pii_entities": []}

    monkeypatch.setattr(openmed, "deidentify", predict)
    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        try:
            response = client.post(
                "/pii/deidentify", json={"text": NOTE, "model_name": ACTIVE}
            )
            assert response.status_code == 200 and response.json() == active
            if enabled:
                assert started.wait(5)
                assert not app.state.shadow_task.done()
                assert (
                    client.post(
                        "/pii/deidentify", json={"text": NOTE, "model_name": ACTIVE}
                    ).json()
                    == active
                )
            else:
                assert calls == [ACTIVE]
        finally:
            release.set()
        if enabled:

            async def settle():
                await asyncio.shield(app.state.shadow_task)

            client.portal.call(settle)
        metrics = app.state.metrics.render()
        if enabled:
            event = "error" if fails else "compared"
            assert f'openmed_service_shadow_total{{outcome="{event}"}} 1' in metrics
            assert 'openmed_service_shadow_total{outcome="busy"} 1' in metrics
            if not fails:
                assert (
                    'openmed_service_shadow_total{outcome="active_only"} 1' in metrics
                )
        else:
            assert "openmed_service_shadow" not in metrics
        for private in [NOTE, "candidate private output"]:
            assert private not in metrics and private not in caplog.text


def test_shadow_refuses_over_budget_and_invalid_sampling(monkeypatch):
    configure(monkeypatch, 1)
    monkeypatch.setenv("OPENMED_SERVICE_PRELOAD_MODELS", ACTIVE)
    monkeypatch.setenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", "100")
    monkeypatch.setenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", "150")
    with pytest.raises(ValueError, match="warm-pool budget"):
        rt.ServiceRuntime.from_env()
    monkeypatch.setenv("OPENMED_SERVICE_SHADOW_SAMPLE_RATE", "nan")
    with pytest.raises(ValueError, match="sample rate"):
        rt.ServiceRuntime.from_env()
