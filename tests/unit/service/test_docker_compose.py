"""Unit tests for docker-compose.yml."""

from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[3]
COMPOSE_FILE = BASE_DIR / "docker-compose.yml"


def _load_compose() -> dict:
    with COMPOSE_FILE.open(encoding="utf-8") as handle:
        compose = yaml.safe_load(handle)
    assert isinstance(compose, dict)
    return compose


def _app_service(compose: dict) -> dict:
    service = compose["services"]["app"]
    assert isinstance(service, dict)
    return service


def test_docker_compose_builds_local_dockerfile():
    service = _app_service(_load_compose())

    assert service["build"] == {"context": "."}


def test_docker_compose_maps_service_port():
    service = _app_service(_load_compose())

    assert "8080:8080" in service["ports"]


def test_docker_compose_persists_hf_and_openmed_model_cache():
    compose = _load_compose()
    service = _app_service(compose)
    environment = service["environment"]

    assert "hf-cache" in compose["volumes"]
    assert "hf-cache:/root/.cache/huggingface" in service["volumes"]
    assert environment["OPENMED_PROFILE"] == "${OPENMED_PROFILE:-prod}"
    assert (
        environment["OPENMED_CACHE_DIR"]
        == "${OPENMED_CACHE_DIR:-/root/.cache/huggingface/openmed}"
    )
    assert (
        environment["OPENMED_SERVICE_MAX_RESIDENT_MODELS"]
        == "${OPENMED_SERVICE_MAX_RESIDENT_MODELS:-}"
    )


def test_docker_compose_app_healthcheck_probes_readyz():
    service = _app_service(_load_compose())

    healthcheck = service["healthcheck"]
    assert isinstance(healthcheck, dict)

    test = healthcheck["test"]
    assert isinstance(test, list)
    joined = " ".join(str(part) for part in test)
    assert "http://127.0.0.1:8080/readyz" in joined
    # The probe is bounded: a few seconds to fail fast, a start period for
    # model preload, and a retry budget rather than an indefinite wait.
    assert "timeout=3" in joined or "timeout 3" in joined
    assert healthcheck["interval"] == "10s"
    assert healthcheck["timeout"] == "5s"
    assert healthcheck["start_period"] == "15s"
    assert healthcheck["retries"] == 6
