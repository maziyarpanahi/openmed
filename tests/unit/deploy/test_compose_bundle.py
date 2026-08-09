"""Tests for the offline-first self-hosted Compose bundle."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = ROOT / "deploy" / "openmed-compose.yaml"
DOCS_FILE = ROOT / "docs" / "deploy" / "self-hosted-compose.md"


def _load_compose() -> dict:
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(compose, dict)
    return compose


def _service(compose: dict) -> dict:
    services = compose["services"]
    assert set(services) == {"openmed"}
    service = services["openmed"]
    assert isinstance(service, dict)
    return service


def test_bundle_builds_the_hardened_service_image():
    service = _service(_load_compose())

    assert service["image"] == "${OPENMED_IMAGE:-openmed:distroless}"
    assert service["pull_policy"] == "never"
    assert service["build"] == {
        "context": "..",
        "dockerfile": "deploy/docker/Dockerfile.distroless",
    }
    assert service["user"] == "65532:65532"


def test_bundle_is_local_only_and_keeps_model_inputs_read_only():
    compose = _load_compose()
    service = _service(compose)
    environment = service["environment"]

    assert environment["OPENMED_OFFLINE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["HF_HOME"] == "/cache/huggingface"
    assert environment["OPENMED_CACHE_DIR"] == "/cache/openmed"
    assert "openmed-cache:/cache" in service["volumes"]
    assert "${OPENMED_MODEL_DIR:-../models}:/models:ro" in service["volumes"]
    assert compose["volumes"]["openmed-cache"]["name"] == (
        "${OPENMED_CACHE_VOLUME_NAME:-openmed-cache}"
    )


def test_bundle_disables_optional_egress_and_uses_hardened_runtime_defaults():
    service = _service(_load_compose())
    environment = service["environment"]

    assert environment["OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT"] == ""
    assert environment["OPENMED_SERVICE_METRICS_ENABLED"] == "false"
    assert environment["OPENMED_SERVICE_TRACING_ENABLED"] == "false"
    assert environment["OPENMED_SERVICE_OTLP_ENDPOINT"] == ""
    assert environment["OPENMED_OPENHIM_MEDIATOR_ENABLED"] == "false"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert "/tmp:rw,noexec,nosuid,nodev,size=128m" in service["tmpfs"]


def test_bundle_healthcheck_probes_readiness_with_bounded_timeout():
    healthcheck = _service(_load_compose())["healthcheck"]
    command = " ".join(healthcheck["test"])

    assert "/readyz" in command
    assert "timeout=3" in command
    assert healthcheck["interval"] == "30s"
    assert healthcheck["timeout"] == "5s"
    assert healthcheck["start_period"] == "30s"
    assert healthcheck["retries"] == 3


def test_bundle_docs_cover_offline_startup_permissions_and_integrations():
    docs = DOCS_FILE.read_text(encoding="utf-8")

    for required_text in (
        "docker compose -f deploy/openmed-compose.yaml up -d --no-build",
        "OPENMED_OFFLINE=1",
        "/cache",
        "/models:ro",
        "65532:65532",
        "OpenHIM",
        "privacy gateway",
        "HF_TOKEN",
        "synthetic",
    ):
        assert required_text in docs
