"""Unit tests for docker-compose.yml"""

from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[3]
COMPOSE_DIR = BASE_DIR / "docker-compose.yml"


def test_docker_compose():
    with open(COMPOSE_DIR, "r", encoding="utf-8") as f:
        docker_compose = yaml.safe_load(f)
        assert docker_compose is not None


def test_docker_compose_ports():
    with open(COMPOSE_DIR, "r", encoding="utf-8") as f:
        docker_compose = yaml.safe_load(f)
        assert "8080:8080" in docker_compose["services"]["app"]["ports"]


def test_docker_compose_volumes():
    with open(COMPOSE_DIR, "r", encoding="utf-8") as f:
        docker_compose = yaml.safe_load(f)
        assert docker_compose["volumes"]["HF_CACHE"] is None
        assert (
            "HF_CACHE:/root/.cache/huggingface"
            in docker_compose["services"]["app"]["volumes"]
        )
