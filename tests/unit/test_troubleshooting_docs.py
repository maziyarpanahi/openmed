"""Source-of-truth guards for the troubleshooting guide."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "troubleshooting.md"
FAQ = ROOT / "docs" / "faq.md"
CONFIGURATION = ROOT / "docs" / "configuration.md"
MKDOCS = ROOT / "mkdocs.yml"
PYPROJECT = ROOT / "pyproject.toml"
REST_GUIDE = ROOT / "docs" / "rest-service.md"
OPENAPI = ROOT / "docs" / "api" / "openapi.json"

REQUIRED_SECTIONS = (
    "## Install / Extras",
    "## Model Download & Offline",
    "## Performance / Cold-start & Memory",
    "## Device (CPU / GPU / MLX)",
)
REQUIRED_EXTRAS = {"cli", "coreml", "hf", "mlx", "service"}


def _documented_extras(text: str) -> set[str]:
    references: set[str] = set()
    for match in re.findall(r"openmed\[([^\]]+)\]", text):
        references.update(extra.strip() for extra in match.split(","))
    return references


def test_troubleshooting_guide_is_published_and_cross_linked() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    faq = FAQ.read_text(encoding="utf-8")
    nav = MKDOCS.read_text(encoding="utf-8")

    assert all(section in guide for section in REQUIRED_SECTIONS)
    assert "Troubleshooting & Common Errors: troubleshooting.md" in nav
    assert "[Troubleshooting & Common Errors](troubleshooting.md)" in faq
    assert "[FAQ](faq.md)" in guide


def test_documented_install_extras_are_declared_in_pyproject() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    with PYPROJECT.open("rb") as handle:
        optional_dependencies = _toml.load(handle)["project"]["optional-dependencies"]
    declared = set(optional_dependencies)

    documented = _documented_extras(guide)
    assert REQUIRED_EXTRAS <= documented
    assert documented <= declared

    hf_dependencies = {
        re.split(r"[<>=!~;\s\[]", requirement, maxsplit=1)[0].lower()
        for requirement in optional_dependencies["hf"]
    }
    if "torch" not in hf_dependencies:
        assert "compatible PyTorch runtime" in guide


def test_rest_troubleshooting_uses_the_client_port_and_safe_local_bind() -> None:
    from openmed.service.client import OpenMedClient

    guide = GUIDE.read_text(encoding="utf-8")
    rest_guide = REST_GUIDE.read_text(encoding="utf-8")
    default_url = OpenMedClient.__init__.__defaults__[0]
    port = urlparse(default_url).port

    assert default_url in rest_guide
    assert f"--host 127.0.0.1 --port {port}" in guide
    assert "--host 0.0.0.0" not in guide


def test_rest_service_inventory_matches_generated_openapi() -> None:
    spec = json.loads(OPENAPI.read_text(encoding="utf-8"))
    rest_guide = REST_GUIDE.read_text(encoding="utf-8")
    normalized_rest_guide = re.sub(r"\s+", " ", rest_guide)
    inventory = rest_guide.split("## Run Locally", maxsplit=1)[0]

    expected_operations = {
        f"{method.upper()} {path}"
        for path, path_item in spec["paths"].items()
        for method in path_item
        if method != "parameters"
    }
    documented_operations = {
        f"{method} {path}"
        for method, path in re.findall(r"`(GET|POST) ([^`]+)`", inventory)
    }
    documented_operations.discard("GET /metrics")

    assert documented_operations == expected_operations
    assert "OpenAPI document is the source of truth" in normalized_rest_guide
    assert "intentionally excluded from the generated schema" in normalized_rest_guide
    assert "0.6.2" not in rest_guide


def test_support_examples_require_synthetic_phi_safe_reproductions() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    rest_guide = REST_GUIDE.read_text(encoding="utf-8")

    assert "Use only synthetic text in reproductions" in guide
    assert "All patient-like content in the examples below is synthetic" in rest_guide
    assert "reversible mappings" in guide
    assert "access tokens" in guide
    assert "keep_mapping=True" not in rest_guide
    assert "timeout=300.0" in rest_guide


def test_troubleshooting_describes_real_loader_and_middleware_lifecycles() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    rest_guide = REST_GUIDE.read_text(encoding="utf-8")
    normalized_guide = re.sub(r"\s+", " ", guide)

    assert (
        "Constructing `ModelLoader()` by itself does not download" in normalized_guide
    )
    assert "separate top-level convenience calls can load" in normalized_guide
    assert "CORS preflight" in rest_guide
    assert "All non-2xx responses use this shape" not in rest_guide


def test_container_guidance_distinguishes_host_bind_and_cache_controls() -> None:
    rest_guide = REST_GUIDE.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", rest_guide)

    assert "docker run --rm -p 127.0.0.1:8080:8080" in rest_guide
    assert "change the mapping to `127.0.0.1:8080:8080`" in normalized
    assert "not a generic `OpenMedConfig.cache_dir` override" in normalized


def test_linked_configuration_examples_use_the_real_public_surface() -> None:
    configuration = CONFIGURATION.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", configuration)

    assert "load_config_from_file" in configuration
    assert "OPENMED_CONFIG" in configuration
    assert "config.toml" in configuration
    assert "MPS → CUDA → CPU" in configuration
    assert "OpenMedConfig.from_file" not in configuration
    assert "OPENMED_CONFIG_FILE" not in configuration
    assert "config.yaml" not in configuration
    assert "strip=True" not in configuration
    assert "json=True" not in configuration
    assert "OPENMED_DISABLE_WARNINGS" not in configuration
    assert "`OpenMedConfig` has no `pipeline` field" in configuration
    assert "does not enforce a model allowlist" in normalized
