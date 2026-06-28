"""Structural validation for the REST API recipes documentation page.

These tests confirm ``docs/rest-recipes.md`` exists, carries copy-paste curl and
Python snippets for the documented endpoints using synthetic data, keeps the
endpoint paths and error envelope aligned with ``docs/rest-service.md``, and is
wired into the mkdocs nav and cross-linked with the reference page.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "docs" / "rest-recipes.md"
REFERENCE = ROOT / "docs" / "rest-service.md"
MKDOCS = ROOT / "mkdocs.yml"

ENDPOINTS = ["/health", "/pii/extract", "/pii/deidentify", "/analyze"]


def _recipes_text() -> str:
    return RECIPES.read_text(encoding="utf-8")


def test_recipes_page_exists():
    assert RECIPES.exists()


def test_recipes_cover_every_endpoint_with_curl_and_python():
    text = _recipes_text()
    for path in ENDPOINTS:
        assert path in text, f"missing endpoint recipe: {path}"
    assert "curl" in text
    assert "import requests" in text
    assert "requests.get(" in text
    assert "requests.post(" in text


def test_recipes_use_synthetic_data_disclaimer():
    assert "Synthetic data only" in _recipes_text()


def test_recipes_reference_docker_compose_run_path():
    assert "docker compose up" in _recipes_text()


def test_recipes_endpoint_paths_match_reference():
    reference = REFERENCE.read_text(encoding="utf-8")
    for path in ENDPOINTS:
        assert path in reference, f"endpoint missing from reference doc: {path}"


def test_recipes_error_envelope_matches_reference():
    envelope = '"code": "validation_error|bad_request|timeout|internal_error"'
    reference = REFERENCE.read_text(encoding="utf-8")
    assert envelope in reference
    assert envelope in _recipes_text()


def test_recipes_and_reference_cross_link():
    assert "rest-recipes.md" in REFERENCE.read_text(encoding="utf-8")
    assert "rest-service.md" in _recipes_text()


def test_recipes_page_is_in_mkdocs_nav():
    assert "rest-recipes.md" in MKDOCS.read_text(encoding="utf-8")
