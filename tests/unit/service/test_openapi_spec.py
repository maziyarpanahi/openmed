"""Tests for the committed REST service OpenAPI artifact."""

from __future__ import annotations

import json
import re
from pathlib import Path

import openmed
from scripts.export_openapi import DEFAULT_OUTPUT_PATH, render_openapi_spec

EXPECTED_PATHS = {
    "/health",
    "/models/loaded",
    "/models/unload",
    "/analyze",
    "/fhir/smart-backend/ingestions",
    "/fhir/smart-backend/ingestions/{job_id}",
    "/fhir/smart-backend/ingestions/{job_id}/summary",
    "/pii/extract",
    "/pii/extract/stream",
    "/pii/deidentify",
    "/jobs",
    "/jobs/{job_id}",
    "/privacy-gateway/complete",
}

REST_RECIPES_PATH = Path(__file__).resolve().parents[3] / "docs/rest-recipes.md"
JSON_FENCE_PATTERN = re.compile(r"```json\n(?P<payload>.*?)\n```", re.DOTALL)
PYTHON_FENCE_PATTERN = re.compile(r"```python\n(?P<source>.*?)\n```", re.DOTALL)


def _rest_recipe_json_examples() -> list[dict]:
    text = REST_RECIPES_PATH.read_text(encoding="utf-8")
    return [
        json.loads(match.group("payload"))
        for match in JSON_FENCE_PATTERN.finditer(text)
    ]


def test_committed_openapi_spec_matches_exporter() -> None:
    assert DEFAULT_OUTPUT_PATH.exists(), (
        "docs/api/openapi.json is missing. Re-run "
        ".venv/bin/python scripts/export_openapi.py."
    )

    actual = DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8")
    expected = render_openapi_spec().decode("utf-8")

    assert actual == expected, (
        "docs/api/openapi.json is stale. Re-run "
        ".venv/bin/python scripts/export_openapi.py."
    )


def test_committed_openapi_spec_lists_current_service_paths() -> None:
    spec = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))

    assert spec["info"]["version"] == openmed.__version__
    assert EXPECTED_PATHS.issubset(set(spec["paths"]))


def test_rest_recipe_examples_are_parseable_and_current() -> None:
    examples = _rest_recipe_json_examples()

    assert len(examples) == 8

    health = next(example for example in examples if example.get("service"))
    assert health["version"] == openmed.__version__

    analysis = next(
        example
        for example in examples
        if example.get("model_name") == "disease_detection_superclinical"
    )
    assert {entity["label"] for entity in analysis["entities"]} <= {"DISEASE"}

    error = next(example["error"] for example in examples if "error" in example)
    assert error["request_id"] == "recipe-validation-error"


def test_rest_recipe_entity_spans_match_their_synthetic_text() -> None:
    for example in _rest_recipe_json_examples():
        text = example.get("text") or example.get("original_text")
        entities = example.get("entities") or example.get("pii_entities") or []
        if text is None:
            continue
        for entity in entities:
            assert text[entity["start"] : entity["end"]] == entity["text"]


def test_rest_recipe_python_snippets_compile() -> None:
    text = REST_RECIPES_PATH.read_text(encoding="utf-8")
    snippets = [match.group("source") for match in PYTHON_FENCE_PATTERN.finditer(text)]

    assert snippets
    for index, snippet in enumerate(snippets, start=1):
        compile(snippet, f"rest-recipes-python-{index}", "exec")
