"""Tests for the committed REST service OpenAPI artifact."""

from __future__ import annotations

import json
import re
from pathlib import Path

import openmed
from openmed.service.schemas import (
    AnalyzeRequest,
    ModelUnloadRequest,
    PIIDeidentifyRequest,
    PIIExtractRequest,
)
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
    "/pii/deidentify/stream",
    "/jobs",
    "/jobs/{job_id}",
    "/privacy-gateway/complete",
}

REST_RECIPES_PATH = Path(__file__).resolve().parents[3] / "docs/rest-recipes.md"
JSON_FENCE_PATTERN = re.compile(r"```json\n(?P<payload>.*?)\n```", re.DOTALL)
PYTHON_FENCE_PATTERN = re.compile(r"```python\n(?P<source>.*?)\n```", re.DOTALL)
BASH_FENCE_PATTERN = re.compile(r"```bash\n(?P<source>.*?)\n```", re.DOTALL)
RECIPE_ENDPOINT_PATTERN = re.compile(
    r"^## .+ — `(?P<method>GET|POST) (?P<path>/[^`]+)`$",
    re.MULTILINE,
)
H2_PATTERN = re.compile(r"^## ", re.MULTILINE)
CURL_URL_PATTERN = re.compile(r'"\$OPENMED_URL(?P<path>/[^"\s]+)"')
CURL_METHOD_PATTERN = re.compile(r"(?:^|\s)-X\s+(?P<method>[A-Z]+)(?:\s|$)")
CURL_TIMEOUT_PATTERN = re.compile(r"--max-time\s+(?P<timeout>\d+)")
CURL_BODY_PATTERN = re.compile(r"-d\s+'(?P<payload>\{.*?\})'", re.DOTALL)


def _rest_recipe_json_examples() -> list[dict]:
    text = REST_RECIPES_PATH.read_text(encoding="utf-8")
    return [
        json.loads(match.group("payload"))
        for match in JSON_FENCE_PATTERN.finditer(text)
    ]


def _rest_recipe_operations() -> set[tuple[str, str]]:
    text = REST_RECIPES_PATH.read_text(encoding="utf-8")
    return {
        (match.group("method").lower(), match.group("path"))
        for match in RECIPE_ENDPOINT_PATTERN.finditer(text)
    }


def _rest_recipe_curl_examples() -> list[dict]:
    text = REST_RECIPES_PATH.read_text(encoding="utf-8")
    examples = []
    for heading in RECIPE_ENDPOINT_PATTERN.finditer(text):
        next_heading = H2_PATTERN.search(text, heading.end())
        section_end = next_heading.start() if next_heading is not None else len(text)
        section = text[heading.end() : section_end]
        for fence in BASH_FENCE_PATTERN.finditer(section):
            source = fence.group("source")
            if "curl " not in source:
                continue
            url_match = CURL_URL_PATTERN.search(source)
            timeout_match = CURL_TIMEOUT_PATTERN.search(source)
            method_match = CURL_METHOD_PATTERN.search(source)
            body_match = CURL_BODY_PATTERN.search(source)
            assert url_match is not None
            assert timeout_match is not None
            examples.append(
                {
                    "heading_method": heading.group("method"),
                    "heading_path": heading.group("path"),
                    "method": method_match.group("method") if method_match else "GET",
                    "path": url_match.group("path"),
                    "timeout": int(timeout_match.group("timeout")),
                    "payload": (
                        json.loads(body_match.group("payload"))
                        if body_match is not None
                        else None
                    ),
                }
            )
    return examples


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


def test_rest_recipe_operations_match_current_openapi() -> None:
    spec = json.loads(DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    operations = _rest_recipe_operations()
    required = {
        ("get", "/health"),
        ("post", "/analyze"),
        ("post", "/pii/extract"),
        ("post", "/pii/deidentify"),
    }

    assert required <= operations
    for method, path in operations:
        assert path in spec["paths"]
        assert method in spec["paths"][path]


def test_rest_recipe_curl_examples_match_request_schemas() -> None:
    examples = _rest_recipe_curl_examples()
    schema_by_path = {
        "/analyze": AnalyzeRequest,
        "/pii/extract": PIIExtractRequest,
        "/pii/deidentify": PIIDeidentifyRequest,
        "/models/unload": ModelUnloadRequest,
    }
    expected_timeouts = {
        "/health": 10,
        "/analyze": 310,
        "/pii/extract": 310,
        "/pii/deidentify": 310,
        "/models/loaded": 10,
        "/models/unload": 30,
    }

    assert len(examples) == 8
    for example in examples:
        assert example["method"] == example["heading_method"]
        assert example["path"] == example["heading_path"]
        assert example["timeout"] == expected_timeouts[example["path"]]

        schema = schema_by_path.get(example["path"])
        if schema is None:
            assert example["payload"] is None
        else:
            assert example["payload"] is not None
            schema(**example["payload"])


def test_rest_recipe_preserves_dependency_and_privacy_contracts() -> None:
    text = REST_RECIPES_PATH.read_text(encoding="utf-8")

    assert "uv pip install requests" in text
    assert "Treat request and response payloads as PHI" in text
    assert '"keep_mapping": true' not in text
    assert "keep_mapping=True" not in text
    assert "publishes port `8080` on every host interface" in text
    assert "CORS is a browser policy, not authentication" in text
    assert "HTTPS and authentication" in text
    assert "Do not send both" in text


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
