"""Structural checks for the static WASM/WebGPU browser demo."""

from __future__ import annotations

import re
from pathlib import Path

from openmed.onnx.transformersjs import REQUIRED_BUNDLE_FILES

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "docs" / "demo" / "web"


def test_web_demo_has_static_entrypoint_and_controls() -> None:
    """The static page must expose model, backend, timing, and result controls."""

    html = (DEMO_DIR / "index.html").read_text(encoding="utf-8")

    assert '<script type="module" src="./app.js"></script>' in html
    assert 'id="repo-id"' in html
    assert 'value="wasm"' in html
    assert 'value="webgpu"' in html
    for element_id in (
        "wasm-load",
        "wasm-first",
        "webgpu-load",
        "webgpu-first",
        "run-selected",
        "benchmark-both",
        "results",
    ):
        assert f'id="{element_id}"' in html


def test_web_demo_wires_transformersjs_backends_and_separate_timings() -> None:
    """Both execution providers must use one token-classification code path."""

    app = (DEMO_DIR / "app.js").read_text(encoding="utf-8")

    assert "@huggingface/transformers@4.2.0" in app
    assert 'const BACKENDS = ["wasm", "webgpu"]' in app
    assert 'pipeline("token-classification"' in app
    assert "device: backend" in app
    assert 'wasm: "q8"' in app
    assert 'webgpu: "fp16"' in app
    assert 'aggregation_strategy: "simple"' in app
    assert app.count("performance.now()") >= 4
    assert "loadMs" in app
    assert "firstInferenceMs" in app
    assert "locateWord" in app
    assert "mergeBioSpans" in app


def test_web_demo_documents_manifest_repo_and_export_bundle_layout() -> None:
    """The runbook must map a manifest repo id to the validated export layout."""

    readme = (DEMO_DIR / "README.md").read_text(encoding="utf-8")

    assert "models.jsonl" in readme
    assert "`repo_id`" in readme
    assert "formats" in readme and "transformersjs" in readme
    assert "python -m http.server" in readme
    for relative_path in REQUIRED_BUNDLE_FILES:
        assert relative_path in readme

    default_match = re.search(
        r'const DEFAULT_REPO_ID = "(?P<repo_id>[^\"]+)";',
        (DEMO_DIR / "app.js").read_text(encoding="utf-8"),
    )
    assert default_match is not None
    assert default_match.group("repo_id").count("/") == 1
