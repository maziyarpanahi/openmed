"""Structural checks for the static WASM/WebGPU browser demo."""

from __future__ import annotations

from pathlib import Path

from openmed.onnx.transformersjs import REQUIRED_BUNDLE_FILES

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "docs" / "demo" / "web"


def test_web_demo_has_static_entrypoint_and_controls() -> None:
    """The static page must expose model, backend, timing, and result controls."""

    html = (DEMO_DIR / "index.html").read_text(encoding="utf-8")

    assert '<script type="module" src="./app.js"></script>' in html
    assert "../../stylesheets/openmed-system.css" in html
    assert "../../stylesheets/openmed-standalone.css" in html
    assert "../../javascripts/openmed-standalone.js" in html
    assert 'rel="canonical"' in html
    assert 'property="og:title"' in html
    assert 'name="twitter:card"' in html
    assert "<style>" not in html
    assert 'id="repo-id"' in html
    assert 'id="runtime-module"' in html
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
    assert "<caption" in html
    assert 'scope="col"' in html
    assert 'scope="row"' in html


def test_web_demo_wires_transformersjs_backends_and_separate_timings() -> None:
    """Both providers use one explicitly configured, same-origin code path."""

    app = (DEMO_DIR / "app.js").read_text(encoding="utf-8")

    assert "https://" not in app
    assert "cdn.jsdelivr" not in app
    assert 'const BACKENDS = ["wasm", "webgpu"]' in app
    assert "import(runtimeUrl.href)" in app
    assert 'runtime.createOpenMedPipeline !== "function"' in app
    assert "resolved.origin !== window.location.origin" in app
    assert 'task: "token-classification"' in app
    assert "modelUrl: configuration.modelUrl.href" in app
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

    app = (DEMO_DIR / "app.js").read_text(encoding="utf-8")
    assert 'query.get("runtime") ?? ""' in app
    assert 'query.get("model") ?? query.get("repo_id") ?? ""' in app
    assert "Supply a same-origin runtime module and model bundle" in app
