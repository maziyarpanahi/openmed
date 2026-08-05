"""Structural and privacy checks for the static Maple WebGPU clinical demo."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "docs" / "demo" / "web"


def test_maple_demo_has_published_local_first_entrypoint() -> None:
    """The standalone page must load only owned assets under a restrictive CSP."""

    html = (DEMO_DIR / "index.html").read_text(encoding="utf-8")

    assert '<script type="module" src="./app.js"></script>' in html
    assert '<link rel="stylesheet" href="./styles.css" />' in html
    assert "../../stylesheets/openmed-system.css" in html
    assert "../../stylesheets/openmed-standalone.css" in html
    assert "../../javascripts/openmed-standalone.js" in html
    assert 'http-equiv="Content-Security-Policy"' in html
    assert "connect-src 'self' blob:" in html
    assert "script-src 'self' 'wasm-unsafe-eval'" in html
    assert 'rel="canonical"' in html
    assert 'property="og:title"' in html
    assert 'name="twitter:card"' in html
    assert "<style>" not in html
    assert "https://cdn" not in html


def test_maple_demo_exposes_model_cache_and_four_workflows() -> None:
    """The UI includes explicit loading, cache status, tasks, and safety controls."""

    html = (DEMO_DIR / "index.html").read_text(encoding="utf-8")

    for element_id in (
        "runtime-module",
        "repo-id",
        "context-tokens",
        "cache-model",
        "load-model",
        "cancel-load",
        "try-preview",
        "clear-model-cache",
        "model-progress",
        "model-state",
        "input-text",
        "question-text",
        "run-task",
        "stop-generation",
        "clear-session",
        "status",
        "results",
        "raw-output",
    ):
        assert f'id="{element_id}"' in html

    for task in ("pii", "entities", "relations", "chat"):
        assert f'data-task="{task}"' in html
    assert 'role="tablist"' in html
    assert html.count('role="tab"') == 4
    assert 'role="tabpanel"' in html
    assert "Human review required" in html
    assert "not a clinical decision system" in html


def test_maple_demo_enforces_same_origin_runtime_contract() -> None:
    """Inference must use one explicit local adapter with no network fallback."""

    app = (DEMO_DIR / "app.js").read_text(encoding="utf-8")

    assert "https://" not in app
    assert "cdn.jsdelivr" not in app
    assert "createOpenMedMapleRuntime" in app
    assert "import(configuration.runtimeUrl.href)" in app
    assert 'networkPolicy: "same-origin-model-assets-only"' in app
    assert "resolved.origin !== window.location.origin" in app
    assert "credentials, query data, or a fragment" in app
    assert "allowRemoteModels" not in app
    assert "fetch(" not in app
    assert "XMLHttpRequest" not in app
    assert "WebSocket" not in app
    assert "localStorage" not in app
    assert "sessionStorage" not in app
    assert "console." not in app
    assert "pagehide" in app


def test_maple_demo_prompts_cover_structured_clinical_tasks() -> None:
    """Every workflow requests bounded, reviewable output from the causal model."""

    app = (DEMO_DIR / "app.js").read_text(encoding="utf-8")

    for marker in (
        "OPENMED_TASK:PII_REDACTION",
        "OPENMED_TASK:ENTITY_EXTRACTION",
        "OPENMED_TASK:RELATION_EXTRACTION",
        "OPENMED_TASK:EVIDENCE_CHAT",
    ):
        assert marker in app
    assert '"text":"exact unmodified source text"' in app
    assert '"entities"' in app
    assert '"relations"' in app
    assert '"evidence"' in app
    assert '"uncertainty"' in app
    assert "Do not reveal hidden chain-of-thought" in app
    assert "stripPrivateReasoning" in app
    assert "resolveExactSurface" in app
    assert "redactFromSpans(note, spans)" in app
    assert "textContent" in app
    assert "innerHTML" not in app
    assert "example.test" in app


def test_maple_demo_documents_model_sources_adapter_and_validation() -> None:
    """The runbook must explain an operational, local-only browser setup."""

    readme = (DEMO_DIR / "README.md").read_text(encoding="utf-8")

    assert "deepgrove/maple-preview" in readme
    assert "deepgrove/maple-preview-2bit-mlx" in readme
    assert "qmoe-4bit-blockwise-128" in readme
    assert "browser adapter cannot load it" in readme
    assert "createOpenMedMapleRuntime" in readme
    assert "clearOpenMedMapleCache" in readme
    assert "same-origin" in readme
    assert "no cloud fallback" in readme
    assert "python -m http.server" in readme
    assert "make docs-build" in readme
    assert "make docs-browser-test" in readme

    ignored = (DEMO_DIR / ".gitignore").read_text(encoding="utf-8")
    assert "/models/" in ignored
    assert "/vendor/" in ignored


def test_maple_demo_checks_in_mock_tested_ort_web_adapter_contract() -> None:
    """The source adapter owns cached decode plumbing, never runtime artifacts."""

    adapter = (DEMO_DIR / "maple-ort-web-adapter.mjs").read_text(encoding="utf-8")
    readme = (DEMO_DIR / "README.md").read_text(encoding="utf-8")

    assert "export async function createOpenMedMapleRuntime" in adapter
    assert '"./vendor/ort.webgpu.min.mjs"' in adapter
    assert '"./vendor/maple-tokenizer.mjs"' in adapter
    assert 'const BUNDLE_MANIFEST = "maple-bundle.json"' in adapter
    assert "qmoe-4bit-blockwise-128" in adapter
    assert "past_key_values." in adapter
    assert "present." in adapter
    assert "new Uint16Array(0)" in adapter
    assert 'preferredOutputLocation[name] = "gpu-buffer"' in adapter
    assert "runOptions.terminate = true" in adapter
    assert "releaseCacheMap" in adapter
    assert "resolveSameOriginHttpUrl" in adapter
    assert "https://" not in adapter

    assert "real browser" in readme
    assert "remain unvalidated release gates" in readme
    assert "node --test tests/web/test_maple_ort_web_adapter.mjs" in readme
