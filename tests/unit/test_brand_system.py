"""Regression tests for the repository-owned OpenMed brand system."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "scripts/brand/validate_system.py"
SOCIAL_RENDERER = REPO_ROOT / "scripts/brand/render_social_assets.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_brand_system", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_social_renderer():
    spec = importlib.util.spec_from_file_location(
        "render_social_assets", SOCIAL_RENDERER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_brand_system_contract_and_reproducibility() -> None:
    validator = _load_validator()
    assert validator.validate() == []


def test_social_masters_are_exact_approved_handoff_exports() -> None:
    source = json.loads(
        (REPO_ROOT / "docs/brand/social/_src/exports.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (REPO_ROOT / "docs/brand/system/handoff-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    approved_hashes = provenance["approved_exports"]
    assert {Path(asset["source"]).name for asset in source["assets"]} == set(
        approved_hashes
    )

    for asset in source["assets"]:
        approved = REPO_ROOT / asset["source"]
        master = REPO_ROOT / asset["master"]
        assert (
            hashlib.sha256(approved.read_bytes()).hexdigest()
            == approved_hashes[approved.name]
        )
        assert master.read_bytes() == approved.read_bytes()


def test_social_manifest_portability_ignores_only_derivative_encoding_bytes() -> None:
    renderer = _load_social_renderer()
    manifest = json.loads(
        (REPO_ROOT / "docs/brand/social/manifest.json").read_text(encoding="utf-8")
    )
    platform_variant = deepcopy(manifest)
    platform_variant["assets"][0]["native_sha256"] = "platform-specific-png"
    platform_variant["assets"][1]["safe_zone_preview_sha256"] = (
        "platform-specific-preview"
    )
    platform_variant["derived_assets"][0]["sha256"] = "platform-specific-encoding"

    assert renderer._manifest_without_derivative_byte_hashes(
        platform_variant
    ) == renderer._manifest_without_derivative_byte_hashes(manifest)

    platform_variant["assets"][0]["native_pixel_sha256"] = "pixel-drift"
    assert renderer._manifest_without_derivative_byte_hashes(
        platform_variant
    ) != renderer._manifest_without_derivative_byte_hashes(manifest)


def test_faq_parity_rejects_visible_answer_drift() -> None:
    validator = _load_validator()
    faq_page = {
        "mainEntity": [
            {
                "name": "Where does clinical text go?",
                "acceptedAnswer": {"text": "The deployment determines the data path."},
            }
        ]
    }
    website = """
    <section id="faq">
      <article class="faq-item">
        <button><span>Where does clinical text go?</span></button>
        <div id="faq-answer-1"><p>The deployment determines the data path.</p></div>
      </article>
    </section>
    """
    errors: list[str] = []
    validator._validate_faq_parity(website, faq_page, errors)
    assert errors == []

    drifted = website.replace(
        "The deployment determines the data path.",
        "Nothing ever leaves the device.",
    )
    validator._validate_faq_parity(drifted, faq_page, errors)
    assert any("visible FAQ disagrees with JSON-LD" in error for error in errors)
