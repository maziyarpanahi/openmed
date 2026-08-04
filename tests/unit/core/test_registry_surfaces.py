"""Single-source regeneration checks for committed registry surfaces."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.core.language_pack_catalog import DEFAULT_MODEL_PLACEHOLDER_LANGUAGES
from openmed.core.manifest_diff import build_registry_surfaces, registry_surface_errors
from openmed.core.model_registry import OPENMED_MODELS
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.core.registry_service import manifest_pii_languages

ROOT = Path(__file__).resolve().parents[3]


def test_committed_registry_surfaces_regenerate_identically() -> None:
    assert registry_surface_errors() == []


def test_registry_cards_have_unique_pointer_metadata() -> None:
    snapshot = build_registry_surfaces()
    titles: set[str] = set()
    descriptions: set[str] = set()

    for card in snapshot.cards.values():
        frontmatter = card.split("---", 2)[1]
        fields = {
            key: json.loads(value.strip())
            for line in frontmatter.splitlines()
            if ":" in line
            for key, value in [line.split(":", 1)]
            if key in {"title", "description"}
        }

        assert "registry checkpoint" in fields["title"]
        assert "registry pointer targeting" in fields["description"]
        assert len(fields["description"]) >= 40
        assert fields["title"] not in titles
        assert fields["description"] not in descriptions
        titles.add(fields["title"])
        descriptions.add(fields["description"])


def test_runtime_registry_and_i18n_surfaces_include_committed_state() -> None:
    latest = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx"

    assert OPENMED_MODELS["pii_latest"].model_id == latest
    assert OPENMED_MODELS["pii_last_green"].model_id == latest
    assert OPENMED_MODELS["pii_latest"].semantic_version == "1.0.0"
    assert SUPPORTED_LANGUAGES == (
        manifest_pii_languages() | set(DEFAULT_MODEL_PLACEHOLDER_LANGUAGES)
    )


def test_registry_coherence_workflow_is_offline_and_diff_guarded() -> None:
    workflow = (ROOT / ".github/workflows/registry-coherence.yml").read_text(
        encoding="utf-8"
    )
    refresh = (ROOT / ".github/workflows/manifest-refresh.yml").read_text(
        encoding="utf-8"
    )

    assert "registry_ctl.py regenerate" in workflow
    assert "registry_ctl.py check" in workflow
    assert "git diff --exit-code" in workflow
    assert "huggingface" not in workflow.casefold()
    assert "registry_ctl.py regenerate" in refresh


def test_committed_registry_state_is_included_in_distributions() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"/gates/registry_state.json"' in pyproject
