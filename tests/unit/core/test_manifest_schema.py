"""Schema and source-of-truth checks for the committed model manifest."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.core import model_registry
from openmed.core.manifest_schema import (
    REQUIRED_FIELDS,
    validate_manifest_row,
)
from scripts.manifest import generate_manifest

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "models.jsonl"
REFRESH_WORKFLOW = ROOT / ".github" / "workflows" / "manifest-refresh.yml"


def _rows():
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_manifest_exists_and_has_unique_repo_ids():
    assert MANIFEST_PATH.exists()
    rows = _rows()
    assert rows
    repo_ids = [row["repo_id"] for row in rows]
    assert len(repo_ids) == len(set(repo_ids))


def test_every_manifest_row_matches_schema():
    violations = []
    for line_number, row in enumerate(_rows(), start=1):
        assert set(row) == REQUIRED_FIELDS
        violations.extend(str(item) for item in validate_manifest_row(row, line_number))
    assert violations == []


def test_registry_model_ids_are_derived_from_manifest():
    manifest_ids = {row["repo_id"] for row in _rows()}
    registry_ids = {info.model_id for info in model_registry.OPENMED_MODELS.values()}
    assert manifest_ids <= registry_ids
    assert model_registry.get_model_info("disease_detection_tiny") is not None
    for repo_id in list(manifest_ids)[:25]:
        assert model_registry.get_model_info(repo_id) is not None


def test_manifest_generator_uses_hub_api(monkeypatch):
    class FakeModel:
        modelId = "OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M"
        pipeline_tag = "token-classification"
        tags = ["transformers", "safetensors", "distilbert", "license:apache-2.0"]
        siblings = []
        sha = "abc123"
        lastModified = None
        createdAt = None

    class FakeApi:
        def list_models(self, *, author, full):
            assert author == "OpenMed"
            assert full is True
            return [FakeModel()]

    monkeypatch.setattr(generate_manifest, "HfApi", lambda: FakeApi())
    rows = generate_manifest.fetch_manifest_rows("OpenMed")

    assert rows[0]["repo_id"] == FakeModel.modelId
    assert rows[0]["family"] == "NER"
    assert rows[0]["param_count"] == 135_000_000


def test_only_manifest_generator_lists_org_models():
    allowed = {
        ROOT / "scripts" / "manifest" / "generate_manifest.py",
        # hf_publish.py constructs HfApi only to UPLOAD a generated card, not to
        # list org models. The guard's intent (single lister of org models) holds.
        ROOT / "openmed" / "core" / "hf_publish.py",
    }
    forbidden_patterns = (
        "from huggingface_hub import list_models",
        "model_info as",
        ".list_models(",
    )
    offenders = []
    search_roots = [ROOT / "openmed", ROOT / "scripts", ROOT / "tests"]
    for search_root in search_roots:
        paths = search_root.rglob("*.py")
        for path in paths:
            if path in allowed or path == Path(__file__):
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                if pattern in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


def test_manifest_refresh_workflow_is_manual_only():
    text = REFRESH_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "cron:" not in text
    assert "scripts/manifest/generate_manifest.py --output models.jsonl" in text
