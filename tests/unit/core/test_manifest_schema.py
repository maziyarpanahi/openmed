"""Schema and source-of-truth checks for the committed model manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.manifest import generate_manifest
from openmed.core import model_registry


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "models.jsonl"
REFRESH_WORKFLOW = ROOT / ".github" / "workflows" / "manifest-refresh.yml"
REQUIRED_FIELDS = {
    "repo_id",
    "family",
    "task",
    "languages",
    "tier",
    "param_count",
    "architecture",
    "base_model",
    "formats",
    "canonical_labels",
    "benchmark",
    "arxiv",
    "license",
    "reproducibility_hash",
    "released",
}


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
    for row in _rows():
        assert set(row) == REQUIRED_FIELDS
        assert isinstance(row["repo_id"], str) and row["repo_id"].startswith("OpenMed/")
        assert isinstance(row["family"], str) and row["family"]
        assert isinstance(row["task"], str) and row["task"]
        assert isinstance(row["languages"], list)
        assert all(isinstance(language, str) for language in row["languages"])
        assert row["tier"] is None or isinstance(row["tier"], str)
        assert row["param_count"] is None or (
            isinstance(row["param_count"], int) and row["param_count"] > 0
        )
        assert row["architecture"] is None or isinstance(row["architecture"], str)
        assert row["base_model"] is None or isinstance(row["base_model"], str)
        assert isinstance(row["formats"], list) and row["formats"]
        assert all(isinstance(format_name, str) for format_name in row["formats"])
        assert isinstance(row["canonical_labels"], list)
        assert all(isinstance(label, str) for label in row["canonical_labels"])
        assert isinstance(row["benchmark"], dict)
        assert {"dataset", "micro_f1", "recall"} <= set(row["benchmark"])
        assert row["arxiv"] is None or isinstance(row["arxiv"], str)
        assert row["license"] is None or isinstance(row["license"], str)
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", row["reproducibility_hash"])
        assert row["released"] is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["released"])


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
    allowed = ROOT / "scripts" / "manifest" / "generate_manifest.py"
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
            if path == allowed or path == Path(__file__):
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
