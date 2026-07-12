"""Schema and source-of-truth checks for the committed model manifest."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.core import model_registry
from openmed.core.manifest_schema import (
    MANIFEST_FIELDS,
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
        assert REQUIRED_FIELDS <= set(row)
        assert set(row) <= MANIFEST_FIELDS
        violations.extend(str(item) for item in validate_manifest_row(row, line_number))
    assert violations == []


def test_enriched_manifest_row_loads_and_validates(tmp_path):
    row = _manifest_row_fixture(
        benchmark=[
            {
                "suite": "shield",
                "dataset": "openmed-golden-pii",
                "micro_f1": 0.9823,
                "recall": 0.991,
                "leakage": 0.0,
            },
            {
                "suite": "clinical-ner",
                "dataset": "synthetic-clinical-ner",
                "micro_f1": 0.901,
                "recall": 0.887,
                "leakage": None,
            },
        ],
        latency_ms={"iphone_15_pro": 18.4, "m2_air": 7},
        peak_ram_mb={"iphone_15_pro": 512, "m2_air": 384.5},
        recommended_tier="phone",
    )
    manifest = tmp_path / "models.jsonl"
    generate_manifest.write_jsonl([row], manifest)

    loaded = model_registry.load_manifest_rows(manifest)
    assert loaded == [row]
    assert validate_manifest_row(loaded[0], line_number=1) == []

    registry = model_registry._build_registry(loaded)
    info = registry["pii_fixture_tiny_65m"]
    assert info.benchmark == row["benchmark"]
    assert info.latency_ms == {"iphone_15_pro": 18.4, "m2_air": 7.0}
    assert info.peak_ram_mb == {"iphone_15_pro": 512.0, "m2_air": 384.5}
    assert info.recommended_tier == "phone"


def test_legacy_manifest_row_without_enrichment_fields_validates():
    row = _manifest_row_fixture()

    assert validate_manifest_row(row, line_number=1) == []
    info = model_registry._build_registry([row])["pii_fixture_tiny_65m"]
    assert info.latency_ms == {}
    assert info.peak_ram_mb == {}
    assert info.recommended_tier is None


def test_manifest_schema_accepts_mlx_4bit_format():
    row = _manifest_row_fixture(
        repo_id="OpenMed/laneformer-2b-it-q4-mlx",
        family="General",
        task="text-generation",
        tier=None,
        param_count=2_320_069_632,
        architecture="laneformer",
        base_model="kogai/laneformer-2b-it",
        formats=["mlx-4bit"],
        canonical_labels=[],
        license="other",
    )

    assert validate_manifest_row(row, line_number=1) == []


def test_manifest_schema_accepts_complete_training_provenance():
    reproducibility_hash = "sha256:" + "1" * 64
    row = _manifest_row_fixture(
        reproducibility_hash=reproducibility_hash,
        training_provenance=_training_provenance_fixture(reproducibility_hash),
    )

    assert validate_manifest_row(row, line_number=1) == []


def test_manifest_schema_rejects_training_provenance_hash_mismatch():
    row = _manifest_row_fixture(
        reproducibility_hash="sha256:" + "1" * 64,
        training_provenance=_training_provenance_fixture("sha256:" + "2" * 64),
    )

    violations = validate_manifest_row(row, line_number=1)

    assert [str(item) for item in violations] == [
        "line 1: training_provenance.reproducibility_hash must match "
        "reproducibility_hash"
    ]


def test_manifest_schema_rejects_incomplete_training_provenance():
    training_provenance = _training_provenance_fixture("sha256:" + "1" * 64)
    training_provenance.pop("rng_seeds")
    row = _manifest_row_fixture(
        reproducibility_hash="sha256:" + "1" * 64,
        training_provenance=training_provenance,
    )

    violations = validate_manifest_row(row, line_number=1)

    assert [
        "line 1: training_provenance missing required key: rng_seeds",
    ] == [str(item) for item in violations]


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
    assert "leakage" in rows[0]["benchmark"]


def test_manifest_generator_infers_language_from_repo_name():
    examples = {
        "Bengali": "bn",
        "Chinese": "zh",
        "Korean": "ko",
        "Vietnamese": "vi",
    }
    for language_name, language_code in examples.items():
        repo_id = f"OpenMed/OpenMed-PII-{language_name}-NomicMed-Large-395M-v1"
        assert generate_manifest._languages(repo_id, []) == [language_code]


def test_pii_language_name_matches_manifest_language():
    language_codes = {
        "Arabic": "ar",
        "Bengali": "bn",
        "Chinese": "zh",
        "Dutch": "nl",
        "French": "fr",
        "German": "de",
        "Hindi": "hi",
        "Italian": "it",
        "Japanese": "ja",
        "Korean": "ko",
        "Portuguese": "pt",
        "Spanish": "es",
        "Telugu": "te",
        "Turkish": "tr",
        "Vietnamese": "vi",
    }
    violations = []
    for row in _rows():
        repo_id = row["repo_id"]
        for language_name, language_code in language_codes.items():
            if f"-PII-{language_name}-" not in repo_id:
                continue
            if row["languages"] != [language_code]:
                violations.append(
                    f"{repo_id}: expected {language_code}, got {row['languages']}"
                )
            break

    assert violations == []


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


def _manifest_row_fixture(**overrides):
    row = {
        "repo_id": "OpenMed/OpenMed-PII-Fixture-Tiny-65M",
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": 65_000_000,
        "architecture": "distilbert",
        "base_model": None,
        "formats": ["pytorch"],
        "canonical_labels": ["PERSON", "DATE"],
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": (
            "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        ),
        "released": "2026-06-24",
    }
    row.update(overrides)
    return row


def _training_provenance_fixture(reproducibility_hash: str):
    return {
        "base_model_revision": "7b4f2ca",
        "data_manifest_hash": "sha256:" + "a" * 64,
        "env_lock_digest": "sha256:" + "b" * 64,
        "git_sha": "abc123",
        "path": "checkpoints/model/training_provenance.json",
        "recipe_config_hash": "sha256:" + "c" * 64,
        "reproducibility_hash": reproducibility_hash,
        "rng_seeds": {"numpy": 21, "python": 13, "torch": 34},
    }
