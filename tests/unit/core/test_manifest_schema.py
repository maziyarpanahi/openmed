"""Schema and source-of-truth checks for the committed model manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core import model_registry
from openmed.core.manifest_schema import (
    LANGUAGE_SCRIPT_TARGETS,
    MANIFEST_FIELDS,
    REQUIRED_FIELDS,
    SCRIPT_COVERAGE_TARGETS,
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
        download_mb=131.794,
        disk_mb=131.794,
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
    assert info.download_mb == 131.794
    assert info.disk_mb == 131.794
    assert info.latency_ms == {"iphone_15_pro": 18.4, "m2_air": 7.0}
    assert info.peak_ram_mb == {"iphone_15_pro": 512.0, "m2_air": 384.5}
    assert info.recommended_tier == "phone"
    assert info.script_coverage == row["script_coverage"]


def test_pii_manifest_row_without_script_coverage_fails():
    row = _manifest_row_fixture()
    del row["script_coverage"]

    assert [str(item) for item in validate_manifest_row(row, line_number=1)] == [
        "line 1: PII entry missing required key: script_coverage"
    ]


def test_script_coverage_verdict_matches_claimed_language_threshold():
    row = _manifest_row_fixture(languages=["hi"])
    row["script_coverage"] = _script_coverage_fixture(["hi"])
    row["script_coverage"]["devanagari"]["unk_rate"] = 0.010001

    violations = validate_manifest_row(row, line_number=1)

    assert [str(item) for item in violations] == [
        "line 1: script_coverage.devanagari.verdict must be unsupported for the "
        "declared languages and UNK rate"
    ]


def test_manifest_row_without_size_enrichment_loads_with_empty_defaults():
    row = _manifest_row_fixture()

    assert validate_manifest_row(row, line_number=1) == []
    info = model_registry._build_registry([row])["pii_fixture_tiny_65m"]
    assert info.latency_ms == {}
    assert info.peak_ram_mb == {}
    assert info.download_mb is None
    assert info.disk_mb is None
    assert info.recommended_tier is None


def test_model_card_release_metadata_validates_for_claimed_scripts():
    row = _manifest_row_fixture(
        languages=["zh"],
        download_mb=2252.843,
        disk_mb=2252.843,
        download_sizes={
            "safetensors": 2235.723,
            "mlx": 2235.717,
            "coreml": None,
            "onnx": 1330.432,
        },
        script_eval={
            "han_simplified": {
                "dataset": "synthetic-zh-pii",
                "recall": 0.7517,
                "leakage_floor": None,
            },
            "han_traditional": {
                "dataset": None,
                "recall": None,
                "leakage_floor": None,
            },
        },
    )
    row["script_coverage"] = _script_coverage_fixture(["zh"])

    assert validate_manifest_row(row, line_number=1) == []


def test_model_card_release_metadata_requires_all_formats_and_claimed_scripts():
    row = _manifest_row_fixture(
        languages=["bn"],
        download_sizes={"safetensors": 100.0},
        script_eval={},
    )
    row["script_coverage"] = _script_coverage_fixture(["bn"])

    violations = {str(item) for item in validate_manifest_row(row, line_number=1)}

    assert "line 1: script_eval missing claimed script: bengali" in violations
    assert "line 1: download_sizes missing required format: coreml" in violations
    assert "line 1: download_sizes missing required format: mlx" in violations
    assert "line 1: download_sizes missing required format: onnx" in violations


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


def test_explicit_ner_repo_name_wins_over_inherited_pii_tags():
    class FakeModel:
        modelId = "OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M-mlx"
        pipeline_tag = "token-classification"
        tags = [
            "pii",
            "de-identification",
            "mlx",
            "base_model:OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M",
        ]
        siblings = []
        sha = "abc123"
        lastModified = None
        createdAt = None

    row = generate_manifest.model_to_manifest_row(FakeModel())

    assert row["family"] == "NER"
    assert row["canonical_labels"] == ["CHEM"]
    assert "PERSON" not in row["canonical_labels"]


def test_generic_pii_repo_name_remains_a_pii_family_fallback():
    assert (
        generate_manifest._family(
            "OpenMed/gliner-multi-pii-v1-mlx",
            [],
            "token-classification",
        )
        == "PII"
    )


def test_manifest_generator_infers_korean_from_repo_name():
    assert generate_manifest._languages(
        "OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1", []
    ) == ["ko"]


@pytest.mark.parametrize(
    ("language_name", "language_code"),
    (("Bengali", "bn"), ("Chinese", "zh"), ("Tamil", "ta")),
)
def test_manifest_generator_infers_v2_languages_from_repo_name(
    language_name,
    language_code,
):
    assert generate_manifest._languages(
        f"OpenMed/OpenMed-PII-{language_name}-Fixture-Large-279M-v1",
        [],
    ) == [language_code]


def test_manifest_generator_preserves_script_coverage(tmp_path):
    output = tmp_path / "models.jsonl"
    previous = _manifest_row_fixture()
    generate_manifest.write_jsonl([previous], output)
    refreshed = _manifest_row_fixture()
    del refreshed["script_coverage"]

    rows = generate_manifest.preserve_existing_enrichment([refreshed], output)

    assert rows[0]["script_coverage"] == previous["script_coverage"]


def test_manifest_generator_preserves_model_card_release_metadata(tmp_path):
    output = tmp_path / "models.jsonl"
    previous = _manifest_row_fixture(
        download_mb=131.794,
        disk_mb=131.794,
        download_sizes={
            "safetensors": 130.0,
            "mlx": 128.0,
            "coreml": None,
            "onnx": 65.0,
        },
        script_eval={},
    )
    generate_manifest.write_jsonl([previous], output)
    refreshed = _manifest_row_fixture()
    for field in (
        "download_mb",
        "disk_mb",
        "download_sizes",
        "script_eval",
    ):
        refreshed.pop(field, None)

    rows = generate_manifest.preserve_existing_enrichment([refreshed], output)

    assert rows[0]["download_mb"] == 131.794
    assert rows[0]["disk_mb"] == 131.794
    assert rows[0]["download_sizes"] == previous["download_sizes"]
    assert rows[0]["script_eval"] == {}


def test_manifest_generator_drops_stale_script_coverage_after_family_change(
    tmp_path,
):
    output = tmp_path / "models.jsonl"
    repo_id = "OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M-mlx"
    previous = _manifest_row_fixture(repo_id=repo_id)
    generate_manifest.write_jsonl([previous], output)
    refreshed = _manifest_row_fixture(
        repo_id=repo_id,
        family="NER",
        canonical_labels=["CHEM"],
    )
    del refreshed["script_coverage"]

    rows = generate_manifest.preserve_existing_enrichment([refreshed], output)

    assert "script_coverage" not in rows[0]


def test_committed_ner_rows_are_not_misclassified_as_pii():
    offenders = []
    for row in _rows():
        if not row["repo_id"].startswith("OpenMed/OpenMed-NER-"):
            continue
        if (
            row["family"] != "NER"
            or "script_coverage" in row
            or row["canonical_labels"] == generate_manifest.PII_CANONICAL_LABELS
        ):
            offenders.append(row["repo_id"])

    assert offenders == []


def test_committed_pharma_rows_use_the_runtime_chem_label():
    rows = [
        row
        for row in _rows()
        if row["repo_id"].startswith("OpenMed/OpenMed-NER-PharmaDetect-")
    ]

    assert rows
    assert all(row["canonical_labels"] == ["CHEM"] for row in rows)


def test_pharma_mlx_registry_entry_uses_ner_metadata():
    repo_id = "OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M-mlx"

    info = model_registry.get_model_info(repo_id)

    assert info is not None
    assert info.family == "NER"
    assert info.category == "Pharmaceutical"
    assert info.recommended_confidence == 0.65
    assert info.script_coverage == {}
    assert repo_id not in {
        model.model_id
        for model in model_registry.get_pii_models_by_language("en").values()
    }


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
        "script_coverage": _script_coverage_fixture(["en"]),
    }
    row.update(overrides)
    return row


def _script_coverage_fixture(languages):
    claimed = {
        script
        for language in languages
        for script in LANGUAGE_SCRIPT_TARGETS.get(language, ())
    }
    return {
        script: {
            "unk_rate": 0.0,
            "byte_fallback_rate": 0.0,
            "tokens_per_grapheme": 1.0,
            "verdict": "supported" if script in claimed else "unclaimed",
        }
        for script in SCRIPT_COVERAGE_TARGETS
    }


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
