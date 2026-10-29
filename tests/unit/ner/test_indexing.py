from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openmed.ner.indexing import (
    GLIFORMER_DISEASE_MODEL_ID,
    ModelIndex,
    ModelRecord,
    build_index,
    load_index,
    resolve_model_record,
    write_index,
)


@pytest.fixture()
def temp_models_dir(tmp_path: Path) -> Path:
    # create fake models with necessary files
    domains = [
        "biomedical",
        "clinical",
        "genomic",
        "finance",
        "legal",
        "news",
        "ecommerce",
        "cybersecurity",
        "chemistry",
        "organism",
        "education",
        "social",
        "public_health",
    ]

    for domain in domains:
        model_dir = tmp_path / f"gliner-{domain}-tiny"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

    return tmp_path


def test_build_index_collects_unique_domains(temp_models_dir: Path) -> None:
    index = build_index(temp_models_dir)
    assert len(index.models) == 13
    expected_domains = {
        "biomedical",
        "clinical",
        "genomic",
        "finance",
        "legal",
        "news",
        "ecommerce",
        "cybersecurity",
        "chemistry",
        "organism",
        "education",
        "social",
        "public_health",
    }
    assert expected_domains.issubset(index.unique_domains)


def test_write_and_load_index_roundtrip(temp_models_dir: Path, tmp_path: Path) -> None:
    index = build_index(temp_models_dir)
    out_path = tmp_path / "index.json"
    write_index(index, out_path, pretty=False)

    loaded = load_index(out_path)
    assert isinstance(loaded, ModelIndex)
    assert len(loaded.models) == len(index.models)
    assert loaded.unique_domains == index.unique_domains


def test_load_index_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in index file") as exc_info:
        load_index(path)

    assert str(path) in str(exc_info.value)


def test_registered_gliformer_disease_defaults():
    record = resolve_model_record(GLIFORMER_DISEASE_MODEL_ID)
    assert record.family == "gliformer"
    assert record.default_labels == ("disease",)
    assert record.license == "Apache-2.0"


def test_unindexed_hub_models_resolve_without_network(subtests):
    for model_id, family in [
        ("OpenMed/GLiFormer-BioNER-Disease-Large-v2", "gliformer"),
        ("knowledgator/gliformer-base-v1", "gliformer"),
        ("fastino/gliner2-base-v1", "gliner2"),
        ("fastino/gliner2.5-base-v1", "gliner2"),
        ("Ihor/gliner-biomed-large-v1.0", "gliner"),
    ]:
        with subtests.test(model_id=model_id, family=family):
            record = resolve_model_record(model_id)
            assert record.family == family
            assert record.path == ""


@pytest.mark.parametrize(
    "config,family",
    [
        ({"model_type": "gliformer-layout"}, "gliformer"),
        ({"model_type": "gliner2", "architecture": "boundary"}, "gliner2"),
        ({"architectures": ["Gliner2ForSchemaExtraction"]}, "gliner2"),
    ],
)
@pytest.mark.parametrize("filename", ["config.json", "gliner_config.json"])
def test_local_config_resolves_arbitrary_directory_name(
    tmp_path, config, family, filename
):
    directory = tmp_path / "arbitrary-export"
    directory.mkdir()
    (directory / filename).write_text(json.dumps(config))
    record = resolve_model_record(str(directory))
    assert record.family == family
    assert record.path == str(directory)
    assert build_index(tmp_path).models[0].family == family


def test_explicit_family_accepts_ambiguous_identifier():
    assert resolve_model_record("synthetic/ner-checkpoint") is None
    assert (
        resolve_model_record("synthetic/ner-checkpoint", family="gliformer").family
        == "gliformer"
    )


def test_explicit_family_does_not_override_checkpoint_config(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "gliner2"}')
    with pytest.raises(ValueError, match="conflicts"):
        resolve_model_record(str(tmp_path), family="gliformer")


def test_family_detection_does_not_match_repository_owner():
    assert resolve_model_record("gliformer/synthetic-bert") is None


def test_default_labels_survive_index_roundtrip(tmp_path):
    record = resolve_model_record(GLIFORMER_DISEASE_MODEL_ID)
    index = ModelIndex(
        models=(record,), generated_at=datetime.now(timezone.utc), source_dir=tmp_path
    )
    path = tmp_path / "index.json"
    write_index(index, path)
    assert load_index(path).models == (record,)


def test_local_token_classifier_with_null_architectures_stays_on_legacy_path(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "bert", "architectures": null}'
    )
    assert resolve_model_record(str(tmp_path)) is None
