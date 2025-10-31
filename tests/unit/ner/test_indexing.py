from __future__ import annotations

import json
from pathlib import Path

import pytest

from datetime import datetime, timezone

from openmed.ner.indexing import ModelIndex, ModelRecord, build_index, load_index, write_index


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
