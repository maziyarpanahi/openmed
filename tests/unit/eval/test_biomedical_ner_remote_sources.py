"""Remote biomedical NER corpora load without running repository code."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from openmed.eval.datasets import (
    BC2GM,
    BC5CDR,
    BIOMEDICAL_NER_CORPORA,
    NCBI_DISEASE,
    SPECIES_800,
    biomedical_ner_suite_metadata,
    load_biomedical_ner_corpus,
)
from openmed.eval.datasets.biomedical_ner import BIOMEDICAL_NER_SOURCES

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class _FakeClassLabel:
    names = ["O", "B", "I"]


class _FakeSequence:
    feature = _FakeClassLabel()


class _FakeDataset(list):
    features = {"ner_tags": _FakeSequence()}


def _install_fake_datasets(
    monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def load_dataset(*args: Any, **kwargs: Any) -> _FakeDataset:
        calls.append((args, kwargs))
        return _FakeDataset(rows)

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_remote_sources_never_request_dataset_code() -> None:
    for corpus in BIOMEDICAL_NER_CORPORA:
        source = BIOMEDICAL_NER_SOURCES[corpus]
        assert not hasattr(source, "trust_remote_code")
        if source.revision is not None:
            assert _COMMIT_SHA.fullmatch(source.revision), corpus


def test_pinned_source_loads_parquet_at_its_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"id": "s1", "tokens": ["BRCA1", "binds"], "ner_tags": [1, 0]},
    ]
    calls = _install_fake_datasets(monkeypatch, rows)
    source = BIOMEDICAL_NER_SOURCES[BC2GM]

    corpus = load_biomedical_ner_corpus(BC2GM, cache_dir=tmp_path)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("parquet",)
    assert kwargs == {
        "cache_dir": str(tmp_path),
        "data_files": {
            "test": (
                f"hf://datasets/bigbio/blurb@{source.revision}/bc2gm/test/*.parquet"
            )
        },
        "split": "test",
    }
    assert "trust_remote_code" not in kwargs
    assert [span.text for span in corpus.records[0].spans] == ["BRCA1"]


def test_data_dir_overrides_the_config_name(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_datasets(monkeypatch, [])
    source = BIOMEDICAL_NER_SOURCES[SPECIES_800]

    load_biomedical_ner_corpus(SPECIES_800, split="validation")

    ((_args, kwargs),) = calls
    assert kwargs["data_files"] == {
        "validation": (
            f"hf://datasets/spyysalo/species_800@{source.revision}/"
            "species_800/validation/*.parquet"
        )
    }


@pytest.mark.parametrize("corpus", [BC5CDR, NCBI_DISEASE])
def test_script_only_sources_require_a_local_export(
    corpus: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_datasets(monkeypatch, [])

    with pytest.raises(RuntimeError, match="does not run dataset code"):
        load_biomedical_ner_corpus(corpus)

    assert calls == []


def test_suite_metadata_records_the_pinned_revision() -> None:
    sources = biomedical_ner_suite_metadata()["sources"]

    assert sources[BC2GM]["revision"] == BIOMEDICAL_NER_SOURCES[BC2GM].revision
    assert sources[BC5CDR]["revision"] is None
