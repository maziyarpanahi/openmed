"""Distinct concepts must not be starved by a fixed alias over-fetch budget."""

import hashlib
import json
import math
from unittest.mock import Mock

import pytest

from openmed.clinical.grounding.index import (
    AliasEmbeddingIndex,
    _AliasRecord,
    build_index,
)
from openmed.clinical.grounding.vocab import VocabLoader, VocabSource


def _index(dominant, others=2):
    records = [
        _AliasRecord(
            "ICD10CM", "SYN-A", "synthetic alpha", f"alpha-{i}", "synthetic-v1"
        )
        for i in range(dominant)
    ]
    vectors = [(1.0, 0.0)] * dominant
    for i in range(others):
        records.append(
            _AliasRecord(
                "ICD10CM", f"SYN-{i}", "synthetic beta", f"beta-{i}", "synthetic-v1"
            )
        )
        similarity = 0.8 - i * 0.02
        vectors.append((similarity, math.sqrt(1.0 - similarity**2)))
    return AliasEmbeddingIndex(
        encoder_id="synthetic-test",
        dimension=2,
        backend="brute",
        index_key="synthetic-key",
        vocab_versions={"ICD10CM": "synthetic-v1"},
        records=records,
        vectors=vectors,
    )


@pytest.mark.parametrize("dominant", [8, 9, 32])
def test_many_aliases_do_not_hide_second_concept(dominant):
    index = _index(dominant)
    found = index.query((1.0, 0.0), k=2)
    assert [(c.code, c.score) for c in found] == [("SYN-A", 1.0), ("SYN-0", 0.8)]
    assert found[0].matched_alias == "alpha-0"
    assert all(c.source == "dense" for c in found)


def test_budget_expands_only_as_far_as_needed(monkeypatch):
    index = _index(20, others=20)
    neighbors = Mock(wraps=index._neighbors)
    monkeypatch.setattr(index, "_neighbors", neighbors)
    assert len(index.query((1.0, 0.0), k=2)) == 2
    assert [call.args[1] for call in neighbors.call_args_list] == [8, 16, 32]


def test_budget_terminates_when_fewer_concepts_exist(monkeypatch):
    index = _index(70, others=0)
    neighbors = Mock(wraps=index._neighbors)
    monkeypatch.setattr(index, "_neighbors", neighbors)
    assert len(index.query((1.0, 0.0), k=2)) == 1
    assert [call.args[1] for call in neighbors.call_args_list] == [8, 16, 32, 64, 70]


def test_ordinary_query_does_not_repeat_neighbor_search(monkeypatch):
    index = _index(2, others=20)
    neighbors = Mock(wraps=index._neighbors)
    monkeypatch.setattr(index, "_neighbors", neighbors)
    assert len(index.query((1.0, 0.0), k=3)) == 3
    assert neighbors.call_count == 1
    assert neighbors.call_args.args[1] == 12


def test_empty_index_remains_empty():
    assert _index(0, others=0).query((1.0, 0.0), k=2) == []


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_remains_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        _index(8).query((1.0, 0.0), k=k)


def test_concept_identity_includes_vocabulary_system():
    index = AliasEmbeddingIndex(
        encoder_id="synthetic-test",
        dimension=2,
        backend="brute",
        index_key="synthetic-key",
        vocab_versions={"ICD10CM": "v1", "LOINC": "v1"},
        records=[
            _AliasRecord("ICD10CM", "SYN-1", "alpha", f"a{i}", "v1") for i in range(10)
        ]
        + [_AliasRecord("LOINC", "SYN-1", "beta", "b", "v1")],
        vectors=[(1.0, 0.0)] * 10 + [(0.8, 0.6)],
    )
    assert [(c.system, c.code) for c in index.query((1.0, 0.0), 2)] == [
        ("ICD10CM", "SYN-1"),
        ("LOINC", "SYN-1"),
    ]


def test_public_build_index_and_query_fill_distinct_concepts(tmp_path):
    class SyntheticEncoder:
        encoder_id = "synthetic-alias-budget"
        dimension = 2

        def encode(self, texts):
            return tuple(
                (1.0, 0.0) if text.startswith("alpha") else (0.8, 0.6) for text in texts
            )

    path = tmp_path / "synthetic.jsonl"
    rows = [
        {
            "system": "icd10cm",
            "concept_id": "SYN-A",
            "canonical_term": "alpha0",
            "aliases": [f"alpha{i}" for i in range(20)],
        },
        {
            "system": "icd10cm",
            "concept_id": "SYN-B",
            "canonical_term": "beta",
            "aliases": ["beta"],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    source = VocabSource(
        system="icd10cm",
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    loader = VocabLoader(cache_dir=tmp_path / "cache", registry={"icd10cm": source})
    index = build_index(
        loader, SyntheticEncoder(), systems=("icd10cm",), backend="brute"
    )
    assert index is not None
    assert [c.code for c in index.query((1.0, 0.0), 2)] == ["SYN-A", "SYN-B"]
