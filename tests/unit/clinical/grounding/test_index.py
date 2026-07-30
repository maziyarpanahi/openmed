"""Tests for the offline alias embedding index.

All vocabulary content is synthetic and algorithmically generated; no real
patient data or licensed terminology is used. Embeddings come from the
dependency-free deterministic hashing encoder, so the suite is fully offline and
byte-for-byte reproducible.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    AliasEmbeddingIndex,
    Candidate,
    DenseCandidateGenerator,
    HashingAliasEncoder,
    build_index,
    build_or_load_index,
    get_linker,
    load_encoder,
    load_index,
    query_index,
)
from openmed.clinical.grounding.index import brute_force_neighbors
from openmed.clinical.grounding.vocab import VocabLoader, VocabSource

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _REPO_ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"
_SYSTEMS = ("icd10cm", "rxnorm", "loinc", "hpo", "mesh")


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in _FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _loader(tmp_path: Path, path: Path = _FIXTURE) -> VocabLoader:
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    registry = {
        system: VocabSource(system=system, path=path, sha256=sha256)
        for system in _SYSTEMS
    }
    return VocabLoader(cache_dir=tmp_path / "cache", registry=registry)


def test_build_then_query_returns_dense_candidates(tmp_path):
    encoder = HashingAliasEncoder()
    index = build_index(_loader(tmp_path), encoder, systems=["icd10cm"])

    assert index is not None
    candidates = index.query_text("type 2 diabetes", encoder, k=5)

    assert candidates
    top = candidates[0]
    assert isinstance(top, Candidate)
    assert top.system == "ICD10CM"
    assert top.code == "E11.9"
    assert top.source == "dense"
    assert top.match_kind == "dense"
    assert top.matched_alias
    assert top.vocab_version.startswith("sha256:")
    assert 0.0 < top.score <= 1.0
    assert len(candidates) <= 5


def test_exact_alias_scores_one(tmp_path):
    encoder = HashingAliasEncoder()
    index = build_index(_loader(tmp_path), encoder, systems=["icd10cm"])

    (vector,) = encoder.encode(["high blood pressure"])
    candidates = index.query(vector, k=3)

    assert candidates[0].code == "I10"
    assert candidates[0].score == pytest.approx(1.0)


def test_query_matches_brute_force_reference_recall_at_10(tmp_path):
    encoder = HashingAliasEncoder()
    loader = _loader(tmp_path)
    index = build_index(loader, encoder, systems=list(_SYSTEMS), backend="brute")
    assert index is not None

    # Independent brute-force reference over the same (concept, alias) rows.
    reference_vectors: list[tuple[float, ...]] = []
    reference_codes: list[str] = []
    for row in _fixture_rows():
        for alias in row["aliases"]:
            (vector,) = encoder.encode([alias])
            reference_vectors.append(vector)
            reference_codes.append(row["concept_id"])

    total = 0.0
    count = 0
    for row in _fixture_rows():
        for alias in row["aliases"]:
            (vector,) = encoder.encode([alias])
            exact_rows = brute_force_neighbors(reference_vectors, vector, 10)
            exact_codes = {reference_codes[r] for r, _ in exact_rows}
            served = {c.code for c in index.query(vector, 10)}
            total += len(exact_codes & served) / len(exact_codes)
            count += 1

    recall = total / count
    assert recall >= 0.95


def test_changing_vocabulary_version_rekeys_and_rebuilds(tmp_path):
    encoder = HashingAliasEncoder()
    cache_dir = tmp_path / "index-cache"

    first_fixture = tmp_path / "v1.jsonl"
    first_fixture.write_text(_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    loader_v1 = _loader(tmp_path / "v1", path=first_fixture)
    index_v1 = build_or_load_index(
        loader_v1, encoder, cache_dir=cache_dir, systems=["icd10cm"]
    )
    assert index_v1 is not None
    persisted_key = load_index(cache_dir).index_key
    assert persisted_key == index_v1.index_key

    # A changed vocabulary edition must change the content hash and the key.
    second_fixture = tmp_path / "v2.jsonl"
    second_fixture.write_text(
        _FIXTURE.read_text(encoding="utf-8")
        + json.dumps(
            {
                "concept_id": "E11.51",
                "system": "icd10cm",
                "canonical_term": "Type 2 diabetes mellitus with diabetic peripheral angiopathy",
                "aliases": ["type 2 diabetes with peripheral angiopathy"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loader_v2 = _loader(tmp_path / "v2", path=second_fixture)
    index_v2 = build_or_load_index(
        loader_v2, encoder, cache_dir=cache_dir, systems=["icd10cm"]
    )

    assert index_v2 is not None
    assert index_v2.index_key != index_v1.index_key
    assert index_v2.vocab_versions != index_v1.vocab_versions
    # The stale payload was overwritten, not silently reused.
    assert load_index(cache_dir).index_key == index_v2.index_key


def test_unchanged_vocabulary_reuses_persisted_index(tmp_path):
    encoder = HashingAliasEncoder()
    cache_dir = tmp_path / "index-cache"
    loader = _loader(tmp_path)

    built = build_or_load_index(
        loader, encoder, cache_dir=cache_dir, systems=["icd10cm"]
    )
    assert built is not None
    reloaded = build_or_load_index(
        loader, encoder, cache_dir=cache_dir, systems=["icd10cm"]
    )
    assert reloaded is not None
    assert reloaded.index_key == built.index_key


def test_cache_hit_does_not_re_encode(tmp_path):
    # Version-keyed persistence must validate a cache hit from content hashes
    # WITHOUT re-encoding the vocabulary (the whole point of the persistence).
    class _CountingEncoder:
        def __init__(self, inner):
            self._inner = inner
            self.encode_calls = 0

        @property
        def encoder_id(self):
            return self._inner.encoder_id

        @property
        def dimension(self):
            return self._inner.dimension

        def encode(self, surfaces):
            self.encode_calls += 1
            return self._inner.encode(surfaces)

    encoder = _CountingEncoder(HashingAliasEncoder())
    cache_dir = tmp_path / "index-cache"
    loader = _loader(tmp_path)

    build_or_load_index(loader, encoder, cache_dir=cache_dir, systems=["icd10cm"])
    calls_after_first_build = encoder.encode_calls
    assert calls_after_first_build >= 1  # first build encodes

    build_or_load_index(loader, encoder, cache_dir=cache_dir, systems=["icd10cm"])
    # Cache hit: no re-encode.
    assert encoder.encode_calls == calls_after_first_build


def test_missing_encoder_degrades_to_sparse_only(tmp_path):
    # load_encoder with no configured weights is the documented no-op.
    assert load_encoder() is None

    index = build_index(_loader(tmp_path), None, systems=["icd10cm"])
    assert index is None
    assert query_index(index, [0.0], k=5) == []

    generator = DenseCandidateGenerator(encoder=None, loader=_loader(tmp_path))
    assert generator.generate("type 2 diabetes", ["icd10cm"]) == []


def test_dense_generator_resolvable_via_registry(tmp_path):
    factory = get_linker("dense")
    assert factory is DenseCandidateGenerator

    generator = factory(HashingAliasEncoder(), loader=_loader(tmp_path))
    candidates = generator.generate("type 2 diabetes", ["icd10cm"], k=3)
    assert candidates[0].code == "E11.9"
    assert all(candidate.source == "dense" for candidate in candidates)


def test_persisted_index_records_provenance(tmp_path):
    encoder = HashingAliasEncoder()
    index = build_index(_loader(tmp_path), encoder, systems=["icd10cm"])
    assert index is not None

    directory = tmp_path / "persist"
    index.save(directory)
    payload = json.loads((directory / "alias_index.json").read_text(encoding="utf-8"))

    provenance = payload["provenance"]
    assert provenance["encoder_id"] == encoder.encoder_id
    assert provenance["index_key"] == index.index_key
    assert "ICD10CM" in provenance["vocab_versions"]
    assert provenance["vocab_versions"]["ICD10CM"].startswith("sha256:")

    restored = AliasEmbeddingIndex.from_payload(payload)
    assert restored.index_key == index.index_key
    assert restored.record_count == index.record_count


def test_results_are_deterministic_across_runs(tmp_path):
    encoder = HashingAliasEncoder()
    first = build_index(_loader(tmp_path), encoder, systems=list(_SYSTEMS))
    second = build_index(_loader(tmp_path), encoder, systems=list(_SYSTEMS))
    assert first is not None and second is not None
    assert first.index_key == second.index_key

    (vector,) = encoder.encode(["pneumonia"])
    assert [repr(c) for c in first.query(vector, 5)] == [
        repr(c) for c in second.query(vector, 5)
    ]


def test_build_and_query_run_with_sockets_blocked(tmp_path, monkeypatch):
    def fail_socket(*args, **kwargs):
        raise AssertionError("network egress attempted")

    monkeypatch.setattr(socket.socket, "connect", fail_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)

    encoder = HashingAliasEncoder()
    index = build_index(_loader(tmp_path), encoder, systems=["icd10cm"])
    assert index is not None
    candidates = index.query_text("type 2 diabetes", encoder, k=3)
    assert candidates[0].code == "E11.9"


def test_wheel_ships_no_vocabulary_content():
    import openmed.clinical.grounding as grounding

    package_dir = Path(grounding.__file__).resolve().parent
    vocab_suffixes = {".tsv", ".rrf", ".obo", ".csv", ".jsonl", ".txt", ".xml"}
    bundled = [
        path
        for path in package_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in vocab_suffixes
    ]
    assert bundled == []
