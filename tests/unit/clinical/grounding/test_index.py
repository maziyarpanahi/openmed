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
from openmed.clinical.normalization import (
    ConceptNormalizationCache,
    RankedCandidateCache,
    SyntheticTerminologyBackend,
)
from openmed.eval.suites.grounding_index_recall import (
    evaluate_grounding_index_recall,
)

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


def _fixture_with_delta(tmp_path: Path) -> Path:
    fixture = tmp_path / "v2.jsonl"
    fixture.write_text(
        _FIXTURE.read_text(encoding="utf-8")
        + json.dumps(
            {
                "concept_id": "E11.51",
                "system": "icd10cm",
                "canonical_term": (
                    "Type 2 diabetes mellitus with diabetic peripheral angiopathy"
                ),
                "aliases": ["type 2 diabetes with peripheral angiopathy"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return fixture


class _RecordingEncoder:
    def __init__(self) -> None:
        self._inner = HashingAliasEncoder()
        self.batches: list[tuple[str, ...]] = []

    @property
    def encoder_id(self):
        return self._inner.encoder_id

    @property
    def dimension(self):
        return self._inner.dimension

    def encode(self, surfaces):
        self.batches.append(tuple(surfaces))
        return self._inner.encode(surfaces)


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


def test_hnsw_backend_matches_brute_force_reference_recall_at_10(tmp_path):
    """The optional ANN path itself must satisfy the issue's recall floor."""

    pytest.importorskip("hnswlib")

    report = evaluate_grounding_index_recall(
        HashingAliasEncoder(),
        cache_dir=tmp_path,
        systems=list(_SYSTEMS),
        k=10,
        backend="hnsw",
    )

    assert report["backend"] == "hnsw"
    assert report["recall_at_k"] >= 0.95


def test_incremental_hnsw_update_replaces_only_changed_ann_shard(tmp_path):
    pytest.importorskip("hnswlib")

    encoder = _RecordingEncoder()
    cache_dir = tmp_path / "index-cache"
    initial = build_or_load_index(
        _loader(tmp_path / "v1"),
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm", "rxnorm"],
        backend="hnsw",
    )
    assert initial is not None
    first_manifest = json.loads(
        (cache_dir / "alias_index.json").read_text(encoding="utf-8")
    )
    first_icd_ann = cache_dir / first_manifest["shards"]["ICD10CM"]["ann_file"]
    first_rxnorm_ann = cache_dir / first_manifest["shards"]["RXNORM"]["ann_file"]
    rxnorm_ann_bytes = first_rxnorm_ann.read_bytes()
    rxnorm_ann_mtime = first_rxnorm_ann.stat().st_mtime_ns

    updated = build_or_load_index(
        _loader(tmp_path / "v2", path=_fixture_with_delta(tmp_path)),
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm", "rxnorm"],
        backend="hnsw",
    )
    assert updated is not None
    updated_manifest = json.loads(
        (cache_dir / "alias_index.json").read_text(encoding="utf-8")
    )
    updated_icd_ann = cache_dir / updated_manifest["shards"]["ICD10CM"]["ann_file"]
    updated_rxnorm_ann = cache_dir / updated_manifest["shards"]["RXNORM"]["ann_file"]

    assert updated.update_summary.reused_shards == ("RXNORM",)
    assert updated.update_summary.rebuilt_shards == ("ICD10CM",)
    assert updated_icd_ann != first_icd_ann
    assert updated_icd_ann.is_file()
    assert not first_icd_ann.exists()
    assert updated_rxnorm_ann == first_rxnorm_ann
    assert updated_rxnorm_ann.read_bytes() == rxnorm_ann_bytes
    assert updated_rxnorm_ann.stat().st_mtime_ns == rxnorm_ann_mtime


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
    second_fixture = _fixture_with_delta(tmp_path)
    loader_v2 = _loader(tmp_path / "v2", path=second_fixture)
    index_v2 = build_or_load_index(
        loader_v2, encoder, cache_dir=cache_dir, systems=["icd10cm"]
    )

    assert index_v2 is not None
    assert index_v2.index_key != index_v1.index_key
    assert index_v2.vocab_versions != index_v1.vocab_versions
    # The stale payload was overwritten, not silently reused.
    assert load_index(cache_dir).index_key == index_v2.index_key


def test_version_drift_evicts_index_bound_dependent_caches(tmp_path):
    encoder = HashingAliasEncoder()
    ranked_cache = RankedCandidateCache()
    normalization_cache = ConceptNormalizationCache()
    terminology_backend = SyntheticTerminologyBackend()
    cache_dir = tmp_path / "index-cache"
    first_fixture = tmp_path / "v1.jsonl"
    first_fixture.write_text(_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    first = build_or_load_index(
        _loader(tmp_path / "v1", path=first_fixture),
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm"],
        dependent_caches=[normalization_cache, ranked_cache],
    )
    assert first is not None
    assert ranked_cache.index_key == first.index_key
    ranked_cache.set("type 2 diabetes", "v1", ("stale",))
    normalization_cache.set("type 2 diabetes", terminology_backend, ("stale",))
    assert ranked_cache.stats().size == 1
    assert normalization_cache.stats().size == 1

    second = build_or_load_index(
        _loader(tmp_path / "v2", path=_fixture_with_delta(tmp_path)),
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm"],
        dependent_caches=[normalization_cache, ranked_cache],
    )

    assert second is not None
    assert second.index_key != first.index_key
    assert ranked_cache.index_key == second.index_key
    assert ranked_cache.invalidation_count == 1
    assert ranked_cache.stats().size == 0
    assert normalization_cache.index_key == second.index_key
    assert normalization_cache.invalidation_count == 1
    assert normalization_cache.stats().size == 0

    ranked_cache.set("type 2 diabetes", "v2", ("fresh",))
    build_or_load_index(
        _loader(tmp_path / "v2-again", path=_fixture_with_delta(tmp_path)),
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm"],
        dependent_caches=[normalization_cache, ranked_cache],
    )
    assert ranked_cache.invalidation_count == 1
    assert ranked_cache.stats().size == 1


def test_incremental_delta_rebuilds_only_changed_shard(tmp_path):
    encoder = _RecordingEncoder()
    cache_dir = tmp_path / "index-cache"
    first_fixture = tmp_path / "v1.jsonl"
    first_fixture.write_text(_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    first = build_or_load_index(
        _loader(tmp_path / "v1", path=first_fixture),
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm", "rxnorm"],
        backend="brute",
    )
    assert first is not None
    assert first.update_summary.rebuilt_shards == ("ICD10CM", "RXNORM")
    assert len(encoder.batches) == 2

    first_manifest = json.loads(
        (cache_dir / "alias_index.json").read_text(encoding="utf-8")
    )
    rxnorm_path = cache_dir / first_manifest["shards"]["RXNORM"]["file"]
    rxnorm_bytes = rxnorm_path.read_bytes()
    rxnorm_mtime = rxnorm_path.stat().st_mtime_ns

    delta_fixture = _fixture_with_delta(tmp_path)
    incremental = build_or_load_index(
        _loader(tmp_path / "v2", path=delta_fixture),
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm", "rxnorm"],
        backend="brute",
    )
    assert incremental is not None
    assert incremental.update_summary.reused_shards == ("RXNORM",)
    assert incremental.update_summary.rebuilt_shards == ("ICD10CM",)
    assert incremental.update_summary.removed_shards == ()
    assert len(encoder.batches) == 3
    assert "type 2 diabetes with peripheral angiopathy" in encoder.batches[-1]
    assert rxnorm_path.read_bytes() == rxnorm_bytes
    assert rxnorm_path.stat().st_mtime_ns == rxnorm_mtime

    full = build_index(
        _loader(tmp_path / "full", path=delta_fixture),
        HashingAliasEncoder(),
        systems=["icd10cm", "rxnorm"],
        backend="brute",
    )
    assert full is not None
    for surface in ("type 2 diabetes", "acetaminophen", "peripheral angiopathy"):
        (vector,) = encoder.encode([surface])
        assert [repr(candidate) for candidate in incremental.query(vector, 10)] == [
            repr(candidate) for candidate in full.query(vector, 10)
        ]


def test_incremental_remove_reuses_remaining_shard_and_prunes_file(tmp_path):
    encoder = _RecordingEncoder()
    cache_dir = tmp_path / "index-cache"
    loader = _loader(tmp_path)
    initial = build_or_load_index(
        loader,
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm", "rxnorm"],
        backend="brute",
    )
    assert initial is not None
    manifest = json.loads((cache_dir / "alias_index.json").read_text(encoding="utf-8"))
    removed_path = cache_dir / manifest["shards"]["RXNORM"]["file"]
    batches_after_build = len(encoder.batches)

    updated = build_or_load_index(
        loader,
        encoder,
        cache_dir=cache_dir,
        systems=["icd10cm"],
        backend="brute",
    )

    assert updated is not None
    assert updated.systems == ("ICD10CM",)
    assert updated.update_summary.reused_shards == ("ICD10CM",)
    assert updated.update_summary.rebuilt_shards == ()
    assert updated.update_summary.removed_shards == ("RXNORM",)
    assert len(encoder.batches) == batches_after_build
    assert not removed_path.exists()


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

    ranked_cache = RankedCandidateCache()
    ranked_cache.bind_index("grounding-index:configured")
    ranked_cache.set("type 2 diabetes", "v1", ("dense",))
    assert (
        build_or_load_index(
            _loader(tmp_path),
            None,
            cache_dir=tmp_path / "index-cache",
            systems=["icd10cm"],
            dependent_caches=[ranked_cache],
        )
        is None
    )
    assert ranked_cache.index_key == "grounding-index:none"
    assert ranked_cache.stats().size == 0

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

    assert provenance["shards"]["ICD10CM"]["record_count"] > 0
    assert (directory / payload["shards"]["ICD10CM"]["file"]).is_file()

    restored = AliasEmbeddingIndex.from_payload(payload, directory=directory)
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
