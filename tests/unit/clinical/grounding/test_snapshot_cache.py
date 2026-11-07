"""Focused synthetic tests for versioned terminology snapshots."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    DEFAULT_CACHE_ENV,
    SnapshotCache,
    SnapshotIntegrityError,
    SnapshotManifest,
    VersionedVocabularyLoader,
    VocabConcept,
    VocabularyIndex,
    VocabularySnapshotManifest,
    default_cache_dir,
    load_snapshot,
)
from openmed.core.offline import OfflineModeError


def test_public_snapshot_manifest_name_keeps_the_v21_cache_contract() -> None:
    from openmed.clinical.grounding.snapshot_cache import (
        SnapshotManifest as CacheSnapshotManifest,
    )
    from openmed.clinical.grounding.vocab import (
        SnapshotManifest as VocabSnapshotManifest,
    )

    assert SnapshotManifest is CacheSnapshotManifest
    assert VocabularySnapshotManifest is VocabSnapshotManifest


def _index(*, delta: bool = False) -> VocabularyIndex:
    concepts = [
        VocabConcept(
            system="icd10cm",
            code="E11.9",
            preferred_term="Type 2 diabetes mellitus",
            synonyms=("T2DM",),
            source="Patient Casey Example must never be cached",
        ),
        VocabConcept(
            system="icd10cm",
            code="I10",
            preferred_term="Essential hypertension",
            synonyms=("high blood pressure",),
        ),
    ]
    if delta:
        concepts.append(
            VocabConcept(
                system="icd10cm",
                code="E11.51",
                preferred_term="Type 2 diabetes with peripheral angiopathy",
            )
        )
    return VocabularyIndex("icd10cm", concepts)


def test_first_load_persists_and_second_load_hits_without_rebuilding(tmp_path: Path):
    cache = SnapshotCache(tmp_path / "snapshots", local_only=False)
    calls = 0

    def build() -> VocabularyIndex:
        nonlocal calls
        calls += 1
        return _index()

    first = cache.load_or_build("icd10cm", "2026-01", build)
    second = cache.load_or_build(
        "icd10cm",
        "2026-01",
        lambda: pytest.fail("cache hit must not rebuild the vocabulary"),
    )

    assert calls == 1
    assert first.hit is False
    assert second.hit is True
    assert second.lookup("T2DM").code == "E11.9"
    assert second.system_uri == "http://hl7.org/fhir/sid/icd-10-cm"
    assert second.release_version == "2026-01"
    assert cache.stats().hits == 1
    assert cache.stats().misses == 1
    assert cache.stats().writes == 1

    manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert manifest["system_uri"] == second.system_uri
    assert manifest["release_version"] == "2026-01"
    assert manifest["content_hash"] == second.content_hash
    assert "Patient Casey Example" not in second.manifest_path.read_text(
        encoding="utf-8"
    )
    assert "source" not in second.artifact_path.read_text(encoding="utf-8")


def test_snapshot_artifact_and_manifest_are_reproducible(tmp_path: Path):
    first_cache = SnapshotCache(tmp_path / "first", local_only=False)
    second_cache = SnapshotCache(tmp_path / "second", local_only=False)

    first = first_cache.load_or_build("icd10cm", "2026-01", _index)
    second = second_cache.load_or_build("icd10cm", "2026-01", _index)

    assert first.artifact_path.read_bytes() == second.artifact_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.snapshot_key == second.snapshot_key


def test_corrupt_artifact_is_rejected_and_rebuilt(tmp_path: Path):
    cache = SnapshotCache(tmp_path / "snapshots", local_only=False)
    first = cache.load_or_build("icd10cm", "2026-01", _index)
    first.artifact_path.write_text('{"corrupted": true}\n', encoding="utf-8")

    calls = 0

    def rebuild() -> VocabularyIndex:
        nonlocal calls
        calls += 1
        return _index()

    second = cache.load_or_build("icd10cm", "2026-01", rebuild)

    assert calls == 1
    assert second.hit is False
    assert second.content_hash == first.content_hash
    assert cache.stats().corruptions == 1
    assert cache.stats().writes == 2
    assert load_snapshot(second.artifact_path).content_hash == first.content_hash


def test_content_hash_pin_rebuilds_a_changed_edition(tmp_path: Path):
    cache = SnapshotCache(tmp_path / "snapshots", local_only=False)
    first = cache.load_or_build("icd10cm", "2026-01", _index)
    changed = _index(delta=True)

    second = cache.load_or_build(
        "icd10cm",
        "2026-01",
        lambda: changed,
        content_hash=changed.content_hash,
    )

    assert second.hit is False
    assert second.content_hash != first.content_hash
    assert second.concept_count == 3
    assert second.snapshot_key != first.snapshot_key


def test_offline_mode_blocks_network_inside_a_cache_miss_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    cache = SnapshotCache(tmp_path / "snapshots")

    def build() -> VocabularyIndex:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        return _index()

    with pytest.raises(OfflineModeError, match="OPENMED_OFFLINE/local_only=True"):
        cache.load_or_build("icd10cm", "2026-01", build)

    assert not (tmp_path / "snapshots").exists()


def test_restricted_snapshots_are_not_persisted_without_opt_in(tmp_path: Path):
    cache = SnapshotCache(tmp_path / "snapshots", local_only=False)
    calls = 0

    def build() -> VocabularyIndex:
        nonlocal calls
        calls += 1
        return _index()

    first = cache.load_or_build(
        "http://snomed.info/sct",
        "2026-01",
        build,
        restricted=True,
    )
    second = cache.load_or_build(
        "http://snomed.info/sct",
        "2026-01",
        build,
        restricted=True,
    )

    assert first.hit is False
    assert second.hit is False
    assert calls == 2
    assert not (tmp_path / "snapshots").exists()

    allowed = SnapshotCache(
        tmp_path / "allowed",
        allow_restricted=True,
        local_only=False,
    )
    stored = allowed.load_or_build(
        "http://snomed.info/sct",
        "2026-01",
        _index,
        restricted=True,
    )
    assert stored.artifact_path.is_file()


def test_direct_loader_rejects_a_hash_mismatched_manifest(tmp_path: Path):
    cache = SnapshotCache(tmp_path / "snapshots", local_only=False)
    snapshot = cache.load_or_build("icd10cm", "2026-01", _index)
    manifest_path = snapshot.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["content_hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SnapshotIntegrityError):
        load_snapshot(snapshot.artifact_path)


def test_clear_removes_only_snapshot_entries(tmp_path: Path):
    cache_dir = tmp_path / "snapshots"
    cache = SnapshotCache(cache_dir, local_only=False)
    cache.load_or_build("icd10cm", "2026-01", _index)
    (cache_dir / "keep-me.txt").write_text("unrelated", encoding="utf-8")

    assert cache.clear() == 1
    assert not tuple(cache_dir.glob("*/manifest.json"))
    assert (cache_dir / "keep-me.txt").read_text(encoding="utf-8") == "unrelated"


def test_versioned_loader_reuses_its_pinned_snapshot(tmp_path: Path):
    calls = 0

    def build() -> VocabularyIndex:
        nonlocal calls
        calls += 1
        return _index()

    loader = VersionedVocabularyLoader(
        "icd10cm",
        "2026-01",
        build,
        cache_dir=tmp_path / "snapshots",
    )

    assert loader.get_index().get("T2DM") == "E11.9"
    assert loader.get_index().get("high blood pressure") == "I10"
    assert calls == 1


def test_default_cache_dir_honors_terminology_cache_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setenv(DEFAULT_CACHE_ENV, str(tmp_path / "configured"))

    assert default_cache_dir() == tmp_path / "configured"
