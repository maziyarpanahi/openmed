"""Tests for free vocabulary loading and alias indexing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    FREE_VOCAB_SYSTEMS,
    RestrictedVocabularyError,
    VocabLoader,
    VocabSource,
    VocabularyChecksumError,
)
from openmed.core.offline import OfflineModeError


def _write_fixture(path: Path, system: str) -> Path:
    target = path / f"{system}.tsv"
    target.write_text(
        "code\tpreferred_term\tsynonyms\n"
        f"{system.upper()}-001\tAspirin\tacetylsalicylic acid|ASA\n"
        f"{system.upper()}-002\tMetformin\tGlucophage\n",
        encoding="utf-8",
    )
    return target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_indexes_fixture_for_all_free_systems(tmp_path):
    registry = {}
    for system in FREE_VOCAB_SYSTEMS:
        fixture = _write_fixture(tmp_path, system)
        registry[system] = VocabSource(
            system=system,
            path=fixture,
            sha256=_sha256(fixture),
        )

    loader = VocabLoader(cache_dir=tmp_path / "cache", registry=registry)

    for system in FREE_VOCAB_SYSTEMS:
        index = loader.get_index(system)
        assert index.concept_count == 2
        assert index["aspirin"] == f"{system.upper()}-001"
        assert index.get("acetylsalicylic acid") == f"{system.upper()}-001"
        assert index.lookup("ASA").preferred_term == "Aspirin"


def test_system_aliases_resolve_to_fixture_index(tmp_path):
    fixture = _write_fixture(tmp_path, "icd10cm")
    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "icd10cm": VocabSource(
                system="icd10cm",
                path=fixture,
                sha256=_sha256(fixture),
            )
        },
    )

    assert loader.get_index("icd-10-cm")["ASA"] == "ICD10CM-001"


def test_offline_mode_blocks_empty_cache_download(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENMED_OFFLINE", "1")

    def fail_download(url: str, target: Path, timeout: float) -> None:
        raise AssertionError("download should not be attempted")

    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "rxnorm": VocabSource(
                system="rxnorm",
                url="https://example.invalid/rxnorm.tsv",
                sha256="0" * 64,
            )
        },
        downloader=fail_download,
    )

    with pytest.raises(OfflineModeError, match="vocabulary download for rxnorm"):
        loader.get_index("rxnorm")


def test_offline_mode_serves_cached_vocab_without_download(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    cached_dir = tmp_path / "cache" / "rxnorm"
    cached_dir.mkdir(parents=True)
    _write_fixture(cached_dir, "concepts")

    def fail_download(url: str, target: Path, timeout: float) -> None:
        raise AssertionError("download should not be attempted")

    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "rxnorm": VocabSource(
                system="rxnorm",
                url="https://example.invalid/rxnorm.tsv",
                sha256="0" * 64,
            )
        },
        downloader=fail_download,
    )

    assert loader.get_index("rxnorm").get("ASA") == "CONCEPTS-001"


@pytest.mark.parametrize("system", ["umls", "snomed", "snomed-ct"])
def test_restricted_vocabularies_point_to_user_key_path(system):
    loader = VocabLoader()

    with pytest.raises(RestrictedVocabularyError, match="user-key-gated"):
        loader.get_index(system)


def test_checksum_mismatch_rejects_local_source(tmp_path):
    fixture = _write_fixture(tmp_path, "mesh")
    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "mesh": VocabSource(
                system="mesh",
                path=fixture,
                sha256="0" * 64,
            )
        },
    )

    with pytest.raises(VocabularyChecksumError, match="Checksum mismatch"):
        loader.get_index("mesh")


def test_downloaded_artifact_requires_checksum(tmp_path):
    loader = VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "hpo": VocabSource(
                system="hpo",
                url="https://example.invalid/hp.obo",
            )
        },
    )

    with pytest.raises(VocabularyChecksumError, match="without a SHA-256 checksum"):
        loader.get_index("hpo")
