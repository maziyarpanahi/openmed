"""Focused tests for the offline terminology grounding workbench."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

import openmed
from openmed.clinical.grounding import (
    RestrictedVocabularyError,
    VocabLoader,
    VocabSource,
    ground,
)
from openmed.core.offline import OfflineModeError

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"


def test_imported_snapshot_is_verified_and_grounding_is_reproducible(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "snapshots"
    loader = VocabLoader(cache_dir=cache_dir, local_only=True)
    source_digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

    manifest = loader.import_snapshot(
        "icd10cm",
        FIXTURE,
        version="synthetic-fixture-1",
        sha256=source_digest,
    )

    assert manifest.version == "synthetic-fixture-1"
    assert manifest.sha256 == f"sha256:{source_digest}"
    assert manifest.system_uri == "http://hl7.org/fhir/sid/icd-10-cm"

    grounded = ground(
        "type 2 diabetes",
        systems=["icd10cm"],
        loader=VocabLoader(cache_dir=cache_dir, local_only=True),
        offline=True,
    )
    repeated = ground(
        "type 2 diabetes",
        systems=["icd10cm"],
        loader=VocabLoader(cache_dir=cache_dir, local_only=True),
        offline=True,
    )

    result = grounded[0]
    assert result.start == 0
    assert result.end == len("type 2 diabetes")
    assert result.code == "E11.9"
    assert result.system_uri == "http://hl7.org/fhir/sid/icd-10-cm"
    assert result.confidence == 1.0
    assert result.to_dict() == repeated[0].to_dict()
    assert result.provenance["snapshot_provenance"]["icd10cm"]["sha256"] == (
        f"sha256:{source_digest}"
    )
    audit = json.dumps(result.to_audit_dict(), sort_keys=True)
    assert "type 2 diabetes" not in audit
    assert result.to_audit_dict()["text_hash"].startswith("sha256:")
    assert (
        openmed.ground(
            "type 2 diabetes",
            systems=["icd10cm"],
            loader=VocabLoader(
                cache_dir=cache_dir,
                local_only=True,
            ),
            offline=True,
        )[0].code
        == "E11.9"
    )

    single_entity = openmed.ground(
        {"text": "type 2 diabetes", "start": 4, "end": 19},
        systems=["icd10cm"],
        loader=VocabLoader(cache_dir=cache_dir, local_only=True),
        offline=True,
    )
    assert single_entity[0].code == "E11.9"


def test_offline_grounding_does_not_attempt_a_missing_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = False

    def fail_connect(*_: object, **__: object) -> None:
        nonlocal attempted
        attempted = True
        raise AssertionError("offline grounding attempted a socket connection")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)
    loader = VocabLoader(
        cache_dir=tmp_path / "empty",
        registry={
            "icd10cm": VocabSource(
                system="icd10cm",
                url="http://example.invalid/synthetic.jsonl",
                sha256="0" * 64,
            )
        },
    )

    with pytest.raises(OfflineModeError):
        ground("type 2 diabetes", systems=["icd10cm"], loader=loader, offline=True)

    assert attempted is False


def test_restricted_request_fails_before_any_network_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = False

    def fail_connect(*_: object, **__: object) -> None:
        nonlocal attempted
        attempted = True
        raise AssertionError("restricted grounding attempted a socket connection")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)
    with pytest.raises(RestrictedVocabularyError):
        ground("synthetic finding", systems=["snomed"], offline=True)

    assert attempted is False


def test_package_content_contains_no_restricted_vocabulary_payloads() -> None:
    from scripts.release.check_license_policy import audit_restricted_vocab_data

    assert audit_restricted_vocab_data(ROOT) == []
