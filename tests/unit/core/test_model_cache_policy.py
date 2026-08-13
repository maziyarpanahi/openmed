"""Focused tests for the local model-cache quota and eviction policy."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

import openmed
from openmed.core import ModelCachePolicy as ExportedModelCachePolicy
from openmed.core.model_cache_policy import (
    CacheIntegrityError,
    CacheOwnershipError,
    ModelCachePolicy,
    sha256_path,
    verify_artifact_checksum,
)


def _write_artifact(root: Path, name: str, payload: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_policy_is_publicly_exported() -> None:
    assert ExportedModelCachePolicy is ModelCachePolicy
    assert openmed.ModelCachePolicy is ModelCachePolicy


def test_reuse_verifies_checksum_and_exposes_only_hashes(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    artifact = _write_artifact(cache_dir, "model-a.bin", b"synthetic model A")
    expected = sha256_path(artifact)
    policy = ModelCachePolicy(cache_dir, quota_bytes=128)

    registered = policy.register_artifact(
        artifact,
        expected_sha256=expected,
        last_accessed_ns=10,
    )
    assert registered.sha256 == expected
    assert registered.verified is True
    assert verify_artifact_checksum(artifact, expected) == expected
    assert (
        policy.reuse_artifact(artifact, expected_sha256=expected) == artifact.resolve()
    )

    artifact.write_bytes(b"tampered synthetic model A")
    with pytest.raises(CacheIntegrityError) as raised:
        policy.reuse_artifact(artifact)

    assert raised.value.path_hash == registered.path_hash
    assert "model-a.bin" not in str(raised.value)
    assert str(tmp_path) not in str(raised.value)
    assert not policy.is_valid(artifact)


def test_policy_is_local_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model cache policy must not use a network socket")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    artifact = _write_artifact(tmp_path / "cache", "model.bin", b"offline bytes")
    policy = ModelCachePolicy(tmp_path / "cache", quota_bytes=64)

    policy.register_artifact(artifact, last_accessed_ns=1)
    plan = policy.plan_eviction()
    assert plan.quota_satisfied
    assert policy.reuse_artifact(artifact) == artifact.resolve()


def test_lru_plan_is_deterministic_and_respects_pins(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    oldest = _write_artifact(cache_dir, "model-old.bin", b"a" * 4)
    middle = _write_artifact(cache_dir, "model-middle.bin", b"b" * 3)
    newest = _write_artifact(cache_dir, "model-new.bin", b"c" * 2)
    pinned = _write_artifact(cache_dir, "model-pinned.bin", b"p" * 5)
    policy = ModelCachePolicy(cache_dir, quota_bytes=7, pinned_artifacts=[pinned])

    oldest_summary = policy.register_artifact(oldest, last_accessed_ns=1)
    middle_summary = policy.register_artifact(middle, last_accessed_ns=2)
    policy.register_artifact(newest, last_accessed_ns=3)
    pinned_summary = policy.register_artifact(pinned, last_accessed_ns=0)

    plan = policy.plan_eviction()
    repeated = ModelCachePolicy(cache_dir, quota_bytes=7).plan_eviction()

    assert plan.current_bytes == 14
    assert plan.bytes_to_free == 7
    assert [item.path_hash for item in plan.evictions] == [
        oldest_summary.path_hash,
        middle_summary.path_hash,
    ]
    assert pinned_summary.path_hash not in {item.path_hash for item in plan.evictions}
    assert plan.remaining_bytes == 7
    assert plan.quota_satisfied
    assert plan.to_dict() == repeated.to_dict()


def test_apply_eviction_removes_only_owned_unchanged_artifacts(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    managed = _write_artifact(cache_dir, "managed.bin", b"managed bytes")
    protected = _write_artifact(cache_dir, "protected.bin", b"protected bytes")
    unowned = _write_artifact(cache_dir, "unowned.bin", b"unowned bytes")
    policy = ModelCachePolicy(cache_dir, quota_bytes=1)

    managed_summary = policy.register_artifact(managed, last_accessed_ns=1)
    policy.register_artifact(protected, pinned=True, last_accessed_ns=2)
    plan = policy.plan_eviction()
    result = policy.apply_eviction(plan)

    assert result.evicted_path_hashes == (managed_summary.path_hash,)
    assert result.skipped_count == 0
    assert not managed.exists()
    assert protected.exists()
    assert unowned.exists()
    assert result.remaining_bytes == protected.stat().st_size


def test_directory_artifact_and_manifest_are_safe_for_reports(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    snapshot = cache_dir / "snapshot"
    _write_artifact(snapshot, "config.json", b'{"synthetic": true}')
    _write_artifact(snapshot, "weights.bin", b"synthetic weights")
    unowned = _write_artifact(cache_dir, "keep.bin", b"keep")
    policy = ModelCachePolicy(cache_dir, quota_bytes=0)

    policy.register_artifact(snapshot, last_accessed_ns=1)
    plan = policy.plan_eviction()
    report = json.dumps(plan.to_dict(), sort_keys=True)
    assert str(cache_dir) not in report
    assert "snapshot" not in report

    result = policy.apply_eviction(plan)
    assert result.quota_satisfied
    assert not snapshot.exists()
    assert unowned.exists()
    manifest = json.loads(policy.manifest_path.read_text(encoding="utf-8"))
    assert str(cache_dir) not in json.dumps(manifest)


def test_artifact_outside_cache_cannot_be_registered(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    outside = _write_artifact(tmp_path, "outside.bin", b"outside")
    policy = ModelCachePolicy(cache_dir, quota_bytes=32)

    with pytest.raises(CacheOwnershipError, match="inside"):
        policy.register_artifact(outside)


def test_pinned_bytes_can_block_quota_without_deleting_the_pin(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    pinned = _write_artifact(cache_dir, "pinned.bin", b"p" * 8)
    policy = ModelCachePolicy(cache_dir, quota_bytes=2)
    policy.register_artifact(pinned, pinned=True, last_accessed_ns=1)

    plan = policy.plan_eviction()
    result = policy.apply_eviction(plan)

    assert plan.evictions == ()
    assert not plan.quota_satisfied
    assert result.skipped_count == 0
    assert not result.quota_satisfied
    assert pinned.exists()
