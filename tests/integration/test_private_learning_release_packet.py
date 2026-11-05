"""Bind a private-learning release packet to a real local Git commit."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openmed.eval.suites.private_learning_release import (
    NOT_READY,
    PRIVATE_LEARNING_RELEASE_CATALOGS,
    PRIVATE_LEARNING_RELEASE_EVIDENCE_SCHEMA_VERSION,
    PRIVATE_LEARNING_RELEASE_MANIFEST_SCHEMA_VERSION,
    READY,
    PrivateLearningReleasePacket,
    evaluate_private_learning_release,
    sign_private_learning_evidence,
)

_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_KEY = b"integration-evidence-key-0123456789abcdef"
_SIGNING_KEY = b"integration-packet-key-0123456789abcdefgh"
_NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Synthetic",
        "GIT_AUTHOR_EMAIL": "synthetic@example.invalid",
        "GIT_COMMITTER_NAME": "Synthetic",
        "GIT_COMMITTER_EMAIL": "synthetic@example.invalid",
    }
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _inputs(commit: str) -> tuple[dict, list[dict]]:
    catalog = PRIVATE_LEARNING_RELEASE_CATALOGS["v3.3"]
    manifest = {
        "schema_version": PRIVATE_LEARNING_RELEASE_MANIFEST_SCHEMA_VERSION,
        "release": {"release_id": "v3.3", "git_commit": commit},
        "policy": {"max_evidence_age_seconds": 86_400},
        "items": [
            {
                "item_id": spec.item_id,
                "priority": spec.priority,
                "privacy_invariant": spec.privacy_invariant,
                "non_compensable": True,
                "required_schema_version": f"openmed.release.{spec.item_id}.v1",
                "required_evidence_digest": _digest(spec.item_id),
            }
            for spec in catalog
        ],
        "moved_work": [],
    }
    evidence = [
        sign_private_learning_evidence(
            {
                "schema_version": PRIVATE_LEARNING_RELEASE_EVIDENCE_SCHEMA_VERSION,
                "item_id": spec.item_id,
                "evidence_schema_version": f"openmed.release.{spec.item_id}.v1",
                "evidence_digest": _digest(spec.item_id),
                "source_commit": commit,
                "generated_at": "2026-11-01T10:00:00Z",
                "status": "passed",
            },
            _EVIDENCE_KEY,
        )
        for spec in catalog
    ]
    return manifest, evidence


@pytest.mark.integration
def test_packet_binds_checkout_commit_and_verifies_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README.txt").write_text("synthetic\n", encoding="utf-8")
    _git(repo, "add", "README.txt")
    _git(repo, "commit", "-q", "-m", "synthetic")
    commit = _git(repo, "rev-parse", "HEAD")
    manifest, evidence = _inputs(commit)

    packet = evaluate_private_learning_release(
        manifest,
        evidence,
        source_commit=commit,
        evidence_key=_EVIDENCE_KEY,
        signing_key=_SIGNING_KEY,
        clock=lambda: _NOW,
    )
    assert packet.decision == READY

    packet_path = tmp_path / "packet.json"
    key_path = tmp_path / "packet.key"
    packet_path.write_text(packet.to_json(), encoding="utf-8")
    key_path.write_bytes(_SIGNING_KEY)
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from openmed.eval.suites.private_learning_release import "
        "PrivateLearningReleasePacket\n"
        "packet = PrivateLearningReleasePacket.from_json("
        "Path(sys.argv[1]).read_text('utf-8'))\n"
        "ok = packet.verify(Path(sys.argv[2]).read_bytes())\n"
        "print(packet.decision if ok else 'UNVERIFIED')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(packet_path), str(key_path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
    )
    assert completed.stdout.strip() == READY

    # A new commit on the checkout makes the same manifest stale for release.
    (repo / "README.txt").write_text("changed\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "changed")
    moved_head = _git(repo, "rev-parse", "HEAD")
    rebound = evaluate_private_learning_release(
        manifest,
        evidence,
        source_commit=moved_head,
        evidence_key=_EVIDENCE_KEY,
        signing_key=_SIGNING_KEY,
        clock=lambda: _NOW,
    )
    assert rebound.decision == NOT_READY
    assert rebound.to_dict()["reason_codes"] == ["release_commit_mismatch"]
    assert PrivateLearningReleasePacket.from_json(rebound.to_json()).verify(
        _SIGNING_KEY
    )
