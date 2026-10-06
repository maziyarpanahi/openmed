"""Caller-governed synthetic git checkout, artifacts and gate evidence."""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

from openmed.eval.agent_release_candidate import SOURCE_SCHEMA
from tests.fixtures.eval.agent_release import passing_evidence

_KEY = b"synthetic-release-signing-key-0001"


def _git(root, *args):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Release Fixture",
            "GIT_AUTHOR_EMAIL": "release@example.invalid",
            "GIT_COMMITTER_NAME": "Release Fixture",
            "GIT_COMMITTER_EMAIL": "release@example.invalid",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
        },
    ).stdout.strip()


def build_candidate_inputs(tmp_path):
    """Build only synthetic local artifacts and aggregate gate evidence."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    (root / "source.txt").write_text("synthetic source\n")
    _git(root, "add", "source.txt")
    _git(root, "commit", "--no-gpg-sign", "-m", "Add synthetic source")
    sha = _git(root, "rev-parse", "HEAD")
    envelope = {
        "schema_version": SOURCE_SCHEMA,
        "source_sha": sha,
        "metrics": [item.to_dict() for item in passing_evidence()],
    }
    paths = {}
    for name in ("wheel", "sdist", "tool_catalog", "policy"):
        path = tmp_path / name
        path.write_bytes(b"synthetic " + name.encode())
        paths[name] = path
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(envelope))
    return SimpleNamespace(
        root=root,
        sha=sha,
        envelope=envelope,
        evidence=evidence,
        params={
            "repo_root": root,
            "source_sha": sha,
            **paths,
            "evidence_files": [evidence],
            "signing_key": _KEY,
        },
        tmp=tmp_path,
    )
