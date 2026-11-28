"""Pin the reproducibility-hash vectors the Swift kit recomputes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openmed.core.model_integrity import _remote_reproducibility_hash
from scripts.manifest.generate_manifest import _reproducibility_hash

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "tests" / "fixtures" / "parity" / "model_repro_hash_v1.json"


def _cases() -> list[dict[str, Any]]:
    document = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert document["schema"] == "openmed.model_repro_hash_vectors"
    assert document["version"] == 1
    return document["cases"]


def _stamp(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def test_vectors_match_the_download_verifier() -> None:
    for case in _cases():
        remote = SimpleNamespace(
            id=case["repo_id"],
            sha=case["sha"],
            last_modified=_stamp(case["last_modified"]),
            created_at=_stamp(case["created_at"]),
            siblings=[SimpleNamespace(rfilename=name) for name in case["siblings"]],
        )

        assert _remote_reproducibility_hash(remote) == case["expected_hash"], case["id"]


def test_vectors_match_the_manifest_generator() -> None:
    for case in _cases():
        stamp = case["last_modified"] or case["created_at"]
        assert case["released"] == (stamp[:10] if stamp else None), case["id"]

        actual = _reproducibility_hash(
            case["repo_id"], case["sha"], case["released"], list(case["siblings"])
        )

        assert actual == case["expected_hash"], case["id"]


def test_vectors_cover_the_canonicalization_edges() -> None:
    cases = {case["id"]: case for case in _cases()}

    assert all(
        case["repo_id"].startswith("OpenMed/Synthetic-Repro-")
        for case in cases.values()
    )
    assert any(case["sha"] is None for case in cases.values())
    assert any(case["released"] is None for case in cases.values())
    assert any(
        case["last_modified"] is None and case["created_at"] for case in cases.values()
    )
    assert any(case["siblings"] != sorted(case["siblings"]) for case in cases.values())
    assert any(
        any(not name.isascii() for name in case["siblings"]) for case in cases.values()
    )
    base = cases["unsorted-siblings-with-commit-and-date"]
    changed = cases["changed-sibling-list-changes-the-hash"]
    assert set(changed["siblings"]) > set(base["siblings"])
    assert changed["expected_hash"] != base["expected_hash"]
