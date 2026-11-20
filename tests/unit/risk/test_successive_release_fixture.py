"""Per-release k checks for the synthetic successive-release fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.risk import kanon_report

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "risk"
    / "successive_releases_v1.jsonl"
)


def _load() -> tuple[dict, dict[str, list[dict]]]:
    rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = rows[0]
    assert meta["kind"] == "meta"
    releases: dict[str, list[dict]] = {}
    for row in rows[1:]:
        assert row["kind"] == "record"
        assert row["synthetic"] is True
        releases.setdefault(row["release"], []).append(row)
    return meta, releases


def test_fixture_is_synthetic_with_two_releases_of_20_to_40_rows() -> None:
    meta, releases = _load()

    assert meta["version"] == 1
    assert meta["synthetic"] is True
    assert sorted(releases) == ["release_1", "release_2"]
    for rows in releases.values():
        assert 20 <= len(rows) <= 40


@pytest.mark.parametrize("release", ["release_1", "release_2"])
def test_each_release_alone_meets_its_committed_k(release: str) -> None:
    meta, releases = _load()

    report = kanon_report(
        releases[release], quasi_identifiers=meta["quasi_identifiers"]
    )

    assert report["k"] == meta["expected"]["k"][release]
