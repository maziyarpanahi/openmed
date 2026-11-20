import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from openmed.risk.release import AnonymityPolicy, anonymize_release
from openmed.risk.release_history import ReleaseHistory


@pytest.fixture
def release():
    rows = [{"unit": f"private-key-{i}", "region": "private-cell"} for i in range(3)]
    return anonymize_release(
        rows,
        AnonymityPolicy(quasi_identifiers=("region",), privacy_unit="unit", target_k=2),
    )


def test_round_trip_two_lineages_with_no_values(tmp_path, release):
    path = tmp_path / "history.db"
    history = ReleaseHistory(path, "private-lineage")
    first = history.append(release, released_on=date(2026, 1, 1))
    second = history.append(release, released_on=date(2026, 2, 1))
    assert first != second
    loaded = ReleaseHistory(path, "private-lineage")
    assert loaded.verify()
    a, b = loaded.entries()
    assert a["dataset_digest"] == release.released_dataset_digest
    assert a["schema_digest"] == release.released_schema_digest
    assert a["quasi_identifiers"] == ["region"]
    assert a["suppressed_rows"] == 0 and a["achieved_k"] == 3
    assert b["released_on"] == "2026-02-01"
    assert ReleaseHistory(path, "other").entries() == ()
    assert b"private-" not in path.read_bytes()
    assert "unit" not in json.dumps(a)


@pytest.mark.parametrize(
    "sql",
    [
        """UPDATE release_entries SET record = replace(record, '2026-01-01', '2026-01-02')""",
        "DELETE FROM release_entries WHERE sequence=0",
    ],
)
def test_edits_and_deletions_fail_before_append(tmp_path, release, sql):
    path = tmp_path / "history.db"
    history = ReleaseHistory(path, "test")
    history.append(release, released_on=date(2026, 1, 1))
    with sqlite3.connect(path) as db:
        db.execute(sql)
    assert not history.verify()
    with pytest.raises(ValueError):
        history.append(release, released_on=date(2026, 2, 1))


def test_concurrent_appends_retain_every_release(tmp_path, release):
    history = ReleaseHistory(tmp_path / "history.db", "test")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda day: history.append(release, released_on=date(2026, 1, day)),
                [1, 2, 3, 4],
            )
        )
    assert history.verify()
    assert len(history.entries()) == 4
