"""Check the shared catalog filter vectors that the Android catalog runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "tests" / "fixtures" / "parity" / "catalog_filters_v1.json"
ROW_FIELDS = (
    "repo_id",
    "formats",
    "tier",
    "param_count",
    "languages",
    "license",
    "reproducibility_hash",
)
QUERY_KEYS = {"format", "tier", "max_param_count", "language", "license"}


def _vectors() -> dict[str, Any]:
    document = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert document["schema"] == "openmed.model_catalog_filters"
    assert document["version"] == 1
    assert document["synthetic_only"] is True
    return document


def _equal(actual: str | None, expected: str | None) -> bool:
    return expected is None or (
        actual is not None and actual.lower() == expected.lower()
    )


def _reference_filter(rows: list[dict[str, Any]], query: dict[str, Any]) -> list[str]:
    """The documented Android ``ModelCatalog.filter`` rules, in catalog order."""
    max_params = query.get("max_param_count")
    return [
        row["repo_id"]
        for row in rows
        if (
            query.get("format") is None
            or any(_equal(value, query["format"]) for value in row["formats"])
        )
        and _equal(row["tier"], query.get("tier"))
        and (
            max_params is None
            or (row["param_count"] is not None and row["param_count"] <= max_params)
        )
        and (
            query.get("language") is None
            or any(_equal(value, query["language"]) for value in row["languages"])
        )
        and _equal(row["license"], query.get("license"))
    ]


def test_rows_are_synthetic_unique_catalog_rows() -> None:
    rows = _vectors()["rows"]
    ids = [row["repo_id"] for row in rows]

    assert len(ids) == len(set(ids))
    for row in rows:
        assert tuple(row) == ROW_FIELDS
        assert row["repo_id"].startswith("Synthetic/")
        assert row["reproducibility_hash"].startswith("sha256:")
        # ASCII tokens keep case folding identical in Kotlin, Swift and JS.
        tokens = [*row["formats"], *row["languages"], row["tier"], row["license"]]
        assert all(token is None or token.isascii() for token in tokens)


def test_every_case_matches_the_documented_rules() -> None:
    document = _vectors()
    rows = document["rows"]

    for case in document["cases"]:
        assert set(case["query"]) <= QUERY_KEYS, case["id"]
        actual = _reference_filter(rows, case["query"])
        assert actual == case["expected_repo_ids"], case["id"]


def test_vectors_cover_the_required_edge_cases() -> None:
    document = _vectors()
    rows = document["rows"]
    queries = [case["query"] for case in document["cases"]]
    known_formats = {value.lower() for row in rows for value in row["formats"]}

    assert any(row["tier"] is None for row in rows)
    assert any(row["param_count"] is None for row in rows)
    assert any(row["license"] is None for row in rows)
    assert any(
        "format" in query and query["format"].lower() not in known_formats
        for query in queries
    )
    assert any(
        value != value.lower() and value != value.upper()
        for query in queries
        for value in query.values()
        if isinstance(value, str)
    )
    # A prefix of a listed format must not match: equality, not prefix.
    assert any(
        case["query"].get("format")
        and case["expected_repo_ids"] == []
        and any(
            value.lower().startswith(case["query"]["format"].lower())
            for row in rows
            for value in row["formats"]
        )
        for case in document["cases"]
    )
    assert {key for query in queries for key in query} == QUERY_KEYS


def test_android_ci_runs_the_vectors() -> None:
    workflow = (ROOT / ".github" / "workflows" / "android-ci.yml").read_text("utf-8")
    test = (
        ROOT
        / "android/openmedkit/src/test/kotlin/com/openmed/openmedkit/parity"
        / "CatalogFilterParityTest.kt"
    )

    assert test.is_file()
    assert "--tests '*CatalogFilterParityTest'" in workflow
    assert workflow.count("tests/fixtures/parity/catalog_filters_v1.json") == 2


def test_lookups_match_repo_ids_exactly() -> None:
    document = _vectors()
    ids = {row["repo_id"] for row in document["rows"]}

    assert any(not lookup["found"] for lookup in document["lookups"])
    for lookup in document["lookups"]:
        assert (lookup["repo_id"] in ids) is lookup["found"], lookup["repo_id"]
