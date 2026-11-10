"""The reference, queries and metadata must follow the evaluated systems."""

import json

import pytest

from openmed.eval.suites.grounding_index_recall import evaluate_grounding_index_recall


class SyntheticEncoder:
    encoder_id = "synthetic-two-axis-encoder"
    dimension = 2

    def __init__(self):
        self.seen = []

    def encode(self, texts):
        self.seen.extend(texts)
        vectors = {"alpha": (1.0, 0.0), "beta": (0.0, 1.0)}
        return tuple(vectors[text] for text in texts)


@pytest.fixture
def synthetic_fixture(tmp_path):
    rows = [
        {
            "system": "icd10cm",
            "concept_id": "SYN-A",
            "canonical_term": "alpha",
            "aliases": ["alpha"],
        },
        {
            "system": "loinc",
            "concept_id": "SYN-B",
            "canonical_term": "beta",
            "aliases": ["beta"],
        },
    ]
    path = tmp_path / "synthetic.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("system", "surface"), [("icd10cm", "alpha"), ("loinc", "beta"), ("LOINC", "beta")]
)
def test_single_system_has_matching_reference_and_query_scope(
    tmp_path, synthetic_fixture, system, surface
):
    encoder = SyntheticEncoder()
    report = evaluate_grounding_index_recall(
        encoder,
        cache_dir=tmp_path / "cache",
        fixture_path=synthetic_fixture,
        systems=(system,),
        k=1,
        backend="brute",
    )
    assert report["query_count"] == 1
    assert report["record_count"] == 1
    assert report["recall_at_k"] == 1.0
    assert set(encoder.seen) == {surface}
    assert report["metadata"]["systems"] == [system.lower()]
    assert set(report["metadata"]["vocab_versions"]) == {system.upper()}


def test_empty_system_selection_has_no_queries(tmp_path, synthetic_fixture):
    encoder = SyntheticEncoder()
    report = evaluate_grounding_index_recall(
        encoder,
        cache_dir=tmp_path / "cache",
        fixture_path=synthetic_fixture,
        systems=(),
        k=1,
        backend="brute",
    )
    assert report["query_count"] == report["record_count"] == 0
    assert report["recall_at_k"] == 0.0
    assert report["metadata"]["systems"] == []
    assert encoder.seen == []


def test_all_selected_systems_retain_exact_recall(tmp_path, synthetic_fixture):
    report = evaluate_grounding_index_recall(
        SyntheticEncoder(),
        cache_dir=tmp_path / "cache",
        fixture_path=synthetic_fixture,
        systems=("icd10cm", "loinc"),
        k=1,
        backend="brute",
    )
    assert report["query_count"] == report["record_count"] == 2
    assert report["recall_at_k"] == 1.0
    assert report["metadata"]["systems"] == ["icd10cm", "loinc"]


def test_duplicate_requested_systems_do_not_duplicate_queries(
    tmp_path, synthetic_fixture
):
    report = evaluate_grounding_index_recall(
        SyntheticEncoder(),
        cache_dir=tmp_path / "cache",
        fixture_path=synthetic_fixture,
        systems=("loinc", "LOINC"),
        k=1,
        backend="brute",
    )
    assert report["query_count"] == 1
    assert report["metadata"]["systems"] == ["loinc"]


def test_adding_unselected_rows_does_not_change_recall(tmp_path, synthetic_fixture):
    kwargs = {"systems": ("icd10cm",), "k": 1, "backend": "brute"}
    first = evaluate_grounding_index_recall(
        SyntheticEncoder(),
        cache_dir=tmp_path / "cache",
        fixture_path=synthetic_fixture,
        **kwargs,
    )
    with synthetic_fixture.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n"
            + json.dumps(
                {
                    "system": "loinc",
                    "concept_id": "SYN-C",
                    "canonical_term": "beta",
                    "aliases": ["beta"],
                }
            )
        )
    second = evaluate_grounding_index_recall(
        SyntheticEncoder(),
        cache_dir=tmp_path / "cache2",
        fixture_path=synthetic_fixture,
        **kwargs,
    )
    assert (
        (first["query_count"], first["recall_at_k"])
        == (second["query_count"], second["recall_at_k"])
        == (1, 1.0)
    )


@pytest.mark.parametrize("alias", ["ICD-10-CM", " icd10cm "])
def test_fixture_system_spelling_uses_loader_normalization(
    tmp_path, synthetic_fixture, alias
):
    rows = [
        json.loads(line)
        for line in synthetic_fixture.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["system"] = alias
    synthetic_fixture.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    report = evaluate_grounding_index_recall(
        SyntheticEncoder(),
        cache_dir=tmp_path / "cache",
        fixture_path=synthetic_fixture,
        systems=("icd10cm",),
        k=1,
        backend="brute",
    )
    assert report["query_count"] == 1
    assert report["recall_at_k"] == 1.0
