"""Invalid executor counts fail before a document iterator is consumed."""

import pytest

from openmed.processing.distributed import (
    assign_document_shard,
    plan_document_shards,
    stable_document_hash,
)

BAD_COUNTS = [True, False, 1.5, float("nan"), float("inf"), "2", 0, -1]


@pytest.mark.parametrize("count", BAD_COUNTS)
def test_assign_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="shard_count"):
        assign_document_shard("synthetic", count)


@pytest.mark.parametrize("count", BAD_COUNTS)
@pytest.mark.parametrize("field", ["shard_count", "worker_count"])
def test_plan_rejects_counts_before_consuming_documents(count, field):
    consumed = []

    def documents():
        consumed.append(True)
        yield {"id": "synthetic"}

    options = {"shard_count": 2, field: count}
    with pytest.raises(ValueError, match=field):
        plan_document_shards(documents(), **options)
    assert consumed == []


@pytest.mark.parametrize("shards", [1, 2, 7])
def test_integer_assignments_and_plan_fingerprints_are_unchanged(shards):
    docs = [{"id": "alpha"}, {"id": "beta"}]
    first = plan_document_shards(docs, shard_count=shards, worker_count=1)
    second = plan_document_shards(reversed(docs), shard_count=shards, worker_count=3)
    assert first == second
    for doc in docs:
        actual = assign_document_shard(doc["id"], shards)
        assert type(actual) is int
        assert actual == int(stable_document_hash(doc["id"]), 16) % shards
        assert first.document_to_shard()[doc["id"]] == actual


def test_unspecified_worker_count_and_empty_plan_still_work():
    plan = plan_document_shards([], shard_count=2)
    assert plan.document_count == 0
    assert len(plan.shards) == 2
