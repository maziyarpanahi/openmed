import pytest

from openmed.training.active_learning import (
    ActiveLearningCandidate,
    ActiveLearningQueue,
)


def candidate():
    return ActiveLearningCandidate(
        0, 3, ("NAME",), "synthetic", 1.0, "test", record_id="synthetic-1"
    )


def fail_append(event):
    raise OSError("synthetic disk failure")


def test_failed_add_does_not_poison_retry_or_persistence(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    queue = ActiveLearningQueue(path)
    item = candidate()
    with monkeypatch.context() as patch:
        patch.setattr(queue, "_append_event", fail_append)
        with pytest.raises(OSError, match="synthetic disk failure"):
            queue.add(item)
    assert queue.pending == ()
    assert queue.labeled_hashes == frozenset()
    restored_after_failure = ActiveLearningQueue(path)
    assert restored_after_failure.pending == ()
    assert restored_after_failure.labeled_hashes == frozenset()
    assert queue.add(item) is True
    assert ActiveLearningQueue(path).pending == (item,)


def test_failed_mark_keeps_pending_item_and_can_be_retried(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    queue = ActiveLearningQueue(path)
    item = candidate()
    queue.add(item)
    with monkeypatch.context() as patch:
        patch.setattr(queue, "_append_event", fail_append)
        with pytest.raises(OSError):
            queue.mark_labeled(item)
    assert queue.pending == (item,)
    assert queue.labeled_hashes == frozenset()
    assert ActiveLearningQueue(path).pending == (item,)
    assert queue.mark_labeled(item) == item.span_hash
    restored = ActiveLearningQueue(path)
    assert restored.pending == ()
    assert restored.labeled_hashes == frozenset({item.span_hash})


def test_real_event_log_open_failure_leaves_memory_unchanged(tmp_path):
    # Opening a directory as a file fails without patching persistence internals.
    path = tmp_path / "events.jsonl"
    queue = ActiveLearningQueue(path)
    path.mkdir()
    with pytest.raises(OSError):
        queue.add(candidate())
    assert queue.pending == ()


def test_in_memory_queue_and_duplicate_handling_unchanged():
    queue = ActiveLearningQueue(None)
    item = candidate()
    assert queue.add(item)
    assert not queue.add(item)
    queue.mark_labeled(item)
    assert not queue.add(item)
    assert queue.pending == ()
