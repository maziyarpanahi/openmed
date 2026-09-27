"""Deterministic thread handoff at the result-cache selection boundary."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import openmed.core.result_cache as result_cache


def test_each_caller_receives_the_cache_selected_under_its_lock(monkeypatch):
    selected = threading.Event()
    replaced = threading.Event()
    first_thread = threading.get_ident()

    class HandoffLock:
        """Release a real lock, then schedule a competing capacity selection."""

        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()
            if threading.get_ident() == first_thread:
                selected.set()
                assert replaced.wait(3.0), "competing caller did not finish"

    monkeypatch.setattr(result_cache, "RESULT_CACHE", None)
    monkeypatch.setattr(result_cache, "_CACHE_LOCK", HandoffLock())

    def competing_call():
        try:
            assert selected.wait(3.0), "first caller did not release its lock"
            return result_cache.get_result_cache(3)
        finally:
            replaced.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        other = executor.submit(competing_call)
        try:
            first = result_cache.get_result_cache(2)
        finally:
            selected.set()
        second = other.result(timeout=3.0)
    assert first.max_entries == 2
    assert second.max_entries == 3
    assert first is not second
    assert result_cache.RESULT_CACHE is second
    for key in range(3):
        first.set(key, key)
    assert len(first) == 2
    assert first.get(0) is None


def test_same_capacity_reuses_cache_and_contents(monkeypatch):
    monkeypatch.setattr(result_cache, "RESULT_CACHE", None)
    first = result_cache.get_result_cache(2)
    first.set("synthetic", 1)
    second = result_cache.get_result_cache(2)
    assert second is first
    assert second.get("synthetic") == 1


@pytest.mark.parametrize("capacity", [0, 1, 128])
def test_requested_capacity_is_unchanged_without_contention(monkeypatch, capacity):
    monkeypatch.setattr(result_cache, "RESULT_CACHE", None)
    cache = result_cache.get_result_cache(capacity)
    assert cache.max_entries == capacity


def test_replacement_does_not_mutate_previously_returned_cache(monkeypatch):
    monkeypatch.setattr(result_cache, "RESULT_CACHE", None)
    first = result_cache.get_result_cache(2)
    first.set("synthetic", 1)
    second = result_cache.get_result_cache(3)
    assert second is not first
    assert first.get("synthetic") == 1
    assert second.get("synthetic") is None
