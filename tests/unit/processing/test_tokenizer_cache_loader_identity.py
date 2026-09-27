"""Loader isolation and lifetime regressions for the shared tokenizer cache."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial

import pytest

from openmed.processing import tokenizer_cache


@pytest.fixture(autouse=True)
def clear_cache():
    tokenizer_cache.clear_tokenizer_cache()
    yield
    tokenizer_cache.clear_tokenizer_cache()


def load(loader, **kwargs):
    return tokenizer_cache.get_tokenizer_with_loader(
        "synthetic/model", loader, **kwargs
    )


def test_distinct_functions_do_not_share_a_tokenizer():
    first_value, second_value = object(), object()

    def first(name, **kwargs):
        return first_value

    def second(name, **kwargs):
        return second_value

    assert load(first) is first_value
    assert load(second) is second_value
    assert load(first) is first_value
    assert load(second) is second_value


@dataclass
class EqualLoader:
    """Unhashable callable with equality that does not identify its tokenizer."""

    name: str = "same"
    calls: int = field(default=0, compare=False)
    tokenizer: object = field(default_factory=object, compare=False)

    def __call__(self, name, **kwargs):
        self.calls += 1
        return self.tokenizer

    def from_pretrained(self, name, **kwargs):
        return self(name, **kwargs)


@pytest.mark.parametrize("bound", [False, True])
def test_equal_unhashable_owners_stay_separate(bound):
    first, second = EqualLoader(), EqualLoader()
    assert first == second
    first_loader = first.from_pretrained if bound else first
    second_loader = second.from_pretrained if bound else second

    assert load(first_loader) is first.tokenizer
    assert load(second_loader) is second.tokenizer
    assert load(first_loader) is first.tokenizer
    assert (first.calls, second.calls) == (1, 1)


def test_repeated_bound_method_access_is_cached():
    loader = EqualLoader()
    assert loader.from_pretrained is not loader.from_pretrained
    assert load(loader.from_pretrained) is load(loader.from_pretrained)
    assert loader.calls == 1


def test_classmethods_are_stable_but_subclasses_are_isolated():
    class First:
        calls = 0
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            cls.calls += 1
            return cls.tokenizer

    class Second(First):
        calls = 0
        tokenizer = object()

    assert load(First.from_pretrained) is First.tokenizer
    assert load(First.from_pretrained) is First.tokenizer
    assert load(Second.from_pretrained) is Second.tokenizer
    assert (First.calls, Second.calls) == (1, 1)


def test_different_methods_on_one_owner_are_isolated():
    class Loaders:
        def first(self, name):
            return "first"

        def second(self, name):
            return "second"

    loaders = Loaders()
    assert load(loaders.first) == "first"
    assert load(loaders.second) == "second"


def test_partial_loaders_keep_their_configuration():
    def loader(name, *, kind):
        return kind

    first = partial(loader, kind="first")
    second = partial(loader, kind="second")
    assert load(first) == "first"
    assert load(second) == "second"
    assert load(first) == "first"


def test_refresh_does_not_replace_another_loaders_entry():
    first_calls, second_calls = [], []

    def first(name):
        first_calls.append(object())
        return first_calls[-1]

    def second(name):
        second_calls.append(object())
        return second_calls[-1]

    first_original = load(first)
    second_original = load(second)
    refreshed = load(first, refresh_cache=True)
    assert refreshed is not first_original
    assert load(first) is refreshed
    assert load(second) is second_original
    assert (len(first_calls), len(second_calls)) == (2, 1)


def test_cached_other_loader_does_not_hide_a_loader_failure():
    good = EqualLoader()
    failure = RuntimeError("synthetic load failure")

    def broken(name):
        raise failure

    original = load(good)
    with pytest.raises(RuntimeError) as caught:
        load(broken)
    assert caught.value is failure
    assert load(good) is original
    assert good.calls == 1


def test_loader_entries_respect_lru_capacity(monkeypatch):
    monkeypatch.setattr(tokenizer_cache, "DEFAULT_TOKENIZER_CACHE_SIZE", 2)
    first, second, third = EqualLoader(), EqualLoader(), EqualLoader()
    load(first)
    load(second)
    load(first)  # Second, not first, is now the least recently used entry.
    load(third)
    load(first)
    load(second)
    assert (first.calls, second.calls, third.calls) == (1, 2, 1)
    assert len(tokenizer_cache._TOKENIZER_CACHE) == 2


@pytest.mark.parametrize("evict", [False, True])
def test_loader_owner_is_released_with_its_cache_entry(monkeypatch, evict):
    monkeypatch.setattr(tokenizer_cache, "DEFAULT_TOKENIZER_CACHE_SIZE", 1)
    owner = EqualLoader()
    reference = weakref.ref(owner)
    load(owner.from_pretrained)
    del owner
    gc.collect()
    assert reference() is not None  # Prevent object-id reuse while cached.

    if evict:
        load(EqualLoader())
    else:
        tokenizer_cache.clear_tokenizer_cache()
    gc.collect()
    assert reference() is None


def test_concurrent_bound_method_calls_still_load_once():
    owner = EqualLoader()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: load(owner.from_pretrained), range(12)))
    assert all(result is owner.tokenizer for result in results)
    assert owner.calls == 1


def test_native_bound_method_access_preserves_cache_hits():
    original, replacement = object(), object()
    owner = {"synthetic/model": original}
    assert load(owner.get) is original
    owner["synthetic/model"] = replacement
    assert load(owner.get) is original
    assert load(owner.get, refresh_cache=True) is replacement


def test_native_bound_methods_on_different_owners_are_isolated():
    first_value, second_value = object(), object()
    first = {"synthetic/model": first_value}
    second = {"synthetic/model": second_value}
    assert load(first.get) is first_value
    assert load(second.get) is second_value
    assert load(first.get) is first_value
