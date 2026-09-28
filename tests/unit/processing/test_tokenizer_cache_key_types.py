"""Verify cache isolation through the explicit-loader public API."""

from collections import OrderedDict
from pathlib import Path

import pytest

from openmed.processing import tokenizer_cache as cache


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear_tokenizer_cache()
    yield
    cache.clear_tokenizer_cache()


@pytest.mark.parametrize(
    "left,right",
    [
        ({"a": 1}, [("a", 1)]),
        ([], ()),
        ([], set()),
        ((), {}),
        ([1, 2], {1, 2}),
        ({1, 2}, frozenset({1, 2})),
        (True, 1),
        (1, 1.0),
        (False, 0),
        ({1: "a"}, {"1": "a"}),
        ({"nested": []}, {"nested": ()}),
    ],
)
def test_different_argument_types_do_not_share(left, right):
    calls = []

    def loader(name, **kwargs):
        calls.append(kwargs)
        return object()

    first = cache.get_tokenizer_with_loader("synthetic", loader, option=left)
    second = cache.get_tokenizer_with_loader("synthetic", loader, option=right)
    assert first is not second
    assert len(calls) == 2


@pytest.mark.parametrize(
    "left,right",
    [
        ({"a": 1, "b": [2]}, OrderedDict([("b", [2]), ("a", 1)])),
        ({1, "two"}, {"two", 1}),
        (frozenset({1, 2}), frozenset({2, 1})),
        (Path("synthetic/path"), str(Path("synthetic/path"))),
        (None, None),
    ],
)
def test_equivalent_options_still_hit(left, right):
    calls = []

    def loader(name, **kwargs):
        calls.append(kwargs)
        return object()

    first = cache.get_tokenizer_with_loader("synthetic", loader, option=left)
    assert cache.get_tokenizer_with_loader("synthetic", loader, option=right) is first
    assert len(calls) == 1


def test_revision_refresh_and_loader_scope_remain_distinct():
    def loader(name, **kwargs):
        return object()

    first = cache.get_tokenizer_with_loader("synthetic", loader, revision="one")
    assert cache.get_tokenizer_with_loader("synthetic", loader, revision="one") is first
    assert (
        cache.get_tokenizer_with_loader("synthetic", loader, revision="two")
        is not first
    )
    refreshed = cache.get_tokenizer_with_loader(
        "synthetic", loader, revision="one", refresh_cache=True
    )
    assert refreshed is not first
    assert (
        cache.get_tokenizer_with_loader("synthetic", loader, revision="one")
        is refreshed
    )
    assert (
        cache.get_tokenizer_with_loader(
            "synthetic", lambda *a, **k: object(), revision="one"
        )
        is not refreshed
    )


def test_cache_stays_bounded(monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_TOKENIZER_CACHE_SIZE", 2)

    def loader(name, **kwargs):
        return object()

    first = cache.get_tokenizer_with_loader("one", loader)
    cache.get_tokenizer_with_loader("two", loader)
    cache.get_tokenizer_with_loader("three", loader)
    assert len(cache._TOKENIZER_CACHE) == 2
    assert cache.get_tokenizer_with_loader("one", loader) is not first
