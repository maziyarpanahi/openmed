"""Tests for process-level tokenizer caching."""

from __future__ import annotations

import pytest

from openmed.processing import tokenizer_cache
from openmed.processing.tokenizer_cache import (
    clear_tokenizer_cache,
    get_tokenizer,
    get_tokenizer_with_loader,
)


@pytest.fixture(autouse=True)
def clear_cache():
    clear_tokenizer_cache()
    yield
    clear_tokenizer_cache()


def _install_fake_auto_tokenizer(monkeypatch):
    class FakeAutoTokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            tokenizer = object()
            cls.calls.append((name, dict(kwargs), tokenizer))
            return tokenizer

    monkeypatch.setattr(tokenizer_cache, "AutoTokenizer", FakeAutoTokenizer)
    return FakeAutoTokenizer


def test_get_tokenizer_reuses_instance_for_same_key(monkeypatch):
    fake = _install_fake_auto_tokenizer(monkeypatch)

    first = get_tokenizer("OpenMed/test-model", revision="main", use_fast=True)
    second = get_tokenizer("OpenMed/test-model", revision="main", use_fast=True)

    assert second is first
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "OpenMed/test-model"
    assert fake.calls[0][1] == {"revision": "main", "use_fast": True}


def test_get_tokenizer_kwargs_order_does_not_change_key(monkeypatch):
    fake = _install_fake_auto_tokenizer(monkeypatch)

    first = get_tokenizer(
        "OpenMed/test-model",
        revision="main",
        cache_dir="/tmp/cache",
        use_fast=True,
    )
    second = get_tokenizer(
        "OpenMed/test-model",
        revision="main",
        use_fast=True,
        cache_dir="/tmp/cache",
    )

    assert second is first
    assert len(fake.calls) == 1


def test_get_tokenizer_distinguishes_revision_and_kwargs(monkeypatch):
    fake = _install_fake_auto_tokenizer(monkeypatch)

    default = get_tokenizer("OpenMed/test-model", revision="main", use_fast=True)
    other_revision = get_tokenizer("OpenMed/test-model", revision="v2", use_fast=True)
    other_kwargs = get_tokenizer("OpenMed/test-model", revision="main", use_fast=False)

    assert other_revision is not default
    assert other_kwargs is not default
    assert len(fake.calls) == 3


def test_clear_tokenizer_cache_empties_cache(monkeypatch):
    fake = _install_fake_auto_tokenizer(monkeypatch)

    first = get_tokenizer("OpenMed/test-model", use_fast=True)
    clear_tokenizer_cache()
    second = get_tokenizer("OpenMed/test-model", use_fast=True)

    assert second is not first
    assert len(fake.calls) == 2


def test_explicit_loader_is_invoked_once_per_distinct_key():
    calls = []

    def loader(name, **kwargs):
        tokenizer = object()
        calls.append((name, dict(kwargs), tokenizer))
        return tokenizer

    first = get_tokenizer_with_loader("OpenMed/test-model", loader, use_fast=True)
    second = get_tokenizer_with_loader("OpenMed/test-model", loader, use_fast=True)
    third = get_tokenizer_with_loader("OpenMed/test-model", loader, use_fast=False)

    assert second is first
    assert third is not first
    assert len(calls) == 2


def test_refresh_cache_reloads_and_replaces_entry():
    calls = []

    def loader(name, **kwargs):
        tokenizer = object()
        calls.append((name, dict(kwargs), tokenizer))
        return tokenizer

    first = get_tokenizer_with_loader("OpenMed/test-model", loader, use_fast=True)
    refreshed = get_tokenizer_with_loader(
        "OpenMed/test-model",
        loader,
        refresh_cache=True,
        use_fast=True,
    )
    later = get_tokenizer_with_loader("OpenMed/test-model", loader, use_fast=True)

    assert refreshed is not first
    assert later is refreshed
    assert len(calls) == 2
