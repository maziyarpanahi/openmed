"""Tests for the redistributable vocabulary loader registry."""

from __future__ import annotations

import pytest

from openmed.clinical.grounding import (
    RestrictedVocabularyLoaderError,
    VocabularyLoader,
    VocabularyLoaderRegistry,
    VocabularyRegistryError,
    validate_vocabulary_loader,
)
from openmed.clinical.grounding.matcher import VocabularyTerms
from openmed.clinical.grounding.registry import InvalidVocabularyLoaderError

SYSTEM_URI = "https://example.org/fhir/CodeSystem/fake-free"


class FakeFreeVocabularyLoader:
    """Minimal conforming loader over a synthetic in-memory snapshot."""

    system_uri = SYSTEM_URI
    redistributable = True

    def load(self) -> VocabularyTerms:
        return {
            "Synthetic finding": {
                "code": "SYN-001",
                "display": "Synthetic finding",
                "provenance": "synthetic-permissive",
            }
        }


class FakeRestrictedVocabularyLoader(FakeFreeVocabularyLoader):
    system_uri = "http://restricted.example/CodeSystem/private"
    redistributable = False


def test_fake_free_loader_conforms_and_builds_matcher():
    loader = FakeFreeVocabularyLoader()
    registry = VocabularyLoaderRegistry()

    assert isinstance(loader, VocabularyLoader)
    assert validate_vocabulary_loader(loader) is loader

    registry.register(loader)
    matcher = registry.matcher(SYSTEM_URI)
    matches = matcher.lookup(" synthetic-finding ")

    assert registry.get(SYSTEM_URI) is loader
    assert registry.available() == (SYSTEM_URI,)
    assert matches[0].code == "SYN-001"
    assert matches[0].metadata["provenance"] == "synthetic-permissive"


def test_registry_refuses_restricted_license_loader_with_clear_error():
    registry = VocabularyLoaderRegistry()

    with pytest.raises(
        RestrictedVocabularyLoaderError,
        match="redistributable must be true.*out of process",
    ):
        registry.register(FakeRestrictedVocabularyLoader())

    assert len(registry) == 0


def test_registry_refuses_explicit_restricted_license_flag():
    class FlaggedRestrictedLoader(FakeFreeVocabularyLoader):
        restricted_license = True

    with pytest.raises(
        RestrictedVocabularyLoaderError,
        match="flagged as restricted-license.*out of process",
    ):
        VocabularyLoaderRegistry().register(FlaggedRestrictedLoader())


def test_registry_requires_explicit_boolean_license_declaration():
    class MissingLicenseFlag:
        system_uri = SYSTEM_URI

        def load(self):
            return {}

    with pytest.raises(InvalidVocabularyLoaderError, match="boolean redistributable"):
        VocabularyLoaderRegistry().register(MissingLicenseFlag())


def test_registry_is_keyed_by_system_uri_and_rejects_accidental_replacement():
    registry = VocabularyLoaderRegistry()
    first = FakeFreeVocabularyLoader()
    second = FakeFreeVocabularyLoader()

    registry.register(first)

    assert SYSTEM_URI in registry
    with pytest.raises(VocabularyRegistryError, match="already registered"):
        registry.register(second)

    registry.register(second, replace=True)
    assert registry.get(SYSTEM_URI) is second
