"""Tests for the pluggable surrogate-key provider contract."""

from __future__ import annotations

import json

import pytest

from openmed.core.surrogate_key_provider import (
    InMemorySurrogateKeyProvider,
    MissingSurrogateKeyCapabilityError,
    SurrogateKeyCapabilities,
    SurrogateKeyCapability,
    SurrogateKeyProvider,
    SurrogateKeyProviderError,
    require_surrogate_key_capabilities,
    validate_surrogate_key_provider,
)


def test_local_provider_is_deterministic_and_declares_local_capabilities():
    first = InMemorySurrogateKeyProvider(seed="synthetic-seed")
    second = InMemorySurrogateKeyProvider(seed="synthetic-seed")

    assert isinstance(first, SurrogateKeyProvider)
    assert first.capabilities == SurrogateKeyCapabilities.all()
    assert first.capability_metadata == {
        "provider": "LocalSyntheticSurrogateKeyProvider",
        "deterministic": True,
        "network_required": False,
        "capabilities": {
            "lookup": True,
            "rotation": True,
            "scope": True,
            "destruction": True,
        },
    }

    first_key = first.lookup("synthetic-scope")
    second_key = second.lookup("synthetic-scope")

    assert first_key is not None
    assert second_key is not None
    assert first_key.material == second_key.material
    assert first_key.as_metadata() == second_key.as_metadata()
    assert first_key.material.hex() not in repr(first_key)
    assert first_key.material.hex() not in json.dumps(first_key.as_metadata())


def test_capability_metadata_fails_closed_before_lifecycle_mutation():
    provider = InMemorySurrogateKeyProvider(
        capabilities=SurrogateKeyCapabilities(lookup=True, scope=True),
    )

    with pytest.raises(MissingSurrogateKeyCapabilityError) as exc_info:
        provider.rotate("synthetic-scope")
    assert exc_info.value.missing == ("rotation",)
    assert provider.scope("synthetic-scope").active_version is None

    with pytest.raises(MissingSurrogateKeyCapabilityError):
        provider.destroy("synthetic-scope")


def test_required_capability_helper_rejects_missing_metadata_without_fallback():
    provider = InMemorySurrogateKeyProvider(
        capabilities=SurrogateKeyCapabilities(lookup=True),
    )

    with pytest.raises(MissingSurrogateKeyCapabilityError, match="rotation"):
        require_surrogate_key_capabilities(provider, ("lookup", "rotation"))

    validate_surrogate_key_provider(provider, (SurrogateKeyCapability.LOOKUP,))

    class IncompleteProvider:
        pass

    with pytest.raises(SurrogateKeyProviderError, match="capability metadata"):
        require_surrogate_key_capabilities(IncompleteProvider(), ("lookup",))


def test_rotation_preserves_old_key_until_explicit_destruction():
    provider = InMemorySurrogateKeyProvider(seed="synthetic-seed")
    first = provider.lookup("synthetic-scope")
    assert first is not None

    rotated = provider.rotate("synthetic-scope")

    assert rotated.version == 2
    assert rotated.material != first.material
    assert provider.lookup("synthetic-scope") == rotated
    assert provider.lookup("synthetic-scope", first.key_id) == first
    assert provider.scope("synthetic-scope").key_versions == (1, 2)

    provider.destroy("synthetic-scope", first.key_id)
    assert provider.lookup("synthetic-scope", first.key_id) is None
    assert provider.lookup("synthetic-scope") == rotated
    assert provider.scope("synthetic-scope").key_versions == (2,)


def test_destroyed_key_version_is_not_deterministically_recreated():
    provider = InMemorySurrogateKeyProvider(seed="synthetic-seed")
    first = provider.lookup("synthetic-scope")
    assert first is not None

    provider.destroy("synthetic-scope", first.key_id)
    replacement = provider.lookup("synthetic-scope")

    assert replacement is not None
    assert replacement.version == 2
    assert replacement.material != first.material
    assert provider.lookup("synthetic-scope", first.key_id) is None


def test_scope_destruction_is_terminal_and_metadata_is_privacy_safe():
    provider = InMemorySurrogateKeyProvider(seed="synthetic-seed")
    key = provider.lookup("synthetic-scope")
    assert key is not None

    provider.destroy_scope("synthetic-scope")
    state = provider.scope("synthetic-scope")

    assert state.destroyed is True
    assert state.active_key_id is None
    assert state.key_versions == ()
    assert key.material.hex() not in json.dumps(state.as_dict())
    assert provider.lookup("synthetic-scope") is None
    with pytest.raises(SurrogateKeyProviderError, match="destroyed"):
        provider.rotate("synthetic-scope")


def test_capability_metadata_serializes_without_sensitive_material():
    provider = InMemorySurrogateKeyProvider(seed="synthetic-seed")
    key = provider.lookup("synthetic-scope")
    assert key is not None

    payload = {
        "key": key.as_metadata(),
        "scope": provider.scope("synthetic-scope").as_dict(),
        "capabilities": provider.capabilities.as_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True)

    assert '"material":' not in encoded
    assert key.material.hex() not in encoded
    assert provider.capabilities.supports("key_lookup")
    assert provider.capabilities.missing(("lookup", "destruction")) == ()
