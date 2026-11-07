"""Tests for the caller-supplied key-provider contract."""

from __future__ import annotations

import copy
import pickle

import pytest

from openmed.core.key_provider import (
    DATA_KEY_BYTES,
    DataKey,
    EncryptionError,
    InMemoryKeyProvider,
    KeyMaterialError,
    KeyProvider,
    KeyRetiredError,
    KeyRevokedError,
    KeyState,
    KeyUnavailableError,
    require_decryption_key,
    require_encryption_key,
    validate_key_provider,
)

PURPOSE = "journey-store"
KEY_ONE = bytes(range(DATA_KEY_BYTES))
KEY_TWO = bytes(reversed(range(DATA_KEY_BYTES)))


def _provider() -> InMemoryKeyProvider:
    provider = InMemoryKeyProvider()
    provider.add_key("key-one", KEY_ONE, purpose=PURPOSE, state=KeyState.ACTIVE)
    return provider


def test_in_memory_provider_implements_the_protocol() -> None:
    provider = _provider()

    assert isinstance(provider, KeyProvider)
    assert validate_key_provider(provider) is provider
    with pytest.raises(TypeError):
        validate_key_provider(object())


def test_rotation_demotes_the_previous_key_to_decrypt_only() -> None:
    provider = _provider()

    rotated = provider.rotate("key-two", KEY_TWO, purpose=PURPOSE)

    assert rotated.key_id == "key-two"
    assert provider.active_key(PURPOSE).key_id == "key-two"
    assert provider.resolve_key("key-one", purpose=PURPOSE).state is (
        KeyState.DECRYPT_ONLY
    )
    assert provider.key_states(PURPOSE) == {
        "key-one": KeyState.DECRYPT_ONLY,
        "key-two": KeyState.ACTIVE,
    }


def test_retired_revoked_and_unknown_keys_fail_closed() -> None:
    provider = _provider()
    provider.rotate("key-two", KEY_TWO, purpose=PURPOSE)

    provider.set_state("key-one", KeyState.RETIRED, purpose=PURPOSE)
    with pytest.raises(KeyRetiredError) as retired:
        provider.resolve_key("key-one", purpose=PURPOSE)
    assert retired.value.code == "key_retired"

    provider.set_state("key-two", KeyState.REVOKED, purpose=PURPOSE)
    with pytest.raises(KeyRevokedError):
        provider.resolve_key("key-two", purpose=PURPOSE)
    with pytest.raises(KeyUnavailableError):
        provider.active_key(PURPOSE)
    with pytest.raises(KeyRevokedError):
        provider.set_state("key-two", KeyState.ACTIVE, purpose=PURPOSE)

    with pytest.raises(KeyUnavailableError):
        provider.resolve_key("key-unknown", purpose=PURPOSE)
    with pytest.raises(KeyUnavailableError):
        provider.resolve_key("key-one", purpose="backup")


def test_retired_key_can_be_restored_but_revoked_material_is_dropped() -> None:
    provider = _provider()
    provider.rotate("key-two", KEY_TWO, purpose=PURPOSE)
    provider.set_state("key-one", KeyState.RETIRED, purpose=PURPOSE)

    provider.set_state("key-one", KeyState.DECRYPT_ONLY, purpose=PURPOSE)

    assert provider.resolve_key("key-one", purpose=PURPOSE).material == KEY_ONE
    provider.set_state("key-one", KeyState.REVOKED, purpose=PURPOSE)
    assert provider._keys["key-one"].material is None


def test_key_material_is_validated_and_never_represented() -> None:
    with pytest.raises(KeyMaterialError):
        DataKey("key-one", PURPOSE, KeyState.ACTIVE, b"short")
    with pytest.raises(KeyMaterialError):
        DataKey("1-invalid", PURPOSE, KeyState.ACTIVE, KEY_ONE)
    with pytest.raises(KeyMaterialError):
        InMemoryKeyProvider().add_key("key-one", "text", purpose=PURPOSE)  # type: ignore[arg-type]
    provider = _provider()
    with pytest.raises(ValueError):
        provider.add_key("key-one", KEY_TWO, purpose=PURPOSE)

    key = provider.active_key(PURPOSE)
    rendered = repr(key) + repr(provider) + str(key)
    assert KEY_ONE.hex() not in rendered
    assert repr(KEY_ONE) not in rendered
    with pytest.raises(TypeError):
        pickle.dumps(key)
    with pytest.raises(TypeError):
        pickle.dumps(provider)
    with pytest.raises(TypeError):
        copy.copy(key)
    with pytest.raises(TypeError):
        copy.deepcopy(key)


def test_generated_provider_holds_one_random_active_key() -> None:
    first = InMemoryKeyProvider.generate(PURPOSE)
    second = InMemoryKeyProvider.generate(PURPOSE)

    assert first.active_key(PURPOSE).key_id == "ephemeral-key-1"
    assert first.active_key(PURPOSE).material != second.active_key(PURPOSE).material


def test_requirement_helpers_enforce_state_purpose_and_identifier() -> None:
    active = DataKey("key-one", PURPOSE, KeyState.ACTIVE, KEY_ONE)
    decrypt_only = DataKey("key-one", PURPOSE, KeyState.DECRYPT_ONLY, KEY_ONE)

    assert require_encryption_key(active, purpose=PURPOSE) is active
    assert require_decryption_key(decrypt_only, purpose=PURPOSE, key_id="key-one")
    with pytest.raises(KeyUnavailableError):
        require_encryption_key(decrypt_only, purpose=PURPOSE)
    with pytest.raises(KeyUnavailableError):
        require_encryption_key(active, purpose="backup")
    with pytest.raises(KeyUnavailableError):
        require_decryption_key(active, purpose=PURPOSE, key_id="key-two")
    with pytest.raises(KeyUnavailableError):
        require_encryption_key(object(), purpose=PURPOSE)
    with pytest.raises(KeyRevokedError):
        require_decryption_key(
            DataKey("key-one", PURPOSE, KeyState.REVOKED, KEY_ONE),
            purpose=PURPOSE,
            key_id="key-one",
        )


def test_error_messages_are_fixed_and_value_free() -> None:
    error = KeyUnavailableError()

    assert isinstance(error, EncryptionError)
    assert error.code == "key_unavailable"
    assert str(error) == "encryption key is unavailable (key_unavailable)"
    assert EncryptionError("Not A Controlled Code!").code == "encryption_failed"
