"""Tests for versioned AEAD envelopes bound to key ids and record identity."""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from openmed.core.aead_envelope import (
    ENVELOPE_ALGORITHM,
    ENVELOPE_MAGIC,
    ENVELOPE_VERSION,
    EncryptionDependencyError,
    EnvelopeAuthenticationError,
    EnvelopeCipher,
    EnvelopeFormatError,
    EnvelopeVersionError,
    envelope_from_text,
    envelope_to_text,
    is_envelope,
    is_text_envelope,
    read_envelope_header,
)
from openmed.core.key_provider import (
    DATA_KEY_BYTES,
    DataKey,
    InMemoryKeyProvider,
    KeyRetiredError,
    KeyState,
    KeyUnavailableError,
)

PURPOSE = "journey-store"
KEY_ONE = bytes(range(DATA_KEY_BYTES))
KEY_TWO = bytes(reversed(range(DATA_KEY_BYTES)))
PLAINTEXT = b"ZQSYNTHETICSENTINEL93 synthetic note"
CONTEXT = {"record": "journey_payload", "table": "facts", "identity": ["fact_1"]}


def _cipher(material: bytes = KEY_ONE, key_id: str = "key-one") -> EnvelopeCipher:
    provider = InMemoryKeyProvider()
    provider.add_key(key_id, material, purpose=PURPOSE, state=KeyState.ACTIVE)
    return EnvelopeCipher(provider, purpose=PURPOSE)


def test_round_trip_header_and_randomized_nonce() -> None:
    cipher = _cipher()

    first = cipher.seal(PLAINTEXT, context=CONTEXT)
    second = cipher.seal(PLAINTEXT, context=CONTEXT)

    assert first != second
    assert PLAINTEXT not in first
    assert first.startswith(ENVELOPE_MAGIC + bytes((ENVELOPE_VERSION,)))
    assert is_envelope(first)
    header = read_envelope_header(first)
    assert (header.version, header.algorithm, header.key_id) == (
        ENVELOPE_VERSION,
        ENVELOPE_ALGORITHM,
        "key-one",
    )
    assert cipher.open(first, context=CONTEXT) == PLAINTEXT
    assert cipher.open(second, context=dict(reversed(CONTEXT.items()))) == PLAINTEXT


def test_text_encoding_round_trips_and_rejects_other_text() -> None:
    cipher = _cipher()

    text = cipher.seal_text(PLAINTEXT, context=CONTEXT)

    assert is_text_envelope(text)
    assert text.isascii()
    assert envelope_from_text(text) == envelope_from_text(
        envelope_to_text(envelope_from_text(text))
    )
    assert cipher.open_text(text, context=CONTEXT) == PLAINTEXT
    for invalid in ('{"plain": true}', "omae:not base64!"):
        with pytest.raises(EnvelopeFormatError):
            envelope_from_text(invalid)


@pytest.mark.parametrize(
    "context",
    [
        {**CONTEXT, "identity": ["fact_2"]},
        {**CONTEXT, "table": "conflicts"},
        {"record": "journey_payload"},
    ],
)
def test_context_mismatch_fails_authentication(context: dict[str, Any]) -> None:
    cipher = _cipher()
    envelope = cipher.seal(PLAINTEXT, context=CONTEXT)

    with pytest.raises(EnvelopeAuthenticationError) as error:
        cipher.open(envelope, context=context)
    assert error.value.code == "envelope_authentication_failed"


def test_tampering_wrong_material_and_other_purpose_fail_closed() -> None:
    cipher = _cipher()
    envelope = bytearray(cipher.seal(PLAINTEXT, context=CONTEXT))

    tampered = bytes(envelope[:-1] + bytes((envelope[-1] ^ 1,)))
    with pytest.raises(EnvelopeAuthenticationError):
        cipher.open(tampered, context=CONTEXT)
    with pytest.raises(EnvelopeAuthenticationError):
        _cipher(KEY_TWO).open(bytes(envelope), context=CONTEXT)

    # The same key material under another purpose derives another cipher key.
    provider = InMemoryKeyProvider()
    provider.add_key("key-one", KEY_ONE, purpose="backup", state=KeyState.ACTIVE)
    with pytest.raises(EnvelopeAuthenticationError):
        EnvelopeCipher(provider, purpose="backup").open(
            bytes(envelope), context=CONTEXT
        )


def test_header_key_identifier_is_authenticated() -> None:
    provider = InMemoryKeyProvider()
    provider.add_key("key-one", KEY_ONE, purpose=PURPOSE, state=KeyState.ACTIVE)
    provider.add_key("key-two", KEY_ONE, purpose=PURPOSE)
    cipher = EnvelopeCipher(provider, purpose=PURPOSE)
    envelope = cipher.seal(PLAINTEXT, context=CONTEXT)

    relabelled = envelope.replace(b"key-one", b"key-two", 1)

    with pytest.raises(EnvelopeAuthenticationError):
        cipher.open(relabelled, context=CONTEXT)


def test_malformed_and_unsupported_envelopes_are_typed() -> None:
    envelope = _cipher().seal(PLAINTEXT, context=CONTEXT)

    with pytest.raises(EnvelopeFormatError):
        read_envelope_header(b"OMAE")
    with pytest.raises(EnvelopeFormatError):
        read_envelope_header(envelope[:30])
    with pytest.raises(EnvelopeFormatError):
        read_envelope_header("text")  # type: ignore[arg-type]
    with pytest.raises(EnvelopeVersionError):
        read_envelope_header(ENVELOPE_MAGIC + bytes((2,)) + envelope[5:])
    with pytest.raises(EnvelopeVersionError):
        read_envelope_header(ENVELOPE_MAGIC + bytes((1, 9)) + envelope[6:])


def test_key_state_is_enforced_on_open_and_seal() -> None:
    provider = InMemoryKeyProvider()
    provider.add_key("key-one", KEY_ONE, purpose=PURPOSE, state=KeyState.ACTIVE)
    cipher = EnvelopeCipher(provider, purpose=PURPOSE)
    envelope = cipher.seal(PLAINTEXT, context=CONTEXT)

    provider.rotate("key-two", KEY_TWO, purpose=PURPOSE)
    assert cipher.open(envelope, context=CONTEXT) == PLAINTEXT
    provider.set_state("key-one", KeyState.RETIRED, purpose=PURPOSE)
    with pytest.raises(KeyRetiredError):
        cipher.open(envelope, context=CONTEXT)
    retired = DataKey("key-one", PURPOSE, KeyState.RETIRED, KEY_ONE)
    with pytest.raises(KeyRetiredError):
        cipher.seal(PLAINTEXT, context=CONTEXT, key=retired)


class _FailingProvider:
    def active_key(self, purpose: str) -> DataKey:
        raise RuntimeError("ZQSYNTHETICSENTINEL93 key-one secret path")

    def resolve_key(self, key_id: str, *, purpose: str) -> DataKey:
        raise OSError("ZQSYNTHETICSENTINEL93")


def test_foreign_provider_failures_become_value_free_key_errors() -> None:
    cipher = EnvelopeCipher(_FailingProvider(), purpose=PURPOSE)
    envelope = _cipher().seal(PLAINTEXT, context=CONTEXT)

    with pytest.raises(KeyUnavailableError) as sealing:
        cipher.seal(PLAINTEXT, context=CONTEXT)
    with pytest.raises(KeyUnavailableError) as opening:
        cipher.open(envelope, context=CONTEXT)
    for error in (sealing.value, opening.value):
        assert "ZQSYNTHETICSENTINEL93" not in str(error)
        assert error.__cause__ is None


def test_invalid_context_and_purpose_are_rejected() -> None:
    cipher = _cipher()

    for context in (
        {"Bad Key": "x"},
        {"flag": True},
        {"blob": b"bytes"},
        {"nested": [["x"]]},
        {"text": "x" * 513},
    ):
        with pytest.raises(ValueError):
            cipher.seal(PLAINTEXT, context=context)
    with pytest.raises(ValueError):
        EnvelopeCipher(InMemoryKeyProvider(), purpose="Not Controlled")


def test_missing_cryptography_dependency_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cipher = _cipher()
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("cryptography"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(EncryptionDependencyError) as error:
        cipher.seal(PLAINTEXT, context=CONTEXT)
    assert error.value.code == "encryption_dependency_unavailable"
