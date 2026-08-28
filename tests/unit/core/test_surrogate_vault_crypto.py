"""Focused tests for encrypted reversible surrogate mappings."""

from __future__ import annotations

import base64
import json
import socket
from pathlib import Path

import pytest

from openmed.core.surrogate_vault_crypto import (
    ENCRYPTION_SCHEME,
    KEY_BYTES,
    SCHEMA_VERSION,
    SurrogateVaultCrypto,
    SurrogateVaultCryptoError,
    SurrogateVaultKeyError,
    SurrogateVaultPayloadError,
    decrypt_mapping,
    encrypt_mapping,
    load_mapping,
    save_mapping,
)

KEY = bytes(range(KEY_BYTES))
OTHER_KEY = bytes(reversed(range(KEY_BYTES)))
MAPPING = {
    "<SURROGATE_A>": "<SYNTHETIC_SOURCE_A>",
    "<SURROGATE_B>": "<SYNTHETIC_SOURCE_B>",
}


def test_encryption_is_deterministic_authenticated_and_round_trips() -> None:
    crypto = SurrogateVaultCrypto(KEY)

    first = crypto.encrypt(MAPPING)
    second = crypto.encrypt(dict(reversed(tuple(MAPPING.items()))))

    assert first == second
    assert crypto.decrypt(first) == MAPPING
    assert ENCRYPTION_SCHEME.encode("ascii") in first
    assert str(SCHEMA_VERSION).encode("ascii") in first
    for value in MAPPING:
        assert value.encode("utf-8") not in first
    for value in MAPPING.values():
        assert value.encode("utf-8") not in first
    assert "SYNTHETIC_SOURCE" not in repr(crypto)


def test_module_helpers_require_a_caller_key_and_round_trip() -> None:
    serialized = encrypt_mapping(MAPPING, KEY)

    assert decrypt_mapping(serialized, KEY) == MAPPING

    with pytest.raises(TypeError, match="key"):
        SurrogateVaultCrypto()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "key",
    [None, b"", b"short", b"x" * (KEY_BYTES - 1), b"x" * (KEY_BYTES + 1), "text"],
)
def test_missing_or_malformed_key_is_rejected_without_echoing_input(key) -> None:
    with pytest.raises(SurrogateVaultKeyError) as raised:
        SurrogateVaultCrypto(key)

    assert "SYNTHETIC" not in str(raised.value)
    assert "key" in str(raised.value)


def test_tampering_and_wrong_keys_fail_without_mapping_values() -> None:
    serialized = encrypt_mapping(MAPPING, KEY)
    envelope = json.loads(serialized)
    ciphertext = bytearray(base64.b64decode(envelope["ciphertext"]))
    ciphertext[-1] ^= 1
    envelope["ciphertext"] = base64.b64encode(ciphertext).decode("ascii")

    for candidate in (json.dumps(envelope), serialized):
        with pytest.raises(
            SurrogateVaultPayloadError, match="authentication"
        ) as raised:
            SurrogateVaultCrypto(OTHER_KEY).decrypt(candidate)
        assert all(value not in str(raised.value) for value in MAPPING.values())


def test_malformed_mapping_and_payload_errors_are_phi_safe() -> None:
    crypto = SurrogateVaultCrypto(KEY)
    with pytest.raises(SurrogateVaultCryptoError) as raised:
        crypto.encrypt({"<SYNTHETIC_SOURCE_A>": None})  # type: ignore[dict-item]
    assert "SYNTHETIC_SOURCE_A" not in str(raised.value)

    raw_payload = json.dumps({"source": "<SYNTHETIC_SOURCE_A>"})
    with pytest.raises(SurrogateVaultPayloadError) as raised:
        crypto.decrypt(raw_payload)
    assert "SYNTHETIC_SOURCE_A" not in str(raised.value)


def test_file_helpers_store_only_ciphertext_and_clean_up_temporary_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "surrogate-mapping.json"
    crypto = SurrogateVaultCrypto(KEY)

    crypto.write(path, MAPPING)

    persisted = path.read_bytes()
    assert all(value.encode("utf-8") not in persisted for value in MAPPING)
    assert not list(tmp_path.glob("*.tmp"))
    assert path.stat().st_mode & 0o077 == 0
    assert crypto.read(path) == MAPPING
    assert load_mapping(path, KEY) == MAPPING

    alternate = tmp_path / "alternate.json"
    save_mapping(alternate, MAPPING, KEY)
    assert load_mapping(alternate, KEY) == MAPPING


def test_crypto_path_is_local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("network access is not part of surrogate encryption")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    assert SurrogateVaultCrypto(KEY).decrypt(encrypt_mapping(MAPPING, KEY)) == MAPPING


def test_malformed_file_does_not_echo_plaintext(tmp_path: Path) -> None:
    path = tmp_path / "surrogate-mapping.json"
    path.write_text("<SYNTHETIC_SOURCE_A>", encoding="utf-8")

    with pytest.raises(SurrogateVaultPayloadError) as raised:
        SurrogateVaultCrypto(KEY).read(path)

    assert "SYNTHETIC_SOURCE_A" not in str(raised.value)
