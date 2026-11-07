"""Tests for webhook signing-key generations, rotation, and rollback (#3573)."""

from __future__ import annotations

import base64
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from openmed.service import _state_files
from openmed.service.signing_keys import (
    KEY_STATE_ACTIVE,
    KEY_STATE_RETIRED,
    KEY_STATE_RETIRING,
    KEYRING_FILE_ENV_VAR,
    KEYRING_FORMAT,
    MAX_KEY_OVERLAP_SECONDS,
    MAX_RETIRED_KEYS,
    SIGNING_KEY_ID_ENV_VAR,
    SIGNING_SECRET_ENV_VAR,
    SIGNING_SECRET_FILE_ENV_VAR,
    KeyringIntegrityError,
    KeyringRollbackError,
    RetiredSigningKeyError,
    SigningKey,
    SigningKeyError,
    SigningKeyring,
    SigningKeyringFile,
    UnknownSigningKeyError,
    derive_key_id,
    keyring_from_document,
    load_signing_keyring,
    signing_keyring_from_env,
)

T0 = 1_800_000_000.0
SECRET_ONE = b"zq-key-sentinel-one-0123456789abcdef"
SECRET_TWO = b"zq-key-sentinel-two-0123456789abcdef"
SECRET_THREE = b"zq-key-sentinel-three-0123456789abcd"
SENTINELS = (SECRET_ONE, SECRET_TWO, SECRET_THREE)


def _assert_no_key_material(blob: Any) -> None:
    text = blob if isinstance(blob, (bytes, str)) else json.dumps(blob)
    data = text.encode("utf-8") if isinstance(text, str) else text
    for secret in SENTINELS:
        assert secret not in data
        assert base64.b64encode(secret) not in data


def _rotated(tmp_path: Path, rotations: int = 2) -> SigningKeyringFile:
    resolver = SigningKeyringFile(tmp_path / "keyring.json")
    for index, secret in enumerate((SECRET_ONE, SECRET_TWO, SECRET_THREE)[:rotations]):
        resolver.rotate(
            new_secret=secret,
            new_key_id=f"k{index + 1}",
            overlap_seconds=600,
            now=T0 + index * 100,
        )
    return resolver


def _document(*keys: dict[str, Any], generation: int = 1, **extra: Any) -> dict:
    return {
        "format": KEYRING_FORMAT,
        "schema_version": 1,
        "generation": generation,
        "rotated_at": T0,
        "keys": list(keys),
        **extra,
    }


def _inline(secret: bytes) -> str:
    return base64.b64encode(secret).decode("ascii")


def test_single_key_ring_derives_a_stable_value_free_key_id() -> None:
    keyring = SigningKeyring.single(SECRET_ONE)

    assert keyring.active_key.key_id == derive_key_id(SECRET_ONE)
    assert keyring.active_key.key_id == derive_key_id(SECRET_ONE.decode())
    assert keyring.active_key.key_id.startswith("hk_")
    assert derive_key_id(SECRET_ONE) != derive_key_id(SECRET_TWO)
    assert keyring.signing_key(now=T0).secret == SECRET_ONE
    _assert_no_key_material(repr(keyring))
    _assert_no_key_material(keyring.describe())


def test_rotation_moves_active_to_retiring_to_retired(tmp_path: Path) -> None:
    resolver = SigningKeyringFile(tmp_path / "keyring.json")
    first = resolver.rotate(new_secret=SECRET_ONE, new_key_id="k1", now=T0)
    second = resolver.rotate(
        new_secret=SECRET_TWO, new_key_id="k2", overlap_seconds=600, now=T0 + 10
    )
    third = resolver.rotate(
        new_secret=SECRET_THREE, new_key_id="k3", overlap_seconds=600, now=T0 + 20
    )

    assert first.to_dict() == {
        "previous_generation": None,
        "generation": 1,
        "rotated_at": T0,
        "active_key_id": "k1",
        "retiring_key_ids": [],
        "retired_key_ids": [],
        "verify_until": None,
    }
    assert second.previous_generation == 1
    assert second.retiring_key_ids == ("k1",)
    assert second.verify_until == T0 + 610
    assert third.generation == 3
    assert third.active_key_id == "k3"
    assert third.retiring_key_ids == ("k2",)
    assert third.retired_key_ids == ("k1",)

    keyring = resolver.current()
    states = {key.key_id: key.state for key in keyring.keys}
    assert states == {
        "k3": KEY_STATE_ACTIVE,
        "k2": KEY_STATE_RETIRING,
        "k1": KEY_STATE_RETIRED,
    }
    assert keyring.signing_key(now=T0 + 20).key_id == "k3"
    for receipt in (first, second, third):
        _assert_no_key_material(receipt.to_dict())
    _assert_no_key_material(keyring.describe())
    _assert_no_key_material(repr(keyring))


def test_old_and_new_keys_verify_only_inside_the_overlap_window(
    tmp_path: Path,
) -> None:
    keyring = _rotated(tmp_path).current()  # k2 active at T0+100, k1 until T0+700

    assert keyring.verification_key("k2", now=T0 + 100).key_id == "k2"
    assert keyring.verification_key("k1", now=T0 + 700).key_id == "k1"
    assert [key.key_id for key in keyring.verification_keys(now=T0 + 700)] == [
        "k2",
        "k1",
    ]
    with pytest.raises(RetiredSigningKeyError) as expired:
        keyring.verification_key("k1", now=T0 + 700.001)
    assert expired.value.code == "key_overlap_expired"
    assert [key.key_id for key in keyring.verification_keys(now=T0 + 701)] == ["k2"]
    with pytest.raises(UnknownSigningKeyError) as premature:
        keyring.verification_key("k2", now=T0 + 99)
    assert premature.value.code == "key_not_yet_active"
    with pytest.raises(UnknownSigningKeyError) as unknown:
        keyring.verification_key("k9", now=T0 + 100)
    assert unknown.value.code == "key_unknown"


def test_retired_keys_fail_deterministically(tmp_path: Path) -> None:
    keyring = _rotated(tmp_path, rotations=3).current()

    for now in (T0, T0 + 200, T0 + 10_000):
        with pytest.raises(RetiredSigningKeyError) as retired:
            keyring.verification_key("k1", now=now)
        assert retired.value.code == "key_retired"
    retired_key = next(key for key in keyring.keys if key.key_id == "k1")
    assert retired_key.secret is None


def test_a_premature_active_key_cannot_sign_or_verify(tmp_path: Path) -> None:
    path = tmp_path / "keyring.json"
    path.write_text(
        json.dumps(
            _document(
                {
                    "key_id": "future",
                    "state": "active",
                    "secret_b64": _inline(SECRET_ONE),
                    "not_before": T0 + 60,
                }
            )
        )
    )
    keyring = SigningKeyringFile(path).current()

    with pytest.raises(UnknownSigningKeyError) as signing:
        keyring.signing_key(now=T0)
    assert signing.value.code == "key_not_yet_active"
    with pytest.raises(UnknownSigningKeyError):
        keyring.verification_key("future", now=T0 + 59)
    assert keyring.signing_key(now=T0 + 60).key_id == "future"


def test_interrupted_rotation_keeps_the_previous_complete_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolver = _rotated(tmp_path)
    path = tmp_path / "keyring.json"
    before = path.read_bytes()

    def crash(_source: str, _target: Any) -> None:
        raise OSError("synthetic crash before the atomic replace")

    with monkeypatch.context() as patched:
        patched.setattr(_state_files.os, "replace", crash)
        with pytest.raises(SigningKeyError) as failed:
            resolver.rotate(new_secret=SECRET_THREE, new_key_id="k3", now=T0 + 200)

    assert failed.value.code == "keyring_write_failed"
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "keyring.json",
        "keyring.json.lock",
    ]
    assert resolver.current().generation == 2
    assert SigningKeyringFile(path).current().active_key.key_id == "k2"
    # The rotation can be retried once the fault clears.
    retried = resolver.rotate(new_secret=SECRET_THREE, new_key_id="k3", now=T0 + 201)
    assert retried.generation == 3


def test_a_partial_or_broken_file_never_replaces_a_loaded_generation(
    tmp_path: Path,
) -> None:
    resolver = _rotated(tmp_path)
    path = tmp_path / "keyring.json"
    complete = path.read_bytes()

    path.write_bytes(complete[: len(complete) // 2])
    assert resolver.current().generation == 2
    assert resolver.last_error == "keyring_invalid"
    with pytest.raises(KeyringIntegrityError):
        resolver.reload()
    with pytest.raises(KeyringIntegrityError):
        SigningKeyringFile(path).current()

    # A generation whose referenced secret is missing is incomplete as a whole.
    path.write_text(
        json.dumps(
            _document(
                {"key_id": "k3", "state": "active", "secret_file": "missing.key"},
                generation=3,
            )
        )
    )
    assert resolver.current().generation == 2
    assert resolver.last_error == "keyring_secret_unavailable"


def test_rollback_to_an_older_generation_is_refused(tmp_path: Path) -> None:
    resolver = _rotated(tmp_path)
    path = tmp_path / "keyring.json"
    generation_two = path.read_bytes()
    resolver.rotate(new_secret=SECRET_THREE, new_key_id="k3", now=T0 + 200)

    path.write_bytes(generation_two)
    keyring = resolver.current()
    assert keyring.generation == 3
    assert resolver.last_error == "keyring_rollback"
    with pytest.raises(RetiredSigningKeyError):
        keyring.verification_key("k1", now=T0 + 200)
    with pytest.raises(KeyringRollbackError):
        resolver.reload()
    with pytest.raises(KeyringRollbackError):
        SigningKeyringFile(path, min_generation=3).current()
    with pytest.raises(KeyringRollbackError):
        resolver.rotate(new_secret=SECRET_THREE, new_key_id="k4")


def test_a_forked_generation_with_the_same_number_is_refused(tmp_path: Path) -> None:
    resolver = _rotated(tmp_path)
    path = tmp_path / "keyring.json"
    document = json.loads(path.read_text())
    document["keys"][0]["secret_b64"] = _inline(SECRET_THREE)
    path.write_text(json.dumps(document))

    assert resolver.current().verification_key("k2", now=T0 + 100).secret == (
        SECRET_TWO
    )
    assert resolver.last_error == "keyring_rollback"


def test_rotation_refuses_reused_ids_and_unbounded_overlap(tmp_path: Path) -> None:
    resolver = _rotated(tmp_path, rotations=3)
    later = T0 + 1_000
    for key_id in ("k1", "k2", "k3"):
        with pytest.raises(SigningKeyError) as reused:
            resolver.rotate(new_secret=b"x" * 32, new_key_id=key_id, now=later)
        assert reused.value.code == "key_id_reused"
    for overlap in (0, -1, MAX_KEY_OVERLAP_SECONDS + 1, float("inf")):
        with pytest.raises(SigningKeyError) as invalid:
            resolver.rotate(new_secret=b"x" * 32, overlap_seconds=overlap, now=later)
        assert invalid.value.code == "rotation_invalid"
    with pytest.raises(SigningKeyError):
        resolver.rotate(new_secret=b"short", now=later)
    with pytest.raises(SigningKeyError):
        resolver.rotate(new_secret=b"x" * 32, new_key_id="bad id", now=later)
    with pytest.raises(SigningKeyError) as backwards:
        resolver.rotate(new_secret=b"x" * 32, now=T0 - 1)
    assert backwards.value.code == "rotation_invalid"
    assert resolver.current().generation == 3


def test_retired_tombstones_are_bounded(tmp_path: Path) -> None:
    resolver = SigningKeyringFile(tmp_path / "keyring.json")
    for index in range(MAX_RETIRED_KEYS + 5):
        resolver.rotate(
            new_secret=bytes([65 + index]) * 32,
            new_key_id=f"k{index}",
            now=T0 + index,
        )

    keyring = resolver.current()
    retired = [key for key in keyring.keys if key.state == KEY_STATE_RETIRED]
    assert len(retired) == MAX_RETIRED_KEYS
    assert retired[0].key_id == f"k{MAX_RETIRED_KEYS + 2}"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_rotation_writes_an_owner_only_file(tmp_path: Path) -> None:
    _rotated(tmp_path)

    mode = stat.S_IMODE((tmp_path / "keyring.json").stat().st_mode)
    assert mode == 0o600


def test_secret_files_and_environment_sources(tmp_path: Path) -> None:
    (tmp_path / "k2.key").write_bytes(SECRET_TWO + b"\n")
    path = tmp_path / "keyring.json"
    path.write_text(
        json.dumps(
            _document(
                {"state": "active", "secret_file": "k2.key"},
                {
                    "key_id": "k1",
                    "state": "retiring",
                    "secret_env": "SYNTHETIC_K1",
                    "verify_until": T0 + 60,
                },
                {"key_id": "k0", "state": "retired"},
            )
        )
    )

    keyring = load_signing_keyring(path, environ={"SYNTHETIC_K1": "env-secret"})

    assert keyring.active_key.key_id == derive_key_id(SECRET_TWO)
    assert keyring.active_key.secret == SECRET_TWO
    assert keyring.verification_key("k1", now=T0).secret == b"env-secret"
    with pytest.raises(KeyringIntegrityError) as missing:
        load_signing_keyring(path, environ={})
    assert missing.value.code == "keyring_secret_unavailable"

    # Rotation keeps the retiring key's secret reference instead of copying it.
    resolver = SigningKeyringFile(path, environ={"SYNTHETIC_K1": "env-secret"})
    resolver.rotate(new_secret=SECRET_THREE, new_key_id="k3", now=T0 + 1)
    rotated = json.loads(path.read_text())
    assert rotated["keys"][1] == {
        "key_id": derive_key_id(SECRET_TWO),
        "state": "retiring",
        "secret_file": "k2.key",
        "not_before": None,
        "verify_until": T0 + 1 + 24 * 60 * 60,
    }
    assert SECRET_TWO not in path.read_bytes()
    assert {entry["key_id"] for entry in rotated["keys"][2:]} == {"k0", "k1"}


@pytest.mark.parametrize(
    "document",
    [
        _document(
            {"key_id": "a", "state": "active", "secret_b64": _inline(SECRET_ONE)},
            {"key_id": "b", "state": "active", "secret_b64": _inline(SECRET_TWO)},
        ),
        _document(
            {"key_id": "a", "state": "retiring", "secret_b64": _inline(SECRET_ONE)},
        ),
        _document(
            {"key_id": "a", "state": "active", "secret_b64": _inline(SECRET_ONE)},
            {"key_id": "b", "state": "retired", "secret_b64": _inline(SECRET_TWO)},
        ),
        _document(
            {"key_id": "a", "state": "active", "secret_b64": _inline(SECRET_ONE)},
            {
                "key_id": "b",
                "state": "retiring",
                "secret_b64": _inline(SECRET_TWO),
                "verify_until": T0 + MAX_KEY_OVERLAP_SECONDS + 1,
            },
        ),
        _document(
            {"key_id": "a", "state": "active", "secret_b64": _inline(SECRET_ONE)},
            {"key_id": "a", "state": "retired"},
        ),
        _document(
            {
                "key_id": "a",
                "state": "active",
                "secret_b64": _inline(SECRET_ONE),
                "secret_env": "ALSO",
            }
        ),
        _document({"key_id": "a", "state": "active", "secret_b64": "not base64!"}),
        _document({"key_id": "a", "state": "pending", "secret_b64": "AA=="}),
        _document({"key_id": "a", "state": ["active"], "secret_b64": "AA=="}),
        _document(),
        _document(
            {"key_id": "a", "state": "active", "secret_b64": _inline(SECRET_ONE)},
            unexpected=True,
        ),
        _document(
            {"key_id": "a", "state": "active", "secret_b64": _inline(SECRET_ONE)},
            generation=0,
        ),
        {**_document(), "format": "other"},
    ],
)
def test_incomplete_or_malformed_generations_are_rejected(
    document: dict[str, Any],
) -> None:
    with pytest.raises(KeyringIntegrityError) as invalid:
        keyring_from_document(document)
    assert invalid.value.code == "keyring_invalid"
    _assert_no_key_material(str(invalid.value))


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "keyring.json"
    path.write_text(
        '{"format":"openmed.signing-keyring","schema_version":1,"generation":1,'
        '"generation":2,"keys":[]}'
    )

    with pytest.raises(KeyringIntegrityError):
        load_signing_keyring(path)


def test_signing_key_repr_and_errors_never_show_secrets() -> None:
    key = SigningKey(key_id="k1", state=KEY_STATE_ACTIVE, secret=SECRET_ONE)

    _assert_no_key_material(repr(key))
    _assert_no_key_material(key.describe())
    assert SigningKey(key_id="k1", state=KEY_STATE_RETIRED).secret is None
    with pytest.raises(KeyringIntegrityError):
        SigningKey(key_id="k1", state=KEY_STATE_RETIRING, secret=SECRET_ONE)


def test_signing_keyring_from_env(tmp_path: Path) -> None:
    assert signing_keyring_from_env({}) is None

    single = signing_keyring_from_env(
        {SIGNING_SECRET_ENV_VAR: "env-secret", SIGNING_KEY_ID_ENV_VAR: "env-k1"}
    )
    assert single is not None
    assert single.current().active_key.key_id == "env-k1"

    (tmp_path / "webhook.key").write_bytes(SECRET_ONE + b"\r\n")
    from_file = signing_keyring_from_env(
        {SIGNING_SECRET_FILE_ENV_VAR: str(tmp_path / "webhook.key")}
    )
    assert from_file is not None
    assert from_file.current().active_key.secret == SECRET_ONE

    _rotated(tmp_path)
    resolver = signing_keyring_from_env(
        {KEYRING_FILE_ENV_VAR: str(tmp_path / "keyring.json")}
    )
    assert isinstance(resolver, SigningKeyringFile)
    assert resolver.current().generation == 2

    for ambiguous in (
        {SIGNING_SECRET_ENV_VAR: "a", SIGNING_SECRET_FILE_ENV_VAR: "b"},
        {KEYRING_FILE_ENV_VAR: "x", SIGNING_SECRET_ENV_VAR: "a"},
        {
            KEYRING_FILE_ENV_VAR: str(tmp_path / "keyring.json"),
            SIGNING_KEY_ID_ENV_VAR: "k",
        },
    ):
        with pytest.raises(KeyringIntegrityError):
            signing_keyring_from_env(ambiguous)
    with pytest.raises(KeyringIntegrityError) as missing:
        signing_keyring_from_env({KEYRING_FILE_ENV_VAR: str(tmp_path / "absent.json")})
    assert missing.value.code == "keyring_unreadable"
    assert str(tmp_path) not in str(missing.value)


def test_retiring_keys_require_a_bounded_overlap() -> None:
    active = SigningKey(key_id="k2", state=KEY_STATE_ACTIVE, secret=SECRET_TWO)
    retiring = SigningKey(
        key_id="k1",
        state=KEY_STATE_RETIRING,
        secret=SECRET_ONE,
        verify_until=T0 + 60,
    )

    with pytest.raises(KeyringIntegrityError):
        SigningKeyring(generation=2, keys=(active, retiring))
    keyring = SigningKeyring(generation=2, keys=(active, retiring), rotated_at=T0)
    assert keyring.verification_key("k1", now=T0 + 60).key_id == "k1"
