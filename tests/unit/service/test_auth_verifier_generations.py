"""Tests for atomic API-key and JWT verifier generations."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
from typing import Any

import pytest
from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import APIKeyCredential, AuthError, hash_api_key
from openmed.service.auth_generations import (
    AUTH_GENERATION_SCOPE_KEY,
    GENERATION_REASON_CODES,
    GenerationalServiceAuth,
    GenerationValidationError,
    VerifierGenerationRegistry,
    VerifierGenerationSpec,
    parse_verifier_generation,
    validate_verifier_generation,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
_ISSUER = "https://issuer.synthetic.invalid"
_AUDIENCE = "openmed-synthetic"
_SECRET_A = b"synthetic-generation-a-hs256-secret-material-0001"
_SECRET_B = b"synthetic-generation-b-hs256-secret-material-0002"
_RSA_N = (
    "3O8uls5vuPQurwdtlL7CclqrswG1af7xVrfnn-kWe5PVdlRG3BQ7vczvWb-wWbF53Pkd"
    "NYCwEXWx9Ym4qSTsJbWmx1NH-b1dEQOYoIVqYhYbMNUZ3R4v9IN2yp6c1ocZ8jTbU_8f"
    "ZSPjFeyaZO5vgvo0-fMhDWuN6YThrZGXAP0"
)
_RSA_E = "AQAB"
_RSA_D = (
    "beOVo7LYRQFHOw2Rxps_MgvBPQ8LgcYpmf1s-s-_vAWi9fEjMZHqyRPmtRgwCdzJhUx"
    "u586zRGvq8ProW1EfFxHKxec5L4iFIDucxSUvhPvydL7PNQuiCIN72mD-VEIhLsft3d"
    "Q-ivi3JyuV-evKRVx0wKalUL4_liDEM6tcTkE"
)
_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")
_NOW = 2_000_000_000.0
_SENTINEL = "SYNTHETIC-SENTINEL-7f3a"


class FakeClock:
    """Injected monotonic-enough test clock."""

    def __init__(self, now: float = _NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeLoader:
    """Minimal model loader double for middleware integration tests."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _b64url_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return _b64url(encoded)


def _decode_int(value: str) -> int:
    padding = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + padding), "big")


def _claims(**overrides: Any) -> dict[str, Any]:
    claims = {
        "sub": "synthetic-client",
        "iss": _ISSUER,
        "aud": _AUDIENCE,
        "iat": _NOW - 10,
        "exp": _NOW + 600,
        "scope": "pii:read",
    }
    claims.update(overrides)
    return claims


def _hs_token(secret: bytes, kid: str, claims: dict[str, Any] | None = None) -> str:
    header = {"alg": "HS256", "kid": kid, "typ": "JWT"}
    signing_input = f"{_b64url_json(header)}.{_b64url_json(claims or _claims())}"
    signature = hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256)
    return f"{signing_input}.{_b64url(signature.digest())}"


def _rs_token(kid: str, claims: dict[str, Any] | None = None) -> str:
    header = {"alg": "RS256", "kid": kid, "typ": "JWT"}
    signing_input = f"{_b64url_json(header)}.{_b64url_json(claims or _claims())}"
    digest = hashlib.sha256(signing_input.encode("ascii")).digest()
    modulus = _decode_int(_RSA_N)
    key_size = (modulus.bit_length() + 7) // 8
    encoded = (
        b"\x00\x01"
        + (b"\xff" * (key_size - len(_SHA256_DIGEST_INFO_PREFIX) - len(digest) - 3))
        + b"\x00"
        + _SHA256_DIGEST_INFO_PREFIX
        + digest
    )
    signature = pow(int.from_bytes(encoded, "big"), _decode_int(_RSA_D), modulus)
    return f"{signing_input}.{_b64url(signature.to_bytes(key_size, 'big'))}"


def _oct_jwk(kid: str, secret: bytes) -> dict[str, str]:
    return {"kty": "oct", "kid": kid, "alg": "HS256", "k": _b64url(secret)}


def _rsa_jwk(kid: str) -> dict[str, str]:
    return {"kty": "RSA", "kid": kid, "alg": "RS256", "n": _RSA_N, "e": _RSA_E}


def _api_key(raw: str, key_id: str, *, scopes: tuple[str, ...] = ("pii:read",)):
    return APIKeyCredential(
        key_id=key_id,
        key_hash=hash_api_key(raw),
        principal=f"principal-{key_id}",
        scopes=scopes,
    )


def _spec(
    generation_id: str,
    serial: int,
    *,
    api_keys: tuple[APIKeyCredential, ...] = (),
    jwks: tuple[dict[str, Any], ...] = (),
    **overrides: Any,
) -> VerifierGenerationSpec:
    values: dict[str, Any] = {
        "generation_id": generation_id,
        "serial": serial,
        "api_keys": api_keys,
        "jwks": jwks,
        "jwt_issuer": _ISSUER if jwks else None,
        "jwt_audiences": (_AUDIENCE,) if jwks else (),
    }
    values.update(overrides)
    return VerifierGenerationSpec(**values)


def _registry(clock: FakeClock | None = None, **kwargs: Any):
    kwargs.setdefault("min_rsa_modulus_bits", 1024)
    return VerifierGenerationRegistry(clock=clock or FakeClock(), **kwargs)


def _generation_a() -> VerifierGenerationSpec:
    return _spec(
        "gen-a",
        1,
        api_keys=(_api_key("raw-key-a", "key-a"),),
        jwks=(_oct_jwk("hs-a", _SECRET_A), _rsa_jwk("rs-a")),
    )


def _generation_b() -> VerifierGenerationSpec:
    return _spec(
        "gen-b",
        2,
        api_keys=(_api_key("raw-key-b", "key-b"),),
        jwks=(_oct_jwk("hs-b", _SECRET_B),),
    )


def _assert_rejected(registry, credential: str, *, bearer: bool = False) -> None:
    with pytest.raises(AuthError) as caught:
        if bearer:
            registry.authenticate_bearer(credential)
        else:
            registry.authenticate_api_key(credential)
    assert caught.value.code == "invalid_credentials"
    assert caught.value.__context__ is None


def test_rotation_admits_declared_overlap_and_rejects_retired_after_boundary():
    clock = FakeClock()
    registry = _registry(clock, max_overlap_seconds=300)
    assert registry.reload(_generation_a()).accepted

    decision = registry.authenticate_api_key("raw-key-a")
    assert (decision.generation_id, decision.serial, decision.role) == (
        "gen-a",
        1,
        "active",
    )
    assert registry.authenticate_bearer(_hs_token(_SECRET_A, "hs-a")).role == "active"
    assert registry.authenticate_bearer(_rs_token("rs-a")).generation_id == "gen-a"

    receipt = registry.reload(_generation_b(), overlap_seconds=120)
    assert receipt.accepted and receipt.reason == "ok"
    assert receipt.retire_at == _NOW + 120

    old = registry.authenticate_api_key("raw-key-a")
    assert (old.generation_id, old.role) == ("gen-a", "overlap")
    new = registry.authenticate_api_key("raw-key-b")
    assert (new.generation_id, new.role) == ("gen-b", "active")
    assert registry.authenticate_bearer(_hs_token(_SECRET_A, "hs-a")).role == "overlap"

    clock.now = _NOW + 119.999
    assert registry.authenticate_api_key("raw-key-a").role == "overlap"

    clock.now = _NOW + 120
    _assert_rejected(registry, "raw-key-a")
    claims = _claims(iat=clock.now - 1, exp=clock.now + 60)
    _assert_rejected(registry, _hs_token(_SECRET_A, "hs-a", claims), bearer=True)
    _assert_rejected(registry, _rs_token("rs-a", claims), bearer=True)
    assert registry.authenticate_api_key("raw-key-b").generation_id == "gen-b"
    assert registry.status().overlap == ()


def test_zero_overlap_cuts_over_immediately_and_staging_does_not_activate():
    registry = _registry()
    assert registry.reload(_generation_a()).accepted
    staged = registry.stage(_generation_b())
    assert staged.accepted and staged.action == "stage"
    _assert_rejected(registry, "raw-key-b")
    assert registry.status().staged == ("gen-b",)

    activated = registry.activate("gen-b")
    assert activated.accepted and activated.retire_at is None
    _assert_rejected(registry, "raw-key-a")
    assert registry.authenticate_api_key("raw-key-b").generation_id == "gen-b"
    assert registry.status().staged == ()


def test_overlap_is_bounded_to_one_previous_generation():
    clock = FakeClock()
    registry = _registry(clock, max_overlap_seconds=600)
    registry.reload(_generation_a())
    registry.reload(_generation_b(), overlap_seconds=600)
    gen_c = _spec("gen-c", 3, api_keys=(_api_key("raw-key-c", "key-c"),))
    assert registry.reload(gen_c, overlap_seconds=600).accepted

    _assert_rejected(registry, "raw-key-a")
    assert registry.authenticate_api_key("raw-key-b").role == "overlap"
    assert registry.authenticate_api_key("raw-key-c").role == "active"

    too_long = _spec("gen-d", 4, api_keys=(_api_key("raw-key-d", "key-d"),))
    rejected = registry.reload(too_long, overlap_seconds=601)
    assert not rejected.accepted and rejected.reason == "overlap_out_of_bounds"
    for bad in (-1.0, float("nan"), float("inf"), True):
        assert registry.reload(too_long, overlap_seconds=bad).reason == (
            "overlap_out_of_bounds"
        )
    assert registry.status().active_generation_id == "gen-c"


def test_revocation_takes_effect_immediately_and_fails_closed():
    clock = FakeClock()
    registry = _registry(clock)
    registry.reload(_generation_a())
    registry.reload(_generation_b(), overlap_seconds=300)

    revoked = registry.revoke("gen-a")
    assert revoked.accepted and revoked.action == "revoke"
    _assert_rejected(registry, "raw-key-a")
    assert registry.authenticate_api_key("raw-key-b").generation_id == "gen-b"

    assert registry.revoke("gen-b").accepted
    _assert_rejected(registry, "raw-key-b")
    assert registry.status().active_generation_id is None
    assert registry.metrics()["decision_no_active_generation"] == 1

    assert registry.revoke("gen-b").reason == "generation_unknown"
    reused = registry.reload(
        _spec("gen-b", 9, api_keys=(_api_key("raw-key-z", "key-z"),))
    )
    assert reused.reason == "generation_id_reused"

    registry.stage(_spec("gen-s", 10, api_keys=(_api_key("raw-key-s", "key-s"),)))
    assert registry.revoke("gen-s").accepted
    assert registry.activate("gen-s").reason == "generation_unknown"


def test_stale_serials_and_unknown_activations_are_rejected():
    registry = _registry()
    registry.reload(_generation_b())
    stale = registry.reload(_generation_a())
    assert stale.reason == "generation_serial_stale"
    assert registry.activate("never-staged").reason == "generation_unknown"
    assert registry.status().active_generation_id == "gen-b"


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (_spec("gen-x", 1), "generation_empty"),
        (_spec("", 1, api_keys=(_api_key("k", "k1"),)), "generation_id_invalid"),
        (
            _spec("bad id/slash", 1, api_keys=(_api_key("k", "k1"),)),
            "generation_id_invalid",
        ),
        (
            _spec("gen-x", 0, api_keys=(_api_key("k", "k1"),)),
            "generation_serial_invalid",
        ),
        (
            _spec("gen-x", True, api_keys=(_api_key("k", "k1"),)),
            "generation_serial_invalid",
        ),
        (
            _spec(
                "gen-x",
                1,
                api_keys=(APIKeyCredential("k1", "sha256:abc", "p"),),
            ),
            "api_key_hash_invalid",
        ),
        (
            _spec("gen-x", 1, api_keys=(_api_key("k", "k1"), _api_key("k", "k2"))),
            "api_key_duplicate",
        ),
        (
            _spec("gen-x", 1, api_keys=(_api_key("k", "k1"), _api_key("j", "k1"))),
            "api_key_duplicate",
        ),
        (_spec("gen-x", 1, api_keys=("not-a-credential",)), "api_keys_malformed"),
        (
            _spec("gen-x", 1, jwks=({"kty": "EC", "kid": "ec", "crv": "P-256"},)),
            "jwk_unsupported",
        ),
        (
            _spec("gen-x", 1, jwks=({"kty": "oct", "alg": "RS256", "kid": "o"},)),
            "jwk_unsupported",
        ),
        (
            _spec("gen-x", 1, jwks=({"kty": "oct", "k": _b64url(_SECRET_A)},)),
            "jwk_kid_missing",
        ),
        (
            _spec(
                "gen-x",
                1,
                jwks=(_oct_jwk("dup", _SECRET_A), _oct_jwk("dup", _SECRET_B)),
            ),
            "jwk_kid_duplicate",
        ),
        (
            _spec("gen-x", 1, jwks=({**_rsa_jwk("rs"), "d": _RSA_D},)),
            "jwk_private_material",
        ),
        (
            _spec("gen-x", 1, jwks=(_oct_jwk("short", b"short-secret"),)),
            "jwk_key_too_weak",
        ),
        (
            _spec("gen-x", 1, jwks=({"kty": "oct", "kid": "o", "k": "***"},)),
            "jwk_key_invalid",
        ),
        (
            _spec("gen-x", 1, jwks=({"kty": "RSA", "kid": "r", "n": _RSA_N},)),
            "jwk_key_invalid",
        ),
        (
            _spec("gen-x", 1, jwks=(_oct_jwk("o", _SECRET_A),), jwt_issuer=None),
            "jwt_binding_incomplete",
        ),
        (
            _spec("gen-x", 1, jwks=(_oct_jwk("o", _SECRET_A),), jwt_audiences=()),
            "jwt_binding_incomplete",
        ),
        (
            _spec(
                "gen-x",
                1,
                jwks=(_oct_jwk("o", _SECRET_A),),
                jwt_leeway_seconds=float("nan"),
            ),
            "jwt_leeway_invalid",
        ),
        (
            _spec("gen-x", 1, jwks=(_oct_jwk("o", _SECRET_A),), jwt_leeway_seconds=301),
            "jwt_leeway_invalid",
        ),
        (
            _spec(
                "gen-x",
                1,
                jwks=(_oct_jwk("o", _SECRET_A),),
                jwt_max_lifetime_seconds=None,
            ),
            "jwt_lifetime_invalid",
        ),
        (
            _spec(
                "gen-x",
                1,
                jwks=(_oct_jwk("o", _SECRET_A),),
                jwt_max_lifetime_seconds=0,
            ),
            "jwt_lifetime_invalid",
        ),
    ],
)
def test_incomplete_or_malformed_generations_never_activate(spec, reason):
    registry = _registry()
    registry.reload(_generation_a())
    before = registry.status()

    receipt = registry.reload(spec)
    assert not receipt.accepted
    assert receipt.reason == reason
    assert reason in GENERATION_REASON_CODES
    assert registry.status() == before
    assert registry.authenticate_api_key("raw-key-a").generation_id == "gen-a"
    with pytest.raises(GenerationValidationError) as caught:
        validate_verifier_generation(spec, min_rsa_modulus_bits=1024)
    assert caught.value.reason == reason
    assert str(caught.value) == reason


def test_default_rsa_floor_rejects_short_synthetic_modulus():
    registry = VerifierGenerationRegistry(clock=FakeClock())
    receipt = registry.reload(_spec("gen-r", 1, jwks=(_rsa_jwk("rs"),)))
    assert receipt.reason == "jwk_key_too_weak"


def test_conflicting_generations_never_activate():
    registry = _registry()
    registry.reload(_generation_a())

    remapped = _spec(
        "gen-c",
        2,
        api_keys=(_api_key("raw-key-a", "key-a", scopes=("*",)),),
    )
    assert registry.reload(remapped, overlap_seconds=60).reason == "api_key_conflict"

    rekeyed = _spec("gen-c", 2, jwks=(_oct_jwk("hs-a", _SECRET_B),))
    assert registry.reload(rekeyed, overlap_seconds=60).reason == "jwk_kid_conflict"

    carried = _spec(
        "gen-c",
        2,
        api_keys=(_api_key("raw-key-a", "key-a"),),
        jwks=(_oct_jwk("hs-a", _SECRET_A),),
    )
    assert registry.reload(carried, overlap_seconds=60).accepted

    duplicate_stage = registry.stage(_spec("gen-d", 3, api_keys=(_api_key("x", "x"),)))
    assert duplicate_stage.accepted
    again = registry.stage(_spec("gen-d", 4, api_keys=(_api_key("y", "y"),)))
    assert again.reason == "generation_id_reused"


def test_staged_conflict_is_rechecked_at_activation():
    registry = _registry()
    registry.reload(_generation_a())
    staged = _spec("gen-late", 5, jwks=(_oct_jwk("hs-z", _SECRET_B),))
    assert registry.stage(staged).accepted
    other = _spec("gen-mid", 3, jwks=(_oct_jwk("hs-z", _SECRET_A),))
    assert registry.reload(other).accepted
    assert registry.activate("gen-late", overlap_seconds=30).reason == (
        "jwk_kid_conflict"
    )
    assert registry.status().active_generation_id == "gen-mid"


def test_parse_document_reuses_service_parsers_and_rejects_malformed():
    document = {
        "generation_id": "gen-doc",
        "serial": 7,
        "api_keys": [
            {
                "id": "doc-key",
                "key_hash": hash_api_key("raw-doc-key"),
                "principal": "doc-principal",
                "scopes": ["pii:read"],
            }
        ],
        "jwks": {"keys": [_oct_jwk("hs-doc", _SECRET_A)]},
        "jwt": {
            "issuer": _ISSUER,
            "audience": [_AUDIENCE],
            "leeway_seconds": 5,
            "max_lifetime_seconds": 900,
        },
    }
    spec = parse_verifier_generation(document)
    assert spec.serial == 7 and spec.jwt_audiences == (_AUDIENCE,)
    registry = _registry()
    assert registry.reload(document).accepted
    decision = registry.authenticate_api_key("raw-doc-key")
    assert decision.principal.subject == "doc-principal"
    assert registry.authenticate_bearer(_hs_token(_SECRET_A, "hs-doc")).serial == 7

    malformed = [
        ("not-a-mapping", "document_malformed"),
        ({**document, "extra": _SENTINEL}, "document_field_unknown"),
        ({**document, "api_keys": {"keys": "nope"}}, "api_keys_malformed"),
        ({**document, "api_keys": [{"key_hash": _SENTINEL}]}, "api_keys_malformed"),
        ({**document, "jwks": "nope"}, "jwks_malformed"),
        ({**document, "jwt": {"issuer": _ISSUER, "x": 1}}, "document_field_unknown"),
        ({**document, "jwt": {"leeway_seconds": "abc"}}, "jwt_leeway_invalid"),
        ({**document, "serial": "7"}, "generation_serial_invalid"),
    ]
    for payload, reason in malformed:
        receipt = registry.reload(payload)
        assert receipt.reason == reason, payload
        assert _SENTINEL not in repr(receipt)
    assert registry.status().active_generation_id == "gen-doc"


def test_caller_mutation_after_reload_cannot_change_active_set():
    registry = _registry()
    jwk = _oct_jwk("hs-m", _SECRET_A)
    keys = [_api_key("raw-key-m", "key-m")]
    spec = _spec("gen-m", 1, api_keys=keys, jwks=(jwk,))
    assert registry.reload(spec).accepted
    jwk["k"] = _b64url(_SECRET_B)
    keys.append(_api_key("raw-key-injected", "key-i"))

    assert registry.authenticate_bearer(_hs_token(_SECRET_A, "hs-m")).serial == 1
    _assert_rejected(registry, _hs_token(_SECRET_B, "hs-m"), bearer=True)
    _assert_rejected(registry, "raw-key-injected")


def test_pinned_view_keeps_one_generation_across_a_reload():
    registry = _registry()
    registry.reload(_generation_a())
    view = registry.view()
    registry.reload(_generation_b())

    assert view.active_generation_id == "gen-a"
    assert view.authenticate_api_key("raw-key-a").generation_id == "gen-a"
    with pytest.raises(AuthError):
        view.authenticate_api_key("raw-key-b")
    assert registry.view().epoch > view.epoch


def test_invalid_clock_fails_closed_for_decisions_and_lifecycle():
    clock = FakeClock()
    registry = _registry(clock)
    registry.reload(_generation_a())
    for bad in (float("nan"), float("inf"), "now", True):
        clock.now = bad
        _assert_rejected(registry, "raw-key-a")
        assert registry.reload(_generation_b()).reason == "clock_invalid"
    clock.now = _NOW
    assert registry.authenticate_api_key("raw-key-a").generation_id == "gen-a"


def test_jwt_claims_use_injected_clock_and_generation_binding():
    clock = FakeClock()
    registry = _registry(clock)
    registry.reload(_generation_a())
    _assert_rejected(
        registry, _hs_token(_SECRET_A, "hs-a", _claims(iss="other")), bearer=True
    )
    _assert_rejected(
        registry, _hs_token(_SECRET_A, "hs-a", _claims(aud="other")), bearer=True
    )
    token = _hs_token(_SECRET_A, "hs-a")
    clock.now = _NOW + 600
    _assert_rejected(registry, token, bearer=True)


def test_reloads_never_expose_a_partial_set_under_concurrency():
    clock = FakeClock()
    registry = _registry(clock)
    keys_per_generation = 12
    generations = 40

    def generation(index: int) -> VerifierGenerationSpec:
        return _spec(
            f"gen-{index}",
            index,
            api_keys=tuple(
                _api_key(f"raw-{index}-{slot}", f"key-{index}-{slot}")
                for slot in range(keys_per_generation)
            ),
        )

    specs = [generation(index) for index in range(1, generations + 1)]
    assert registry.reload(specs[0]).accepted
    stop = threading.Event()
    failures: list[str] = []
    observed: set[str] = set()

    def reader() -> None:
        while not stop.is_set():
            view = registry.view()
            active = view.active_generation_id
            if active is None:
                failures.append("no-active")
                return
            index = int(active.split("-")[1])
            observed.add(active)
            for slot in range(keys_per_generation):
                try:
                    decision = view.authenticate_api_key(f"raw-{index}-{slot}")
                except AuthError:
                    failures.append("partial")
                    return
                if decision.generation_id != active:
                    failures.append("unpinned")
                    return
            try:
                view.authenticate_api_key(f"raw-{index + 1}-0")
            except AuthError:
                pass
            else:
                failures.append("future-key")
                return
            try:
                direct = registry.authenticate_api_key(f"raw-{index}-0")
            except AuthError:
                continue
            if direct.generation_id != active:
                failures.append("direct-mismatch")
                return

    def writer() -> None:
        for spec in specs[1:]:
            receipt = registry.reload(spec)
            if not receipt.accepted:
                failures.append("reload")
                return
            time.sleep(0.002)
        stop.set()

    readers = [threading.Thread(target=reader) for _ in range(6)]
    for thread in readers:
        thread.start()
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    writer_thread.join(timeout=60)
    stop.set()
    for thread in readers:
        thread.join(timeout=60)

    assert failures == []
    assert registry.status().active_generation_id == f"gen-{generations}"
    assert len(observed) > 1


def test_receipts_metrics_status_and_errors_expose_no_key_material():
    registry = _registry()
    raw_key = f"raw-{_SENTINEL}"
    secret = f"{_SENTINEL}-hs256-secret-padding-padding".encode()
    spec = _spec(
        "gen-leak",
        1,
        api_keys=(
            APIKeyCredential(
                key_id=f"id-{_SENTINEL}",
                key_hash=hash_api_key(raw_key),
                principal=f"principal-{_SENTINEL}",
                scopes=("pii:read",),
            ),
        ),
        jwks=(_oct_jwk(f"kid-{_SENTINEL}", secret),),
    )
    receipt = registry.reload(spec)
    assert receipt.accepted
    token = _hs_token(
        secret, f"kid-{_SENTINEL}", _claims(sub=f"sub-{_SENTINEL}", email=_SENTINEL)
    )
    decision = registry.authenticate_bearer(token)
    assert decision.principal.subject == f"sub-{_SENTINEL}"

    rejected = [
        registry.reload(_spec(f"bad {_SENTINEL}", 2, api_keys=spec.api_keys)),
        registry.reload({"generation_id": _SENTINEL, "serial": 2, "x": _SENTINEL}),
        registry.reload(
            _spec("gen-2", 2, jwks=({**_oct_jwk("o", secret), "d": _SENTINEL},))
        ),
        registry.revoke(_SENTINEL),
        registry.activate(_SENTINEL),
    ]
    with pytest.raises(AuthError) as caught:
        registry.authenticate_bearer(f"{token}{_SENTINEL}")
    with pytest.raises(GenerationValidationError) as invalid:
        validate_verifier_generation(
            _spec("gen-x", 1, jwks=({"kty": "oct", "kid": "o", "k": _SENTINEL},))
        )

    surfaces = [
        repr(receipt),
        repr(spec),
        repr(registry.view()),
        repr(registry.status()),
        repr(decision.pin),
        json.dumps(registry.metrics()),
        json.dumps(receipt.to_dict()),
        str(caught.value),
        str(invalid.value),
        repr(invalid.value.__context__),
        *(repr(item) for item in rejected),
        *(json.dumps(item.to_dict()) for item in rejected),
    ]
    for surface in surfaces:
        assert _SENTINEL not in surface
        assert _b64url(secret) not in surface
        assert hash_api_key(raw_key).split(":", 1)[1] not in surface
    assert all(item.reason in GENERATION_REASON_CODES for item in rejected)
    assert all(not item.accepted for item in rejected)


def test_receipt_digest_is_deterministic_and_tamper_evident():
    first = _registry().reload(_generation_a())
    second = _registry().reload(_generation_a())
    assert first.digest == second.digest
    assert first.to_dict()["digest"] == first.digest
    assert first.digest.startswith("sha256:") and len(first.digest) == 71
    later = _registry(FakeClock(_NOW + 1)).reload(_generation_a())
    assert later.digest != first.digest


def test_metrics_count_codes_only():
    clock = FakeClock()
    registry = _registry(clock)
    registry.reload(_generation_a())
    registry.reload(_generation_b(), overlap_seconds=30)
    registry.reload(_spec("gen-x", 1, api_keys=(_api_key("raw-x", "key-x"),)))
    registry.authenticate_api_key("raw-key-a")
    registry.authenticate_api_key("raw-key-b")
    with pytest.raises(AuthError):
        registry.authenticate_api_key("unknown")
    metrics = registry.metrics()
    assert metrics["reload_accepted"] == 2
    assert metrics["reload_rejected:generation_serial_stale"] == 1
    assert metrics["decision_active"] == 1
    assert metrics["decision_overlap"] == 1
    assert metrics["decision_rejected"] == 1
    assert all(isinstance(value, int) for value in metrics.values())


def test_registry_rejects_invalid_configuration():
    for kwargs in (
        {"max_overlap_seconds": -1},
        {"max_overlap_seconds": float("inf")},
        {"max_staged": 0},
        {"min_rsa_modulus_bits": 256},
    ):
        with pytest.raises(ValueError):
            VerifierGenerationRegistry(clock=FakeClock(), **kwargs)


def test_staged_capacity_is_bounded():
    registry = _registry(max_staged=2)
    for index in range(1, 3):
        spec = _spec(f"gen-{index}", index, api_keys=(_api_key(f"k{index}", "k"),))
        assert registry.stage(spec).accepted
    overflow = _spec("gen-3", 3, api_keys=(_api_key("k3", "k"),))
    assert registry.stage(overflow).reason == "staged_capacity_exceeded"


@pytest.fixture
def generational_app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    for env_var in (
        "OPENMED_SERVICE_AUTH_ENABLED",
        "OPENMED_SERVICE_AUTH_API_KEYS",
        "OPENMED_SERVICE_AUTH_JWKS",
        "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
        json.dumps({"GET /models/loaded": ["pii:read"]}),
    )
    app = create_app()
    original = next(route for route in app.routes if route.path == "/models/loaded")

    async def whoami(request: Request) -> dict[str, Any]:
        pin = request.scope[AUTH_GENERATION_SCOPE_KEY]
        return {
            "subject": request.state.auth_principal.subject,
            "generation_id": pin.generation_id,
            "role": pin.role,
        }

    app.routes[app.routes.index(original)] = APIRoute(
        "/models/loaded", whoami, methods=["GET"]
    )
    clock = FakeClock()
    registry = _registry(clock)
    app.state.auth = GenerationalServiceAuth.replacing(
        app.state.auth, registry=registry
    )
    return app, registry, clock


@pytest.mark.integration
def test_middleware_pins_requests_and_rotates_without_restart(generational_app):
    app, registry, clock = generational_app
    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False
    ) as client:
        response = client.get("/models/loaded", headers={"X-API-Key": "raw-key-a"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_credentials"

        registry.reload(_generation_a())
        response = client.get("/models/loaded", headers={"X-API-Key": "raw-key-a"})
        assert response.status_code == 200
        assert response.json() == {
            "subject": "principal-key-a",
            "generation_id": "gen-a",
            "role": "active",
        }

        registry.reload(_generation_b(), overlap_seconds=60)
        bearer = {"Authorization": f"Bearer {_hs_token(_SECRET_A, 'hs-a')}"}
        response = client.get("/models/loaded", headers=bearer)
        assert response.json()["role"] == "overlap"
        apikey = {"Authorization": "ApiKey raw-key-b"}
        assert client.get("/models/loaded", headers=apikey).json()["role"] == "active"

        rejected = registry.reload(_spec("gen-bad", 9))
        assert not rejected.accepted
        response = client.get("/models/loaded", headers={"X-API-Key": "raw-key-b"})
        assert response.json()["generation_id"] == "gen-b"

        clock.now = _NOW + 60
        response = client.get("/models/loaded", headers={"X-API-Key": "raw-key-a"})
        assert response.status_code == 401
        assert "raw-key-a" not in response.text

        assert client.get("/models/loaded").status_code == 401
        assert client.get("/health").status_code != 401
