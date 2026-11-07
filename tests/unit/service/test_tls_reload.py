"""Unit tests for versioned TLS/mTLS credential generations and reload.

Every certificate and key is ephemeral and synthetic. Assertions record only
generation IDs, fingerprints, states, counts and reason codes.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import signal
import sys
import threading
import weakref
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from openmed.service import tls_reload
from openmed.service.grpc_security import (
    GrpcCredentialReloader,
    GrpcTLSConfig,
    resolve_grpc_listener,
)
from openmed.service.mtls import (
    SERVICE_MTLS_CA_BUNDLE_ENV_VAR,
    SERVICE_MTLS_ENABLED_ENV_VAR,
    SERVICE_MTLS_RELOAD_INTERVAL_ENV_VAR,
    SERVICE_MTLS_TRUST_OVERLAP_ENV_VAR,
    ServiceMTLS,
    ServiceMTLSConfig,
    parse_service_mtls_config,
)
from openmed.service.tls_reload import (
    ACTIVATION_ACTIVATED,
    ACTIVATION_FAILED,
    ACTIVATION_NOT_ATTEMPTED,
    ACTIVATION_UNCHANGED,
    RELOAD_OUTCOME_ACTIVATED,
    RELOAD_OUTCOME_REJECTED,
    RELOAD_OUTCOME_UNCHANGED,
    TLS_CATEGORY_CERTIFICATE_CHAIN,
    TLS_CATEGORY_CLIENT_AUTH,
    TLS_CATEGORY_CLIENT_CA,
    TLS_CATEGORY_PRIVATE_KEY,
    TLS_CATEGORY_RELOAD,
    TLS_CATEGORY_SERVER_CA,
    TLS_MODE_CLIENT_TRUST,
    TLS_MODE_MTLS,
    TLS_MODE_TLS,
    TLS_REASON_ACTIVATED,
    TLS_REASON_ACTIVATION_FAILED,
    TLS_REASON_CANDIDATE_CHANGED,
    TLS_REASON_CERTIFICATE_EXPIRED,
    TLS_REASON_CERTIFICATE_EXPIRING,
    TLS_REASON_CERTIFICATE_NOT_YET_VALID,
    TLS_REASON_CHAIN_UNTRUSTED,
    TLS_REASON_CLIENT_AUTH_AMBIGUOUS,
    TLS_REASON_CLIENT_CA_INVALID,
    TLS_REASON_CLIENT_TRUST_UNENFORCEABLE,
    TLS_REASON_INCOMPLETE,
    TLS_REASON_KEY_MISMATCH,
    TLS_REASON_MALFORMED,
    TLS_REASON_POLICY_INCOMPATIBLE,
    TLS_REASON_RELOAD_IN_PROGRESS,
    TLS_REASON_SERVING,
    TLS_REASON_UNCHANGED,
    TLS_REASON_UNREADABLE,
    VALIDATION_INVALID,
    VALIDATION_NOT_RUN,
    VALIDATION_VALID,
    TLSActivationMaterial,
    TLSCredentialError,
    TLSCredentialProvider,
    TLSCredentialSource,
    TLSCredentialWatcher,
    install_reload_signal_handler,
)
from tests.fixtures.tls_reload import (
    CredentialFiles,
    FakeClock,
    SyntheticGeneration,
    assert_value_free,
    build_generation,
    credential_files,
    install,
    issue,
    write_atomic,
)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def generations(clock: FakeClock) -> dict[str, SyntheticGeneration]:
    return {label: build_generation(label, now=clock.now) for label in ("a", "b", "c")}


@pytest.fixture
def files(
    tmp_path: Path, generations: dict[str, SyntheticGeneration]
) -> CredentialFiles:
    files = credential_files(tmp_path / "credentials")
    install(files, generations["a"])
    return files


def _server_source(files: CredentialFiles, **overrides: Any) -> TLSCredentialSource:
    values: dict[str, Any] = {
        "certificate_chain_file": files.certificate_chain_file,
        "private_key_file": files.private_key_file,
        "client_ca_file": files.client_ca_file,
    }
    values.update(overrides)
    return TLSCredentialSource(**values)


def _client_trust_source(files: CredentialFiles) -> TLSCredentialSource:
    return TLSCredentialSource(client_ca_file=files.client_ca_file)


def _provider(
    files: CredentialFiles, clock: FakeClock, **kwargs: Any
) -> TLSCredentialProvider:
    source = kwargs.pop("source", None) or _server_source(files)
    return TLSCredentialProvider(source, clock=clock, **kwargs)


def test_initial_generation_is_validated_and_reported_value_free(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock)
    active = provider.active

    assert provider.mode == TLS_MODE_MTLS
    assert active.sequence == 1
    assert active.generation_id.startswith("tlsgen-")
    assert active.certificate_sha256 == generations["a"].server.sha256
    assert active.client_ca_sha256 == (generations["a"].root.sha256,)
    assert active.require_client_cert is True
    assert provider.readiness().serving is True
    assert provider.readiness().reason == TLS_REASON_SERVING
    rendered = json.dumps(provider.report()) + repr(active) + repr(provider.snapshot())
    assert_value_free(rendered, files, *generations.values())


def test_modes_follow_the_configured_files(
    files: CredentialFiles, clock: FakeClock
) -> None:
    tls_only = _provider(
        files, clock, source=_server_source(files, client_ca_file=None)
    )
    client_trust = _provider(files, clock, source=_client_trust_source(files))

    assert tls_only.mode == TLS_MODE_TLS
    assert tls_only.active.require_client_cert is False
    assert client_trust.mode == TLS_MODE_CLIENT_TRUST
    assert client_trust.active.certificate_sha256 is None
    assert client_trust.active.client_trust_store is not None


@pytest.mark.parametrize(
    ("overrides", "reason", "category"),
    [
        ({"private_key_file": None}, TLS_REASON_INCOMPLETE, TLS_CATEGORY_PRIVATE_KEY),
        (
            {"certificate_chain_file": None},
            TLS_REASON_INCOMPLETE,
            TLS_CATEGORY_CERTIFICATE_CHAIN,
        ),
        (
            {"client_ca_file": None, "require_client_cert": True},
            TLS_REASON_INCOMPLETE,
            TLS_CATEGORY_CLIENT_CA,
        ),
        (
            {"require_client_cert": False},
            TLS_REASON_CLIENT_AUTH_AMBIGUOUS,
            TLS_CATEGORY_CLIENT_AUTH,
        ),
        (
            {"require_client_cert": "yes"},
            TLS_REASON_CLIENT_AUTH_AMBIGUOUS,
            TLS_CATEGORY_CLIENT_AUTH,
        ),
        (
            {"private_key_file": b"key.pem"},
            TLS_REASON_INCOMPLETE,
            TLS_CATEGORY_PRIVATE_KEY,
        ),
    ],
)
def test_incomplete_or_ambiguous_configuration_fails_closed(
    files: CredentialFiles,
    clock: FakeClock,
    overrides: dict[str, Any],
    reason: str,
    category: str,
) -> None:
    with pytest.raises(TLSCredentialError) as error:
        _provider(files, clock, source=_server_source(files, **overrides))

    assert error.value.reason == reason
    assert error.value.category == category
    assert_value_free(str(error.value), files)


def test_client_trust_rejects_server_ca_or_optional_client_auth(
    files: CredentialFiles, clock: FakeClock
) -> None:
    with pytest.raises(TLSCredentialError) as server_ca:
        _provider(
            files,
            clock,
            source=TLSCredentialSource(
                client_ca_file=files.client_ca_file,
                server_ca_file=files.client_ca_file,
            ),
        )
    with pytest.raises(TLSCredentialError) as optional:
        _provider(
            files,
            clock,
            source=TLSCredentialSource(
                client_ca_file=files.client_ca_file, require_client_cert=False
            ),
        )
    with pytest.raises(TLSCredentialError) as empty:
        _provider(files, clock, source=TLSCredentialSource())

    assert server_ca.value.reason == TLS_REASON_INCOMPLETE
    assert optional.value.reason == TLS_REASON_CLIENT_AUTH_AMBIGUOUS
    assert empty.value.reason == TLS_REASON_INCOMPLETE


def test_generation_id_is_content_derived_and_stable_across_restart(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    first = _provider(files, clock)
    install(files, generations["b"])
    assert first.reload().outcome == RELOAD_OUTCOME_ACTIVATED

    restarted = _provider(files, clock)

    assert restarted.active.generation_id == first.active.generation_id
    assert restarted.active.sequence == 1
    assert first.active.sequence == 2


def test_restart_with_incomplete_material_fails_closed(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    write_atomic(files.certificate_chain_file, generations["b"].server.cert_pem)

    with pytest.raises(TLSCredentialError) as error:
        _provider(files, clock)

    assert error.value.reason == TLS_REASON_KEY_MISMATCH
    assert error.value.fingerprint_sha256 == generations["b"].server.sha256


def test_reload_of_identical_material_is_unchanged(
    files: CredentialFiles, clock: FakeClock
) -> None:
    provider = _provider(files, clock)
    before = provider.active

    result = provider.reload()

    assert result.outcome == RELOAD_OUTCOME_UNCHANGED
    assert result.reason == TLS_REASON_UNCHANGED
    assert result.validation_state == VALIDATION_VALID
    assert result.activation_state == ACTIVATION_UNCHANGED
    assert result.generation_id == result.previous_generation_id
    assert provider.active is before


def test_reload_activates_a_complete_candidate_with_typed_result(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock, trust_overlap_seconds=120)
    old = provider.active
    install(files, generations["b"])

    result = provider.reload()

    assert result.outcome == RELOAD_OUTCOME_ACTIVATED
    assert result.activated is True
    assert result.reason == TLS_REASON_ACTIVATED
    assert result.validation_state == VALIDATION_VALID
    assert result.activation_state == ACTIVATION_ACTIVATED
    assert result.previous_generation_id == old.generation_id
    assert result.generation_id == provider.active.generation_id
    assert result.candidate_generation_id == result.generation_id
    assert result.sequence == 2
    assert result.overlap_deadline == clock.now + timedelta(seconds=120)
    assert provider.active.certificate_sha256 == generations["b"].server.sha256
    assert provider.last_reload is result
    assert set(result.to_dict()) == {
        "outcome",
        "reason",
        "generation_id",
        "previous_generation_id",
        "candidate_generation_id",
        "validation_state",
        "activation_state",
        "overlap_deadline",
        "sequence",
        "completed_at",
        "category",
        "fingerprint_sha256",
    }


def _bad_candidates(
    files: CredentialFiles,
    generations: dict[str, SyntheticGeneration],
    clock: FakeClock,
) -> dict[str, tuple[Any, str, str]]:
    now = clock.now
    b = generations["b"]
    expired = issue(
        "openmed-reload.test",
        issuer=b.root,
        now=now,
        usage=None,
        not_before=now - timedelta(days=9),
        not_after=now - timedelta(days=1),
        key=b.server.key,
    )
    future = issue(
        "openmed-reload.test",
        issuer=b.root,
        now=now,
        not_before=now + timedelta(days=1),
        not_after=now + timedelta(days=9),
        key=b.server.key,
    )
    expiring = issue(
        "openmed-reload.test",
        issuer=b.root,
        now=now,
        not_before=now - timedelta(days=1),
        not_after=now + timedelta(seconds=60),
        key=b.server.key,
    )
    leaf_ca = issue("reload-synthetic-not-a-ca", issuer=None, now=now, ca=False)
    expired_ca = issue(
        "reload-synthetic-expired-ca",
        issuer=None,
        now=now,
        ca=True,
        not_before=now - timedelta(days=9),
        not_after=now - timedelta(days=1),
    )

    def cert_and_key(cert: bytes, key: bytes) -> None:
        write_atomic(files.certificate_chain_file, cert)
        write_atomic(files.private_key_file, key)

    return {
        "key_mismatch_partial_write": (
            lambda: write_atomic(files.certificate_chain_file, b.server.cert_pem),
            TLS_REASON_KEY_MISMATCH,
            TLS_CATEGORY_PRIVATE_KEY,
        ),
        "truncated_certificate": (
            lambda: write_atomic(
                files.certificate_chain_file,
                b.server.cert_pem[: len(b.server.cert_pem) // 2],
            ),
            TLS_REASON_MALFORMED,
            TLS_CATEGORY_CERTIFICATE_CHAIN,
        ),
        "empty_key": (
            lambda: write_atomic(files.private_key_file, b"\n"),
            TLS_REASON_MALFORMED,
            TLS_CATEGORY_PRIVATE_KEY,
        ),
        "missing_key": (
            lambda: files.private_key_file.unlink(),
            TLS_REASON_UNREADABLE,
            TLS_CATEGORY_PRIVATE_KEY,
        ),
        "oversized_chain": (
            lambda: write_atomic(
                files.certificate_chain_file,
                b.server.cert_pem + b"#" * tls_reload.MAX_CREDENTIAL_FILE_BYTES,
            ),
            TLS_REASON_MALFORMED,
            TLS_CATEGORY_CERTIFICATE_CHAIN,
        ),
        "expired_leaf": (
            lambda: cert_and_key(expired.cert_pem, b.server.key_pem),
            TLS_REASON_CERTIFICATE_EXPIRED,
            TLS_CATEGORY_CERTIFICATE_CHAIN,
        ),
        "not_yet_valid_leaf": (
            lambda: cert_and_key(future.cert_pem, b.server.key_pem),
            TLS_REASON_CERTIFICATE_NOT_YET_VALID,
            TLS_CATEGORY_CERTIFICATE_CHAIN,
        ),
        "expiring_leaf": (
            lambda: cert_and_key(expiring.cert_pem, b.server.key_pem),
            TLS_REASON_CERTIFICATE_EXPIRING,
            TLS_CATEGORY_CERTIFICATE_CHAIN,
        ),
        "broken_chain": (
            lambda: cert_and_key(
                b.server.cert_pem + generations["c"].root.cert_pem, b.server.key_pem
            ),
            TLS_REASON_CHAIN_UNTRUSTED,
            TLS_CATEGORY_CERTIFICATE_CHAIN,
        ),
        "client_ca_not_a_ca": (
            lambda: write_atomic(files.client_ca_file, leaf_ca.cert_pem),
            TLS_REASON_CLIENT_CA_INVALID,
            TLS_CATEGORY_CLIENT_CA,
        ),
        "client_ca_expired": (
            lambda: write_atomic(
                files.client_ca_file, b.root.cert_pem + expired_ca.cert_pem
            ),
            TLS_REASON_CERTIFICATE_EXPIRED,
            TLS_CATEGORY_CLIENT_CA,
        ),
        "client_ca_garbage": (
            lambda: write_atomic(files.client_ca_file, b"not a certificate\n"),
            TLS_REASON_MALFORMED,
            TLS_CATEGORY_CLIENT_CA,
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "key_mismatch_partial_write",
        "truncated_certificate",
        "empty_key",
        "missing_key",
        "oversized_chain",
        "expired_leaf",
        "not_yet_valid_leaf",
        "expiring_leaf",
        "broken_chain",
        "client_ca_not_a_ca",
        "client_ca_expired",
        "client_ca_garbage",
    ],
)
def test_invalid_candidate_keeps_previous_generation_fully_active(
    case: str,
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider(files, clock)
    before = provider.snapshot()
    mutate, reason, category = _bad_candidates(files, generations, clock)[case]
    mutate()

    with caplog.at_level(logging.DEBUG, logger=tls_reload.__name__):
        result = provider.reload()

    assert result.outcome == RELOAD_OUTCOME_REJECTED
    assert result.reason == reason
    assert result.category == category
    assert result.validation_state == VALIDATION_INVALID
    assert result.activation_state == ACTIVATION_NOT_ATTEMPTED
    assert result.generation_id == before.active.generation_id
    after = provider.snapshot()
    assert after.active is before.active
    assert after.artifacts is before.artifacts
    assert provider.readiness().serving is True
    assert provider.readiness().last_reload_reason == reason
    assert provider.metrics()["reload_rejected_by_reason"] == {reason: 1}
    rendered = json.dumps(result.to_dict()) + caplog.text
    rendered += json.dumps(provider.report())
    assert_value_free(rendered, files, *generations.values())


def test_server_ca_anchor_is_enforced(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    tmp_path: Path,
) -> None:
    server_ca = tmp_path / "server-ca.pem"
    server_ca.write_bytes(generations["a"].root.cert_pem)
    provider = _provider(
        files, clock, source=_server_source(files, server_ca_file=server_ca)
    )
    install(files, generations["b"])

    result = provider.reload()

    assert result.reason == TLS_REASON_CHAIN_UNTRUSTED
    assert result.category == TLS_CATEGORY_SERVER_CA
    write_atomic(server_ca, generations["b"].root.cert_pem)
    assert provider.reload().outcome == RELOAD_OUTCOME_ACTIVATED


def test_intermediate_chain_validates_against_server_ca(
    files: CredentialFiles,
    clock: FakeClock,
    tmp_path: Path,
) -> None:
    root = issue("reload-synthetic-chain-root", issuer=None, now=clock.now, ca=True)
    intermediate = issue(
        "reload-synthetic-chain-intermediate", issuer=root, now=clock.now, ca=True
    )
    leaf = issue("openmed-reload.test", issuer=intermediate, now=clock.now)
    write_atomic(files.certificate_chain_file, leaf.cert_pem + intermediate.cert_pem)
    write_atomic(files.private_key_file, leaf.key_pem)
    server_ca = tmp_path / "server-ca.pem"
    server_ca.write_bytes(root.cert_pem)

    provider = _provider(
        files, clock, source=_server_source(files, server_ca_file=server_ca)
    )

    assert provider.active.certificate_sha256 == leaf.sha256


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX file permissions do not restrict this user",
)
def test_unreadable_candidate_keeps_previous_generation(
    files: CredentialFiles, clock: FakeClock
) -> None:
    provider = _provider(files, clock)
    before = provider.active
    files.private_key_file.chmod(0)
    try:
        result = provider.reload()
    finally:
        files.private_key_file.chmod(0o600)

    assert result.reason == TLS_REASON_UNREADABLE
    assert result.category == TLS_CATEGORY_PRIVATE_KEY
    assert provider.active is before


def test_material_changing_during_validation_is_rejected(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(files, clock)
    before = provider.active
    install(files, generations["b"])
    real_read = tls_reload._read_bounded
    reads: dict[Path, int] = {}

    def racing_read(path: Path, category: str) -> bytes:
        reads[path] = reads.get(path, 0) + 1
        if path == files.client_ca_file and reads[path] == 2:
            write_atomic(files.client_ca_file, generations["c"].root.cert_pem)
        return real_read(path, category)

    monkeypatch.setattr(tls_reload, "_read_bounded", racing_read)
    result = provider.reload()

    assert result.reason == TLS_REASON_CANDIDATE_CHANGED
    assert result.category == TLS_CATEGORY_CLIENT_CA
    assert provider.active is before


def test_activation_failure_keeps_previous_generation(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    fail = {"enabled": False}

    def hook(material: TLSActivationMaterial) -> str:
        if fail["enabled"]:
            raise RuntimeError("synthetic activation failure")
        return material.generation.generation_id

    provider = _provider(files, clock, activation_hooks={"probe": hook})
    before = provider.snapshot()
    install(files, generations["b"])
    fail["enabled"] = True

    result = provider.reload()

    assert result.reason == TLS_REASON_ACTIVATION_FAILED
    assert result.validation_state == VALIDATION_VALID
    assert result.activation_state == ACTIVATION_FAILED
    assert result.category == TLS_CATEGORY_RELOAD
    assert provider.snapshot().active is before.active
    assert provider.snapshot().artifacts["probe"] == before.active.generation_id
    assert "synthetic activation failure" not in str(result.to_dict())


def test_activation_failure_at_startup_fails_closed(
    files: CredentialFiles, clock: FakeClock
) -> None:
    def hook(material: TLSActivationMaterial) -> None:
        raise RuntimeError("synthetic activation failure")

    with pytest.raises(TLSCredentialError) as error:
        _provider(files, clock, activation_hooks={"probe": hook})

    assert error.value.reason == TLS_REASON_ACTIVATION_FAILED
    assert error.value.__cause__ is None


def test_activation_material_is_cleared_and_never_serialized(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    seen: list[TLSActivationMaterial] = []

    def hook(material: TLSActivationMaterial) -> None:
        assert material.private_key_pem is not None
        with pytest.raises(TypeError):
            pickle.dumps(material)
        seen.append(material)

    provider = _provider(files, clock, activation_hooks={"probe": hook})
    install(files, generations["b"])
    provider.reload()

    assert len(seen) == 2
    for material in seen:
        assert material.private_key_pem is None
        assert material.certificate_chain_pem is None
        assert material.client_ca_pem is None
        assert material.overlap_client_ca_pem is None
        assert_value_free(repr(material), files, *generations.values())


def test_policy_incompatible_source_is_rejected(
    files: CredentialFiles, clock: FakeClock
) -> None:
    provider = _provider(files, clock)
    before = provider.active

    result = provider.reload(_server_source(files, client_ca_file=None))

    assert result.reason == TLS_REASON_POLICY_INCOMPATIBLE
    assert result.category == TLS_CATEGORY_CLIENT_AUTH
    assert provider.active is before
    assert provider.watched_files()[-1] == files.client_ca_file


def test_reload_can_switch_to_a_new_source_directory(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    tmp_path: Path,
) -> None:
    provider = _provider(files, clock)
    staged = credential_files(tmp_path / "next")
    install(staged, generations["b"])

    result = provider.reload(_server_source(staged))

    assert result.outcome == RELOAD_OUTCOME_ACTIVATED
    assert provider.watched_files()[0] == staged.certificate_chain_file
    assert provider.reload(object()).reason == TLS_REASON_INCOMPLETE  # type: ignore[arg-type]


def test_concurrent_reload_is_rejected_deterministically(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = {"count": 0}

    def hook(material: TLSActivationMaterial) -> None:
        calls["count"] += 1
        if calls["count"] > 1:
            entered.set()
            assert release.wait(10)

    provider = _provider(files, clock, activation_hooks={"probe": hook})
    install(files, generations["b"])
    results: list[Any] = []
    worker = threading.Thread(target=lambda: results.append(provider.reload()))
    worker.start()
    assert entered.wait(10)

    concurrent = provider.reload()
    release.set()
    worker.join(10)

    assert concurrent.outcome == RELOAD_OUTCOME_REJECTED
    assert concurrent.reason == TLS_REASON_RELOAD_IN_PROGRESS
    assert concurrent.validation_state == VALIDATION_NOT_RUN
    assert concurrent.category == TLS_CATEGORY_RELOAD
    assert concurrent.generation_id == concurrent.previous_generation_id
    assert results[0].outcome == RELOAD_OUTCOME_ACTIVATED
    assert provider.active.sequence == 2


def test_trust_overlap_is_bounded_then_retired(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock, trust_overlap_seconds=60)
    old = provider.active
    install(files, generations["b"])
    provider.reload()
    new = provider.active

    during = provider.snapshot()
    assert during.trusted_generations == (new, old)
    assert provider.readiness().overlap_active is True
    clock.advance(59)
    assert provider.snapshot().overlap is old
    clock.advance(1)

    after = provider.snapshot()
    assert after.trusted_generations == (new,)
    assert after.overlap_deadline is None
    assert provider.metrics()["generations_retired_total"] == 1
    assert provider.metrics()["retained_generations"] == 1


def test_no_overlap_when_disabled_or_client_ca_unchanged(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    disabled = _provider(files, clock, trust_overlap_seconds=0)
    same_ca = _provider(files, clock)
    replacement = issue(
        "openmed-reload.test", issuer=generations["a"].root, now=clock.now
    )
    install(files, generations["b"])
    disabled_result = disabled.reload()
    write_atomic(files.certificate_chain_file, replacement.cert_pem)
    write_atomic(files.private_key_file, replacement.key_pem)
    write_atomic(files.client_ca_file, generations["a"].root.cert_pem)
    same_ca_result = same_ca.reload()

    assert disabled_result.overlap_deadline is None
    assert disabled.snapshot().overlap is None
    assert same_ca_result.outcome == RELOAD_OUTCOME_ACTIVATED
    assert same_ca_result.overlap_deadline is None


def test_reload_during_overlap_retires_the_oldest_generation(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock, trust_overlap_seconds=300)
    install(files, generations["b"])
    provider.reload()
    b = provider.active
    install(files, generations["c"])
    provider.reload()

    snapshot = provider.snapshot()

    assert snapshot.trusted_generations == (provider.active, b)
    assert provider.metrics()["retained_generations"] == 2
    assert provider.metrics()["generations_retired_total"] == 1


def test_overlap_material_reaches_activation_hooks(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    overlaps: list[bool] = []

    def hook(material: TLSActivationMaterial) -> None:
        overlaps.append(
            material.overlap_client_ca_pem == generations["a"].root.cert_pem
        )

    provider = _provider(files, clock, activation_hooks={"probe": hook})
    install(files, generations["b"])
    provider.reload()

    assert overlaps == [False, True]


def test_rollback_restores_the_prior_generation_id(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock)
    original = provider.active.generation_id
    install(files, generations["b"])
    provider.reload()
    install(files, generations["a"])

    result = provider.reload()

    assert result.outcome == RELOAD_OUTCOME_ACTIVATED
    assert provider.active.generation_id == original
    assert provider.active.sequence == 3


def test_expiry_edges_are_deterministic(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    window = 300
    b = generations["b"]
    edge = issue(
        "openmed-reload.test",
        issuer=b.root,
        now=clock.now,
        not_before=clock.now - timedelta(hours=1),
        not_after=clock.now + timedelta(seconds=window + 10),
        key=b.server.key,
    )
    # X.509 times have one-second precision; anchor the edges on the parsed value.
    not_before = edge.certificate.not_valid_before_utc
    not_after = edge.certificate.not_valid_after_utc
    write_atomic(files.client_ca_file, b.root.cert_pem)
    write_atomic(files.certificate_chain_file, edge.cert_pem)
    write_atomic(files.private_key_file, b.server.key_pem)

    clock.now = not_after - timedelta(seconds=window)
    exactly = _provider(files, clock, min_remaining_validity_seconds=window)
    clock.now = not_after - timedelta(seconds=window - 1)
    with pytest.raises(TLSCredentialError) as late:
        _provider(files, clock, min_remaining_validity_seconds=window)

    assert exactly.active.certificate_sha256 == edge.sha256
    assert late.value.reason == TLS_REASON_CERTIFICATE_EXPIRING
    assert late.value.fingerprint_sha256 == edge.sha256
    assert exactly.readiness(not_after).serving is True
    expired = exactly.readiness(not_after + timedelta(microseconds=1))
    assert expired.serving is False
    assert expired.reason == TLS_REASON_CERTIFICATE_EXPIRED
    assert exactly.readiness(not_before).serving is True
    early = exactly.readiness(not_before - timedelta(microseconds=1))
    assert early.reason == TLS_REASON_CERTIFICATE_NOT_YET_VALID


def test_readiness_reports_unenforceable_client_trust(
    files: CredentialFiles, clock: FakeClock, tmp_path: Path
) -> None:
    root = issue(
        "reload-synthetic-short-root",
        issuer=None,
        now=clock.now,
        ca=True,
        not_after=clock.now + timedelta(days=1),
    )
    ca_file = tmp_path / "short-ca.pem"
    ca_file.write_bytes(root.cert_pem)
    provider = TLSCredentialProvider(
        TLSCredentialSource(client_ca_file=ca_file), clock=clock
    )
    assert provider.readiness().client_trust_enforceable is True

    clock.advance(2 * 86_400)
    readiness = provider.readiness()

    assert readiness.serving is False
    assert readiness.client_trust_enforceable is False
    assert readiness.reason == TLS_REASON_CLIENT_TRUST_UNENFORCEABLE


def test_repeated_reload_cycles_release_generations_and_handles(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock, trust_overlap_seconds=30)
    refs: list[weakref.ref[Any]] = [weakref.ref(provider.active)]
    fd_directory = Path("/dev/fd")
    fds_before = len(os.listdir(fd_directory)) if fd_directory.is_dir() else None

    for cycle in range(24):
        install(files, generations["b" if cycle % 2 == 0 else "a"])
        assert provider.reload().outcome == RELOAD_OUTCOME_ACTIVATED
        refs.append(weakref.ref(provider.active))
        if cycle % 3 == 0:
            clock.advance(31)
        assert provider.metrics()["retained_generations"] <= 2

    clock.advance(31)
    provider.snapshot()
    gc.collect()

    alive = [ref for ref in refs if ref() is not None]
    assert alive == [refs[-1]]
    assert provider.metrics()["active_generation_sequence"] == 25
    if fds_before is not None:
        assert len(os.listdir(fd_directory)) <= fds_before


def test_invalid_provider_parameters_are_refused(
    files: CredentialFiles, clock: FakeClock
) -> None:
    for value in (-1, float("nan"), tls_reload.MAX_TRUST_OVERLAP_SECONDS + 1, "60"):
        with pytest.raises(ValueError):
            _provider(files, clock, trust_overlap_seconds=value)
    with pytest.raises(ValueError):
        _provider(files, clock, activation_hooks={"Bad Name": lambda material: None})
    with pytest.raises(TypeError):
        _provider(files, clock, activation_hooks={"probe": "not callable"})
    with pytest.raises(TLSCredentialError):
        TLSCredentialProvider(object())  # type: ignore[arg-type]


def test_source_and_error_reprs_hide_paths(
    files: CredentialFiles, generations: dict[str, SyntheticGeneration]
) -> None:
    error = TLSCredentialError(
        "uncontrolled reason", category="/etc/secret", fingerprint_sha256="nope"
    )

    assert_value_free(repr(_server_source(files)), files)
    assert error.reason == tls_reload.TLS_REASON_FAILED
    assert error.category is None
    assert error.fingerprint_sha256 is None


def test_watcher_reloads_only_when_metadata_changes(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock)
    watcher = TLSCredentialWatcher(provider, interval_seconds=0.05)

    first = watcher.poll_once()
    assert first is not None and first.outcome == RELOAD_OUTCOME_UNCHANGED
    assert watcher.poll_once() is None

    write_atomic(files.certificate_chain_file, generations["b"].server.cert_pem)
    partial = watcher.poll_once()
    assert partial is not None and partial.reason == TLS_REASON_KEY_MISMATCH
    assert watcher.poll_once() is None

    write_atomic(files.private_key_file, generations["b"].server.key_pem)
    write_atomic(files.client_ca_file, generations["b"].root.cert_pem)
    completed = watcher.poll_once()
    assert completed is not None and completed.outcome == RELOAD_OUTCOME_ACTIVATED


def test_watcher_thread_starts_and_stops_without_leaks(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock)

    def watcher_threads() -> list[threading.Thread]:
        return [t for t in threading.enumerate() if t.name == "openmed-tls-reload"]

    baseline = len(watcher_threads())
    for _ in range(3):
        watcher = TLSCredentialWatcher(provider, interval_seconds=0.01)
        with watcher:
            watcher.start()
            assert watcher.running
        assert not watcher.running
    assert len(watcher_threads()) == baseline

    install(files, generations["b"])
    with TLSCredentialWatcher(provider, interval_seconds=0.01):
        deadline = threading.Event()
        for _ in range(500):
            if provider.active.sequence == 2:
                break
            deadline.wait(0.01)
    assert provider.active.sequence == 2
    assert len(watcher_threads()) == baseline


def test_watcher_rejects_invalid_arguments(
    files: CredentialFiles, clock: FakeClock
) -> None:
    provider = _provider(files, clock)
    with pytest.raises(TypeError):
        TLSCredentialWatcher(object(), interval_seconds=1)
    for interval in (0, -1, tls_reload.MAX_WATCH_INTERVAL_SECONDS + 1):
        with pytest.raises(ValueError):
            TLSCredentialWatcher(provider, interval_seconds=interval)


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP is POSIX only")
def test_sighup_reloads_every_target(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers can only be installed from the main thread")
    provider = _provider(files, clock)
    other = _provider(files, clock, source=_client_trust_source(files))
    install(files, generations["b"])
    previous = signal.getsignal(signal.SIGHUP)
    restore = install_reload_signal_handler(provider, other)
    try:
        signal.raise_signal(signal.SIGHUP)
        waiter = threading.Event()
        for _ in range(500):
            if provider.active.sequence == 2 and other.active.sequence == 2:
                break
            waiter.wait(0.01)
    finally:
        restore()

    assert provider.active.sequence == 2
    assert other.active.sequence == 2
    assert signal.getsignal(signal.SIGHUP) == previous


def test_signal_handler_requires_targets() -> None:
    with pytest.raises(ValueError):
        install_reload_signal_handler()
    with pytest.raises(ValueError):
        install_reload_signal_handler(object())  # type: ignore[arg-type]


def test_grpc_reloader_fetcher_switches_only_on_change(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    reloader = GrpcCredentialReloader(
        GrpcTLSConfig(
            certificate_chain_file=files.certificate_chain_file,
            private_key_file=files.private_key_file,
            client_ca_file=files.client_ca_file,
        ),
        trust_overlap_seconds=60,
        clock=clock,
    )
    first_id = reloader.installed_generation_id

    assert reloader._fetch_certificate_configuration() is None
    install(files, generations["b"])
    assert reloader.reload().outcome == RELOAD_OUTCOME_ACTIVATED
    overlap_config = reloader._fetch_certificate_configuration()
    assert overlap_config is reloader.provider.snapshot().artifacts["grpc"].overlap
    assert reloader._fetch_certificate_configuration() is None
    clock.advance(60)
    steady_config = reloader._fetch_certificate_configuration()

    assert steady_config is reloader.provider.snapshot().artifacts["grpc"].steady
    assert reloader.installed_generation_id != first_id
    assert reloader.report()["handshake_config_switches"] == 2
    assert_value_free(json.dumps(reloader.report()), files, *generations.values())


def test_grpc_reloader_fetcher_never_raises(
    files: CredentialFiles,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloader = GrpcCredentialReloader(_server_source(files), clock=clock)

    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("synthetic snapshot failure")

    monkeypatch.setattr(reloader.provider, "snapshot", broken)

    assert reloader._fetch_certificate_configuration() is None


def test_grpc_reloader_binds_through_listener_policy(
    files: CredentialFiles, clock: FakeClock
) -> None:
    reloader = GrpcCredentialReloader(_server_source(files), clock=clock)

    listener = resolve_grpc_listener("0.0.0.0:50051", tls=reloader, profile="prod")

    assert listener.transport == "mtls"
    assert listener.credentials is not None
    assert (
        listener.credentials.certificate_sha256
        == reloader.provider.active.certificate_sha256
    )
    assert dict(reloader.server_options()) == {
        "grpc.max_connection_age_ms": 3_600_000,
        "grpc.max_connection_age_grace_ms": 300_000,
    }
    unbounded = GrpcCredentialReloader(
        _server_source(files), clock=clock, max_connection_age_seconds=None
    )
    assert unbounded.server_options() == ()


def test_grpc_reloader_requires_server_material_and_valid_ages(
    files: CredentialFiles, clock: FakeClock
) -> None:
    with pytest.raises(TLSCredentialError) as client_only:
        GrpcCredentialReloader(_client_trust_source(files), clock=clock)
    with pytest.raises(TLSCredentialError):
        GrpcCredentialReloader(object(), clock=clock)  # type: ignore[arg-type]
    for value in (0, -1, float("inf"), "60"):
        with pytest.raises(ValueError):
            GrpcCredentialReloader(
                _server_source(files), clock=clock, max_connection_age_seconds=value
            )

    assert client_only.value.reason == TLS_REASON_INCOMPLETE


def test_grpc_reloader_accepts_a_replacement_grpc_config(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    tmp_path: Path,
) -> None:
    reloader = GrpcCredentialReloader(_server_source(files), clock=clock)
    staged = credential_files(tmp_path / "next")
    install(staged, generations["b"])

    result = reloader.reload(
        GrpcTLSConfig(
            certificate_chain_file=staged.certificate_chain_file,
            private_key_file=staged.private_key_file,
            client_ca_file=staged.client_ca_file,
        )
    )

    assert result.outcome == RELOAD_OUTCOME_ACTIVATED
    assert reloader.readiness().serving is True
    assert reloader.watched_files()[0] == staged.certificate_chain_file


def test_rest_mtls_config_parses_reload_settings(
    monkeypatch: pytest.MonkeyPatch, files: CredentialFiles
) -> None:
    monkeypatch.setenv(SERVICE_MTLS_ENABLED_ENV_VAR, "true")
    monkeypatch.setenv(SERVICE_MTLS_CA_BUNDLE_ENV_VAR, str(files.client_ca_file))
    monkeypatch.delenv(SERVICE_MTLS_RELOAD_INTERVAL_ENV_VAR, raising=False)
    monkeypatch.delenv(SERVICE_MTLS_TRUST_OVERLAP_ENV_VAR, raising=False)
    defaults = parse_service_mtls_config()
    monkeypatch.setenv(SERVICE_MTLS_RELOAD_INTERVAL_ENV_VAR, "5")
    monkeypatch.setenv(SERVICE_MTLS_TRUST_OVERLAP_ENV_VAR, "0")
    configured = parse_service_mtls_config()

    assert defaults.reload_interval_seconds is None
    assert defaults.trust_overlap_seconds == tls_reload.DEFAULT_TRUST_OVERLAP_SECONDS
    assert configured.reload_interval_seconds == 5.0
    assert configured.trust_overlap_seconds == 0.0
    for env_var, value in (
        (SERVICE_MTLS_RELOAD_INTERVAL_ENV_VAR, "0.5"),
        (SERVICE_MTLS_RELOAD_INTERVAL_ENV_VAR, "soon"),
        (SERVICE_MTLS_TRUST_OVERLAP_ENV_VAR, "-1"),
        (SERVICE_MTLS_TRUST_OVERLAP_ENV_VAR, "nan"),
    ):
        monkeypatch.setenv(env_var, value)
        with pytest.raises(ValueError, match=env_var):
            parse_service_mtls_config()
        monkeypatch.delenv(env_var)


def test_rest_mtls_uses_a_client_trust_generation(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    config = ServiceMTLSConfig(enabled=True, ca_bundle=files.client_ca_file)
    mtls = ServiceMTLS(config, error_response=lambda *a, **k: None)
    disabled = ServiceMTLS(ServiceMTLSConfig(), error_response=lambda *a, **k: None)

    assert mtls.credentials is not None
    assert mtls.credentials.mode == TLS_MODE_CLIENT_TRUST
    assert mtls.readiness() is not None and mtls.readiness().serving
    install(files, generations["b"])
    assert mtls.reload().outcome == RELOAD_OUTCOME_ACTIVATED
    assert disabled.reload() is None
    assert disabled.readiness() is None
    disabled.start_reload_watcher()
    disabled.stop_reload_watcher()


def test_rest_mtls_refuses_invalid_bundles_and_server_only_providers(
    files: CredentialFiles, clock: FakeClock, tmp_path: Path
) -> None:
    garbage = tmp_path / "garbage.pem"
    garbage.write_bytes(b"not a certificate\n")
    tls_only = _provider(
        files, clock, source=_server_source(files, client_ca_file=None)
    )

    with pytest.raises(ValueError) as invalid:
        ServiceMTLS(
            ServiceMTLSConfig(enabled=True, ca_bundle=garbage),
            error_response=lambda *a, **k: None,
        )
    with pytest.raises(ValueError):
        ServiceMTLS(
            ServiceMTLSConfig(enabled=True),
            error_response=lambda *a, **k: None,
        )
    with pytest.raises(ValueError):
        ServiceMTLS(
            ServiceMTLSConfig(enabled=True),
            error_response=lambda *a, **k: None,
            credentials=tls_only,
        )

    assert isinstance(invalid.value, TLSCredentialError)
    assert invalid.value.reason == TLS_REASON_MALFORMED
    assert str(garbage) not in str(invalid.value)


def test_generation_replace_keeps_value_free_repr(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    provider = _provider(files, clock)
    copy = replace(provider.active, sequence=9)

    assert copy.sequence == 9
    assert_value_free(repr(copy), files, *generations.values())
