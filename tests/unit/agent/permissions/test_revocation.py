"""Offline negative controls for runtime status, generations and safe receipts."""

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from openmed.agent.permissions.revocation import (
    AuthorityBinding,
    AuthorityBoundary,
    AuthorityContract,
    AuthorityDeniedError,
    AuthorityKind,
    AuthorityReason,
    AuthorityRuntime,
    AuthorityStatus,
    AuthorityVersion,
    InMemoryAuthorityGenerationStore,
)
from tests.fixtures.agent.authority_status import SyntheticAuthority


@pytest.mark.parametrize("kind", list(AuthorityKind))
@pytest.mark.parametrize("boundary", list(AuthorityBoundary))
def test_fresh_status_checked_at_every_boundary(kind, boundary):
    fixture = SyntheticAuthority()
    contract = next(c for c in fixture.contracts if c.kind is kind)
    binding = fixture.runtime.bind((contract,))
    calls = fixture.provider.calls
    assert fixture.runtime.check(binding, boundary).reason is AuthorityReason.ACTIVE
    assert fixture.provider.calls == calls + 1
    fixture.provider.revoke(contract)
    with pytest.raises(AuthorityDeniedError) as error:
        fixture.runtime.check(binding, boundary)
    assert error.value.receipt.reason is AuthorityReason.REVOKED


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("missing", AuthorityReason.UNAVAILABLE),
        ("unavailable", AuthorityReason.UNAVAILABLE),
        ("wrong_digest", AuthorityReason.UNAVAILABLE),
        ("wrong_kind", AuthorityReason.UNAVAILABLE),
        ("invalid_response", AuthorityReason.UNAVAILABLE),
        ("stale", AuthorityReason.STALE),
        ("future", AuthorityReason.STALE),
        ("lower_generation", AuthorityReason.ROLLBACK),
        ("changed_generation", AuthorityReason.GENERATION_CHANGED),
    ],
)
def test_status_failures_are_closed_and_content_free(failure, reason):
    fixture = SyntheticAuthority()
    contract = fixture.contracts[0]
    fixture.provider.records[(contract.kind, contract.digest)] = (2, False)
    binding = fixture.runtime.bind((contract,))
    status = fixture.current_status(contract)
    if failure == "missing":
        fixture.provider.records.clear()
    elif failure == "unavailable":
        fixture.provider.unavailable = True
    else:
        changes = {
            "wrong_digest": {"digest": "sha256:" + "f" * 64},
            "wrong_kind": {"kind": AuthorityKind.PURPOSE_TICKET},
            "stale": {"observed_at": fixture.clock() - 1},
            "future": {"observed_at": fixture.clock() + 1},
            "lower_generation": {"generation": 1},
            "changed_generation": {"generation": 3},
        }
        fixture.provider.override = (
            object()
            if failure == "invalid_response"
            else replace(status, **changes[failure])
        )
    with pytest.raises(AuthorityDeniedError) as error:
        fixture.runtime.check(binding, AuthorityBoundary.EFFECT_DISPATCH)
    assert error.value.receipt.reason is reason
    output = json.dumps(error.value.receipt.to_dict()) + "".join(
        traceback.format_exception(error.value)
    )
    assert "synthetic-private-provider-payload" not in output
    assert fixture.ticket.purpose not in output
    assert fixture.grant.signature not in output
    assert fixture.ticket.record_selectors[0].digest not in output


def test_expiry_is_distinct_from_revocation_even_with_historical_clock():
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind((fixture.contracts[0],))
    fixture.clock.now = 100
    with pytest.raises(AuthorityDeniedError) as error:
        fixture.runtime.check(binding, AuthorityBoundary.RESUME)
    assert error.value.code == "authority_expired"
    fixture.provider.revoke(fixture.contracts[0])
    with pytest.raises(AuthorityDeniedError) as error:
        fixture.runtime.check(binding, AuthorityBoundary.RESUME)
    assert error.value.code == "authority_revoked"


def test_restored_high_generation_cannot_be_downgraded_by_fresh_low_status():
    fixture = SyntheticAuthority()
    original = fixture.runtime.bind((fixture.contracts[0],))
    binding = AuthorityBinding((replace(original.versions[0], generation=8),))
    restarted = AuthorityRuntime(
        fixture.provider, InMemoryAuthorityGenerationStore(), clock=fixture.clock
    )
    with pytest.raises(AuthorityDeniedError, match="authority_status_rollback"):
        restarted.check(binding, AuthorityBoundary.RESUME)


@pytest.mark.parametrize("generation", [0, 1, 2])
def test_tombstone_rejects_resurrection_at_any_generation(generation):
    store = InMemoryAuthorityGenerationStore()
    status = AuthorityStatus(AuthorityKind.GRANT, "sha256:" + "a" * 64, 1, True, 10)
    assert store.observe(status)
    assert not store.observe(replace(status, generation=generation, revoked=False))
    assert store.observe(replace(status, generation=2))


def test_equal_generation_cannot_change_state_and_concurrent_rollback_is_rejected():
    store = InMemoryAuthorityGenerationStore()
    status = AuthorityStatus(AuthorityKind.GRANT, "sha256:" + "a" * 64, 0, False, 10)
    assert store.observe(status)
    assert not store.observe(replace(status, revoked=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                store.observe, [replace(status, generation=g) for g in range(1, 20)]
            )
        )
    assert not store.observe(replace(status, generation=18))
    assert store.observe(replace(status, generation=19))


@pytest.mark.parametrize("failure", ["exception", "invalid"])
def test_generation_store_failure_is_closed(failure):
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind((fixture.contracts[0],))

    class BadStore:
        def observe(self, status):
            if failure == "exception":
                raise RuntimeError("synthetic-private-store-payload")
            return 1

    runtime = AuthorityRuntime(fixture.provider, BadStore(), clock=fixture.clock)
    with pytest.raises(
        AuthorityDeniedError, match="authority_generation_store_unavailable"
    ) as error:
        runtime.check(binding, AuthorityBoundary.RESUME)
    assert "synthetic-private-store-payload" not in "".join(
        traceback.format_exception(error.value)
    )


@pytest.mark.parametrize("value", [-1, True, 2**63, "synthetic-private-value"])
def test_clock_failure_is_closed(value):
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind((fixture.contracts[0],))
    runtime = AuthorityRuntime(fixture.provider, fixture.store, clock=lambda: value)
    with pytest.raises(AuthorityDeniedError, match="authority_status_unavailable"):
        runtime.check(binding, AuthorityBoundary.RESUME)


def test_explicit_bounded_age_policy_and_unknown_preview():
    fixture = SyntheticAuthority()
    contract = fixture.contracts[0]
    fixture.provider.override = replace(fixture.current_status(contract), observed_at=8)
    runtime = AuthorityRuntime(
        fixture.provider, fixture.store, clock=fixture.clock, max_status_age=2
    )
    binding = runtime.bind((contract,))
    assert (
        runtime.check(binding, AuthorityBoundary.SENSITIVE_READ).reason
        is AuthorityReason.ACTIVE
    )
    fixture.clock.now = 11
    with pytest.raises(AuthorityDeniedError, match="authority_status_stale"):
        runtime.check(binding, AuthorityBoundary.RESUME)
    fixture.provider.override = None
    fixture.provider.records.clear()
    with pytest.raises(AuthorityDeniedError, match="authority_status_unavailable"):
        runtime.bind((contract,))


@pytest.mark.parametrize(
    "max_age,reason", [(0, "authority_status_stale"), (1000, "authority_expired")]
)
def test_slow_generation_store_cannot_admit_aged_or_expired_status(max_age, reason):
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind((fixture.contracts[0],))

    class DelayedStore:
        def observe(self, status):
            fixture.clock.now = 100
            return True

    runtime = AuthorityRuntime(
        fixture.provider, DelayedStore(), clock=fixture.clock, max_status_age=max_age
    )
    with pytest.raises(AuthorityDeniedError, match=reason):
        runtime.check(binding, AuthorityBoundary.EFFECT_DISPATCH)


def test_binding_roundtrip_preserves_exact_generations_and_canonical_digest():
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind(fixture.contracts)
    restored = AuthorityBinding.from_dict(json.loads(json.dumps(binding.to_dict())))
    assert restored == binding
    assert restored.digest() == binding.digest()
    assert (
        AuthorityBinding(tuple(reversed(binding.versions))).digest() == binding.digest()
    )


@pytest.mark.parametrize(
    "mutation", ["extra", "missing", "schema", "generation", "kind", "duplicate"]
)
def test_malformed_recovery_binding_is_rejected(mutation):
    fixture = SyntheticAuthority()
    payload = fixture.runtime.bind((fixture.contracts[0],)).to_dict()
    if mutation == "extra":
        payload["source"] = "synthetic-private-payload"
    elif mutation == "missing":
        del payload["versions"][0]["generation"]
    elif mutation == "schema":
        payload["schema_version"] = "future"
    elif mutation == "generation":
        payload["versions"][0]["generation"] = True
    elif mutation == "kind":
        payload["versions"][0]["kind"] = "synthetic-private-payload"
    else:
        payload["versions"] *= 2
    with pytest.raises(ValueError, match="^invalid_authority_binding$"):
        AuthorityBinding.from_dict(payload)


def test_preview_denies_revoked_expired_or_duplicate_contracts():
    fixture = SyntheticAuthority()
    contract = fixture.contracts[0]
    with pytest.raises(ValueError, match="duplicate_authority_contract"):
        fixture.runtime.bind((contract, contract))
    fixture.clock.now = 100
    with pytest.raises(AuthorityDeniedError, match="authority_expired"):
        fixture.runtime.bind((contract,))
    fixture.provider.revoke(contract)
    with pytest.raises(AuthorityDeniedError, match="authority_revoked"):
        fixture.runtime.bind((contract,))


@pytest.mark.parametrize("generation", [-1, True, 2**63])
def test_contract_and_status_validate_controlled_metadata(generation):
    contract = AuthorityContract(AuthorityKind.GRANT, "sha256:" + "a" * 64, 100)
    with pytest.raises(ValueError, match="invalid_authority_integer"):
        AuthorityVersion(contract, generation)
    with pytest.raises(ValueError, match="invalid_authority_integer"):
        AuthorityStatus(contract.kind, contract.digest, generation, False, 10)
