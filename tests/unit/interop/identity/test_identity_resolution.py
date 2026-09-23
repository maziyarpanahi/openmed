"""Synthetic patient and encounter identity-resolution tests."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.clinical.journey_contracts import canonical_digest
from openmed.interop.identity import (
    IDENTITY_SCHEMA_NAMES,
    CompositeIdentityResolver,
    ExactIdentityResolver,
    IdentityContractError,
    IdentityEvidence,
    IdentityLink,
    IdentityResolution,
    IdentityResolutionRequest,
    IdentityResolutionStore,
    IdentityReviewDecision,
    ProbabilisticIdentityCandidate,
    SourceIdentityKey,
    load_all_identity_schemas,
)
from openmed.structured.store import StoreResult, StoreState

T0 = "2026-01-02T03:04:05Z"
T1 = "2026-01-02T04:04:05Z"
POLICY_ID = "openmed.identity.default"
POLICY_VERSION = "1.0.0"


def _source(index: int, *, entity_type: str = "patient") -> SourceIdentityKey:
    return SourceIdentityKey(
        entity_type=entity_type,
        source_id=f"source_{index:016d}",
        local_key=f"local_{index:016d}",
    )


def _canonical(index: int, *, entity_type: str = "patient") -> str:
    return f"{entity_type}_{index:016d}"


def _request(
    *keys: SourceIdentityKey,
    role: str = "clinician",
    attributes: tuple[str, ...] = ("identified_access",),
) -> IdentityResolutionRequest:
    return IdentityResolutionRequest(
        request_id="request_aaaaaaaaaaaaaaaa",
        entity_type=keys[0].entity_type,
        source_keys=tuple(keys),
        purpose="care",
        role=role,
        attributes=attributes,
        policy_id=POLICY_ID,
        policy_version=POLICY_VERSION,
        requested_at=T0,
    )


def _link(
    source: SourceIdentityKey,
    canonical_key: str,
    index: int,
) -> IdentityLink:
    return IdentityLink(
        link_id=f"link_{index:016d}",
        source_key=source,
        canonical_key=canonical_key,
        evidence_digest=canonical_digest({"evidence": index, "synthetic": True}),
        policy_id=POLICY_ID,
        policy_version=POLICY_VERSION,
        recorded_at=T0,
    )


def _review(
    resolution: IdentityResolution,
    action: str,
    selected: str | None,
    index: int = 1,
) -> IdentityReviewDecision:
    return IdentityReviewDecision(
        decision_id=f"decision_{index:016d}",
        resolution_id=resolution.resolution_id,
        action=action,
        selected_canonical_key=selected,
        reviewer_digest=canonical_digest({"reviewer": "synthetic"}),
        evidence_digest=canonical_digest({"review": index, "synthetic": True}),
        policy_id=resolution.policy_id,
        policy_version=resolution.policy_version,
        decided_at=T1,
    )


def test_contracts_round_trip_without_raw_identifiers() -> None:
    canary = "SYNTHETIC raw patient identifier"
    request = _request(_source(1))
    link = _link(_source(1), _canonical(1), 1)

    assert SourceIdentityKey.from_json(_source(1).to_json()) == _source(1)
    assert IdentityResolutionRequest.from_json(request.to_json()) == request
    assert IdentityLink.from_json(link.to_json()) == link
    assert canary not in request.to_json()
    assert canary not in repr(request)

    with pytest.raises(IdentityContractError) as captured:
        replace(request, role=canary)
    assert canary not in str(captured.value)


def test_golden_identity_journey_fixture(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/interop/identity_resolution_golden.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    request = IdentityResolutionRequest.from_dict(fixture["request"])
    links = tuple(IdentityLink.from_dict(item) for item in fixture["links"])
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    for link in links:
        assert store.add_link(link).ok

    result = ExactIdentityResolver(store).resolve(request)

    assert result.ok and result.value is not None
    actual = {
        "candidate_keys": list(result.value.candidate_keys),
        "canonical_key": result.value.canonical_key,
        "evidence_count": len(result.value.evidence),
        "review_required": result.value.review_required,
        "state": result.value.state,
    }
    assert fixture["schema_version"] == 1
    assert actual == fixture["expected"]
    store.close()


def test_public_identity_records_validate_against_bundled_schemas(
    tmp_path: Path,
) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    key = _source(1)
    first = _link(key, _canonical(1), 1)
    second = _link(key, _canonical(2), 2)
    assert store.add_link(first).ok
    matched = ExactIdentityResolver(store).resolve(_request(key)).value
    assert matched is not None and matched.evidence
    assert store.add_link(second).ok
    conflict = ExactIdentityResolver(store).resolve(_request(key)).value
    assert conflict is not None
    review = _review(conflict, "merge", _canonical(1))

    records = {
        "source_key": key,
        "request": _request(key),
        "evidence": matched.evidence[0],
        "resolution": conflict,
        "link": first,
        "review_decision": review,
    }
    schemas = load_all_identity_schemas()

    assert set(schemas) == set(IDENTITY_SCHEMA_NAMES)
    for name, record in records.items():
        schema = schemas[name]
        validator = validator_for(schema)
        validator.check_schema(schema)
        assert schema["schema_version"] == 1
        assert not tuple(validator(schema).iter_errors(record.to_dict()))
        assert type(record).from_json(record.to_json()) == record

    invalid_uncertain = conflict.to_dict()
    invalid_uncertain["review_required"] = False
    resolution_schema = schemas["resolution"]
    assert tuple(
        validator_for(resolution_schema)(resolution_schema).iter_errors(
            invalid_uncertain
        )
    )
    store.close()


def test_resolution_rejects_duplicate_source_keys_and_evidence(
    tmp_path: Path,
) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    key = _source(1)
    assert store.add_link(_link(key, _canonical(1), 1)).ok
    resolution = ExactIdentityResolver(store).resolve(_request(key)).value
    assert resolution is not None

    with pytest.raises(IdentityContractError, match="source_keys must be unique"):
        replace(resolution, source_keys=(key, key))
    with pytest.raises(IdentityContractError, match="evidence must be unique"):
        replace(resolution, evidence=(resolution.evidence[0],) * 2)
    store.close()


@given(st.permutations((_source(1), _source(2), _source(3))))
def test_request_source_key_canonicalization_is_order_independent(
    keys: list[SourceIdentityKey],
) -> None:
    request = _request(*keys)
    baseline = _request(_source(1), _source(2), _source(3))

    assert request.source_keys == baseline.source_keys
    assert request.request_digest == baseline.request_digest


def test_unmatched_and_duplicate_exact_links_are_deterministic(tmp_path: Path) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = ExactIdentityResolver(store)
    first_key = _source(1)
    second_key = _source(2)

    unmatched = resolver.resolve(_request(first_key))
    assert unmatched.ok and unmatched.value is not None
    assert unmatched.value.state == "unmatched"

    canonical_key = _canonical(1)
    assert store.add_link(_link(first_key, canonical_key, 1)).ok
    assert store.add_link(_link(second_key, canonical_key, 2)).ok
    request = _request(first_key, second_key)
    first = resolver.resolve(request)
    second = resolver.resolve(request)

    assert first.ok and first.value is not None
    assert first.value.state == "matched"
    assert first.value.canonical_key == canonical_key
    assert second.ok and not second.created
    assert second.value == first.value
    store.close()


def test_reprocessing_with_new_request_metadata_reuses_first_resolution(
    tmp_path: Path,
) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = ExactIdentityResolver(store)
    key = _source(1)
    assert store.add_link(_link(key, _canonical(1), 1)).ok
    original_request = _request(key)
    replay_request = replace(
        original_request,
        request_id="request_bbbbbbbbbbbbbbbb",
        requested_at=T1,
    )

    original = resolver.resolve(original_request)
    replay = resolver.resolve(replay_request)

    assert original.ok and original.value is not None and original.created
    assert replay.ok and replay.value is not None and not replay.created
    assert replay_request.request_digest == original_request.request_digest
    assert replay.value == original.value
    store.close()


def test_collision_is_conflict_until_merge_review(tmp_path: Path) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = ExactIdentityResolver(store)
    key = _source(1)
    first_candidate = _canonical(1)
    second_candidate = _canonical(2)
    assert store.add_link(_link(key, first_candidate, 1)).ok
    assert store.add_link(_link(key, second_candidate, 2)).ok

    conflict = resolver.resolve(_request(key))
    assert conflict.ok and conflict.value is not None
    assert conflict.value.state == "conflict"
    assert conflict.value.review_required
    assert conflict.value.canonical_key is None

    review = _review(conflict.value, "merge", first_candidate)
    assert store.apply_review(review).ok
    resolved = resolver.resolve(_request(key))
    assert resolved.ok and resolved.value is not None
    assert resolved.value.state == "matched"
    assert resolved.value.canonical_key == first_candidate
    store.close()


def test_amendment_review_replaces_wrong_exact_link(tmp_path: Path) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = ExactIdentityResolver(store)
    key = _source(1)
    wrong = _canonical(1)
    amended = _canonical(2)
    assert store.add_link(_link(key, wrong, 1)).ok
    assert store.add_link(_link(key, amended, 2)).ok
    conflict = resolver.resolve(_request(key)).value
    assert conflict is not None

    decision = _review(conflict, "confirm_match", amended)
    assert store.apply_review(decision).ok
    result = resolver.resolve(_request(key))

    assert result.ok and result.value is not None
    assert result.value.state == "matched"
    assert result.value.canonical_key == amended
    store.close()


class SyntheticPlugin:
    """Deterministic candidate plugin for synthetic tests."""

    def __init__(self, candidates: tuple[ProbabilisticIdentityCandidate, ...]) -> None:
        self._candidates = candidates

    def candidates(self, request: IdentityResolutionRequest):
        return StoreResult.success(self._candidates)


class FailedPlugin:
    """Plugin that returns a typed value-free failure."""

    def candidates(self, request: IdentityResolutionRequest):
        return StoreResult.outcome(StoreState.FAILURE, "identity_plugin_failed")


def _candidate(index: int, score: int = 9500) -> ProbabilisticIdentityCandidate:
    return ProbabilisticIdentityCandidate(
        canonical_key=_canonical(index),
        score_basis_points=score,
        evidence_digest=canonical_digest({"candidate": index, "synthetic": True}),
        plugin_id="synthetic.matcher",
        plugin_version="1.0.0",
    )


def test_probabilistic_candidate_never_auto_matches_and_split_needs_review(
    tmp_path: Path,
) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    exact = ExactIdentityResolver(store)
    resolver = CompositeIdentityResolver(exact, SyntheticPlugin((_candidate(1),)))
    key = _source(1)

    ambiguous = resolver.resolve(_request(key))
    assert ambiguous.ok and ambiguous.value is not None
    assert ambiguous.value.state == "ambiguous"
    assert ambiguous.value.canonical_key is None
    assert ambiguous.value.review_required
    still_isolated = exact.evaluate(_request(key))
    assert still_isolated.ok and still_isolated.value is not None
    assert still_isolated.value.state == "unmatched"

    split_key = _canonical(9)
    decision = _review(ambiguous.value, "split", split_key)
    assert store.apply_review(decision).ok
    resolved = exact.resolve(_request(key))
    assert resolved.ok and resolved.value is not None
    assert resolved.value.state == "matched"
    assert resolved.value.canonical_key == split_key
    store.close()


def test_plugin_failure_is_not_converted_to_unmatched(tmp_path: Path) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = CompositeIdentityResolver(ExactIdentityResolver(store), FailedPlugin())

    result = resolver.resolve(_request(_source(1)))

    assert result.state is StoreState.FAILURE
    assert result.code == "identity_plugin_failed"
    rows = store._connection.execute(
        "SELECT COUNT(*) FROM identity_resolutions"
    ).fetchone()
    assert rows[0] == 0
    store.close()


@pytest.mark.parametrize("mode", ["raises", "invalid_result", "invalid_candidates"])
def test_plugin_errors_are_value_free_and_do_not_persist(
    tmp_path: Path, mode: str
) -> None:
    canary = "synthetic private plugin failure payload"

    class BrokenPlugin:
        def candidates(self, request):
            if mode == "raises":
                raise RuntimeError(canary)
            if mode == "invalid_result":
                return canary
            return StoreResult.success((canary,))

    with IdentityResolutionStore(tmp_path / "identity.sqlite3") as store:
        resolver = CompositeIdentityResolver(
            ExactIdentityResolver(store), BrokenPlugin()
        )
        result = resolver.resolve(_request(_source(1)))
        assert result.state is StoreState.FAILURE
        assert result.code == (
            "identity_plugin_failed" if mode == "raises" else "identity_plugin_invalid"
        )
        assert canary not in repr(result)
        assert (
            store._connection.execute(
                "SELECT COUNT(*) FROM identity_resolutions"
            ).fetchone()[0]
            == 0
        )


def test_confirm_unmatched_deactivates_collision_without_merge(tmp_path: Path) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = ExactIdentityResolver(store)
    key = _source(1)
    assert store.add_link(_link(key, _canonical(1), 1)).ok
    assert store.add_link(_link(key, _canonical(2), 2)).ok
    conflict = resolver.resolve(_request(key)).value
    assert conflict is not None

    assert store.apply_review(_review(conflict, "confirm_unmatched", None)).ok
    result = resolver.resolve(_request(key))

    assert result.ok and result.value is not None
    assert result.value.state == "unmatched"
    store.close()


def test_review_candidate_mismatch_rolls_back_without_changing_links(
    tmp_path: Path,
) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = ExactIdentityResolver(store)
    key = _source(1)
    assert store.add_link(_link(key, _canonical(1), 1)).ok
    assert store.add_link(_link(key, _canonical(2), 2)).ok
    conflict = resolver.resolve(_request(key)).value
    assert conflict is not None

    rejected = store.apply_review(_review(conflict, "merge", _canonical(9)))
    unchanged = resolver.resolve(_request(key))

    assert rejected.state is StoreState.CONFLICT
    assert rejected.code == "review_candidate_mismatch"
    assert unchanged.ok and unchanged.value is not None
    assert unchanged.value.state == "conflict"
    store.close()


def test_policy_denial_happens_before_identity_lookup(tmp_path: Path) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = ExactIdentityResolver(store)
    denied = resolver.resolve(_request(_source(1), attributes=()))

    assert denied.state is StoreState.DENIED
    assert denied.code == "identified_attribute_required"
    rows = store._connection.execute(
        "SELECT COUNT(*) FROM identity_resolutions"
    ).fetchone()
    assert rows[0] == 0
    store.close()


def test_encounter_keys_use_same_contract_without_cross_entity_match(
    tmp_path: Path,
) -> None:
    store = IdentityResolutionStore(tmp_path / "identity.sqlite3")
    resolver = ExactIdentityResolver(store)
    encounter = _source(1, entity_type="encounter")
    patient = _source(1, entity_type="patient")
    assert store.add_link(
        _link(encounter, _canonical(1, entity_type="encounter"), 1)
    ).ok

    encounter_result = resolver.resolve(_request(encounter))
    patient_result = resolver.resolve(_request(patient))

    assert encounter_result.value is not None
    assert encounter_result.value.state == "matched"
    assert patient_result.value is not None
    assert patient_result.value.state == "unmatched"
    store.close()


def test_restart_integrity_and_newer_schema_refusal(tmp_path: Path) -> None:
    path = tmp_path / "identity.sqlite3"
    store = IdentityResolutionStore(path)
    resolver = ExactIdentityResolver(store)
    key = _source(1)
    assert store.add_link(_link(key, _canonical(1), 1)).ok
    assert store.add_link(_link(key, _canonical(2), 2)).ok
    conflict = resolver.resolve(_request(key)).value
    assert conflict is not None
    assert store.apply_review(_review(conflict, "merge", _canonical(1))).ok
    store.close()

    reopened = IdentityResolutionStore(path)
    assert reopened.get_resolution(conflict.resolution_id).value == conflict
    assert (
        reopened.get_resolution("resolution_ffffffffffffffff").state
        is StoreState.UNKNOWN
    )
    integrity = reopened.integrity_check()
    assert integrity.ok and integrity.value is not None
    assert integrity.value["identity_reviews"] == 1
    recovered = ExactIdentityResolver(reopened).resolve(_request(key))
    assert recovered.ok and recovered.value is not None
    assert recovered.value.state == "matched"
    assert recovered.value.canonical_key == _canonical(1)
    reopened.close()

    connection = sqlite3.connect(path)
    connection.execute("UPDATE schema_migrations SET version = 99")
    connection.commit()
    connection.close()
    unsupported = IdentityResolutionStore.open(path)
    assert unsupported.state is StoreState.UNSUPPORTED
    assert unsupported.code == "schema_unsupported"


def test_persisted_records_contain_opaque_keys_not_raw_values(tmp_path: Path) -> None:
    canary = "synthetic raw patient value canary"
    path = tmp_path / "identity.sqlite3"
    store = IdentityResolutionStore(path)
    key = _source(1)
    assert store.add_link(_link(key, _canonical(1), 1)).ok
    assert ExactIdentityResolver(store).resolve(_request(key)).ok

    payloads = []
    for table in ("identity_links", "identity_resolutions"):
        payloads.extend(
            row[0]
            for row in store._connection.execute(
                f"SELECT payload_json FROM {table}"  # noqa: S608
            ).fetchall()
        )
    assert canary not in json.dumps(payloads)
    assert all("local_" in payload for payload in payloads)
    store.close()
