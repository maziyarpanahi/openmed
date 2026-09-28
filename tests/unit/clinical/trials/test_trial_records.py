"""Public clinical-trial source and local cache tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from openmed.clinical.trials import (
    ClinicalTrialSource,
    LocalTrialStore,
    TrialCacheCorruptionError,
    TrialContractError,
    TrialQuery,
    TrialSchemaDriftError,
    TrialSourceUnavailableError,
    load_trial_study_schema,
    parse_trial_source_page,
)
from openmed.clinical.trials import source as trial_source_module
from openmed.clinical.trials import store as trial_store_module

FIXTURES = Path(__file__).parents[3] / "fixtures" / "clinical" / "trials"


class FrozenTransport:
    """Record metadata-only URLs and return one frozen page."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[tuple[str, float]] = []

    def fetch(self, url: str, *, timeout_seconds: float) -> bytes:
        self.calls.append((url, timeout_seconds))
        return self.payload


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_frozen_source_page_is_deterministic_and_schema_valid() -> None:
    first = parse_trial_source_page(
        _fixture("initial.json"), retrieved_at="2026-09-21T08:00:00Z"
    )
    second = parse_trial_source_page(
        _fixture("initial.json"), retrieved_at="2026-09-21T08:00:00Z"
    )
    assert first == second
    assert [item.study_id for item in first.studies] == ["NCT00000001", "NCT00000002"]
    assert first.next_page_token == "synthetic-page-2"
    assert first.studies[0].eligibility_text.startswith("Inclusion")

    schema = load_trial_study_schema()
    validator_for(schema).check_schema(schema)
    validator = validator_for(schema)(schema)
    assert not list(validator.iter_errors(first.studies[0].to_dict()))
    assert (
        type(first.studies[0]).from_dict(first.studies[0].to_dict()) == first.studies[0]
    )


def test_changed_and_withdrawn_studies_create_append_only_versions(
    tmp_path: Path,
) -> None:
    store = LocalTrialStore(tmp_path / "trials")
    initial = parse_trial_source_page(
        _fixture("initial.json"), retrieved_at="2026-09-21T08:00:00Z"
    )
    update = parse_trial_source_page(
        _fixture("update.json"), retrieved_at="2026-09-21T09:00:00Z"
    )

    assert store.apply_page(initial).created_versions == 2
    repeated = store.apply_page(initial)
    assert repeated.created_versions == 0
    assert repeated.unchanged_studies == 2
    assert store.apply_page(update).created_versions == 1

    history = store.history("NCT00000001")
    assert [item.overall_status for item in history] == ["RECRUITING", "WITHDRAWN"]
    assert history[0].version_id != history[1].version_id
    assert store.latest("NCT00000001") == history[1]


def test_offline_queries_use_latest_cached_versions(tmp_path: Path) -> None:
    store = LocalTrialStore(tmp_path / "trials")
    store.apply_page(
        parse_trial_source_page(
            _fixture("initial.json"), retrieved_at="2026-09-21T08:00:00Z"
        )
    )
    store.apply_page(
        parse_trial_source_page(
            _fixture("update.json"), retrieved_at="2026-09-21T09:00:00Z"
        )
    )

    assert [
        item.study_id for item in store.query(TrialQuery(countries=("France",)))
    ] == ["NCT00000001"]
    assert store.query(TrialQuery(statuses=("recruiting",))) == ()
    assert [
        item.study_id for item in store.query(TrialQuery(conditions=("Asthma",)))
    ] == ["NCT00000002"]


def test_cache_corruption_and_schema_drift_raise_typed_errors(tmp_path: Path) -> None:
    store = LocalTrialStore(tmp_path / "trials")
    page = parse_trial_source_page(
        _fixture("initial.json"), retrieved_at="2026-09-21T08:00:00Z"
    )
    store.apply_page(page)
    envelope = json.loads(store.path.read_text(encoding="utf-8"))
    envelope["records"][0]["overall_status"] = "COMPLETED"
    store.path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(TrialCacheCorruptionError, match="digest"):
        store.query()

    with pytest.raises(TrialSchemaDriftError, match="statusModule"):
        parse_trial_source_page(
            b'{"studies":[{"protocolSection":{"identificationModule":{}}}]}',
            retrieved_at="2026-09-21T08:00:00Z",
        )


def test_source_request_contains_only_public_pagination_controls() -> None:
    transport = FrozenTransport(_fixture("initial.json"))
    source = ClinicalTrialSource(transport=transport)
    page = source.fetch_page(
        retrieved_at="2026-09-21T08:00:00Z",
        page_token="opaque-public-cursor",
        page_size=25,
    )
    assert len(page.studies) == 2
    url, timeout = transport.calls[0]
    assert timeout == 30.0
    assert "format=json" in url
    assert "pageSize=25" in url
    assert "pageToken=opaque-public-cursor" in url
    forbidden = ("patient", "subject", "name", "birth", "email", "phone")
    assert not any(token in url.casefold() for token in forbidden)


def test_explicit_source_failure_is_typed() -> None:
    class FailingTransport:
        def fetch(self, url: str, *, timeout_seconds: float) -> bytes:
            raise OSError("synthetic transport failure")

    source = ClinicalTrialSource(transport=FailingTransport())
    with pytest.raises(TrialSourceUnavailableError, match="metadata fetch failed"):
        source.fetch_page(retrieved_at="2026-09-21T08:00:00Z")


@pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf])
def test_source_rejects_nonfinite_timeout(timeout: float) -> None:
    with pytest.raises(TrialContractError):
        ClinicalTrialSource(timeout_seconds=timeout)


def test_unexpected_transport_exception_is_value_free() -> None:
    class FailingTransport:
        def fetch(self, url: str, *, timeout_seconds: float) -> bytes:
            raise RuntimeError("private-input-should-not-leak")

    source = ClinicalTrialSource(transport=FailingTransport())
    with pytest.raises(TrialSourceUnavailableError) as raised:
        source.fetch_page(retrieved_at="2026-09-21T08:00:00Z")
    assert "private-input-should-not-leak" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_source_response_limit_rejects_oversized_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trial_source_module, "_MAX_SOURCE_PAGE_BYTES", 100)
    source = ClinicalTrialSource(transport=FrozenTransport(_fixture("initial.json")))
    with pytest.raises(TrialSchemaDriftError, match="size"):
        source.fetch_page(retrieved_at="2026-09-21T08:00:00Z")


def test_cache_write_without_posix_fchmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(trial_store_module.os, "fchmod", raising=False)
    store = LocalTrialStore(tmp_path / "trials")
    page = parse_trial_source_page(
        _fixture("initial.json"), retrieved_at="2026-09-21T08:00:00Z"
    )
    assert store.apply_page(page).created_versions == 2
    assert store.latest("NCT00000001") is not None
