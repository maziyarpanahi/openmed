from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from openmed.clinical.grounding.vocab import VocabConcept, VocabularyIndex
from openmed.clinical.terminology import (
    SemanticCandidate,
    SQLiteTerminologyMappingStore,
    TerminologyCoverageSummary,
    TerminologyQuery,
    TerminologyRelationshipRule,
    TerminologyResolutionPolicy,
    TerminologyResolver,
    TerminologySnapshot,
    load_terminology_mapping_schema,
)
from openmed.structured.store import StoreState

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "terminology_resolution_cascade.json"
)
SECRET = "synthetic-test-secret-at-least-16-bytes"
CANARY = "SYNTHETIC-PRIVATE-TERMINOLOGY-CANARY"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _snapshot(*, version: str | None = None) -> TerminologySnapshot:
    payload = _fixture()
    snapshot_payload = payload["snapshot"]
    assert isinstance(snapshot_payload, dict)
    concepts_payload = payload["concepts"]
    assert isinstance(concepts_payload, list)
    concepts = [
        VocabConcept(
            system=str(snapshot_payload["system"]),
            code=str(item["code"]),
            preferred_term=str(item["preferred_term"]),
            synonyms=tuple(item["synonyms"]),
        )
        for item in concepts_payload
    ]
    return TerminologySnapshot(
        vocabulary=str(snapshot_payload["vocabulary"]),
        version=version or str(snapshot_payload["version"]),
        index=VocabularyIndex(str(snapshot_payload["system"]), concepts),
    )


def _relationships() -> tuple[TerminologyRelationshipRule, ...]:
    payload = _fixture()["relationships"]
    assert isinstance(payload, list)
    return tuple(TerminologyRelationshipRule(**item) for item in payload)


def _resolver(
    *,
    snapshot: TerminologySnapshot | None = None,
    policy: TerminologyResolutionPolicy | None = None,
    semantic_provider: object | None = None,
) -> TerminologyResolver:
    return TerminologyResolver(
        snapshot or _snapshot(),
        hmac_secret=SECRET,
        relationships=_relationships(),
        policy=policy,
        semantic_provider=semantic_provider,
    )


@pytest.mark.parametrize("case", _fixture()["cases"], ids=lambda case: case["name"])
def test_cascade_fixture_is_deterministic(case: dict[str, object]) -> None:
    resolver = _resolver()
    first = resolver.resolve(TerminologyQuery(**case["query"]))
    second = resolver.resolve(TerminologyQuery(**case["query"]))

    assert first.value is not None and second.value is not None
    assert first.value.to_dict() == second.value.to_dict()
    assert first.value.state == case["state"]
    assert first.ok is (case["state"] == "mapped")
    if "rule" in case:
        assert first.value.candidates[0].mapping_rule == case["rule"]
    if "code" in case:
        assert first.value.selected_candidate is not None
        assert first.value.selected_candidate.code == case["code"]


def test_ambiguous_and_unmapped_never_look_like_success() -> None:
    ambiguous = _resolver().resolve(TerminologyQuery(source_value="Headache"))
    unmapped = _resolver().resolve(
        TerminologyQuery(source_value="No synthetic concept")
    )

    assert ambiguous.state is StoreState.CONFLICT
    assert ambiguous.code == "terminology_ambiguous"
    assert ambiguous.value is not None and ambiguous.value.review_required
    assert unmapped.state is StoreState.UNKNOWN
    assert unmapped.code == "terminology_unmapped"
    assert unmapped.value is not None and unmapped.value.review_required


@given(
    surface=st.sampled_from(("hypertension", "Hypertension", "HYPERTENSION")),
    left=st.integers(min_value=0, max_value=8),
    right=st.integers(min_value=0, max_value=8),
)
def test_alias_normalization_property_preserves_mapping(
    surface: str, left: int, right: int
) -> None:
    result = _resolver().resolve(
        TerminologyQuery(source_value=" " * left + surface + " " * right)
    )

    assert result.ok and result.value is not None
    assert result.value.selected_candidate is not None
    assert result.value.selected_candidate.code == "1000-1"
    assert result.value.selected_candidate.mapping_rule == "alias"


class _SemanticProvider:
    local_only = True
    provider_id = "synthetic.semantic"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls = 0

    def candidates(self, *_: object, **__: object) -> tuple[SemanticCandidate, ...]:
        self.calls += 1
        return (
            SemanticCandidate(code="1000-1", confidence=0.94),
            SemanticCandidate(code="1000-2", confidence=0.80),
        )


def test_semantic_fallback_is_disabled_by_default() -> None:
    provider = _SemanticProvider()
    result = _resolver(semantic_provider=provider).resolve(
        TerminologyQuery(source_value="Semantic only surface")
    )

    assert result.state is StoreState.UNKNOWN
    assert provider.calls == 0


def test_opt_in_local_semantic_provider_uses_margin() -> None:
    provider = _SemanticProvider()
    policy = TerminologyResolutionPolicy(semantic_enabled=True)
    result = _resolver(policy=policy, semantic_provider=provider).resolve(
        TerminologyQuery(source_value="Semantic only surface")
    )

    assert result.ok
    assert result.value is not None
    assert result.value.selected_candidate is not None
    assert result.value.selected_candidate.code == "1000-1"
    assert result.value.selected_candidate.mapping_rule == "semantic"
    assert provider.calls == 1


def test_enabled_semantic_without_provider_is_explicitly_unsupported() -> None:
    result = _resolver(
        policy=TerminologyResolutionPolicy(semantic_enabled=True)
    ).resolve(TerminologyQuery(source_value="Semantic only surface"))

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "semantic_provider_unavailable"
    assert result.value is not None
    assert result.value.state == "unmapped"


def test_public_results_never_serialize_source_surface() -> None:
    result = _resolver().resolve(TerminologyQuery(source_value=CANARY))

    assert result.value is not None
    assert CANARY not in repr(result.value)
    assert CANARY not in result.value.to_json()
    assert result.value.source_digest.startswith("hmac-sha256:")


def test_store_appends_snapshot_versions_and_queues_non_success(tmp_path: Path) -> None:
    old = _resolver(snapshot=_snapshot(version="synthetic-2026.1")).resolve(
        TerminologyQuery(source_value=CANARY)
    )
    new = _resolver(snapshot=_snapshot(version="synthetic-2026.2")).resolve(
        TerminologyQuery(source_value=CANARY)
    )
    assert old.value is not None and new.value is not None
    assert old.value.mapping_id != new.value.mapping_id
    assert old.value.source_digest == new.value.source_digest

    path = tmp_path / "terminology.db"
    with SQLiteTerminologyMappingStore(path) as store:
        assert store.record(old.value, recorded_at="2026-09-20T10:00:00+00:00").created
        assert store.record(new.value, recorded_at="2026-09-20T11:00:00+00:00").created
        assert not store.record(new.value).created
        history = store.history(old.value.source_digest)
        queue = store.review_queue()

    assert history.ok and history.value is not None
    assert [item["snapshot"]["version"] for item in history.value] == [
        "synthetic-2026.1",
        "synthetic-2026.2",
    ]
    assert queue.ok and queue.value is not None
    assert len(queue.value) == 2
    assert {item.state for item in queue.value} == {"unmapped"}
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert CANARY.encode() not in path.read_bytes()


def test_rejected_decision_gets_its_own_review_record(tmp_path: Path) -> None:
    ambiguous = _resolver().resolve(TerminologyQuery(source_value="Headache"))
    assert ambiguous.value is not None
    rejected = ambiguous.value.reject("review_rejected")

    with SQLiteTerminologyMappingStore(tmp_path / "terminology.db") as store:
        written = store.record(rejected, recorded_at="2026-09-20T10:00:00Z")
        queued = store.review_queue(states={"rejected"})

    assert written.ok
    assert queued.ok and queued.value is not None
    assert queued.value[0].mapping_id == rejected.mapping_id
    assert queued.value[0].state == "rejected"


def test_coverage_summary_contains_counts_and_rates_only() -> None:
    mapped = _resolver().resolve(TerminologyQuery(source_value="Hypertension"))
    ambiguous = _resolver().resolve(TerminologyQuery(source_value="Headache"))
    unmapped = _resolver().resolve(TerminologyQuery(source_value=CANARY))
    assert mapped.value and ambiguous.value and unmapped.value
    rejected = replace(ambiguous.value.reject(), reason_code="review_rejected")

    summary = TerminologyCoverageSummary.from_results(
        (mapped.value, ambiguous.value, unmapped.value, rejected)
    )
    payload = summary.to_dict()

    assert payload["total"] == 4
    assert payload["mapped_rate"] == 0.25
    assert payload["unmapped_rate"] == 0.25
    assert CANARY not in json.dumps(payload)
    assert not any("source" in key or "candidate" in key for key in payload)


def test_public_json_schemas_validate_result_queue_and_summary(tmp_path: Path) -> None:
    result = _resolver().resolve(TerminologyQuery(source_value=CANARY))
    assert result.value is not None
    summary = TerminologyCoverageSummary.from_results((result.value,))
    with SQLiteTerminologyMappingStore(tmp_path / "terminology.db") as store:
        assert store.record(result.value, recorded_at="2026-09-20T10:00:00+00:00").ok
        queue = store.review_queue()
    assert queue.value is not None

    payloads = {
        "terminology_mapping_result": result.value.to_dict(),
        "terminology_review_item": queue.value[0].to_dict(),
        "terminology_coverage_summary": summary.to_dict(),
    }
    for name, payload in payloads.items():
        schema = load_terminology_mapping_schema(name)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)


def test_corrupt_store_payload_is_not_returned(tmp_path: Path) -> None:
    result = _resolver().resolve(TerminologyQuery(source_value=CANARY))
    assert result.value is not None
    path = tmp_path / "terminology.db"
    with SQLiteTerminologyMappingStore(path) as store:
        assert store.record(result.value).ok
        store._connection.execute(  # noqa: SLF001 - deliberate integrity test
            "UPDATE terminology_mappings SET payload_json = '{}'"
        )
        history = store.history(result.value.source_digest)

    assert history.state is StoreState.FAILURE
    assert history.code == "mapping_integrity_failed"


def test_store_rejects_symlink_path(tmp_path: Path) -> None:
    target = tmp_path / "target.db"
    target.touch()
    symlink = tmp_path / "link.db"
    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")

    opened = SQLiteTerminologyMappingStore.open(symlink)

    assert opened.state is StoreState.FAILURE
    assert os.path.islink(symlink)


def test_store_rejects_dangling_symlink_path(tmp_path: Path) -> None:
    symlink = tmp_path / "dangling.db"
    try:
        symlink.symlink_to(tmp_path / "missing.db")
    except OSError:
        pytest.skip("symlinks unavailable")

    opened = SQLiteTerminologyMappingStore.open(symlink)

    assert opened.state is StoreState.FAILURE
    assert os.path.islink(symlink)
