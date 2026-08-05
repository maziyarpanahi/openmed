"""Tests for PHI-free operator reports over distributed batch runs.

Synthetic data only: document identifiers are generated from an index, and the
canary strings reuse the invented PHI vocabulary already used by the repository's
no-raw-text logging guard. Nothing here reads a corpus or the network.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from openmed.processing.distributed import plan_document_shards
from openmed.processing.resume import resume_plan, worker_ref
from openmed.processing.run_manifest import (
    BatchRunManifest,
    ShardOutputValidation,
    ShardRecord,
    ShardStatus,
    build_run_manifest,
)
from openmed.processing.run_report import (
    RUN_STATE_COMPLETE,
    RUN_STATE_EXHAUSTED,
    RUN_STATE_IN_PROGRESS,
    UNRECOGNIZED_ERROR_TYPE,
    RunReportError,
    RunReportPrivacyError,
    assert_no_raw_text,
    build_run_report,
)

# Invented values. Any of these appearing in a serialized report is a leak.
PHI_TEXT = (
    "Patient Evelyn Quantum, MRN ZQ-7391, DOB 04/17/1972, "
    "SSN 042-66-9001, phone 919-555-0188."
)
PHI_SUBSTRINGS = (
    "Evelyn Quantum",
    "ZQ-7391",
    "04/17/1972",
    "042-66-9001",
    "919-555-0188",
)

# A worker id that is bounded and control-character free -- so the run manifest
# accepts it unchanged -- yet still names a person.
LEAKY_WORKER_ID = "onc-ward-dr-evelyn-quantum-01"

HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
PREFIXED_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def _documents(count: int) -> list[dict[str, str]]:
    return [{"id": f"doc-{index:05d}"} for index in range(count)]


def _manifest(shard_count: int = 4, documents: int = 12) -> BatchRunManifest:
    plan = plan_document_shards(_documents(documents), shard_count=shard_count)
    return build_run_manifest(plan, run_id="run-0001", created_at=1000.0)


def _completed(record: ShardRecord, *, index: int, **overrides) -> ShardRecord:
    defaults = {
        "status": ShardStatus.COMPLETED,
        "attempts": 1,
        "started_at": 1000.0 + index,
        "completed_at": 1000.0 + index + (index + 1),
        "output_path": f"shard-{record.shard_id:05d}.jsonl",
        "output_digest": "sha256:" + f"{index:064x}",
        "output_bytes": 128 + index,
        "worker_id": LEAKY_WORKER_ID,
    }
    defaults.update(overrides)
    return ShardRecord(
        shard_id=record.shard_id,
        fingerprint=record.fingerprint,
        document_count=record.document_count,
        **defaults,
    )


def _finished_manifest(**overrides) -> BatchRunManifest:
    manifest = _manifest()
    records = tuple(
        _completed(record, index=index, **overrides)
        for index, record in enumerate(manifest.shards)
    )
    return BatchRunManifest(
        run_id=manifest.run_id,
        created_at=manifest.created_at,
        updated_at=2000.0,
        algorithm=manifest.algorithm,
        shard_count=manifest.shard_count,
        document_count=manifest.document_count,
        plan_fingerprint=manifest.plan_fingerprint,
        openmed_version=manifest.openmed_version,
        shards=records,
    )


# ---------------------------------------------------------------------------
# Acceptance: counts, durations, failures, fingerprints
# ---------------------------------------------------------------------------


def test_report_carries_counts_durations_failures_and_fingerprints() -> None:
    manifest = _finished_manifest()
    report = build_run_report(manifest, generated_at=3000.0)
    payload = report.to_dict()

    assert payload["shard_count"] == 4
    assert payload["document_count"] == 12
    assert payload["status_counts"]["completed"] == 4
    assert payload["total_attempts"] == 4

    assert payload["duration_seconds"]["measured_shards"] == 4
    assert payload["duration_seconds"]["total"] == pytest.approx(1 + 2 + 3 + 4)
    assert payload["duration_seconds"]["max"] == pytest.approx(4.0)
    assert payload["duration_seconds"]["p50"] == pytest.approx(2.5)

    assert HEX_DIGEST.fullmatch(payload["plan_fingerprint"])
    for row in payload["shards"]:
        assert HEX_DIGEST.fullmatch(row["fingerprint"])
        assert PREFIXED_DIGEST.fullmatch(row["output_digest"])

    assert payload["failures"] == []
    assert payload["run_state"] == RUN_STATE_COMPLETE
    assert payload["algorithm"]
    assert payload["openmed_version"]


def test_failed_shards_are_reported_by_error_type_only() -> None:
    manifest = _finished_manifest()
    broken = ShardRecord(
        shard_id=0,
        fingerprint=manifest.shards[0].fingerprint,
        document_count=manifest.shards[0].document_count,
        status=ShardStatus.FAILED,
        attempts=3,
        error_type="RuntimeError",
    )
    manifest = BatchRunManifest(
        run_id=manifest.run_id,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        algorithm=manifest.algorithm,
        shard_count=manifest.shard_count,
        document_count=manifest.document_count,
        plan_fingerprint=manifest.plan_fingerprint,
        openmed_version=manifest.openmed_version,
        shards=(broken,) + manifest.shards[1:],
    )

    payload = build_run_report(manifest, generated_at=3000.0).to_dict()

    assert payload["failures"] == [
        {"shard_id": 0, "error_type": "RuntimeError", "attempts": 3}
    ]
    assert payload["status_counts"]["failed"] == 1
    assert payload["run_state"] == RUN_STATE_IN_PROGRESS


# ---------------------------------------------------------------------------
# Acceptance: no document text reaches a serialized report
# ---------------------------------------------------------------------------


def test_serialized_report_is_free_of_planted_phi() -> None:
    manifest = _finished_manifest()
    report = build_run_report(
        manifest,
        generated_at=3000.0,
        resume=resume_plan(manifest, clock=lambda: 3000.0),
        validation=ShardOutputValidation(valid=(0, 1, 2, 3)),
    )

    blob = json.dumps(report.to_dict(), sort_keys=True)
    leaked = [needle for needle in PHI_SUBSTRINGS if needle in blob]
    assert leaked == [], f"PHI leaked into the report: {leaked!r}"
    assert LEAKY_WORKER_ID not in blob
    assert "doc-" not in blob
    assert blob == json.dumps(report.to_dict(), sort_keys=True)


def test_no_worker_id_key_survives_anywhere_in_the_payload() -> None:
    """Structural, because a value check provably cannot catch this.

    ``onc-ward-dr-evelyn-quantum-01`` is a perfectly ordinary token: it has no
    whitespace and no control characters, so every allowlist in this module
    accepts it. Only removing the field protects it, so the test asserts the
    field is gone rather than asserting something about its value.
    """

    manifest = _finished_manifest()
    report = build_run_report(
        manifest,
        generated_at=3000.0,
        resume=resume_plan(manifest, clock=lambda: 3000.0),
    )

    def keys(node) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.items():
                found.add(key)
                found |= keys(value)
        elif isinstance(node, list):
            for item in node:
                found |= keys(item)
        return found

    present = keys(report.to_dict())
    assert "worker_id" not in present
    assert "output_path" not in present
    assert "worker_ref" in present


def test_worker_ref_matches_the_shared_helper_across_both_report_paths() -> None:
    """One worker must carry one token, whichever table it appears in."""

    manifest = _finished_manifest()
    payload = build_run_report(manifest, generated_at=3000.0).to_dict()

    expected = worker_ref(LEAKY_WORKER_ID)
    assert expected is not None and expected.startswith("w-")
    assert {row["worker_ref"] for row in payload["shards"]} == {expected}


# ---------------------------------------------------------------------------
# The guard must be able to fail
# ---------------------------------------------------------------------------


def test_guard_rejects_phi_as_a_mapping_key() -> None:
    """A value-only scanner cannot catch this: the value is a benign integer."""

    payload = {"shards": [{PHI_TEXT: 1}]}

    # Demonstrate the bypass a values-only check would suffer.
    values_only_leak = [
        value for value in payload["shards"][0].values() if isinstance(value, str)
    ]
    assert values_only_leak == []

    with pytest.raises(RunReportPrivacyError, match="mapping key"):
        assert_no_raw_text(payload)


def test_guard_inspects_every_node_when_temporaries_are_freed_as_it_walks() -> None:
    """A node skipped in silence means published record content.

    Memoizing visited nodes by ``id()`` is unsound for a walker that frees each
    temporary as it advances: the interpreter reissues the address and the next
    node inherits a memoized id. This payload builds every member on access, so
    each is collectable the moment the walk moves on, and the planted value sits
    in a late node rather than a shallow one.
    """

    from collections.abc import Mapping as _Mapping

    class _Ephemeral(_Mapping):
        def __init__(self, depth: int, phi_at: int) -> None:
            self.depth, self.phi_at = depth, phi_at

        def __getitem__(self, key):
            if key != "child":
                raise KeyError(key)
            if self.depth == self.phi_at:
                return {PHI_TEXT: 1}
            if self.depth == 0:
                return {"leaf": 1}
            return _Ephemeral(self.depth - 1, self.phi_at)

        def __iter__(self):
            yield "child"

        def __len__(self) -> int:
            return 1

    for depth, phi_at in ((100, 1), (200, 3), (60, 0)):
        with pytest.raises(RunReportPrivacyError):
            assert_no_raw_text({"root": _Ephemeral(depth, phi_at)})

    # Wide rather than deep: many short-lived siblings, planted value last.
    rows = [{f"k{index}": index} for index in range(500)]
    rows.append({PHI_TEXT: 1})
    with pytest.raises(RunReportPrivacyError):
        assert_no_raw_text({"rows": rows})


def test_guard_rejects_phi_as_a_value_under_a_benign_key() -> None:
    with pytest.raises(RunReportPrivacyError, match="string value"):
        assert_no_raw_text({"run_id": PHI_TEXT})


def test_guard_rejects_an_exception_message_smuggled_as_a_field() -> None:
    try:
        raise ValueError(f"failed while reading {PHI_TEXT}")
    except ValueError as exc:
        smuggled = {"failures": [{"error": str(exc)}]}

    with pytest.raises(RunReportPrivacyError, match="string value"):
        assert_no_raw_text(smuggled)


def test_guard_rejects_forbidden_keys_even_when_their_values_are_safe() -> None:
    with pytest.raises(RunReportPrivacyError, match="forbidden key"):
        assert_no_raw_text({"worker_id": "w-abcdef123456"})


def test_guard_error_messages_never_quote_the_offending_value() -> None:
    """The message is itself operator-visible, and the CLI prints it verbatim."""

    with pytest.raises(RunReportPrivacyError) as excinfo:
        assert_no_raw_text({"run_id": PHI_TEXT})
    rendered = str(excinfo.value)
    assert all(needle not in rendered for needle in PHI_SUBSTRINGS)

    with pytest.raises(RunReportPrivacyError) as excinfo:
        assert_no_raw_text({PHI_TEXT: 1})
    rendered = str(excinfo.value)
    assert all(needle not in rendered for needle in PHI_SUBSTRINGS)


def test_guard_rejects_unsupported_types_rather_than_coercing_them() -> None:
    with pytest.raises(RunReportPrivacyError, match="unsupported value type"):
        assert_no_raw_text({"generated_at": object()})


def test_guard_accepts_the_tokens_a_report_legitimately_carries() -> None:
    assert_no_raw_text(
        {
            "run_id": "run-0001",
            "status": "completed",
            "output_digest": "sha256:" + "a" * 64,
            "worker_ref": "w-0123456789ab",
            "generated_at": 1000.5,
            "shards": [0, 1, 2],
            "enabled": True,
            "baseline": None,
        }
    )


# ---------------------------------------------------------------------------
# Coerced-string defences and their honest limit
# ---------------------------------------------------------------------------


def test_a_coerced_none_is_caught_on_digest_fields() -> None:
    manifest = _finished_manifest(output_digest="sha256:" + "b" * 64)
    record = manifest.shards[0]
    broken = ShardRecord.__new__(ShardRecord)
    object.__setattr__(broken, "shard_id", record.shard_id)
    object.__setattr__(broken, "fingerprint", "None")
    object.__setattr__(broken, "document_count", record.document_count)
    object.__setattr__(broken, "status", ShardStatus.PENDING)
    object.__setattr__(broken, "attempts", 0)
    for name in ("started_at", "completed_at", "output_path", "output_digest"):
        object.__setattr__(broken, name, None)
    for name in ("output_bytes", "worker_id", "error_type"):
        object.__setattr__(broken, name, None)

    manifest = BatchRunManifest(
        run_id=manifest.run_id,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        algorithm=manifest.algorithm,
        shard_count=manifest.shard_count,
        document_count=manifest.document_count,
        plan_fingerprint=manifest.plan_fingerprint,
        openmed_version=manifest.openmed_version,
        shards=(broken,) + manifest.shards[1:],
    )

    with pytest.raises(RunReportPrivacyError, match="not a sha256 digest"):
        build_run_report(manifest, generated_at=3000.0).to_dict()


def test_plan_fingerprint_is_shape_checked_like_every_other_digest() -> None:
    """The most prominent digest must not be the one route that skips the check.

    A shard fingerprint and a plan fingerprint are the same kind of value, so a
    value blocked in one and published in the other is an inconsistency, not a
    policy.
    """

    manifest = _finished_manifest()
    forged = BatchRunManifest.__new__(BatchRunManifest)
    for name, value in (
        ("run_id", manifest.run_id),
        ("created_at", manifest.created_at),
        ("updated_at", manifest.updated_at),
        ("algorithm", manifest.algorithm),
        ("shard_count", manifest.shard_count),
        ("document_count", manifest.document_count),
        ("plan_fingerprint", "MRN-ZQ7391-Evelyn-Quantum-DOB-1972-04-17"),
        ("openmed_version", manifest.openmed_version),
        ("shards", manifest.shards),
        ("schema_version", manifest.schema_version),
    ):
        object.__setattr__(forged, name, value)

    with pytest.raises(RunReportPrivacyError, match="plan_fingerprint"):
        build_run_report(forged, generated_at=3000.0)

    # The same value in a shard fingerprint was already blocked; both now agree.
    object.__setattr__(forged, "plan_fingerprint", "None")
    with pytest.raises(RunReportPrivacyError, match="plan_fingerprint"):
        build_run_report(forged, generated_at=3000.0)


def test_is_publishable_token_matches_what_the_guard_accepts() -> None:
    """The pre-flight check and the publication check must not disagree."""

    from openmed.processing.run_report import is_publishable_token

    for value in ("run-0001", "run_2026-08-02", "w-0123456789ab", "sha256:" + "a" * 64):
        assert is_publishable_token(value) is True
        assert_no_raw_text({"run_id": value})

    for value in ("nightly run", "run#42", PHI_TEXT, "x" * 129, "", "run\n"):
        assert is_publishable_token(value) is False
        with pytest.raises(RunReportPrivacyError):
            assert_no_raw_text({"run_id": value})

    assert is_publishable_token(None) is False
    assert is_publishable_token(7) is False


def test_a_malformed_error_type_renders_as_a_constant() -> None:
    manifest = _finished_manifest()
    record = manifest.shards[0]
    broken = ShardRecord.__new__(ShardRecord)
    for name, value in (
        ("shard_id", record.shard_id),
        ("fingerprint", record.fingerprint),
        ("document_count", record.document_count),
        ("status", ShardStatus.FAILED),
        ("attempts", 2),
        ("started_at", None),
        ("completed_at", None),
        ("output_path", None),
        ("output_digest", None),
        ("output_bytes", None),
        ("worker_id", None),
        ("error_type", "not an identifier\n| forged | row"),
    ):
        object.__setattr__(broken, name, value)

    manifest = BatchRunManifest(
        run_id=manifest.run_id,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        algorithm=manifest.algorithm,
        shard_count=manifest.shard_count,
        document_count=manifest.document_count,
        plan_fingerprint=manifest.plan_fingerprint,
        openmed_version=manifest.openmed_version,
        shards=(broken,) + manifest.shards[1:],
    )

    payload = build_run_report(manifest, generated_at=3000.0).to_dict()
    assert payload["failures"][0]["error_type"] == UNRECOGNIZED_ERROR_TYPE
    assert "forged" not in json.dumps(payload)


def test_the_allowlist_cannot_have_the_trailing_newline_bug() -> None:
    """A character-class fullmatch has no anchor for a newline to slip past."""

    with pytest.raises(RunReportPrivacyError, match="string value"):
        assert_no_raw_text({"error_type": "RuntimeError\n"})
    with pytest.raises(RunReportPrivacyError, match="string value"):
        assert_no_raw_text({"error_type": "RuntimeError\n| forged | row"})


# ---------------------------------------------------------------------------
# Resume, run state, stragglers and empty shards
# ---------------------------------------------------------------------------


def test_exhausted_run_is_terminal_not_complete() -> None:
    manifest = _manifest(shard_count=2, documents=4)
    failed = tuple(
        ShardRecord(
            shard_id=record.shard_id,
            fingerprint=record.fingerprint,
            document_count=record.document_count,
            status=ShardStatus.FAILED,
            attempts=5,
            error_type="RuntimeError",
        )
        for record in manifest.shards
    )
    manifest = BatchRunManifest(
        run_id=manifest.run_id,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        algorithm=manifest.algorithm,
        shard_count=manifest.shard_count,
        document_count=manifest.document_count,
        plan_fingerprint=manifest.plan_fingerprint,
        openmed_version=manifest.openmed_version,
        shards=failed,
    )

    plan = resume_plan(manifest, max_attempts=3, clock=lambda: 3000.0)
    payload = build_run_report(manifest, generated_at=3000.0, resume=plan).to_dict()

    assert plan.is_exhausted is True
    assert plan.is_complete is False
    assert payload["run_state"] == RUN_STATE_EXHAUSTED
    assert payload["resume"]["exhausted"] == [0, 1]
    assert payload["resume"]["is_exhausted"] is True


def test_report_distinguishes_not_measured_from_nothing_lagging() -> None:
    manifest = _finished_manifest()
    plan = resume_plan(manifest, clock=lambda: 3000.0)
    payload = build_run_report(manifest, generated_at=3000.0, resume=plan).to_dict()

    resume = payload["resume"]
    assert resume["straggler_detection_enabled"] == plan.straggler_detection_enabled
    assert resume["stragglers"] == []

    markdown = build_run_report(
        manifest, generated_at=3000.0, resume=plan
    ).to_markdown()
    if plan.straggler_detection_enabled:
        assert "None lagging." in markdown
    else:
        assert "Not measured" in markdown


def test_zero_document_completed_shard_is_normal_not_an_anomaly() -> None:
    plan = plan_document_shards(_documents(2), shard_count=6)
    manifest = build_run_manifest(plan, run_id="run-empty", created_at=1000.0)
    records = tuple(
        _completed(record, index=index) for index, record in enumerate(manifest.shards)
    )
    manifest = BatchRunManifest(
        run_id=manifest.run_id,
        created_at=manifest.created_at,
        updated_at=2000.0,
        algorithm=manifest.algorithm,
        shard_count=manifest.shard_count,
        document_count=manifest.document_count,
        plan_fingerprint=manifest.plan_fingerprint,
        openmed_version=manifest.openmed_version,
        shards=records,
    )

    payload = build_run_report(manifest, generated_at=3000.0).to_dict()
    empty = [row for row in payload["shards"] if row["document_count"] == 0]

    assert empty, "fixture should produce at least one empty shard"
    assert all(row["status"] == "completed" for row in empty)
    assert payload["run_state"] == RUN_STATE_COMPLETE
    assert payload["failures"] == []


def test_preserved_digest_on_a_requeued_shard_is_not_an_inconsistency() -> None:
    """``prepare_resume`` keeps the digest as the reproducibility anchor."""

    from openmed.processing.resume import prepare_resume

    manifest = _finished_manifest()
    plan = resume_plan(manifest, root=None, clock=lambda: 3000.0)
    requeued = prepare_resume(manifest, plan, updated_at=3000.0)

    payload = build_run_report(requeued, generated_at=3000.0).to_dict()
    for row in payload["shards"]:
        if row["status"] == "pending" and row["output_digest"] is not None:
            assert PREFIXED_DIGEST.fullmatch(row["output_digest"])
    assert payload["failures"] == []


def test_outputs_block_uses_the_validation_object_not_a_list() -> None:
    manifest = _finished_manifest()
    validation = ShardOutputValidation(valid=(0, 1), missing=(2,), mismatched=(3,))
    payload = build_run_report(
        manifest, generated_at=3000.0, validation=validation
    ).to_dict()

    assert payload["outputs"] == {
        "all_valid": False,
        "valid": [0, 1],
        "missing": [2],
        "mismatched": [3],
    }
    assert payload["run_state"] == RUN_STATE_IN_PROGRESS


def test_output_validation_can_veto_an_apparently_complete_manifest() -> None:
    manifest = _finished_manifest()
    validation = ShardOutputValidation(valid=(0, 1, 2), missing=(3,))

    payload = build_run_report(
        manifest,
        generated_at=3000.0,
        validation=validation,
    ).to_dict()

    assert payload["status_counts"]["completed"] == manifest.shard_count
    assert payload["run_state"] == RUN_STATE_IN_PROGRESS


def test_report_rejects_a_resume_plan_for_another_manifest() -> None:
    manifest = _finished_manifest()
    other = _manifest(shard_count=5, documents=15)
    foreign_plan = resume_plan(other, clock=lambda: 3000.0)

    with pytest.raises(RunReportError, match="does not describe"):
        build_run_report(manifest, generated_at=3000.0, resume=foreign_plan)


@pytest.mark.parametrize("generated_at", [math.nan, math.inf, -math.inf, -1.0])
def test_generated_at_must_be_finite_and_non_negative(generated_at: float) -> None:
    with pytest.raises(RunReportError, match="finite non-negative"):
        build_run_report(_finished_manifest(), generated_at=generated_at)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_markdown_is_byte_stable_for_a_fixed_generated_at() -> None:
    manifest = _finished_manifest()
    first = build_run_report(manifest, generated_at=3000.0).to_markdown()
    second = build_run_report(manifest, generated_at=3000.0).to_markdown()

    assert first == second
    assert first.startswith("# Batch Run Report: run-0001")
    assert first.endswith("\n")
    assert "| Shards | 4 |" in first
    assert all(needle not in first for needle in PHI_SUBSTRINGS)
    assert LEAKY_WORKER_ID not in first


def test_json_rendering_round_trips_and_is_sorted() -> None:
    manifest = _finished_manifest()
    report = build_run_report(manifest, generated_at=3000.0)

    rendered = report.to_json()
    assert json.loads(rendered) == report.to_dict()
    assert rendered == report.to_json()
