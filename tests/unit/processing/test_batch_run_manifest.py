"""Durability, resume and PHI-free tests for the distributed run manifest."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from openmed.processing import (
    MAX_LABEL_LENGTH,
    RUN_MANIFEST_SCHEMA_VERSION,
    BatchRunManifest,
    InMemoryRunManifestStore,
    LocalFileRunManifestStore,
    ManifestSchemaError,
    RunManifestError,
    RunManifestStore,
    ShardOutputDigestMismatchError,
    ShardOutputMissingError,
    ShardOutputValidation,
    ShardRecord,
    ShardStatus,
    UnknownShardError,
    build_run_manifest,
    plan_document_shards,
    shard_output_digest,
    shards_to_execute,
    validate_shard_outputs,
)

# Obviously fake markers mirrored from tests/unit/processing/test_distributed_sharding.py
# so the PHI-free assertions below have something to catch if a field ever leaks.
DOCUMENT_TEMPLATES = (
    "Patient Jane Doe called from 555-0100.",
    "Patient John Roe emailed john.roe@example.org.",
    "Patient Alex Kim lives at 12 Oak Street.",
    "Patient Maria Lee has diabetes.",
)
PHI_MARKERS = (
    "Jane Doe",
    "555-0100",
    "John Roe",
    "john.roe@example.org",
    "Alex Kim",
    "12 Oak Street",
    "Maria Lee",
    "diabetes",
)
SHARD_COUNT = 4

# Windows does not enforce POSIX permission bits for owner reads, and root
# bypasses them everywhere, so chmod(0o000) is not a portable way to make a
# file unreadable.
_POSIX_PERMISSIONS_ENFORCED = (
    os.name != "nt" and getattr(os, "geteuid", lambda: 1)() != 0
)


def _documents(count: int = 12) -> list[dict[str, str]]:
    """Build a synthetic corpus algorithmically from the fake templates."""

    return [
        {
            "id": f"note-{index:03d}",
            "text": DOCUMENT_TEMPLATES[index % len(DOCUMENT_TEMPLATES)],
        }
        for index in range(count)
    ]


def _manifest(count: int = 12) -> BatchRunManifest:
    plan = plan_document_shards(_documents(count), shard_count=SHARD_COUNT)
    return build_run_manifest(plan, run_id="run-0001", created_at=1_000.0)


def _shard_payload(shard_id: int) -> bytes:
    """Synthetic redacted output bytes for a shard."""

    return json.dumps({"shard_id": shard_id, "records": shard_id + 1}).encode("utf-8")


def _complete_shard(
    manifest: BatchRunManifest,
    shard_id: int,
    root: Path,
    *,
    payload: bytes | None = None,
) -> BatchRunManifest:
    """Write a shard output and mark its record completed."""

    output_path = f"shards/shard-{shard_id:05d}.json"
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    body = _shard_payload(shard_id) if payload is None else payload
    target.write_bytes(body)

    record = replace(
        manifest.shard(shard_id),
        status=ShardStatus.COMPLETED,
        attempts=1,
        started_at=1_010.0,
        completed_at=1_012.5,
        output_path=output_path,
        output_digest=shard_output_digest(target),
        output_bytes=len(body),
        worker_id="worker-0",
    )
    return manifest.with_shard(record, updated_at=1_020.0)


def _strings(value: Any) -> Iterator[str]:
    """Yield every key and every string value in a JSON-safe structure."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


# --- Construction ----------------------------------------------------------


def test_build_run_manifest_mirrors_the_shard_plan() -> None:
    documents = _documents()
    plan = plan_document_shards(documents, shard_count=SHARD_COUNT)

    manifest = build_run_manifest(plan, run_id="run-0001", created_at=1_000.0)

    assert manifest.schema_version == RUN_MANIFEST_SCHEMA_VERSION
    assert manifest.algorithm == plan.algorithm
    assert manifest.plan_fingerprint == plan.fingerprint
    assert manifest.shard_count == SHARD_COUNT
    assert manifest.document_count == len(documents)
    assert manifest.created_at == manifest.updated_at == 1_000.0
    assert [record.shard_id for record in manifest.shards] == list(range(SHARD_COUNT))
    assert [record.fingerprint for record in manifest.shards] == [
        shard.fingerprint for shard in plan.shards
    ]
    assert all(record.status is ShardStatus.PENDING for record in manifest.shards)
    assert manifest.pending_shards() == manifest.shards
    assert manifest.completed_shards() == ()


def test_with_shard_is_immutable_and_never_touches_attempts() -> None:
    manifest = _manifest()
    original = manifest.shard(1)

    updated = manifest.with_shard(
        replace(original, status=ShardStatus.RUNNING, attempts=3, worker_id="worker-7"),
        updated_at=1_005.0,
    )

    # The manifest is a record, not an executor: it stores attempts verbatim.
    assert updated.shard(1).attempts == 3
    assert updated.shard(1).status is ShardStatus.RUNNING
    assert updated.updated_at == 1_005.0
    assert manifest.shard(1) == original
    assert manifest.updated_at == 1_000.0


def test_shard_record_duration_needs_both_timings() -> None:
    manifest = _manifest()

    assert manifest.shard(0).duration_seconds is None
    timed = replace(manifest.shard(0), started_at=10.0, completed_at=12.5)
    assert timed.duration_seconds == 2.5


def test_with_shard_rejects_an_unknown_shard_id() -> None:
    manifest = _manifest()
    stranger = replace(manifest.shard(0), shard_id=SHARD_COUNT + 1)

    # UnknownShardError derives from both hierarchies, so callers written
    # against either keep working.
    with pytest.raises(UnknownShardError, match="unknown shard id"):
        manifest.with_shard(stranger)
    with pytest.raises(RunManifestError):
        manifest.shard(SHARD_COUNT + 1)
    with pytest.raises(KeyError):
        manifest.shard(SHARD_COUNT + 1)
    assert str(UnknownShardError("unknown shard id: 9")) == "unknown shard id: 9"


def test_with_shard_refuses_to_rewrite_the_shard_plan_binding() -> None:
    manifest = _manifest()
    original = manifest.shard(0)

    with pytest.raises(RunManifestError, match="fingerprint must not change"):
        manifest.with_shard(replace(original, fingerprint="TOTALLY_DIFFERENT"))
    with pytest.raises(RunManifestError, match="document_count must not change"):
        manifest.with_shard(replace(original, document_count=99_999))

    # Execution state may still move freely.
    moved = manifest.with_shard(
        replace(original, status=ShardStatus.RUNNING, worker_id="worker-3"),
        updated_at=1_006.0,
    )
    assert moved.shard(0).status is ShardStatus.RUNNING
    assert sum(record.document_count for record in moved.shards) == 12


# --- Field-level guards ----------------------------------------------------


def test_completed_shard_requires_an_output_path_and_digest() -> None:
    manifest = _manifest()

    with pytest.raises(RunManifestError, match="requires an output path and digest"):
        replace(manifest.shard(0), status=ShardStatus.COMPLETED)


def test_absolute_and_escaping_output_paths_are_refused() -> None:
    manifest = _manifest()
    digest = f"sha256:{'a' * 64}"

    with pytest.raises(RunManifestError, match="relative to the run root"):
        replace(manifest.shard(0), output_path="/var/run/shard-0.json")
    with pytest.raises(RunManifestError, match="escape the run root"):
        replace(manifest.shard(0), output_path="../shard-0.json", output_digest=digest)


def test_error_type_rejects_anything_that_looks_like_a_message() -> None:
    manifest = _manifest()

    ok = replace(manifest.shard(0), status=ShardStatus.FAILED, error_type="ValueError")
    assert ok.error_type == "ValueError"

    with pytest.raises(RunManifestError, match=r"never str\(exc\)"):
        replace(
            manifest.shard(0),
            status=ShardStatus.FAILED,
            error_type="ValueError: Patient Jane Doe called from 555-0100.",
        )


def test_error_type_guard_is_a_shape_check_not_a_content_guarantee() -> None:
    """Documented honestly: it stops the realistic leak, not a determined one.

    A bounded dotted identifier is accepted, so a caller who deliberately
    encodes content still gets through. The guard exists to turn the mistake
    of passing ``str(exc)`` into a construction error.
    """

    manifest = _manifest()

    smuggled = replace(
        manifest.shard(0),
        status=ShardStatus.FAILED,
        error_type="Patient_Jane_Doe_MRN_12345",
    )
    assert smuggled.error_type == "Patient_Jane_Doe_MRN_12345"

    # What it does reliably stop: messages, newlines and unbounded text.
    for bad in (
        "ValueError: patient 12 Oak Street",
        "RuntimeError\nX",
        "E" * (MAX_LABEL_LENGTH + 1),
    ):
        with pytest.raises(RunManifestError):
            replace(manifest.shard(0), status=ShardStatus.FAILED, error_type=bad)


def test_output_digest_must_carry_the_sha256_prefix() -> None:
    manifest = _manifest()

    with pytest.raises(RunManifestError, match="sha256: digest"):
        replace(
            manifest.shard(0),
            status=ShardStatus.COMPLETED,
            output_path="shards/shard-00000.json",
            output_digest="a" * 64,
        )


def test_manifest_refuses_a_shard_list_that_disagrees_with_shard_count() -> None:
    manifest = _manifest()

    with pytest.raises(RunManifestError, match="shards for shard_count"):
        replace(manifest, shards=manifest.shards[:-1])


# --- Persistence and durability -------------------------------------------


def test_manifest_round_trips_through_local_store(tmp_path: Path) -> None:
    store = LocalFileRunManifestStore(tmp_path / "run-manifest.json")
    assert store.load() is None

    manifest = _complete_shard(_manifest(), 2, tmp_path)
    store.save(manifest)

    assert store.load() == manifest
    assert BatchRunManifest.from_dict(manifest.to_dict()) == manifest


def test_partial_run_manifest_loads_and_validates(tmp_path: Path) -> None:
    store = LocalFileRunManifestStore(tmp_path / "run-manifest.json")
    manifest = _complete_shard(_complete_shard(_manifest(), 0, tmp_path), 3, tmp_path)
    manifest = manifest.with_shard(
        replace(manifest.shard(1), status=ShardStatus.RUNNING, attempts=1),
        updated_at=1_030.0,
    )
    store.save(manifest)

    reloaded = store.load()
    assert reloaded is not None
    assert {record.shard_id for record in reloaded.completed_shards()} == {0, 3}
    assert validate_shard_outputs(reloaded, root=tmp_path).all_valid


def test_in_memory_store_round_trips() -> None:
    manifest = _manifest()
    store = InMemoryRunManifestStore()

    assert store.load() is None
    store.save(manifest)
    assert store.load() == manifest


def test_unsupported_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    payload = _manifest().to_dict()
    payload["schema_version"] = RUN_MANIFEST_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestSchemaError, match="unsupported run manifest schema"):
        LocalFileRunManifestStore(path).load()


def test_malformed_manifest_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ManifestSchemaError, match="not valid JSON"):
        LocalFileRunManifestStore(path).load()


def test_manifest_missing_a_required_field_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    payload = _manifest().to_dict()
    del payload["plan_fingerprint"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestSchemaError, match="missing"):
        LocalFileRunManifestStore(path).load()


@pytest.mark.parametrize(
    ("crash_phase", "expect_replaced"),
    [
        ("after_write", False),
        ("after_fsync", False),
        ("before_replace", False),
        ("after_replace", True),
        ("after_directory_fsync", True),
    ],
)
def test_manifest_survives_every_crash_point(
    tmp_path: Path,
    crash_phase: str,
    expect_replaced: bool,
) -> None:
    path = tmp_path / "run-manifest.json"
    reader = LocalFileRunManifestStore(path)

    first = _manifest()
    reader.save(first)
    second = replace(first, updated_at=2_000.0)

    observed: list[str] = []

    def crash_hook(phase: str, target: Path) -> None:
        assert target == path
        observed.append(phase)
        if phase == crash_phase:
            raise RuntimeError("simulated power loss")

    with pytest.raises(RuntimeError, match="simulated power loss"):
        LocalFileRunManifestStore(path, atomic_write_hook=crash_hook).save(second)

    # The crash point really fired, and fired where the write sequence says it does.
    assert observed[-1] == crash_phase

    persisted = reader.load()
    assert persisted is not None
    assert persisted in {first, second}
    # Non-vacuous: the on-disk generation differs across the rename boundary.
    assert persisted.updated_at == (2_000.0 if expect_replaced else 1_000.0)
    assert persisted == (second if expect_replaced else first)
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


# --- Resume semantics ------------------------------------------------------


def test_shards_to_execute_skips_completed_shards_with_matching_digests(
    tmp_path: Path,
) -> None:
    manifest = _complete_shard(_complete_shard(_manifest(), 0, tmp_path), 2, tmp_path)

    validation = validate_shard_outputs(manifest, root=tmp_path)

    assert validation.valid == (0, 2)
    assert validation.missing == ()
    assert validation.mismatched == ()
    assert validation.all_valid
    assert shards_to_execute(manifest, root=tmp_path) == (1, 3)


def test_missing_output_marks_shard_for_reexecution(tmp_path: Path) -> None:
    manifest = _complete_shard(_complete_shard(_manifest(), 0, tmp_path), 2, tmp_path)
    (tmp_path / manifest.shard(0).output_path).unlink()

    validation = validate_shard_outputs(manifest, root=tmp_path)

    assert validation.missing == (0,)
    assert validation.valid == (2,)
    assert not validation.all_valid
    assert shards_to_execute(manifest, root=tmp_path) == (0, 1, 3)


def test_mutated_output_digest_marks_shard_for_reexecution(tmp_path: Path) -> None:
    manifest = _complete_shard(_complete_shard(_manifest(), 0, tmp_path), 2, tmp_path)
    (tmp_path / manifest.shard(2).output_path).write_bytes(b'{"truncated":true}')

    validation = validate_shard_outputs(manifest, root=tmp_path)

    assert validation.mismatched == (2,)
    assert validation.valid == (0,)
    assert shards_to_execute(manifest, root=tmp_path) == (1, 2, 3)


def test_strict_validation_raises_for_missing_and_corrupted_outputs(
    tmp_path: Path,
) -> None:
    manifest = _complete_shard(_manifest(), 1, tmp_path)
    output = tmp_path / manifest.shard(1).output_path

    output.write_bytes(b"corrupted")
    with pytest.raises(ShardOutputDigestMismatchError, match="shard 1"):
        validate_shard_outputs(manifest, root=tmp_path, strict=True)

    output.unlink()
    with pytest.raises(ShardOutputMissingError, match="shard 1"):
        validate_shard_outputs(manifest, root=tmp_path, strict=True)


def test_failed_shards_are_requeued_without_touching_their_attempt_count(
    tmp_path: Path,
) -> None:
    manifest = _complete_shard(_manifest(), 0, tmp_path)
    manifest = manifest.with_shard(
        replace(
            manifest.shard(1),
            status=ShardStatus.FAILED,
            attempts=2,
            error_type="TimeoutError",
        ),
        updated_at=1_040.0,
    )

    assert shards_to_execute(manifest, root=tmp_path) == (1, 2, 3)
    assert manifest.shard(1).attempts == 2


def test_output_paths_resolve_against_a_relocatable_run_root(tmp_path: Path) -> None:
    source = tmp_path / "run-a"
    source.mkdir()
    manifest = _complete_shard(_manifest(), 3, source)

    relocated = tmp_path / "run-b"
    (relocated / "shards").mkdir(parents=True)
    (relocated / manifest.shard(3).output_path).write_bytes(_shard_payload(3))

    assert validate_shard_outputs(manifest, root=relocated).valid == (3,)
    assert shards_to_execute(manifest, root=relocated) == (0, 1, 2)


# --- PHI-free guarantees ---------------------------------------------------


def test_manifest_json_excludes_document_text_and_ids(tmp_path: Path) -> None:
    documents = _documents()
    manifest = _manifest()
    for shard_id in range(SHARD_COUNT):
        manifest = _complete_shard(manifest, shard_id, tmp_path)
    manifest = manifest.with_shard(
        replace(
            manifest.shard(1),
            status=ShardStatus.FAILED,
            attempts=2,
            error_type="RuntimeError",
            output_path=None,
            output_digest=None,
            output_bytes=None,
        ),
        updated_at=1_050.0,
    )

    payload = manifest.to_dict()
    rendered = json.dumps(payload, sort_keys=True)
    # Keys are checked alongside values: a leak can ride through either side.
    strings = list(_strings(payload))

    for marker in PHI_MARKERS:
        assert marker not in rendered
        assert all(marker not in item for item in strings)
    for document in documents:
        assert document["id"] not in rendered
        assert all(document["id"] not in item for item in strings)
        assert document["text"] not in rendered

    assert "document_hashes" not in rendered
    assert "document_ids" not in rendered
    assert "text" not in {key for key in payload}
    assert "plan_fingerprint" in payload


def test_persisted_manifest_file_contains_no_document_text(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    manifest = _complete_shard(_manifest(), 0, tmp_path)
    LocalFileRunManifestStore(path).save(manifest)

    written = path.read_text(encoding="utf-8")

    for marker in PHI_MARKERS:
        assert marker not in written
    for document in _documents():
        assert document["id"] not in written
    assert written.endswith("\n")
    assert json.loads(written)["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION


def test_shard_record_field_names_are_the_reviewed_phi_free_set() -> None:
    record = ShardRecord(shard_id=0, fingerprint="a" * 64, document_count=3)

    assert set(record.to_dict()) == {
        "shard_id",
        "fingerprint",
        "document_count",
        "status",
        "attempts",
        "started_at",
        "completed_at",
        "output_path",
        "output_digest",
        "output_bytes",
        "worker_id",
        "error_type",
    }


# --- Deserialization is a trust boundary -----------------------------------


def test_from_dict_refuses_foreign_types_instead_of_coercing_them(
    tmp_path: Path,
) -> None:
    """Coercion here would launder arbitrary content back onto disk.

    ``str()`` never fails, so a bare coercion accepts a dict, a list or None
    for any string field and renders it into the manifest. Non-Python writers
    feed this path, so every field is type-checked instead.
    """

    leak = {"note": "Patient Jane Doe called from 555-0100."}

    payload = _manifest().to_dict()
    payload["shards"][0]["worker_id"] = leak
    with pytest.raises(ManifestSchemaError, match="worker_id must be a string"):
        BatchRunManifest.from_dict(payload)

    for field_name in ("run_id", "algorithm", "plan_fingerprint", "openmed_version"):
        payload = _manifest().to_dict()
        payload[field_name] = None
        with pytest.raises(ManifestSchemaError, match=f"{field_name} must be a string"):
            BatchRunManifest.from_dict(payload)

    payload = _manifest().to_dict()
    payload["shards"][0]["fingerprint"] = None
    with pytest.raises(ManifestSchemaError, match="fingerprint must be a string"):
        BatchRunManifest.from_dict(payload)

    payload = _manifest().to_dict()
    payload["shards"][0]["output_path"] = ["a", "b"]
    with pytest.raises(ManifestSchemaError, match="output_path must be a string"):
        BatchRunManifest.from_dict(payload)

    payload = _manifest().to_dict()
    payload["shards"][0]["status"] = 3
    with pytest.raises(ManifestSchemaError, match="status must be a string"):
        BatchRunManifest.from_dict(payload)


def test_foreign_worker_id_cannot_be_laundered_back_onto_disk(tmp_path: Path) -> None:
    """End-to-end through the real store: load must refuse, not re-serialize."""

    path = tmp_path / "run-manifest.json"
    payload = _manifest().to_dict()
    payload["shards"][0]["worker_id"] = {
        "note": "Patient Jane Doe called from 555-0100."
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = LocalFileRunManifestStore(path)

    with pytest.raises(ManifestSchemaError):
        store.load()

    # Nothing reached a manifest, so nothing can be re-saved from one.
    reserialized = tmp_path / "reserialized.json"
    LocalFileRunManifestStore(reserialized).save(_manifest())
    written = reserialized.read_text(encoding="utf-8")
    assert "Jane Doe" not in written
    assert "555-0100" not in written


def test_null_fingerprint_cannot_masquerade_as_a_valid_plan_binding() -> None:
    """A 'None' fingerprint would let a bogus manifest look plan-bound."""

    payload = _manifest().to_dict()
    for shard in payload["shards"]:
        shard["fingerprint"] = None

    with pytest.raises(ManifestSchemaError, match="fingerprint must be a string"):
        BatchRunManifest.from_dict(payload)


# --- Trailing-newline and control-character injection ----------------------


def test_trailing_newline_is_rejected_in_digests_and_error_types() -> None:
    """`$` matches before a final newline; a smuggled one never re-digests."""

    manifest = _manifest()
    digest = f"sha256:{'a' * 64}"

    with pytest.raises(RunManifestError, match="sha256: digest"):
        replace(
            manifest.shard(0),
            status=ShardStatus.COMPLETED,
            output_path="shards/shard-00000.json",
            output_digest=f"{digest}\n",
        )
    with pytest.raises(RunManifestError, match="control characters"):
        replace(
            manifest.shard(0), status=ShardStatus.FAILED, error_type="RuntimeError\n"
        )
    with pytest.raises(RunManifestError, match="control characters"):
        replace(manifest.shard(0), worker_id="worker-0\nINJECTED: true")
    with pytest.raises(RunManifestError, match="control characters"):
        replace(
            manifest.shard(0),
            status=ShardStatus.COMPLETED,
            output_path="shards/a\nb.json",
            output_digest=digest,
        )


def test_error_type_and_worker_id_are_length_capped() -> None:
    manifest = _manifest()

    ok = replace(manifest.shard(0), worker_id="w" * MAX_LABEL_LENGTH)
    assert ok.worker_id is not None

    with pytest.raises(RunManifestError, match="at most"):
        replace(manifest.shard(0), worker_id="w" * (MAX_LABEL_LENGTH + 1))
    with pytest.raises(RunManifestError, match="at most"):
        replace(
            manifest.shard(0),
            status=ShardStatus.FAILED,
            error_type="E" * (MAX_LABEL_LENGTH + 1),
        )


def test_run_id_carries_the_same_bounds_as_the_other_free_text_labels(
    tmp_path: Path,
) -> None:
    """run_id flows verbatim into reports and CLI output, so bound it too."""

    manifest = _manifest()

    assert replace(manifest, run_id="r" * MAX_LABEL_LENGTH).run_id

    with pytest.raises(RunManifestError, match="at most"):
        replace(manifest, run_id="r" * (MAX_LABEL_LENGTH + 1))
    with pytest.raises(RunManifestError, match="control characters"):
        replace(manifest, run_id="run-0001\nINJECTED: true")
    with pytest.raises(RunManifestError, match="non-empty"):
        replace(manifest, run_id="   ")

    # The guard also holds on the deserialization boundary, where a foreign
    # writer -- not the operator -- controls the value.
    path = tmp_path / "run-manifest.json"
    payload = manifest.to_dict()
    payload["run_id"] = "run-0001\nstatus: completed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunManifestError, match="control characters"):
        LocalFileRunManifestStore(path).load()


# --- Non-finite floats -----------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_timestamps_are_refused(bad: float) -> None:
    manifest = _manifest()

    with pytest.raises(RunManifestError, match="finite"):
        replace(manifest.shard(0), started_at=bad)
    with pytest.raises(RunManifestError, match="finite"):
        replace(manifest, updated_at=bad)


def test_non_finite_timestamps_are_refused_on_load(tmp_path: Path) -> None:
    # json.dumps emits a bare NaN token by default, which json.loads accepts
    # and strict parsers in other languages reject; refuse it on the way in.
    path = tmp_path / "run-manifest.json"
    payload = _manifest().to_dict()
    payload["updated_at"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert "NaN" in path.read_text(encoding="utf-8")
    with pytest.raises(RunManifestError, match="finite"):
        LocalFileRunManifestStore(path).load()

    payload["updated_at"] = float("inf")
    with pytest.raises(RunManifestError, match="finite"):
        BatchRunManifest.from_dict(payload)


def test_persisted_manifest_is_strict_json_with_no_nan_tokens(tmp_path: Path) -> None:
    path = tmp_path / "run-manifest.json"
    LocalFileRunManifestStore(path).save(_complete_shard(_manifest(), 0, tmp_path))

    written = path.read_text(encoding="utf-8")

    assert "NaN" not in written
    assert "Infinity" not in written
    # parse_constant fires only for NaN/Infinity/-Infinity tokens.
    json.loads(
        written,
        parse_constant=lambda token: pytest.fail(f"non-strict JSON token: {token}"),
    )


# --- I/O failures stay inside the error hierarchy --------------------------


def test_corrupted_manifests_raise_run_manifest_error(tmp_path: Path) -> None:
    """A sibling recovering with `except RunManifestError` must not crash."""

    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b'{"schema_version": 1, "run_id": "\xff\xfe"}')
    with pytest.raises(RunManifestError, match="not valid UTF-8"):
        LocalFileRunManifestStore(invalid_utf8).load()

    directory = tmp_path / "a-directory"
    directory.mkdir()
    with pytest.raises(RunManifestError, match="not readable"):
        LocalFileRunManifestStore(directory).load()


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(13, "Permission denied"),
        OSError(5, "Input/output error"),
    ],
    ids=["permission-denied", "io-error"],
)
def test_read_failures_surface_as_run_manifest_error_on_every_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: OSError,
) -> None:
    """The product contract, verified without depending on POSIX permissions.

    What matters to callers is that no read failure escapes the
    ``RunManifestError`` hierarchy, so ``except RunManifestError`` is enough to
    recover. Injecting the ``OSError`` keeps that covered on every platform;
    the POSIX-only test below covers the permission-bit mechanism itself.
    """

    path = tmp_path / "run-manifest.json"
    path.write_text("{}", encoding="utf-8")

    def deny(*args: Any, **kwargs: Any) -> str:
        raise error

    monkeypatch.setattr(Path, "read_text", deny)

    with pytest.raises(RunManifestError, match="not readable"):
        LocalFileRunManifestStore(path).load()


@pytest.mark.skipif(
    not _POSIX_PERMISSIONS_ENFORCED,
    reason=(
        "POSIX permission bits are not enforced for owner reads on Windows, "
        "and root bypasses them; the portable contract is covered above"
    ),
)
def test_posix_permission_bits_make_a_manifest_unreadable(tmp_path: Path) -> None:
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{}", encoding="utf-8")
    unreadable.chmod(0o000)
    try:
        with pytest.raises(RunManifestError, match="not readable"):
            LocalFileRunManifestStore(unreadable).load()
    finally:
        unreadable.chmod(0o600)


# --- Containment -----------------------------------------------------------


def test_symlinked_output_escaping_the_run_root_is_not_trusted(
    tmp_path: Path,
) -> None:
    """The relative-path rule is enforced against the filesystem, not the string."""

    root = tmp_path / "run"
    (root / "shards").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(_shard_payload(0))

    manifest = _manifest().with_shard(
        replace(
            _manifest().shard(0),
            status=ShardStatus.COMPLETED,
            attempts=1,
            output_path="shards/shard-00000.json",
            output_digest=shard_output_digest(outside),
            output_bytes=len(_shard_payload(0)),
        ),
        updated_at=1_020.0,
    )
    (root / "shards" / "shard-00000.json").symlink_to(outside)

    validation = validate_shard_outputs(manifest, root=root)

    assert validation.missing == (0,)
    assert validation.valid == ()
    assert 0 in shards_to_execute(manifest, root=root)
    with pytest.raises(ShardOutputMissingError, match="outside the run root"):
        validate_shard_outputs(manifest, root=root, strict=True)


# --- Frozen means frozen ---------------------------------------------------


def test_manifest_coerces_its_shard_sequence_so_it_cannot_be_mutated() -> None:
    manifest = _manifest()
    mutable = list(manifest.shards)

    rebuilt = replace(manifest, shards=mutable)
    mutable.append(mutable[0])

    assert isinstance(rebuilt.shards, tuple)
    assert len(rebuilt.shards) == rebuilt.shard_count
    assert hash(rebuilt) == hash(rebuilt)

    with pytest.raises(RunManifestError, match="sequence of shard records"):
        replace(manifest, shards="not-a-sequence")
    with pytest.raises(RunManifestError, match="ShardRecord entries"):
        replace(manifest, shards=(1, 2, 3, 4))


def test_shard_output_validation_coerces_its_buckets_to_tuples() -> None:
    validation = ShardOutputValidation(valid=[1], missing=[2], mismatched=[3])

    assert validation.valid == (1,)
    assert validation.missing == (2,)
    assert validation.mismatched == (3,)
    assert hash(validation) == hash(validation)


# --- Consumer-facing contract ---------------------------------------------


def test_stores_satisfy_the_runtime_checkable_store_protocol(tmp_path: Path) -> None:
    assert isinstance(InMemoryRunManifestStore(), RunManifestStore)
    assert isinstance(LocalFileRunManifestStore(tmp_path / "m.json"), RunManifestStore)
    assert not isinstance(object(), RunManifestStore)


def test_running_shards_are_returned_for_the_executor_to_adjudicate(
    tmp_path: Path,
) -> None:
    """Straggler-versus-orphan policy needs liveness data this module lacks."""

    manifest = _complete_shard(_manifest(), 0, tmp_path)
    manifest = manifest.with_shard(
        replace(manifest.shard(2), status=ShardStatus.RUNNING, attempts=1),
        updated_at=1_060.0,
    )

    assert shards_to_execute(manifest, root=tmp_path) == (1, 2, 3)
