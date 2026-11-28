"""Value-free model lifecycle events from prefetch, verification and eviction."""

from __future__ import annotations

import json
import socket
import sys
import types
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.core import hf_hub, model_cache_policy, model_lifecycle
from openmed.core.hf_hub import DownloadIntegrityError, prefetch_model
from openmed.core.model_cache_policy import ModelCachePolicy
from openmed.core.model_lifecycle import (
    EVENT_FIELDS,
    MODEL_LIFECYCLE_EVENT_TYPES,
    MODEL_LIFECYCLE_REASON_CODES,
    MODEL_LIFECYCLE_SCHEMA_VERSION,
    NULL_MODEL_LIFECYCLE_SINK,
    ModelLifecycleEvent,
    ModelLifecycleEventError,
    ModelLifecycleEventType,
    RecordingModelLifecycleSink,
    emit_model_activated,
    emit_model_rolled_back,
)
from tests.fixtures.model_lifecycle import (
    VECTORS_PATH,
    install_fake_hub,
    run_scenario,
)

SCHEMA = json.loads(
    (
        Path(model_lifecycle.__file__).with_name("schemas")
        / "json"
        / "model_lifecycle_event.schema.json"
    ).read_text(encoding="utf-8")
)
VECTORS = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
SEQUENCES = {sequence["id"]: sequence for sequence in VECTORS["sequences"]}
SYNTHETIC_TOKEN = "hf_syntheticTokenValueForLifecycleTests0001"


def _scenario(case_id: str) -> dict:
    return json.loads(json.dumps(SEQUENCES[case_id]["scenario"]))


def _refuse_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model lifecycle events must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_schema_and_vectors_agree_with_the_python_vocabulary() -> None:
    Draft202012Validator.check_schema(SCHEMA)
    validator = Draft202012Validator(SCHEMA)

    assert VECTORS["synthetic"] is True
    assert VECTORS["version"] == 1
    assert VECTORS["event_schema_version"] == MODEL_LIFECYCLE_SCHEMA_VERSION
    assert VECTORS["event_schema"] == SCHEMA["$id"]
    assert tuple(VECTORS["event_types"]) == MODEL_LIFECYCLE_EVENT_TYPES
    assert tuple(VECTORS["event_fields"]) == EVENT_FIELDS
    assert tuple(VECTORS["reason_codes"]) == MODEL_LIFECYCLE_REASON_CODES
    assert SCHEMA["properties"]["event"]["enum"] == list(MODEL_LIFECYCLE_EVENT_TYPES)
    assert sorted(SCHEMA["required"]) == sorted(EVENT_FIELDS)
    seen = set()
    for sequence in VECTORS["sequences"]:
        for event in sequence["events"]:
            assert list(event) == list(EVENT_FIELDS)
            assert validator.is_valid(event)
            seen.add(event["event"])
            if event["reason"] is not None and event["event"] != "rolled_back":
                assert event["reason"] in MODEL_LIFECYCLE_REASON_CODES
    assert seen == set(MODEL_LIFECYCLE_EVENT_TYPES)


@pytest.mark.parametrize("case_id", sorted(SEQUENCES))
def test_each_vector_sequence_replays_exactly(
    case_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_network(monkeypatch)

    events, outcome = run_scenario(_scenario(case_id), tmp_path, monkeypatch)

    assert events == SEQUENCES[case_id]["events"]
    assert outcome == SEQUENCES[case_id]["outcome"]


def test_fake_hub_run_emits_ordered_events_whose_bytes_sum_to_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _scenario("prefetch_two_files")
    events, outcome = run_scenario(scenario, tmp_path, monkeypatch)
    downloaded = sum(
        path.stat().st_size
        for path in (tmp_path / "hub" / "snapshots").rglob("*")
        if path.is_file()
    )

    assert outcome == "ok"
    order = [event["event"] for event in events]
    assert order == [
        "requested",
        "fetching",
        "verified",
        "fetching",
        "verified",
        "ready",
    ]
    fetching = [event for event in events if event["event"] == "fetching"]
    verified = [event for event in events if event["event"] == "verified"]
    assert sum(event["byte_count"] for event in fetching) == downloaded
    assert sum(event["byte_count"] for event in verified) == downloaded
    assert events[-1]["byte_count"] == downloaded
    assert events[-1]["file_count"] == len(scenario["files"])
    assert {event["revision_sha"] for event in events[1:]} == {scenario["commit"]}


def test_corrupted_file_reports_a_reason_without_url_token_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", SYNTHETIC_TOKEN)
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example.invalid")
    scenario = _scenario("prefetch_corruption_persists")

    events, outcome = run_scenario(scenario, tmp_path, monkeypatch)

    assert outcome == "download_integrity_error"
    failures = [event for event in events if event["event"] == "integrity_failed"]
    assert failures and all(event["reason"] == "digest_mismatch" for event in failures)
    assert events[-1]["event"] == "failed"
    assert events[-1]["reason"] == "integrity_retry_exhausted"
    rendered = json.dumps(events)
    for forbidden in (
        SYNTHETIC_TOKEN,
        "http",
        "example.invalid",
        str(tmp_path),
        "/",
        "config.json",
    ):
        if forbidden == "/":
            # The only slash allowed is the one inside the Hub repo id.
            assert all(
                "/" not in str(value)
                for event in events
                for key, value in event.items()
                if key != "repo_id"
            )
        else:
            assert forbidden not in rendered


def test_integrity_failure_keeps_the_existing_error_and_adds_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_hub(monkeypatch, _scenario("prefetch_corruption_persists"), tmp_path)
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    with pytest.raises(DownloadIntegrityError) as raised:
        prefetch_model(
            "OpenMed/synthetic-lifecycle-model",
            cache_dir=str(tmp_path / "hub"),
            retries=0,
        )

    assert raised.value.reason == "integrity_retry_exhausted"
    assert "after one forced re-fetch" in str(raised.value)


def test_eviction_emits_one_event_per_artifact_with_digest_and_size(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    sink = RecordingModelLifecycleSink()
    policy = ModelCachePolicy(cache, quota_bytes=0, lifecycle_sink=sink)
    expected = {}
    for index, content in enumerate((b"synthetic-a", b"synthetic-bb")):
        path = cache / f"artifact-{index}.bin"
        path.write_bytes(content)
        artifact = policy.register_artifact(path, last_accessed_ns=index)
        expected[artifact.sha256] = len(content)

    result = policy.enforce_quota()

    events = sink.to_list()
    assert [event["event"] for event in events] == ["evicted", "evicted"]
    assert {event["digest"]: event["byte_count"] for event in events} == expected
    assert sum(event["byte_count"] for event in events) == result.bytes_freed
    assert all(event["repo_id"] is None for event in events)
    assert str(tmp_path) not in json.dumps(events)


def test_dry_run_and_skipped_evictions_emit_nothing(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    sink = RecordingModelLifecycleSink()
    path = cache / "pinned.bin"
    path.write_bytes(b"synthetic")
    policy = ModelCachePolicy(cache, quota_bytes=0, lifecycle_sink=sink)
    policy.register_artifact(path)

    policy.apply_eviction(policy.plan_eviction(), dry_run=True)
    policy.pin_artifact(path)
    policy.enforce_quota()

    assert sink.events == ()


def test_default_sink_records_nothing_and_makes_no_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_network(monkeypatch)

    def no_event(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the default sink must not build events")

    monkeypatch.setattr(hf_hub, "ModelLifecycleEvent", no_event)
    monkeypatch.setattr(model_cache_policy, "ModelLifecycleEvent", no_event)
    install_fake_hub(monkeypatch, _scenario("prefetch_single_file"), tmp_path)
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    path = prefetch_model(
        "OpenMed/synthetic-lifecycle-model", cache_dir=str(tmp_path / "hub")
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.bin").write_bytes(b"synthetic")
    policy = ModelCachePolicy(cache, quota_bytes=0)
    policy.register_artifact(cache / "a.bin")
    result = policy.enforce_quota()

    assert Path(path).is_dir() and result.evicted_count == 1
    assert not hasattr(NULL_MODEL_LIFECYCLE_SINK, "__dict__")
    assert (
        NULL_MODEL_LIFECYCLE_SINK.emit(
            ModelLifecycleEvent(ModelLifecycleEventType.REQUESTED)
        )
        is None
    )


def test_a_failing_sink_never_changes_the_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        def emit(self, event: ModelLifecycleEvent) -> None:
            raise RuntimeError("synthetic sink failure")

    install_fake_hub(monkeypatch, _scenario("prefetch_single_file"), tmp_path)
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)

    path = prefetch_model(
        "OpenMed/synthetic-lifecycle-model",
        cache_dir=str(tmp_path / "hub"),
        lifecycle_sink=Broken(),
    )

    assert Path(path).is_dir()


def test_a_sink_without_emit_is_rejected_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        ModelCachePolicy(tmp_path, quota_bytes=0, lifecycle_sink=object())
    with pytest.raises(TypeError):
        prefetch_model("OpenMed/synthetic-lifecycle-model", lifecycle_sink=object())


def test_activation_hooks_emit_activated_and_rolled_back() -> None:
    sink = RecordingModelLifecycleSink()
    digest = "sha256:" + "c" * 64

    emit_model_activated(
        sink, repo_id="OpenMed/synthetic-model", revision_sha="d" * 40, digest=digest
    )
    emit_model_rolled_back(
        sink,
        repo_id="OpenMed/synthetic-model",
        revision_sha="d" * 40,
        reason="health_window_failed",
    )

    assert [event["event"] for event in sink.to_list()] == ["activated", "rolled_back"]
    assert sink.to_list()[1]["reason"] == "health_window_failed"
    validator = Draft202012Validator(SCHEMA)
    assert all(validator.is_valid(event) for event in sink.to_list())


@pytest.mark.parametrize(
    "fields",
    [
        {"event": "downloaded"},
        {"event": "requested", "repo_id": "https://huggingface.co/OpenMed/model"},
        {"event": "requested", "repo_id": "/synthetic/private/model"},
        {"event": "ready", "revision_sha": "main"},
        {"event": "evicted", "byte_count": 1, "digest": "/synthetic/cache/a.bin"},
        {"event": "failed", "reason": "Synthetic failure text"},
        {"event": "failed"},
        {"event": "rolled_back"},
        {"event": "fetching", "file_count": 1, "byte_count": 1},
        {"event": "evicted", "digest": "sha256:" + "a" * 64},
        {"event": "ready", "byte_count": -1},
        {"event": "ready", "file_count": True},
    ],
)
def test_events_reject_values_outside_the_value_free_vocabulary(
    fields: dict,
) -> None:
    event_type = fields.pop("event")

    with pytest.raises(ModelLifecycleEventError) as raised:
        ModelLifecycleEvent(event_type, **fields)

    message = str(raised.value)
    assert "synthetic" not in message.lower() and "https" not in message
    record = {
        "schema_version": MODEL_LIFECYCLE_SCHEMA_VERSION,
        "event": event_type,
        **{name: None for name in EVENT_FIELDS[2:]},
        **fields,
    }
    assert not Draft202012Validator(SCHEMA).is_valid(record)


def test_offline_cache_hit_reports_ready_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_network(monkeypatch)
    calls = install_fake_hub(monkeypatch, _scenario("offline_cache_hit"), tmp_path)
    sink = RecordingModelLifecycleSink()

    prefetch_model(
        "OpenMed/synthetic-lifecycle-model",
        config=types.SimpleNamespace(local_only=True),
        lifecycle_sink=sink,
    )

    assert calls["snapshot"] == [True] and calls["download"] == []
    assert [event.event.value for event in sink.events] == ["requested", "ready"]


@pytest.mark.parametrize(
    "parent",
    ["C:\\Users\\synthetic\\hub\\snapshots", "/home/synthetic/hub/snapshots"],
    ids=["windows", "posix"],
)
def test_offline_cache_hit_reads_the_commit_from_either_path_style(
    parent: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_network(monkeypatch)
    scenario = _scenario("offline_cache_hit")
    install_fake_hub(monkeypatch, scenario, tmp_path)
    separator = "\\" if "\\" in parent else "/"
    snapshot = f"{parent}{separator}{scenario['commit']}"
    monkeypatch.setattr(
        sys.modules["huggingface_hub"],
        "snapshot_download",
        lambda **_kwargs: snapshot,
    )
    sink = RecordingModelLifecycleSink()

    result = prefetch_model(
        scenario["repo_id"],
        config=types.SimpleNamespace(local_only=True),
        lifecycle_sink=sink,
    )

    assert result == snapshot
    assert [event["revision_sha"] for event in sink.to_list()] == [
        None,
        scenario["commit"],
    ]
