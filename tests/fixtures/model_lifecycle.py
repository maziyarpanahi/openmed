"""Synthetic fake-hub and cache scenarios for model lifecycle event vectors.

Each scenario in ``tests/fixtures/parity/model_lifecycle_v1.json`` describes
invented repository files or cache artifacts. :func:`run_scenario` replays it
through the real Python emitters against an in-process fake Hub, with no
network access, and returns the recorded events.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Any

from openmed.core.hf_hub import DownloadIntegrityError, prefetch_model
from openmed.core.model_cache_policy import ModelCachePolicy
from openmed.core.model_lifecycle import (
    RecordingModelLifecycleSink,
    emit_model_activated,
    emit_model_rolled_back,
)
from openmed.core.offline import OfflineModeError

VECTORS_PATH = Path(__file__).resolve().parent / "parity" / "model_lifecycle_v1.json"


class FakeLocalEntryNotFoundError(FileNotFoundError):
    """Stand-in for ``huggingface_hub`` ``LocalEntryNotFoundError``."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def install_fake_hub(monkeypatch: Any, scenario: dict, root: Path) -> dict:
    """Install a fake ``huggingface_hub`` that serves ``scenario`` files."""

    files = {item["filename"]: item for item in scenario.get("files", [])}
    snapshot = root / "hub" / "snapshots" / scenario.get("commit", "0" * 40)
    attempts = {name: 0 for name in files}
    calls: dict[str, list] = {"model_info": [], "download": [], "snapshot": []}

    def model_info(repo_id: str, **kwargs: Any) -> Any:
        calls["model_info"].append(repo_id)
        siblings = []
        for name, item in files.items():
            content = item["content"].encode("utf-8")
            lfs = types.SimpleNamespace(sha256=_sha256(content))
            if scenario.get("metadata") == "missing_digest":
                lfs = None
            siblings.append(
                types.SimpleNamespace(
                    rfilename=name, size=len(content), blob_id=None, lfs=lfs
                )
            )
        return types.SimpleNamespace(sha=scenario["commit"], siblings=siblings)

    class FakeHfApi:
        def model_info(self, repo_id: str, **kwargs: Any) -> Any:
            return model_info(repo_id, **kwargs)

    def hf_hub_download(**kwargs: Any) -> str:
        name = kwargs["filename"]
        calls["download"].append(name)
        attempts[name] += 1
        content = files[name]["content"].encode("utf-8")
        if attempts[name] <= files[name].get("corrupt_downloads", 0):
            content = b"#" * len(content)
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        progress = kwargs["tqdm_class"](total=len(content), initial=0)
        progress.update(len(content))
        return str(path)

    def snapshot_download(**kwargs: Any) -> str:
        calls["snapshot"].append(kwargs["local_files_only"])
        if scenario["mode"] == "offline_miss":
            raise FakeLocalEntryNotFoundError("synthetic cache miss")
        return str(snapshot)

    module = types.ModuleType("huggingface_hub")
    module.HfApi = FakeHfApi
    module.hf_hub_download = hf_hub_download
    module.snapshot_download = snapshot_download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError = FakeLocalEntryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return calls


def _run_prefetch(scenario: dict, root: Path, monkeypatch: Any) -> tuple:
    install_fake_hub(monkeypatch, scenario, root)
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)
    sink = RecordingModelLifecycleSink()
    config = types.SimpleNamespace(local_only=scenario["mode"] != "online")
    try:
        prefetch_model(
            scenario["repo_id"],
            revision=scenario.get("revision"),
            cache_dir=str(root / "hub"),
            config=config,
            retries=0,
            lifecycle_sink=sink,
        )
        outcome = "ok"
    except DownloadIntegrityError:
        outcome = "download_integrity_error"
    except OfflineModeError:
        outcome = "offline_mode_error"
    return sink.to_list(), outcome


def _run_eviction(scenario: dict, root: Path) -> tuple:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    sink = RecordingModelLifecycleSink()
    policy = ModelCachePolicy(
        cache, quota_bytes=scenario["quota_bytes"], lifecycle_sink=sink
    )
    for item in scenario["artifacts"]:
        path = cache / item["name"]
        path.write_bytes(item["content"].encode("utf-8"))
        policy.register_artifact(path, last_accessed_ns=item["last_accessed_ns"])
    plan = policy.plan_eviction()
    for item in scenario["artifacts"]:
        if item.get("deleted_after_plan"):
            (cache / item["name"]).unlink()
    result = policy.apply_eviction(plan)
    return sink.to_list(), f"evicted_{result.evicted_count}"


def _run_activation(scenario: dict) -> tuple:
    sink = RecordingModelLifecycleSink()
    for step in scenario["steps"]:
        if step["hook"] == "activated":
            emit_model_activated(
                sink,
                repo_id=scenario["repo_id"],
                revision_sha=step["revision_sha"],
                digest=step.get("digest"),
            )
        else:
            emit_model_rolled_back(
                sink,
                repo_id=scenario["repo_id"],
                revision_sha=step["revision_sha"],
                reason=step["reason"],
                digest=step.get("digest"),
            )
    return sink.to_list(), "ok"


def run_scenario(scenario: dict, root: Path, monkeypatch: Any) -> tuple:
    """Replay one scenario and return ``(events, outcome)``."""

    operation = scenario["operation"]
    if operation == "prefetch":
        return _run_prefetch(scenario, root, monkeypatch)
    if operation == "evict":
        return _run_eviction(scenario, root)
    return _run_activation(scenario)
