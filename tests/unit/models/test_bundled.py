"""Focused tests for the opt-in bundled offline model bootstrap."""

from __future__ import annotations

import socket
from dataclasses import replace
from typing import Any

import pytest

from openmed.core.config import OpenMedConfig
from openmed.core.offline import OfflineModeError
from openmed.models.bundled import (
    BUNDLED_MODEL_MANIFEST,
    BundledModelError,
    get_bundled_model_manifest,
    load_bundled_model,
    validate_bundled_model_manifest,
)


class _RecordingLoader:
    """ModelLoader-compatible synthetic loader for offline tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def load_model(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((args, kwargs))
        return {"model": "synthetic-bundled-model"}


class _NetworkProbeLoader:
    """Loader that would attempt egress if the bootstrap were not guarded."""

    def load_model(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        return {}


def test_bundled_manifest_is_versioned_and_matches_registry() -> None:
    manifest = validate_bundled_model_manifest()

    assert manifest is BUNDLED_MODEL_MANIFEST
    assert manifest.version == "1.0.0"
    assert manifest["schema_version"] == "openmed.bundled-model.v1"
    assert manifest.checksum.startswith("sha256:")
    assert manifest.license == "apache-2.0"
    assert manifest.opt_in is True
    assert manifest.offline is True


def test_only_the_declared_bundle_is_opted_in() -> None:
    assert get_bundled_model_manifest("pii_detection") is BUNDLED_MODEL_MANIFEST
    assert (
        get_bundled_model_manifest(BUNDLED_MODEL_MANIFEST.model_id)
        is BUNDLED_MODEL_MANIFEST
    )

    with pytest.raises(KeyError, match="no bundled model"):
        get_bundled_model_manifest("disease_detection_tiny")


def test_stale_checksum_is_rejected_before_loading() -> None:
    stale = replace(
        BUNDLED_MODEL_MANIFEST,
        checksum="sha256:" + "0" * 64,
    )

    with pytest.raises(BundledModelError, match="stale registry checksum"):
        validate_bundled_model_manifest(stale)


def test_load_uses_registry_key_and_forces_local_only() -> None:
    loader = _RecordingLoader()
    config = OpenMedConfig(local_only=False)

    result = load_bundled_model(config=config, loader=loader)

    assert result == {"model": "synthetic-bundled-model"}
    assert config.local_only is False
    assert loader.calls == [
        ((BUNDLED_MODEL_MANIFEST.model_key,), {"local_files_only": True})
    ]


def test_load_rejects_an_explicit_network_fallback() -> None:
    loader = _RecordingLoader()

    with pytest.raises(BundledModelError, match="network fallback is disabled"):
        load_bundled_model(loader=loader, local_files_only=False)

    assert loader.calls == []


def test_socket_egress_is_blocked_for_the_entire_load() -> None:
    with pytest.raises(OfflineModeError, match="network access"):
        load_bundled_model(loader=_NetworkProbeLoader())
