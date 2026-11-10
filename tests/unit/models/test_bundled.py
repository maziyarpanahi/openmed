"""Focused tests for the opt-in bundled offline model bootstrap."""

from __future__ import annotations

import socket
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import pytest

from openmed.core.config import OpenMedConfig
from openmed.core.models import ModelLoader
from openmed.core.offline import OfflineModeError
from openmed.models.bundled import (
    BUNDLED_MODEL_MANIFEST,
    BundledModelError,
    BundledModelManifest,
    BundledModelUnavailableError,
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


class _FailingLoader:
    """Loader whose raw diagnostic must not escape the bootstrap boundary."""

    def load_model(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise OSError("synthetic-sensitive-cache-location")


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
        (
            (BUNDLED_MODEL_MANIFEST.model_key,),
            {"local_files_only": True, "require_integrity": True},
        )
    ]


def test_integrity_required_load_does_not_reuse_unverified_memory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openmed.core.models.HF_AVAILABLE", True)
    loader = ModelLoader(OpenMedConfig())
    full_model_name = BUNDLED_MODEL_MANIFEST.model_id
    cached_model = Mock(config=Mock())
    cached_tokenizer = Mock()
    loader._models[full_model_name] = cached_model
    loader._tokenizers[full_model_name] = cached_tokenizer
    local_loading = Mock(return_value={"local_files_only": True})
    monkeypatch.setattr(loader, "_local_loading_kwargs", local_loading)
    monkeypatch.setattr(
        loader,
        "_prepare_model_reference",
        Mock(return_value="/verified/synthetic-model"),
    )
    monkeypatch.setattr(loader, "_resolve_torch_device", Mock(return_value="cpu"))
    monkeypatch.setattr(loader, "_resolve_attn_implementation", Mock(return_value=None))
    monkeypatch.setattr(
        loader,
        "_build_load_quantization_config",
        Mock(return_value=None),
    )

    verified_config = Mock(
        num_labels=2,
        problem_type="token_classification",
        architectures=["BertForTokenClassification"],
    )
    verified_model = Mock(config=verified_config)
    verified_tokenizer = Mock()
    auto_config = Mock()
    auto_config.from_pretrained.return_value = verified_config
    auto_tokenizer = Mock()
    auto_model = Mock()
    auto_model.from_pretrained.return_value = verified_model
    tokenizer_loader = Mock(return_value=verified_tokenizer)
    monkeypatch.setattr("openmed.core.models.AutoConfig", auto_config)
    monkeypatch.setattr("openmed.core.models.AutoTokenizer", auto_tokenizer)
    monkeypatch.setattr(
        "openmed.core.models.AutoModelForTokenClassification", auto_model
    )
    monkeypatch.setattr("openmed.core.models._ensure_hf_auto_config", Mock())
    monkeypatch.setattr("openmed.core.models._ensure_hf_auto_tokenizer", Mock())
    monkeypatch.setattr("openmed.core.models._ensure_hf_auto_model", Mock())
    monkeypatch.setattr(
        "openmed.core.models.get_tokenizer_with_loader",
        tokenizer_loader,
    )

    result = loader.load_model(
        BUNDLED_MODEL_MANIFEST.model_key,
        local_files_only=True,
        require_integrity=True,
    )

    assert result["model"] is verified_model
    assert result["tokenizer"] is verified_tokenizer
    assert result["model"] is not cached_model
    assert result["tokenizer"] is not cached_tokenizer
    assert local_loading.call_count == 2
    assert tokenizer_loader.call_args.kwargs["refresh_cache"] is True


def test_load_rejects_an_explicit_network_fallback() -> None:
    loader = _RecordingLoader()

    with pytest.raises(BundledModelError, match="network fallback is disabled"):
        load_bundled_model(loader=loader, local_files_only=False)

    assert loader.calls == []

    with pytest.raises(BundledModelError, match="verified artifact integrity"):
        load_bundled_model(loader=loader, require_integrity=False)

    assert loader.calls == []


def test_socket_egress_is_blocked_for_the_entire_load() -> None:
    with pytest.raises(OfflineModeError, match="network access"):
        load_bundled_model(loader=_NetworkProbeLoader())


def test_loader_failures_are_content_free() -> None:
    with pytest.raises(BundledModelUnavailableError) as raised:
        load_bundled_model(loader=_FailingLoader())

    assert "synthetic-sensitive-cache-location" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_manifest_rejects_string_subclasses_without_echoing_values() -> None:
    class SensitiveString(str):
        pass

    with pytest.raises(ValueError) as raised:
        BundledModelManifest(
            schema_version=SensitiveString("synthetic-sensitive-schema"),
            version="1.0.0",
            model_key="pii_detection",
            model_id=BUNDLED_MODEL_MANIFEST.model_id,
            checksum=BUNDLED_MODEL_MANIFEST.checksum,
            license="apache-2.0",
        )

    assert "synthetic-sensitive-schema" not in str(raised.value)
