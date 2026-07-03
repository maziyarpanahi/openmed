"""Tests for OPENMED_OFFLINE/local-only inference mode."""

from __future__ import annotations

import os
import socket
from unittest.mock import Mock, patch

import pytest

from openmed.core.config import OpenMedConfig
from openmed.core.offline import (
    HF_OFFLINE_ENV_VARS,
    OFFLINE_ENV_VAR,
    OfflineModeError,
    network_blocked_if_offline,
)


def _clear_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OFFLINE_ENV_VAR, raising=False)
    for name in HF_OFFLINE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_openmed_offline_env_sets_local_only_and_hf_flags(monkeypatch):
    _clear_offline_env(monkeypatch)
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")

    config = OpenMedConfig()

    assert config.local_only is True
    for name in HF_OFFLINE_ENV_VARS:
        assert os.environ[name] == "1"


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.get_all_models")
def test_local_only_model_listing_uses_manifest(mock_get_all_models, monkeypatch):
    _clear_offline_env(monkeypatch)
    mock_get_all_models.return_value = {
        "pii": Mock(model_id="OpenMed/local-pii"),
    }

    from openmed.core.models import ModelLoader

    loader = ModelLoader(OpenMedConfig(local_only=True))
    assert loader.list_available_models() == ["OpenMed/local-pii"]
    mock_get_all_models.assert_called_once()


@patch("openmed.core.models.HF_AVAILABLE", True)
@patch("openmed.core.models.pipeline")
def test_local_only_hf_pipeline_uses_cached_files(mock_pipeline, monkeypatch):
    _clear_offline_env(monkeypatch)

    from openmed.core.models import ModelLoader

    loader = ModelLoader(OpenMedConfig(local_only=True, backend="hf"))
    loader.create_pipeline("OpenMed/local-pii")

    assert mock_pipeline.call_args.kwargs["local_files_only"] is True


def test_disallowed_socket_connection_raises_clear_offline_error(monkeypatch):
    _clear_offline_env(monkeypatch)
    config = OpenMedConfig(local_only=True)

    with network_blocked_if_offline(config):
        with pytest.raises(OfflineModeError, match="OPENMED_OFFLINE/local_only=True"):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)


class _FakeLocalPiiPipeline:
    tokenizer = None

    @staticmethod
    def _entities(text: str):
        return [
            {
                "entity_group": "NAME",
                "score": 0.99,
                "word": "John Doe",
                "start": text.index("John Doe"),
                "end": text.index("John Doe") + len("John Doe"),
            },
            {
                "entity_group": "PHONE",
                "score": 0.98,
                "word": "555-1234",
                "start": text.index("555-1234"),
                "end": text.index("555-1234") + len("555-1234"),
            },
        ]

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return [self._entities(item) for item in text]
        return self._entities(text)


class _FakeLocalLoader:
    def __init__(self, config: OpenMedConfig):
        self.config = config
        self.pipeline = _FakeLocalPiiPipeline()

    def create_pipeline(self, *args, **kwargs):
        return self.pipeline

    def get_max_sequence_length(self, *args, **kwargs):
        return None


def test_pii_deidentification_path_runs_with_sockets_blocked(monkeypatch):
    _clear_offline_env(monkeypatch)
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")
    blocked_attempts = []

    def fail_socket(*args, **kwargs):
        blocked_attempts.append((args, kwargs))
        raise AssertionError("network egress attempted")

    monkeypatch.setattr(socket.socket, "connect", fail_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)

    from openmed.core.pii import deidentify, extract_pii

    text = "Patient John Doe called 555-1234."
    config = OpenMedConfig(use_medical_tokenizer=False)
    loader = _FakeLocalLoader(config)

    pii_result = extract_pii(
        text,
        model_name="local-pii",
        config=config,
        loader=loader,
        use_smart_merging=False,
    )

    assert [(e.label, e.start, e.end, e.text) for e in pii_result.entities] == [
        ("NAME", 8, 16, "John Doe"),
        ("PHONE", 24, 32, "555-1234"),
    ]

    deid_result = deidentify(
        text,
        model_name="local-pii",
        config=config,
        loader=loader,
        use_smart_merging=False,
    )

    assert deid_result.deidentified_text == "Patient [NAME] called [PHONE]."
    assert blocked_attempts == []
