"""Opt-in smoke tests for user-supplied Indic NER checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openmed.ner.families.indic import (
    IndicNerCheckpointUnavailable,
    load_indic_ner_adapter,
)

_LOCAL_CHECKPOINT_ENV = "OPENMED_INDIC_NER_COMPAT_LOCAL_MODEL"
_REMOTE_CHECKPOINT_ENV = "OPENMED_INDIC_NER_COMPAT_REMOTE_MODEL"
_SYNTHETIC_INPUT = "व्यक्ति नगर संस्था"

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _configured_checkpoint(environment_variable: str) -> str:
    source = os.environ.get(environment_variable, "").strip()
    if not source:
        pytest.skip(f"set {environment_variable} to an explicitly approved checkpoint")
    pytest.importorskip(
        "transformers",
        reason="install openmed[hf] to run Indic NER compatibility tests",
    )
    pytest.importorskip(
        "torch",
        reason="install torch to run Indic NER compatibility tests",
    )
    return source


def _exercise_checkpoint(source: str, *, local_files_only: bool) -> None:
    try:
        adapter = load_indic_ner_adapter(
            source,
            local_files_only=local_files_only,
        )
    except IndicNerCheckpointUnavailable as exc:
        pytest.skip(f"configured checkpoint is unavailable: {exc}")

    predictions = adapter.predict(_SYNTHETIC_INPUT, max_length=64)

    assert all(
        0 <= prediction.start < prediction.end <= len(_SYNTHETIC_INPUT)
        for prediction in predictions
    )
    assert all(
        set(prediction.to_dict()) == {"confidence", "end", "label", "start"}
        for prediction in predictions
    )


def test_user_supplied_local_checkpoint_contract() -> None:
    source = _configured_checkpoint(_LOCAL_CHECKPOINT_ENV)
    checkpoint_path = Path(source).expanduser()
    if not checkpoint_path.is_dir():
        pytest.skip(
            f"{_LOCAL_CHECKPOINT_ENV} must name an accessible checkpoint directory"
        )

    _exercise_checkpoint(str(checkpoint_path), local_files_only=True)


def test_explicit_remote_checkpoint_contract() -> None:
    source = _configured_checkpoint(_REMOTE_CHECKPOINT_ENV)
    if Path(source).expanduser().exists():
        pytest.skip(f"{_REMOTE_CHECKPOINT_ENV} must name a model repository")

    _exercise_checkpoint(source, local_files_only=False)
