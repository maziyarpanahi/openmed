from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

from openmed.interop import adapter_spec, available_adapters, get_adapter, ray_data
from openmed.interop.ray_data import DeidentifyBatch, deidentify_dataset

ROOT = Path(__file__).resolve().parents[3]


def fake_deidentifier(text: str, **kwargs):
    assert kwargs["policy"] == "hipaa_safe_harbor"
    redacted = text.replace("Jane Roe", "[PERSON]").replace("555-0100", "[PHONE]")
    return SimpleNamespace(deidentified_text=redacted)


def test_batch_callable_redacts_plain_dict_of_lists_without_ray():
    batch = {
        "note": ["Patient Jane Roe, call 555-0100", "no PII here", None],
        "encounter_id": [1, 2, 3],
    }
    operator = DeidentifyBatch(
        column="note",
        policy="hipaa_safe_harbor",
        deidentifier=fake_deidentifier,
    )

    result = operator(batch)

    assert result == {
        "note": ["Patient [PERSON], call [PHONE]", "no PII here", None],
        "encounter_id": [1, 2, 3],
    }
    assert batch["note"][0] == "Patient Jane Roe, call 555-0100"


def test_batch_callable_preserves_numpy_column_format():
    operator = DeidentifyBatch(column="note", deidentifier=fake_deidentifier)
    batch = {
        "note": np.asarray(["Jane Roe", "555-0100"]),
        "encounter_id": np.asarray([1, 2]),
    }

    result = operator(batch)

    assert isinstance(result["note"], np.ndarray)
    assert result["note"].tolist() == ["[PERSON]", "[PHONE]"]
    assert result["encounter_id"] is batch["encounter_id"]


def test_default_loader_is_created_once_and_reused_across_batches(monkeypatch):
    loader = object()
    loader_creations = 0
    observed_loaders: list[object] = []

    def make_loader():
        nonlocal loader_creations
        loader_creations += 1
        return loader

    def deidentifier(text: str, **kwargs):
        observed_loaders.append(kwargs["loader"])
        return SimpleNamespace(deidentified_text="[REDACTED]")

    monkeypatch.setattr(ray_data, "_default_model_loader", make_loader)
    monkeypatch.setattr(ray_data, "_default_deidentifier", lambda: deidentifier)

    operator = DeidentifyBatch(column="note")
    operator({"note": ["Jane Roe"]})
    operator({"note": ["555-0100"]})

    assert loader_creations == 1
    assert observed_loaders == [loader, loader]


def test_batch_callable_error_does_not_expose_source_text():
    def malformed_deidentifier(text: str, **kwargs):
        return {"redacted": False}

    operator = DeidentifyBatch(
        column="note",
        deidentifier=malformed_deidentifier,
    )

    with pytest.raises(TypeError, match="deidentified_text") as exc_info:
        operator({"note": ["Jane Roe"]})

    assert "Jane Roe" not in str(exc_info.value)


def test_registry_loads_ray_adapter_without_importing_ray():
    ray_modules_before = {
        name for name in sys.modules if name == "ray" or name.startswith("ray.")
    }

    adapter = get_adapter("ray")

    assert adapter is ray_data
    assert "ray" in available_adapters()
    assert adapter_spec("ray").extra == "ray"
    assert hasattr(adapter, "DeidentifyBatch")
    assert {
        name for name in sys.modules if name == "ray" or name.startswith("ray.")
    } == ray_modules_before


def test_ray_extra_declares_data_dependency():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = _toml.load(handle)["project"]["optional-dependencies"]["ray"]

    assert any(requirement.startswith("ray[data]") for requirement in dependencies)


class _FakeDataset:
    def __init__(self) -> None:
        self.map_batches_call = None

    def map_batches(self, operator, **kwargs):
        self.map_batches_call = (operator, kwargs)
        return "transformed-dataset"


def test_deidentify_dataset_uses_actor_callable(monkeypatch):
    class FakeActorPoolStrategy:
        pass

    monkeypatch.setattr(
        ray_data,
        "_load_ray_data",
        lambda: SimpleNamespace(ActorPoolStrategy=FakeActorPoolStrategy),
    )
    dataset = _FakeDataset()

    result = deidentify_dataset(
        dataset,
        "note",
        "hipaa_safe_harbor",
        method="mask",
    )

    assert result == "transformed-dataset"
    operator, kwargs = dataset.map_batches_call
    assert operator is DeidentifyBatch
    assert kwargs["batch_format"] == "numpy"
    assert isinstance(kwargs["compute"], FakeActorPoolStrategy)
    assert kwargs["fn_constructor_kwargs"] == {
        "column": "note",
        "policy": "hipaa_safe_harbor",
        "method": "mask",
    }
    assert kwargs["zero_copy_batch"] is True


def test_deidentify_dataset_reports_missing_ray_extra(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(ray_data, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[ray\]"):
        deidentify_dataset(_FakeDataset(), "note")
