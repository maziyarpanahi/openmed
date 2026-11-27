"""The shared error-code vectors agree with the registry and run in Python.

``tests/fixtures/parity/error_codes_v1.json`` lists every portable code from
the error-code registry and the offline scenarios each runtime must trigger.
Python runs every scenario marked ``required`` for it with the fixture's
synthetic inputs and checks the code only: messages are not part of the
contract. The coded errors keep their builtin bases and messages, so
existing ``except`` clauses still catch them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from openmed.core.error_registry import load_error_registry
from openmed.core.errors import (
    ERROR_CODES,
    ConfigurationError,
    InputError,
    ModelLoadError,
    OpenMedError,
    PolicyError,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "parity" / "error_codes_v1.json"
RUNTIMES = ("python", "swift", "kotlin", "web")
STATUSES = ("required", "pending", "not_applicable")
CELL_FIELDS = frozenset({"status", "entry", "today", "reason", "builtin"})
BUILTINS: Mapping[str, type[BaseException]] = {
    "ValueError": ValueError,
    "TypeError": TypeError,
    "RuntimeError": RuntimeError,
}


@pytest.fixture(scope="module")
def vectors() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry():
    return load_error_registry()


def _scenario(vectors: Mapping[str, Any], scenario_id: str) -> Mapping[str, Any]:
    return next(item for item in vectors["scenarios"] if item["id"] == scenario_id)


# ---------------------------------------------------------------------------
# Python triggers, one per scenario id
# ---------------------------------------------------------------------------


def _empty_hash_secret(inputs: Mapping[str, Any]) -> None:
    from openmed.core.schemas.span import hmac_text_hash

    hmac_text_hash("synthetic surface", inputs["secret"])


def _empty_pipeline_secret(inputs: Mapping[str, Any]) -> None:
    from openmed.core.pipeline import Pipeline

    Pipeline(hmac_secret=inputs["secret"])


def _invalid_offsets(inputs: Mapping[str, Any]) -> None:
    from openmed.core.schemas import OpenMedSpan

    OpenMedSpan(
        doc_id="synthetic-doc",
        start=inputs["start"],
        end=inputs["end"],
        text_hash="hmac-sha256:" + "0" * 64,
        canonical_label="NAME",
        entity_type="NAME",
    )


def _unknown_policy_profile(inputs: Mapping[str, Any]) -> None:
    from openmed.core.policy import canonical_policy_name

    canonical_policy_name(inputs["profile"])


def _model_missing_offline(inputs: Mapping[str, Any]) -> None:
    from openmed.core.config import OpenMedConfig
    from openmed.core.hf_hub import prefetch_model

    prefetch_model(
        inputs["model_id"],
        config=OpenMedConfig(local_only=inputs["offline"]),
    )


PYTHON_TRIGGERS: Mapping[str, tuple[Callable[[Mapping[str, Any]], None], ...]] = {
    "empty_hash_secret": (_empty_hash_secret, _empty_pipeline_secret),
    "invalid_offsets": (_invalid_offsets,),
    "unknown_policy_profile": (_unknown_policy_profile,),
    "model_missing_offline": (_model_missing_offline,),
}


class _LocalEntryNotFound(Exception):
    pass


@pytest.fixture
def offline_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make the Hub cache empty and forbid any network fetch."""
    from openmed.core import hf_hub

    def snapshot_download(**kwargs: Any) -> str:
        assert kwargs["local_files_only"] is True
        raise _LocalEntryNotFound("synthetic cache miss")

    monkeypatch.setattr(
        hf_hub,
        "_import_snapshot_download",
        lambda: (snapshot_download, _LocalEntryNotFound),
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Fixture shape
# ---------------------------------------------------------------------------


def test_fixture_shape_is_closed_and_synthetic(vectors) -> None:
    assert vectors["schema"] == "openmed.error_code_vectors"
    assert vectors["schema_version"] == 1
    assert vectors["synthetic"] is True
    assert tuple(vectors["runtimes"]) == RUNTIMES
    assert tuple(vectors["statuses"]) == STATUSES
    ids = [scenario["id"] for scenario in vectors["scenarios"]]
    assert len(ids) == len(set(ids))
    assert {
        "empty_hash_secret",
        "invalid_offsets",
        "unknown_policy_profile",
        "model_missing_offline",
        "invalid_sdk_config",
    } <= set(ids)
    for scenario in vectors["scenarios"]:
        assert set(scenario) == {"id", "summary", "code", "inputs", "runtimes"}
        assert tuple(scenario["runtimes"]) == RUNTIMES, scenario["id"]
        assert "message" not in json.dumps(scenario["inputs"])
        for runtime, cell in scenario["runtimes"].items():
            assert set(cell) <= CELL_FIELDS, (scenario["id"], runtime)
            assert cell["status"] in STATUSES, (scenario["id"], runtime)
            if cell["status"] == "required":
                assert "reason" not in cell, (scenario["id"], runtime)
            else:
                assert cell["reason"].strip(), (scenario["id"], runtime)
            if "builtin" in cell:
                assert runtime == "python"
                assert cell["builtin"] in BUILTINS


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def test_fixture_inputs_are_synthetic_and_value_free(vectors) -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    assert "/Users/" not in text and "C:\\" not in text
    for scenario in vectors["scenarios"]:
        for value in _strings(scenario["inputs"]):
            assert value == "" or "synthetic" in value, scenario["id"]


# ---------------------------------------------------------------------------
# Agreement with the registry
# ---------------------------------------------------------------------------


def test_fixture_codes_are_the_registry_portable_codes(vectors, registry) -> None:
    assert vectors["registry"]["contract"] == "openmed.errors.code-registry"
    assert vectors["registry"]["contract_version"] == registry.contract_version
    assert (ROOT / vectors["registry"]["path"]).is_file()
    fixture_codes = {item["code"]: item for item in vectors["codes"]}
    assert list(fixture_codes) == sorted(fixture_codes)
    assert set(fixture_codes) == {entry.code for entry in registry.portable_codes()}
    for code, item in fixture_codes.items():
        entry = registry.entry(code)
        assert item["category"] == entry.category, code
        assert item["http_status"] == entry.http_status, code
        assert item["cli_exit_code"] == entry.cli_exit_code, code
        assert item["retryable"] is entry.retryable, code


def test_scenario_codes_are_portable_and_cross_referenced(vectors, registry) -> None:
    portable = {entry.code for entry in registry.portable_codes()}
    listed = {item["code"]: set(item["scenarios"]) for item in vectors["codes"]}
    for scenario in vectors["scenarios"]:
        assert scenario["code"] in portable, scenario["id"]
        assert scenario["id"] in listed[scenario["code"]], scenario["id"]
    for code, scenario_ids in listed.items():
        for scenario_id in scenario_ids:
            assert _scenario(vectors, scenario_id)["code"] == code


def test_python_pending_cells_name_their_owner(vectors) -> None:
    sdk = _scenario(vectors, "invalid_sdk_config")["runtimes"]["python"]
    assert sdk["status"] == "pending"
    assert "#5342" in sdk["reason"]


def test_every_required_python_scenario_has_a_trigger(vectors) -> None:
    required = {
        scenario["id"]
        for scenario in vectors["scenarios"]
        if scenario["runtimes"]["python"]["status"] == "required"
    }
    assert required == set(PYTHON_TRIGGERS)


# ---------------------------------------------------------------------------
# Python runs its scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_id,trigger",
    [
        (scenario_id, trigger)
        for scenario_id, triggers in PYTHON_TRIGGERS.items()
        for trigger in triggers
    ],
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
def test_python_scenarios_raise_the_expected_code(
    vectors, registry, offline_hub, scenario_id, trigger
) -> None:
    scenario = _scenario(vectors, scenario_id)
    cell = scenario["runtimes"]["python"]
    with pytest.raises(OpenMedError) as excinfo:
        trigger(scenario["inputs"])
    error = excinfo.value
    assert error.code == scenario["code"]
    assert isinstance(error, BUILTINS[cell["builtin"]])
    entry = registry.entry(error.code)
    assert any(isinstance(error, _import_class(path)) for path in entry.python), (
        scenario_id
    )
    assert error.to_dict()["code"] == scenario["code"]


def _import_class(path: str) -> type[BaseException]:
    import importlib

    module_name, _, attribute = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


# ---------------------------------------------------------------------------
# Compatibility of the newly coded errors
# ---------------------------------------------------------------------------


def test_empty_hash_secret_keeps_value_error_and_message() -> None:
    from openmed.core.schemas.span import hmac_text_hash

    with pytest.raises(ValueError, match="^hmac_secret must be non-empty$") as excinfo:
        hmac_text_hash("synthetic surface", b"")
    assert isinstance(excinfo.value, ConfigurationError)
    assert ERROR_CODES["ConfigurationError"] == excinfo.value.code
    with pytest.raises(TypeError, match="explicit key"):
        hmac_text_hash("synthetic surface", None)  # type: ignore[arg-type]


def test_invalid_span_offsets_keep_value_error_and_message() -> None:
    from openmed.core.schemas import OpenMedSpan

    with pytest.raises(ValueError, match="non-negative offsets") as excinfo:
        _invalid_offsets({"start": -1, "end": 2})
    assert isinstance(excinfo.value, InputError)
    assert str(excinfo.value) == (
        "start/end must be non-negative offsets with end >= start"
    )
    assert excinfo.value.details == {"start": -1, "end": 2}
    with pytest.raises(TypeError, match="integers") as type_error:
        OpenMedSpan(
            doc_id="synthetic-doc",
            start="0",  # type: ignore[arg-type]
            end=1,
            text_hash="hmac-sha256:" + "0" * 64,
            canonical_label="NAME",
            entity_type="NAME",
        )
    assert not isinstance(type_error.value, OpenMedError)


def test_missing_offline_model_keeps_offline_mode_error_and_message(
    offline_hub,
) -> None:
    from openmed.core.offline import OfflineModeError, OfflineModelUnavailableError

    with pytest.raises(OfflineModeError) as excinfo:
        _model_missing_offline(
            {"model_id": "OpenMed/synthetic-missing-model", "offline": True}
        )
    error = excinfo.value
    assert isinstance(error, OfflineModelUnavailableError)
    assert isinstance(error, (RuntimeError, ModelLoadError))
    assert str(error).startswith(
        "OpenMed/synthetic-missing-model is not available in the local cache "
        "and offline mode blocks the download."
    )
    assert error.details == {"model_name": "OpenMed/synthetic-missing-model"}
    assert isinstance(error.__cause__, _LocalEntryNotFound)


def test_offline_network_guard_stays_an_uncoded_offline_error() -> None:
    from openmed.core.offline import OfflineModeError, raise_offline_error

    with pytest.raises(OfflineModeError) as excinfo:
        raise_offline_error("synthetic action")
    assert not isinstance(excinfo.value, OpenMedError)


def test_unknown_policy_profile_was_already_coded() -> None:
    with pytest.raises(PolicyError) as excinfo:
        _unknown_policy_profile({"profile": "synthetic_unknown_profile"})
    assert isinstance(excinfo.value, ValueError)
    assert "synthetic_unknown_profile" not in json.dumps(excinfo.value.to_dict())


def test_errors_doc_lists_every_scenario_with_its_code(vectors) -> None:
    text = (ROOT / "docs" / "api" / "errors.md").read_text(encoding="utf-8")
    assert "tests/fixtures/parity/error_codes_v1.json" in text
    for scenario in vectors["scenarios"]:
        assert f"| `{scenario['id']}` | `{scenario['code']}` |" in text
