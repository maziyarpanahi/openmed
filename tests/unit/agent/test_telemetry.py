"""Offline negative controls for the agent telemetry boundary."""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

import pytest

from openmed.agent import telemetry as module
from openmed.agent.action_phases import ActionPhaseError
from openmed.agent.event_attributes import EventAttributeError, EventAttributes
from openmed.agent.telemetry import AgentTelemetry


class Sink:
    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.metrics: list[tuple[str, Any, Any]] = []

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        self.spans.append({"name": name, **kwargs})
        yield self

    def set_attribute(self, key: str, value: Any) -> None:
        self.spans[-1]["attributes"][key] = value

    def create_counter(self, name: str, **kwargs: Any) -> Any:
        return self

    def create_histogram(self, name: str, **kwargs: Any) -> Any:
        return self

    def add(self, value: int, *, attributes: Any) -> None:
        self.metrics.append(("counter", value, attributes))

    def record(self, value: float, *, attributes: Any) -> None:
        self.metrics.append(("duration", value, attributes))


def test_disabled_ignores_injected_sinks_and_clock(monkeypatch: Any) -> None:
    def unexpected(*args: Any) -> Any:
        pytest.fail("disabled telemetry touched optional runtime")

    monkeypatch.setattr(module, "import_module", unexpected)
    monkeypatch.setattr(module, "find_spec", unexpected)
    sink = Sink()
    telemetry = AgentTelemetry(tracer=sink, meter=sink, clock=unexpected)
    with telemetry.event_span("running", EventAttributes.from_mapping({})) as span:
        assert span is None
    assert not telemetry.enabled
    assert sink.spans == sink.metrics == []


@pytest.mark.parametrize("enabled,absent", [(False, False), (True, True)])
def test_fresh_process_never_imports_otel_when_disabled_or_absent(
    enabled: bool, absent: bool
) -> None:
    script = f"""
import builtins
import importlib.abc
import importlib.util
import sys

class RejectOTel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('opentelemetry'):
            raise AssertionError('OpenTelemetry import attempted')

sys.meta_path.insert(0, RejectOTel())
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('opentelemetry'):
        raise AssertionError('OpenTelemetry import attempted')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
if {absent!r}:
    real_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name: (
        None if name == 'opentelemetry' else real_find_spec(name)
    )
from openmed.agent.telemetry import AgentTelemetry
from openmed.agent.event_attributes import EventAttributes
runtime = AgentTelemetry(enabled={enabled!r})
with runtime.event_span('running', EventAttributes.from_mapping({{}})):
    pass
assert not runtime.enabled
assert not any(name.startswith('opentelemetry') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_partial_otel_installation_degrades_to_noop(monkeypatch: Any) -> None:
    def missing(name: str) -> Any:
        raise ModuleNotFoundError("optional API missing")

    monkeypatch.setattr(module, "find_spec", lambda name: object())
    monkeypatch.setattr(module, "import_module", missing)
    telemetry = AgentTelemetry(enabled=True)
    with telemetry.event_span("running", EventAttributes.from_mapping({})):
        pass
    assert not telemetry.enabled


def test_opt_in_uses_only_global_api_sinks(monkeypatch: Any) -> None:
    from types import SimpleNamespace

    names: list[str] = []
    sink = Sink()

    def get_sink(name: str) -> Sink:
        names.append(name)
        return sink

    api = SimpleNamespace(get_tracer=get_sink, get_meter=get_sink)
    monkeypatch.setattr(module, "find_spec", lambda name: object())
    monkeypatch.setattr(module, "import_module", lambda name: api)
    telemetry = AgentTelemetry(enabled=True)
    with telemetry.event_span(
        "completed", EventAttributes.from_mapping({"duration_ms": 5})
    ):
        pass
    assert names == ["openmed.agent", "openmed.agent"]
    assert len(sink.spans) == 1


@pytest.mark.parametrize("key", ["arguments", "output", "patient_name", "file_path"])
def test_tampered_attributes_fail_before_sink_changes(key: str) -> None:
    attributes = EventAttributes.from_mapping({})
    object.__setattr__(
        attributes, "values", MappingProxyType({key: "Synthetic Juniper MRN JS-1188"})
    )
    sink = Sink()
    telemetry = AgentTelemetry(enabled=True, tracer=sink, meter=sink)
    with pytest.raises(EventAttributeError) as error:
        with telemetry.event_span("running", attributes):
            pytest.fail("invalid event entered")
    assert "Juniper" not in str(error.value)
    assert sink.spans == sink.metrics == []


@pytest.mark.parametrize("phase", ["Juniper MRN JS-1188", None, [], 1])
def test_unknown_phase_is_value_free_and_atomic(phase: Any) -> None:
    sink = Sink()
    telemetry = AgentTelemetry(enabled=True, tracer=sink, meter=sink)
    with pytest.raises(ActionPhaseError, match="unknown_phase") as error:
        with telemetry.event_span(phase, EventAttributes.from_mapping({})):
            pytest.fail("invalid phase entered")
    assert "Juniper" not in str(error.value)
    assert error.value.__context__ is None
    assert sink.spans == sink.metrics == []


def test_mapping_is_not_an_event_contract() -> None:
    with pytest.raises(EventAttributeError, match="invalid_attribute_type"):
        with AgentTelemetry().event_span("running", {"duration_ms": 2}):
            pass


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"enabled": "true"}, TypeError),
        ({"run_id_mode": "raw"}, ValueError),
        ({"run_id_mode": []}, ValueError),
        ({"clock": None}, TypeError),
    ],
)
def test_invalid_configuration_fails_closed(kwargs: Any, error: Any) -> None:
    with pytest.raises(error):
        AgentTelemetry(**kwargs)


def test_measured_duration_uses_injected_clock_without_import(monkeypatch: Any) -> None:
    def unexpected(*args: Any) -> Any:
        pytest.fail("injected sinks must not import OpenTelemetry")

    monkeypatch.setattr(module, "import_module", unexpected)
    monkeypatch.setattr(module, "find_spec", unexpected)
    sink = Sink()
    times = iter([10.0, 10.125])
    telemetry = AgentTelemetry(
        enabled=True, tracer=sink, meter=sink, clock=lambda: next(times)
    )
    with telemetry.event_span("running", EventAttributes.from_mapping({})):
        pass
    assert sink.spans[0]["attributes"] == {"openmed.agent.duration_ms": 125.0}
    assert sink.metrics == [
        ("counter", 1, {"openmed.agent.phase": "running"}),
        ("duration", 125.0, {"openmed.agent.phase": "running"}),
    ]


@pytest.mark.parametrize("end", [0.0, float("nan"), float("inf"), 1e20])
def test_invalid_clock_duration_never_reaches_sink(end: float) -> None:
    sink = Sink()
    times = iter([1.0, end])
    telemetry = AgentTelemetry(
        enabled=True, tracer=sink, meter=sink, clock=lambda: next(times)
    )
    with pytest.raises(EventAttributeError):
        with telemetry.event_span("running", EventAttributes.from_mapping({})):
            pass
    assert "openmed.agent.duration_ms" not in sink.spans[0]["attributes"]
    assert len(sink.metrics) == 1
