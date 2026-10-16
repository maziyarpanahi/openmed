"""Synthetic runtime probes test policy, not actual GPU kernels."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openmed.torch import attention


@pytest.fixture(autouse=True)
def reset_logs():
    original = set(attention._LOGGED_BACKENDS)
    attention._LOGGED_BACKENDS.clear()
    yield
    attention._LOGGED_BACKENDS.clear()
    attention._LOGGED_BACKENDS.update(original)


def runtime(monkeypatch, probe):
    monkeypatch.setattr(attention.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        attention,
        "_import_torch",
        lambda: SimpleNamespace(cuda=SimpleNamespace(is_available=probe)),
    )


@pytest.mark.parametrize("error", [RuntimeError, OSError, ValueError])
@pytest.mark.parametrize("prefer", ["flash_attention_2", "flash"])
def test_runtime_probe_failure_downgrades(monkeypatch, error, prefer):
    def probe():
        raise error("SYNTHETIC-NATIVE-DETAIL")

    runtime(monkeypatch, probe)
    log = Mock()
    assert attention.select_attn_implementation(prefer, log=log) == "eager"
    assert log.warning.call_count == 1
    assert "SYNTHETIC-NATIVE-DETAIL" not in str(log.method_calls)


@pytest.mark.parametrize(
    "available,expected", [(True, "flash_attention_2"), (False, "eager")]
)
def test_successful_probe_keeps_selection(monkeypatch, available, expected):
    runtime(monkeypatch, lambda: available)
    assert attention.select_attn_implementation("flash_attention_2") == expected


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_are_not_swallowed(monkeypatch, error):
    def probe():
        raise error()

    runtime(monkeypatch, probe)
    with pytest.raises(error):
        attention.select_attn_implementation("flash_attention_2")


def test_no_probe_and_auto_controls(monkeypatch):
    runtime(monkeypatch, None)
    assert attention.select_attn_implementation("flash_attention_2") == "eager"
    assert attention.select_attn_implementation("auto") is None
    assert attention.select_attn_implementation("eager") == "eager"
