"""Tests for intentional public API deprecation metadata and warnings."""

from __future__ import annotations

import asyncio
import inspect
import sys
import warnings
from types import ModuleType

import pytest

from openmed.utils import deprecated
from openmed.utils.deprecation import (
    deprecated_attribute,
    deprecated_module,
    deprecated_parameter,
    warn_deprecated,
)


def test_deprecated_function_warns_and_preserves_metadata():
    @deprecated(since="1.9", remove_in="2.0", replacement="modern")
    def legacy(value: str) -> str:
        """Return the supplied value."""

        return value

    with pytest.warns(DeprecationWarning, match="Use modern instead"):
        assert legacy("value") == "value"

    assert legacy.__name__ == "legacy"
    assert legacy.__openmed_deprecated__ == {
        "since": "1.9",
        "remove_in": "2.0",
        "replacement": "modern",
    }


def test_deprecated_class_warns_without_replacing_class_identity():
    @deprecated(since="1.9", remove_in="2.0")
    class Legacy:
        """Small deprecated class fixture."""

        def __init__(self, value: str) -> None:
            self.value = value

    with pytest.warns(DeprecationWarning, match="Legacy is deprecated"):
        instance = Legacy("value")

    assert isinstance(instance, Legacy)
    assert instance.value == "value"


def test_deprecated_rejects_empty_version(subtests):
    for keyword in ["since", "remove_in"]:
        with subtests.test(keyword=keyword):
            arguments = {"since": "1.9", "remove_in": "2.0"}
            arguments[keyword] = " "

            with pytest.raises(ValueError, match=keyword):
                deprecated(**arguments)


@pytest.mark.parametrize("args, kwargs", [((None,), {}), ((), {"legacy": False})])
def test_deprecated_parameter_warns_only_when_supplied(args, kwargs):
    def query(legacy=True, *, mode="safe"):
        return legacy, mode

    wrapped = deprecated_parameter(
        "legacy", since="4.3", remove_in="5.0", replacement="mode"
    )(query)

    assert inspect.signature(wrapped) == inspect.signature(query)
    assert inspect.unwrap(wrapped) is query
    assert wrapped.__qualname__ == query.__qualname__
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert wrapped(mode="safe") == (True, "safe")
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            wrapped(False, unknown=True)
    with pytest.warns(
        DeprecationWarning, match=r"'legacy'.*4\.3.*5\.0.*Use mode instead"
    ) as caught:
        assert wrapped(*args, **kwargs) == query(*args, **kwargs)
    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_deprecated_parameter_preserves_async_methods():
    class Client:
        @deprecated_parameter("flag", since="4.3", remove_in="5.0")
        async def query(self, flag=False, /):
            return flag

    query = Client().query
    assert inspect.iscoroutinefunction(query)
    assert str(inspect.signature(query)) == "(flag=False, /)"
    with pytest.warns(DeprecationWarning, match="'flag'") as caught:
        assert asyncio.run(query(True)) is True
    assert len(caught) == 1


def test_deprecated_parameter_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown deprecated parameter 'missing'"):
        deprecated_parameter("missing", since="4.3", remove_in="5.0")(lambda: None)


def test_deprecated_attribute_forwards_reads_writes_and_deletion():
    class Record:
        __slots__ = ("current",)
        old = deprecated_attribute("current", since="4.3", remove_in="5.0")

    record = Record()
    value = object()
    match = r"Record\.old.*4\.3.*5\.0.*Use current instead"
    with pytest.warns(DeprecationWarning, match=match) as write:
        record.old = value
    assert record.current is value
    with pytest.warns(DeprecationWarning, match=match) as read:
        assert record.old is value
    with pytest.warns(DeprecationWarning, match=match) as deletion:
        del record.old
    assert len(write) == len(read) == len(deletion) == 1
    assert not hasattr(record, "current")


def test_deprecated_module_warns_and_returns_the_same_export(monkeypatch):
    target = ModuleType("example_current")
    target.export = object()
    monkeypatch.setitem(sys.modules, target.__name__, target)
    legacy = ModuleType("example_legacy")
    legacy.__getattr__ = deprecated_module(
        legacy.__name__, since="4.3", remove_in="5.0", replacement=target.__name__
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(AttributeError, match="example_legacy.*missing"):
            legacy.missing
        assert not hasattr(legacy, "__path__")
    with pytest.warns(
        DeprecationWarning, match=r"example_legacy.*4\.3.*5\.0.*example_current"
    ) as caught:
        assert legacy.export is target.export
    assert len(caught) == 1


def test_warn_deprecated_reports_a_stable_code():
    with pytest.warns(
        DeprecationWarning, match=r"^\[old_config\].*4\.3.*5\.0.*Use new_config instead"
    ) as caught:
        warn_deprecated("old_config", "4.3", "5.0", replacement="new_config")
    assert len(caught) == 1
    assert caught[0].filename == __file__
