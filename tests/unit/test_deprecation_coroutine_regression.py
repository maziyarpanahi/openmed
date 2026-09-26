"""Coroutine deprecation preserves async dispatch and callable behavior."""

import asyncio
import inspect
import warnings

import pytest

from openmed.utils.deprecation import deprecated


def _decorate(target):
    return deprecated(since="2.5", remove_in="3.0", replacement="new_api")(target)


def test_decorated_async_function_retains_identity_and_metadata():
    async def original(value: int = 3) -> int:
        """Synthetic coroutine."""
        return value

    wrapped = _decorate(original)
    assert inspect.iscoroutinefunction(wrapped)
    assert inspect.signature(wrapped) == inspect.signature(original)
    assert wrapped.__name__ == original.__name__
    assert wrapped.__doc__ == original.__doc__
    assert wrapped.__wrapped__ is original
    assert wrapped.__openmed_deprecated__["replacement"] == "new_api"


def test_async_warning_is_emitted_on_await_not_creation():
    @_decorate
    async def operation(value):
        return value + 1

    async def run():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coroutine = operation(4)
            before_await = len(caught)
            result = await coroutine
            assert before_await == 0
            assert result == 5
            assert len(caught) == 1
            assert caught[0].category is DeprecationWarning
            assert caught[0].filename == __file__

    asyncio.run(run())


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_async_exception_and_cancellation_propagate(error_type):
    @_decorate
    async def operation():
        raise error_type("synthetic")

    async def run():
        with pytest.warns(DeprecationWarning), pytest.raises(error_type):
            await operation()

    asyncio.run(run())


def test_sync_and_class_behavior_remain_unchanged():
    @_decorate
    def operation(value):
        return value + 1

    assert not inspect.iscoroutinefunction(operation)
    with pytest.warns(DeprecationWarning):
        assert operation(2) == 3

    class Original:
        def __init__(self, value):
            self.value = value

    wrapped = _decorate(Original)
    assert wrapped is Original
    with pytest.warns(DeprecationWarning):
        assert wrapped(7).value == 7
