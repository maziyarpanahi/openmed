"""Doctest coverage for public core examples."""

from __future__ import annotations

import doctest
import importlib

import pytest

_TARGET_MODULES = (
    "openmed",
    "openmed.core.pii",
)


@pytest.mark.doctest_examples
@pytest.mark.parametrize("module_name", _TARGET_MODULES)
def test_public_core_doctests(module_name: str) -> None:
    """Run doctests for the targeted public modules."""
    module = importlib.import_module(module_name)
    result = doctest.testmod(
        module,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0
