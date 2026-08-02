"""Tests for intentional public API deprecation metadata and warnings."""

from __future__ import annotations

import pytest

from openmed.utils import deprecated


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


@pytest.mark.parametrize("keyword", ["since", "remove_in"])
def test_deprecated_rejects_empty_version(keyword):
    arguments = {"since": "1.9", "remove_in": "2.0"}
    arguments[keyword] = " "

    with pytest.raises(ValueError, match=keyword):
        deprecated(**arguments)
