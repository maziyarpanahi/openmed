"""Public API docstring coverage tests."""

import inspect

import openmed


def test_public_api_callable_exports_have_docstrings() -> None:
    """Every public function/class export resolves and has a docstring."""
    unresolved = []
    missing_docstrings = []

    for name in openmed.__all__:
        try:
            obj = getattr(openmed, name)
        except AttributeError:
            unresolved.append(name)
            continue

        if not inspect.isfunction(obj) and not inspect.isclass(obj):
            continue

        if not getattr(obj, "__doc__", None):
            missing_docstrings.append(name)

    assert unresolved == []
    assert missing_docstrings == []
