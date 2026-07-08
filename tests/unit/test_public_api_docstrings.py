"""Public API docstring coverage tests."""

import inspect

import openmed


def test_public_api_exports_have_docstrings() -> None:
    """Every top-level public export resolves and has a docstring."""
    unresolved = []
    missing_docstrings = []

    for name in openmed.__all__:
        try:
            obj = getattr(openmed, name)
        except AttributeError:
            unresolved.append(name)
            continue

        if not inspect.getdoc(obj):
            missing_docstrings.append(name)

    assert unresolved == []
    assert missing_docstrings == []
