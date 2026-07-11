"""Regression tests for PEP 561 typing support.

These guard three things that keep ``openmed`` a typed package for downstream
consumers:

* the ``py.typed`` marker actually sits next to ``openmed/__init__.py`` (PEP 561
  requires the marker to live inside the installed package), and
* the hatch build ``include`` list still declares ``openmed/py.typed`` so the
  marker ships in the built wheel and sdist, and
* the newer v1.6 public surfaces expose resolvable type hints and type-check
  cleanly with the pinned development checker over just those modules.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import typing
from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

import openmed

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(openmed.__file__).resolve().parent
PY_TYPED_INCLUDE = "openmed/py.typed"

# The newer v1.6 public modules whose annotations this task hardened. Keeping
# the list here means the type-check step and the "annotations resolve" spot
# check stay in lockstep with the packaging guarantee.
TYPED_MODULES = (
    "openmed.core.audit",
    "openmed.core.pipeline",
    "openmed.clinical.context",
    "openmed.clinical.exporters.fhir.bundle",
)


def _load_pyproject() -> dict:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        return _toml.load(handle)


def test_py_typed_marker_present_in_package() -> None:
    """The PEP 561 marker must live inside the installed package tree."""
    marker = PACKAGE_ROOT / "py.typed"
    assert marker.is_file(), (
        "openmed/py.typed marker is missing; downstream type checkers will treat "
        "openmed as untyped without it"
    )


def test_py_typed_declared_in_hatch_build_include() -> None:
    """The build config must ship the marker in the wheel and sdist."""
    config = _load_pyproject()
    include = config["tool"]["hatch"]["build"]["include"]
    assert PY_TYPED_INCLUDE in include, (
        f"{PY_TYPED_INCLUDE!r} must be in [tool.hatch.build].include so the "
        "typing marker is packaged"
    )


def _public_annotation_targets(
    export_name: str, obj: object
) -> list[tuple[str, object]]:
    """Return an export and the public callables declared directly on it."""
    targets = [(export_name, obj)]
    if not inspect.isclass(obj):
        return targets

    for member_name, descriptor in vars(obj).items():
        if member_name.startswith("_"):
            continue
        if isinstance(descriptor, (classmethod, staticmethod)):
            descriptor = descriptor.__func__
        elif isinstance(descriptor, property):
            descriptor = descriptor.fget
        if inspect.isfunction(descriptor):
            targets.append((f"{export_name}.{member_name}", descriptor))
    return targets


@pytest.mark.parametrize("module_name", TYPED_MODULES)
def test_typed_modules_expose_resolvable_hints(module_name: str) -> None:
    """Each scoped v1.6 module imports and its public hints resolve.

    ``typing.get_type_hints`` forces evaluation of the string annotations these
    modules declare under ``from __future__ import annotations``; a broken or
    dangling annotation on a public class or function would raise here. Public
    aliases (e.g. ``Literal`` type aliases) are not callables carrying their own
    annotations and are intentionally skipped. Exported classes are also walked
    so annotations on their public methods and properties cannot hide behind a
    resolvable class-level annotation.
    """
    module = importlib.import_module(module_name)
    for name in getattr(module, "__all__", ()):
        obj = getattr(module, name)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        for qualname, target in _public_annotation_targets(name, obj):
            try:
                typing.get_type_hints(target)
            except (NameError, TypeError) as exc:  # pragma: no cover - failure path
                pytest.fail(
                    f"{module_name}.{qualname} has an unresolvable type hint: {exc}"
                )


def test_scoped_modules_type_check_cleanly() -> None:
    """Type-check the scoped v1.6 modules with the pinned mypy dependency."""

    module_paths = [
        str(PACKAGE_ROOT.parent / f"{name.replace('.', '/')}.py")
        for name in TYPED_MODULES
    ]

    cmd = [sys.executable, "-m", "mypy", *module_paths]

    result = subprocess.run(  # noqa: S603 - fixed, non-user-supplied argv
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "mypy reported type errors on the scoped v1.6 modules:\n"
        f"{result.stdout}\n{result.stderr}"
    )
