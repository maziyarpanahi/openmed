"""Gate: code in ``openmed/`` must not read an undeclared ``OPENMED_*`` variable.

Every environment variable the package reads needs a declaration in
``openmed/core/settings.schema.json`` with its type, default and secret flag.
The gate has two layers, both static and offline:

* **Reads.** An AST scan resolves ``os.environ[...]``, ``os.environ.get``,
  ``os.getenv`` and the package's own environment helper wrappers to literal
  ``OPENMED_*`` names (see ``tests/fixtures/settings_env_scan.py``).  Each
  resolved name must be declared.  A key built at runtime with an f-string,
  ``+``, ``%`` or ``format`` cannot be checked and needs a reasoned entry in
  the ``dynamic`` allowlist.
* **Identifiers.** Every whole-string ``OPENMED_*`` literal outside a
  docstring must be declared too, which catches table-driven reads the
  resolver cannot follow.  Constants that only look like environment
  variables, such as FHIR system names or error codes, carry a reasoned
  allowlist entry.

The allowlist in ``tests/fixtures/settings_undeclared_allowlist.json`` seeds
the sections that are not declared yet.  It only shrinks: entries that are no
longer needed fail the gate, and :data:`ALLOWLIST_CEILING` pins its size.
Failure messages name variables and source locations only, never values.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Mapping
from pathlib import Path

import pytest

from openmed.core.settings_registry import declared_setting_names
from tests.fixtures.settings_env_scan import (
    PACKAGE_ROOT,
    REPO_ROOT,
    EnvScan,
    literal_env_names,
    scan_env_reads,
)

ALLOWLIST_PATH = REPO_ROOT / "tests" / "fixtures" / "settings_undeclared_allowlist.json"
# Committed size of the undeclared allowlist.  Lower it with every entry that
# is removed; it must never grow.
ALLOWLIST_CEILING = 140
SCHEMA = "openmed/core/settings.schema.json"
SENTINEL = "synthetic-undeclared-value-91d2"


def find_undeclared(
    scan: EnvScan,
    identifiers: Mapping[str, list[str]],
    *,
    declared: Collection[str],
    allowed: Collection[str] = (),
    dynamic_allowed: Collection[str] = (),
) -> list[str]:
    """Return one failure message per undeclared name or unverifiable read.

    Args:
        scan: Environment reads found by the AST scan.
        identifiers: Whole-string ``OPENMED_*`` literals and their locations.
        declared: Names declared in the settings schema.
        allowed: Names on the reasoned undeclared allowlist.
        dynamic_allowed: ``path::expression`` keys of allowed dynamic reads.

    Returns:
        Messages naming each variable and its locations; never a value.
    """

    known = set(declared) | set(allowed)
    problems: list[str] = []
    read_names = scan.names()
    for name in sorted(read_names - known):
        problems.append(
            f"{name} is read at {', '.join(scan.locations(name))} but is not "
            f"declared in {SCHEMA}"
        )
    for name in sorted(set(identifiers) - read_names - known):
        problems.append(
            f"{name} appears at {', '.join(identifiers[name])} but is not declared "
            f"in {SCHEMA}; declare it, or allowlist it with a reason if it is a "
            "constant rather than an environment variable"
        )
    for read in scan.dynamic:
        if dynamic_key(read.path, read.expression) not in dynamic_allowed:
            problems.append(
                f"{read.location} reads an OPENMED_* name built at runtime, which "
                "cannot be checked; add a reasoned 'dynamic' allowlist entry for "
                f"{dynamic_key(read.path, read.expression)!r}"
            )
    return problems


def dynamic_key(path: str, expression: str) -> str:
    """Return the allowlist key for one dynamic read: ``path::expression``."""

    return f"{path}::{expression}"


def _load_allowlist() -> dict[str, dict[str, str]]:
    return json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def package_scan() -> EnvScan:
    return scan_env_reads([PACKAGE_ROOT])


@pytest.fixture(scope="module")
def package_identifiers() -> dict[str, list[str]]:
    return literal_env_names([PACKAGE_ROOT])


# -- the gate on the real package ---------------------------------------------


def test_every_openmed_environment_read_is_declared(
    package_scan, package_identifiers
) -> None:
    allowlist = _load_allowlist()
    problems = find_undeclared(
        package_scan,
        package_identifiers,
        declared=declared_setting_names(),
        allowed=allowlist["undeclared"],
        dynamic_allowed=allowlist["dynamic"],
    )
    assert problems == [], "\n".join(problems)


def test_scan_sees_the_declared_sections(package_scan) -> None:
    # Guard against a scanner that silently resolves nothing.
    read = package_scan.names()
    assert "OPENMED_SERVICE_AUTH_API_KEYS" in read
    assert "OPENMED_SERVICE_AUTH_API_KEYS_FILE" in read
    assert "OPENMED_OFFLINE" in read
    assert len(read & declared_setting_names()) >= 150


def test_allowlist_only_shrinks() -> None:
    entries = _load_allowlist()["undeclared"]
    assert len(entries) <= ALLOWLIST_CEILING, (
        "the undeclared allowlist must not grow; declare the new setting in "
        f"{SCHEMA} instead"
    )
    assert len(entries) == ALLOWLIST_CEILING, (
        f"the allowlist shrank to {len(entries)} entries; lower "
        "ALLOWLIST_CEILING to match"
    )


def test_allowlist_entries_are_reasoned_and_still_needed(
    package_scan, package_identifiers
) -> None:
    allowlist = _load_allowlist()
    declared = declared_setting_names()
    present = package_scan.names() | set(package_identifiers)
    dynamic_present = {
        dynamic_key(read.path, read.expression) for read in package_scan.dynamic
    }

    assert set(allowlist) == {"description", "undeclared", "dynamic"}
    for name, reason in allowlist["undeclared"].items():
        assert isinstance(reason, str) and len(reason.strip()) >= 10, name
        assert name not in declared, f"{name} is declared; remove its allowlist entry"
        assert name in present, f"{name} no longer appears; remove its entry"
    for key, reason in allowlist["dynamic"].items():
        assert isinstance(reason, str) and len(reason.strip()) >= 10, key
        assert key in dynamic_present, f"{key} no longer matches a dynamic read"
    assert list(allowlist["undeclared"]) == sorted(allowlist["undeclared"])


# -- negative controls on synthetic modules -------------------------------------


def _synthetic_package(tmp_path: Path, source: str) -> Path:
    package = tmp_path / "openmed"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "synthetic_gate.py").write_text(source, encoding="utf-8")
    return package


def _check(tmp_path: Path, **kwargs) -> list[str]:
    package = tmp_path / "openmed"
    return find_undeclared(
        scan_env_reads([package], root=tmp_path),
        literal_env_names([package], root=tmp_path),
        declared=kwargs.pop("declared", declared_setting_names()),
        **kwargs,
    )


def test_synthetic_undeclared_read_fails_with_path_and_line(tmp_path) -> None:
    _synthetic_package(
        tmp_path,
        'import os\n\nFLAG = os.getenv("OPENMED_SYNTHETIC_UNDECLARED_FLAG")\n',
    )

    problems = _check(tmp_path)

    assert problems == [
        "OPENMED_SYNTHETIC_UNDECLARED_FLAG is read at openmed/synthetic_gate.py:3 "
        f"but is not declared in {SCHEMA}"
    ]


def test_failure_output_names_variables_and_locations_only(tmp_path) -> None:
    _synthetic_package(
        tmp_path,
        "import os\n"
        'TOKEN_ENV = "OPENMED_SYNTHETIC_UNDECLARED_TOKEN"\n'
        "def load(environ=os.environ):\n"
        "    return environ.get(TOKEN_ENV)\n",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OPENMED_SYNTHETIC_UNDECLARED_TOKEN", SENTINEL)
        problems = _check(tmp_path)

    rendered = "\n".join(problems)
    assert "OPENMED_SYNTHETIC_UNDECLARED_TOKEN" in rendered
    assert "openmed/synthetic_gate.py:4" in rendered
    assert SENTINEL not in rendered
    assert os.environ.get("OPENMED_SYNTHETIC_UNDECLARED_TOKEN") is None


def test_declared_and_allowlisted_reads_pass(tmp_path) -> None:
    _synthetic_package(
        tmp_path,
        "import os\n"
        'LEVEL = os.getenv("OPENMED_SERVICE_LOG_LEVEL")\n'
        'PENDING = os.environ.get("OPENMED_SYNTHETIC_PENDING")\n',
    )

    assert _check(tmp_path, allowed={"OPENMED_SYNTHETIC_PENDING": "pending"}) == []
    assert len(_check(tmp_path)) == 1


def test_dynamic_names_are_unverifiable_until_allowlisted(tmp_path) -> None:
    _synthetic_package(
        tmp_path,
        "import os\n"
        "def read(suffix):\n"
        '    return os.getenv(f"OPENMED_SYNTHETIC_{suffix}")\n',
    )

    problems = _check(tmp_path)
    assert len(problems) == 1
    assert "openmed/synthetic_gate.py:3" in problems[0]
    assert "cannot be checked" in problems[0]

    # Keys use the expression as ``ast.unparse`` renders it.
    key = dynamic_key("openmed/synthetic_gate.py", "f'OPENMED_SYNTHETIC_{suffix}'")
    assert _check(tmp_path, dynamic_allowed={key: "synthetic"}) == []


def test_dynamic_names_built_from_a_named_prefix_are_unverifiable(tmp_path) -> None:
    _synthetic_package(
        tmp_path,
        "import os\n"
        'PREFIX = "OPENMED_SYNTHETIC_"\n'
        'JOINED = os.environ.get(PREFIX + "JOINED")\n'
        'FORMATTED = os.getenv(f"{PREFIX}FORMATTED")\n',
    )

    problems = _check(tmp_path)
    assert len(problems) == 2
    assert "openmed/synthetic_gate.py:3" in problems[0]
    assert "openmed/synthetic_gate.py:4" in problems[1]
    assert all("cannot be checked" in problem for problem in problems)


def test_constant_identifiers_need_a_declaration_or_an_entry(tmp_path) -> None:
    _synthetic_package(
        tmp_path,
        'SYSTEM = "OPENMED_SYNTHETIC_SYSTEM_URL"\n__all__ = ["SYSTEM"]\n',
    )

    problems = _check(tmp_path)
    assert len(problems) == 1
    assert problems[0].startswith(
        "OPENMED_SYNTHETIC_SYSTEM_URL appears at openmed/synthetic_gate.py:1"
    )
    allowed = {"OPENMED_SYNTHETIC_SYSTEM_URL": "constant"}
    assert _check(tmp_path, allowed=allowed) == []
