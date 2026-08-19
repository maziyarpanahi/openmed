"""Tests for the static release API-surface comparison and migration gate."""

from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "api_surface_diff.py"
FIXTURES = ROOT / "tests" / "fixtures" / "api_surface"

spec = importlib.util.spec_from_file_location("api_surface_diff", SCRIPT)
assert spec is not None and spec.loader is not None
api_surface_diff = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api_surface_diff
spec.loader.exec_module(api_surface_diff)


def fixture_diff():
    """Return the hand-audited before/after fixture comparison."""

    before = api_surface_diff.extract_surface_from_path(
        FIXTURES / "before" / "fixturepkg"
    )
    after = api_surface_diff.extract_surface_from_path(
        FIXTURES / "after" / "fixturepkg"
    )
    return api_surface_diff.diff_surfaces(
        before,
        after,
        before_ref="fixture-before",
        after_ref="fixture-after",
        package="fixturepkg",
    )


def test_breaking_classification_matches_hand_audited_fixture_exactly():
    diff = fixture_diff()

    assert {
        (change.change, change.symbol, change.replacement) for change in diff.breaking
    } == {
        (
            "signature-narrowed",
            "fixturepkg.api.LegacyClient.request",
            None,
        ),
        ("removed", "fixturepkg.api.LegacyClient.removed_attribute", None),
        ("signature-narrowed", "fixturepkg.api.narrowed", None),
        ("removed", "fixturepkg.api.removed", None),
        (
            "renamed",
            "fixturepkg.api.renamed",
            "fixturepkg.api.renamed_replacement",
        ),
    }


def test_deprecated_decorator_is_not_classified_as_breaking():
    diff = fixture_diff()

    assert [change.symbol for change in diff.deprecated] == ["fixturepkg.api.retained"]
    assert "fixturepkg.api.retained" not in {change.symbol for change in diff.breaking}


def test_module_all_hides_unlisted_symbol():
    before = api_surface_diff.extract_surface_from_path(
        FIXTURES / "before" / "fixturepkg"
    )

    assert "fixturepkg.api.hidden_even_though_publicly_named" not in before
    assert "fixturepkg.api.removed" in before


def test_reexported_function_keeps_static_signature_without_importing():
    sources = {
        PurePosixPath("fixturepkg/__init__.py"): (
            "from .api import public\n__all__ = ['public']\n"
        ),
        PurePosixPath("fixturepkg/api.py"): (
            "def public(value: str, optional: int = 1) -> str:\n    return value\n"
        ),
    }

    surface = api_surface_diff.extract_surface_from_sources(sources, "fixturepkg")

    assert surface["fixturepkg.public"].signature == (
        "(value: str, optional: int = 1) -> str"
    )


def test_transitive_class_reexport_keeps_members_without_importing():
    sources = {
        PurePosixPath("fixturepkg/__init__.py"): (
            "from .facade import Client\n__all__ = ['Client']\n"
        ),
        PurePosixPath("fixturepkg/facade.py"): (
            "from .api import Client\n__all__ = ['Client']\n"
        ),
        PurePosixPath("fixturepkg/api.py"): (
            "class Client:\n"
            "    def request(self, value: str) -> str:\n"
            "        return value\n"
        ),
    }

    surface = api_surface_diff.extract_surface_from_sources(sources, "fixturepkg")

    assert surface["fixturepkg.Client"].kind == "class"
    assert surface["fixturepkg.Client.request"].signature == "(value: str) -> str"


def test_lazy_reexport_map_keeps_signatures_and_class_members():
    sources = {
        PurePosixPath("fixturepkg/__init__.py"): (
            "_LAZY_IMPORTS = {'public': '.api', 'Client': '.api'}\n"
            "__all__ = ['public', 'Client']\n"
        ),
        PurePosixPath("fixturepkg/api.py"): (
            "def public(value: str, optional: int = 1) -> str:\n"
            "    return value\n\n"
            "class Client:\n"
            "    def request(self, value: str) -> str:\n"
            "        return value\n"
        ),
    }

    surface = api_surface_diff.extract_surface_from_sources(sources, "fixturepkg")

    assert surface["fixturepkg.public"].signature == (
        "(value: str, optional: int = 1) -> str"
    )
    assert surface["fixturepkg.Client"].kind == "class"
    assert surface["fixturepkg.Client.request"].signature == "(value: str) -> str"


def test_lazy_reexport_map_honors_attribute_aliases():
    sources = {
        PurePosixPath("fixturepkg/__init__.py"): (
            "_LAZY_IMPORTS = {'public': '.api'}\n"
            "_LAZY_ATTRIBUTE_NAMES = {'public': 'implementation'}\n"
            "__all__ = ['public']\n"
        ),
        PurePosixPath("fixturepkg/api.py"): (
            "def implementation(value: str) -> str:\n    return value\n"
        ),
    }

    surface = api_surface_diff.extract_surface_from_sources(sources, "fixturepkg")

    assert surface["fixturepkg.public"].signature == "(value: str) -> str"
    assert surface["fixturepkg.public"].source_target == "fixturepkg.api.implementation"


def test_resolving_a_previously_opaque_import_is_not_breaking():
    before = {
        "fixturepkg.VALUE": api_surface_diff.Symbol(
            name="fixturepkg.VALUE",
            module="fixturepkg",
            qualname="VALUE",
            kind="import",
            source_target="fixturepkg.api.VALUE",
        )
    }
    after = {
        "fixturepkg.VALUE": api_surface_diff.Symbol(
            name="fixturepkg.VALUE",
            module="fixturepkg",
            qualname="VALUE",
            kind="data",
            source_target="fixturepkg.api.VALUE",
        )
    }

    diff = api_surface_diff.diff_surfaces(before, after, package="fixturepkg")

    assert diff.breaking == ()


def test_package_local_reexport_without_all_remains_public():
    sources = {
        PurePosixPath("fixturepkg/api.py"): "VALUE = {'en'}\n",
        PurePosixPath("fixturepkg/bridge.py"): (
            "from typing import Any\nfrom .api import VALUE\n"
        ),
    }

    surface = api_surface_diff.extract_surface_from_sources(sources, "fixturepkg")

    assert surface["fixturepkg.bridge.VALUE"].kind == "data"
    assert "fixturepkg.bridge.Any" not in surface


def test_json_diff_is_machine_readable_and_stable():
    payload = fixture_diff().to_dict()

    assert payload["schema_version"] == 1
    assert payload["summary"] == {
        "before_symbols": 8,
        "after_symbols": 7,
        "added": 1,
        "deprecated": 1,
        "breaking": 5,
    }
    assert [change["symbol"] for change in payload["added"]] == [
        "fixturepkg.api.added_later"
    ]
    assert all("fingerprint" not in change for change in payload["breaking"])


def test_missing_entry_fails_and_names_the_symbol():
    diff = fixture_diff()
    complete = "\n".join(change.symbol for change in (*diff.breaking, *diff.deprecated))

    assert api_surface_diff.missing_migration_symbols(diff, complete) == ()
    missing_symbol = "fixturepkg.api.removed"
    incomplete = complete.replace(missing_symbol, "")
    assert api_surface_diff.missing_migration_symbols(diff, incomplete) == (
        missing_symbol,
    )


def test_missing_deprecated_entry_fails_and_names_the_symbol():
    diff = fixture_diff()
    complete = "\n".join(change.symbol for change in (*diff.breaking, *diff.deprecated))
    missing_symbol = diff.deprecated[0].symbol
    incomplete = complete.replace(missing_symbol, "")

    assert api_surface_diff.missing_migration_symbols(diff, incomplete) == (
        missing_symbol,
    )


def test_check_cli_fails_closed_then_passes_when_entry_is_restored(
    tmp_path, monkeypatch, capsys
):
    diff = fixture_diff()
    complete = "\n".join(change.symbol for change in (*diff.breaking, *diff.deprecated))
    missing_symbol = "fixturepkg.api.removed"
    guide = tmp_path / "migration.md"
    guide.write_text(complete.replace(missing_symbol, ""), encoding="utf-8")
    monkeypatch.setattr(api_surface_diff, "compare_refs", lambda *args: diff)

    assert api_surface_diff.main(["before", "after", "--check", str(guide)]) == 1
    assert missing_symbol in capsys.readouterr().err

    guide.write_text(complete, encoding="utf-8")
    assert api_surface_diff.main(["before", "after", "--check", str(guide)]) == 0
    assert "completeness check passed" in capsys.readouterr().out


def test_json_stdout_stays_machine_readable_when_check_passes(
    tmp_path, monkeypatch, capsys
):
    diff = fixture_diff()
    guide = tmp_path / "migration.md"
    guide.write_text(
        "\n".join(change.symbol for change in (*diff.breaking, *diff.deprecated)),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_surface_diff, "compare_refs", lambda *args: diff)

    assert (
        api_surface_diff.main(["before", "after", "--json", "-", "--check", str(guide)])
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["schema_version"] == 1
    assert "completeness check passed" in captured.err


def test_full_package_extraction_is_ast_only_and_under_thirty_seconds(monkeypatch):
    imported_before = set(sys.modules)
    original_import = builtins.__import__

    def reject_openmed_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openmed" or name.startswith("openmed."):
            raise AssertionError(f"extractor imported {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_openmed_import)
    started = time.monotonic()
    surface = api_surface_diff.extract_surface(ROOT, "WORKTREE")
    elapsed = time.monotonic() - started

    assert surface
    assert elapsed < 30
    assert set(sys.modules) - imported_before == set()


def test_real_migration_guide_covers_every_detected_break(tmp_path):
    baseline = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            "--verify",
            "--quiet",
            "v1.8.0^{commit}",
        ],
        capture_output=True,
        check=False,
    )
    if baseline.returncode != 0:
        pytest.skip(
            "v1.8.0 is unavailable in this shallow checkout; tag builds fetch history"
        )
    diff = api_surface_diff.compare_refs(ROOT, "v1.8.0", "WORKTREE")
    guide = ROOT / "docs" / "migration" / "1.8-to-1.9.md"
    text = guide.read_text(encoding="utf-8")

    assert api_surface_diff.missing_migration_symbols(diff, text) == ()
    if not diff.breaking:
        return
    omitted = diff.breaking[0].symbol
    incomplete = tmp_path / "incomplete.md"
    incomplete.write_text(text.replace(omitted, "", 1), encoding="utf-8")

    assert api_surface_diff.check_migration_document(diff, incomplete) == (omitted,)


def test_model_release_workflow_is_separate_from_sdk_tags():
    model_workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )
    publish_workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )

    assert "\n  push:" not in model_workflow
    assert "workflow_dispatch:" in model_workflow
    assert "schedule:" in model_workflow
    assert "fetch-depth: 0" in model_workflow
    assert "Check API migration guide completeness" in model_workflow
    assert "if: github.event_name == 'workflow_dispatch'" in model_workflow
    assert "scripts/release/api_surface_diff.py" in model_workflow
    assert 'pip install -e ".[dev,hf,zh,indic]"' in model_workflow
    assert "default: v2.1.0" in model_workflow
    assert "default: docs/migration/2.1-to-2.2.md" in model_workflow
    assert "API migration guide completeness gate passed." in model_workflow
    assert "tags:\n      - 'v*'" in publish_workflow
