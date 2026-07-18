"""Tests for the static release API-surface comparison and migration gate."""

from __future__ import annotations

import builtins
import importlib.util
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
    complete = "\n".join(change.symbol for change in diff.breaking)

    assert api_surface_diff.missing_migration_symbols(diff, complete) == ()
    missing_symbol = "fixturepkg.api.removed"
    incomplete = complete.replace(missing_symbol, "")
    assert api_surface_diff.missing_migration_symbols(diff, incomplete) == (
        missing_symbol,
    )


def test_check_cli_fails_closed_then_passes_when_entry_is_restored(
    tmp_path, monkeypatch, capsys
):
    diff = fixture_diff()
    complete = "\n".join(change.symbol for change in diff.breaking)
    missing_symbol = "fixturepkg.api.removed"
    guide = tmp_path / "migration.md"
    guide.write_text(complete.replace(missing_symbol, ""), encoding="utf-8")
    monkeypatch.setattr(api_surface_diff, "compare_refs", lambda *args: diff)

    assert api_surface_diff.main(["before", "after", "--check", str(guide)]) == 1
    assert missing_symbol in capsys.readouterr().err

    guide.write_text(complete, encoding="utf-8")
    assert api_surface_diff.main(["before", "after", "--check", str(guide)]) == 0
    assert "completeness check passed" in capsys.readouterr().out


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


def test_release_workflow_runs_gate_only_for_tags():
    workflow = (ROOT / ".github" / "workflows" / "release-gates.yml").read_text(
        encoding="utf-8"
    )

    assert 'tags:\n      - "v*"' in workflow
    assert "fetch-depth: 0" in workflow
    assert "Check API migration guide completeness" in workflow
    assert "if: startsWith(github.ref, 'refs/tags/')" in workflow
    assert "scripts/release/api_surface_diff.py" in workflow
    assert "API migration guide completeness gate passed." in workflow
