import json
from pathlib import Path

import pytest

from scripts.release.check_removals_ledger import LEDGER, _has_marker, validate


def test_committed_ledger():
    root = Path(__file__).resolve().parents[3]
    assert validate(root) == []
    assert {"E2", "P3", "P5", "G2", "C2", "android-mlx"} <= {
        entry["id"] for entry in json.loads((root / LEDGER).read_text())
    }


def test_deprecated_entry_needs_the_named_marker_and_reverse_coverage(tmp_path):
    ledger = tmp_path / LEDGER
    ledger.parent.mkdir(parents=True)
    row = dict(
        id="test",
        surfaces=["python"],
        symbols={"openmed/test.py": ["Old"]},
        status="deprecated",
        replacement="New",
        migration_anchor="docs/migration.md#test",
    )
    ledger.write_text(json.dumps([row]))
    source = tmp_path / "openmed/test.py"
    source.parent.mkdir()
    source.write_text(
        "class Old: pass\n@deprecated(remove_in='5.0')\nclass Other: pass"
    )
    assert validate(tmp_path) == [
        "test: missing deprecation marker for openmed/test.py#Old",
        "unlisted 5.0 removal: openmed/test.py#Other",
    ]
    source.write_text("@deprecated(remove_in='5.0')\nclass Old: pass")
    assert validate(tmp_path) == []


@pytest.mark.parametrize(
    ("suffix", "source"),
    [
        (
            "kt",
            '@Deprecated("removed", ReplaceWith("new()"))\nfun old() {}\nfun other() {}',
        ),
        (
            "ts",
            "/** @deprecated use New */\nexport class old {}\nexport class other {}",
        ),
        (
            "swift",
            '@available(*, deprecated, renamed: "new")\npublic func old() {}\npublic func other() {}',
        ),
        ("go", "// Deprecated: use New.\ntype old struct{}\ntype other struct{}"),
    ],
)
def test_marker_must_apply_to_named_declaration(suffix, source):
    path = Path("fixture." + suffix)
    assert _has_marker(path, "old", source)
    assert not _has_marker(path, "other", source)


def test_parameter_marker_tracks_only_the_named_parameter(tmp_path):
    from scripts.release.check_removals_ledger import _python_markers

    source = tmp_path / "fixture.py"
    source.write_text(
        '@deprecated_parameter("old", since="4.3", remove_in="5.0", replacement="new")\ndef setup(*, old=None, new=None): pass'
    )
    assert _python_markers(source) == {"setup.old"}
