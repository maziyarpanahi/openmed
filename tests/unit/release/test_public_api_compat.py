"""Public-API backwards-compatibility and deprecation-policy checker tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_public_api.py"
BASELINE = ROOT / "scripts" / "release" / "public_api_baseline.json"
ALLOWLIST = ROOT / "scripts" / "release" / "public_api_allowlist.json"
POLICY_DOC = ROOT / "docs" / "compliance" / "api-deprecation-policy.md"

spec = importlib.util.spec_from_file_location("check_public_api", SCRIPT)
assert spec is not None
api = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = api
spec.loader.exec_module(api)


def _member(name: str, kind: str = "function", signature: str | None = "(text=...)"):
    return api.MemberSnapshot(name=name, kind=kind, signature=signature)


def _surface(module: str, *members):
    return api.SurfaceSnapshot(module=module, members=tuple(members))


# --- Baseline integrity and round-tripping ---------------------------------


def test_committed_baseline_exists_and_is_wellformed():
    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert document["schema_version"] == api.SCHEMA_VERSION
    assert "openmed" in document["modules"]
    # The top-level package must at least carry its documented public surface.
    assert document["modules"]["openmed"]


def test_current_surface_matches_committed_baseline():
    """The committed baseline must pass against today's surface (green CI)."""

    baseline = api.load_baseline(BASELINE)
    current = api.capture_surface()
    changes = api.diff_surface(baseline, current)
    allowlist = api.load_allowlist(ALLOWLIST)
    failures = api.unannounced_breaks(changes, allowlist)
    assert failures == [], [change.detail for change in failures]


def test_run_check_passes_against_committed_baseline():
    assert api.run_check(BASELINE, ALLOWLIST) == 0


def test_capture_and_document_round_trip(tmp_path):
    original = api.capture_surface()
    document = api.surface_to_document(original)

    path = tmp_path / "baseline.json"
    api.write_baseline(path, original)
    reloaded_document = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded_document == document

    rehydrated = api.document_to_surface(reloaded_document)
    # Re-diffing a surface against its own serialized baseline yields no change.
    assert api.diff_surface(rehydrated, original) == []


def test_baseline_is_deterministic():
    first = api.surface_to_document(api.capture_surface())
    second = api.surface_to_document(api.capture_surface())
    assert first == second


# --- Removal is flagged as breaking ----------------------------------------


def test_injected_removal_is_flagged_breaking():
    baseline = [_surface("openmed", _member("stays"), _member("removed_symbol"))]
    current = [_surface("openmed", _member("stays"))]

    changes = api.diff_surface(baseline, current)
    removed = [change for change in changes if change.name == "removed_symbol"]
    assert len(removed) == 1
    assert removed[0].kind == "removed"
    assert removed[0].breaking is True

    failures = api.unannounced_breaks(changes, allowlist=set())
    assert [change.location for change in failures] == ["openmed.removed_symbol"]


def test_removal_can_be_announced_via_allowlist():
    baseline = [_surface("openmed", _member("removed_symbol"))]
    current = [_surface("openmed")]

    changes = api.diff_surface(baseline, current)
    allowlisted = api.unannounced_breaks(changes, allowlist={"openmed.removed_symbol"})
    assert allowlisted == []


def test_kind_change_is_breaking():
    baseline = [_surface("openmed", _member("thing", kind="function"))]
    current = [_surface("openmed", _member("thing", kind="class"))]

    changes = api.diff_surface(baseline, current)
    assert len(changes) == 1
    assert changes[0].kind == "kind_changed"
    assert changes[0].breaking is True


# --- Additions are NOT breaking --------------------------------------------


def test_injected_addition_is_not_breaking():
    baseline = [_surface("openmed", _member("stays"))]
    current = [_surface("openmed", _member("stays"), _member("brand_new"))]

    changes = api.diff_surface(baseline, current)
    added = [change for change in changes if change.name == "brand_new"]
    assert len(added) == 1
    assert added[0].kind == "added"
    assert added[0].breaking is False

    assert api.unannounced_breaks(changes, allowlist=set()) == []


def test_new_module_is_all_additions():
    baseline = [_surface("openmed")]
    current = [
        _surface("openmed"),
        _surface("openmed.newpkg", _member("one"), _member("two")),
    ]

    changes = api.diff_surface(baseline, current)
    assert {change.name for change in changes} == {"one", "two"}
    assert all(change.kind == "added" for change in changes)
    assert api.unannounced_breaks(changes, allowlist=set()) == []


# --- Signature-change classification ---------------------------------------


def test_adding_optional_parameter_is_not_breaking():
    baseline = [_surface("openmed", _member("f", signature="(text=...)"))]
    current = [_surface("openmed", _member("f", signature="(text=..., extra=...)"))]

    changes = api.diff_surface(baseline, current)
    assert len(changes) == 1
    assert changes[0].kind == "signature_changed"
    assert changes[0].breaking is False
    assert api.unannounced_breaks(changes, allowlist=set()) == []


def test_adding_required_parameter_is_breaking():
    baseline = [_surface("openmed", _member("f", signature="(text=...)"))]
    current = [_surface("openmed", _member("f", signature="(text=..., required)"))]

    changes = api.diff_surface(baseline, current)
    assert changes[0].breaking is True


def test_removing_parameter_is_breaking():
    baseline = [_surface("openmed", _member("f", signature="(a, b=...)"))]
    current = [_surface("openmed", _member("f", signature="(a)"))]

    changes = api.diff_surface(baseline, current)
    assert changes[0].breaking is True


def test_making_optional_parameter_required_is_breaking():
    baseline = [_surface("openmed", _member("f", signature="(a=...)"))]
    current = [_surface("openmed", _member("f", signature="(a)"))]

    changes = api.diff_surface(baseline, current)
    assert changes[0].breaking is True


def test_reordering_positional_parameters_is_breaking():
    baseline = [_surface("openmed", _member("f", signature="(a, b)"))]
    current = [_surface("openmed", _member("f", signature="(b, a)"))]

    changes = api.diff_surface(baseline, current)
    assert changes[0].breaking is True


def test_widening_with_kwargs_is_not_breaking():
    baseline = [_surface("openmed", _member("f", signature="(a=...)"))]
    current = [_surface("openmed", _member("f", signature="(a=..., **kwargs)"))]

    changes = api.diff_surface(baseline, current)
    assert changes[0].breaking is False


def test_changing_default_value_is_not_breaking():
    # The capture format records only whether a default exists, so a changed
    # default value never even registers as a diff.
    assert api._member_signature.__doc__  # sanity: helper is documented
    baseline = [_surface("openmed", _member("f", signature="(a=...)"))]
    current = [_surface("openmed", _member("f", signature="(a=...)"))]
    assert api.diff_surface(baseline, current) == []


# --- Missing optional dependencies must not cause false removals -----------


def test_module_absent_from_current_is_not_reported_removed():
    baseline = [
        _surface("openmed", _member("stays")),
        _surface("openmed.optional", _member("gone_because_uninstalled")),
    ]
    current = [_surface("openmed", _member("stays"))]

    changes = api.diff_surface(baseline, current)
    assert changes == []


# --- CLI behaviour ----------------------------------------------------------


def test_update_writes_baseline_then_check_passes(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(json.dumps({"announced_breaks": {}}), encoding="utf-8")

    assert api.main(["--update", "--baseline", str(baseline_path)]) == 0
    assert baseline_path.exists()
    assert (
        api.main(
            [
                "--baseline",
                str(baseline_path),
                "--allowlist",
                str(allowlist_path),
            ]
        )
        == 0
    )


def test_missing_baseline_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        api.load_baseline(tmp_path / "does-not-exist.json")


def test_allowlist_accepts_bare_mapping(tmp_path):
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(
        json.dumps({"openmed.gone": "removed in 2.0"}), encoding="utf-8"
    )
    assert api.load_allowlist(allowlist_path) == {"openmed.gone"}


def test_committed_allowlist_loads():
    # The committed allowlist must be parseable and currently empty of stale
    # entries (nothing announced yet on a fresh baseline).
    assert api.load_allowlist(ALLOWLIST) == set()


# --- Policy documentation ---------------------------------------------------


def test_deprecation_policy_doc_exists():
    text = POLICY_DOC.read_text(encoding="utf-8")
    assert "Deprecation" in text
    assert "public_api_allowlist.json" in text
    assert "check_public_api.py" in text
