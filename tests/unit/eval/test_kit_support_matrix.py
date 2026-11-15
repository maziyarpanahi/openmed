"""Coherence and negative-control tests for the frozen OpenMedKit support matrix.

The matrix in ``openmed/eval/data/kit_support_matrix.json`` must agree with the
package manifests, ``docs/export-matrix.md``, ``openmed.eval.tiers`` and its
rendered page. Negative controls edit a copied manifest set under ``tmp_path``
so every drift case is shown to fail, not only the happy path.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from openmed.eval.kit_support_matrix import (
    DEFAULT_KIT_SUPPORT_MATRIX,
    DEVICE_CLASSES,
    EPIC_SURFACES,
    EXPERIMENTAL,
    PENDING_EVIDENCE,
    SUPPORTED,
    UNSUPPORTED,
    KitSupportMatrixError,
    entries_with_status,
    load_kit_support_matrix,
    render_kit_support_matrix_markdown,
    tier_names,
    validate_kit_support_matrix,
)
from openmed.eval.tiers import TIERS

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_kit_support_matrix.py"
SPEC = importlib.util.spec_from_file_location("check_kit_support_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)

DOC = ROOT / "docs" / "openmedkit-support-matrix.md"


@pytest.fixture
def matrix() -> dict[str, Any]:
    return copy.deepcopy(load_kit_support_matrix())


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Copy the manifests, docs and evidence anchors the check reads."""
    files = [
        *checker.manifest_paths(ROOT),
        checker.EXPORT_MATRIX_DOC,
        checker.DOC_PATH.as_posix(),
        checker.MKDOCS.as_posix(),
    ]
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    for record in load_kit_support_matrix()["evidence"]:
        for ref in record["refs"]:
            target = tmp_path / ref
            if (ROOT / ref).is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")
    return tmp_path


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture anchor missing in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _entry(matrix: dict[str, Any], entry_id: str) -> dict[str, Any]:
    for key in ("cells", "unsupported"):
        for row in matrix[key]:
            if row["id"] == entry_id:
                return row
    raise AssertionError(f"matrix entry {entry_id} is missing")


def _problems(matrix: dict[str, Any], root: Path = ROOT) -> list[str]:
    return checker.check_repository_coherence(matrix, root)


# --- committed matrix -------------------------------------------------------


def test_committed_matrix_is_valid_and_coherent_with_the_repository() -> None:
    matrix = load_kit_support_matrix(validate=False)

    assert validate_kit_support_matrix(matrix) == ()
    assert _problems(matrix) == []
    assert matrix["release"] == "v4.0"
    assert matrix["state"] == "frozen"


def test_checker_script_reports_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert checker.main([]) == 0
    assert "kit support matrix: ok" in capsys.readouterr().out


def test_copied_repository_is_coherent(repo: Path, matrix: dict[str, Any]) -> None:
    assert _problems(matrix, repo) == []


def test_every_epic_surface_states_each_support_level_explicitly() -> None:
    matrix = load_kit_support_matrix()
    page = DOC.read_text(encoding="utf-8")
    surfaces = {surface["id"]: surface for surface in matrix["surfaces"]}

    assert tuple(surfaces) == EPIC_SURFACES
    for surface_id, surface in surfaces.items():
        supported = entries_with_status(matrix, SUPPORTED, surface=surface_id)
        experimental = entries_with_status(matrix, EXPERIMENTAL, surface=surface_id)
        unsupported = entries_with_status(matrix, UNSUPPORTED, surface=surface_id)
        assert supported or experimental, surface_id
        assert unsupported, surface_id
        assert surface["support_claim"] == (
            "supported" if supported else "experimental-only"
        )

        section = page.split(f"\n## {surface['name']}\n", 1)[1].split("\n## ", 1)[0]
        for heading in ("Supported", "Experimental", "Unsupported combinations"):
            assert f"### {heading}" in section, (surface_id, heading)


def test_p2_surfaces_claim_no_supported_cells() -> None:
    matrix = load_kit_support_matrix()
    for surface in matrix["surfaces"]:
        if surface["epic_scope"] == "P2":
            assert not entries_with_status(matrix, SUPPORTED, surface=surface["id"])


def test_every_supported_cell_names_existing_evidence() -> None:
    matrix = load_kit_support_matrix()
    records = {record["id"]: record for record in matrix["evidence"]}
    supported = entries_with_status(matrix, SUPPORTED)

    assert supported
    for cell in supported:
        assert cell["evidence"] != PENDING_EVIDENCE
        for evidence_id in cell["evidence"]:
            for ref in records[evidence_id]["refs"]:
                assert (ROOT / ref).exists(), (cell["id"], evidence_id)


def test_runtime_skew_matches_the_manifest_pins() -> None:
    matrix = load_kit_support_matrix()
    pins = {pin["id"]: pin for pin in matrix["runtime_pins"]}
    skews = {skew["runtime"]: skew for skew in matrix["runtime_skew"]}

    runtimes = {pin["runtime"] for pin in pins.values() if pin["pinning"] != "system"}
    for runtime in sorted(runtimes):
        members = [pin for pin in pins.values() if pin["runtime"] == runtime]
        declared = {
            checker.pin_value(ROOT, source, pin)
            for pin in members
            for source in pin["sources"]
        }
        if len(declared) > 1:
            # Differing kit pins must be listed, never implied equal.
            assert set(skews[runtime]["pins"]) == {pin["id"] for pin in members}
        else:
            assert runtime not in skews


def test_matrix_uses_only_coarse_device_classes() -> None:
    matrix = load_kit_support_matrix()
    for entry in [*matrix["cells"], *matrix["unsupported"]]:
        assert entry["device_class"] in DEVICE_CLASSES | {"*"}
    text = DEFAULT_KIT_SUPPORT_MATRIX.read_text(encoding="utf-8")
    assert not re.search(r"\b(?:iPhone|iPad|Watch)\d+,\d+\b", text)
    assert "/Users/" not in text and "@" not in text


def test_every_tier_resolves_in_the_canonical_tier_table() -> None:
    matrix = load_kit_support_matrix()
    names = tier_names(TIERS)
    tiers = {cell["tier"] for cell in matrix["cells"] if cell["tier"] is not None}

    assert tiers <= names
    assert {"Nano", "Tiny", "Base"} <= tiers


def test_docs_page_is_rendered_from_the_matrix_and_published() -> None:
    matrix = load_kit_support_matrix()
    page = DOC.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    publication = yaml.safe_load(
        (ROOT / "docs" / "brand" / "system" / "publication.yml").read_text(
            encoding="utf-8"
        )
    )

    assert checker.extract_docs_block(page) == render_kit_support_matrix_markdown(
        matrix
    )
    assert "openmedkit-support-matrix.md" in nav
    assert "openmedkit-support-matrix.md" in publication["classification"]["navigated"]
    assert "v3.5" in page and "v4.0" in page
    assert "export-matrix.md" in page and "tiers.md" in page
    export_page = (ROOT / "docs" / "export-matrix.md").read_text(encoding="utf-8")
    assert "openmedkit-support-matrix.md" in export_page


# --- evidence rule ----------------------------------------------------------


def test_supported_cell_with_pending_evidence_is_rejected(
    matrix: dict[str, Any],
) -> None:
    _entry(matrix, "swift-macos-mlx")["evidence"] = PENDING_EVIDENCE

    problems = validate_kit_support_matrix(matrix)

    assert "cell_supported_without_evidence:swift-macos-mlx" in problems


def test_pending_evidence_is_allowed_on_an_experimental_cell(
    matrix: dict[str, Any],
) -> None:
    cell = _entry(matrix, "swift-macos-mlx")
    cell["evidence"] = PENDING_EVIDENCE
    cell["status"] = EXPERIMENTAL
    for surface in matrix["surfaces"]:
        if surface["id"] == "swift":
            surface["support_claim"] = "experimental-only"

    assert validate_kit_support_matrix(matrix) == ()


@pytest.mark.parametrize("evidence", [[], ["ci-made-up-lane"], "missing", None])
def test_supported_cell_needs_a_known_evidence_id(
    matrix: dict[str, Any], evidence: Any
) -> None:
    _entry(matrix, "python-linux-torch-cpu")["evidence"] = evidence

    problems = validate_kit_support_matrix(matrix)

    assert any(
        code
        in {
            "cell_supported_without_evidence:python-linux-torch-cpu",
            "cell_evidence_unknown:python-linux-torch-cpu",
        }
        for code in problems
    )


def test_evidence_reference_must_exist_in_the_repository(
    matrix: dict[str, Any],
) -> None:
    matrix["evidence"][0]["refs"] = ["tests/fixtures/not-a-real-anchor.json"]

    assert f"evidence_ref_missing:{matrix['evidence'][0]['id']}" in _problems(matrix)


def test_surface_claim_must_match_its_cells(matrix: dict[str, Any]) -> None:
    for surface in matrix["surfaces"]:
        if surface["id"] == "android":
            surface["support_claim"] = "supported"

    assert "surface_claim_mismatch:android" in validate_kit_support_matrix(matrix)


def test_missing_epic_surface_is_rejected(matrix: dict[str, Any]) -> None:
    matrix["surfaces"] = [s for s in matrix["surfaces"] if s["id"] != "tauri"]
    matrix["cells"] = [c for c in matrix["cells"] if c["surface"] != "tauri"]
    matrix["unsupported"] = [
        u for u in matrix["unsupported"] if u["surface"] != "tauri"
    ]

    assert "surfaces_mismatch_epic" in validate_kit_support_matrix(matrix)


def test_surface_without_unsupported_rows_is_rejected(
    matrix: dict[str, Any],
) -> None:
    matrix["unsupported"] = [u for u in matrix["unsupported"] if u["surface"] != "web"]

    assert "surface_without_unsupported:web" in validate_kit_support_matrix(matrix)


# --- privacy boundary -------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "Measured on iPhone15,2 only.",
        "Run 6f1c2a9e-3b4d-4e5f-8a7b-1c2d3e4f5a6b recorded.",
        "Copied from /Users/someone/models.",
        "Reported by person@example.org.",
        "Device 00008110-001A2B3C4D5E6F70 failed.",
        "IMEI 356938035643809 attached.",
        "Board serial recorded.",
    ],
)
def test_device_identifiers_and_user_data_are_rejected_without_echo(
    matrix: dict[str, Any], value: str
) -> None:
    _entry(matrix, "android-arm64-onnx-cpu")["note"] = value

    problems = validate_kit_support_matrix(matrix)

    assert any(code.startswith("matrix_identifier_value:") for code in problems)
    assert all(value not in code for code in problems)


def test_unknown_device_class_and_fields_are_rejected(matrix: dict[str, Any]) -> None:
    cell = _entry(matrix, "android-arm64-onnx-cpu")
    cell["device_class"] = "pixel-8-pro"
    cell["device_model"] = "Pixel 8 Pro"

    problems = validate_kit_support_matrix(matrix)

    assert "cell_device_class_invalid:android-arm64-onnx-cpu" in problems
    assert "cell_field_unknown:android-arm64-onnx-cpu" in problems


# --- tiers and families -----------------------------------------------------


def test_unknown_tier_is_rejected(matrix: dict[str, Any]) -> None:
    _entry(matrix, "swift-ios-mlx")["tier"] = "Huge"

    assert "cell_tier_unknown:swift-ios-mlx" in validate_kit_support_matrix(matrix)


def test_tier_table_change_without_matrix_update_is_detected(
    matrix: dict[str, Any],
) -> None:
    tiers = copy.deepcopy(TIERS)
    tiers["Tiny"].pop("sub_tiers")

    problems = validate_kit_support_matrix(matrix, tiers=tiers)

    assert "cell_tier_unknown:swift-watchos-coreml-nano" in problems


def test_family_list_change_without_matrix_update_is_detected(
    repo: Path, matrix: dict[str, Any]
) -> None:
    export = repo / "docs" / "export-matrix.md"
    _edit(
        export,
        "| ModernBERT |",
        "| EuroBERT | `eurobert` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |\n"
        "| ModernBERT |",
    )

    assert "families_mismatch_export_matrix" in _problems(matrix, repo)


def test_cell_for_a_family_the_export_matrix_rejects_is_detected(
    repo: Path, matrix: dict[str, Any]
) -> None:
    export = repo / "docs" / "export-matrix.md"
    _edit(
        export,
        "| BERT | `bert` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ✅ | ✅ |",
        "| BERT | `bert` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ❌ | ✅ |",
    )

    assert "cell_family_not_exported:swift-macos-coreml" in _problems(matrix, repo)


def test_unsupported_family_that_is_exported_is_detected(
    matrix: dict[str, Any],
) -> None:
    entry = _entry(matrix, "swift-coreml-unexported-families")
    entry["families"] = [*entry["families"], "bert"]

    assert "unsupported_family_is_exported:swift-coreml-unexported-families" in (
        _problems(matrix)
    )


def test_precision_without_an_export_column_is_detected(
    matrix: dict[str, Any],
) -> None:
    _entry(matrix, "swift-macos-coreml")["precisions"] = ["fp32"]

    assert "cell_precision_not_exported:swift-macos-coreml" in _problems(matrix)


# --- manifests --------------------------------------------------------------


# Each pattern captures the text just before a declared version; ``\g<1>9``
# prefixes a digit so the edit changes the value whatever the current pin is.
_BUMP = r"\g<1>9"


@pytest.mark.parametrize(
    ("relative", "pattern", "replacement", "code"),
    [
        (
            "android/gradle/libs.versions.toml",
            r'(onnxruntime = ")',
            _BUMP,
            "pin_mismatch:onnxruntime-android",
        ),
        (
            "js/openmedkit-flutter/android/build.gradle",
            r"(def onnxRuntimeVersion = ')",
            _BUMP,
            "pin_mismatch:onnxruntime-flutter-android",
        ),
        (
            "js/openmedkit-flutter/ios/openmedkit_flutter.podspec",
            r"(s\.dependency 'onnxruntime-c', ')",
            _BUMP,
            "pin_mismatch:onnxruntime-flutter-ios",
        ),
        (
            "js/openmedkit-web/package.json",
            r'("onnxruntime-web": ">=)',
            _BUMP,
            "pin_mismatch:onnxruntime-web",
        ),
        (
            "Package.swift",
            r'(mlx-swift\.git",\s*exact: ")',
            _BUMP,
            "pin_mismatch:mlx-swift",
        ),
        ("Package.swift", r"(\.iOS\(\.v)", _BUMP, "floor_mismatch:swift-ios"),
        (
            "swift/OpenMedKit/Package.swift",
            r"(\.macOS\(\.v)",
            _BUMP,
            "floor_mismatch:swift-macos",
        ),
        (
            "Package.swift",
            r"(\.visionOS\(\.v\d+\),)",
            r"\1\n        .macCatalyst(.v17),",
            "floor_platforms_mismatch:swift",
        ),
        (
            "android/openmedkit/build.gradle.kts",
            r"(minSdk = )",
            _BUMP,
            "floor_mismatch:android-min-sdk",
        ),
        (
            "js/openmedkit-flutter/android/build.gradle",
            r"(minSdk )",
            _BUMP,
            "floor_mismatch:flutter-android-min-sdk",
        ),
        (
            "js/openmedkit-flutter/macos/openmedkit_flutter.podspec",
            r"(s\.platform = :osx, ')",
            _BUMP,
            "floor_mismatch:flutter-macos",
        ),
        (
            "pyproject.toml",
            r'(requires-python = ">=)',
            _BUMP,
            "floor_mismatch:python-requires",
        ),
        (
            "js/openmedkit-react-native/package.json",
            r'("react-native": ">=)',
            _BUMP,
            "floor_mismatch:react-native-host",
        ),
    ],
)
def test_manifest_change_without_matrix_update_is_detected(
    repo: Path,
    matrix: dict[str, Any],
    relative: str,
    pattern: str,
    replacement: str,
    code: str,
) -> None:
    path = repo / relative
    text, count = re.subn(pattern, replacement, path.read_text(encoding="utf-8"), 1)
    assert count == 1, f"fixture anchor missing in {path.name}"
    path.write_text(text, encoding="utf-8")

    assert code in _problems(matrix, repo)


def test_new_onnxruntime_pin_in_an_unlisted_kit_manifest_is_detected(
    repo: Path, matrix: dict[str, Any]
) -> None:
    manifest = repo / "js" / "openmedkit-react-native" / "package.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document.setdefault("dependencies", {})["onnxruntime-react-native"] = "1.22.0"
    manifest.write_text(json.dumps(document, indent=2), encoding="utf-8")

    assert (
        "runtime_pin_unlisted:onnxruntime:js/openmedkit-react-native/package.json"
        in _problems(matrix, repo)
    )


def test_mlx_linkage_change_is_detected(repo: Path, matrix: dict[str, Any]) -> None:
    _edit(
        repo / "Package.swift",
        'name: "MLX",\n                    package: "mlx-swift",\n'
        "                    condition: .when(platforms: [.iOS, .macOS])",
        'name: "MLX",\n                    package: "mlx-swift",\n'
        "                    condition: .when(platforms: [.macOS])",
    )

    assert "cell_runtime_not_linked:swift-ios-mlx" in _problems(matrix, repo)


def test_cell_on_a_platform_without_mlx_linkage_is_detected(
    matrix: dict[str, Any],
) -> None:
    cell = copy.deepcopy(_entry(matrix, "swift-watchos-coreml-nano"))
    cell.update(
        id="swift-watchos-mlx-experiment",
        runtime="mlx-swift",
        provider="metal",
        artifact="mlx",
    )
    matrix["cells"].append(cell)

    assert "cell_runtime_not_linked:swift-watchos-mlx-experiment" in _problems(matrix)


def test_cell_below_its_platform_floor_is_detected(matrix: dict[str, Any]) -> None:
    _entry(matrix, "react-native-ios-mlx")["platform"]["min"] = "16"

    assert "entry_below_floor:react-native-ios-mlx" in _problems(matrix)


# --- runtime skew -----------------------------------------------------------


def test_removing_the_skew_entry_is_detected(matrix: dict[str, Any]) -> None:
    matrix["runtime_skew"] = [
        skew for skew in matrix["runtime_skew"] if skew["runtime"] != "onnxruntime"
    ]

    assert "runtime_skew_missing:onnxruntime" in _problems(matrix)


def test_partial_skew_entry_is_detected(matrix: dict[str, Any]) -> None:
    for skew in matrix["runtime_skew"]:
        if skew["runtime"] == "onnxruntime":
            skew["pins"].remove("onnxruntime-web")

    assert "runtime_skew_incomplete:onnxruntime" in _problems(matrix)


def test_skew_entry_for_aligned_pins_is_stale(
    repo: Path, matrix: dict[str, Any]
) -> None:
    pins = {pin["id"]: pin for pin in matrix["runtime_pins"]}
    _edit(
        repo / "swift" / "OpenMedKit" / "Package.swift",
        f'exact: "{pins["mlx-swift-nested"]["version"]}"',
        f'exact: "{pins["mlx-swift"]["version"]}"',
    )
    pins["mlx-swift-nested"]["version"] = pins["mlx-swift"]["version"]

    problems = _problems(matrix, repo)

    assert "runtime_skew_stale:mlx-swift" in problems
    assert "pin_mismatch:mlx-swift-nested" not in problems


# --- docs -------------------------------------------------------------------


def test_docs_drift_is_detected(repo: Path, matrix: dict[str, Any]) -> None:
    _edit(repo / checker.DOC_PATH, "| reference | supported |", "| reference | maybe |")

    assert "docs_out_of_sync" in _problems(matrix, repo)


def test_docs_block_round_trips_through_replace(matrix: dict[str, Any]) -> None:
    page = DOC.read_text(encoding="utf-8")
    block = render_kit_support_matrix_markdown(matrix)

    assert checker.replace_docs_block(page, block) == page


def test_docs_without_markers_are_rejected() -> None:
    with pytest.raises(checker.ManifestError):
        checker.extract_docs_block("# no markers here\n")


# --- loader -----------------------------------------------------------------


def test_loader_rejects_unreadable_and_invalid_documents(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(KitSupportMatrixError) as unreadable:
        load_kit_support_matrix(broken)
    assert unreadable.value.problems == ("matrix_unreadable",)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(KitSupportMatrixError) as rejected:
        load_kit_support_matrix(invalid)
    assert "schema_version_unsupported" in rejected.value.problems
