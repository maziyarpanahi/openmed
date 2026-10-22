"""Apple privacy manifest and required-reason API scan tests."""

from __future__ import annotations

import importlib.util
import plistlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_privacy_manifests.py"

spec = importlib.util.spec_from_file_location("check_privacy_manifests", SCRIPT)
assert spec is not None
privacy_manifests = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = privacy_manifests
spec.loader.exec_module(privacy_manifests)

KIT_TARGETS = {
    "OpenMedKit": "OpenMedKit",
    "OpenMedExtensionSupport": "OpenMedExtensionSupport",
    "OpenMedShareExtension": "ShareExtension",
    "OpenMedActionExtension": "ActionExtension",
}
MANIFEST_NAME = privacy_manifests.MANIFEST_NAME
KIT_SOURCES = ROOT / privacy_manifests.KIT_SOURCES
MANIFESTS = {
    target: KIT_SOURCES / directory / "Resources" / MANIFEST_NAME
    for target, directory in KIT_TARGETS.items()
} | {"OpenMedScanDemo": ROOT / privacy_manifests.SCAN_DEMO / MANIFEST_NAME}


def _write_manifest(path: Path, api_types: list[dict] | None = None, **overrides):
    manifest = {
        "NSPrivacyTracking": False,
        "NSPrivacyTrackingDomains": [],
        "NSPrivacyCollectedDataTypes": [],
        "NSPrivacyAccessedAPITypes": api_types or [],
        **overrides,
    }
    path.write_bytes(plistlib.dumps(manifest))
    return path


def _declaration(category: str, *reasons: str) -> dict:
    return {
        "NSPrivacyAccessedAPIType": category,
        "NSPrivacyAccessedAPITypeReasons": list(reasons),
    }


def test_repository_manifests_pass(capsys):
    assert privacy_manifests.main() == 0
    assert "declare every required-reason API" in capsys.readouterr().out


@pytest.mark.parametrize("target", MANIFESTS)
def test_manifests_declare_no_tracking_or_collected_data(target):
    data = plistlib.loads(MANIFESTS[target].read_bytes())

    assert data["NSPrivacyTracking"] is False
    assert data["NSPrivacyTrackingDomains"] == []
    assert data["NSPrivacyCollectedDataTypes"] == []


@pytest.mark.parametrize("package", ["Package.swift", "swift/OpenMedKit/Package.swift"])
def test_package_targets_process_their_privacy_manifest(package):
    text = (ROOT / package).read_text(encoding="utf-8")

    for target in KIT_TARGETS:
        block = re.search(rf'\.target\(\s*name: "{target}",.*?\n        \)', text, re.S)
        assert block is not None, target
        assert '.process("Resources")' in block.group(), target


def test_demo_user_defaults_need_declaration(tmp_path):
    for source in (ROOT / privacy_manifests.SCAN_DEMO).rglob("*.swift"):
        copy = tmp_path / source.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(source.read_bytes())
    demo = tmp_path / privacy_manifests.SCAN_DEMO
    # The demo manifest before user defaults access was declared.
    manifest = _write_manifest(demo / MANIFEST_NAME)

    errors = privacy_manifests.check_target(demo, manifest, tmp_path)

    assert any("ScanFlowViewModel.swift:" in error for error in errors)
    assert any("ClinicalPresetsStore.swift:" in error for error in errors)
    assert all("NSPrivacyAccessedAPICategoryUserDefaults" in error for error in errors)

    fixed = ROOT / privacy_manifests.SCAN_DEMO / MANIFEST_NAME
    manifest.write_bytes(fixed.read_bytes())
    assert privacy_manifests.check_target(demo, manifest, tmp_path) == []


def test_undeclared_file_timestamp_api_fails(tmp_path):
    source = tmp_path / "Store.swift"
    source.write_text(
        "/* Cache freshness check.\n */\n"
        "let keys: Set<URLResourceKey> = [.contentModificationDateKey]\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path / MANIFEST_NAME)

    assert privacy_manifests.check_target(tmp_path, manifest, tmp_path) == [
        "Store.swift:3 contentModificationDateKey: "
        "NSPrivacyAccessedAPICategoryFileTimestamp is not declared in "
        "PrivacyInfo.xcprivacy"
    ]

    category = "NSPrivacyAccessedAPICategoryFileTimestamp"
    _write_manifest(manifest, [_declaration(category, "C617.1")])
    assert privacy_manifests.check_target(tmp_path, manifest, tmp_path) == []


def test_comments_are_not_api_use(tmp_path):
    (tmp_path / "Notes.swift").write_text(
        '/// Persists to UserDefaults.\nlet url = "https://example.invalid" // stat\n'
        "/* systemUptime\n   mach_absolute_time() */\nlet value = 1\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path / MANIFEST_NAME)

    assert privacy_manifests.check_target(tmp_path, manifest, tmp_path) == []


def test_string_literal_text_is_not_api_use(tmp_path):
    (tmp_path / "Labels.swift").write_text(
        'let message = "UserDefaults \\"stat\\" // creationDate"\n'
        'let raw = #"systemUptime "quoted" \\(AppStorage)"#\n'
        'let help = """\n  Reads "UserDefaults" and \\\n  activeInputModes\n  """\n'
        "let value = 1\n",
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path / MANIFEST_NAME)

    assert privacy_manifests.check_target(tmp_path, manifest, tmp_path) == []


def test_api_use_inside_string_interpolation_is_detected(tmp_path):
    (tmp_path / "Status.swift").write_text(
        'let text = """\n  Saved: \\(UserDefaults.standard.string(forKey: "k") ?? "-")\n'
        '  """\nlet raw = #"Up \\#(ProcessInfo.processInfo.systemUptime)s"#\n',
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path / MANIFEST_NAME)

    assert privacy_manifests.check_target(tmp_path, manifest, tmp_path) == [
        "Status.swift:4 systemUptime: NSPrivacyAccessedAPICategorySystemBootTime "
        "is not declared in PrivacyInfo.xcprivacy",
        "Status.swift:2 UserDefaults: NSPrivacyAccessedAPICategoryUserDefaults "
        "is not declared in PrivacyInfo.xcprivacy",
    ]


def test_tracking_collected_data_and_unapproved_reasons_fail(tmp_path):
    manifest = _write_manifest(
        tmp_path / MANIFEST_NAME,
        [_declaration("NSPrivacyAccessedAPICategoryUserDefaults", "35F9.1")],
        NSPrivacyTracking=True,
        NSPrivacyTrackingDomains=["tracker.example.invalid"],
        NSPrivacyCollectedDataTypes=[{"NSPrivacyCollectedDataType": "Health"}],
    )

    assert privacy_manifests.check_target(tmp_path, manifest, tmp_path) == [
        "PrivacyInfo.xcprivacy: NSPrivacyTracking must be false",
        "PrivacyInfo.xcprivacy: NSPrivacyTrackingDomains must be empty",
        "PrivacyInfo.xcprivacy: NSPrivacyCollectedDataTypes must be empty",
        "PrivacyInfo.xcprivacy: NSPrivacyAccessedAPICategoryUserDefaults needs "
        "approved reasons",
    ]


def test_kit_target_without_manifest_fails(tmp_path):
    (tmp_path / privacy_manifests.KIT_SOURCES / "NewTarget").mkdir(parents=True)
    demo = tmp_path / privacy_manifests.SCAN_DEMO
    demo.mkdir(parents=True)
    _write_manifest(demo / MANIFEST_NAME)

    assert privacy_manifests.check_repository(tmp_path) == [
        "swift/OpenMedKit/Sources/NewTarget/Resources/PrivacyInfo.xcprivacy: "
        "missing or unreadable privacy manifest"
    ]
