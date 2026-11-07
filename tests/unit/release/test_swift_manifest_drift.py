"""Root versus nested OpenMedKit SwiftPM manifest drift check."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_swift_manifest_drift.py"
FIXTURES = ROOT / "tests" / "fixtures" / "swift_manifests"
SWIFT_WORKFLOW = ROOT / ".github" / "workflows" / "swift-test.yml"
CONSUMER_SCRIPT = ROOT / "scripts" / "check_swift_root_consumer.sh"
CONSUMER_SOURCE = (
    ROOT
    / "swift"
    / "RootPackageConsumer"
    / "Sources"
    / "RootPackageConsumer"
    / "RootPackageConsumer.swift"
)

spec = importlib.util.spec_from_file_location("check_swift_manifest_drift", SCRIPT)
assert spec is not None
drift = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = drift
spec.loader.exec_module(drift)


def _fixture(role: str) -> dict[str, Any]:
    path = FIXTURES / f"pre_alignment_{role}.dump.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _dependency(dump: dict[str, Any], identity: str) -> dict[str, Any]:
    for entry in dump["dependencies"]:
        details = entry["sourceControl"][0]
        if details["identity"] == identity:
            return details
    raise AssertionError(f"missing dependency {identity}")


def _target(dump: dict[str, Any], name: str) -> dict[str, Any]:
    for target in dump["targets"]:
        if target["name"] == name:
            return target
    raise AssertionError(f"missing target {name}")


def _root_style(nested: dict[str, Any]) -> dict[str, Any]:
    """Return a root-manifest dump equivalent to ``nested`` (explicit paths)."""

    root = copy.deepcopy(nested)
    root["packageKind"] = {"root": ["/synthetic/openmed"]}
    for target in root["targets"]:
        default = "Tests" if target["type"] == "test" else "Sources"
        relative = target.get("path") or f"{default}/{target['name']}"
        target["path"] = f"swift/OpenMedKit/{relative}"
    return root


def _aligned_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    nested = _fixture("nested")
    return _root_style(nested), nested


def _resolved(pins: dict[str, str]) -> dict[str, Any]:
    return {
        "pins": [
            {
                "identity": identity,
                "kind": "remoteSourceControl",
                "location": f"https://example.invalid/{identity}.git",
                "state": {"revision": "0" * 40, "version": version},
            }
            for identity, version in pins.items()
        ],
        "version": 3,
    }


def test_pre_alignment_manifests_report_mlx_drift() -> None:
    diffs = drift.compare_manifests(_fixture("root"), _fixture("nested"))

    assert diffs == [
        drift.SectionDiff(
            "dependencies",
            ("mlx-swift exact 0.31.3",),
            (
                "mlx-swift exact 0.31.6",
                "mlx-swift-lm revision 42f08a872075fd07f9f1f40ec1a5e191e6aad86e",
            ),
        ),
        drift.SectionDiff(
            "targets",
            (),
            (
                "OpenMedKit depends product:MLXLLM@mlx-swift-lm when "
                "platforms=ios,macos",
                "OpenMedKit depends product:MLXLMCommon@mlx-swift-lm when "
                "platforms=ios,macos",
            ),
        ),
    ]


def test_pre_alignment_fixtures_differ_only_in_mlx_and_source_paths() -> None:
    root = _fixture("root")
    nested = _fixture("nested")

    root_targets = {target["name"]: target for target in root["targets"]}
    nested_targets = {target["name"]: target for target in nested["targets"]}
    assert set(root_targets) == set(nested_targets)
    assert all(
        target["path"].startswith("swift/OpenMedKit/")
        for target in root_targets.values()
    )
    # Nested targets may rely on SwiftPM's default Sources/ and Tests/ layout.
    assert "path" not in nested_targets["OpenMedKit"]
    assert "path" not in nested_targets["OpenMedKitTests"]


def test_aligned_manifests_pass_with_only_source_path_differences() -> None:
    root, nested = _aligned_pair()

    assert drift.compare_manifests(root, nested) == []
    assert "differ only in source paths" in drift.format_report([])


def test_one_manifest_dependency_change_reports_identity_and_requirement_only() -> None:
    root, nested = _aligned_pair()
    _dependency(nested, "mlx-swift")["requirement"] = {"exact": ["0.33.0"]}

    diffs = drift.compare_manifests(root, nested)

    assert diffs == [
        drift.SectionDiff(
            "dependencies",
            ("mlx-swift exact 0.31.6",),
            ("mlx-swift exact 0.33.0",),
        )
    ]
    report = drift.format_report(diffs).splitlines()
    assert report[-3:] == [
        "[dependencies]",
        "- mlx-swift exact 0.31.6",
        "+ mlx-swift exact 0.33.0",
    ]


def test_dependency_declared_in_only_one_manifest_is_reported_by_identity() -> None:
    root, nested = _aligned_pair()
    extra = copy.deepcopy(root["dependencies"][0])
    extra["sourceControl"][0]["identity"] = "swift-synthetic"
    extra["sourceControl"][0]["location"] = {
        "remote": [{"urlString": "https://example.invalid/swift-synthetic.git"}]
    }
    extra["sourceControl"][0]["requirement"] = {
        "range": [{"lowerBound": "1.2.0", "upperBound": "2.0.0"}]
    }
    root["dependencies"].append(extra)

    diffs = drift.compare_manifests(root, nested)

    assert diffs == [
        drift.SectionDiff("dependencies", ("swift-synthetic range 1.2.0..<2.0.0",), ())
    ]


def test_dependency_location_change_is_reported_without_requirements() -> None:
    root, nested = _aligned_pair()
    _dependency(nested, "zipfoundation")["location"] = {
        "remote": [{"urlString": "https://example.invalid/fork/ZIPFoundation.git"}]
    }

    diffs = drift.compare_manifests(root, nested)

    assert diffs == [
        drift.SectionDiff(
            "dependencies",
            (
                "zipfoundation location sourceControl "
                "https://github.com/weichsel/ZIPFoundation.git",
            ),
            (
                "zipfoundation location sourceControl "
                "https://example.invalid/fork/ZIPFoundation.git",
            ),
        )
    ]


def test_target_condition_drift_is_reported() -> None:
    root, nested = _aligned_pair()
    for dependency in _target(root, "OpenMedKit")["dependencies"]:
        product = dependency.get("product")
        if product and product[0] == "MLXLLM":
            product[3] = None

    diffs = drift.compare_manifests(root, nested)

    assert diffs == [
        drift.SectionDiff(
            "targets",
            ("OpenMedKit depends product:MLXLLM@mlx-swift-lm",),
            (
                "OpenMedKit depends product:MLXLLM@mlx-swift-lm when platforms=ios,macos",
            ),
        )
    ]


def test_resource_product_and_platform_drift_is_reported() -> None:
    root, nested = _aligned_pair()
    _target(nested, "OpenMedShareExtension")["resources"] = [
        {"path": "Resources/PrivacyInfo.xcprivacy", "rule": {"copy": {}}}
    ]
    nested["products"] = [
        product
        for product in nested["products"]
        if product["name"] != "OpenMedActionExtension"
    ]
    root["platforms"][0]["version"] = "18.0"

    sections = {diff.section: diff for diff in drift.compare_manifests(root, nested)}

    assert sections["package"].root_only == ("platform ios 18.0",)
    assert sections["package"].nested_only == ("platform ios 17.0",)
    assert sections["products"].root_only == (
        "OpenMedActionExtension library(automatic) targets=OpenMedActionExtension",
    )
    assert sections["targets"].nested_only == (
        "OpenMedShareExtension resource copy Resources/PrivacyInfo.xcprivacy",
    )


def test_root_target_outside_nested_package_is_drift() -> None:
    root, nested = _aligned_pair()
    _target(root, "OpenMedKit")["path"] = "Sources/OpenMedKit"

    diffs = drift.compare_manifests(root, nested)

    assert diffs == [
        drift.SectionDiff(
            "targets",
            ("OpenMedKit path ../../Sources/OpenMedKit",),
            ("OpenMedKit path Sources/OpenMedKit",),
        )
    ]


def test_resolved_pins_must_satisfy_agreed_requirements() -> None:
    root, nested = _aligned_pair()
    pins = {
        "argmax-oss-swift": "1.1.1",
        "mlx-swift": "0.31.6",
        "swift-transformers": "1.0.0",
        "zipfoundation": "0.9.20",
    }

    assert drift.check_resolved_pins(root, nested, _resolved(pins)) == [
        "mlx-swift-lm revision 42f08a872075fd07f9f1f40ec1a5e191e6aad86e pinned nothing"
    ]

    stale = dict(pins, **{"mlx-swift": "0.31.3", "zipfoundation": "1.0.0"})
    problems = drift.check_resolved_pins(root, nested, _resolved(stale))
    assert "mlx-swift exact 0.31.6 pinned 0.31.3" in problems
    assert "zipfoundation range 0.9.19..<1.0.0 pinned 1.0.0" in problems


def test_resolved_check_skips_dependencies_the_manifests_disagree_on() -> None:
    root = _fixture("root")
    nested = _fixture("nested")
    resolved = _resolved(
        {
            "argmax-oss-swift": "1.1.1",
            "mlx-swift": "0.31.3",
            "swift-transformers": "1.0.0",
            "zipfoundation": "0.9.20",
        }
    )

    assert drift.check_resolved_pins(root, nested, resolved) == []


def test_main_reports_drift_from_captured_dumps(tmp_path, capsys) -> None:
    resolved = tmp_path / "Package.resolved"
    resolved.write_text(json.dumps(_resolved({"mlx-swift": "0.31.3"})), "utf-8")

    code = drift.main(
        [
            "--root-dump",
            str(FIXTURES / "pre_alignment_root.dump.json"),
            "--nested-dump",
            str(FIXTURES / "pre_alignment_nested.dump.json"),
            "--resolved",
            str(resolved),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "[dependencies]\n- mlx-swift exact 0.31.3\n+ mlx-swift exact 0.31.6" in (
        captured.err
    )
    assert "Package.resolved is stale" in captured.err
    assert "! swift-transformers exact 1.0.0 pinned nothing" in captured.err
    assert str(tmp_path) not in captured.err
    assert "/synthetic" not in captured.err


def test_main_passes_for_aligned_dumps(tmp_path, capsys) -> None:
    root, nested = _aligned_pair()
    root_path = tmp_path / "root.json"
    nested_path = tmp_path / "nested.json"
    root_path.write_text(json.dumps(root), "utf-8")
    nested_path.write_text(json.dumps(nested), "utf-8")

    code = drift.main(
        ["--root-dump", str(root_path), "--nested-dump", str(nested_path)]
    )

    assert code == 0
    assert "Swift manifests agree" in capsys.readouterr().out


def test_main_uses_value_free_reason_codes_for_unreadable_input(
    tmp_path, capsys
) -> None:
    broken = tmp_path / "private-location" / "root.json"
    broken.parent.mkdir()
    broken.write_text("not json", "utf-8")

    code = drift.main(
        [
            "--root-dump",
            str(broken),
            "--nested-dump",
            str(FIXTURES / "pre_alignment_nested.dump.json"),
        ]
    )

    err = capsys.readouterr().err
    assert code == 2
    assert "dump_not_json role=root" in err
    assert "private-location" not in err


def test_dump_manifest_runs_swift_package_dump_package(tmp_path) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout='{"name": "X"}')

    assert drift.dump_manifest(tmp_path, role="root", runner=runner) == {"name": "X"}
    command, kwargs = calls[0]
    assert command == [
        "swift",
        "package",
        "--package-path",
        str(tmp_path),
        "dump-package",
    ]
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False


def test_dump_manifest_failure_hides_swift_diagnostics(tmp_path) -> None:
    def failing(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr=f"error: {tmp_path}/Package.swift:3"
        )

    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])

    with pytest.raises(drift.ManifestDumpError) as failed:
        drift.dump_manifest(tmp_path, role="nested", runner=failing)
    assert str(failed.value) == "dump_package_failed role=nested exit=1"

    with pytest.raises(drift.ManifestDumpError) as unavailable:
        drift.dump_manifest(tmp_path, role="root", runner=missing)
    assert str(unavailable.value) == "swift_unavailable role=root"


def test_swift_workflow_runs_drift_check_and_versioned_consumer() -> None:
    workflow = yaml.load(SWIFT_WORKFLOW.read_text("utf-8"), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]
    macos_steps = [step.get("run", "") for step in jobs["build-and-test"]["steps"]]
    ios_steps = [step.get("run", "") for step in jobs["build-ios"]["steps"]]

    drift_step = macos_steps.index(
        "python3 scripts/release/check_swift_manifest_drift.py"
    )
    assert drift_step < macos_steps.index("swift package resolve")
    assert "scripts/check_swift_root_consumer.sh macos" in macos_steps
    assert "scripts/check_swift_root_consumer.sh ios" in ios_steps
    for event in ("push", "pull_request"):
        paths = workflow["on"][event]["paths"]
        assert "scripts/release/check_swift_manifest_drift.py" in paths
        assert "scripts/check_swift_root_consumer.sh" in paths
        assert "swift/**" in paths


def test_versioned_consumer_requires_the_root_by_version_and_uses_gated_symbols():
    script = CONSUMER_SCRIPT.read_text("utf-8")
    source = CONSUMER_SOURCE.read_text("utf-8")

    assert '.package(url: "file://$SNAPSHOT", exact: "$SNAPSHOT_VERSION")' in script
    assert '.product(name: "OpenMedKit", package: "openmed")' in script
    assert "#if" not in source
    for symbol in (
        "OpenMedMaple(modelDirectoryURL:",
        "maple.brief(",
        "OpenMedMapleConfiguration",
        "OpenMedVisionLanguageModel.load(modelDirectory:",
        "OpenMedVisionLanguageGeneration",
    ):
        assert symbol in source


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swift") is None,
    reason="needs the macOS Swift toolchain; CI runs this in swift-test.yml",
)
def test_repository_manifests_agree() -> None:
    assert drift.main([]) == 0
