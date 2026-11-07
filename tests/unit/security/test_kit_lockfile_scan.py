"""OpenMedKit lockfile vulnerability scan: workflow, fixtures and gate behavior."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/vuln-scan.yml"
SETUP_TRIVY = ROOT / ".github/actions/setup-trivy/action.yml"
FIXTURES = ROOT / "tests/fixtures/security/kit-lockfiles"
NPM_FIXTURE = FIXTURES / "seeded-npm-package-lock.json"
GRADLE_FIXTURE = FIXTURES / "seeded-gradle-lockfile.txt"
ALLOWLIST = ROOT / "deploy/security/cve-allowlist.yaml"
# Each Trivy target of the kit scan, relative to the repository root.
KIT_SCAN_TARGETS = {
    "swiftpm-root": "Package.resolved",
    "swiftpm-demos": "swift",
    "js": "js",
    "clients": "clients",
}
LOCKFILE_NAMES = (
    "Package.resolved",
    "package-lock.json",
    "Cargo.lock",
    "gradle.lockfile",
    "yarn.lock",
    "pnpm-lock.yaml",
)
GENERATED_DIRECTORIES = {"node_modules", "build", "target", ".gradle", ".build"}


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gradle_lockfile_view = _load(
    "gradle_lockfile_view", "scripts/security/gradle_lockfile_view.py"
)
vulnerability_scan_gate = _load(
    "kit_vulnerability_scan_gate", "scripts/security/vulnerability_scan_gate.py"
)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _kit_steps() -> dict[str, str]:
    steps = _workflow()["jobs"]["kit-lockfile-scan"]["steps"]
    return {step.get("name", ""): step.get("run", "") for step in steps}


def _committed_kit_lockfiles() -> list[Path]:
    patterns = (
        "Package.resolved",
        "swift/**/Package.resolved",
        "js/**/package-lock.json",
        "js/**/Cargo.lock",
        "clients/**/package-lock.json",
        "android/**/gradle.lockfile",
    )
    found = {
        path
        for pattern in patterns
        for path in ROOT.glob(pattern)
        if not GENERATED_DIRECTORIES.intersection(path.relative_to(ROOT).parts)
    }
    return sorted(found)


def _kit_trivy_report(path: Path, vulnerabilities: list[dict]) -> Path:
    """Write a Trivy filesystem report shaped like a kit lockfile scan."""
    path.write_text(
        json.dumps(
            {
                "SchemaVersion": 2,
                "ArtifactName": "js",
                "ArtifactType": "filesystem",
                "Results": [
                    {
                        "Target": "openmedkit-web/package-lock.json",
                        "Class": "lang-pkgs",
                        "Type": "npm",
                        "Vulnerabilities": [
                            {
                                "Title": "Synthetic advisory title",
                                "Description": "Synthetic advisory description",
                                "PrimaryURL": "https://example.invalid/advisory",
                                "References": ["https://example.invalid/ref"],
                                **vulnerability,
                            }
                            for vulnerability in vulnerabilities
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _lodash_finding(**overrides: str) -> dict:
    return {
        "VulnerabilityID": "CVE-2021-23337",
        "PkgName": "lodash",
        "InstalledVersion": "4.17.15",
        "FixedVersion": "4.17.21",
        "Severity": "HIGH",
        **overrides,
    }


# Workflow structure ---------------------------------------------------------


def test_kit_scan_is_a_separate_job_sharing_the_pinned_trivy_setup() -> None:
    jobs = _workflow()["jobs"]
    workflow_text = WORKFLOW.read_text(encoding="utf-8")

    assert {"vulnerability-scan", "kit-lockfile-scan"} <= set(jobs)
    for job in ("vulnerability-scan", "kit-lockfile-scan"):
        uses = [step.get("uses") for step in jobs[job]["steps"]]
        assert "./.github/actions/setup-trivy" in uses, job
    # The scanner download and checksum live in one place only.
    assert "releases/download" not in workflow_text
    action = SETUP_TRIVY.read_text(encoding="utf-8")
    assert 'TRIVY_VERSION: "0.76.1"' in action
    assert "TRIVY_SHA256:" in action
    assert "sha256sum --check --status" in action


def test_kit_scan_covers_every_surface_with_the_lockfile_scanner_options() -> None:
    scan = _kit_steps()["Scan OpenMedKit lockfiles"]

    assert "--scanners vuln" in scan
    assert "--pkg-types library" in scan
    assert "--format json" in scan
    for name, target in KIT_SCAN_TARGETS.items():
        assert f"scan {name} {target}\n" in scan
    assert "gradle_lockfile_view.py" in scan
    assert "--source android" in scan
    assert 'scan android "$raw/android-release-runtime"' in scan
    # Raw scanner output, with advisory text and paths, never leaves the runner.
    assert 'raw="$RUNNER_TEMP/kit-lockfile-scans"' in scan


def test_kit_gate_uses_the_shared_threshold_and_allowlist() -> None:
    gate = _kit_steps()["Enforce OpenMedKit lockfile gate"]
    workflow = _workflow()

    for name in [*KIT_SCAN_TARGETS, "android"]:
        assert f'--report "$raw/trivy-{name}.json"' in gate
    assert "--allowlist deploy/security/cve-allowlist.yaml" in gate
    assert "--lock-report" not in gate
    assert "kit-lockfile-reports/kit-lockfile-scan-summary.json" in gate
    assert "VULN_SCAN_SEVERITY_THRESHOLD" in workflow["env"]
    assert "--threshold" not in gate


def test_kit_job_uploads_only_sanitized_gate_summaries() -> None:
    steps = _workflow()["jobs"]["kit-lockfile-scan"]["steps"]
    uploads = [step for step in steps if "upload-artifact" in step.get("uses", "")]
    runs = "\n".join(step.get("run", "") for step in steps)

    assert len(uploads) == 1
    assert uploads[0]["with"]["path"] == "kit-lockfile-reports/"
    assert uploads[0]["if"] == "always()"
    assert "sarif" not in runs
    for line in runs.splitlines():
        if (
            "--output" in line
            and "trivy" not in line
            and "kit-lockfile-reports" in line
        ):
            assert "summary" in line


def test_seeded_check_requires_the_gate_to_block_each_seeded_package() -> None:
    seeded = _kit_steps()["Verify kit scan rejects seeded vulnerable lockfiles"]

    assert '"$seeded/npm/package-lock.json"' in seeded
    assert '"$seeded/gradle-source/app/gradle.lockfile"' in seeded
    assert 'if [ "$status" -eq 0 ]; then' in seeded
    assert "for package in lodash org.apache.logging.log4j:log4j-core; do" in seeded
    assert '.source_kind == "filesystem"' in seeded
    assert '.package == "com.google.guava:guava"' in seeded


def test_workflow_triggers_on_kit_lockfile_and_gate_changes() -> None:
    on = _workflow()[True]  # PyYAML reads the bare ``on`` key as a boolean.
    expected = {
        ".github/actions/setup-trivy/**",
        "Package.resolved",
        "swift/**/Package.resolved",
        "js/**/package-lock.json",
        "js/**/Cargo.lock",
        "clients/**/package-lock.json",
        "android/**/gradle.lockfile",
        "scripts/security/gradle_lockfile_view.py",
        "tests/fixtures/security/kit-lockfiles/**",
        "tests/unit/security/test_kit_lockfile_scan.py",
    }

    for event in ("push", "pull_request"):
        assert expected <= set(on[event]["paths"]), event


# Coverage of committed lockfiles and isolation of fixtures -----------------


def test_every_committed_kit_lockfile_is_under_a_scan_target() -> None:
    lockfiles = _committed_kit_lockfiles()
    targets = [ROOT / target for target in KIT_SCAN_TARGETS.values()]
    targets.append(ROOT / "android")

    names = {path.name for path in lockfiles}
    # Negative control: discovery sees every ecosystem the issue covers.
    assert {"Package.resolved", "package-lock.json", "Cargo.lock"} <= names
    assert "gradle.lockfile" in names
    for lockfile in lockfiles:
        assert any(
            lockfile == target or target in lockfile.parents for target in targets
        ), lockfile.relative_to(ROOT)


def test_seeded_fixtures_stay_outside_every_real_scan_path() -> None:
    targets = [ROOT / target for target in KIT_SCAN_TARGETS.values()]
    targets += [ROOT / "android", ROOT / "uv.lock"]

    for fixture in (NPM_FIXTURE, GRADLE_FIXTURE):
        assert fixture.is_file()
        assert not any(target in fixture.parents for target in targets)
        # Names no scanner or dependency graph treats as a lockfile.
        assert fixture.name not in LOCKFILE_NAMES
        assert not fixture.name.endswith(("gradle.lockfile", ".lock"))
    assert sorted(path.name for path in FIXTURES.iterdir()) == sorted(
        [NPM_FIXTURE.name, GRADLE_FIXTURE.name]
    )


def test_seeded_npm_fixture_pins_a_known_vulnerable_production_dependency() -> None:
    lock = json.loads(NPM_FIXTURE.read_text(encoding="utf-8"))

    assert lock["lockfileVersion"] == 3
    root = lock["packages"][""]
    assert root["dependencies"] == {"lodash": "4.17.15"}
    entry = lock["packages"]["node_modules/lodash"]
    # lodash 4.17.15 is affected by CVE-2020-8203 and CVE-2021-23337 (HIGH).
    assert entry["version"] == "4.17.15"
    assert not entry.get("dev", False)
    assert "resolved" not in entry


def test_seeded_gradle_fixture_ships_a_known_vulnerable_entry() -> None:
    entries = gradle_lockfile_view.shipped_entries(
        GRADLE_FIXTURE.read_text(encoding="utf-8"), "releaseRuntimeClasspath"
    )

    # log4j-core 2.14.1 is affected by CVE-2021-44228 (CRITICAL).
    assert "org.apache.logging.log4j:log4j-core:2.14.1" in entries
    assert not any(entry.startswith("com.google.guava:") for entry in entries)


# Gradle release runtime view ------------------------------------------------


def _write_lockfile(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# This is a Gradle generated file for dependency locking.\n"
    path.write_text(header + "".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def test_view_keeps_only_release_runtime_entries_at_the_same_relative_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "android"
    _write_lockfile(
        source / "library/gradle.lockfile",
        [
            "com.example:shipped:1.0.0=debugRuntimeClasspath,releaseRuntimeClasspath",
            "com.example:compile-only:2.0.0=releaseCompileClasspath",
            "com.example:test-only:3.0.0=releaseUnitTestRuntimeClasspath",
            "com.example:tooling:4.0.0=kotlinCompilerClasspath",
            "com.example:compiler-plugin:1.4.8=kotlin-extension",
            "empty=androidApis,lintChecks",
        ],
    )
    _write_lockfile(source / "settings-gradle.lockfile", ["empty=incomingCatalog"])
    _write_lockfile(
        source / "library/build/intermediates/gradle.lockfile",
        ["com.example:generated:9.9.9=releaseRuntimeClasspath"],
    )

    summary = gradle_lockfile_view.write_view(
        source, tmp_path / "view", "releaseRuntimeClasspath"
    )

    assert summary == gradle_lockfile_view.ViewSummary(lockfiles=2, entries=1)
    view = (tmp_path / "view/library/gradle.lockfile").read_text(encoding="utf-8")
    assert "com.example:shipped:1.0.0=releaseRuntimeClasspath\n" in view
    for dropped in ("compile-only", "test-only", "tooling", "compiler-plugin"):
        assert f"com.example:{dropped}" not in view
    assert "empty=" not in view
    assert sorted(
        path.relative_to(tmp_path / "view").as_posix()
        for path in (tmp_path / "view").rglob("*")
        if path.is_file()
    ) == ["library/gradle.lockfile"]


def test_view_of_committed_android_lockfiles_includes_each_shipped_module(
    tmp_path: Path,
) -> None:
    summary = gradle_lockfile_view.write_view(
        ROOT / "android", tmp_path / "view", "releaseRuntimeClasspath"
    )
    library = (tmp_path / "view/openmedkit/gradle.lockfile").read_text(encoding="utf-8")

    assert summary.lockfiles >= 4
    assert summary.entries > 0
    assert "com.microsoft.onnxruntime:onnxruntime-android:" in library
    assert "org.robolectric:" not in library
    assert "junit:junit:" not in library
    for demo in ("OpenMedDemo", "OpenMedMapleDemo", "OpenMedScanDemo"):
        assert (tmp_path / f"view/{demo}/app/gradle.lockfile").is_file()


@pytest.mark.parametrize(
    ("lines", "code"),
    [
        ([], "no_gradle_lockfiles"),
        (["com.example:test:1.0=testRuntimeClasspath"], "no_shipped_entries"),
        (["/private/secret-path\tINJECTED"], "malformed_lockfile_entry"),
        (
            ["com.example:bad version:1=releaseRuntimeClasspath"],
            "malformed_lockfile_entry",
        ),
    ],
)
def test_view_fails_closed_with_value_free_codes(
    tmp_path: Path, capsys, lines: list[str], code: str
) -> None:
    source = tmp_path / "android"
    source.mkdir()
    if lines:
        _write_lockfile(source / "app/gradle.lockfile", lines)

    status = gradle_lockfile_view.main(
        ["--source", str(source), "--output", str(tmp_path / "view")]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.err == f"gradle lockfile view failed: {code}\n"
    assert "secret" not in captured.out + captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert not (tmp_path / "view").exists()


def test_view_refuses_overlapping_or_non_empty_output(tmp_path: Path) -> None:
    source = tmp_path / "android"
    _write_lockfile(
        source / "app/gradle.lockfile",
        ["com.example:shipped:1.0.0=releaseRuntimeClasspath"],
    )
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")

    for output, code in (
        (source / "view", "output_overlaps_source"),
        (occupied, "output_not_empty"),
    ):
        with pytest.raises(gradle_lockfile_view.ViewError, match=code):
            gradle_lockfile_view.write_view(source, output, "releaseRuntimeClasspath")
    with pytest.raises(gradle_lockfile_view.ViewError, match="invalid_configuration"):
        gradle_lockfile_view.write_view(source, tmp_path / "view", "../escape")


def test_view_cli_reports_counts_only(tmp_path: Path, capsys) -> None:
    source = tmp_path / "android"
    _write_lockfile(
        source / "app/gradle.lockfile",
        ["com.example:shipped:1.0.0=releaseRuntimeClasspath"],
    )

    status = gradle_lockfile_view.main(
        ["--source", str(source), "--output", str(tmp_path / "view")]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == (
        "gradle lockfile view written: 1 lockfiles, 1 shipped entries\n"
    )
    assert "com.example" not in captured.out


def test_view_cli_errors_do_not_echo_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as exited:
        gradle_lockfile_view.main(["--source", "/private/secret-path"])

    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == "gradle lockfile view failed: invalid_cli_arguments\n"


# Gate behavior for kit lockfile reports ------------------------------------


def test_seeded_high_npm_finding_fails_the_gate(tmp_path: Path, capsys) -> None:
    report = _kit_trivy_report(tmp_path / "trivy-js.json", [_lodash_finding()])
    output = tmp_path / "summary.json"

    status = vulnerability_scan_gate.main(
        [
            "--report",
            str(report),
            "--allowlist",
            str(ALLOWLIST),
            "--threshold",
            "HIGH",
            "--output",
            str(output),
            "--today",
            "2026-11-07",
        ]
    )

    assert status == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["blocking_findings"] == [
        {
            "vulnerability_id": "CVE-2021-23337",
            "severity": "HIGH",
            "package": "lodash",
            "installed_version": "4.17.15",
            "fixed_version": "4.17.21",
            "source_kind": "filesystem",
            "exposure": None,
        }
    ]
    assert (
        "CVE-2021-23337 HIGH lodash@4.17.15 fixed by 4.17.21" in capsys.readouterr().err
    )


def test_kit_reports_contain_package_names_versions_and_advisory_ids_only(
    tmp_path: Path, capsys
) -> None:
    findings = [
        _lodash_finding(),
        _lodash_finding(
            VulnerabilityID="GHSA-jfh8-c2jp-5v3q",
            PkgName="org.apache.logging.log4j:log4j-core",
            InstalledVersion="2.14.1",
            FixedVersion="2.15.0, 2.12.2",
            Severity="CRITICAL",
        ),
        _lodash_finding(
            VulnerabilityID="CVE-2026-90001",
            PkgName="github.com/apple/swift-nio",
            InstalledVersion="2.40.0",
            FixedVersion="",
        ),
        _lodash_finding(
            VulnerabilityID="CVE-2026-90002",
            PkgName="@tauri-apps/api",
            InstalledVersion="2.0.0",
            FixedVersion="2.0.1",
            Severity="MEDIUM",
        ),
    ]
    report = _kit_trivy_report(tmp_path / "trivy-kit.json", findings)
    output = tmp_path / "summary.json"

    assert (
        vulnerability_scan_gate.main(
            [
                "--report",
                str(report),
                "--allowlist",
                str(ALLOWLIST),
                "--threshold",
                "HIGH",
                "--output",
                str(output),
            ]
        )
        == 1
    )

    serialized = output.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    summary = json.loads(serialized)
    assert [item["package"] for item in summary["findings"]] == [
        "lodash",
        "org.apache.logging.log4j:log4j-core",
        "github.com/apple/swift-nio",
        "@tauri-apps/api",
    ]
    assert summary["findings"][1]["fixed_version"] == "2.15.0, 2.12.2"
    # MEDIUM is below the shared HIGH threshold and does not block.
    assert len(summary["blocking_findings"]) == 3
    for item in summary["findings"]:
        assert set(item) == {
            "vulnerability_id",
            "severity",
            "package",
            "installed_version",
            "fixed_version",
            "source_kind",
            "exposure",
        }
    for omitted in (
        "Synthetic advisory",
        "example.invalid",
        "openmedkit-web/package-lock.json",
        str(tmp_path),
    ):
        assert omitted not in serialized + captured.out + captured.err


@pytest.mark.parametrize(
    "package",
    [
        "https://user:token@example.invalid/owner/repo",
        "example.invalid//double-slash",
        "/private/secret-path",
        "name with spaces",
        "scope@inner",
    ],
)
def test_url_or_path_shaped_package_names_are_not_echoed(package: str) -> None:
    finding = vulnerability_scan_gate.Finding(
        vulnerability_id="CVE-2026-99999",
        severity="HIGH",
        package=package,
        installed_version="1.0.0",
        fixed_version="",
        target="package-lock.json",
        scanner="lang-pkgs",
        title="",
        primary_url="",
        report="report.json",
    )

    assert vulnerability_scan_gate.finding_metadata(finding)["package"] == "unknown"


def test_unfixed_kit_finding_obeys_the_shared_waiver_process(tmp_path: Path) -> None:
    report = _kit_trivy_report(
        tmp_path / "trivy-js.json", [_lodash_finding(FixedVersion="")]
    )
    waiver = vulnerability_scan_gate.AllowlistEntry(
        "CVE-2021-23337", vulnerability_scan_gate.dt.date(2099, 1, 1), "Synthetic"
    )

    allowed = vulnerability_scan_gate.evaluate_reports([report], [waiver], "HIGH")
    fixed = _kit_trivy_report(tmp_path / "fixed.json", [_lodash_finding()])
    blocked = vulnerability_scan_gate.evaluate_reports([fixed], [waiver], "HIGH")

    assert len(allowed.allowed_findings) == 1
    assert not allowed.blocking_findings
    # A fixed version keeps the finding blocking even with a matching waiver.
    assert len(blocked.blocking_findings) == 1
