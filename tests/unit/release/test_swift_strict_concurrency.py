"""OpenMedKit complete-concurrency-checking lane."""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_swift_strict_concurrency.py"
SWIFT_WORKFLOW = ROOT / ".github" / "workflows" / "swift-test.yml"
SWIFT_TESTS = ROOT / "swift" / "OpenMedKit" / "Tests"
SWIFT_SOURCES = ROOT / "swift" / "OpenMedKit" / "Sources"

spec = importlib.util.spec_from_file_location("check_swift_strict_concurrency", SCRIPT)
assert spec is not None
strict = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = strict
spec.loader.exec_module(strict)

SOURCE_FILE = ROOT / "swift/OpenMedKit/Sources/OpenMedKit/PostProcessing.swift"
TEST_FILE = ROOT / "swift/OpenMedKit/Tests/OpenMedKitTests/OpenMedMLXTests.swift"
DEPENDENCY_FILE = Path(
    "/tmp/runner/openmed-strict-concurrency.x/macos/checkouts/"
    "mlx-swift/Source/MLX/MLXArray.swift"
)


def _log(*lines: str) -> list[str]:
    return [line + "\n" for line in lines]


def test_first_party_warnings_and_errors_are_reported_relative_to_the_root() -> None:
    log = _log(
        f"{SOURCE_FILE}:225:16: warning: static property 'defaultPIIPatterns' is not "
        "concurrency-safe [#MutableGlobalVariable]",
        f"{TEST_FILE}:17:28: error: static property 'files' is not concurrency-safe",
        f"{SOURCE_FILE}:225:16: warning: static property 'defaultPIIPatterns' is not "
        "concurrency-safe [#MutableGlobalVariable]",
        f"{SOURCE_FILE}:226:5: note: add '@MainActor' to make it main actor-isolated",
    )

    diagnostics = strict.parse_diagnostics(log, root=ROOT)

    assert [item.render() for item in diagnostics] == [
        "swift/OpenMedKit/Sources/OpenMedKit/PostProcessing.swift:225:16: warning: "
        "static property 'defaultPIIPatterns' is not concurrency-safe "
        "[#MutableGlobalVariable]",
        "swift/OpenMedKit/Tests/OpenMedKitTests/OpenMedMLXTests.swift:17:28: error: "
        "static property 'files' is not concurrency-safe",
    ]
    assert all(str(ROOT) not in item.render() for item in diagnostics)


def test_dependency_and_non_swift_diagnostics_are_negative_controls() -> None:
    log = _log(
        f"{DEPENDENCY_FILE}:10:4: warning: sending 'array' risks causing data races",
        f"{ROOT}/.build/checkouts/swift-transformers/Sources/Hub/HubApi.swift:1:1: "
        "warning: non-Sendable type",
        "/tmp/mlx-swift/Source/Cmlx/mlx-generated/metal/utils.h:80:3: warning: "
        "inline variables are a C++17 extension",
        f"{ROOT}/swift/OpenMedKitScanDemo/Demo.swift:3:1: warning: unrelated demo",
        "Build complete! (12.0s)",
    )

    assert strict.parse_diagnostics(log, root=ROOT) == []


def test_messages_drop_repository_path_prefixes() -> None:
    other = ROOT / "swift/OpenMedKit/Sources/OpenMedKit/OpenMedKit.swift"
    log = _log(f"{SOURCE_FILE}:1:1: warning: conflicts with declaration in {other}")

    (diagnostic,) = strict.parse_diagnostics(log, root=ROOT)

    assert diagnostic.message == (
        "conflicts with declaration in swift/OpenMedKit/Sources/OpenMedKit/OpenMedKit.swift"
    )


def test_build_commands_enable_complete_checking_for_every_first_party_target(
    tmp_path: Path,
) -> None:
    macos, macos_cwd = strict.build_command("macos", tmp_path)
    ios, ios_cwd = strict.build_command("ios", tmp_path)

    assert macos[:2] == ["swift", "build"]
    assert "--build-tests" in macos
    assert macos[macos.index("--package-path") + 1] == str(ROOT)
    assert macos[macos.index("--scratch-path") + 1] == str(tmp_path)
    assert macos[macos.index("-Xswiftc") + 1] == "-strict-concurrency=complete"
    assert macos_cwd == ROOT

    assert ios[:2] == ["xcodebuild", "build"]
    assert ios[ios.index("-scheme") + 1] == "OpenMedKit-Package"
    assert ios[ios.index("-destination") + 1] == "generic/platform=iOS Simulator"
    assert ios[ios.index("-derivedDataPath") + 1] == str(tmp_path)
    assert "SWIFT_STRICT_CONCURRENCY=complete" in ios
    assert ios_cwd == ROOT / "swift" / "OpenMedKit"

    with pytest.raises(ValueError, match="unsupported_platform"):
        strict.build_command("watchos", tmp_path)


class _FakeBuild:
    def __init__(self, output: str, returncode: int = 0) -> None:
        self.output = output
        self.returncode = returncode
        self.commands: list[list[str]] = []

    def __call__(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[Any]:
        self.commands.append(command)
        kwargs["stdout"].write(self.output)
        return subprocess.CompletedProcess(command, self.returncode)


def test_clean_build_passes(tmp_path: Path) -> None:
    out = io.StringIO()
    build = _FakeBuild("Compiling OpenMedKit PostProcessing.swift\nBuild complete!\n")

    status = strict.check_platform(
        "macos", work_dir=tmp_path / "scratch", log_dir=tmp_path, runner=build, out=out
    )

    assert status == 0
    assert "macos: 0 first-party concurrency diagnostics" in out.getvalue()
    assert (tmp_path / "macos.log").is_file()


def test_first_party_diagnostic_fails_even_when_the_build_succeeds(
    tmp_path: Path,
) -> None:
    out = io.StringIO()
    build = _FakeBuild(f"{SOURCE_FILE}:225:16: warning: not concurrency-safe\n")

    status = strict.check_platform(
        "ios", work_dir=tmp_path / "dd", log_dir=tmp_path, runner=build, out=out
    )

    assert status == 1
    assert (
        "swift/OpenMedKit/Sources/OpenMedKit/PostProcessing.swift:225:16"
        in out.getvalue()
    )
    assert str(ROOT) not in out.getvalue()


def test_failed_build_without_first_party_diagnostic_reports_a_relative_tail(
    tmp_path: Path,
) -> None:
    out = io.StringIO()
    work_dir = tmp_path / "scratch"
    build = _FakeBuild(
        f"error: could not resolve {work_dir}/checkouts/mlx-swift\n"
        f"error: {ROOT}/Package.swift: manifest parse failed\n",
        returncode=1,
    )

    status = strict.check_platform(
        "macos", work_dir=work_dir, log_dir=tmp_path, runner=build, out=out
    )

    assert status == 2
    report = out.getvalue()
    assert "macos: build_failed exit=1" in report
    assert "error: could not resolve checkouts/mlx-swift" in report
    assert "error: Package.swift: manifest parse failed" in report
    assert str(ROOT) not in report
    assert str(work_dir) not in report


def test_missing_toolchain_is_a_controlled_refusal(tmp_path: Path) -> None:
    def missing(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        raise FileNotFoundError(command[0])

    out = io.StringIO()
    status = strict.check_platform(
        "ios", work_dir=tmp_path / "dd", log_dir=tmp_path, runner=missing, out=out
    )

    assert status == 2
    assert "ios: toolchain_unavailable tool=xcodebuild" in out.getvalue()


def test_main_builds_each_requested_platform_once_and_reports_the_worst_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    build = _FakeBuild(f"{SOURCE_FILE}:1:1: warning: not concurrency-safe\n")

    status = strict.main(
        ["--platform", "macos", "--platform", "ios", "--platform", "macos"],
        runner=build,
    )

    assert status == 1
    assert [command[0] for command in build.commands] == ["swift", "xcodebuild"]
    assert list(tmp_path.iterdir()) == []


def test_main_checks_an_existing_log(tmp_path: Path) -> None:
    clean = tmp_path / "clean.log"
    clean.write_text("Build complete!\n", encoding="utf-8")
    dirty = tmp_path / "dirty.log"
    dirty.write_text(f"{SOURCE_FILE}:1:1: warning: not concurrency-safe\n", "utf-8")

    assert strict.main(["--from-log", str(clean)]) == 0
    assert strict.main(["--from-log", str(dirty)]) == 1


def test_swift_workflow_runs_the_strict_concurrency_lane() -> None:
    workflow = yaml.load(SWIFT_WORKFLOW.read_text("utf-8"), Loader=yaml.BaseLoader)
    steps = [
        step.get("run", "") for step in workflow["jobs"]["strict-concurrency"]["steps"]
    ]

    assert workflow["jobs"]["strict-concurrency"]["runs-on"].startswith("macos")
    script = "python3 scripts/release/check_swift_strict_concurrency.py"
    assert f"{script} --platform macos" in steps
    assert f"{script} --platform ios" in steps
    sanitizer = next(step for step in steps if "--sanitize=thread" in step)
    assert "ConcurrencySafetyTests" in sanitizer
    assert "SpeechDecoderProgressConcurrencyTests" in sanitizer
    for event in ("push", "pull_request"):
        paths = workflow["on"][event]["paths"]
        assert "scripts/release/check_swift_strict_concurrency.py" in paths
        assert "swift/**" in paths


def _unchecked_sendable_types() -> set[str]:
    pattern = re.compile(
        r"(?:class|struct|enum)\s+(\w+)\s*:[^{]*@unchecked\s+Sendable", re.MULTILINE
    )
    found: set[str] = set()
    for path in SWIFT_SOURCES.rglob("*.swift"):
        found.update(pattern.findall(path.read_text("utf-8")))
    return found


def test_every_unchecked_sendable_source_type_has_a_concurrent_use_test() -> None:
    tests = "\n".join(
        path.read_text("utf-8")
        for path in (
            SWIFT_TESTS / "OpenMedKitTests" / "ConcurrencySafetyTests.swift",
            SWIFT_TESTS
            / "OpenMedSpeechTests"
            / "SpeechDecoderProgressConcurrencyTests.swift",
        )
    )
    # Private implementation details are reached through their public owner.
    covered_by_owner = {
        "GeneratedDetectorBox": "GeneratedTextGuard",
        "ThermalNotificationSubscription": "AppleSystemThermalSource",
    }

    unchecked = _unchecked_sendable_types()

    assert unchecked, "the scan should find the lock-backed runtime types"
    missing = sorted(
        name
        for name in unchecked
        if not re.search(rf"\b{covered_by_owner.get(name, name)}\b", tests)
    )
    assert missing == []
