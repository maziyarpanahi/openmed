"""Installed-package consumer verifier, fixtures, workflow and migration docs."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "verify_installed_consumers.py"
CONSUMERS = ROOT / "tests" / "consumers"
CONTRACT = CONSUMERS / "fixtures" / "installed_consumer_contract_v1.json"
WORKFLOW = ROOT / ".github" / "workflows" / "installed-consumers.yml"
GUIDE = ROOT / "docs" / "migration" / "openmedkit-installed-packages.md"
GUIDE_NAV_PATH = "migration/openmedkit-installed-packages.md"
FULL_SHA = re.compile(r"@[0-9a-f]{40}(\s|$)")

spec = importlib.util.spec_from_file_location("verify_installed_consumers", SCRIPT)
assert spec is not None and spec.loader is not None
consumers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = consumers
spec.loader.exec_module(consumers)


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _utf16(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def _which(*available: str):
    return lambda name: f"/usr/bin/{name}" if name in available else None


def _context(tmp_path: Path, runner, **overrides):
    values = {
        "root": ROOT,
        "work_dir": tmp_path / "work",
        "runner": runner,
        "which": _which("swift", "git", "java", "node", "npm"),
        "system": "Darwin",
        "environ": {},
    }
    values.update(overrides)
    values["work_dir"].mkdir(parents=True, exist_ok=True)
    return consumers.Context(**values)


def _arg_value(argv: tuple[str, ...], prefix: str) -> str:
    return next(item.removeprefix(prefix) for item in argv if item.startswith(prefix))


def _write_aar(path: Path, *, omit: tuple[str, ...] = (), extra: tuple[str, ...] = ()):
    classes = io.BytesIO()
    with zipfile.ZipFile(classes, "w") as jar:
        for entry in consumers.REQUIRED_AAR_CLASSES_ENTRIES:
            if f"classes.jar!{entry}" not in omit:
                jar.writestr(entry, "{}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as aar:
        for entry in consumers.REQUIRED_AAR_ENTRIES:
            if entry in omit:
                continue
            aar.writestr(entry, classes.getvalue() if entry == "classes.jar" else "x")
        for entry in extra:
            aar.writestr(entry, "x")


# -- Synthetic contract ------------------------------------------------------


def test_contract_is_synthetic_and_offsets_are_internally_consistent() -> None:
    contract = _contract()
    assert contract["schema_version"] == 1
    assert contract["synthetic"] is True
    assert contract["offset_unit"] == "unicode_scalar"
    assert {case["id"] for case in contract["offset_cases"]} >= {
        "cjk-extension-b-astral",
        "emoji-zwj-prefix",
        "combining-marks",
    }
    for case in contract["offset_cases"]:
        text, start, end = case["text"], case["start"], case["end"]
        assert case["scalar_length"] == len(text), case["id"]
        assert case["utf16_length"] == _utf16(text, len(text)), case["id"]
        assert case["utf16_start"] == _utf16(text, start), case["id"]
        assert case["utf16_end"] == _utf16(text, end), case["id"]
        expected = text[:start] + case["replacement"] + text[end:]
        assert case["expected_redacted"] == expected, case["id"]
    # At least one case must make scalar and UTF-16 offsets disagree.
    assert any(case["utf16_end"] != case["end"] for case in contract["offset_cases"])


def test_contract_policy_actions_match_python_and_bundled_kit_profiles() -> None:
    sources = (
        ROOT / "openmed" / "core" / "policies",
        ROOT / "swift" / "OpenMedKit" / "Sources" / "OpenMedKit" / "Resources",
        ROOT / "android" / "openmedkit" / "src" / "main" / "assets",
    )
    for case in _contract()["policy_cases"]:
        for index, source in enumerate(sources):
            folder = source if index == 0 else source / "policies"
            profile = json.loads(
                (folder / f"{case['profile']}.json").read_text(encoding="utf-8")
            )
            for label, action in case["label_actions"].items():
                assert profile["actions"][label] == action, (case["profile"], label)


def test_contract_redaction_cases_follow_their_policy_actions() -> None:
    contract = _contract()
    actions = {
        case["profile"]: case["label_actions"] for case in contract["policy_cases"]
    }
    for case in contract["redaction_cases"]:
        redacted = case["text"]
        for entity in sorted(case["entities"], key=lambda item: -item["start"]):
            assert entity["utf16_start"] == _utf16(case["text"], entity["start"])
            if actions[case["profile"]][entity["label"]] == "mask":
                redacted = (
                    redacted[: entity["start"]]
                    + f"[{entity['label']}]"
                    + redacted[entity["end"] :]
                )
        assert redacted == case["expected_redacted"], case["id"]
    divergence = contract["kit_divergences"]["npm_web"]
    target = next(
        case
        for case in contract["redaction_cases"]
        if case["id"] == divergence["redaction_case"]
    )
    assert divergence["expected_redacted"] != target["expected_redacted"]


def test_contract_model_fixture_digests_match_contents() -> None:
    fixture = _contract()["model_fixture"]
    assert re.fullmatch(r"[0-9a-f]{40}", fixture["revision_sha"])
    for group in ("onnx_files", "mlx_legacy_files"):
        for entry in fixture[group]:
            payload = entry["content"].encode("utf-8")
            assert entry["size_bytes"] == len(payload), entry["path"]
            assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert ".." in fixture["unsafe_manifest_config_path"].split("/")


def test_contract_contains_no_identifier_shaped_values() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", text)
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    assert not re.search(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b", text)


# -- Consumer sources --------------------------------------------------------


def test_consumers_never_import_package_sources_by_repository_path() -> None:
    assert consumers.scan_consumer_sources(CONSUMERS) == []
    for name in ("swiftpm", "gradle", "npm"):
        assert (CONSUMERS / name).is_dir()
        assert not (CONSUMERS / name / consumers.STAGED_CONTRACT_NAME).exists()


def test_source_scan_flags_repository_relative_imports(tmp_path: Path) -> None:
    (tmp_path / "npm" / "test").mkdir(parents=True)
    (tmp_path / "npm" / "test" / "bad.test.mjs").write_text(
        'import "../../../js/openmedkit-web/src/index.ts";\n', encoding="utf-8"
    )
    (tmp_path / "gradle").mkdir()
    (tmp_path / "gradle" / "build.gradle.kts").write_text(
        'dependencies { implementation(project(":openmedkit")) }\n',
        encoding="utf-8",
    )
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "contract.json").write_text(
        '{"unsafe": "../../escape"}', encoding="utf-8"
    )
    assert consumers.scan_consumer_sources(tmp_path) == [
        "gradle/build.gradle.kts",
        "npm/test/bad.test.mjs",
    ]


def test_consumers_compile_the_documented_migration_targets() -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    for relative in (
        "swiftpm/Sources/OpenMedKitInstalledConsumer/Migration.swift",
        "gradle/app/src/main/kotlin/org/openmed/installedconsumer/Migration.kt",
        "npm/src/web-migration.ts",
        "npm/src/react-native-migration.ts",
    ):
        source = (CONSUMERS / relative).read_text(encoding="utf-8")
        assert Path(relative).name in guide
        if relative.endswith((".swift", ".kt")):
            assert "openmedkit-installed-packages.md" in source
    swift = (
        CONSUMERS / "swiftpm/Sources/OpenMedKitInstalledConsumer/Migration.swift"
    ).read_text(encoding="utf-8")
    assert "allowNetworkAccess: false" in swift
    kotlin = (
        CONSUMERS
        / "gradle/app/src/main/kotlin/org/openmed/installedconsumer/Migration.kt"
    ).read_text(encoding="utf-8")
    for symbol in (
        "OpenMedKit.fromDirectory",
        "ModelDownloader(context)",
        "ModelCache",
    ):
        assert symbol in kotlin


# -- Swift staging -----------------------------------------------------------


def _root_dump() -> str:
    """A ``dump-package`` description of the root manifest's declared paths.

    The real lane asks SwiftPM; this offline stand-in reads the explicit
    ``path:`` arguments, which every root target declares.
    """
    manifest = (ROOT / "Package.swift").read_text(encoding="utf-8")
    targets = [
        {
            "name": Path(path).name,
            "type": "test" if "/Tests/" in f"/{path}/" else "regular",
            "path": path,
        }
        for path in re.findall(r'\bpath:\s*"([^"]+)"', manifest)
    ]
    assert targets, "root manifest declares no target paths"
    return "warning: synthetic\n" + json.dumps(
        {"name": "OpenMedKit", "targets": targets}
    )


def test_swift_targets_from_dump_use_explicit_and_default_paths() -> None:
    dump = json.dumps(
        {
            "targets": [
                {"name": "Tool", "type": "executable", "path": None},
                {"name": "ToolTests", "type": "test"},
                {"name": "Lib", "type": "regular", "path": "Custom/Lib"},
            ]
        }
    )
    assert consumers.swift_targets_from_dump(dump) == [
        ("executable", "Tool", "Sources/Tool"),
        ("test", "ToolTests", "Tests/ToolTests"),
        ("regular", "Lib", "Custom/Lib"),
    ]
    for broken in ("not json", '{"targets": {}}', '{"targets": [{"type": "x"}]}'):
        with pytest.raises(consumers.LaneFailure, match="swift_manifest_unreadable"):
            consumers.swift_targets_from_dump(broken)


def test_stage_swift_package_copies_only_manifest_and_non_test_sources(
    tmp_path: Path,
) -> None:
    targets = consumers.swift_targets_from_dump(_root_dump())
    staged = consumers.stage_swift_package(ROOT, tmp_path / "OpenMedKit", targets)
    assert {"Package.swift", "Package.resolved", "LICENSE"} <= set(staged)
    assert "swift/OpenMedKit/Sources/OpenMedKit" in staged
    assert not any("Tests" in Path(item).parts for item in staged)
    package = tmp_path / "OpenMedKit"
    resources = package / "swift/OpenMedKit/Sources/OpenMedKit/Resources/policies"
    assert (resources / "hipaa_safe_harbor.json").is_file()
    assert not (package / "swift" / "OpenMedKit" / "Tests").exists()
    assert not (package / "tests").exists()
    assert consumers.tree_digest(package) == consumers.tree_digest(package)


def test_stage_swift_package_rejects_missing_or_escaping_target_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "Package.swift").write_text("// synthetic\n", encoding="utf-8")
    cases = {
        "swift_target_path_missing": ("regular", "Missing", "Sources/Missing"),
        "swift_target_path_outside_package": ("regular", "Up", "../Sources/Up"),
    }
    for reason, target in cases.items():
        with pytest.raises(consumers.LaneFailure, match=reason):
            consumers.stage_swift_package(root, tmp_path / reason, [target])


def test_rewrite_swift_dependency_pins_the_staged_git_tag() -> None:
    manifest = (CONSUMERS / "swiftpm" / "Package.swift").read_text(encoding="utf-8")
    assert '.package(path: "../OpenMedKit")' in manifest
    rewritten = consumers.rewrite_swift_dependency(
        manifest, "file:///stage/OpenMedKit", consumers.SWIFT_STAGED_TAG
    )
    assert '.package(path: "../OpenMedKit")' not in rewritten
    assert (
        '.package(url: "file:///stage/OpenMedKit", exact: "0.0.0-installed-consumer")'
        in rewritten
    )
    assert rewritten.count(consumers.SWIFT_DEPENDENCY_MARKER) == 1
    for broken in ("no marker\n", f"a {consumers.SWIFT_DEPENDENCY_MARKER}\n" * 2):
        with pytest.raises(consumers.LaneFailure, match="swift_dependency_marker"):
            consumers.rewrite_swift_dependency(broken, "file:///x", "1.0.0")


# -- Android helpers ---------------------------------------------------------


def test_android_versions_come_from_the_library_catalog(tmp_path: Path) -> None:
    versions = consumers.read_android_versions(
        ROOT / "android" / "gradle" / "libs.versions.toml"
    )
    assert set(versions) == set(consumers.ANDROID_VERSION_KEYS)
    properties = (CONSUMERS / "gradle" / "gradle.properties").read_text()
    for key in consumers.ANDROID_VERSION_KEYS:
        assert re.search(rf"(?m)^{key}=\S+$", properties), key
    broken = tmp_path / "libs.versions.toml"
    broken.write_text('[versions]\nagp = "7.4.2"\n', encoding="utf-8")
    with pytest.raises(consumers.LaneFailure, match="android_version_missing"):
        consumers.read_android_versions(broken)


def test_inspect_aar_reports_missing_resources_and_fixture_leaks(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "complete.aar"
    _write_aar(complete)
    assert consumers.inspect_aar(complete) == []

    broken = tmp_path / "broken.aar"
    _write_aar(
        broken,
        omit=(
            "assets/openmed_model_catalog.jsonl",
            "classes.jar!policies/hipaa_safe_harbor.json",
        ),
        extra=("assets/fixtures/sample.json",),
    )
    assert consumers.inspect_aar(broken) == [
        "assets/openmed_model_catalog.jsonl",
        "classes.jar!policies/hipaa_safe_harbor.json",
        "assets/fixtures/sample.json",
    ]


# -- Output parsers ----------------------------------------------------------


def test_parsers_keep_only_value_free_check_ids(tmp_path: Path) -> None:
    tap = (
        "TAP version 13\n"
        "# Subtest: resolves openmed\n"
        "    ok 1 - nested subtest\n"
        "ok 1 - resolves openmed\n"
        "not ok 2 - redacts the session\n"
        "ok 3 - snapshot # SKIP not staged\n"
    )
    assert consumers.parse_tap(tap) == [
        {"id": "resolves openmed", "status": "passed"},
        {"id": "redacts the session", "status": "failed"},
        {"id": "snapshot", "status": "skipped"},
    ]
    swift = 'Build complete!\n{"checks": [{"id": "a", "status": "pass"},' + (
        ' {"id": "b", "status": "fail", "code": "x"}], "lane": "swiftpm"}\n'
    )
    assert consumers.parse_swift_report(swift) == [
        {"id": "a", "status": "passed"},
        {"id": "b", "status": "failed"},
    ]
    assert consumers.parse_swift_report("no json") == []
    (tmp_path / "TEST-x.xml").write_text(
        '<testsuite><testcase classname="org.openmed.A" name="ok"/>'
        '<testcase classname="org.openmed.A" name="bad"><failure message="m"/>'
        "</testcase></testsuite>",
        encoding="utf-8",
    )
    assert consumers.parse_junit(tmp_path) == [
        {"id": "A.ok", "status": "passed"},
        {"id": "A.bad", "status": "failed"},
    ]


# -- Lane orchestration ------------------------------------------------------


@pytest.mark.parametrize(
    ("lane", "overrides", "reason"),
    [
        ("swiftpm", {"system": "Linux"}, "platform_unsupported"),
        ("swiftpm", {"which": _which("git")}, "toolchain_missing:swift"),
        ("gradle", {"which": _which()}, "toolchain_missing:java"),
        ("gradle", {"environ": {}}, "toolchain_missing:android_sdk"),
        ("npm-web", {"which": _which("npm")}, "toolchain_missing:node"),
        ("npm-react-native", {"which": _which()}, "toolchain_missing:node"),
    ],
)
def test_missing_toolchains_skip_without_qualifying(
    tmp_path: Path, monkeypatch, lane: str, overrides: dict, reason: str
) -> None:
    if lane == "gradle" and "environ" in overrides:
        monkeypatch.setattr(consumers, "android_sdk_configured", lambda ctx: False)

    def runner(step, log):
        raise AssertionError("a skipped lane must not run commands")

    result = consumers.run_lane(_context(tmp_path, runner, **overrides), lane)
    assert (result.status, result.reason, result.qualified) == (
        "skipped",
        reason,
        False,
    )
    assert consumers.exit_code([result], require=False) == 0
    assert consumers.exit_code([result], require=True) == 1
    report = consumers.build_report([result])
    assert report["qualified_lanes"] == []
    assert report["publication"] is False


def test_swiftpm_lane_consumes_a_content_addressed_git_tag(tmp_path: Path) -> None:
    seen: list[tuple[str, tuple[str, ...]]] = []

    def runner(step, log):
        seen.append((step.step_id, step.argv))
        if step.step_id == "dump_package":
            return consumers.StepOutcome(0, _root_dump())
        if step.step_id == "swift_run":
            checks = [{"id": "policy.label_actions", "status": "pass", "code": "ok"}]
            return consumers.StepOutcome(0, json.dumps({"checks": checks}))
        return consumers.StepOutcome(0, "")

    ctx = _context(tmp_path, runner)
    result = consumers.run_lane(ctx, "swiftpm")

    assert result.status == "passed", result.reason
    assert [step["id"] for step in result.steps] == [
        "dump_package",
        "stage_package",
        "stage_consumer",
        "git_init",
        "git_add",
        "git_commit",
        "git_tag",
        "pin_git_tag",
        "swift_build",
        "swift_run",
    ]
    lane_dir = ctx.work_dir / "swiftpm"
    tagged = [path for path in lane_dir.glob("git-*/OpenMedKit") if path.is_dir()]
    assert len(tagged) == 1
    assert not (lane_dir / "OpenMedKit").exists()
    manifest = (lane_dir / "consumer" / "Package.swift").read_text(encoding="utf-8")
    assert f'.package(url: "{tagged[0].as_uri()}", exact: ' in manifest
    assert (lane_dir / "consumer" / consumers.STAGED_CONTRACT_NAME).is_file()
    git_steps = [argv for step_id, argv in seen if step_id.startswith("git_")]
    assert all(argv[0] == "/usr/bin/git" for argv in git_steps)
    report = json.dumps(consumers.build_report([result]))
    assert str(tmp_path) not in report and str(ROOT) not in report


def test_swiftpm_path_mode_and_failed_checks_fail_the_lane(tmp_path: Path) -> None:
    def runner(step, log):
        if step.step_id == "dump_package":
            return consumers.StepOutcome(0, _root_dump())
        if step.step_id == "swift_run":
            checks = [{"id": "offsets", "status": "fail", "code": "case:utf16"}]
            return consumers.StepOutcome(1, json.dumps({"checks": checks}))
        return consumers.StepOutcome(0, "")

    ctx = _context(tmp_path, runner, swift_source="path")
    result = consumers.run_lane(ctx, "swiftpm")

    assert (result.status, result.reason) == ("failed", "consumer_checks_failed")
    assert (ctx.work_dir / "swiftpm" / "OpenMedKit" / "Package.swift").is_file()
    manifest = (ctx.work_dir / "swiftpm" / "consumer" / "Package.swift").read_text()
    assert '.package(path: "../OpenMedKit")' in manifest
    assert not any(step["id"].startswith("git_") for step in result.steps)
    assert consumers.exit_code([result], require=False) == 1


def test_gradle_lane_publishes_to_an_isolated_repository(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    calls: dict[str, tuple[str, ...]] = {}

    def runner(step, log):
        calls[step.step_id] = step.argv
        if step.step_id == "publish_aar":
            repo = Path(_arg_value(step.argv, "-Dmaven.repo.local="))
            version = consumers.ANDROID_CONSUMER_VERSION
            _write_aar(
                repo / "org/openmed/openmedkit" / version / f"openmedkit-{version}.aar"
            )
        if step.step_id == "consumer_verify":
            results = step.cwd / "app/build/test-results/testDebugUnitTest"
            results.mkdir(parents=True)
            (results / "TEST-a.xml").write_text(
                '<testsuite><testcase classname="x.InstalledPackageTest" '
                'name="offsets"/></testsuite>',
                encoding="utf-8",
            )
        return consumers.StepOutcome(0, "")

    ctx = _context(
        tmp_path, runner, environ={"ANDROID_HOME": str(sdk)}, gradle_offline=True
    )
    result = consumers.run_lane(ctx, "gradle")

    assert result.status == "passed", result.reason
    publish, verify = calls["publish_aar"], calls["consumer_verify"]
    repo = _arg_value(publish, "-Dmaven.repo.local=")
    assert Path(repo).is_relative_to(ctx.work_dir)
    assert _arg_value(verify, "-Dmaven.repo.local=") == repo
    assert "--offline" in publish and "--offline" in verify
    assert "verifyInstalledConsumer" in verify
    expected = consumers.read_android_versions(
        ROOT / "android/gradle/libs.versions.toml"
    )
    for key, value in expected.items():
        assert f"-P{key}={value}" in verify
    assert f"-PopenmedkitVersion={consumers.ANDROID_CONSUMER_VERSION}" in verify
    assert result.artifacts[0]["kind"] == "android-aar"
    assert re.fullmatch(r"[0-9a-f]{64}", result.artifacts[0]["sha256"])
    assert result.checks == [{"id": "InstalledPackageTest.offsets", "status": "passed"}]


def test_gradle_lane_fails_on_an_incomplete_aar(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()

    def runner(step, log):
        repo = Path(_arg_value(step.argv, "-Dmaven.repo.local="))
        version = consumers.ANDROID_CONSUMER_VERSION
        _write_aar(
            repo / "org/openmed/openmedkit" / version / f"openmedkit-{version}.aar",
            omit=("assets/openmed_model_catalog.jsonl",),
        )
        return consumers.StepOutcome(0, "")

    ctx = _context(tmp_path, runner, environ={"ANDROID_HOME": str(sdk)})
    result = consumers.run_lane(ctx, "gradle")
    assert (result.status, result.reason) == ("failed", "aar_incomplete")


def _fake_js_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    shutil.copytree(CONSUMERS / "npm", root / consumers.CONSUMERS_DIR / "npm")
    (root / consumers.CONTRACT_PATH.parent).mkdir(parents=True)
    shutil.copy2(CONTRACT, root / consumers.CONTRACT_PATH)
    snapshot = root / consumers.WEB_API_SNAPSHOT
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text('{"exports": []}', encoding="utf-8")
    for package in (consumers.WEB_PACKAGE_DIR, consumers.REACT_NATIVE_PACKAGE_DIR):
        (root / package).mkdir(parents=True)
        (root / package / "package.json").write_text("{}", encoding="utf-8")
    return root


@pytest.mark.parametrize("lane", ["npm-web", "npm-react-native"])
def test_npm_lanes_install_the_packed_tarball_into_a_staged_consumer(
    tmp_path: Path, lane: str
) -> None:
    root = _fake_js_root(tmp_path)
    seen: list[tuple[str, Path, tuple[str, ...]]] = []

    def runner(step, log):
        seen.append((step.step_id, step.cwd, step.argv))
        if step.step_id.startswith("npm_ci:"):
            for tool in (("typescript", "bin", "tsc"), ("tsx", "dist", "cli.mjs")):
                path = step.cwd / "node_modules" / Path(*tool)
                path.parent.mkdir(parents=True)
                path.write_text("", encoding="utf-8")
        if step.step_id == "npm_pack":
            name = (
                "openmed-openmedkit-react-native-0.1.0.tgz"
                if lane == "npm-react-native"
                else "openmed-3.2.0.tgz"
            )
            Path(step.argv[-1], name).write_bytes(b"synthetic tarball")
        if step.step_id == "node_test":
            return consumers.StepOutcome(0, "ok 1 - resolves the install\n")
        return consumers.StepOutcome(0, "")

    ctx = _context(tmp_path, runner, root=root, install_js_deps=True)
    result = consumers.run_lane(ctx, lane)

    assert result.status == "passed", result.reason
    steps = [step_id for step_id, _cwd, _argv in seen]
    assert steps[1:] == ["npm_pack", "npm_install", "node_test", "typecheck_migration"]
    consumer = ctx.work_dir / lane / "consumer"
    install = next(argv for step_id, _cwd, argv in seen if step_id == "npm_install")
    assert "--offline" in install and install[-1].endswith(".tgz")
    assert all(
        cwd == consumer
        for step_id, cwd, _argv in seen
        if step_id in {"npm_install", "node_test", "typecheck_migration"}
    )
    tests = (
        consumers.REACT_NATIVE_TESTS
        if lane == "npm-react-native"
        else (consumers.WEB_TESTS)
    )
    node_test = next(argv for step_id, _cwd, argv in seen if step_id == "node_test")
    assert node_test[-len(tests) :] == tests
    assert (consumer / "public-api.json").exists() is (lane == "npm-web")
    assert result.checks[-1] == {
        "id": "typescript.migration_compiles",
        "status": "passed",
    }
    assert result.artifacts[0]["name"].endswith(".tgz")


def test_npm_lane_requires_dependencies_unless_install_is_allowed(
    tmp_path: Path,
) -> None:
    root = _fake_js_root(tmp_path)

    def runner(step, log):
        raise AssertionError("no command may run without dependencies")

    result = consumers.run_lane(_context(tmp_path, runner, root=root), "npm-web")
    assert (result.status, result.reason) == ("failed", "js_dependencies_missing")


def test_main_refuses_a_work_dir_inside_the_checkout(capsys) -> None:
    def runner(step, log):
        raise AssertionError("no lane may run")

    code = consumers.main(
        ["--lane", "npm-web", "--work-dir", str(ROOT / "build" / "consumers")],
        runner=runner,
    )
    assert code == 2
    assert "outside the checkout" in capsys.readouterr().err


# -- CI and documentation ----------------------------------------------------


def test_workflow_runs_each_lane_on_its_platform_with_pinned_actions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    assert workflow["permissions"] == {"contents": "read"}
    triggers = workflow.get("on") or workflow.get(True)
    assert "tests/consumers/**" in triggers["pull_request"]["paths"]
    assert (
        "scripts/release/verify_installed_consumers.py"
        in (triggers["pull_request"]["paths"])
    )
    expected = {
        "swiftpm": ("macos-", ["--lane swiftpm"]),
        "gradle": ("ubuntu-", ["--lane gradle"]),
        "npm": ("ubuntu-", ["--lane npm-web", "--lane npm-react-native"]),
    }
    for job_id, (runner, lanes) in expected.items():
        job = workflow["jobs"][job_id]
        assert job["runs-on"].startswith(runner), job_id
        commands = " ".join(step.get("run", "") for step in job["steps"])
        assert "verify_installed_consumers.py" in commands
        assert "--require" in commands
        assert "publish" not in commands.replace("Publication", "")
        for lane in lanes:
            assert lane in commands, (job_id, lane)
        for step in job["steps"]:
            if "uses" in step:
                assert FULL_SHA.search(step["uses"] + " "), step["uses"]


def test_migration_guide_is_published_and_names_typed_incompatibilities() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    publication = (ROOT / "docs/brand/system/publication.yml").read_text("utf-8")
    assert f": {GUIDE_NAV_PATH}" in mkdocs
    assert f"- {GUIDE_NAV_PATH}" in publication
    guide = GUIDE.read_text(encoding="utf-8")
    for symbol in (
        "OpenMedModelStoreError.invalidManifestPath",
        "PolicyError.unknownProfile",
        "ModelIntegrityException",
        "UnsupportedOperationException",
        "IllegalArgumentException",
        "OpenMedKitRN is not installed in the React Native runtime",
        "kotlinx-coroutines-core",
        "slf4j-api",
        "`OTHER`",
        "scripts/release/verify_installed_consumers.py",
    ):
        assert symbol in guide, symbol
