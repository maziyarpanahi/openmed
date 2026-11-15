#!/usr/bin/env python3
"""Verify built OpenMedKit packages in consumer projects outside the checkout.

Each lane builds or packs one distributable, stages a consumer project from
``tests/consumers/`` into a work directory outside the source tree, installs the
artifact there and runs offline checks against a synthetic contract:

* ``swiftpm``: stages the root ``Package.swift`` with only its non-test target
  sources, tags it in a throwaway git repository (or uses a path dependency)
  and builds and runs an executable consumer.
* ``gradle``: publishes the OpenMedKit AAR with
  ``publishReleasePublicationToMavenLocal`` into an isolated Maven repository
  and builds a consumer app that resolves it from ``mavenLocal``.
* ``npm-web`` and ``npm-react-native``: install the ``npm pack`` tarballs into a
  fresh consumer and run Node tests plus a TypeScript migration compile check.

Nothing is published. Lanes whose toolchain is unavailable are reported as
``skipped`` and are never counted as qualified; ``--require`` turns a skip into a
failure. The JSON report holds lane ids, step ids, check ids, artifact names and
SHA-256 digests only: no paths, fixture text or command output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
CONSUMERS_DIR = Path("tests") / "consumers"
CONTRACT_PATH = CONSUMERS_DIR / "fixtures" / "installed_consumer_contract_v1.json"
WEB_API_SNAPSHOT = (
    Path("tests") / "web" / "__snapshots__" / "openmedkit-web-public-api.json"
)
STAGED_CONTRACT_NAME = "consumer_contract.json"

LANES = ("swiftpm", "gradle", "npm-web", "npm-react-native")
REPORT_SCHEMA = "openmed.installed_consumers"

SWIFT_PACKAGE_FILES = ("Package.swift", "Package.resolved", "LICENSE")
SWIFT_DEPENDENCY_MARKER = "// openmedkit-dependency"
SWIFT_CONSUMER_PRODUCT = "OpenMedKitInstalledConsumer"
SWIFT_STAGED_TAG = "0.0.0-installed-consumer"
STAGED_COMMIT_DATE = "2000-01-01T00:00:00+00:00"

ANDROID_CONSUMER_VERSION = "0.0.0-installed-consumer"
ANDROID_GROUP = "org.openmed"
ANDROID_VERSION_KEYS = {
    "consumerAgpVersion": "agp",
    "consumerKotlinVersion": "kotlin",
    "consumerCoroutinesVersion": "kotlinx-coroutines",
    "consumerSerializationVersion": "kotlinx-serialization-json",
    "consumerJunitVersion": "junit",
    "consumerRobolectricVersion": "robolectric",
    "consumerSlf4jVersion": "slf4j-android-compatible",
    "consumerErrorproneVersion": "errorprone-android-compatible",
}
REQUIRED_AAR_ENTRIES = (
    "AndroidManifest.xml",
    "classes.jar",
    "proguard.txt",
    "assets/openmed_model_catalog.jsonl",
    "assets/policies/hipaa_safe_harbor.json",
)
REQUIRED_AAR_CLASSES_ENTRIES = ("policies/hipaa_safe_harbor.json",)

WEB_PACKAGE_DIR = Path("js") / "openmedkit-web"
REACT_NATIVE_PACKAGE_DIR = Path("js") / "openmedkit-react-native"
WEB_TESTS = ("test/web-esm.test.mjs", "test/web-cjs.test.cjs")
REACT_NATIVE_TESTS = ("test/react-native.test.mts",)

# Consumer sources must not reach back into package sources. ``../OpenMedKit``
# (the staged sibling) is the only parent-relative reference a consumer uses.
FORBIDDEN_SOURCE_REFERENCES = (
    re.compile(r"\.\./\.\."),
    re.compile(r"swift/OpenMedKit/Sources"),
    re.compile(r"android/openmedkit/src"),
    re.compile(r"js/openmedkit-[a-z-]+/(src|dist)"),
    re.compile(r"project\(\s*\":openmedkit\""),
)
SOURCE_SUFFIXES = {
    ".swift",
    ".kt",
    ".kts",
    ".ts",
    ".mts",
    ".mjs",
    ".cjs",
    ".js",
    ".json",
    ".properties",
}
GENERATED_DIRECTORIES = frozenset(
    {".build", ".swiftpm", ".gradle", "build", "node_modules", "xcuserdata"}
)
STAGE_IGNORE = shutil.ignore_patterns(
    *GENERATED_DIRECTORIES,
    ".DS_Store",
    STAGED_CONTRACT_NAME,
    "public-api.json",
)
PACKAGE_IGNORE = shutil.ignore_patterns(".build", ".swiftpm", "xcuserdata", ".DS_Store")

TAP_LINE = re.compile(r"^(not ok|ok) \d+ - (.+?)(?: # (SKIP|TODO)\b.*)?$")


class LaneSkip(Exception):
    """A lane cannot run here; it is reported but never qualified."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LaneFailure(Exception):
    """A lane ran and failed with a value-free reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Step:
    """One external command run by a lane."""

    step_id: str
    argv: tuple[str, ...]
    cwd: Path
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class StepOutcome:
    """Exit status and captured standard output of a step."""

    returncode: int
    stdout: str = ""


Runner = Callable[[Step, Path], StepOutcome]


@dataclass
class LaneResult:
    """Outcome of one consumer lane, safe to serialize into the report."""

    lane: str
    status: str = "skipped"
    reason: str = "not_run"
    steps: list[dict[str, str]] = field(default_factory=list)
    checks: list[dict[str, str]] = field(default_factory=list)
    artifacts: list[dict[str, str]] = field(default_factory=list)

    @property
    def qualified(self) -> bool:
        """Return whether the lane actually ran and every check passed."""
        return self.status == "passed"

    def to_json(self) -> dict[str, object]:
        """Return the value-free report record for this lane."""
        return {
            "lane": self.lane,
            "status": self.status,
            "reason": self.reason,
            "qualified": self.qualified,
            "steps": list(self.steps),
            "checks": list(self.checks),
            "artifacts": list(self.artifacts),
        }


@dataclass
class Context:
    """Inputs shared by every lane."""

    root: Path
    work_dir: Path
    runner: Runner
    which: Callable[[str], str | None] = shutil.which
    system: str = field(default_factory=platform.system)
    environ: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    swift_source: str = "git"
    gradle_offline: bool = False
    install_js_deps: bool = False
    tsc: Path | None = None
    tsx: Path | None = None
    python: str = sys.executable


def subprocess_runner(step: Step, log_path: Path) -> StepOutcome:
    """Run ``step`` and append its combined output to the lane log.

    Args:
        step: Command to execute.
        log_path: Lane log file inside the work directory.

    Returns:
        The exit status and captured standard output.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **dict(step.env)}
    completed = subprocess.run(
        step.argv,
        cwd=step.cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"== step {step.step_id} (exit {completed.returncode})\n")
        log.write(completed.stdout or "")
    return StepOutcome(completed.returncode, completed.stdout or "")


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(directory: Path) -> str:
    """Return a SHA-256 over every file's relative path and content."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def json_object_from_output(output: str) -> dict[str, object] | None:
    """Return the outermost JSON object in command output, if any."""
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def swift_targets_from_dump(output: str) -> list[tuple[str, str, str]]:
    """Return ``(type, name, path)`` for targets in ``swift package dump-package``.

    SwiftPM evaluates the manifest, so no Swift source is parsed here. Targets
    without an explicit path use SwiftPM's default ``Sources/<name>`` or
    ``Tests/<name>`` layout.

    Args:
        output: Standard output of ``swift package dump-package``.

    Returns:
        Declared targets in manifest order.

    Raises:
        LaneFailure: When the output is not a package description.
    """
    payload = json_object_from_output(output)
    targets = payload.get("targets") if payload is not None else None
    if not isinstance(targets, list):
        raise LaneFailure("swift_manifest_unreadable")
    resolved: list[tuple[str, str, str]] = []
    for target in targets:
        if not isinstance(target, dict) or not isinstance(target.get("name"), str):
            raise LaneFailure("swift_manifest_unreadable")
        kind = str(target.get("type", "regular"))
        name = target["name"]
        default = f"{'Tests' if kind == 'test' else 'Sources'}/{name}"
        path = target.get("path")
        resolved.append((kind, name, path if isinstance(path, str) else default))
    return resolved


def stage_swift_package(
    root: Path, destination: Path, targets: Sequence[tuple[str, str, str]]
) -> list[str]:
    """Copy the root Swift package with only its non-test target sources.

    Args:
        root: Repository root holding ``Package.swift``.
        destination: Empty directory that receives the staged package.
        targets: ``(type, name, path)`` entries from ``swift_targets_from_dump``.

    Returns:
        Staged relative paths, sorted.

    Raises:
        LaneFailure: When the manifest or a declared target path is missing.
    """
    if not (root / "Package.swift").is_file():
        raise LaneFailure("swift_manifest_missing")
    destination.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for name in SWIFT_PACKAGE_FILES:
        source = root / name
        if source.is_file():
            shutil.copy2(source, destination / name)
            staged.append(name)
    for kind, _name, relative in targets:
        if kind == "test":
            continue
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise LaneFailure("swift_target_path_outside_package")
        source = root / relative
        if not source.is_dir():
            raise LaneFailure("swift_target_path_missing")
        shutil.copytree(source, destination / relative, ignore=PACKAGE_IGNORE)
        staged.append(relative)
    return sorted(staged)


def rewrite_swift_dependency(manifest: str, url: str, version: str) -> str:
    """Point the consumer manifest at a tagged git repository.

    Args:
        manifest: Consumer ``Package.swift`` source text.
        url: ``file://`` URL of the staged package repository.
        version: Exact semantic-version tag to depend on.

    Returns:
        The rewritten manifest.

    Raises:
        LaneFailure: When the dependency marker is absent or ambiguous.
    """
    lines = manifest.splitlines(keepends=True)
    marked = [
        index for index, line in enumerate(lines) if SWIFT_DEPENDENCY_MARKER in line
    ]
    if len(marked) != 1:
        raise LaneFailure("swift_dependency_marker")
    line = lines[marked[0]]
    indent = line[: len(line) - len(line.lstrip())]
    newline = "\n" if line.endswith("\n") else ""
    lines[marked[0]] = (
        f'{indent}.package(url: "{url}", exact: "{version}")  '
        f"{SWIFT_DEPENDENCY_MARKER}{newline}"
    )
    return "".join(lines)


def read_android_versions(catalog: Path) -> dict[str, str]:
    """Map consumer Gradle properties to versions from the Android catalog.

    Args:
        catalog: ``android/gradle/libs.versions.toml``.

    Returns:
        ``-P`` property names and their pinned versions.

    Raises:
        LaneFailure: When a required version is missing.
    """
    versions = tomllib.loads(catalog.read_text("utf-8")).get("versions", {})
    resolved: dict[str, str] = {}
    for prop, key in ANDROID_VERSION_KEYS.items():
        value = versions.get(key)
        if not isinstance(value, str) or not value:
            raise LaneFailure("android_version_missing")
        resolved[prop] = value
    return resolved


def inspect_aar(aar: Path) -> list[str]:
    """Return required AAR entries that the built archive is missing.

    Args:
        aar: Published OpenMedKit AAR.

    Returns:
        Missing entry names; empty when the archive is complete.
    """
    with zipfile.ZipFile(aar) as archive:
        names = set(archive.namelist())
        missing = [entry for entry in REQUIRED_AAR_ENTRIES if entry not in names]
        if "classes.jar" in names:
            with zipfile.ZipFile(archive.open("classes.jar")) as classes:
                class_names = set(classes.namelist())
            missing.extend(
                f"classes.jar!{entry}"
                for entry in REQUIRED_AAR_CLASSES_ENTRIES
                if entry not in class_names
            )
        missing.extend(
            name
            for name in sorted(names)
            if "fixtures/" in name or name.startswith("tests/")
        )
    return missing


def scan_consumer_sources(consumers: Path) -> list[str]:
    """Return consumer files that reference package sources by repository path.

    Args:
        consumers: ``tests/consumers`` directory.

    Returns:
        Relative file names with forbidden references, sorted.
    """
    violations: list[str] = []
    for path in sorted(consumers.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        parts = path.relative_to(consumers).parts
        # The synthetic contract is data: it deliberately carries an escaping
        # manifest path as a negative control.
        if parts[0] == "fixtures" or GENERATED_DIRECTORIES.intersection(parts):
            continue
        text = path.read_text("utf-8")
        if any(pattern.search(text) for pattern in FORBIDDEN_SOURCE_REFERENCES):
            violations.append(path.relative_to(consumers).as_posix())
    return violations


def parse_swift_report(stdout: str) -> list[dict[str, str]]:
    """Extract value-free checks from the Swift consumer's JSON report."""
    payload = json_object_from_output(stdout)
    checks = payload.get("checks") if payload is not None else None
    if not isinstance(checks, list):
        return []
    return [
        {
            "id": str(check.get("id", "")),
            "status": "passed" if check.get("status") == "pass" else "failed",
        }
        for check in checks
        if isinstance(check, dict)
    ]


def parse_tap(stdout: str) -> list[dict[str, str]]:
    """Extract top-level test results from Node's TAP reporter output."""
    checks: list[dict[str, str]] = []
    for line in stdout.splitlines():
        match = TAP_LINE.match(line)
        if match is None:
            continue
        verdict, name, directive = match.groups()
        status = "skipped" if directive else "passed" if verdict == "ok" else "failed"
        checks.append({"id": name.strip(), "status": status})
    return checks


def parse_junit(results_dir: Path) -> list[dict[str, str]]:
    """Extract test case results from Gradle JUnit XML reports."""
    checks: list[dict[str, str]] = []
    for report in sorted(results_dir.glob("TEST-*.xml")):
        tree = ElementTree.parse(report)
        for case in tree.getroot().iter("testcase"):
            suite = case.get("classname", "").rsplit(".", 1)[-1]
            status = "passed"
            if case.find("failure") is not None or case.find("error") is not None:
                status = "failed"
            elif case.find("skipped") is not None:
                status = "skipped"
            checks.append({"id": f"{suite}.{case.get('name', '')}", "status": status})
    return checks


def fresh_directory(path: Path) -> Path:
    """Return ``path`` as a new empty directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def stage_consumer(root: Path, name: str, destination: Path) -> Path:
    """Copy a consumer template and the synthetic contract outside the checkout.

    Args:
        root: Repository root.
        name: Consumer directory under ``tests/consumers``.
        destination: Directory that receives the consumer copy.

    Returns:
        The staged consumer directory.
    """
    shutil.copytree(root / CONSUMERS_DIR / name, destination, ignore=STAGE_IGNORE)
    shutil.copy2(root / CONTRACT_PATH, destination / STAGED_CONTRACT_NAME)
    return destination


def android_sdk_configured(ctx: Context) -> bool:
    """Return whether Gradle can locate an Android SDK."""
    for key in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        value = ctx.environ.get(key)
        if value and Path(value).is_dir():
            return True
    local_properties = ctx.root / "android" / "local.properties"
    if local_properties.is_file():
        return any(
            line.strip().startswith("sdk.dir=")
            for line in local_properties.read_text("utf-8").splitlines()
        )
    return False


def resolve_js_tool(explicit: Path | None, candidates: Iterable[Path]) -> Path | None:
    """Return an explicit or discovered JavaScript build tool entry point."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    return next((path for path in candidates if path.is_file()), None)


def single_tarball(directory: Path, pattern: str) -> Path:
    """Return the only tarball matching ``pattern`` in ``directory``."""
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise LaneFailure("tarball_not_unique")
    return matches[0]


class LaneRun:
    """Step bookkeeping for one lane."""

    def __init__(self, ctx: Context, result: LaneResult) -> None:
        self.ctx = ctx
        self.result = result
        self.log_path = ctx.work_dir / "logs" / f"{result.lane}.log"

    def note(self, step_id: str) -> None:
        """Record an in-process staging step that succeeded."""
        self.result.steps.append({"id": step_id, "status": "passed"})

    def step(
        self,
        step_id: str,
        argv: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
        *,
        check: bool = True,
    ) -> StepOutcome:
        """Run one external command and record its status.

        Raises:
            LaneFailure: When ``check`` is set and the command fails.
        """
        step = Step(step_id, tuple(argv), cwd, tuple(sorted((env or {}).items())))
        outcome = self.ctx.runner(step, self.log_path)
        status = "passed" if outcome.returncode == 0 else "failed"
        self.result.steps.append({"id": step_id, "status": status})
        if check and outcome.returncode != 0:
            raise LaneFailure(f"step_failed:{step_id}")
        return outcome

    def artifact(self, kind: str, path: Path) -> None:
        """Record a built artifact by file name and digest only."""
        self.result.artifacts.append(
            {"kind": kind, "name": path.name, "sha256": sha256_file(path)}
        )


def run_swiftpm(lane: LaneRun) -> None:
    """Stage the Swift package and run the SwiftPM consumer."""
    ctx = lane.ctx
    if ctx.system != "Darwin":
        raise LaneSkip("platform_unsupported")
    swift = ctx.which("swift")
    if swift is None:
        raise LaneSkip("toolchain_missing:swift")
    git = ctx.which("git")
    if ctx.swift_source == "git" and git is None:
        raise LaneSkip("toolchain_missing:git")

    lane_dir = fresh_directory(ctx.work_dir / "swiftpm")
    package_dir = lane_dir / "OpenMedKit"
    description = lane.step(
        "dump_package",
        (swift, "package", "dump-package", "--package-path", str(ctx.root)),
        ctx.root,
    )
    targets = swift_targets_from_dump(description.stdout)
    staged = stage_swift_package(ctx.root, package_dir, targets)
    if any("Tests" in Path(item).parts for item in staged):
        raise LaneFailure("swift_tests_staged")
    if ctx.swift_source == "git":
        # SwiftPM caches git dependencies by URL and does not move an existing
        # tag, so the repository URL is addressed by the staged content.
        addressed = lane_dir / f"git-{tree_digest(package_dir)[:16]}" / "OpenMedKit"
        addressed.parent.mkdir()
        package_dir = package_dir.rename(addressed)
    lane.note("stage_package")

    consumer = stage_consumer(ctx.root, "swiftpm", lane_dir / "consumer")
    resolved = ctx.root / "Package.resolved"
    if resolved.is_file():
        shutil.copy2(resolved, consumer / "Package.resolved")
    lane.note("stage_consumer")

    if ctx.swift_source == "git" and git is not None:
        git_env = {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            # Fixed dates make the tagged commit a function of the content.
            "GIT_AUTHOR_DATE": STAGED_COMMIT_DATE,
            "GIT_COMMITTER_DATE": STAGED_COMMIT_DATE,
        }
        identity = (
            "-c",
            "user.name=OpenMed installed consumer",
            "-c",
            "user.email=installed-consumer@openmed.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
        )
        lane.step("git_init", (git, "init", "-q"), package_dir, git_env)
        lane.step("git_add", (git, "add", "-A"), package_dir, git_env)
        lane.step(
            "git_commit",
            (git, *identity, "commit", "-q", "-m", "Stage OpenMedKit package"),
            package_dir,
            git_env,
        )
        lane.step(
            "git_tag", (git, *identity, "tag", SWIFT_STAGED_TAG), package_dir, git_env
        )
        manifest = consumer / "Package.swift"
        manifest.write_text(
            rewrite_swift_dependency(
                manifest.read_text("utf-8"), package_dir.as_uri(), SWIFT_STAGED_TAG
            ),
            "utf-8",
        )
        lane.note("pin_git_tag")

    lane.step(
        "swift_build", (swift, "build", "--package-path", str(consumer)), consumer
    )
    outcome = lane.step(
        "swift_run",
        (
            swift,
            "run",
            "--skip-build",
            "--package-path",
            str(consumer),
            SWIFT_CONSUMER_PRODUCT,
        ),
        consumer,
        check=False,
    )
    lane.result.checks = parse_swift_report(outcome.stdout)
    finish_checks(lane, outcome.returncode)


def run_gradle(lane: LaneRun) -> None:
    """Publish the AAR to an isolated Maven repository and run the Gradle consumer."""
    ctx = lane.ctx
    android_dir = ctx.root / "android"
    gradlew = android_dir / "gradlew"
    if not gradlew.is_file():
        raise LaneSkip("toolchain_missing:gradlew")
    if ctx.which("java") is None and not ctx.environ.get("JAVA_HOME"):
        raise LaneSkip("toolchain_missing:java")
    if not android_sdk_configured(ctx):
        raise LaneSkip("toolchain_missing:android_sdk")
    versions = read_android_versions(android_dir / "gradle" / "libs.versions.toml")

    lane_dir = fresh_directory(ctx.work_dir / "gradle")
    maven_repo = lane_dir / "m2"
    offline = ("--offline",) if ctx.gradle_offline else ()
    lane.step(
        "publish_aar",
        (
            str(gradlew),
            "-p",
            str(android_dir),
            ":openmedkit:publishReleasePublicationToMavenLocal",
            f"-Dmaven.repo.local={maven_repo}",
            f"-PopenmedAndroidGroup={ANDROID_GROUP}",
            f"-PopenmedAndroidVersion={ANDROID_CONSUMER_VERSION}",
            f"-PopenmedPython={ctx.python}",
            "--stacktrace",
            *offline,
        ),
        android_dir,
    )
    aar = (
        maven_repo.joinpath(*ANDROID_GROUP.split("."))
        / "openmedkit"
        / ANDROID_CONSUMER_VERSION
        / f"openmedkit-{ANDROID_CONSUMER_VERSION}.aar"
    )
    if not aar.is_file():
        raise LaneFailure("aar_missing")
    if inspect_aar(aar):
        raise LaneFailure("aar_incomplete")
    lane.artifact("android-aar", aar)
    lane.note("inspect_aar")

    consumer = stage_consumer(ctx.root, "gradle", lane_dir / "consumer")
    lane.note("stage_consumer")
    properties = {
        **versions,
        "openmedkitGroup": ANDROID_GROUP,
        "openmedkitVersion": ANDROID_CONSUMER_VERSION,
    }
    outcome = lane.step(
        "consumer_verify",
        (
            str(gradlew),
            "-p",
            str(consumer),
            "verifyInstalledConsumer",
            f"-Dmaven.repo.local={maven_repo}",
            *(f"-P{key}={value}" for key, value in sorted(properties.items())),
            "--stacktrace",
            *offline,
        ),
        consumer,
        check=False,
    )
    lane.result.checks = parse_junit(
        consumer / "app" / "build" / "test-results" / "testDebugUnitTest"
    )
    finish_checks(lane, outcome.returncode)


def ensure_js_dependencies(lane: LaneRun, package_dir: Path, npm: str) -> None:
    """Install a package's locked dev dependencies when allowed and missing.

    Raises:
        LaneFailure: When dependencies are missing and installing is not allowed.
    """
    if (package_dir / "node_modules").is_dir():
        return
    if not lane.ctx.install_js_deps:
        raise LaneFailure("js_dependencies_missing")
    lane.step(
        f"npm_ci:{package_dir.name}",
        (npm, "ci", "--no-audit", "--no-fund"),
        package_dir,
    )


def js_tool_candidates(ctx: Context, package_dir: Path, *parts: str) -> list[Path]:
    """Return dev-dependency locations of a JS tool, package first."""
    return [
        directory / "node_modules" / Path(*parts)
        for directory in (package_dir, ctx.root / WEB_PACKAGE_DIR)
    ]


def run_npm_lane(lane: LaneRun, *, react_native: bool) -> None:
    """Pack one npm package and run the consumer tests against the tarball."""
    ctx = lane.ctx
    node = ctx.which("node")
    npm = ctx.which("npm")
    if node is None or npm is None:
        raise LaneSkip("toolchain_missing:node")
    relative = REACT_NATIVE_PACKAGE_DIR if react_native else WEB_PACKAGE_DIR
    package_dir = ctx.root / relative
    if not (package_dir / "package.json").is_file():
        raise LaneSkip("package_missing")
    # The web tarball is built by its prepack script; the bridge ships sources
    # and only needs dev tools for the consumer checks.
    ensure_js_dependencies(lane, package_dir, npm)
    tsc = resolve_js_tool(
        ctx.tsc, js_tool_candidates(ctx, package_dir, "typescript", "bin", "tsc")
    )
    if tsc is None:
        raise LaneFailure("typescript_missing")
    tsx = resolve_js_tool(
        ctx.tsx, js_tool_candidates(ctx, package_dir, "tsx", "dist", "cli.mjs")
    )
    if react_native and tsx is None:
        raise LaneFailure("tsx_missing")

    lane_dir = fresh_directory(ctx.work_dir / lane.result.lane)
    packs = lane_dir / "packs"
    packs.mkdir()
    lane.step("npm_pack", (npm, "pack", "--pack-destination", str(packs)), package_dir)
    pattern = (
        "openmed-openmedkit-react-native-*.tgz" if react_native else "openmed-*.tgz"
    )
    tarball = single_tarball(packs, pattern)
    lane.artifact("npm-tarball", tarball)

    consumer = stage_consumer(ctx.root, "npm", lane_dir / "consumer")
    snapshot = ctx.root / WEB_API_SNAPSHOT
    if not react_native and snapshot.is_file():
        shutil.copy2(snapshot, consumer / "public-api.json")
    lane.note("stage_consumer")
    lane.step(
        "npm_install",
        (
            npm,
            "install",
            "--no-audit",
            "--no-fund",
            "--no-package-lock",
            "--offline",
            str(tarball),
        ),
        consumer,
    )
    if react_native and tsx is not None:
        test_argv: tuple[str, ...] = (
            node,
            str(tsx),
            "--test",
            "--test-reporter=tap",
            *REACT_NATIVE_TESTS,
        )
        tsconfig = "tsconfig.react-native.json"
    else:
        test_argv = (node, "--test", "--test-reporter=tap", *WEB_TESTS)
        tsconfig = "tsconfig.web.json"
    outcome = lane.step("node_test", test_argv, consumer, check=False)
    lane.result.checks = parse_tap(outcome.stdout)
    typecheck = lane.step(
        "typecheck_migration", (node, str(tsc), "-p", tsconfig), consumer, check=False
    )
    lane.result.checks.append(
        {
            "id": "typescript.migration_compiles",
            "status": "passed" if typecheck.returncode == 0 else "failed",
        }
    )
    finish_checks(lane, outcome.returncode or typecheck.returncode)


def finish_checks(lane: LaneRun, returncode: int) -> None:
    """Fail the lane unless the consumer ran checks and all of them passed."""
    checks = lane.result.checks
    if returncode != 0 or any(check["status"] == "failed" for check in checks):
        raise LaneFailure("consumer_checks_failed")
    if not any(check["status"] == "passed" for check in checks):
        raise LaneFailure("consumer_checks_missing")


LANE_RUNNERS: dict[str, Callable[[LaneRun], None]] = {
    "swiftpm": run_swiftpm,
    "gradle": run_gradle,
    "npm-web": lambda lane: run_npm_lane(lane, react_native=False),
    "npm-react-native": lambda lane: run_npm_lane(lane, react_native=True),
}


def run_lane(ctx: Context, name: str) -> LaneResult:
    """Run one lane and convert its outcome to a value-free result."""
    result = LaneResult(lane=name)
    lane = LaneRun(ctx, result)
    try:
        LANE_RUNNERS[name](lane)
    except LaneSkip as skip:
        result.status, result.reason = "skipped", skip.reason
    except LaneFailure as failure:
        result.status, result.reason = "failed", failure.reason
    except (OSError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        result.status, result.reason = "failed", f"error:{type(error).__name__}"
    else:
        result.status, result.reason = "passed", "ok"
    return result


def build_report(results: Sequence[LaneResult]) -> dict[str, object]:
    """Return the JSON report; it never contains paths or fixture values."""
    return {
        "schema_version": 1,
        "report": REPORT_SCHEMA,
        "publication": False,
        "lanes": [result.to_json() for result in results],
        "qualified_lanes": [result.lane for result in results if result.qualified],
    }


def exit_code(results: Sequence[LaneResult], *, require: bool) -> int:
    """Return 1 when a lane failed, or was skipped while lanes are required."""
    for result in results:
        if result.status == "failed" or (require and result.status != "passed"):
            return 1
    return 0


def tail(path: Path, lines: int = 60) -> str:
    """Return the last ``lines`` lines of a lane log."""
    if not path.is_file():
        return ""
    return "\n".join(path.read_text("utf-8", errors="replace").splitlines()[-lines:])


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--lane",
        action="append",
        choices=LANES,
        help="Lane to run; repeat for several. Defaults to every lane.",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Fail when a requested lane is skipped instead of qualifying it.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Staging directory outside the checkout. Defaults to a temporary "
        "directory that is removed afterwards unless --keep is set.",
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep the staging directory."
    )
    parser.add_argument("--report", type=Path, help="Write the JSON report here.")
    parser.add_argument(
        "--swift-source",
        choices=("git", "path"),
        default="git",
        help="Consume the staged Swift package through a git tag or a path.",
    )
    parser.add_argument(
        "--gradle-offline", action="store_true", help="Pass --offline to Gradle."
    )
    parser.add_argument(
        "--install-js-deps",
        action="store_true",
        help="Run npm ci in a JS package whose node_modules is missing.",
    )
    parser.add_argument("--tsc", type=Path, help="TypeScript compiler entry point.")
    parser.add_argument("--tsx", type=Path, help="tsx CLI entry point for the RN lane.")
    parser.add_argument(
        "--print-logs-on-failure",
        action="store_true",
        help="Print the tail of a failed lane's log to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, runner: Runner = subprocess_runner) -> int:
    """Run the requested lanes and return the process exit code."""
    args = parse_args(argv)
    violations = scan_consumer_sources(ROOT / CONSUMERS_DIR)
    if violations:
        print(
            "installed-consumers: consumer sources reference package sources: "
            + ", ".join(violations),
            file=sys.stderr,
        )
        return 2

    created = args.work_dir is None
    work_dir = (
        Path(tempfile.mkdtemp(prefix="openmed-installed-consumers-"))
        if created
        else args.work_dir
    ).resolve()
    if work_dir.is_relative_to(ROOT):
        print(
            "installed-consumers: --work-dir must be outside the checkout",
            file=sys.stderr,
        )
        return 2
    work_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(
        root=ROOT,
        work_dir=work_dir,
        runner=runner,
        swift_source=args.swift_source,
        gradle_offline=args.gradle_offline,
        install_js_deps=args.install_js_deps,
        tsc=args.tsc,
        tsx=args.tsx,
    )
    results = []
    for name in args.lane or LANES:
        result = run_lane(ctx, name)
        results.append(result)
        passed = sum(1 for check in result.checks if check["status"] == "passed")
        print(
            f"installed-consumers: {name} {result.status} ({result.reason}; "
            f"{passed}/{len(result.checks)} checks)"
        )
        if result.status == "failed" and args.print_logs_on_failure:
            print(tail(work_dir / "logs" / f"{name}.log"), file=sys.stderr)

    report = build_report(results)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8"
        )
    if created and not args.keep:
        shutil.rmtree(work_dir, ignore_errors=True)
    return exit_code(results, require=args.require)


if __name__ == "__main__":
    raise SystemExit(main())
