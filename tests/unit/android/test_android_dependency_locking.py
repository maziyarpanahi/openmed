"""Android Gradle dependency locking and verification metadata policy tests."""

from __future__ import annotations

import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[3]
ANDROID = ROOT / "android"
LOCKING_SCRIPT = ANDROID / "gradle" / "dependency-locking.gradle.kts"
ROOT_BUILD = ANDROID / "build.gradle.kts"
CATALOG = ANDROID / "gradle" / "libs.versions.toml"
VERIFICATION = ANDROID / "gradle" / "verification-metadata.xml"
LIBRARY_BUILD = ANDROID / "openmedkit" / "build.gradle.kts"
WORKFLOW = ROOT / ".github" / "workflows" / "android-ci.yml"
NAMESPACE = {"v": "https://schema.gradle.org/dependency-verification"}
SHIPPED = "releaseRuntimeClasspath"
ENTRY = re.compile(r"(?P<group>[^:=\s]+):(?P<name>[^:=\s]+):(?P<version>[^:=\s]+)=(.+)")


def _modules() -> list[Path]:
    """Return every Android Gradle module directory (not the root build)."""
    return sorted(
        path.parent
        for path in ANDROID.rglob("build.gradle.kts")
        if path.parent != ANDROID and "build" not in path.relative_to(ANDROID).parts
    )


def _lock_entries(module: Path) -> dict[tuple[str, str], tuple[str, set[str]]]:
    entries = {}
    for line in (module / "gradle.lockfile").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("empty="):
            continue
        match = ENTRY.fullmatch(line)
        assert match is not None, line
        configurations = set(match.group(4).split(","))
        entries[(match["group"], match["name"])] = (match["version"], configurations)
    return entries


def _verified_components() -> dict[tuple[str, str, str], set[str]]:
    root = ET.parse(VERIFICATION).getroot()
    return {
        (item.get("group"), item.get("name"), item.get("version")): {
            artifact.get("name") for artifact in item.iterfind("v:artifact", NAMESPACE)
        }
        for item in root.iterfind("v:components/v:component", NAMESPACE)
    }


def test_every_module_applies_strict_locking_for_all_build_configurations() -> None:
    script = LOCKING_SCRIPT.read_text(encoding="utf-8")
    root_build = ROOT_BUILD.read_text(encoding="utf-8")

    assert "subprojects {" in root_build
    assert 'apply(from = rootProject.file("gradle/dependency-locking.gradle.kts"))' in (
        root_build
    )
    assert "lockMode.set(LockMode.STRICT)" in script
    assert "resolutionStrategy.activateDependencyLocking()" in script
    # Only configurations the build never resolves are left unlocked.
    assert 'val unlockedConfigurations = setOf("default", "archives")' in script
    assert '!name.endsWith("DependenciesMetadata")' in script
    assert 'tasks.register("verifyDependencyLocks")' in script
    assert "UnresolvedDependencyResult" in script


def test_every_module_commits_a_lockfile_with_shipped_entries() -> None:
    modules = _modules()

    assert {module.relative_to(ANDROID).as_posix() for module in modules} >= {
        "openmedkit",
        "OpenMedDemo/app",
        "OpenMedMapleDemo/app",
        "OpenMedScanDemo/app",
    }
    for module in modules:
        entries = _lock_entries(module)
        shipped = [name for name, (_, confs) in entries.items() if SHIPPED in confs]
        assert shipped, module.relative_to(ANDROID)
        assert any(
            "releaseUnitTestRuntimeClasspath" in confs for _, confs in entries.values()
        ), module.relative_to(ANDROID)


def test_library_lock_pins_each_catalog_dependency_it_declares() -> None:
    catalog = tomllib.loads(CATALOG.read_text(encoding="utf-8"))
    versions = catalog["versions"]
    libraries = catalog["libraries"]
    build = LIBRARY_BUILD.read_text(encoding="utf-8")
    declared = re.findall(r"^\s*implementation\(libs\.([a-z0-9.]+)\)", build, re.M)
    entries = _lock_entries(ANDROID / "openmedkit")

    assert declared
    for accessor in declared:
        library = libraries[accessor.replace(".", "-")]
        group, name = library["module"].split(":")
        version = versions[library["version"]["ref"]]
        # A catalog bump without --write-locks fails resolution in Gradle; this
        # catches the same drift offline.
        locked_version, configurations = entries[(group, name)]
        assert locked_version == version, library["module"]
        assert SHIPPED in configurations, library["module"]


def test_verification_metadata_checks_sha256_for_every_locked_module() -> None:
    root = ET.parse(VERIFICATION).getroot()
    configuration = root.find("v:configuration", NAMESPACE)
    components = _verified_components()
    text = VERIFICATION.read_text(encoding="utf-8")

    assert configuration is not None
    assert configuration.findtext("v:verify-metadata", namespaces=NAMESPACE) == "true"
    assert "<sha256 " in text
    for weaker in ("<sha1 ", "<md5 ", "<sha512 "):
        assert weaker not in text
    # Only IDE-downloaded sources and javadoc are trusted without a checksum.
    trusted = {
        item.get("file")
        for item in root.iterfind(
            "v:configuration/v:trusted-artifacts/v:trust", NAMESPACE
        )
    }
    assert trusted == {".*-javadoc[.]jar", ".*-sources[.]jar"}
    for module in _modules():
        for (group, name), (version, _) in _lock_entries(module).items():
            assert (group, name, version) in components, f"{group}:{name}:{version}"


def test_verification_metadata_covers_aapt2_for_every_ci_and_developer_host() -> None:
    catalog = tomllib.loads(CATALOG.read_text(encoding="utf-8"))
    agp = catalog["versions"]["agp"]
    aapt2 = {
        version: artifacts
        for (group, name, version), artifacts in _verified_components().items()
        if group == "com.android.tools.build" and name == "aapt2"
    }

    (version,) = [version for version in aapt2 if version.startswith(f"{agp}-")]
    for host in ("linux", "osx", "windows"):
        assert f"aapt2-{version}-{host}.jar" in aapt2[version], host


def test_lock_and_verification_files_hold_coordinates_only() -> None:
    files = [module / "gradle.lockfile" for module in _modules()]
    files += [ANDROID / "settings-gradle.lockfile", VERIFICATION]

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/home/" not in text
        assert "file:" not in text


def test_android_ci_verifies_locks_and_rejects_an_unlocked_dependency() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = {
        step.get("name"): step for step in workflow["jobs"]["build-and-test"]["steps"]
    }
    cache_paths = steps["Set up Java"]["with"]["cache-dependency-path"]

    assert steps["Verify Gradle dependency locks"]["run"] == (
        "./gradlew verifyDependencyLocks --stacktrace"
    )
    negative = steps["Verify an unlocked dependency change fails the build"]["run"]
    assert "dependencies.add('implementation', 'org.json:json:20240303')" in negative
    assert ":openmedkit:compileReleaseKotlin --init-script" in negative
    assert "which is not part of the dependency lock state" in negative
    assert "android/**/gradle.lockfile" in cache_paths
    assert "android/gradle/verification-metadata.xml" in cache_paths
    # The negative-control dependency must already be checksum-verified, so the
    # failure can only come from the lock state.
    assert ("org.json", "json", "20240303") in _verified_components()
