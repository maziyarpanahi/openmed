import os
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github/workflows"
BRANCH = "release/v4.3"
REQUIRED = (
    "ci",
    "tests",
    "secret-scan",
    "sdk-release-candidate",
    "swift-test",
    "android-ci",
    "edge-benchmark",
    "eval",
    "helm-lint",
    "installed-consumers",
    "low-resource",
    "mlx-test",
    "nix",
    "notebooks-execute",
    "reproducible-lock",
    "tauri-sidecar",
    "vuln-scan",
)


def workflow(name):
    return yaml.load((WORKFLOWS / (name + ".yml")).read_text(), Loader=yaml.BaseLoader)


def covers(config):
    def matches(pattern):
        # A single GitHub Actions wildcard cannot consume a slash.
        if pattern == "**":
            return True
        parts = pattern.split("/")
        return len(parts) == 2 and all(
            fnmatchcase(value, p)
            for value, p in zip(BRANCH.split("/"), parts, strict=True)
        )

    included = "branches" not in config
    for pattern in config.get("branches", ()):
        if matches(pattern.lstrip("!")):
            included = not pattern.startswith("!")
    return included and not any(
        matches(pattern) for pattern in config.get("branches-ignore", ())
    )


@pytest.mark.parametrize("name", REQUIRED)
def test_lts_branch_coverage(name):
    events = workflow(name)["on"]
    for event in ("pull_request", "push"):
        assert event in events and covers(events[event] or {}), (name, event)


def test_attribution_runs_for_lts_pr_heads():
    scan = workflow("secret-scan")
    event = scan["on"]["pull_request_target"]
    assert covers(event)
    assert {"opened", "synchronize", "reopened"} <= set(event["types"])
    job = scan["jobs"]["commit-attribution"]
    assert job["if"] == "github.event_name != 'pull_request'"
    assert any(
        "Commit attribution policy" in step.get("run", "") for step in job["steps"]
    )


def test_filters_that_exclude_lts_are_detected():
    assert not covers({"branches": ["master", "release/v*", "!release/v4.3"]})
    assert not covers({"branches-ignore": ["release/v*"]})
    assert not covers({"branches": ["*"]})


@pytest.mark.parametrize(("version", "expected_code"), [("4.3.2", 0), ("4.4.0", 1)])
def test_sdk_resolver_on_lts(tmp_path, version, expected_code):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    metadata = tmp_path / "openmed/__about__.py"
    metadata.parent.mkdir()
    metadata.write_text(f'__version__ = "{version}"\n')
    git("add", "openmed/__about__.py")
    git(
        "-c",
        "user.name=Release fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Synthetic SDK candidate",
    )
    for tag in ("v4.2.9", "v4.3.1", "v4.3.2-rc1", "v4.30.0"):
        git("tag", tag)
    script = next(
        step["run"]
        for step in workflow("sdk-release-candidate")["jobs"]["readiness"]["steps"]
        if step.get("id") == "candidate"
    )
    source = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    output = tmp_path / "output"
    env = dict(
        os.environ,
        RELEASE_BASE_REF=BRANCH,
        GITHUB_EVENT_NAME="push",
        CANDIDATE_SHA=git("rev-parse", "HEAD"),
        REQUESTED_API_BASE="v3.2.0",
        REQUESTED_MIGRATION="old.md",
        GITHUB_OUTPUT=str(output),
    )
    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == expected_code, proc.stderr
    if not expected_code:
        assert "api_base=v4.3.1" in output.read_text()
        assert "migration=docs/migration/4.2-to-4.3.md" in output.read_text()
