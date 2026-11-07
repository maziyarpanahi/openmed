"""Dependabot configuration policy tests."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEPENDABOT_CONFIG = ROOT / ".github" / "dependabot.yml"
KIT_ECOSYSTEMS = ("npm", "gradle", "swift", "cargo")


def _github_actions_section() -> str:
    text = DEPENDABOT_CONFIG.read_text(encoding="utf-8")
    _, section = text.split('package-ecosystem: "github-actions"', 1)
    return section.split('package-ecosystem: "pre-commit"', 1)[0]


def _updates() -> list[dict]:
    config = yaml.safe_load(DEPENDABOT_CONFIG.read_text(encoding="utf-8"))
    return config["updates"]


def _directories(update: dict) -> set[str]:
    if "directories" in update:
        return set(update["directories"])
    return {update["directory"]}


def _as_dependabot_directory(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    return "/" if relative == "." else f"/{relative}"


def _kit_manifest_directories() -> dict[str, set[str]]:
    """Return the committed kit manifest directories for each kit ecosystem."""
    manifests = {
        "npm": [*ROOT.glob("js/*/package.json"), *ROOT.glob("clients/*/package.json")],
        "gradle": [ROOT / "android" / "settings.gradle.kts"],
        "swift": [ROOT / "Package.swift", ROOT / "swift/OpenMedKit/Package.swift"],
        "cargo": list(ROOT.glob("js/*/src-tauri/Cargo.toml")),
    }
    return {
        ecosystem: {
            _as_dependabot_directory(path.parent) for path in paths if path.is_file()
        }
        for ecosystem, paths in manifests.items()
    }


def test_github_actions_group_excludes_major_updates():
    section = _github_actions_section()

    assert "applies-to: version-updates" in section
    assert "update-types:" in section
    assert '- "minor"' in section
    assert '- "patch"' in section
    assert '- "major"' not in section


def test_every_kit_ecosystem_has_one_update_entry():
    ecosystems = [update["package-ecosystem"] for update in _updates()]

    for ecosystem in KIT_ECOSYSTEMS:
        assert ecosystems.count(ecosystem) == 1, ecosystem


def test_kit_update_entries_cover_every_committed_manifest_directory():
    expected = _kit_manifest_directories()
    by_ecosystem = {update["package-ecosystem"]: update for update in _updates()}

    # Negative controls: discovery must see the kits this policy is about.
    assert expected["npm"] >= {
        "/js/openmedkit-electron",
        "/js/openmedkit-react-native",
        "/js/openmedkit-tauri",
        "/js/openmedkit-web",
        "/clients/typescript",
    }
    assert expected["gradle"] == {"/android"}
    assert expected["swift"] == {"/", "/swift/OpenMedKit"}
    assert expected["cargo"] == {"/js/openmedkit-tauri/src-tauri"}
    for ecosystem in KIT_ECOSYSTEMS:
        assert _directories(by_ecosystem[ecosystem]) == expected[ecosystem], ecosystem


def test_kit_updates_follow_the_weekly_grouping_policy():
    kit_updates = [
        update for update in _updates() if update["package-ecosystem"] in KIT_ECOSYSTEMS
    ]

    assert len(kit_updates) == len(KIT_ECOSYSTEMS)
    for update in kit_updates:
        schedule = update["schedule"]
        assert schedule["interval"] == "weekly"
        assert schedule["day"] == "monday"
        assert schedule["timezone"] == "Etc/UTC"
        assert update["open-pull-requests-limit"] == 5
        assert len(update["groups"]) == 1
        (group,) = update["groups"].values()
        # Majors and security updates stay out of the group, as for Actions.
        assert group["applies-to"] == "version-updates"
        assert group["patterns"] == ["*"]
        assert group["update-types"] == ["minor", "patch"]


def test_update_schedules_do_not_collide():
    slots = [
        (update["schedule"]["day"], update["schedule"]["time"]) for update in _updates()
    ]

    assert len(slots) == len(set(slots))
