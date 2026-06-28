"""GitHub Actions workflow reference policy tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_github_actions_refs.py"

spec = importlib.util.spec_from_file_location("check_github_actions_refs", SCRIPT)
assert spec is not None
actions_refs = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = actions_refs
spec.loader.exec_module(actions_refs)


def test_iter_action_refs_finds_remote_refs_and_ignores_local_refs(tmp_path):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    workflow = workflows / "ci.yml"
    workflow.write_text(
        "\n".join(
            [
                "jobs:",
                "  test:",
                "    steps:",
                "      - uses: actions/checkout@v6",
                "      - uses: ./.github/actions/local",
                "      - uses: docker://alpine:3.20",
                "      - uses: owner/repo/.github/workflows/reuse.yml@v1",
                "      - uses: actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd",
            ]
        ),
        encoding="utf-8",
    )

    refs = list(actions_refs.iter_action_refs(workflows))

    assert [(ref.spec, ref.repository, ref.ref) for ref in refs] == [
        ("actions/checkout@v6", "actions/checkout", "v6"),
        (
            "owner/repo/.github/workflows/reuse.yml@v1",
            "owner/repo",
            "v1",
        ),
        (
            "actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd",
            "actions/github-script",
            "ed597411d8f924073f98dfc5c65a23a2325f34cd",
        ),
    ]


def test_audit_action_refs_reports_missing_tags_without_rechecking_duplicates(tmp_path):
    action_ref = actions_refs.ActionRef(
        tmp_path / "ci.yml",
        10,
        "astral-sh/setup-uv@v8",
        "astral-sh/setup-uv",
        "v8",
    )
    duplicate_ref = actions_refs.ActionRef(
        tmp_path / "ci.yml",
        20,
        "astral-sh/setup-uv@v8",
        "astral-sh/setup-uv",
        "v8",
    )
    calls: list[tuple[str, str]] = []

    def resolver(repository: str, ref: str) -> tuple[bool, str]:
        calls.append((repository, ref))
        return False, "missing tag"

    results = actions_refs.audit_action_refs([action_ref, duplicate_ref], resolver)

    assert calls == [("astral-sh/setup-uv", "v8")]
    assert [result.ok for result in results] == [False, False]
    assert "missing tag" in actions_refs.format_result(results[0])


def test_audit_action_refs_accepts_commit_sha_without_network(tmp_path):
    action_ref = actions_refs.ActionRef(
        tmp_path / "ci.yml",
        12,
        "actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd",
        "actions/github-script",
        "ed597411d8f924073f98dfc5c65a23a2325f34cd",
    )

    def resolver(repository: str, ref: str) -> tuple[bool, str]:
        raise AssertionError("SHA refs should not need remote lookups")

    results = actions_refs.audit_action_refs([action_ref], resolver)

    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].reason == "pinned commit SHA"


def test_main_fails_when_remote_ref_does_not_parse(tmp_path, capsys):
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "ci.yml").write_text(
        "jobs:\n  test:\n    uses: owner/repo/.github/workflows/reuse.yml\n",
        encoding="utf-8",
    )

    assert actions_refs.main(["--workflows-dir", str(workflows)]) == 1

    captured = capsys.readouterr()
    assert "remote action refs must be static" in captured.err
