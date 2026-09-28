"""GitHub Actions workflow reference policy tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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


def test_sha_policy_rejects_tags_branches_and_short_shas_without_network(tmp_path):
    def resolver(repository: str, ref: str) -> tuple[bool, str]:
        raise AssertionError("SHA policy must not resolve mutable references")

    refs = [
        actions_refs.ActionRef(
            tmp_path / "ci.yml", 1, f"owner/action@{ref}", "owner/action", ref
        )
        for ref in ("v1", "main", "ed597411d8f9")
    ]
    results = actions_refs.audit_action_refs(refs, resolver, require_sha=True)

    assert all(not result.ok for result in results)
    assert all("full commit SHA" in result.reason for result in results)


def test_sha_policy_checks_nested_composite_actions(tmp_path, capsys):
    actions = tmp_path / "actions"
    nested = actions / "example"
    nested.mkdir(parents=True)
    action = nested / "action.yml"
    action.write_text("runs:\n  steps:\n    - uses: actions/checkout@v7\n")
    args = ["--workflows-dir", str(actions), "--require-sha"]

    assert actions_refs.main(args) == 1
    assert "full commit SHA" in capsys.readouterr().err

    action.write_text(
        "runs:\n  steps:\n"
        "    - uses: ./local-action\n"
        "    - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1\n"
    )
    assert actions_refs.main(args) == 0


def test_repository_actions_remain_pinned():
    refs = [
        *actions_refs.iter_action_refs(ROOT / ".github" / "workflows"),
        *actions_refs.iter_action_refs(ROOT / ".github" / "actions"),
    ]
    assert refs
    results = actions_refs.audit_action_refs(refs, require_sha=True)
    assert all(result.ok for result in results), [
        actions_refs.format_result(result) for result in results if not result.ok
    ]


@pytest.mark.parametrize(
    "step",
    [
        '- "uses": actions/checkout@v7',
        "- {uses: actions/checkout@v7}",
        "- uses: >-\n        actions/checkout@v7",
        "- &checkout {uses: actions/checkout@v7}\n    - *checkout",
    ],
)
def test_sha_policy_parses_equivalent_yaml_forms(tmp_path, capsys, step):
    (tmp_path / "ci.yml").write_text(f"jobs:\n  test:\n    steps:\n    {step}\n")

    refs = list(actions_refs.iter_action_refs(tmp_path))
    assert refs and all(ref.spec == "actions/checkout@v7" for ref in refs)
    assert actions_refs.main(["--workflows-dir", str(tmp_path), "--require-sha"]) == 1
    assert "full commit SHA" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["[broken", "null", "[actions/checkout@v7]"])
def test_sha_policy_fails_closed_on_invalid_yaml_or_uses_values(
    tmp_path, capsys, value
):
    (tmp_path / "ci.yml").write_text(f"jobs:\n  test:\n    uses: {value}\n")

    assert actions_refs.main(["--workflows-dir", str(tmp_path), "--require-sha"]) == 1
    assert "policy failed" in capsys.readouterr().err
