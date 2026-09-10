"""Tests for trusted-contributor issue self-assignment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github" / "scripts" / "trusted_self_assign.py"

spec = importlib.util.spec_from_file_location("trusted_self_assign", SCRIPT)
assert spec is not None
trusted_self_assign = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = trusted_self_assign
spec.loader.exec_module(trusted_self_assign)


class FakeClient:
    """Record calls made by the event processor."""

    def __init__(self, merged_count: int = 1) -> None:
        self.merged_count = merged_count
        self.merged_queries: list[tuple[str, str]] = []
        self.assignments: list[tuple[str, int, str]] = []
        self.comments: list[tuple[str, int, str]] = []

    def merged_pull_request_count(self, repository: str, login: str) -> int:
        self.merged_queries.append((repository, login))
        return self.merged_count

    def add_assignee(self, repository: str, issue_number: int, login: str) -> None:
        self.assignments.append((repository, issue_number, login))

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        self.comments.append((repository, issue_number, body))


def issue_comment_event(
    *,
    action: str = "created",
    body: str = "/assign",
    state: str = "open",
    login: str = "merged-user",
    user_type: str = "User",
    assignees: list[str] | None = None,
    pull_request: bool = False,
) -> dict[str, Any]:
    """Build a minimal issue-comment event payload."""

    issue: dict[str, Any] = {
        "number": 42,
        "state": state,
        "assignees": [{"login": assignee} for assignee in (assignees or [])],
    }
    if pull_request:
        issue["pull_request"] = {
            "url": "https://api.github.com/repos/maziyarpanahi/openmed/pulls/42"
        }
    return {
        "action": action,
        "issue": issue,
        "comment": {
            "body": body,
            "user": {"login": login, "type": user_type},
        },
    }


def test_parse_assignment_request_accepts_exact_open_issue_command():
    request = trusted_self_assign.parse_assignment_request(issue_comment_event())

    assert request == trusted_self_assign.AssignmentRequest(
        issue_number=42,
        login="merged-user",
        existing_assignees=frozenset(),
    )


@pytest.mark.parametrize(
    "event",
    [
        issue_comment_event(action="edited"),
        issue_comment_event(body=" /assign"),
        issue_comment_event(body="/assign\n"),
        issue_comment_event(body="/assign please"),
        issue_comment_event(state="closed"),
        issue_comment_event(pull_request=True),
        issue_comment_event(user_type="Bot"),
        issue_comment_event(login="dependabot[bot]"),
    ],
)
def test_parse_assignment_request_ignores_unsupported_events(event):
    assert trusted_self_assign.parse_assignment_request(event) is None


def test_parse_assignment_request_rejects_invalid_relevant_login():
    event = issue_comment_event(login="bad login")

    with pytest.raises(
        trusted_self_assign.EventPayloadError,
        match="invalid GitHub login",
    ):
        trusted_self_assign.parse_assignment_request(event)


def test_process_event_assigns_author_with_one_merged_pull_request():
    client = FakeClient(merged_count=1)

    outcome = trusted_self_assign.process_event(
        issue_comment_event(), "maziyarpanahi/openmed", client
    )

    assert outcome == "assigned"
    assert client.merged_queries == [("maziyarpanahi/openmed", "merged-user")]
    assert client.assignments == [("maziyarpanahi/openmed", 42, "merged-user")]
    assert client.comments == []


def test_process_event_rejects_author_without_a_merged_pull_request():
    client = FakeClient(merged_count=0)

    outcome = trusted_self_assign.process_event(
        issue_comment_event(), "maziyarpanahi/openmed", client
    )

    assert outcome == "ineligible"
    assert client.assignments == []
    assert len(client.comments) == 1
    assert "after at least one pull request has been merged" in client.comments[0][2]
    assert "ask a maintainer" in client.comments[0][2]


def test_process_event_adds_user_without_replacing_existing_assignees():
    client = FakeClient()
    event = issue_comment_event(assignees=["current-owner"])

    outcome = trusted_self_assign.process_event(event, "maziyarpanahi/openmed", client)

    assert outcome == "assigned"
    assert client.assignments == [("maziyarpanahi/openmed", 42, "merged-user")]


def test_process_event_is_idempotent_for_existing_assignee():
    client = FakeClient()
    event = issue_comment_event(assignees=["MERGED-USER"])

    outcome = trusted_self_assign.process_event(event, "maziyarpanahi/openmed", client)

    assert outcome == "already-assigned"
    assert client.merged_queries == []
    assert client.assignments == []
    assert client.comments == []


def test_process_event_rejects_invalid_repository():
    with pytest.raises(
        trusted_self_assign.EventPayloadError,
        match="GITHUB_REPOSITORY",
    ):
        trusted_self_assign.process_event(
            issue_comment_event(), "maziyarpanahi/openmed other:repo", FakeClient()
        )


def test_client_uses_scoped_merged_pull_request_search(monkeypatch):
    client = trusted_self_assign.GitHubClient("test-token")
    calls: list[tuple[str, str, dict[str, str] | None, Any]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        calls.append((method, path, query, payload))
        return {"total_count": 1, "incomplete_results": False, "items": []}

    monkeypatch.setattr(client, "_request_json", fake_request)

    assert client.merged_pull_request_count("maziyarpanahi/openmed", "merged-user") == 1
    assert calls == [
        (
            "GET",
            "/search/issues",
            {
                "q": ("repo:maziyarpanahi/openmed is:pr is:merged author:merged-user"),
                "per_page": "1",
            },
            None,
        )
    ]


@pytest.mark.parametrize(
    "response",
    [
        {"total_count": 1, "incomplete_results": True},
        {"total_count": 1},
        {"total_count": True, "incomplete_results": False},
        {"total_count": -1, "incomplete_results": False},
        {"total_count": "1", "incomplete_results": False},
    ],
)
def test_client_fails_closed_on_incomplete_or_invalid_search(monkeypatch, response):
    client = trusted_self_assign.GitHubClient("test-token")
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(trusted_self_assign.GitHubResponseError):
        client.merged_pull_request_count("maziyarpanahi/openmed", "merged-user")


def test_client_posts_additive_assignee_payload(monkeypatch):
    client = trusted_self_assign.GitHubClient("test-token")
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del query
        calls.append((method, path, payload))
        return {}

    monkeypatch.setattr(client, "_request_json", fake_request)

    client.add_assignee("maziyarpanahi/openmed", 42, "merged-user")

    assert calls == [
        (
            "POST",
            "/repos/maziyarpanahi/openmed/issues/42/assignees",
            {"assignees": ["merged-user"]},
        )
    ]


def test_workflow_has_least_privilege_and_exact_command_gate():
    workflow = (
        ROOT / ".github" / "workflows" / "trusted-contributor-self-assign.yml"
    ).read_text(encoding="utf-8")

    assert "issue_comment:" in workflow
    assert "contents: read" in workflow
    assert "issues: write" in workflow
    assert "pull-requests: read" in workflow
    assert "github.event.comment.body == '/assign'" in workflow
    assert "persist-credentials: false" in workflow
