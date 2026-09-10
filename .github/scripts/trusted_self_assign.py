#!/usr/bin/env python3
"""Self-assign trusted contributors who comment ``/assign`` on an issue."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

COMMAND = "/assign"
MIN_MERGED_PULL_REQUESTS = 1
GITHUB_API_VERSION = "2026-03-10"
GITHUB_API_URL = "https://api.github.com"
LOGIN_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class EventPayloadError(RuntimeError):
    """Raised when a relevant event has an unsafe or malformed payload."""


class GitHubResponseError(RuntimeError):
    """Raised when GitHub returns an unusable response."""


@dataclass(frozen=True)
class AssignmentRequest:
    """Validated self-assignment request extracted from an issue event."""

    issue_number: int
    login: str
    existing_assignees: frozenset[str]


class AssignmentClient(Protocol):
    """GitHub operations required by the self-assignment handler."""

    def merged_pull_request_count(self, repository: str, login: str) -> int:
        """Return the number of merged pull requests authored by ``login``."""

    def add_assignee(self, repository: str, issue_number: int, login: str) -> None:
        """Add ``login`` to the issue's current assignees."""

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        """Post an issue comment."""


class GitHubClient:
    """Small REST client for the GitHub operations used by this workflow."""

    def __init__(self, token: str, api_url: str = GITHUB_API_URL) -> None:
        if not token:
            raise ValueError("GITHUB_TOKEN must not be empty")
        self._token = token
        self._api_url = api_url.rstrip("/")

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._api_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"

        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        request = Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "User-Agent": "openmed-trusted-self-assign",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            },
        )
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310
                raw_body = response.read()
        except HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")[:500]
            raise GitHubResponseError(
                f"GitHub API {method} {path} returned HTTP {error.code}: "
                f"{response_body}"
            ) from error
        except URLError as error:
            raise GitHubResponseError(
                f"GitHub API {method} {path} could not be reached: {error.reason}"
            ) from error

        if not raw_body:
            return {}
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as error:
            raise GitHubResponseError(
                f"GitHub API {method} {path} returned invalid JSON"
            ) from error
        if not isinstance(parsed, dict):
            raise GitHubResponseError(
                f"GitHub API {method} {path} returned a non-object response"
            )
        return parsed

    def merged_pull_request_count(self, repository: str, login: str) -> int:
        """Return a complete merged-PR count for ``login`` or fail closed."""

        query = f"repo:{repository} is:pr is:merged author:{login}"
        result = self._request_json(
            "GET",
            "/search/issues",
            query={"q": query, "per_page": "1"},
        )
        if result.get("incomplete_results") is not False:
            raise GitHubResponseError(
                "GitHub returned incomplete merged pull request search results"
            )

        total_count = result.get("total_count")
        if (
            isinstance(total_count, bool)
            or not isinstance(total_count, int)
            or total_count < 0
        ):
            raise GitHubResponseError(
                "GitHub returned an invalid merged pull request count"
            )
        return total_count

    def add_assignee(self, repository: str, issue_number: int, login: str) -> None:
        """Add ``login`` without replacing the issue's existing assignees."""

        self._request_json(
            "POST",
            f"/repos/{repository}/issues/{issue_number}/assignees",
            payload={"assignees": [login]},
        )

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        """Post an issue comment explaining an ineligible request."""

        self._request_json(
            "POST",
            f"/repos/{repository}/issues/{issue_number}/comments",
            payload={"body": body},
        )


def parse_assignment_request(event: dict[str, Any]) -> AssignmentRequest | None:
    """Return a validated request, or ``None`` for events that must be ignored."""

    if event.get("action") != "created":
        return None

    issue = event.get("issue")
    comment = event.get("comment")
    if not isinstance(issue, dict) or not isinstance(comment, dict):
        return None
    if "pull_request" in issue or issue.get("state") != "open":
        return None
    if comment.get("body") != COMMAND:
        return None

    user = comment.get("user")
    if not isinstance(user, dict):
        raise EventPayloadError("Relevant comment is missing its user")
    login = user.get("login")
    user_type = user.get("type")
    if user_type == "Bot" or (
        isinstance(login, str) and login.casefold().endswith("[bot]")
    ):
        return None
    if not isinstance(login, str) or LOGIN_PATTERN.fullmatch(login) is None:
        raise EventPayloadError("Relevant comment has an invalid GitHub login")

    issue_number = issue.get("number")
    if (
        isinstance(issue_number, bool)
        or not isinstance(issue_number, int)
        or issue_number < 1
    ):
        raise EventPayloadError("Relevant comment has an invalid issue number")

    assignees = issue.get("assignees", [])
    if not isinstance(assignees, list):
        raise EventPayloadError("Relevant issue has an invalid assignee list")
    existing_assignees: set[str] = set()
    for assignee in assignees:
        if not isinstance(assignee, dict) or not isinstance(assignee.get("login"), str):
            raise EventPayloadError("Relevant issue has an invalid assignee")
        existing_assignees.add(assignee["login"].casefold())

    return AssignmentRequest(
        issue_number=issue_number,
        login=login,
        existing_assignees=frozenset(existing_assignees),
    )


def process_event(
    event: dict[str, Any], repository: str, client: AssignmentClient
) -> str:
    """Process one issue-comment event and return a machine-readable outcome."""

    if REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise EventPayloadError("GITHUB_REPOSITORY has an invalid value")

    assignment = parse_assignment_request(event)
    if assignment is None:
        return "ignored"
    if assignment.login.casefold() in assignment.existing_assignees:
        return "already-assigned"

    merged_count = client.merged_pull_request_count(repository, assignment.login)
    if merged_count < MIN_MERGED_PULL_REQUESTS:
        client.add_comment(
            repository,
            assignment.issue_number,
            (
                f"@{assignment.login}, automatic `/assign` self-assignment is "
                "available after at least one pull request has been merged into "
                "this repository. Please ask a maintainer to assign you for now."
            ),
        )
        return "ineligible"

    client.add_assignee(repository, assignment.issue_number, assignment.login)
    return "assigned"


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def main() -> int:
    """Load the workflow event and run the self-assignment handler."""

    event_path = Path(_required_environment("GITHUB_EVENT_PATH"))
    event = json.loads(event_path.read_text(encoding="utf-8"))
    if not isinstance(event, dict):
        raise EventPayloadError("GITHUB_EVENT_PATH must contain a JSON object")

    client = GitHubClient(_required_environment("GITHUB_TOKEN"))
    outcome = process_event(
        event,
        _required_environment("GITHUB_REPOSITORY"),
        client,
    )
    print(f"Trusted self-assignment outcome: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
