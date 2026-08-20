"""Offline MCP authorization conformance fixtures and runner.

The conformance matrix is deliberately a small protocol model rather than an
OAuth client.  It exercises the authorization boundaries that matter to the
OpenMed MCP surface with an in-process transport and synthetic references.  No
socket, credential, authorization header, or clinical payload is required.

The public report is intentionally privacy-safe: each result contains only a
case identifier and a stable failure category.  Fixture details stay local to
the runner and are never copied into reports, exceptions, or logs.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "AUTHORIZATION_CONFORMANCE_SCHEMA_VERSION",
    "DEFAULT_FIXTURE_PATH",
    "ConformanceCase",
    "ConformanceCaseResult",
    "ConformanceCoverageError",
    "ConformanceFixtureError",
    "ConformanceManifest",
    "ConformanceReport",
    "ConformanceViolation",
    "MockAuthorizationTransport",
    "load_authorization_conformance_manifest",
    "load_conformance_manifest",
    "load_fixture_manifest",
    "manifest_case_ids",
    "render_authorization_matrix",
    "render_conformance_matrix",
    "run_authorization_conformance",
    "run_conformance",
    "validate_case_coverage",
    "write_conformance_matrix",
]

AUTHORIZATION_CONFORMANCE_SCHEMA_VERSION = "openmed.mcp.authorization_conformance.v1"
DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "mcp"
    / "authorization"
    / "conformance_manifest.json"
)

FAILURE_CATEGORIES = frozenset(
    {
        "wrong_audience",
        "missing_resource_indicator",
        "token_passthrough",
        "invalid_pkce",
        "redirect_mismatch",
        "insufficient_tool_scope",
        "oversized_payload",
        "unapproved_state_change",
        "token_binding_mismatch",
    }
)

FAILURE_BOUNDARIES = frozenset(
    {
        "authorization_request",
        "authorization_code_exchange",
        "redirect_validation",
        "token_validation",
        "transport_policy",
        "tool_authorization",
        "payload_policy",
        "state_change_policy",
    }
)

_RESULT_FAILURE_MISSING = "expected_failure_missing"
_RESULT_RUNNER_ERROR = "runner_error"


class ConformanceFixtureError(ValueError):
    """Raised when a conformance manifest does not match its versioned schema."""


class ConformanceCoverageError(ConformanceFixtureError):
    """Raised when focused tests do not declare exactly the manifest's cases."""

    def __init__(
        self,
        *,
        missing: Sequence[str] = (),
        unexpected: Sequence[str] = (),
    ) -> None:
        self.missing = tuple(sorted(missing))
        self.unexpected = tuple(sorted(unexpected))
        parts: list[str] = []
        if self.missing:
            parts.append("missing case declarations")
        if self.unexpected:
            parts.append("unknown case declarations")
        super().__init__(
            "Focused conformance tests must cover every declared case"
            + (f" ({', '.join(parts)})." if parts else ".")
        )


class ConformanceViolation(ValueError):
    """A safe, categorized authorization failure raised by the mock transport."""

    def __init__(self, category: str, boundary: str) -> None:
        if category not in FAILURE_CATEGORIES:
            raise ValueError("unknown conformance failure category")
        if boundary not in FAILURE_BOUNDARIES:
            raise ValueError("unknown conformance failure boundary")
        self.category = category
        self.boundary = boundary
        super().__init__(f"authorization conformance failure: {category}")


def _schema_error(path: str) -> ConformanceFixtureError:
    """Build a manifest error without echoing fixture values."""

    return ConformanceFixtureError(f"Invalid authorization fixture field: {path}.")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _schema_error(path)
    return value


def _string(value: Any, path: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise _schema_error(path)
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _schema_error(path)
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _schema_error(path)
    return value


def _strings(value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _schema_error(path)
    normalized: list[str] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        item_string = _string(item, item_path)
        assert item_string is not None
        normalized.append(item_string)
    if not normalized and not allow_empty:
        raise _schema_error(path)
    if len(set(normalized)) != len(normalized):
        raise _schema_error(path)
    return tuple(normalized)


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


@dataclass(frozen=True)
class ConformanceCase:
    """One synthetic positive or negative authorization scenario."""

    case_id: str
    kind: str
    behavior: str
    failure_boundary: str
    expected_failure: str | None
    authorization_request: Mapping[str, Any]
    token_claims: Mapping[str, Any]
    tool_call: Mapping[str, Any]
    transport: Mapping[str, Any]

    @property
    def is_negative(self) -> bool:
        """Return whether this case must fail at its declared boundary."""

        return self.kind == "negative"


@dataclass(frozen=True)
class ConformanceManifest:
    """Validated versioned authorization conformance manifest."""

    schema_version: str
    synthetic: bool
    protected_resource: Mapping[str, Any]
    authorization_server: Mapping[str, Any]
    policy: Mapping[str, Any]
    cases: tuple[ConformanceCase, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConformanceManifest":
        """Validate and normalize a decoded manifest mapping."""

        _mapping(payload, "manifest")
        schema_version = _string(payload.get("schema_version"), "schema_version")
        assert schema_version is not None
        if schema_version != AUTHORIZATION_CONFORMANCE_SCHEMA_VERSION:
            raise _schema_error("schema_version")
        if _boolean(payload.get("synthetic"), "synthetic") is not True:
            raise _schema_error("synthetic")

        protected = _mapping(payload.get("protected_resource"), "protected_resource")
        protected_resource = {
            "resource": _string(
                protected.get("resource"), "protected_resource.resource"
            ),
            "authorization_servers": _strings(
                protected.get("authorization_servers"),
                "protected_resource.authorization_servers",
            ),
            "scopes_supported": _strings(
                protected.get("scopes_supported"),
                "protected_resource.scopes_supported",
            ),
            "bearer_methods_supported": _strings(
                protected.get("bearer_methods_supported"),
                "protected_resource.bearer_methods_supported",
            ),
        }
        if "header" not in protected_resource["bearer_methods_supported"]:
            raise _schema_error("protected_resource.bearer_methods_supported")

        authorization = _mapping(
            payload.get("authorization_server"), "authorization_server"
        )
        authorization_server = {
            "issuer": _string(
                authorization.get("issuer"), "authorization_server.issuer"
            ),
            "authorization_endpoint": _string(
                authorization.get("authorization_endpoint"),
                "authorization_server.authorization_endpoint",
            ),
            "token_endpoint": _string(
                authorization.get("token_endpoint"),
                "authorization_server.token_endpoint",
            ),
            "code_challenge_methods_supported": _strings(
                authorization.get("code_challenge_methods_supported"),
                "authorization_server.code_challenge_methods_supported",
            ),
            "registered_redirect_uris": _strings(
                authorization.get("registered_redirect_uris"),
                "authorization_server.registered_redirect_uris",
            ),
        }
        if "S256" not in authorization_server["code_challenge_methods_supported"]:
            raise _schema_error("authorization_server.code_challenge_methods_supported")

        policy = _mapping(payload.get("policy"), "policy")
        normalized_policy = {
            "max_tool_payload_bytes": _integer(
                policy.get("max_tool_payload_bytes"),
                "policy.max_tool_payload_bytes",
                minimum=1,
            )
        }

        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, Sequence) or isinstance(
            raw_cases, (str, bytes, bytearray)
        ):
            raise _schema_error("cases")
        if not raw_cases:
            raise _schema_error("cases")

        cases: list[ConformanceCase] = []
        seen_case_ids: set[str] = set()
        for index, raw_case in enumerate(raw_cases):
            case_path = f"cases[{index}]"
            case = _mapping(raw_case, case_path)
            case_id = _string(case.get("id"), f"{case_path}.id")
            assert case_id is not None
            if case_id in seen_case_ids:
                raise _schema_error(f"{case_path}.id")
            seen_case_ids.add(case_id)
            kind = _string(case.get("kind"), f"{case_path}.kind")
            assert kind is not None
            if kind not in {"positive", "negative"}:
                raise _schema_error(f"{case_path}.kind")
            behavior = _string(case.get("behavior"), f"{case_path}.behavior")
            assert behavior is not None
            boundary = _string(
                case.get("failure_boundary"), f"{case_path}.failure_boundary"
            )
            assert boundary is not None
            if boundary not in FAILURE_BOUNDARIES:
                raise _schema_error(f"{case_path}.failure_boundary")
            expected_failure = _string(
                case.get("expected_failure"),
                f"{case_path}.expected_failure",
                allow_none=True,
            )
            if kind == "positive" and expected_failure is not None:
                raise _schema_error(f"{case_path}.expected_failure")
            if kind == "negative" and expected_failure not in FAILURE_CATEGORIES:
                raise _schema_error(f"{case_path}.expected_failure")

            request = _mapping(
                case.get("authorization_request"),
                f"{case_path}.authorization_request",
            )
            pkce = _mapping(
                request.get("pkce"), f"{case_path}.authorization_request.pkce"
            )
            normalized_request = {
                "resource": _string(
                    request.get("resource"),
                    f"{case_path}.authorization_request.resource",
                    allow_none=True,
                ),
                "scopes": _strings(
                    request.get("scopes"),
                    f"{case_path}.authorization_request.scopes",
                ),
                "redirect_uri": _string(
                    request.get("redirect_uri"),
                    f"{case_path}.authorization_request.redirect_uri",
                ),
                "pkce": {
                    "method": _string(
                        pkce.get("method"),
                        f"{case_path}.authorization_request.pkce.method",
                    ),
                    "verifier_ref": _string(
                        pkce.get("verifier_ref"),
                        f"{case_path}.authorization_request.pkce.verifier_ref",
                    ),
                    "challenge_ref": _string(
                        pkce.get("challenge_ref"),
                        f"{case_path}.authorization_request.pkce.challenge_ref",
                    ),
                },
                "client_key_ref": _string(
                    request.get("client_key_ref"),
                    f"{case_path}.authorization_request.client_key_ref",
                ),
            }

            claims = _mapping(case.get("token_claims"), f"{case_path}.token_claims")
            normalized_claims = {
                "audience": _string(
                    claims.get("audience"), f"{case_path}.token_claims.audience"
                ),
                "issuer": _string(
                    claims.get("issuer"), f"{case_path}.token_claims.issuer"
                ),
                "resource": _string(
                    claims.get("resource"),
                    f"{case_path}.token_claims.resource",
                    allow_none=True,
                ),
                "scopes": _strings(
                    claims.get("scopes"),
                    f"{case_path}.token_claims.scopes",
                    allow_empty=True,
                ),
                "confirmation_key_ref": _string(
                    claims.get("confirmation_key_ref"),
                    f"{case_path}.token_claims.confirmation_key_ref",
                ),
            }

            tool = _mapping(case.get("tool_call"), f"{case_path}.tool_call")
            normalized_tool = {
                "name": _string(tool.get("name"), f"{case_path}.tool_call.name"),
                "required_scope": _string(
                    tool.get("required_scope"),
                    f"{case_path}.tool_call.required_scope",
                ),
                "payload_bytes": _integer(
                    tool.get("payload_bytes"),
                    f"{case_path}.tool_call.payload_bytes",
                ),
                "state_change": _boolean(
                    tool.get("state_change"), f"{case_path}.tool_call.state_change"
                ),
                "approval": _boolean(
                    tool.get("approval"), f"{case_path}.tool_call.approval"
                ),
            }

            transport = _mapping(case.get("transport"), f"{case_path}.transport")
            normalized_transport = {
                "forward_authorization": _boolean(
                    transport.get("forward_authorization"),
                    f"{case_path}.transport.forward_authorization",
                )
            }
            cases.append(
                ConformanceCase(
                    case_id=case_id,
                    kind=kind,
                    behavior=behavior,
                    failure_boundary=boundary,
                    expected_failure=expected_failure,
                    authorization_request=normalized_request,
                    token_claims=normalized_claims,
                    tool_call=normalized_tool,
                    transport=normalized_transport,
                )
            )

        assert protected_resource["resource"] is not None
        assert authorization_server["issuer"] is not None
        if (
            authorization_server["issuer"]
            not in protected_resource["authorization_servers"]
        ):
            raise _schema_error("protected_resource.authorization_servers")
        return cls(
            schema_version=schema_version,
            synthetic=True,
            protected_resource=protected_resource,
            authorization_server=authorization_server,
            policy=normalized_policy,
            cases=tuple(cases),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible copy of the normalized manifest."""

        return {
            "schema_version": self.schema_version,
            "synthetic": self.synthetic,
            "protected_resource": _copy(self.protected_resource),
            "authorization_server": _copy(self.authorization_server),
            "policy": _copy(self.policy),
            "cases": [
                {
                    "id": case.case_id,
                    "kind": case.kind,
                    "behavior": case.behavior,
                    "failure_boundary": case.failure_boundary,
                    "expected_failure": case.expected_failure,
                    "authorization_request": _copy(case.authorization_request),
                    "token_claims": _copy(case.token_claims),
                    "tool_call": _copy(case.tool_call),
                    "transport": _copy(case.transport),
                }
                for case in self.cases
            ],
        }


@dataclass(frozen=True)
class ConformanceCaseResult:
    """Sanitized result for one manifest case."""

    case_id: str
    failure_category: str | None
    failure_boundary: str | None
    expected_failure: str | None
    conformant: bool

    def to_dict(self) -> dict[str, str | None]:
        """Return the only fields permitted in public runner output."""

        return {
            "case_id": self.case_id,
            "failure_category": self.failure_category,
        }


@dataclass(frozen=True)
class ConformanceReport:
    """Deterministic, privacy-safe report produced by :func:`run_conformance`."""

    results: tuple[ConformanceCaseResult, ...]

    @property
    def ok(self) -> bool:
        """Return whether every positive and negative case met its declaration."""

        return all(result.conformant for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Return case IDs and failure categories, with no fixture contents."""

        return {"results": [result.to_dict() for result in self.results]}

    def to_json(self) -> str:
        """Return canonical JSON suitable for a deterministic test artifact."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class ConformanceTransport(Protocol):
    """Minimal transport contract used by the offline runner."""

    def discover_protected_resource(
        self, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def discover_authorization_server(
        self, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def authorize(
        self,
        case: ConformanceCase,
        protected_resource: Mapping[str, Any],
        authorization_server: Mapping[str, Any],
    ) -> Any: ...

    def exchange_code(
        self,
        case: ConformanceCase,
        authorization_code: Any,
        authorization_server: Mapping[str, Any],
    ) -> Any: ...

    def call_tool(
        self,
        case: ConformanceCase,
        access_token: Any,
        protected_resource: Mapping[str, Any],
        authorization_server: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> Any: ...


@dataclass(frozen=True, repr=False)
class _AuthorizationCode:
    """Opaque in-process authorization code."""

    case_id: str

    def __repr__(self) -> str:
        return "<opaque-authorization-code>"


@dataclass(frozen=True, repr=False)
class _AccessToken:
    """Opaque in-process access token with claims kept out of reports."""

    case_id: str
    claims: Mapping[str, Any]
    client_key_ref: str

    def __repr__(self) -> str:
        return "<opaque-access-token>"


class MockAuthorizationTransport:
    """Execute the conformance flow without an HTTP client or network socket.

    The transport records only operation names for optional test inspection.
    It never stores or returns an authorization header or token string.
    """

    def __init__(self) -> None:
        self.operations: list[str] = []

    def _record(self, operation: str) -> None:
        self.operations.append(operation)

    def discover_protected_resource(
        self, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self._record("protected_resource_metadata")
        return copy.deepcopy(dict(metadata))

    def discover_authorization_server(
        self, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self._record("authorization_server_metadata")
        return copy.deepcopy(dict(metadata))

    def authorize(
        self,
        case: ConformanceCase,
        protected_resource: Mapping[str, Any],
        authorization_server: Mapping[str, Any],
    ) -> _AuthorizationCode:
        self._record("authorize")
        request = case.authorization_request
        if request["resource"] is None:
            raise ConformanceViolation(
                "missing_resource_indicator", "authorization_request"
            )
        if request["resource"] != protected_resource["resource"]:
            raise ConformanceViolation(
                "missing_resource_indicator", "authorization_request"
            )
        if not set(request["scopes"]).issubset(
            set(protected_resource["scopes_supported"])
        ):
            raise ConformanceViolation(
                "insufficient_tool_scope", "authorization_request"
            )
        if (
            request["redirect_uri"]
            not in authorization_server["registered_redirect_uris"]
        ):
            raise ConformanceViolation("redirect_mismatch", "redirect_validation")
        if (
            request["pkce"]["method"]
            not in authorization_server["code_challenge_methods_supported"]
        ):
            raise ConformanceViolation("invalid_pkce", "authorization_code_exchange")
        return _AuthorizationCode(case.case_id)

    def exchange_code(
        self,
        case: ConformanceCase,
        authorization_code: _AuthorizationCode,
        authorization_server: Mapping[str, Any],
    ) -> _AccessToken:
        del authorization_code, authorization_server
        self._record("token_exchange")
        pkce = case.authorization_request["pkce"]
        if pkce["verifier_ref"] != pkce["challenge_ref"]:
            raise ConformanceViolation("invalid_pkce", "authorization_code_exchange")
        return _AccessToken(
            case_id=case.case_id,
            claims=copy.deepcopy(dict(case.token_claims)),
            client_key_ref=case.authorization_request["client_key_ref"],
        )

    def call_tool(
        self,
        case: ConformanceCase,
        access_token: _AccessToken,
        protected_resource: Mapping[str, Any],
        authorization_server: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> Mapping[str, str]:
        self._record("tool_call")
        if case.transport["forward_authorization"]:
            raise ConformanceViolation("token_passthrough", "transport_policy")

        claims = access_token.claims
        if claims["audience"] != protected_resource["resource"]:
            raise ConformanceViolation("wrong_audience", "token_validation")
        if claims["issuer"] != authorization_server["issuer"]:
            raise ConformanceViolation("wrong_audience", "token_validation")
        if claims["resource"] is None:
            raise ConformanceViolation("missing_resource_indicator", "token_validation")
        if claims["resource"] != protected_resource["resource"]:
            raise ConformanceViolation("missing_resource_indicator", "token_validation")
        if claims["confirmation_key_ref"] != access_token.client_key_ref:
            raise ConformanceViolation("token_binding_mismatch", "token_validation")

        tool = case.tool_call
        if tool["required_scope"] not in claims["scopes"]:
            raise ConformanceViolation("insufficient_tool_scope", "tool_authorization")
        if tool["payload_bytes"] > policy["max_tool_payload_bytes"]:
            raise ConformanceViolation("oversized_payload", "payload_policy")
        if tool["state_change"] and not tool["approval"]:
            raise ConformanceViolation("unapproved_state_change", "state_change_policy")
        return {"status": "completed"}


def _coerce_manifest(
    value: ConformanceManifest | Mapping[str, Any],
) -> ConformanceManifest:
    if isinstance(value, ConformanceManifest):
        return value
    if isinstance(value, Mapping):
        return ConformanceManifest.from_mapping(value)
    raise ConformanceFixtureError("Expected an authorization conformance manifest.")


def load_conformance_manifest(
    path: str | Path | None = None,
) -> ConformanceManifest:
    """Load and validate the committed synthetic conformance manifest."""

    fixture_path = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        del error
        raise ConformanceFixtureError(
            "Unable to load the authorization conformance manifest."
        ) from None
    return ConformanceManifest.from_mapping(payload)


load_fixture_manifest = load_conformance_manifest
load_authorization_conformance_manifest = load_conformance_manifest


def manifest_case_ids(
    manifest: ConformanceManifest | Mapping[str, Any],
) -> frozenset[str]:
    """Return the declared case IDs as a stable set."""

    return frozenset(case.case_id for case in _coerce_manifest(manifest).cases)


def validate_case_coverage(
    manifest: ConformanceManifest | Mapping[str, Any],
    covered_case_ids: Sequence[str],
) -> None:
    """Fail closed when focused tests omit or invent a manifest case."""

    declared = manifest_case_ids(manifest)
    covered = frozenset(covered_case_ids)
    missing = declared - covered
    unexpected = covered - declared
    if missing or unexpected:
        raise ConformanceCoverageError(missing=missing, unexpected=unexpected)


def _execute_case(
    case: ConformanceCase,
    manifest: ConformanceManifest,
    transport: ConformanceTransport,
) -> None:
    protected_resource = transport.discover_protected_resource(
        manifest.protected_resource
    )
    authorization_server = transport.discover_authorization_server(
        manifest.authorization_server
    )
    authorization_code = transport.authorize(
        case,
        protected_resource,
        authorization_server,
    )
    access_token = transport.exchange_code(
        case,
        authorization_code,
        authorization_server,
    )
    transport.call_tool(
        case,
        access_token,
        protected_resource,
        authorization_server,
        manifest.policy,
    )


def run_conformance(
    manifest: ConformanceManifest | Mapping[str, Any] | None = None,
    *,
    transport: ConformanceTransport | None = None,
    covered_case_ids: Sequence[str] | None = None,
) -> ConformanceReport:
    """Run every declared case through a local mocked authorization flow.

    Args:
        manifest: Validated manifest or decoded manifest mapping. The committed
            fixture is loaded when omitted.
        transport: Optional compatible mocked transport. The default is local
            and never imports an HTTP client.
        covered_case_ids: Optional focused-test declaration. When supplied,
            coverage must match the manifest exactly.

    Returns:
        A deterministic report whose public payload contains only case IDs and
        failure categories.
    """

    effective_manifest = _coerce_manifest(
        load_conformance_manifest() if manifest is None else manifest
    )
    if covered_case_ids is not None:
        validate_case_coverage(effective_manifest, covered_case_ids)
    effective_transport = (
        transport if transport is not None else MockAuthorizationTransport()
    )
    results: list[ConformanceCaseResult] = []

    for case in sorted(effective_manifest.cases, key=lambda item: item.case_id):
        actual_category: str | None = None
        actual_boundary: str | None = None
        try:
            _execute_case(case, effective_manifest, effective_transport)
        except ConformanceViolation as error:
            actual_category = error.category
            actual_boundary = error.boundary
        except Exception:
            actual_category = _RESULT_RUNNER_ERROR
            actual_boundary = None

        if case.expected_failure is not None and actual_category is None:
            actual_category = _RESULT_FAILURE_MISSING

        conformant = (
            actual_category is None
            if case.expected_failure is None
            else actual_category == case.expected_failure
            and actual_boundary == case.failure_boundary
        )
        results.append(
            ConformanceCaseResult(
                case_id=case.case_id,
                failure_category=actual_category,
                failure_boundary=actual_boundary,
                expected_failure=case.expected_failure,
                conformant=conformant,
            )
        )
    return ConformanceReport(results=tuple(results))


run_authorization_conformance = run_conformance


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def render_conformance_matrix(
    manifest: ConformanceManifest | Mapping[str, Any] | None = None,
) -> str:
    """Render a concise deterministic Markdown matrix from the manifest."""

    effective_manifest = _coerce_manifest(
        load_conformance_manifest() if manifest is None else manifest
    )
    lines = [
        "<!-- Generated by render_conformance_matrix; keep the manifest canonical. -->",
        "# Offline MCP authorization conformance matrix",
        "",
        "> Synthetic, offline regression coverage for authorization boundaries. "
        "This is not a certification claim.",
        "",
        "| Case | Protocol behavior | Boundary | Declared outcome |",
        "| --- | --- | --- | --- |",
    ]
    for case in sorted(effective_manifest.cases, key=lambda item: item.case_id):
        outcome = "pass" if case.expected_failure is None else case.expected_failure
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(case.case_id),
                    _markdown_cell(case.behavior),
                    _markdown_cell(case.failure_boundary),
                    _markdown_cell(outcome),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


render_authorization_matrix = render_conformance_matrix


def write_conformance_matrix(
    path: str | Path,
    manifest: ConformanceManifest | Mapping[str, Any] | None = None,
) -> None:
    """Write the generated matrix with UTF-8 text and deterministic newlines."""

    Path(path).write_text(
        render_conformance_matrix(manifest),
        encoding="utf-8",
        newline="\n",
    )
