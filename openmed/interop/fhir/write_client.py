"""Bounded execution of approved FHIR R4 writes through trusted injections.

Planning, human approval, token custody, lineage verification, admission and
durable attempt storage belong to caller-owned boundaries. No HTTP library,
OAuth flow, retry, compensating write, or clinical decision is supplied here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.interop.fhir_capability_preflight import (
    FHIRCapabilityMetadata,
    FHIRWriteInteraction,
    FHIRWritePlan,
    evaluate_fhir_write_plan,
)

from .versions import SUPPORTED_RESOURCE_TYPES

__all__ = [
    "FHIRCredentialCustody",
    "FHIRHTTPResponse",
    "FHIRPreparedWrite",
    "FHIRTransportRequest",
    "FHIRWriteClient",
    "FHIRWriteError",
    "FHIRWriteLedger",
    "FHIRWriteLimits",
    "FHIRWriteOutcome",
    "FHIRWriteStatus",
]

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BARE_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_KEY = re.compile(r"fhir-cw-v1-[0-9a-f]{64}\Z")
_HANDLE = re.compile(r"smart_[A-Za-z0-9_-]{43}\Z")
_ID = r"[A-Za-z0-9.\-]{1,64}"
_ETAG = re.compile(r'W/"' + _ID + r'"\Z')
_REASONS = frozenset(
    {
        "invalid_configuration",
        "invalid_plan",
        "invalid_payload",
        "invalid_lineage",
        "limit_exceeded",
        "unsupported_capability",
        "authorization_rejected",
        "authorization_expired",
        "credential_rejected",
        "ledger_unavailable",
        "idempotency_mismatch",
        "attempt_in_progress",
        "dispatch_uncertain",
        "redirect_received",
        "response_limit_exceeded",
        "unexpected_response",
        "server_rejected",
        "server_conflict",
        "precondition_failed",
        "precondition_required",
        "server_acknowledged",
        "receipt_uncertain",
    }
)


class FHIRWriteError(ValueError):
    """A closed, value-free local rejection code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = (
            reason_code if reason_code in _REASONS else "invalid_configuration"
        )
        super().__init__(self.reason_code)


class FHIRWriteStatus(str, Enum):
    """Server acknowledgement, known refusal, conflict, or uncertain commit."""

    COMMITTED = "committed"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    UNKNOWN = "unknown_commit"


@dataclass(frozen=True, slots=True)
class FHIRWriteLimits:
    """Inclusive request/response limits forwarded to the trusted transport."""

    max_request_bytes: int = 1_048_576
    max_response_bytes: int = 262_144
    max_entries: int = 100
    max_json_nodes: int = 20_000
    max_json_depth: int = 32
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        for value, maximum in (
            (self.max_request_bytes, 16_777_216),
            (self.max_response_bytes, 16_777_216),
            (self.max_entries, 1000),
            (self.max_json_nodes, 100_000),
            (self.max_json_depth, 64),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise FHIRWriteError("invalid_configuration")
        if (
            type(self.timeout_seconds) not in (float, int)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 120
        ):
            raise FHIRWriteError("invalid_configuration")


@dataclass(frozen=True, slots=True, repr=False)
class FHIRTransportRequest:
    """Private exact wire request; transport must enforce caps and avoid retries.

    Headers and payload may contain PHI or credentials. Never log this object,
    its fields, URLs, bodies, or driver exceptions. Redirect following and all
    automatic transport retries must be disabled by the injected implementation.
    """

    method: str
    url: str
    headers: tuple[tuple[str, str], ...]
    body: bytes
    timeout_seconds: float
    max_response_bytes: int
    follow_redirects: bool = False
    retries: int = 0

    def __repr__(self) -> str:
        """Hide sensitive request content."""
        return "FHIRTransportRequest(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class FHIRHTTPResponse:
    """Private response retained only while classifying a bounded outcome."""

    status_code: int
    body: bytes = b""
    headers: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:
        """Hide all server response values."""
        return "FHIRHTTPResponse(<redacted>)"


@dataclass(frozen=True, slots=True)
class FHIRWriteOutcome:
    """Value-free execution receipt; counts describe proposed resource writes.

    ``committed`` means a valid server acknowledgement, including a conditional
    create that matched an existing resource. It does not prove clinical
    correctness or mean that every proposed write created a new resource.
    """

    status: FHIRWriteStatus
    reason_code: str
    action_digest: str
    payload_digest: str
    resource_count: int
    response_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.status) is not FHIRWriteStatus
            or self.reason_code not in _REASONS
            or not _valid_digest(self.action_digest)
            or not _valid_digest(self.payload_digest)
        ):
            raise FHIRWriteError("invalid_configuration")
        if (
            type(self.resource_count) is not int
            or self.resource_count < 1
            or self.response_digest is not None
            and not _valid_digest(self.response_digest)
        ):
            raise FHIRWriteError("invalid_configuration")

    @property
    def reconciliation_required(self) -> bool:
        """Require caller-owned reconciliation for every ambiguous attempt."""
        return self.status is FHIRWriteStatus.UNKNOWN

    def to_dict(self) -> dict[str, str | int | bool | None]:
        """Export only controlled codes, counts and keyed commitments."""
        return {
            "status": self.status.value,
            "reason_code": self.reason_code,
            "action_digest": self.action_digest,
            "payload_digest": self.payload_digest,
            "resource_count": self.resource_count,
            "response_digest": self.response_digest,
            "reconciliation_required": self.reconciliation_required,
        }


class FHIRCredentialCustody(Protocol):
    """Existing SMART custody dispatch interface, without token extraction."""

    def dispatch(
        self, handle: str, *, audience: str, required_scopes: Iterable[str]
    ) -> object:
        """Check custody bounds and invoke the sender bound by its trusted factory."""
        ...


class FHIRWriteLedger(Protocol):
    """Required caller-owned durable, atomic attempt ledger.

    Claim keys are globally unique within the endpoint's write namespace, not
    merely within a process. A claim remains reserved after a crash. ``finish``
    must atomically persist an immutable outcome for the same claim. Only exact
    True acknowledgements count. Failed claims must never silently release an
    uncertain attempt. This adapter supplies no in-memory production ledger.
    """

    def claim(self, idempotency_key: str, action_digest: str) -> bool:
        """Atomically reserve a never-before-dispatched key and action."""
        ...

    def lookup(self, idempotency_key: str) -> FHIRWriteOutcome | None:
        """Read a durable result; None includes pending or uncertain storage."""
        ...

    def finish(
        self, idempotency_key: str, action_digest: str, outcome: FHIRWriteOutcome
    ) -> bool:
        """Persist one matching outcome and acknowledge only after durable storage."""
        ...


@dataclass(frozen=True, slots=True, repr=False)
class FHIRPreparedWrite:
    """Frozen wire proposal for protected review, not proof of authorization.

    Lineage and request details stay private. The outer action digest binds the
    exact payload, predicate, version headers, endpoint, credential handle,
    limits, scopes and lineage snapshot. Existing lineage approval alone does
    not bind all of these wire details; require an ApprovalReceipt for this
    outer action as well as the existing lineage gate at dispatch.
    """

    action_digest: str
    payload_digest: str
    resource_count: int
    idempotency_key: str
    _method: str = field(repr=False)
    _path: str = field(repr=False)
    _headers: tuple[tuple[str, str], ...] = field(repr=False)
    _body: bytes = field(repr=False)
    _handle: str = field(repr=False)
    _lineage: bytes = field(repr=False)
    _scopes: tuple[str, ...] = field(repr=False)
    _types: tuple[str, ...] = field(repr=False)

    def __repr__(self) -> str:
        """Hide private wire and lineage values."""
        return "FHIRPreparedWrite(<redacted>)"

    @property
    def payload(self) -> dict[str, Any]:
        """Return a detached payload only for trusted protected review or gates."""
        return json.loads(self._body)

    def to_dict(self) -> dict[str, str | int]:
        """Return the metadata-only outer review commitment."""
        return {
            "action_digest": self.action_digest,
            "payload_digest": self.payload_digest,
            "resource_count": self.resource_count,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(slots=True)
class _Attempt:
    prepared: FHIRPreparedWrite
    receipt: ApprovalReceipt
    submitted: bool = False
    response: FHIRHTTPResponse | None = None


class FHIRWriteClient:
    """Execute only the declared R4 subset through trusted caller boundaries.

    Args:
        audience: Operator-configured HTTPS FHIR base, never chosen by a plan.
        secret: Private commitment key of at least 32 bytes.
        capabilities: R4 metadata observed for this exact configured audience.
        transport: One-shot bounded sender; must not log, redirect, or retry.
        custody_factory: Trusted factory binding the existing SMART custody
            sender. The operator may retain the broker to store credentials.
        ledger: Durable atomic reservation and outcome receipt storage.
        authorize: Local verifier of the exact receipt, active grant, patient
            scope, default-off admission/emergency stop, and fresh update
            evidence. Return exact True only after all applicable gates pass.
        verify_lineage: Re-run the existing lineage gate against this exact
            detached payload, original plans and targets; return its manifest.
            A copied manifest alone does not prove coverage of payload fields.
        clock: Aware clock checked before custody and at the send boundary.
        limits: Inclusive limits. The transport must honor the response cap
            while reading, since this adapter cannot undo driver allocation.
        scope_context: Operator-selected SMART context, never payload-selected.
        allow_insecure_loopback: Explicit HTTP loopback opt-in for offline labs.
    """

    def __init__(
        self,
        *,
        audience: str,
        secret: bytes,
        capabilities: FHIRCapabilityMetadata,
        transport: Callable[[FHIRTransportRequest], FHIRHTTPResponse],
        custody_factory: Callable[[Callable[[str, str], None]], FHIRCredentialCustody],
        ledger: FHIRWriteLedger,
        authorize: Callable[[FHIRPreparedWrite, ApprovalReceipt], bool],
        verify_lineage: Callable[[FHIRPreparedWrite], object],
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        limits: FHIRWriteLimits = FHIRWriteLimits(),
        scope_context: str = "user",
        allow_insecure_loopback: bool = False,
    ) -> None:
        if (
            type(secret) is not bytes
            or len(secret) < 32
            or type(limits) is not FHIRWriteLimits
            or type(capabilities) is not FHIRCapabilityMetadata
            or scope_context not in {"user", "patient", "system"}
            or type(allow_insecure_loopback) is not bool
        ):
            raise FHIRWriteError("invalid_configuration")
        if any(
            not callable(c)
            for c in (transport, custody_factory, authorize, verify_lineage, clock)
        ) or any(
            not callable(getattr(ledger, n, None))
            for n in ("claim", "lookup", "finish")
        ):
            raise FHIRWriteError("invalid_configuration")
        self._audience = _audience(audience, allow_insecure_loopback)
        self._secret, self._capabilities, self._limits = secret, capabilities, limits
        self._transport, self._ledger = transport, ledger
        self._authorize, self._verify_lineage, self._clock = (
            authorize,
            verify_lineage,
            clock,
        )
        self._scope_context = scope_context
        self._active: ContextVar[_Attempt | None] = ContextVar(
            "fhir_write_attempt", default=None
        )
        try:
            self._custody = custody_factory(self._send)
        except Exception:
            raise FHIRWriteError("invalid_configuration") from None
        if not callable(getattr(self._custody, "dispatch", None)):
            raise FHIRWriteError("invalid_configuration")

    def prepare_conditional(
        self,
        plan: object,
        resource: Mapping[str, Any],
        *,
        credential_handle: str,
        lineage: object,
        precondition: object | None = None,
    ) -> FHIRPreparedWrite:
        """Snapshot an existing ConditionalWritePlan and UpdatePrecondition.

        Supports conditional POST with If-None-Exist, or conditional PUT with
        required If-Match. Predicate bytes and the planner's idempotency key
        are preserved. Match readiness and fresh reads belong to authorization.
        No plan, precondition, lineage approval or token is constructed here.
        """
        try:
            kind = getattr(plan, "kind")
            kind = getattr(kind, "value", kind)
            resource_type = getattr(plan, "resource_type")
            predicate = getattr(getattr(plan, "predicate"), "canonical_query")
            key = getattr(plan, "idempotency_key")
            if kind not in {
                "create",
                "update",
            } or resource_type not in SUPPORTED_RESOURCE_TYPES - {"Bundle"}:
                raise FHIRWriteError("invalid_plan")
            _predicate(predicate)
            body = _serialize(resource, self._limits)
            payload = _parse(body, self._limits, "invalid_payload")
            if (
                not isinstance(payload, dict)
                or payload.get("resourceType") != resource_type
            ):
                raise FHIRWriteError("invalid_payload")
            if "id" in payload and (
                type(payload["id"]) is not str
                or re.fullmatch(_ID, payload["id"]) is None
            ):
                raise FHIRWriteError("invalid_payload")
            headers = [("If-None-Exist", predicate)] if kind == "create" else []
            if kind == "update":
                version = getattr(precondition, "if_match", None)
                if type(version) is not str or not _ETAG.fullmatch(version):
                    raise FHIRWriteError("invalid_plan")
                headers.append(("If-Match", version))
            elif precondition is not None:
                raise FHIRWriteError("invalid_plan")
            self._require_capability(
                FHIRWritePlan(FHIRWriteInteraction(kind), resource_type, True)
            )
            manifest = _lineage(lineage, self._limits)
            if any(
                record["plan_key"] != key for record in json.loads(manifest)["records"]
            ):
                raise FHIRWriteError("invalid_lineage")
            return self._prepare(
                "POST" if kind == "create" else "PUT",
                resource_type if kind == "create" else resource_type + "?" + predicate,
                tuple(headers),
                body,
                key,
                credential_handle,
                manifest,
                (resource_type,),
                1,
            )
        except FHIRWriteError:
            raise
        except Exception:
            raise FHIRWriteError("invalid_plan") from None

    def prepare_transaction(
        self,
        transaction: object,
        *,
        idempotency_key: str,
        credential_handle: str,
        lineage: object,
    ) -> FHIRPreparedWrite:
        """Consume exact AssembledTransaction bytes without rewriting or splitting.

        Its existing bundle SHA-256 and entry count must match. The declared
        subset is POST creates and version-guarded PUT updates, plus the
        assembler's final Provenance POST. Missing PUT versions are refused.
        The action ledger's existing idempotency identity is passed unchanged.
        """
        try:
            body = getattr(transaction, "serialized")
            if type(body) is not bytes or len(body) > self._limits.max_request_bytes:
                raise FHIRWriteError("limit_exceeded")
            if (
                getattr(transaction, "bundle_digest")
                != "sha256:" + hashlib.sha256(body).hexdigest()
            ):
                raise FHIRWriteError("invalid_payload")
            payload = _parse(body, self._limits, "invalid_payload")
            if (
                type(payload) is not dict
                or payload.get("resourceType") != "Bundle"
                or payload.get("type") != "transaction"
                or set(payload) != {"resourceType", "type", "entry"}
            ):
                raise FHIRWriteError("invalid_payload")
            entries = payload["entry"]
            if (
                type(entries) is not list
                or not 2 <= len(entries) <= self._limits.max_entries
                or getattr(transaction, "entry_count") != len(entries)
            ):
                raise FHIRWriteError("invalid_payload")
            types = []
            full_urls = set()
            for index, entry in enumerate(entries):
                if (
                    type(entry) is not dict
                    or set(entry) != {"fullUrl", "resource", "request"}
                    or type(entry["resource"]) is not dict
                ):
                    raise FHIRWriteError("invalid_payload")
                full_url = entry["fullUrl"]
                if (
                    type(full_url) is not str
                    or not full_url.startswith("urn:uuid:")
                    or str(UUID(full_url[9:])) != full_url[9:]
                    or full_url in full_urls
                ):
                    raise FHIRWriteError("invalid_payload")
                full_urls.add(full_url)
                resource_type = entry["resource"].get("resourceType")
                request = entry["request"]
                if (
                    resource_type not in SUPPORTED_RESOURCE_TYPES - {"Bundle"}
                    or type(request) is not dict
                ):
                    raise FHIRWriteError("invalid_payload")
                if index < len(entries) - 1 and resource_type == "Provenance":
                    raise FHIRWriteError("invalid_payload")
                if index == len(entries) - 1 and (
                    resource_type != "Provenance"
                    or request != {"method": "POST", "url": "Provenance"}
                ):
                    raise FHIRWriteError("invalid_payload")
                _transaction_request(request, resource_type)
                interaction = (
                    FHIRWriteInteraction.CREATE
                    if request["method"] == "POST"
                    else FHIRWriteInteraction.UPDATE
                )
                conditional = "ifNoneExist" in request or "?" in request["url"]
                self._require_capability(
                    FHIRWritePlan(interaction, resource_type, conditional)
                )
                types.append(resource_type)
            self._require_capability(FHIRWritePlan(FHIRWriteInteraction.TRANSACTION))
            return self._prepare(
                "POST",
                "",
                (),
                body,
                idempotency_key,
                credential_handle,
                _lineage(lineage, self._limits),
                tuple(types),
                len(types) - 1,
            )
        except FHIRWriteError:
            raise
        except Exception:
            raise FHIRWriteError("invalid_payload") from None

    def submit(
        self, prepared: FHIRPreparedWrite, receipt: ApprovalReceipt
    ) -> FHIRWriteOutcome:
        """Submit once after repeated authorization; never retry or compensate.

        A duplicate exact key reads its durable result. A reserved key without
        a verifiable outcome requires reconciliation, including after restart.
        Transport failures after entering the sender are always ambiguous.
        """
        self._require_prepared(prepared)
        try:
            self._guard(prepared, receipt)
        except FHIRWriteError as exc:
            return self._outcome(prepared, FHIRWriteStatus.REJECTED, exc.reason_code)
        try:
            claimed = self._ledger.claim(
                prepared.idempotency_key, prepared.action_digest
            )
            if claimed is not True:
                if claimed is not False:
                    raise FHIRWriteError("ledger_unavailable")
                previous = self._ledger.lookup(prepared.idempotency_key)
                if previous is None:
                    return self._outcome(
                        prepared, FHIRWriteStatus.UNKNOWN, "attempt_in_progress"
                    )
                if (
                    type(previous) is not FHIRWriteOutcome
                    or previous.action_digest != prepared.action_digest
                    or previous.payload_digest != prepared.payload_digest
                    or previous.resource_count != prepared.resource_count
                ):
                    return self._outcome(
                        prepared, FHIRWriteStatus.CONFLICT, "idempotency_mismatch"
                    )
                return previous
        except Exception:
            return self._outcome(
                prepared, FHIRWriteStatus.REJECTED, "ledger_unavailable"
            )
        attempt = _Attempt(prepared, receipt)
        token = self._active.set(attempt)
        try:
            try:
                self._guard(prepared, receipt)
                self._custody.dispatch(
                    prepared._handle,
                    audience=self._audience,
                    required_scopes=prepared._scopes,
                )
                if not attempt.submitted or attempt.response is None:
                    raise FHIRWriteError("credential_rejected")
                outcome = self._classify(prepared, attempt.response)
            except Exception:
                outcome = self._outcome(
                    prepared,
                    FHIRWriteStatus.UNKNOWN
                    if attempt.submitted
                    else FHIRWriteStatus.REJECTED,
                    "dispatch_uncertain"
                    if attempt.submitted
                    else "credential_rejected",
                )
            try:
                stored = self._ledger.finish(
                    prepared.idempotency_key, prepared.action_digest, outcome
                )
            except Exception:
                stored = False
            if stored is not True:
                return self._outcome(
                    prepared, FHIRWriteStatus.UNKNOWN, "receipt_uncertain"
                )
            return outcome
        finally:
            self._active.reset(token)

    def reconcile(self, prepared: FHIRPreparedWrite) -> FHIRWriteOutcome:
        """Read durable evidence only; never contact the server or dispatch again.

        Missing evidence remains unknown. A caller-owned recovery workflow must
        query the server under its own read authorization and review policy.
        """
        self._require_prepared(prepared)
        try:
            previous = self._ledger.lookup(prepared.idempotency_key)
            if (
                type(previous) is FHIRWriteOutcome
                and previous.action_digest == prepared.action_digest
                and previous.payload_digest == prepared.payload_digest
                and previous.resource_count == prepared.resource_count
            ):
                return previous
        except Exception:
            pass
        return self._outcome(prepared, FHIRWriteStatus.UNKNOWN, "receipt_uncertain")

    def _send(self, audience: str, authorization: str) -> None:
        attempt = self._active.get()
        if (
            attempt is None
            or attempt.submitted
            or audience != self._audience
            or type(authorization) is not str
            or re.fullmatch(r"Bearer [A-Za-z0-9._~+/-]+={0,}", authorization) is None
        ):
            raise FHIRWriteError("credential_rejected")
        self._guard(attempt.prepared, attempt.receipt)
        prepared = attempt.prepared
        request = FHIRTransportRequest(
            prepared._method,
            self._audience + ("/" + prepared._path if prepared._path else ""),
            (
                ("Content-Type", "application/fhir+json"),
                ("Accept", "application/fhir+json"),
                ("Prefer", "return=representation"),
                ("Idempotency-Key", prepared.idempotency_key),
                *prepared._headers,
                ("Authorization", authorization),
            ),
            prepared._body,
            self._limits.timeout_seconds,
            self._limits.max_response_bytes,
        )
        attempt.submitted = True
        attempt.response = self._transport(request)

    def _require_capability(self, plan: FHIRWritePlan) -> None:
        if (
            self._capabilities.fhir_version != "4.0.1"
            or not evaluate_fhir_write_plan(self._capabilities, plan).is_compatible
        ):
            raise FHIRWriteError("unsupported_capability")

    def _prepare(
        self,
        method: str,
        path: str,
        headers: tuple[tuple[str, str], ...],
        body: bytes,
        key: str,
        handle: str,
        lineage: bytes,
        types: tuple[str, ...],
        count: int,
    ) -> FHIRPreparedWrite:
        if (
            type(key) is not str
            or not _KEY.fullmatch(key)
            or type(handle) is not str
            or not _HANDLE.fullmatch(handle)
        ):
            raise FHIRWriteError("invalid_plan")
        operations: dict[str, set[str]] = {}
        if path:
            operations[types[0]] = {"c" if method == "POST" else "u"}
        else:
            for entry in json.loads(body)["entry"]:
                operations.setdefault(entry["resource"]["resourceType"], set()).add(
                    "c" if entry["request"]["method"] == "POST" else "u"
                )
        scopes = tuple(
            f"{self._scope_context}/{resource_type}."
            + "".join(operation for operation in "cu" if operation in required)
            for resource_type, required in sorted(operations.items())
        )
        if (
            len(
                {record["resource_handle"] for record in json.loads(lineage)["records"]}
            )
            != count
        ):
            raise FHIRWriteError("invalid_lineage")
        prepared = FHIRPreparedWrite(
            "",
            self._digest(b"payload", body),
            count,
            key,
            method,
            path,
            headers,
            body,
            handle,
            lineage,
            scopes,
            types,
        )
        return FHIRPreparedWrite(
            self._action(prepared),
            prepared.payload_digest,
            count,
            key,
            method,
            path,
            headers,
            body,
            handle,
            lineage,
            scopes,
            types,
        )

    def _action(self, prepared: FHIRPreparedWrite) -> str:
        content = json.dumps(
            {
                "audience": self._audience,
                "method": prepared._method,
                "path": prepared._path,
                "headers": prepared._headers,
                "body": self._digest(b"payload", prepared._body),
                "key": prepared.idempotency_key,
                "credential": prepared._handle,
                "lineage": self._digest(b"lineage", prepared._lineage),
                "types": prepared._types,
                "count": prepared.resource_count,
                "scopes": prepared._scopes,
                "limits": [
                    self._limits.max_request_bytes,
                    self._limits.max_response_bytes,
                    self._limits.max_entries,
                    self._limits.max_json_nodes,
                    self._limits.max_json_depth,
                    self._limits.timeout_seconds,
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return self._digest(b"action", content)

    def _digest(self, domain: bytes, content: bytes) -> str:
        return (
            "sha256:"
            + hmac.new(
                self._secret,
                b"openmed.fhir.write-client.v1\0" + domain + b"\0" + content,
                hashlib.sha256,
            ).hexdigest()
        )

    def _require_prepared(self, prepared: FHIRPreparedWrite) -> None:
        try:
            valid = (
                type(prepared) is FHIRPreparedWrite
                and _valid_digest(prepared.action_digest)
                and hmac.compare_digest(prepared.action_digest, self._action(prepared))
                and hmac.compare_digest(
                    prepared.payload_digest, self._digest(b"payload", prepared._body)
                )
            )
        except Exception:
            valid = False
        if not valid:
            raise FHIRWriteError("invalid_plan")

    def _guard(self, prepared: FHIRPreparedWrite, receipt: ApprovalReceipt) -> None:
        self._require_prepared(prepared)
        try:
            now = self._clock()
            if (
                type(now) is not datetime
                or now.tzinfo is None
                or now.utcoffset() is None
                or type(receipt) is not ApprovalReceipt
                or receipt.action_digest != prepared.action_digest
            ):
                raise FHIRWriteError("authorization_rejected")
            instant = now.timestamp()
            if not receipt.consumed_at <= instant < receipt.expires_at:
                raise FHIRWriteError("authorization_expired")
            if self._authorize(prepared, receipt) is not True:
                raise FHIRWriteError("authorization_rejected")
            if not hmac.compare_digest(
                _lineage(self._verify_lineage(prepared), self._limits),
                prepared._lineage,
            ):
                raise FHIRWriteError("invalid_lineage")
            # Trusted gates may perform fresh local reads. Approval must still
            # be live after they complete, at the actual effect boundary.
            current = self._clock()
            if (
                type(current) is not datetime
                or current.tzinfo is None
                or current.utcoffset() is None
            ):
                raise FHIRWriteError("authorization_rejected")
            if not receipt.consumed_at <= current.timestamp() < receipt.expires_at:
                raise FHIRWriteError("authorization_expired")
        except FHIRWriteError:
            raise
        except Exception:
            raise FHIRWriteError("authorization_rejected") from None

    def _outcome(
        self,
        prepared: FHIRPreparedWrite,
        status: FHIRWriteStatus,
        reason: str,
        response_digest: str | None = None,
    ) -> FHIRWriteOutcome:
        return FHIRWriteOutcome(
            status,
            reason,
            prepared.action_digest,
            prepared.payload_digest,
            prepared.resource_count,
            response_digest,
        )

    def _classify(
        self, prepared: FHIRPreparedWrite, response: FHIRHTTPResponse
    ) -> FHIRWriteOutcome:
        if (
            type(response) is not FHIRHTTPResponse
            or type(response.status_code) is not int
            or not 100 <= response.status_code <= 599
            or type(response.body) is not bytes
        ):
            return self._outcome(
                prepared, FHIRWriteStatus.UNKNOWN, "unexpected_response"
            )
        if len(response.body) > self._limits.max_response_bytes:
            return self._outcome(
                prepared, FHIRWriteStatus.UNKNOWN, "response_limit_exceeded"
            )
        commitment = self._digest(b"response", response.body)
        status = response.status_code
        if 300 <= status < 400:
            return self._outcome(
                prepared, FHIRWriteStatus.UNKNOWN, "redirect_received", commitment
            )
        if status in {409, 412, 428}:
            return self._outcome(
                prepared,
                FHIRWriteStatus.CONFLICT,
                {
                    409: "server_conflict",
                    412: "precondition_failed",
                    428: "precondition_required",
                }[status],
                commitment,
            )
        if status in {400, 401, 403, 404, 405, 406, 410, 413, 415, 422}:
            return self._outcome(
                prepared, FHIRWriteStatus.REJECTED, "server_rejected", commitment
            )
        if status not in {200, 201, 204}:
            return self._outcome(
                prepared, FHIRWriteStatus.UNKNOWN, "unexpected_response", commitment
            )
        try:
            headers = _response_headers(response.headers)
            if status == 204 and response.body:
                raise FHIRWriteError("unexpected_response")
            if response.body:
                if (
                    headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    != "application/fhir+json"
                ):
                    raise FHIRWriteError("unexpected_response")
                payload = _parse(response.body, self._limits, "unexpected_response")
            else:
                payload = None
            if len(prepared._types) == 1:
                if payload is None:
                    acknowledged_id = _ack(headers, prepared._types[0], self._audience)
                else:
                    acknowledged_id = _resource_ack(payload, prepared._types[0])
                expected_id = prepared.payload.get("id")
                if expected_id is not None and expected_id != acknowledged_id:
                    raise FHIRWriteError("unexpected_response")
            else:
                if (
                    status != 200
                    or type(payload) is not dict
                    or payload.get("resourceType") != "Bundle"
                    or payload.get("type") != "transaction-response"
                    or type(payload.get("entry")) is not list
                    or len(payload["entry"]) != len(prepared._types)
                ):
                    raise FHIRWriteError("unexpected_response")
                requested = prepared.payload["entry"]
                for entry, resource_type, proposal in zip(
                    payload["entry"], prepared._types, requested
                ):
                    if (
                        type(entry) is not dict
                        or type(entry.get("response")) is not dict
                    ):
                        raise FHIRWriteError("unexpected_response")
                    part = entry["response"]
                    if (
                        type(part.get("status")) is not str
                        or re.fullmatch(
                            r"(?:200|201|204)(?: [A-Za-z ]{1,40})?", part["status"]
                        )
                        is None
                    ):
                        raise FHIRWriteError("unexpected_response")
                    acknowledged_id = _ack(
                        {"location": part.get("location"), "etag": part.get("etag")},
                        resource_type,
                        self._audience,
                    )
                    expected_id = proposal["resource"].get("id")
                    if expected_id is not None and expected_id != acknowledged_id:
                        raise FHIRWriteError("unexpected_response")
                    if "resource" in entry:
                        if (
                            _resource_ack(entry["resource"], resource_type)
                            != acknowledged_id
                        ):
                            raise FHIRWriteError("unexpected_response")
            return self._outcome(
                prepared, FHIRWriteStatus.COMMITTED, "server_acknowledged", commitment
            )
        except Exception:
            return self._outcome(
                prepared, FHIRWriteStatus.UNKNOWN, "unexpected_response", commitment
            )


def _valid_digest(value: object) -> bool:
    return type(value) is str and _DIGEST.fullmatch(value) is not None


def _audience(value: str, allow_loopback: bool) -> str:
    try:
        if (
            type(value) is not str
            or len(value) > 2048
            or any(ord(c) < 33 or ord(c) == 127 for c in value)
        ):
            raise ValueError
        parsed = urlsplit(value)
        parsed.port
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or "%" in parsed.path
            or any(p in {".", ".."} for p in parsed.path.split("/"))
            or not (
                parsed.scheme == "https"
                or allow_loopback
                and parsed.scheme == "http"
                and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        ):
            raise ValueError
        # Preserve the custody audience exactly; a trailing slash is refused,
        # rather than silently normalizing a broker's audience binding.
        if value.endswith("/"):
            raise ValueError
        return value
    except Exception:
        raise FHIRWriteError("invalid_configuration") from None


def _predicate(value: object) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 4096
        or not re.fullmatch(r"[A-Za-z0-9._~!$&'()*+,;=:@%\-]+", value)
        or re.search(r"%(?![0-9A-Fa-f]{2})", value)
    ):
        raise FHIRWriteError("invalid_plan")
    try:
        pairs = parse_qsl(
            value,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=64,
            errors="strict",
        )
        if (
            not pairs
            or any(
                not re.fullmatch(
                    r"_?[A-Za-z][A-Za-z0-9-]*(?::[A-Za-z][A-Za-z0-9-]*)?", key
                )
                or key
                in {
                    "_count",
                    "_sort",
                    "_include",
                    "_revinclude",
                    "_summary",
                    "_elements",
                    "_total",
                    "_format",
                    "_pretty",
                    "_contained",
                    "_containedType",
                    "_since",
                }
                or not val
                or len(val) > 2048
                or "," in val
                or any(ord(c) < 32 or ord(c) == 127 for c in val)
                for key, val in pairs
            )
            or len({key for key, _ in pairs}) != len(pairs)
        ):
            raise ValueError
    except Exception:
        raise FHIRWriteError("invalid_plan") from None


def _transaction_request(request: dict[str, Any], resource_type: str) -> None:
    method, url = request.get("method"), request.get("url")
    if method == "POST":
        if url != resource_type or set(request) - {"method", "url", "ifNoneExist"}:
            raise FHIRWriteError("invalid_payload")
        if "ifNoneExist" in request:
            _predicate(request["ifNoneExist"])
    elif method == "PUT":
        if (
            type(url) is not str
            or set(request) != {"method", "url", "ifMatch"}
            or type(request["ifMatch"]) is not str
            or not _ETAG.fullmatch(request["ifMatch"])
        ):
            raise FHIRWriteError("invalid_payload")
        if url.startswith(resource_type + "?"):
            _predicate(url[len(resource_type) + 1 :])
        elif re.fullmatch(re.escape(resource_type) + "/" + _ID, url) is None:
            raise FHIRWriteError("invalid_payload")
    else:
        raise FHIRWriteError("invalid_payload")


def _check_json(value: Any, limits: FHIRWriteLimits) -> None:
    count = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if depth > limits.max_json_depth or count > limits.max_json_nodes:
            raise FHIRWriteError("limit_exceeded")
        if type(node) is dict:
            for key, val in node.items():
                if type(key) is not str:
                    raise FHIRWriteError("invalid_payload")
                walk(val, depth + 1)
        elif type(node) is list:
            for val in node:
                walk(val, depth + 1)
        elif node is not None and type(node) not in (str, int, float, bool):
            raise FHIRWriteError("invalid_payload")
        elif type(node) is float and not math.isfinite(node):
            raise FHIRWriteError("invalid_payload")

    walk(value, 0)


def _serialize(value: object, limits: FHIRWriteLimits) -> bytes:
    try:
        if not isinstance(value, Mapping):
            raise FHIRWriteError("invalid_payload")
        value = dict(value)
        _check_json(value, limits)
        size, parts = 0, []
        for part in json.JSONEncoder(
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).iterencode(value):
            encoded = part.encode("utf-8")
            size += len(encoded)
            if size > limits.max_request_bytes:
                raise FHIRWriteError("limit_exceeded")
            parts.append(encoded)
        return b"".join(parts)
    except FHIRWriteError:
        raise
    except Exception:
        raise FHIRWriteError("invalid_payload") from None


def _parse(body: bytes, limits: FHIRWriteLimits, reason: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        result = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
        _check_json(result, limits)
        return result
    except Exception:
        raise FHIRWriteError(reason) from None


def _lineage(manifest: object, limits: FHIRWriteLimits) -> bytes:
    try:
        action = getattr(manifest, "action_digest")
        records = getattr(manifest, "records")
        if (
            type(action) is not str
            or not _BARE_DIGEST.fullmatch(action)
            or type(records) is not tuple
            or not 1 <= len(records) <= limits.max_json_nodes
        ):
            raise ValueError
        values = []
        for record in records:
            value = {
                name: getattr(record, name)
                for name in (
                    "resource_handle",
                    "field_path",
                    "policy_digest",
                    "approval_receipt_digest",
                    "target_digest",
                    "plan_key",
                )
            }
            if (
                type(value["resource_handle"]) is not str
                or not re.fullmatch(r"res_[0-9a-f]{32}", value["resource_handle"])
                or type(value["field_path"]) is not str
                or len(value["field_path"]) > 160
                or not re.fullmatch(
                    r"[a-z][A-Za-z0-9]*(?:\.[a-z][A-Za-z0-9]*)*", value["field_path"]
                )
            ):
                raise ValueError
            if (
                any(
                    type(value[n]) is not str or not _BARE_DIGEST.fullmatch(value[n])
                    for n in (
                        "policy_digest",
                        "approval_receipt_digest",
                        "target_digest",
                    )
                )
                or type(value["plan_key"]) is not str
                or not _KEY.fullmatch(value["plan_key"])
            ):
                raise ValueError
            steps = getattr(record, "step_digests")
            evidence = getattr(record, "evidence")
            if (
                type(steps) is not tuple
                or not steps
                or len(steps) > limits.max_json_nodes
                or len(set(steps)) != len(steps)
                or any(
                    type(s) is not str or not _BARE_DIGEST.fullmatch(s) for s in steps
                )
                or type(evidence) is not tuple
                or not evidence
                or len(evidence) > limits.max_json_nodes
            ):
                raise ValueError
            spans = []
            for span in evidence:
                digest, start, end = span.digest, span.start, span.end
                if (
                    type(digest) is not str
                    or not _BARE_DIGEST.fullmatch(digest)
                    or type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end
                ):
                    raise ValueError
                spans.append([digest, start, end])
            value.update(step_digests=list(steps), evidence=spans)
            values.append(value)
        if len({(v["resource_handle"], v["field_path"]) for v in values}) != len(
            values
        ):
            raise ValueError
        values.sort(key=lambda v: (v["resource_handle"], v["field_path"]))
        return _serialize({"action_digest": action, "records": values}, limits)
    except Exception:
        raise FHIRWriteError("invalid_lineage") from None


def _response_headers(values: tuple[tuple[str, str], ...]) -> dict[str, str]:
    if type(values) is not tuple or len(values) > 64:
        raise FHIRWriteError("unexpected_response")
    result = {}
    size = 0
    for item in values:
        if (
            type(item) is not tuple
            or len(item) != 2
            or any(type(v) is not str for v in item)
        ):
            raise FHIRWriteError("unexpected_response")
        name, value = item
        size += len(name) + len(value)
        if (
            size > 32768
            or not re.fullmatch(r"[A-Za-z0-9-]{1,64}", name)
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or name.lower() in result
        ):
            raise FHIRWriteError("unexpected_response")
        result[name.lower()] = value
    return result


def _resource_ack(value: object, resource_type: str) -> str:
    if (
        type(value) is not dict
        or value.get("resourceType") != resource_type
        or type(value.get("id")) is not str
        or re.fullmatch(_ID, value["id"]) is None
        or type(value.get("meta")) is not dict
        or type(value["meta"].get("versionId")) is not str
        or re.fullmatch(_ID, value["meta"]["versionId"]) is None
    ):
        raise FHIRWriteError("unexpected_response")
    return value["id"]


def _ack(headers: Mapping[str, Any], resource_type: str, audience: str) -> str:
    location, etag = headers.get("location"), headers.get("etag")
    if (
        type(location) is not str
        or len(location) > 4096
        or type(etag) is not str
        or not _ETAG.fullmatch(etag)
    ):
        raise FHIRWriteError("unexpected_response")
    if location.startswith(audience + "/"):
        location = location[len(audience) + 1 :]
    match = re.fullmatch(
        re.escape(resource_type) + "/(" + _ID + ")/_history/(" + _ID + ")", location
    )
    if match is None or etag != 'W/"' + match.group(2) + '"':
        raise FHIRWriteError("unexpected_response")
    return match.group(1)
