"""Strict governed-workflow consumers over the existing metadata wire contract.

Only inspection calls may retry. Receipt submission and cancellation intent are
explicit single calls; failed or malformed mutation replies preserve uncertainty.
No raw response, server exception message, credential or protected preview is
attached to a client error. This module has no server or approval issuer.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.agent.outcomes import WorkflowOutcome
from openmed.agent.workflows.recovery import EffectRecord

from .governed_workflows import (
    MAX_WORKFLOW_EFFECTS,
    MAX_WORKFLOW_REQUEST_BYTES,
    MAX_WORKFLOW_RESPONSE_BYTES,
    WORKFLOW_ERROR_STATUSES,
    WORKFLOW_RESPONSE_VERSION,
    WorkflowReference,
    WorkflowView,
)

_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "workflow_id",
        "action_digest",
        "state_digest",
        "phase",
        "effects",
        "preview_digest",
        "proposed_effect_count",
        "committed_effect_count",
        "outcome",
        "receipt_digest",
        "cancellation_requested",
    }
)
_CLIENT_CODES = frozenset(
    {
        "workflow_invalid_request",
        "workflow_unsupported_version",
        "workflow_malformed_response",
        "workflow_response_too_large",
        "workflow_binding_mismatch",
        "workflow_transport_failed",
        "workflow_server_refused",
        "workflow_poll_cancelled",
        "workflow_poll_timeout",
        "workflow_clock_failed",
    }
)
_RETRY_CODES = frozenset(
    {"workflow_unavailable", "workflow_service_failed", "workflow_transport_failed"}
)
_REFUSAL_CODES = frozenset(WORKFLOW_ERROR_STATUSES) - {
    "workflow_mutation_unknown",
    "workflow_service_failed",
    "workflow_unavailable",
    "workflow_invalid_result",
}
_SAFE_REQUEST_ID = re.compile(
    r"(?:req_[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\Z"
)
MutationOutcome = Literal["not_applicable", "not_attempted", "refused", "unknown"]


class WorkflowClientError(RuntimeError):
    """Fixed diagnostics that preserve mutation uncertainty without response text.

    Args:
        code: A closed workflow server/client code.
        status_code: HTTP status, when available.
        mutation_outcome: Whether a mutation may have reached custody.
        request_id: Optional validated opaque correlation identifier.
        last_view: Last strictly validated poll snapshot, when available.
    """

    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        mutation_outcome: MutationOutcome = "not_applicable",
        request_id: str | None = None,
        last_view: WorkflowView | None = None,
    ) -> None:
        self.code = (
            code
            if code in _CLIENT_CODES or code in WORKFLOW_ERROR_STATUSES
            else "workflow_server_refused"
        )
        self.status_code = status_code
        self.mutation_outcome = mutation_outcome
        self.request_id = (
            request_id
            if type(request_id) is str and _SAFE_REQUEST_ID.fullmatch(request_id)
            else None
        )
        self.last_view = last_view
        super().__init__(f"{self.code}: governed workflow request did not complete.")


def _finite(value: Any, maximum: float, *, allow_zero: bool = False) -> bool:
    return (
        type(value) in (int, float)
        and (0 <= value if allow_zero else 0 < value)
        and value <= maximum
        and math.isfinite(value)
    )


@dataclass(frozen=True, slots=True)
class WorkflowReadOptions:
    """Bound explicit inspection retries; mutations do not accept this policy.

    Args:
        max_attempts: At most three inspection attempts; default one.
        retry_delay_seconds: Interruptible delay between retryable read failures.
        timeout_seconds: At most thirty seconds per request.
    """

    max_attempts: int = 1
    retry_delay_seconds: float = 0.0
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            type(self.max_attempts) is not int
            or not 1 <= self.max_attempts <= 3
            or not _finite(self.retry_delay_seconds, 5, allow_zero=True)
            or not _finite(self.timeout_seconds, 30)
        ):
            raise WorkflowClientError("workflow_invalid_request")


@dataclass(frozen=True, slots=True)
class WorkflowPollPolicy:
    """Bound status requests and elapsed time, stopping for external review.

    Args:
        max_requests: Maximum actual status requests, at most one hundred.
        timeout_seconds: Total poll deadline, at most five minutes.
        interval_seconds: Interruptible delay, at most thirty seconds.
    """

    max_requests: int = 20
    timeout_seconds: float = 60.0
    interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if (
            type(self.max_requests) is not int
            or not 1 <= self.max_requests <= 100
            or not _finite(self.timeout_seconds, 300)
            or not _finite(self.interval_seconds, 30, allow_zero=True)
        ):
            raise WorkflowClientError("workflow_invalid_request")


def _json(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > MAX_WORKFLOW_RESPONSE_BYTES:
        raise WorkflowClientError("workflow_response_too_large")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, value in items:
            if key in values:
                raise ValueError()
            values[key] = value
        return values

    def constant(_value: str) -> None:
        raise ValueError()

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
        pending = [(value, 0)]
        nodes = 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            if depth > 8 or nodes > 4_096:
                raise ValueError()
            if type(item) is dict:
                pending.extend((v, depth + 1) for v in item.values())
            elif type(item) is list:
                pending.extend((v, depth + 1) for v in item)
            elif type(item) is float and not math.isfinite(item):
                raise ValueError()
        if type(value) is not dict:
            raise ValueError()
        return value
    except Exception:
        pass
    # Raise after the handler so JSONDecodeError.doc is not retained as context.
    raise WorkflowClientError("workflow_malformed_response")


def parse_workflow_response(raw: bytes, reference: WorkflowReference) -> WorkflowView:
    """Parse exact fields using native outcomes/effects and recompute commitments.

    Args:
        raw: Bounded response JSON bytes.
        reference: Exact run/workflow/action the caller inspected.

    Returns:
        A native immutable metadata snapshot; never clinical execution authority.

    Raises:
        WorkflowClientError: On unsupported, malformed or mismatched metadata.
    """
    values = _json(raw)
    if values.keys() != _RESPONSE_FIELDS:
        raise WorkflowClientError("workflow_malformed_response")
    if values["schema_version"] != WORKFLOW_RESPONSE_VERSION:
        raise WorkflowClientError("workflow_unsupported_version")
    try:
        if type(reference) is not WorkflowReference:
            raise ValueError()
        native_reference = WorkflowReference(
            RunId.parse(values["run_id"]),
            WorkflowId.parse(values["workflow_id"]),
            values["action_digest"],
        )
        if (
            native_reference.run_id != reference.run_id
            or native_reference.workflow_id != reference.workflow_id
            or native_reference.action_digest != reference.action_digest
        ):
            raise WorkflowClientError("workflow_binding_mismatch")
        for field in ("proposed_effect_count", "committed_effect_count"):
            if (
                type(values[field]) is not int
                or not 0 <= values[field] <= MAX_WORKFLOW_EFFECTS
            ):
                raise ValueError()
        if (
            type(values["cancellation_requested"]) is not bool
            or type(values["effects"]) is not list
            or len(values["effects"]) > MAX_WORKFLOW_EFFECTS
        ):
            raise ValueError()
        outcome = None
        if values["outcome"] is not None:
            if type(values["outcome"]) is not dict or set(values["outcome"]) != {
                "schema_version",
                "outcome_class",
                "reason_code",
            }:
                raise ValueError()
            outcome = WorkflowOutcome.from_dict(values["outcome"])
        view = WorkflowView(
            native_reference,
            values["state_digest"],
            ActionPhase(values["phase"]),
            tuple(EffectRecord.from_dict(item) for item in values["effects"]),
            outcome,
            values["receipt_digest"],
            values["cancellation_requested"],
        )
        if view.to_dict() != values:
            raise ValueError()
        return view
    except WorkflowClientError:
        raise
    except Exception:
        pass
    raise WorkflowClientError("workflow_malformed_response")


class WorkflowClientMixin:
    """Typed workflow operations on the existing OpenMedClient HTTP transport."""

    _client: httpx.Client
    _request_id: str | None

    def workflow_preflight(
        self,
        reference: WorkflowReference,
        *,
        options: WorkflowReadOptions | None = None,
        stop_event: threading.Event | None = None,
    ) -> WorkflowView:
        """Inspect preflight metadata with optional bounded read-only retries."""
        return self._workflow_read("preflight", reference, options, stop_event)

    def workflow_preview(
        self,
        reference: WorkflowReference,
        *,
        options: WorkflowReadOptions | None = None,
        stop_event: threading.Event | None = None,
    ) -> WorkflowView:
        """Inspect the existing content-free preview, never dispatching an effect."""
        return self._workflow_read("preview", reference, options, stop_event)

    def workflow_status(
        self,
        reference: WorkflowReference,
        *,
        options: WorkflowReadOptions | None = None,
        stop_event: threading.Event | None = None,
    ) -> WorkflowView:
        """Inspect current phase, review evidence and committed-effect counts."""
        return self._workflow_read("status", reference, options, stop_event)

    def workflow_submit_receipt(
        self,
        reference: WorkflowReference,
        receipt: ApprovalReceipt,
        *,
        timeout_seconds: float = 30,
        stop_event: threading.Event | None = None,
    ) -> WorkflowView:
        """Submit an existing receipt exactly once; failures may leave it recorded."""
        payload = self._workflow_payload(reference, mutation=True)
        try:
            if type(receipt) is not ApprovalReceipt:
                raise ValueError()
            parsed = ApprovalReceipt.from_dict(receipt.to_dict())
            if parsed.action_digest != reference.action_digest:
                raise ValueError()
            payload["receipt"] = parsed.to_dict()
        except Exception:
            raise WorkflowClientError(
                "workflow_invalid_request", mutation_outcome="not_attempted"
            ) from None
        return self._workflow_send(
            "review-receipts",
            payload,
            reference,
            timeout_seconds,
            stop_event,
            mutation=True,
        )

    def workflow_cancel(
        self,
        reference: WorkflowReference,
        *,
        timeout_seconds: float = 30,
        stop_event: threading.Event | None = None,
    ) -> WorkflowView:
        """Request server cancellation intent once; this cannot roll back effects."""
        payload = self._workflow_payload(reference, mutation=True)
        return self._workflow_send(
            "cancel", payload, reference, timeout_seconds, stop_event, mutation=True
        )

    def poll_workflow(
        self,
        reference: WorkflowReference,
        *,
        policy: WorkflowPollPolicy | None = None,
        stop_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> WorkflowView:
        """Poll status within request/time bounds; stop at review or terminal state.

        Args:
            reference: Exact action to inspect; never a mutation request.
            policy: Validated request/deadline/interval bounds.
            stop_event: Cooperative caller cancellation between requests/chunks.
            clock: Injectable monotonic clock for deterministic offline controls.

        Returns:
            The first review-required or terminal metadata snapshot.

        Raises:
            WorkflowClientError: On cancellation, timeout, clock or transport failure.
        """
        settings = WorkflowPollPolicy() if policy is None else policy
        if type(settings) is not WorkflowPollPolicy:
            raise WorkflowClientError("workflow_invalid_request")
        stop = threading.Event() if stop_event is None else stop_event
        if not isinstance(stop, threading.Event):
            raise WorkflowClientError("workflow_invalid_request")

        def now() -> float:
            try:
                value = clock()
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError()
                return float(value)
            except Exception:
                raise WorkflowClientError("workflow_clock_failed") from None

        start = previous = now()
        last = None
        for index in range(settings.max_requests):
            if stop.is_set():
                raise WorkflowClientError("workflow_poll_cancelled", last_view=last)
            current = now()
            if current < previous:
                raise WorkflowClientError("workflow_clock_failed", last_view=last)
            remaining = settings.timeout_seconds - (current - start)
            if remaining <= 0:
                raise WorkflowClientError("workflow_poll_timeout", last_view=last)
            failure = None
            try:
                last = self.workflow_status(
                    reference,
                    options=WorkflowReadOptions(timeout_seconds=min(30.0, remaining)),
                    stop_event=stop,
                )
            except WorkflowClientError as error:
                failure = WorkflowClientError(
                    error.code,
                    status_code=error.status_code,
                    request_id=error.request_id,
                    last_view=last,
                )
            if failure is not None:
                raise failure
            if last is None:
                raise WorkflowClientError("workflow_malformed_response")
            current_after = now()
            if current_after < current:
                raise WorkflowClientError("workflow_clock_failed", last_view=last)
            if stop.is_set():
                raise WorkflowClientError("workflow_poll_cancelled", last_view=last)
            if current_after - start >= settings.timeout_seconds:
                raise WorkflowClientError("workflow_poll_timeout", last_view=last)
            if last.phase in (
                ActionPhase.WAITING_REVIEW,
                ActionPhase.COMPLETED,
                ActionPhase.ABORTED,
            ):
                return last
            if index + 1 == settings.max_requests:
                break
            previous = current_after
            wait = min(
                settings.interval_seconds,
                settings.timeout_seconds - (current_after - start),
            )
            if stop.wait(wait):
                raise WorkflowClientError("workflow_poll_cancelled", last_view=last)
        raise WorkflowClientError("workflow_poll_timeout", last_view=last)

    def _workflow_payload(
        self, reference: WorkflowReference, *, mutation: bool
    ) -> dict[str, Any]:
        try:
            if type(reference) is not WorkflowReference:
                raise ValueError()
            reference = WorkflowReference.from_dict(reference.to_dict())
            if mutation:
                reference.require_mutation()
            return reference.to_dict()
        except Exception:
            raise WorkflowClientError(
                "workflow_invalid_request",
                mutation_outcome="not_attempted" if mutation else "not_applicable",
            ) from None

    def _workflow_read(
        self,
        operation: str,
        reference: WorkflowReference,
        options: WorkflowReadOptions | None,
        stop_event: threading.Event | None,
    ) -> WorkflowView:
        settings = WorkflowReadOptions() if options is None else options
        if type(settings) is not WorkflowReadOptions:
            raise WorkflowClientError("workflow_invalid_request")
        payload = self._workflow_payload(reference, mutation=False)
        for attempt in range(settings.max_attempts):
            try:
                return self._workflow_send(
                    operation,
                    payload,
                    reference,
                    settings.timeout_seconds,
                    stop_event,
                    mutation=False,
                )
            except WorkflowClientError as error:
                if (
                    attempt + 1 == settings.max_attempts
                    or error.code not in _RETRY_CODES
                ):
                    raise
            stop = threading.Event() if stop_event is None else stop_event
            if stop.wait(settings.retry_delay_seconds):
                raise WorkflowClientError("workflow_poll_cancelled")
        raise WorkflowClientError("workflow_transport_failed")

    def _workflow_send(
        self,
        operation: str,
        payload: dict[str, Any],
        reference: WorkflowReference,
        timeout: float,
        stop_event: threading.Event | None,
        *,
        mutation: bool,
    ) -> WorkflowView:
        before: MutationOutcome = "not_attempted" if mutation else "not_applicable"
        uncertain: MutationOutcome = "unknown" if mutation else "not_applicable"
        if not _finite(timeout, 30) or (
            stop_event is not None and not isinstance(stop_event, threading.Event)
        ):
            raise WorkflowClientError(
                "workflow_invalid_request", mutation_outcome=before
            )
        if stop_event is not None and stop_event.is_set():
            raise WorkflowClientError(
                "workflow_poll_cancelled", mutation_outcome=before
            )
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        if len(encoded) > MAX_WORKFLOW_REQUEST_BYTES:
            raise WorkflowClientError(
                "workflow_invalid_request", mutation_outcome=before
            )
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if type(self._request_id) is str and _SAFE_REQUEST_ID.fullmatch(
            self._request_id
        ):
            headers["X-Request-ID"] = self._request_id
        status = None
        correlation = None
        deadline = time.monotonic() + timeout
        try:
            with self._client.stream(
                "POST",
                "/v1/workflows/" + operation,
                content=encoded,
                headers=headers,
                timeout=timeout,
                follow_redirects=False,
            ) as response:
                status = response.status_code
                correlation = response.headers.get("X-Request-ID")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    if stop_event is not None and stop_event.is_set():
                        raise WorkflowClientError(
                            "workflow_poll_cancelled", mutation_outcome=uncertain
                        )
                    if time.monotonic() >= deadline:
                        raise WorkflowClientError(
                            "workflow_mutation_unknown"
                            if mutation
                            else "workflow_transport_failed",
                            mutation_outcome=uncertain,
                        )
                    if len(raw) + len(chunk) > MAX_WORKFLOW_RESPONSE_BYTES:
                        raise WorkflowClientError(
                            "workflow_response_too_large",
                            status_code=status,
                            mutation_outcome=uncertain,
                        )
                    raw.extend(chunk)
                if stop_event is not None and stop_event.is_set():
                    raise WorkflowClientError(
                        "workflow_poll_cancelled", mutation_outcome=uncertain
                    )
                if time.monotonic() >= deadline:
                    raise WorkflowClientError(
                        "workflow_mutation_unknown"
                        if mutation
                        else "workflow_transport_failed",
                        mutation_outcome=uncertain,
                    )
                if (
                    response.headers.get("Content-Type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                    != "application/json"
                ):
                    raise WorkflowClientError(
                        "workflow_malformed_response",
                        status_code=status,
                        mutation_outcome=uncertain,
                    )
                if not 200 <= status < 300:
                    values = _json(bytes(raw))
                    code = "workflow_server_refused"
                    error = values.get("error")
                    if (
                        set(values) == {"error"}
                        and type(error) is dict
                        and {"code", "message", "details"}
                        <= set(error)
                        <= {"code", "message", "details", "request_id"}
                    ):
                        candidate = error["code"]
                        if (
                            type(candidate) is str
                            and WORKFLOW_ERROR_STATUSES.get(candidate) == status
                        ):
                            code = candidate
                    outcome: MutationOutcome = (
                        "refused" if mutation and code in _REFUSAL_CODES else uncertain
                    )
                    raise WorkflowClientError(
                        code,
                        status_code=status,
                        mutation_outcome=outcome,
                        request_id=correlation,
                    )
                return parse_workflow_response(bytes(raw), reference)
        except WorkflowClientError as error:
            # Parser errors must retain mutation uncertainty too, without causes.
            failure = WorkflowClientError(
                error.code,
                status_code=status,
                mutation_outcome=error.mutation_outcome
                if error.mutation_outcome != "not_applicable"
                else uncertain,
                request_id=correlation,
            )
        except Exception:
            failure = WorkflowClientError(
                "workflow_mutation_unknown"
                if mutation
                else "workflow_transport_failed",
                status_code=status,
                mutation_outcome=uncertain,
                request_id=correlation,
            )
        raise failure


__all__ = [
    "WorkflowClientError",
    "WorkflowClientMixin",
    "WorkflowPollPolicy",
    "WorkflowReadOptions",
    "parse_workflow_response",
]
