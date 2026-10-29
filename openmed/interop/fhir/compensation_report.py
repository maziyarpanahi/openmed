"""Content-free review packets for partially failed FHIR R4 bundle writes.

The caller supplies in-memory request and response Bundles. This module never
issues a request, retains resource content, or authorizes a corrective write.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "CompensationAction",
    "CompensationEffect",
    "CompensationPacket",
    "EffectClass",
    "build_compensation_report",
]

_STATUS = re.compile(r"([1-5][0-9]{2})(?: [^\r\n]*)?\Z")
_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class EffectClass(str, Enum):
    """Conservative classification of an effect reported by the server."""

    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    REVIEW_REQUIRED = "review_required"


class CompensationAction(str, Enum):
    """Fixed, non-executable proposal for an authorized reviewer."""

    VERIFY_AND_CONSIDER_DELETE = "verify_and_consider_delete"
    REVIEW_DELETION = "review_deletion"
    REVIEW_PRIOR_VERSION = "review_prior_version"
    RECONCILE_SERVER_STATE = "reconcile_server_state"


@dataclass(frozen=True, slots=True)
class CompensationEffect:
    """One positional bundle effect with no URLs, IDs, or resource values."""

    entry_index: int
    method: str
    status_code: int | None
    classification: EffectClass
    reason_code: str
    proposed_action: CompensationAction


@dataclass(frozen=True, slots=True)
class CompensationPacket:
    """Review-only proposal; it has no target or executable request data."""

    effects: tuple[CompensationEffect, ...]
    has_partial_failure: bool

    @property
    def approval_required(self) -> bool:
        """Compensation always requires separate human authorization."""

        return True


def _entries(bundle: Mapping[str, Any], expected_type: str) -> list[Any]:
    if bundle.get("resourceType") != "Bundle" or bundle.get("type") != expected_type:
        raise ValueError("FHIR bundle type is invalid")
    entries = bundle.get("entry")
    if type(entries) is not list:
        raise ValueError("FHIR bundle entries are invalid")
    return entries


def _response_status(entry: Any) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    response = entry.get("response")
    if not isinstance(response, Mapping):
        return None
    status = response.get("status")
    if type(status) is not str:
        return None
    match = _STATUS.fullmatch(status)
    return int(match.group(1)) if match else None


def build_compensation_report(
    intended: Mapping[str, Any], received: Mapping[str, Any] | None
) -> CompensationPacket:
    """Compare positional FHIR request and response entries without I/O.

    A reported successful create is only a *candidate* for reversal. Its
    location must be verified in the clinical system before a separately
    approved deletion. Successful updates and failed or missing responses
    require reconciliation; successful deletions have no safe automatic undo.

    Args:
        intended: In-memory R4 transaction or batch request Bundle.
        received: Corresponding response Bundle, or ``None`` after uncertainty.

    Returns:
        A value-free packet of proposed review actions in request order.

    Raises:
        TypeError: If the bundle input is not a mapping.
        ValueError: If bundle structure or request methods are invalid.
    """

    if not isinstance(intended, Mapping) or (
        received is not None and not isinstance(received, Mapping)
    ):
        raise TypeError("FHIR bundles must be mappings")
    bundle_type = intended.get("type")
    if type(bundle_type) is not str or bundle_type not in {"transaction", "batch"}:
        raise ValueError("FHIR request bundle type is invalid")
    requests = _entries(intended, bundle_type)
    responses: list[Any] = []
    if received is not None:
        responses = _entries(received, f"{bundle_type}-response")
        if len(responses) > len(requests):
            raise ValueError("FHIR response has excess entries")

    effects: list[CompensationEffect] = []
    reported_success = False
    uncertain_or_failed = False
    for index, entry in enumerate(requests):
        if not isinstance(entry, Mapping) or not isinstance(
            entry.get("request"), Mapping
        ):
            raise ValueError("FHIR request entry is invalid")
        request = entry["request"]
        method = request.get("method")
        if type(method) is not str or method not in _METHODS:
            raise ValueError("FHIR request method is invalid")
        if type(request.get("url")) is not str or not request["url"]:
            raise ValueError("FHIR request target is invalid")

        response_entry = responses[index] if index < len(responses) else None
        status = _response_status(response_entry)
        correlated = isinstance(response_entry, Mapping) and (
            "fullUrl" not in response_entry
            or response_entry["fullUrl"] == entry.get("fullUrl")
        )
        succeeded = status is not None and 200 <= status < 300 and correlated
        reported_success |= succeeded
        uncertain_or_failed |= not succeeded

        if not succeeded:
            classification = EffectClass.REVIEW_REQUIRED
            reason = "unconfirmed_or_failed"
            action = CompensationAction.RECONCILE_SERVER_STATE
        elif (
            method == "POST"
            and status == 201
            and isinstance(response_entry.get("response", {}).get("location"), str)
            and response_entry["response"]["location"]
        ):
            classification = EffectClass.REVERSIBLE
            reason = "created_location_reported"
            action = CompensationAction.VERIFY_AND_CONSIDER_DELETE
        elif method == "DELETE":
            classification = EffectClass.IRREVERSIBLE
            reason = "deletion_reported"
            action = CompensationAction.REVIEW_DELETION
        else:
            classification = EffectClass.REVIEW_REQUIRED
            reason = "prior_state_or_target_unverified"
            action = CompensationAction.REVIEW_PRIOR_VERSION

        effects.append(
            CompensationEffect(index, method, status, classification, reason, action)
        )
    return CompensationPacket(
        effects=tuple(effects),
        has_partial_failure=reported_success and uncertain_or_failed,
    )
