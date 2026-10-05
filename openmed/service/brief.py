"""Shared, bounded brief transport and application-owned review lookup."""

import asyncio
import re
from typing import Any, Callable

from openmed.clinical.brief import (
    BriefContext,
    BriefRefusal,
    _result,
    build_clinical_brief,
)
from openmed.clinical.brief_cancellation import (
    BriefCancellation,
    BriefInterrupted,
    call_with_cancellation,
    check_cancellation,
)
from openmed.core.pii import DeidentificationResult

LOCAL_BRIEF_MODELS = ("extractive", "mlx", "maple", "maple-preview")
BRIEF_PROFILES = (
    "bhc",
    "brief_hospital_course",
    "clinical_handoff",
    "discharge_summary",
    "problem_oriented",
)


def brief_response(
    text: str,
    *,
    model: str = "mlx",
    profile: str = "bhc",
    review_id: str | None = None,
    context_provider: Callable | None = None,
    cancellation: BriefCancellation | None = None,
) -> dict[str, Any]:
    """Map caller, CLI and service interruption to the same value-free result.

    The application may pass a started budget context; no remote provider or
    wire-supplied review approval is admitted. Cancelled calls discard output.
    """
    try:
        check_cancellation(cancellation)
        result = _brief_response(
            text,
            model=model,
            profile=profile,
            review_id=review_id,
            context_provider=context_provider,
            cancellation=cancellation,
        )
        check_cancellation(cancellation)
        return result
    except BriefInterrupted as error:
        reason = BriefRefusal(error.reason)
    except (KeyboardInterrupt, asyncio.CancelledError):
        if cancellation is not None:
            cancellation.cancel()
        reason = BriefRefusal.CANCELLED
    return _result("", reason, []).to_response()


def _brief_response(
    text: str,
    *,
    model: str = "mlx",
    profile: str = "bhc",
    review_id: str | None = None,
    context_provider: Callable | None = None,
    cancellation: BriefCancellation | None = None,
) -> dict[str, Any]:
    """Run the shared contract without accepting network backends or approvals.

    ``context_provider(text, review_id)`` is trusted local application code. It
    resolves an already-reviewed artifact/context under the caller's access
    policy. Neither review histories nor NLI scores are accepted over the wire.
    """
    if type(text) is not str or not text or len(text.encode("utf-8")) > 16384:
        raise ValueError("invalid brief input")
    if model not in LOCAL_BRIEF_MODELS or profile not in BRIEF_PROFILES:
        raise ValueError("invalid local brief configuration")
    if review_id is not None and (
        not isinstance(review_id, str) or not re.fullmatch(r"[a-f0-9]{64}", review_id)
    ):
        raise ValueError("invalid brief review reference")
    value, context = text, None
    if review_id is not None:
        if context_provider is None:
            return _result("", BriefRefusal.REVIEW_REQUIRED, []).to_response()
        failed = False
        try:
            value, context = call_with_cancellation(
                context_provider, text, review_id, cancellation=cancellation
            )
            if (
                type(value) is not DeidentificationResult
                or type(context) is not BriefContext
                or value.original_text != text
            ):
                failed = True
        except BriefInterrupted:
            raise
        except Exception:
            failed = True
        if failed:
            return _result("", BriefRefusal.INVALID_EVIDENCE, []).to_response()
    return build_clinical_brief(
        value, model=model, profile=profile, context=context, cancellation=cancellation
    ).to_response()


def brief_response_schema() -> dict[str, Any]:
    """Return the shared JSON shape used by MCP and compatibility tests."""
    properties = {
        "schema_version": {"type": "integer", "const": 1},
        "status": {"type": "string", "enum": ["needs_review", "refused"]},
        "summary": {"type": "string"},
        "refusal_reason": {"type": ["string", "null"]},
        "summary_digest": {"type": "string"},
        "summary_characters": {"type": "integer", "minimum": 0},
        "digest": {"type": "string"},
        "stages": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "object"}},
        "verdicts": {"type": "array", "items": {"type": "object"}},
        "metrics": {"type": "object"},
        "envelope": {"type": "object"},
        "review_packet": {"type": "object"},
        "provenance": {"type": "object"},
        "profile_digest": {"type": ["string", "null"]},
        "backend_id": {"type": ["string", "null"]},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
