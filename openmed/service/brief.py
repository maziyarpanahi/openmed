"""Shared, bounded brief transport and application-owned review lookup."""

import re
from typing import Any, Callable

from openmed.clinical.brief import (
    BriefContext,
    BriefRefusal,
    ReviewedLocalBriefContext,
    _result,
    build_clinical_brief,
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
            value, context = context_provider(text, review_id)
            if (
                type(value) is not DeidentificationResult
                or type(context) not in (BriefContext, ReviewedLocalBriefContext)
                or value.original_text != text
            ):
                failed = True
        except Exception:
            failed = True
        if failed:
            return _result("", BriefRefusal.INVALID_EVIDENCE, []).to_response()
    return build_clinical_brief(
        value, model=model, profile=profile, context=context
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
    required = list(properties)
    properties.update(
        generation_contract={
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "schema_version"],
            "properties": {
                "kind": {"const": "explicit_evidence"},
                "schema_version": {"type": "integer", "const": 1},
            },
        },
        claim_bindings={
            "type": "array",
            "minItems": 1,
            "maxItems": 64,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_index", "reference_digest"],
                "properties": {
                    "claim_index": {"type": "integer", "minimum": 0, "maximum": 63},
                    "reference_digest": {
                        "type": "string",
                        "pattern": "^sha256:[a-f0-9]{64}$",
                    },
                },
            },
        },
    )
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        "dependentRequired": {
            "generation_contract": ["claim_bindings"],
            "claim_bindings": ["generation_contract"],
        },
    }
