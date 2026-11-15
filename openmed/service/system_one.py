"""De-identify protected System One requests before routing them to a local backend."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from dataclasses import replace
from typing import Any, Callable

from openmed.service.privacy_gateway import safety_sweep_tripwire
from openmed.structured.decision import DecisionInputFormat, DecisionMode, decide
from openmed.structured.decision_interop import (
    from_system_one_request,
    to_system_one_response,
)

_AUDIT = logging.getLogger("openmed.service.systemone.audit")


class SystemOneInputError(ValueError):
    """A malformed, over-budget or non-text System One body."""


class SystemOnePrivacyError(ValueError):
    """A local de-identification or residual check refused dispatch."""


def _safe_text(result: Any, source: str) -> tuple[str, int]:
    value = getattr(result, "deidentified_text", None)
    entities = getattr(result, "pii_entities", None)
    meta = getattr(result, "metadata", None)
    if meta is None:
        meta = {}
    if (
        not isinstance(meta, dict)
        or not isinstance(value, str)
        or not value.strip()
        or not isinstance(entities, (tuple, list))
    ):
        raise SystemOnePrivacyError()
    for signals in (result, meta):
        get = (
            signals.get
            if isinstance(signals, dict)
            else lambda key, default: getattr(signals, key, default)
        )
        if (
            get("abstained", False)
            or get("requires_review", False)
            or get("state", "success") not in ("success", "ok")
            or get("status", "success") not in ("success", "ok")
        ):
            raise SystemOnePrivacyError()
    # A detector claiming redactions must not leave those exact source spans behind.
    for e in entities:
        if getattr(e, "action", None) in {"abstain", "review", "retain"}:
            raise SystemOnePrivacyError()
        start, end = getattr(e, "start", None), getattr(e, "end", None)
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(source)
            or source[start:end].casefold() in value.casefold()
        ):
            raise SystemOnePrivacyError()
    if safety_sweep_tripwire(value):
        raise SystemOnePrivacyError()
    return value, len(entities)


def protect_and_decide(
    body: dict[str, Any],
    *,
    cleaner: Callable[[str], Any],
    backend: Any,
    policy: Any,
    profiles: Any,
    audit_key: bytes,
) -> dict[str, Any]:
    """Screen all model-bound text and adapt safe scores back to caller-owned IDs."""
    try:
        if set(body) != {"state", "questions"}:
            raise SystemOneInputError()
        original = from_system_one_request(body)
        digest = hmac.new(
            audit_key,
            json.dumps(
                body,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode(),
            hashlib.sha256,
        ).hexdigest()
    except (ValueError, TypeError, RecursionError):
        raise SystemOneInputError() from None
    decision_id = uuid.uuid4().hex
    questions = original.as_questions()
    count = 0
    cache: dict[str, str] = {}

    def clean(text: str) -> str:
        nonlocal count
        if text not in cache:
            if len(cache) >= 256:
                raise SystemOneInputError()
            s, n = _safe_text(cleaner(text), text)
            cache[text] = s
            count += n
        return cache[text]

    safe_questions = []
    try:
        state = clean(original.input_text)
        for i, q in enumerate(questions):
            options = q.options
            descriptions = q.option_descriptions
            instruction = clean(q.instructions) if q.instructions else None
            if q.mode is DecisionMode.FIXED_CHOICE:
                # Option IDs are model-visible in System One. Give every option a
                # local ordinal and a screened description to retain its meaning.
                descriptions = tuple(
                    clean(name) + ((": " + clean(desc)) if desc else "")
                    for name, desc in zip(
                        options, descriptions or (None,) * len(options), strict=True
                    )
                )
                options = tuple(f"option_{j}" for j in range(len(options)))
            elif descriptions:
                descriptions = tuple(
                    clean(desc) if desc else None for desc in descriptions
                )
            levels = tuple(clean(level) for level in q.levels)
            safe_questions.append(
                replace(
                    q,
                    question_id=f"q{i}",
                    options=options,
                    option_descriptions=descriptions,
                    instructions=instruction,
                    levels=levels,
                )
            )
        safe = replace(
            original,
            questions=tuple(safe_questions),
            input_text=state,
            input_format=DecisionInputFormat.TEXT,
        )
    except Exception:
        _AUDIT.info(
            "systemone %s",
            json.dumps(
                {
                    "decision_id": decision_id,
                    "request_digest": digest,
                    "state": "privacy_refused",
                    "entity_count": count,
                }
            ),
        )
        raise SystemOnePrivacyError() from None
    result = decide(safe, backend=backend, policy=policy, calibration_profiles=profiles)
    wire = to_system_one_response(safe, result)
    answers = {}
    audit = []
    for q, clean_q, outcome in zip(
        questions, safe_questions, result.question_results(), strict=True
    ):
        answer = wire["answers"][clean_q.question_id]
        if q.mode is DecisionMode.FIXED_CHOICE:
            mapping = dict(zip(clean_q.options, q.options, strict=True))
            if answer.get("choice") is not None:
                answer["choice"] = mapping[answer["choice"]]
            if "probabilities" in answer:
                answer["probabilities"] = {
                    mapping[k]: v for k, v in answer["probabilities"].items()
                }
        elif q.levels:
            answer["legend"] = {str(i): label for i, label in enumerate(q.levels)}
        answers[q.question_id] = answer
        audit.append(
            {
                "question_id": hmac.new(
                    audit_key, q.question_id.encode(), hashlib.sha256
                ).hexdigest(),
                "mode": q.mode.value,
                "state": outcome.state.value,
            }
        )
    _AUDIT.info(
        "systemone %s",
        json.dumps(
            {
                "decision_id": decision_id,
                "request_digest": digest,
                "questions": audit,
                "entity_count": count,
            },
            separators=(",", ":"),
        ),
    )
    return {"decision_id": decision_id, "model": wire["model"], "answers": answers}
