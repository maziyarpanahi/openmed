"""Source-aligned clinical context composition for bounded service callers.

This module consumes already extracted entities. It does not load a model or
contact a service, and it does not qualify an extraction as a patient fact.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from itertools import islice
from typing import Any

from openmed.clinical.context import assert_context, scan_context_cues
from openmed.clinical.experiencer import resolve_experiencer
from openmed.clinical.sections import detect_sections, validate_section_spans
from openmed.core.clinical_language import resolve_clinical_language

CLINICAL_CONTEXT_VERSION = "clinical-context-v5"
DEFAULT_CONTEXT_TASKS = ("sections", "entities", "assertions")
STRUCTURED_TASKS = ("medications", "labs", "vitals", "relations")
TEMPORAL_TASKS = ("events", "timeline")
CONTEXT_TASKS = (*DEFAULT_CONTEXT_TASKS, *STRUCTURED_TASKS, *TEMPORAL_TASKS)
CONTEXT_LANGUAGES = ("en", "de")
MAX_ANALYSIS_ENTITIES = 2000
_LABELS = {
    "disease": "Disease",
    "condition": "Condition",
    "problem": "Problem",
    "symptom": "Symptom",
    "sign": "Sign",
    "drug": "Drug",
    "medication": "Drug",
    "chemical": "Chemical",
    "anatomy": "Anatomy",
    "bodypart": "Anatomy",
    "procedure": "Procedure",
    "labtest": "Lab Test",
    "labname": "Lab Test",
    "labvalue": "Lab Value",
    "referencerange": "Reference Range",
    "abnormalflag": "Abnormal Flag",
    "vitalsign": "Vital Sign",
    "dose": "Dose",
    "dosage": "Dose",
    "route": "Route",
    "frequency": "Frequency",
    "duration": "Duration",
    "form": "Form",
    "strength": "Strength",
    "severity": "Severity",
    "date": "Date",
    "time": "Time",
    "medicaldevice": "Medical Device",
}


class ClinicalAnalysisError(ValueError):
    """A finite, source-free error code for the clinical composition boundary."""


def validated_clinical_entities(
    text: str,
    entities: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate original offsets and return deterministic, surface-free entities.

    Args:
        text: Original source document used for upstream extraction.
        entities: Extracted spans with start/end, a supported clinical label and
            optional score. A supplied text value must match the source exactly.

    Returns:
        Deduplicated entities ordered by source offsets, with stable local IDs.
        Missing confidence remains unknown rather than becoming a perfect score.

    Raises:
        ClinicalAnalysisError: Input bounds, label, score or source offsets fail.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
        raise ClinicalAnalysisError("invalid_clinical_source")
    if isinstance(entities, (str, bytes, Mapping)):
        raise ClinicalAnalysisError("invalid_clinical_entities")
    records = list(islice(entities, MAX_ANALYSIS_ENTITIES + 1))
    if len(records) > MAX_ANALYSIS_ENTITIES:
        raise ClinicalAnalysisError("clinical_entity_limit")
    selected = {}
    for entity in records:
        if not isinstance(entity, Mapping):
            raise ClinicalAnalysisError("invalid_clinical_entity")
        start, end = entity.get("start"), entity.get("end")
        label = entity.get("label")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
            or not isinstance(label, str)
        ):
            raise ClinicalAnalysisError("invalid_clinical_offsets")
        canonical = _LABELS.get(
            label.lower().replace(" ", "").replace("_", "").replace("-", "")
        )
        if canonical is None:
            raise ClinicalAnalysisError("unsupported_clinical_label")
        if "text" in entity and entity["text"] != text[start:end]:
            raise ClinicalAnalysisError("clinical_source_mismatch")
        score = entity.get("score", entity.get("confidence"))
        if score is not None and (
            type(score) not in (float, int)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ClinicalAnalysisError("invalid_clinical_score")
        key = start, end, canonical
        previous = selected.get(key)
        if previous is None or (score if score is not None else -1) > (
            previous["score"] if previous["score"] is not None else -1
        ):
            selected[key] = {
                "start": start,
                "end": end,
                "label": canonical,
                "score": score,
            }
    return [
        {"id": f"e{i}", **entity}
        for i, (_, entity) in enumerate(sorted(selected.items()), start=1)
    ]


def _section_view(sections):
    allowed = (
        "label",
        "start",
        "end",
        "header_start",
        "header_end",
        "content_start",
        "language",
        "confidence",
        "coding",
    )
    return [
        {"id": f"s{i}", **{key: section[key] for key in allowed if key in section}}
        for i, section in enumerate(sections, start=1)
    ]


def analyze_clinical_context(
    text: str,
    entities: Iterable[Mapping[str, Any]],
    *,
    language: str = "auto",
    tasks: Sequence[str] = DEFAULT_CONTEXT_TASKS,
    entity_coverage_complete: bool = True,
    timeout_seconds: float = 30,
    cancel_check: Callable[[], bool] | None = None,
    reference_date: str | None = None,
) -> dict[str, Any]:
    """Compose sections, clinical spans, context and structured source evidence.

    Args:
        text: Original, unmodified clinical source, at most 100,000 characters.
        entities: Upstream spans whose full input coverage was verified by the
            caller. Entity source surfaces are excluded from returned records.
        language: Explicit language/locale or conservative automatic detection.
        tasks: Requested tasks drawn from CONTEXT_TASKS; sections, entities and
            assertions are the default. Medication/lab/vital/relation tasks
            consume compatible, already-extracted entity and attribute spans.
        entity_coverage_complete: Whether the upstream engine processed every
            input token. False cannot produce successful entity/assertion tasks.
        timeout_seconds: Cooperative postprocessing deadline, at most 30 seconds.
        cancel_check: Optional cancellation predicate checked between stages.
        reference_date: Explicit ISO calendar date for the timeline task's
            relative expressions. None leaves them unanchored; the current
            clock and patient birth date are never substituted.

    Returns:
        Per-task status, records, offset-based evidence and language provenance.
        Every successful task remains review-required. Mixed/uncertain or
        unsupported assertion language returns an explicit unsupported task,
        while independent section/entity results remain separately inspectable.

    Raises:
        ClinicalAnalysisError: Invalid inputs, expired deadline or cancellation.
    """
    if (
        isinstance(tasks, str)
        or not tasks
        or len(tasks) > len(CONTEXT_TASKS)
        or len(set(tasks)) != len(tasks)
        or set(tasks) - set(CONTEXT_TASKS)
    ):
        raise ClinicalAnalysisError("invalid_clinical_tasks")
    if type(entity_coverage_complete) is not bool:
        raise ClinicalAnalysisError("invalid_clinical_coverage")
    if reference_date is not None:
        if (
            "timeline" not in tasks
            or not isinstance(reference_date, str)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reference_date)
        ):
            raise ClinicalAnalysisError("invalid_clinical_reference_date")
        try:
            date.fromisoformat(reference_date)
        except ValueError:
            raise ClinicalAnalysisError("invalid_clinical_reference_date") from None
    if (
        type(timeout_seconds) not in (float, int)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 30
    ):
        raise ClinicalAnalysisError("invalid_clinical_deadline")
    deadline = time.monotonic() + timeout_seconds

    def check():
        if cancel_check and cancel_check():
            raise ClinicalAnalysisError("clinical_cancelled")
        if time.monotonic() >= deadline:
            raise ClinicalAnalysisError("clinical_timeout")

    check()
    spans = validated_clinical_entities(text, entities)
    try:
        resolved = resolve_clinical_language(text, language=language)
    except (TypeError, ValueError):
        raise ClinicalAnalysisError("invalid_clinical_language") from None
    check()
    if (
        entity_coverage_complete
        and not resolved.needs_review
        and not resolved.mixed
        and resolved.language in CONTEXT_LANGUAGES
    ):
        from openmed.clinical.quantity_evidence import _repair_drug_decimal_boundaries

        spans = _repair_drug_decimal_boundaries(text, spans, resolved.language, check)
    section_language = (
        resolved.language
        if resolved.source == "explicit" or not resolved.needs_review
        else None
    )
    sections = detect_sections(text, language=section_language)
    validate_section_spans(text, sections)
    visible_sections = _section_view(sections)
    for span in spans:
        containing = next(
            (
                section
                for section in visible_sections
                if section["start"] <= span["start"] < span["end"] <= section["end"]
            ),
            None,
        )
        span["section_id"] = containing["id"] if containing else None
    check()
    results = {}
    for task in tasks:
        if task != "sections" and not entity_coverage_complete:
            results[task] = {
                "status": "failed",
                "complete": False,
                "records": [],
                "error": "incomplete_entity_coverage",
            }
        elif task not in {"sections", "entities"} and (
            resolved.mixed
            or resolved.needs_review
            or resolved.language not in CONTEXT_LANGUAGES
        ):
            results[task] = {
                "status": "unsupported",
                "complete": False,
                "records": [],
                "error": "assertion_language_requires_supported_override"
                if task == "assertions"
                else "structured_language_requires_supported_override",
            }
        else:
            results[task] = {
                "status": "needs_review",
                "complete": True,
                "records": [],
                "warnings": ["clinical_task_not_qualified"],
            }
    if "sections" in results:
        results["sections"]["records"] = visible_sections
    if "entities" in results and results["entities"]["complete"]:
        results["entities"]["records"] = spans
    if any(
        results.get(task, {}).get("complete")
        for task in ("assertions", *STRUCTURED_TASKS, *TEMPORAL_TASKS)
    ):
        assertions = assert_context(
            text, spans, language=resolved.language, sections=sections
        )
        cues = scan_context_cues(text, spans, language=resolved.language)
        records = []
        for span, assertion in zip(spans, assertions, strict=True):
            check()
            axes = {
                key: assertion[key]
                for key in ("negation", "uncertainty", "experiencer", "temporality")
            }
            section = next(
                (s for s in visible_sections if s["id"] == span["section_id"]), None
            )

            def in_header(start, end):
                return (
                    section is not None
                    and "header_start" in section
                    and start < section["header_end"]
                    and section["header_start"] < end
                )

            evidence = [
                {
                    "start": hit.start,
                    "end": hit.end,
                    "category": hit.category,
                    "direction": hit.direction,
                    "source": "context_cue",
                }
                for hit in cues[span]
                if not in_header(hit.start, hit.end)
            ]
            assignment = resolve_experiencer(
                text,
                span,
                language=resolved.language,
                section_experiencer=axes["experiencer"],
            )
            if (
                assignment.cue_offset
                and assertion.get("context_sources", {}).get("experiencer") == "local"
                and not in_header(*assignment.cue_offset)
            ):
                evidence.append(
                    {
                        "start": assignment.cue_offset[0],
                        "end": assignment.cue_offset[1],
                        "category": "experiencer",
                        "source": "subject_cue",
                    }
                )
            if span["section_id"]:
                section = next(
                    s for s in visible_sections if s["id"] == span["section_id"]
                )
                if "header_start" in section:
                    evidence.append(
                        {
                            "start": section["header_start"],
                            "end": section["header_end"],
                            "category": "section",
                            "source": "section_header",
                        }
                    )
            records.append(
                {
                    "entity_id": span["id"],
                    "start": span["start"],
                    "end": span["end"],
                    **axes,
                    "context_sources": assertion.get("context_sources", {}),
                    "evidence": evidence,
                }
            )
        if "assertions" in results:
            results["assertions"]["records"] = records
        requested_structured = [
            task for task in STRUCTURED_TASKS if results.get(task, {}).get("complete")
        ]
        if requested_structured:
            from openmed.clinical.structured_analysis import _structure_clinical_tasks

            results.update(
                _structure_clinical_tasks(
                    text,
                    spans,
                    visible_sections,
                    records,
                    language=resolved.language,
                    tasks=requested_structured,
                    check=check,
                )
            )
        requested_temporal = [
            task for task in TEMPORAL_TASKS if results.get(task, {}).get("complete")
        ]
        if requested_temporal:
            from openmed.clinical.temporal_analysis import _temporal_tasks

            results.update(
                _temporal_tasks(
                    text,
                    spans,
                    visible_sections,
                    records,
                    language=resolved.language,
                    tasks=requested_temporal,
                    reference_date=reference_date,
                    check=check,
                )
            )
    check()
    complete = all(result["complete"] for result in results.values())
    return {
        "schema_version": 1,
        "version": CLINICAL_CONTEXT_VERSION,
        "status": "needs_review"
        if complete
        else "partial"
        if any(result["complete"] for result in results.values())
        else "failed",
        "complete": complete,
        "source_chars": len(text),
        "language": {
            "language": resolved.language,
            "locale": resolved.locale,
            "source": resolved.source,
            "confidence": resolved.confidence,
            "mixed": resolved.mixed,
            "needs_review": resolved.needs_review,
        },
        "tasks": results,
        "qualification": {
            "status": "preview",
            "qualified_languages": [],
            "assertion_preview_languages": list(CONTEXT_LANGUAGES),
            "structured_preview_languages": list(CONTEXT_LANGUAGES),
        },
    }
