"""Bounded event and timeline composition over validated clinical source spans."""

from __future__ import annotations

import bisect
import re
from datetime import date
from itertools import islice

from openmed.clinical.context import assert_context, scan_context_cues
from openmed.clinical.events import (
    extract_lab_trend_events,
    extract_medication_change_events,
)
from openmed.clinical.medication_sig import (
    filter_medication_candidates,
    normalize_medication_attribute,
)
from openmed.clinical.structured_analysis import (
    _quantity_boundary_complete,
    _reference,
    _repair_numeric_attributes,
)
from openmed.clinical.temporal_german import GERMAN_TIMEX_RE
from openmed.clinical.temporal_normalizer import normalize_temporal
from openmed.clinical.timeline import assemble_timeline
from openmed.clinical.timeline.timex import detect_timexes

TEMPORAL_TASKS = ("events", "timeline")
MAX_TEMPORAL_SPANS = 2048
MAX_TEMPORAL_RECORDS = 4096
MAX_SCOPE_ENTITIES = 64
MAX_SCOPE_TRIGGERS = 128
_AXES = ("negation", "uncertainty", "temporality", "experiencer")
_HEADS = {
    "Disease",
    "Condition",
    "Problem",
    "Symptom",
    "Sign",
    "Procedure",
    "Drug",
    "Chemical",
    "Lab Test",
    "Vital Sign",
}
_BOUNDARIES = re.compile(
    r"[\r\n;!?]|\.(?=\s|$)|\b(?:but|however|aber|jedoch)\b", re.IGNORECASE
)
_IDENTIFIER_DATE = re.compile(
    r"\b(?:dob|date\s+of\s+birth|born|geburtsdatum|geboren|geb)\b", re.IGNORECASE
)


def _timexes(text, language, reference_date, check):
    from openmed.clinical.analysis import ClinicalAnalysisError

    candidates = (
        ((m.start(), m.end()) for m in GERMAN_TIMEX_RE.finditer(text))
        if language == "de"
        else ((m.start, m.end) for m in detect_timexes(text))
    )
    spans = list(islice(candidates, MAX_TEMPORAL_SPANS + 1))
    if len(spans) > MAX_TEMPORAL_SPANS:
        raise ClinicalAnalysisError("clinical_temporal_span_limit")
    check()
    normalized = normalize_temporal(text, spans, reference_date, language=language)
    records = []
    for timex in normalized:
        check()
        start, end = timex.start, timex.end
        # Birth dates must never become event dates. Also keep Morgen in
        # am/jeden Morgen unresolved rather than interpreting time of day as tomorrow.
        prefix = text[max(0, start - 64) : start]
        identifier = bool(_IDENTIFIER_DATE.search(prefix))
        morning = (
            language == "de"
            and text[start:end].casefold() == "morgen"
            and bool(re.search(r"\b(?:am|jeden|heute)\s*$", prefix, re.IGNORECASE))
        )
        records.append(
            dict(
                start=start,
                end=end,
                value=None if identifier or morning else timex.value,
                anchor=timex.anchor,
                type=timex.timex_type,
                flags=list(timex.granularity_flags)
                + (["identifier_date"] if identifier else [])
                + (["ambiguous_time_of_day"] if morning else []),
            )
        )
    return records


def _windows(text, entities, sections, timexes, check):
    from openmed.clinical.analysis import ClinicalAnalysisError

    cuts = sorted(
        {
            0,
            len(text),
            *(s["start"] for s in sections),
            *(s["end"] for s in sections),
            *(
                m.end()
                for m in _BOUNDARIES.finditer(text)
                if not any(t["start"] <= m.start() < t["end"] for t in timexes)
            ),
        }
    )
    groups = {}
    for entity in entities:
        check()
        first = bisect.bisect_right(cuts, entity["start"]) - 1
        last = bisect.bisect_left(cuts, entity["end"]) - 1
        if first == last:
            groups.setdefault(first, []).append(entity)
    if any(len(group) > MAX_SCOPE_ENTITIES for group in groups.values()):
        raise ClinicalAnalysisError("clinical_scope_entity_limit")
    return [(cuts[index], cuts[index + 1], group) for index, group in groups.items()]


def _context(text, start, end, language, sections):
    span = dict(start=start, end=end, label="EVENT")
    record = assert_context(text, [span], language=language, sections=sections)[0]
    cues = scan_context_cues(text, [span], language=language)
    return {
        **{axis: record[axis] for axis in _AXES},
        "context_sources": record.get("context_sources", {}),
        "evidence": [
            dict(
                start=cue.start,
                end=cue.end,
                category=cue.category,
                source="context_cue",
            )
            for cue in cues[span]
        ],
    }


def _events(text, entities, sections, assertions, language, timexes, check):
    from openmed.clinical.analysis import ClinicalAnalysisError

    contexts = {record["entity_id"]: record for record in assertions}
    entities = [
        e
        for e in entities
        if not any(
            "header_start" in s
            and e["start"] < s["header_end"]
            and s["header_start"] < e["end"]
            for s in sections
        )
    ]
    if len({(e["start"], e["end"]) for e in entities}) != len(entities):
        raise ClinicalAnalysisError("clinical_ambiguous_entity_offsets")
    entities, contexts = _repair_numeric_attributes(
        text, entities, contexts, language, check
    )
    accepted = {
        (c.start, c.end)
        for c in filter_medication_candidates(text, entities, language=language)
    }
    entities = [
        e
        for e in entities
        if e["label"] not in {"Drug", "Chemical"} or (e["start"], e["end"]) in accepted
    ]
    records = []
    for left, right, group in _windows(text, entities, sections, timexes, check):
        check()
        local = text[left:right]
        heads = [e for e in group if e["label"] in _HEADS]
        drugs = [e for e in heads if e["label"] in {"Drug", "Chemical"}]
        labs = [e for e in heads if e["label"] == "Lab Test"]
        by_id = {e["id"]: e for e in group}
        frames = []
        for engine, selected_heads, label in (
            (extract_medication_change_events, drugs, "drug"),
            (extract_lab_trend_events, labs, "analyte"),
        ):
            if not selected_heads:
                continue
            # A single scope with several competing heads cannot establish
            # which one an action modifies from proximity alone.
            if len(selected_heads) != 1:
                continue
            selected = [
                {
                    **e,
                    "label": label,
                    "start": e["start"] - left,
                    "end": e["end"] - left,
                }
                for e in selected_heads
            ]
            for entity in group:
                attr_label = (
                    "dose"
                    if label == "drug" and entity["label"] == "Dose"
                    else "lab_value"
                    if label == "analyte" and entity["label"] == "Lab Value"
                    else None
                )
                if attr_label and _quantity_boundary_complete(
                    text, entity["start"], entity["end"]
                ):
                    selected.append(
                        {
                            **entity,
                            "label": attr_label,
                            "start": entity["start"] - left,
                            "end": entity["end"] - left,
                        }
                    )
            try:
                frames.extend(
                    engine(
                        local,
                        selected,
                        language=language,
                        include_detected_time_anchors=False,
                        max_events=MAX_SCOPE_TRIGGERS,
                    )
                )
            except ValueError as exc:
                if str(exc) == "clinical_event_trigger_limit":
                    raise ClinicalAnalysisError(str(exc)) from None
                raise
        linked = set()
        for frame in frames:
            check()
            head_role, trigger_role = (
                ("drug", "action")
                if frame.event_type == "medication_change"
                else ("analyte", "direction")
            )
            head = by_id[frame.role_slots(head_role)[0].source_id]
            trigger = frame.role_slots(trigger_role)[0]
            start, end = trigger.start + left, trigger.end + left
            if frame.event_type == "medication_change" and trigger.value in {
                "increased",
                "decreased",
            }:

                def gap(entity):
                    return max(0, entity["start"] - end, start - entity["end"])

                if any(gap(lab) < gap(head) for lab in labs):
                    continue  # Elevated CRP beside a medication is not a dose change.
            trigger_context = _context(text, start, end, language, sections)
            attributes = []
            for role, slots in frame.roles.items():
                if role in {head_role, trigger_role}:
                    continue
                for slot in slots:
                    entity = by_id[slot.source_id]
                    attribute = {"role": role, "source": _reference(entity)}
                    if role in {"old_dose", "new_dose"}:
                        parsed = normalize_medication_attribute(
                            "dose",
                            text[entity["start"] : entity["end"]],
                            language=language,
                        )
                        attribute["normalized"] = (
                            {
                                key: parsed[key]
                                for key in (
                                    "recognized",
                                    "value",
                                    "unit",
                                    "canonical_value",
                                    "canonical_unit",
                                    "dimension",
                                )
                                if key in parsed
                            }
                            if parsed["recognized"]
                            else {"recognized": False, "reason": "unsupported_quantity"}
                        )
                    attributes.append(attribute)
            linked.add(head["id"])
            records.append(
                dict(
                    id=f"event-{head['id']}-{start}",
                    type=frame.event_type,
                    source=_reference(head),
                    trigger=dict(
                        start=start,
                        end=end,
                        value=trigger.value,
                        source="clinical_event_lexicon",
                        score_kind="heuristic",
                    ),
                    context=contexts[head["id"]],
                    trigger_context=trigger_context,
                    attributes=attributes,
                    conflicts=[conflict.conflict_type for conflict in frame.conflicts],
                    coding_eligible=False,
                    scope={"start": left, "end": right},
                )
            )
        for head in heads:
            if head["id"] not in linked:
                records.append(
                    dict(
                        id=f"event-{head['id']}",
                        type="clinical_mention",
                        source=_reference(head),
                        context=contexts[head["id"]],
                        trigger=None,
                        attributes=[],
                        conflicts=[],
                        coding_eligible=False,
                        scope={"start": left, "end": right},
                    )
                )
        if len(records) > MAX_TEMPORAL_RECORDS:
            raise ClinicalAnalysisError("clinical_temporal_output_limit")
    return sorted(
        records,
        key=lambda e: (
            e["source"]["start"],
            e["trigger"]["start"] if e["trigger"] else -1,
            e["id"],
        ),
    )


def _date_interval(value):
    if not isinstance(value, str):
        return None
    parts = value.split("/")
    if len(parts) > 2 or any(
        not re.fullmatch(r"\d{4}-\d{2}-\d{2}", part) for part in parts
    ):
        return None
    try:
        dates = [date.fromisoformat(part) for part in parts]
    except ValueError:
        return None
    if dates[-1] < dates[0]:
        return None
    return dict(start=dates[0].isoformat(), end=dates[-1].isoformat())


def _timeline(events, timexes, reference_date, check):
    records, input_events = {}, []
    for event in events:
        check()
        scope = event["scope"]
        candidates = [
            t
            for t in timexes
            if scope["start"] <= t["start"] < t["end"] <= scope["end"]
        ]
        # No nearest-date guess in a clause containing competing dates. No
        # document-date fallback for absent or unresolved temporal evidence.
        anchor = None
        if len(candidates) == 1 and (
            interval := _date_interval(candidates[0]["value"])
        ):
            candidate = candidates[0]
            anchor = dict(
                **interval,
                value=candidate["value"],
                source=dict(start=candidate["start"], end=candidate["end"]),
                reference_date=reference_date if candidate["anchor"] else None,
                basis="explicit_reference_date"
                if candidate["anchor"]
                else "source_date",
            )
        records[event["id"]] = {
            **event,
            "anchor": anchor,
            "temporal_evidence": candidates,
            "ordering_group": "anchored" if anchor else "unanchored",
            "ordering_is_presentation_only": True,
        }
        input_events.append(
            dict(
                id=event["id"],
                start=event["source"]["start"],
                end=event["source"]["end"],
                label=event["source"]["label"],
                event_kind=event["type"],
                normalized_time=anchor["value"] if anchor else None,
            )
        )
    # The established assembler sorts timestamps and retains a separate
    # unanchored bucket. Preserve the original four-axis contexts above rather
    # than projecting them through the assembler's narrower historical enums.
    timeline = assemble_timeline(input_events)
    return [records[event.entity] for event in timeline.events]


def _temporal_tasks(
    text, entities, sections, assertions, *, language, tasks, reference_date, check
):
    from openmed.clinical.analysis import ClinicalAnalysisError

    try:
        timexes = _timexes(text, language, reference_date, check)
        events = _events(text, entities, sections, assertions, language, timexes, check)
        output = {}
        for task in tasks:
            check()
            output[task] = dict(
                status="needs_review",
                complete=True,
                records=events
                if task == "events"
                else _timeline(events, timexes, reference_date, check),
                warnings=[
                    "clinical_task_not_qualified",
                    "event_and_timeline_links_are_candidates",
                ],
            )
        return output
    except ClinicalAnalysisError as exc:
        if str(exc) in {"clinical_cancelled", "clinical_timeout"}:
            raise
        error = str(exc)
    except Exception:
        error = "clinical_temporal_processing_failed"
    return {
        task: dict(status="failed", complete=False, records=[], error=error)
        for task in tasks
    }
