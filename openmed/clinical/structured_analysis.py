"""Bounded, source-aligned composition of existing clinical structuring helpers.

Only the validated analysis boundary calls this module. It never extracts spans
from a model, resolves codes, or confirms a candidate as a patient fact.
"""

from __future__ import annotations

import bisect
import re

from openmed.clinical.lab_values import (
    derive_abnormal_flag,
    link_lab_value_attributes,
    parse_reference_range,
)
from openmed.clinical.medication_sig import (
    filter_medication_candidates,
    normalize_medication_attribute,
)
from openmed.clinical.relations.medication_links import extract_medication_relations
from openmed.clinical.relations_lite import extract_relation_candidates
from openmed.clinical.units import parse_measurement
from openmed.clinical.vital_signs import structure_vital_sign

STRUCTURED_TASKS = ("medications", "labs", "vitals", "relations")
MAX_SCOPE_ENTITIES = 64
MAX_STRUCTURED_RECORDS = 4096
_BOUNDARY = re.compile(
    r"[\r\n;!?]|(?<!\d)\.(?=\s|$)|\b(?:but|however|aber|jedoch)\b", re.IGNORECASE
)
_NUMBERS = re.compile(r"(?<![\w])[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)")
_LAB_ROLES = {
    "Lab Test": "lab_name",
    "Lab Value": "lab_value",
    "Reference Range": "reference_range",
    "Abnormal Flag": "abnormal_flag",
}


def _scopes(text, entities, sections, check):
    cuts = sorted(
        {
            0,
            len(text),
            *(m.end() for m in _BOUNDARY.finditer(text)),
            *(s["start"] for s in sections),
            *(s["end"] for s in sections),
        }
    )
    groups = {}
    for entity in entities:
        check()
        first = bisect.bisect_right(cuts, entity["start"]) - 1
        last = bisect.bisect_left(cuts, entity["end"]) - 1
        if first != last:
            continue  # A span crossing a scope cannot provide a linking edge.
        groups.setdefault(first, []).append(entity)
    from openmed.clinical.analysis import ClinicalAnalysisError

    if any(len(group) > MAX_SCOPE_ENTITIES for group in groups.values()):
        raise ClinicalAnalysisError("clinical_scope_entity_limit")
    return list(groups.values())


def _reference(entity):
    return {
        **{
            key: entity.get(key)
            for key in ("id", "start", "end", "label", "score", "section_id")
        },
        **{
            key: entity[key] for key in ("span_repair", "source_parts") if key in entity
        },
    }


def _repair_numeric_attributes(text, entities, assertions, language, check):
    """Join only same-label adjacent fragments of one valid decimal quantity."""
    separator = {"de": ",", "en": "."}.get(language)
    repaired, contexts = [], dict(assertions)
    position = 0
    axes = ("negation", "uncertainty", "temporality", "experiencer")
    while position < len(entities):
        check()
        left = entities[position]
        right = entities[position + 1] if position + 1 < len(entities) else None
        if (
            right
            and separator
            and left["label"] in {"Dose", "Strength"}
            and right["label"] == left["label"]
            and left["section_id"] == right["section_id"]
            and left["end"] < right["start"]
            and text[left["end"] : right["start"]] == separator
            and re.fullmatch(r"[+-]?\d+", text[left["start"] : left["end"]])
            and (
                left["start"] == 0 or not re.match(r"[\w.,+-]", text[left["start"] - 1])
            )
            and (
                right["end"] == len(text)
                or not re.match(r"[\w/^*%µμ]", text[right["end"]])
            )
            and all(
                assertions[left["id"]][axis] == assertions[right["id"]][axis]
                for axis in axes
            )
            and normalize_medication_attribute(
                "dose", text[left["start"] : right["end"]], language=language
            )["recognized"]
        ):
            combined = {
                **left,
                "id": f"repair:{left['id']}:{right['id']}",
                "end": right["end"],
                "score": min(left["score"], right["score"])
                if left["score"] is not None and right["score"] is not None
                else None,
                "span_repair": "adjacent_decimal_fragments",
                "source_parts": [_reference(left), _reference(right)],
            }
            context = {
                **assertions[left["id"]],
                "entity_id": combined["id"],
                "end": combined["end"],
            }
            context["evidence"] = [
                dict(record) for record in assertions[left["id"]]["evidence"]
            ]
            for record in assertions[right["id"]]["evidence"]:
                if record not in context["evidence"]:
                    context["evidence"].append(dict(record))
            contexts[combined["id"]] = context
            repaired.append(combined)
            position += 2
        else:
            repaired.append(left)
            position += 1
    return repaired, contexts


def _context(entity, assertions):
    return assertions[entity["id"]]


def _quantity_boundary_complete(text, start, end):
    """Reject a number or unit cut out of a larger original-source quantity."""
    if start and (text[start - 1].isalnum() or text[start - 1] in "_µμ"):
        return False
    if start and text[start - 1] in "+-" and text[start].isdigit():
        return False
    if (
        start > 1
        and text[start - 1] in ".,/"
        and text[start - 2].isdigit()
        and text[start].isdigit()
    ):
        return False
    if end < len(text) and (text[end].isalnum() or text[end] in "_/^*%µμ"):
        return False
    if (
        end + 1 < len(text)
        and text[end] in ".,/"
        and text[end - 1].isdigit()
        and text[end + 1].isdigit()
    ):
        return False
    return True


def _measurement(text, entity, language):
    if not _quantity_boundary_complete(text, entity["start"], entity["end"]):
        return {
            "status": "unknown",
            "canonical_magnitude": None,
            "canonical_unit": None,
            "dimension": None,
            "reason": "incomplete_quantity_span",
        }, {}
    result = parse_measurement(text[entity["start"] : entity["end"]], language=language)
    return {
        key: result.get(key)
        for key in ("status", "canonical_magnitude", "canonical_unit", "dimension")
    }, result


def _normalized_range(reference, unit, language):
    bounds = {
        name: parse_measurement(reference[name], unit, language=language)
        for name in ("low", "high")
        if reference[name] is not None
    }
    complete = bool(bounds) and all(item["status"] == "ok" for item in bounds.values())
    return {
        "status": "ok" if complete else "unknown",
        "low": bounds.get("low", {}).get("canonical_magnitude") if complete else None,
        "high": bounds.get("high", {}).get("canonical_magnitude") if complete else None,
        "unit": next(iter(bounds.values()))["canonical_unit"] if complete else None,
        "low_inclusive": reference["low_inclusive"],
        "high_inclusive": reference["high_inclusive"],
    }


def _medications(text, entities, sections, assertions, language, check):
    entities, assertions = _repair_numeric_attributes(
        text, entities, assertions, language, check
    )
    accepted = {
        (c.start, c.end)
        for c in filter_medication_candidates(text, entities, language=language)
    }
    drugs = [
        e
        for e in entities
        if e["label"] in {"Drug", "Chemical"} and (e["start"], e["end"]) in accepted
    ]
    relevant = [
        e
        for e in entities
        if e in drugs
        or e["label"] in {"Dose", "Route", "Frequency", "Duration", "Form", "Strength"}
    ]
    by_offset = {(e["start"], e["end"]): e for e in relevant}
    attributes = {e["id"]: [] for e in drugs}
    for group in _scopes(text, relevant, sections, check):
        check()
        links = extract_medication_relations(
            text, [{**e, "score": e["score"] or 0.0} for e in group], sections=sections
        )
        for link in links:
            check()
            head = by_offset[link.head.offset_key()]
            tail = by_offset[link.tail.offset_key()]
            if head["id"] not in attributes:
                continue
            normalized = normalize_medication_attribute(
                "dose" if link.type == "strength" else link.type,
                text[tail["start"] : tail["end"]],
                language=language,
            )
            if normalized is not None:
                if link.type in {
                    "dose",
                    "strength",
                } and not _quantity_boundary_complete(text, tail["start"], tail["end"]):
                    normalized = {
                        "recognized": False,
                        "reason": "incomplete_quantity_span",
                    }
                elif not normalized.get("recognized"):
                    normalized = {
                        "recognized": False,
                        "reason": "unrecognized_medication_attribute",
                    }
                else:
                    normalized = {
                        key: value
                        for key, value in normalized.items()
                        if key not in {"raw", "cue", "advisory", "reason"}
                    }
            attributes[head["id"]].append(
                {
                    "type": link.type,
                    "source": _reference(tail),
                    "normalized": normalized,
                    "link_score": link.score,
                    "score_kind": "heuristic",
                }
            )
    return [
        {
            "source": _reference(drug),
            "context": _context(drug, assertions),
            "attributes": attributes[drug["id"]],
            "grounding_performed": False,
            "candidate_status": "unconfirmed",
            "coding_eligible": False,
        }
        for drug in drugs
    ]


def _labs(text, entities, sections, assertions, language, check):
    relevant = [e for e in entities if e["label"] in _LAB_ROLES]
    by_id = {e["id"]: e for e in relevant}
    links = {}
    for group in _scopes(text, relevant, sections, check):
        check()
        graph = link_lab_value_attributes(
            [{**e, "label": _LAB_ROLES[e["label"]]} for e in group]
        )
        for edge in graph.edges:
            links[edge.head, edge.label] = edge.tail
    records, used_values = [], set()
    for head in relevant:
        if head["label"] != "Lab Test":
            continue
        check()
        value_id = links.get((head["id"], "has_value"))
        value = by_id.get(value_id)
        record = {
            "source": _reference(head),
            "context": _context(head, assertions),
            "measurement": None,
            "reference_range": None,
            "abnormal_flag": "unknown",
            "abnormal_flag_source": None,
            "evidence": [],
            "extraction_status": "unlinked",
            "coding_eligible": False,
        }
        if value:
            used_values.add(value_id)
            measurement, parsed = _measurement(text, value, language)
            record["measurement"] = measurement
            record["extraction_status"] = (
                "parsed" if measurement["status"] == "ok" else "unparsed"
            )
            record["evidence"].append({"role": "value", **_reference(value)})
            range_entity = by_id.get(links.get((value_id, "has_reference_range")))
            flag_entity = by_id.get(links.get((value_id, "has_abnormal_flag")))
            reference = None
            if range_entity:
                reference = parse_reference_range(
                    text[range_entity["start"] : range_entity["end"]]
                    if _quantity_boundary_complete(
                        text, range_entity["start"], range_entity["end"]
                    )
                    else "",
                    language=language,
                )
                record["evidence"].append(
                    {"role": "reference_range", **_reference(range_entity)}
                )
                range_unit = reference.get("unit") or parsed.get("provenance", {}).get(
                    "input_unit"
                )
                record["reference_range"] = _normalized_range(
                    reference, range_unit, language
                )
            explicit_flag = (
                text[flag_entity["start"] : flag_entity["end"]] if flag_entity else None
            )
            if flag_entity:
                record["evidence"].append(
                    {"role": "abnormal_flag", **_reference(flag_entity)}
                )
            if measurement["status"] == "ok":
                record["abnormal_flag"] = derive_abnormal_flag(
                    measurement["canonical_magnitude"],
                    reference,
                    explicit_flag,
                    value_unit=measurement["canonical_unit"],
                    reference_unit=reference.get("unit")
                    if reference and reference.get("unit")
                    else parsed.get("provenance", {}).get("input_unit"),
                    language=language,
                )
                record["abnormal_flag_source"] = (
                    "explicit"
                    if flag_entity
                    else "reference_comparison"
                    if reference
                    else None
                )
        records.append(record)
    for value in relevant:
        if value["label"] == "Lab Value" and value["id"] not in used_values:
            records.append(
                {
                    "source": _reference(value),
                    "context": _context(value, assertions),
                    "extraction_status": "unlinked_value",
                    "measurement": None,
                    "coding_eligible": False,
                }
            )
    return sorted(records, key=lambda r: (r["source"]["start"], r["source"]["end"]))


def _vitals(text, entities, sections, assertions, language, check):
    records = []
    for entity in entities:
        if entity["label"] != "Vital Sign":
            continue
        check()
        surface = text[entity["start"] : entity["end"]]
        result = structure_vital_sign(surface, language=language)
        expected = 2 if result["kind"] == "blood_pressure" else 1
        unambiguous = len(_NUMBERS.findall(surface)) == expected
        parsed = result["kind"] != "unknown" and unambiguous
        records.append(
            {
                "source": _reference(entity),
                "context": _context(entity, assertions),
                "extraction_status": "parsed" if parsed else "unparsed",
                "reason": None if parsed else "unknown_or_ambiguous_vital_span",
                "measurement": result if parsed else None,
                "coding_eligible": False,
            }
        )
    return records


def _relations(text, entities, sections, assertions, language, check):
    entities, assertions = _repair_numeric_attributes(
        text, entities, assertions, language, check
    )
    entities = [
        e
        for e in entities
        if e["label"] not in {"Dose", "Strength"}
        or _quantity_boundary_complete(text, e["start"], e["end"])
    ]
    records = []
    for group in _scopes(text, entities, sections, check):
        check()
        contextual = [
            {**e, "score": e["score"] or 0.0, "assertion": assertions[e["id"]]}
            for e in group
        ]
        for relation in extract_relation_candidates(
            text,
            contextual,
            sections=sections,
            allow_cross_sentence=False,
            asserted_only=False,
        ):
            check()
            if len(records) >= MAX_STRUCTURED_RECORDS:
                from openmed.clinical.analysis import ClinicalAnalysisError

                raise ClinicalAnalysisError("clinical_structured_output_limit")
            by_offset = {(e["start"], e["end"]): e for e in group}
            head, tail = (
                by_offset[relation.head.offset],
                by_offset[relation.tail.offset],
            )
            records.append(
                {
                    "head": _reference(head),
                    "tail": _reference(tail),
                    "type": "DRUG_STRENGTH"
                    if relation.relation_type == "DRUG_DOSE"
                    and tail["label"] == "Strength"
                    else relation.relation_type,
                    "score": relation.confidence,
                    "score_kind": "heuristic",
                    "head_context": _context(head, assertions),
                    "tail_context": _context(tail, assertions),
                    "coding_eligible": False,
                }
            )
    return records


def _structure_clinical_tasks(
    text, entities, sections, assertions, *, language, tasks, check
):
    """Compose requested tasks from validated source spans and context records."""
    from openmed.clinical.analysis import ClinicalAnalysisError

    context = {record["entity_id"]: record for record in assertions}
    entities = [
        entity
        for entity in entities
        if not any(
            "header_start" in section
            and entity["start"] < section["header_end"]
            and section["header_start"] < entity["end"]
            for section in sections
        )
    ]
    ambiguous_offsets = len({(e["start"], e["end"]) for e in entities}) != len(entities)
    processors = {
        "medications": _medications,
        "labs": _labs,
        "vitals": _vitals,
        "relations": _relations,
    }
    output = {}
    for task in tasks:
        check()
        try:
            if ambiguous_offsets:
                raise ClinicalAnalysisError("clinical_ambiguous_entity_offsets")
            records = processors[task](
                text, entities, sections, context, language, check
            )
            check()
            output[task] = {
                "status": "needs_review",
                "complete": True,
                "records": records,
                "warnings": ["clinical_task_not_qualified"],
            }
            if task == "medications":
                output[task]["warnings"].append(
                    "medication_candidate_threshold_0.75_no_grounding"
                )
        except ClinicalAnalysisError as exc:
            if str(exc) not in {
                "clinical_scope_entity_limit",
                "clinical_structured_output_limit",
                "clinical_ambiguous_entity_offsets",
            }:
                raise
            output[task] = {
                "status": "failed",
                "complete": False,
                "records": [],
                "error": str(exc),
            }
        except Exception:
            output[task] = {
                "status": "failed",
                "complete": False,
                "records": [],
                "error": "clinical_structuring_failed",
            }
    return output
