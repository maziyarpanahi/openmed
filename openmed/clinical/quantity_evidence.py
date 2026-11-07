"""Source-only medication quantity evidence beside a model-detected drug.

Written amounts do not establish a prescribed dose or product strength. These
helpers retain original model offsets when correcting a split decimal boundary.
"""

from __future__ import annotations

import re

from openmed.clinical.medication_sig import normalize_medication_attribute

_QUANTITY = re.compile(
    r"(?<![\w.,/+\-])(?P<number>[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))"
    r"[ \t\u00a0\u202f]*(?:µg|μg|mcg|mg|kg|g|ml|l|iu|ie)"
    r"(?![\w/^*%µμ])",
    re.IGNORECASE,
)


def _quantities(text, language, check):
    """Yield complete, unit-normalizable source quantities without surface text."""
    from openmed.clinical.analysis import ClinicalAnalysisError

    for index, match in enumerate(_QUANTITY.finditer(text)):
        check()
        if index >= 2000:
            raise ClinicalAnalysisError("clinical_quantity_limit")
        parsed = normalize_medication_attribute(
            "dose", match.group(), language=language
        )
        if not parsed["recognized"]:
            continue
        # A regex match ending before a hyphen/slash could be part of a range,
        # concentration or compound unit; never release the truncated amount.
        if re.match(r"[ \t]*[-–—/]", text[match.end() :]):
            continue
        yield (
            match,
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
            },
        )


def _repair_drug_decimal_boundaries(text, entities, language, check):
    """Trim a Drug span only when it ends inside a complete decimal amount.

    Integer drug-name suffixes and absent/ambiguous units cannot trigger repair.
    The retained score describes the original model span, explicitly marked as
    such; it is not a newly calibrated confidence for the corrected boundary.
    """
    if not any(entity["label"] == "Drug" for entity in entities):
        return entities
    quantities = list(_quantities(text, language, check))
    repaired = []
    for entity in entities:
        check()
        replacement = entity
        if entity["label"] == "Drug":
            for match, _ in quantities:
                number_start, number_end = match.span("number")
                if (
                    entity["start"] < number_start < entity["end"] < number_end
                    and number_start > 0
                    and text[number_start - 1].isspace()
                    and ("," if language == "de" else ".") in match.group("number")
                ):
                    end = number_start
                    while end > entity["start"] and text[end - 1] in " \t\u00a0\u202f":
                        end -= 1
                    if end > entity["start"]:
                        replacement = {
                            **entity,
                            "end": end,
                            "score_kind": "model_score_before_boundary_repair",
                            "span_repair": "drug_boundary_inside_decimal_quantity",
                            "source_parts": [dict(entity)],
                            "quantity_evidence": {
                                "start": match.start(),
                                "end": match.end(),
                            },
                        }
                    break
        repaired.append(replacement)
    return repaired


def _written_amounts(text, drug, entities, language, check):
    """Find one immediately adjacent written amount without resolving its meaning."""
    # Limit scanning to the drug's immediate suffix, preserving absolute offsets.
    # Newlines, another drug, and narrative words cannot provide an implicit link.
    end = drug["end"]
    suffix = text[end : min(len(text), end + 128)]
    prefix = re.match(r"[ \t\u00a0\u202f]*(?::[ \t\u00a0\u202f]*)?", suffix)
    start = end + prefix.end()
    if start and (text[start - 1].isalnum() or text[start - 1] in "_µμ.,/+-"):
        return []
    for match, normalized in _quantities(text[start : start + 128], language, check):
        if match.start() != 0:
            return []
        finish = start + match.end()
        if finish < len(text) and re.match(r"[\w/^*%µμ]", text[finish]):
            return []
        if re.match(r"[ \t]*[-–—/]", text[finish:]):
            return []
        supporting = [
            {key: entity[key] for key in ("id", "start", "end", "label", "score")}
            for entity in entities
            if entity["label"] in {"Dose", "Strength"}
            and entity["start"] < finish
            and start < entity["end"]
        ]
        return [
            {
                "source": {"start": start, "end": finish},
                "normalized": normalized,
                "semantic_type": "unspecified",
                "score_kind": "deterministic_source_pattern",
                "source_parts": supporting,
                "requires_review": True,
            }
        ]
    return []
