"""Offline synthetic SDOH/timeline contracts; no clinical validation.

Run ``python -m examples.v30_sdoh_timeline`` from the repository root. The
example accepts no arguments, files, stdin notes, model or network providers.
Printed JSONL contains controlled labels, source offsets and states only.
The fixed reference date and normalized temporal values remain internal.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from openmed.clinical import TimeExpr, build_timeline, normalize_temporal
from openmed.clinical.sdoh import extract_sdoh
from openmed.clinical.sections import detect_sections
from openmed.core.offline import network_blocked_if_offline

SYNTHETIC_NOTE = (
    "Assessment: retired teacher.\n"
    "Social History: unemployed; lives alone; food insecurity.\n"
    "History of Present Illness: fever 3 days ago; cough today; "
    "fatigue since 03/04/2026.\n"
    "Plan: possible follow-up in 2 days."
)
REFERENCE_TIME = "2026-06-15"


def _span(surface: str) -> tuple[int, int]:
    start = SYNTHETIC_NOTE.index(surface)
    return start, start + len(surface)


def build_demo_records() -> list[dict[str, Any]]:
    """Run actual public APIs on embedded synthetic data without model loading.

    Returns:
        Controlled-label, offset and state projections suitable for this demo.
        Source text, values, scores, dates and arbitrary metadata are omitted.

    Raises:
        RuntimeError: If the fixed synthetic public-API contract has drifted.
    """
    with network_blocked_if_offline(local_only=True):
        sections = detect_sections(SYNTHETIC_NOTE)
        findings = extract_sdoh(SYNTHETIC_NOTE, spans=[], sections=sections)
        sdoh_rows = sorted((item.category, item.status, item.span) for item in findings)
        if sdoh_rows != [
            ("employment", "unemployed", (45, 55)),
            ("food_insecurity", "current", (70, 85)),
            ("living_status", "lives_alone", (57, 68)),
        ]:
            raise RuntimeError("synthetic_sdoh_contract_drift")

        temporal_surfaces = ("3 days ago", "today", "03/04/2026", "in 2 days")
        normalized = normalize_temporal(
            SYNTHETIC_NOTE,
            [_span(surface) for surface in temporal_surfaces],
            reference_time=REFERENCE_TIME,
        )
        if tuple(item.value for item in normalized) != (
            "2026-06-12",
            "2026-06-15",
            None,
            "2026-06-17",
        ):
            raise RuntimeError("synthetic_temporal_contract_drift")

        # build_timeline accepts TimeExpr, not NormalizedTimex directly. Preserve
        # the normalizer's offsets/value; never substitute a guessed date.
        timexes = tuple(
            TimeExpr(
                text=item.text,
                start=item.start,
                end=item.end,
                kind=item.type,
                value=item.value,
                anchor=item.anchor,
                reference_time=REFERENCE_TIME,
            )
            for item in normalized
        )
        # These axes are explicitly authored synthetic fixture annotations,
        # rather than claims of automatic event/assertion extraction.
        event_annotations = (
            ("fever", "historical", "certain"),
            ("cough", "recent", "certain"),
            ("fatigue", "recent", "uncertain"),
            ("follow-up", "hypothetical", "uncertain"),
        )
        spans = [
            {
                "text": surface,
                "label": "symptom" if index < 3 else "event",
                "start": _span(surface)[0],
                "end": _span(surface)[1],
                "temporality": temporality,
                "certainty": certainty,
                "negation": "affirmed",
                "experiencer": "patient",
                "time_expr": timexes[index],
            }
            for index, (surface, temporality, certainty) in enumerate(event_annotations)
        ]
        timeline = build_timeline(spans)
        if tuple(
            len(timeline.lanes[lane])
            for lane in ("historical", "recent", "hypothetical")
        ) != (1, 2, 1):
            raise RuntimeError("synthetic_timeline_contract_drift")

    records: list[dict[str, Any]] = [
        {
            "record": "demonstration",
            "data": "synthetic",
            "clinical_validation": "not_validated",
            "review": "required",
        }
    ]
    records.extend(
        {"record": "sdoh", "category": category, "status": status, "span": list(span)}
        for category, status, span in sdoh_rows
    )
    records.extend(
        {
            "record": "temporal",
            "kind": item.type,
            "span": list(item.span),
            "state": (
                "ambiguous"
                if "ambiguous" in item.granularity_flags
                else "anchored"
                if item.value is not None
                else "unanchored"
            ),
        }
        for item in normalized
    )
    records.extend(
        {
            "record": "timeline",
            "kind": event.event_kind,
            "span": list(event.span),
            "temporality": event.assertion.temporality,
            "certainty": event.assertion.certainty,
            "negation": event.assertion.negation,
            "experiencer": event.assertion.experiencer,
            "time_state": evidence.state if evidence is not None else "unanchored",
            "review": "required",
        }
        for event, evidence in zip(timeline.events, timeline.time_evidence, strict=True)
    )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """Print the fixed synthetic demonstration or a content-free failure code.

    Args:
        argv: CLI argument vector, used only to refuse any supplied arguments.

    Returns:
        Zero on success, two for arguments, or one for contract failure.
    """
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        print("synthetic_example_accepts_no_arguments", file=sys.stderr)
        return 2
    failed = False
    try:
        records = build_demo_records()
    except Exception:
        failed = True
    if failed:
        print("synthetic_example_contract_failed", file=sys.stderr)
        return 1
    for record in records:
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
