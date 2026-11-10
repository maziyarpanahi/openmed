---
name: building-patient-timelines
description: "Assemble a chronological patient timeline from OpenMed-extracted clinical events, normalizing dates and resolving relative time expressions on-device. Use when the user wants to build a patient timeline, order events from clinical notes, reconstruct a longitudinal history, plot a course of illness, or turn analyze_text/deidentify output into a sorted sequence of dated encounters, diagnoses, medications, and procedures. Covers temporal normalization (absolute and relative), event modeling toward FHIR Encounter/Condition.onsetDateTime, anchoring to a document/admission date, and handling undated or ambiguous events. Consumes OpenMed analyze_text entities plus clinical temporality (resolving-clinical-context); produces a sorted event list ready for charting or FHIR export."
license: Apache-2.0
metadata:
  project: OpenMed
  category: analytics-reporting
  pairs: after
  version: "1.0"
---

# Building patient timelines

A patient timeline is a chronologically ordered list of clinical events —
diagnoses, medications, procedures, encounters — each carrying a normalized
date. OpenMed gives you the **events** (via `analyze_text`) and the **clinical
temporality** of each mention (current vs. historical, see
`resolving-clinical-context`); this skill turns those into a sorted timeline.
Everything runs **on-device** — de-identify first if the source notes contain
PHI, and keep raw identifiers out of logs.

## When to use this skill

After you have extracted entities from one or more notes and want them ordered
in time: a longitudinal history, a "course of illness" view, a feed for a
summary card, or a pre-step before FHIR export. If you only need to *extract*
entities, use `extracting-clinical-entities`. If you need negation/temporality
on a single mention, use `resolving-clinical-context`.

## Quick start

```python
import datetime as dt
import openmed

note = (
    "Discharge summary, 2024-03-12. Patient admitted 2024-03-08 with chest pain. "
    "History of type 2 diabetes diagnosed in 2019. Started on metformin two days "
    "after admission. Cardiac catheterization performed yesterday."
)

# 1) Extract clinical events (entities carry char offsets: start/end)
result = openmed.analyze_text(note, output_format="dict")
events = result["entities"]   # each: {text, label, confidence, start, end}

# 2) Normalize the temporal frame: an explicit document/anchor date drives
#    resolution of relative expressions ("two days after", "yesterday").
anchor = dt.date(2024, 3, 12)  # parsed from the note header or document metadata
```

`analyze_text` returns `{text, entities, model_name, timestamp, ...}`; each
entity is `{text, label, confidence, start, end}`. Use `start`/`end` to locate
each event in the source and to find the nearest date expression.

## Workflow

1. **De-identify if needed.** If notes carry PHI, run `openmed.deidentify(...)`
   first, or keep the timeline keyed by stable internal IDs — never log raw
   names/MRNs.
2. **Extract events.** `openmed.analyze_text(note)` for conditions, drugs,
   procedures; pick the model that matches your target entities
   (`choosing-openmed-models`).
3. **Resolve temporality.** For each event, use `resolving-clinical-context`
   to tag it `current` / `historical` / `hypothetical` and to drop negated or
   family-history mentions that should not appear on the patient's own line.
4. **Normalize dates.** Map each event to a date:
   - **Absolute** (`2024-03-08`, `March 2019`) → parse directly. Record the
     granularity (day / month / year) — a year-only event sorts to a coarse
     bucket, not a fake `Jan 1`.
   - **Relative** (`two days after admission`, `yesterday`, `on POD 2`) →
     resolve against an **anchor**: the document date, admission date, or a
     prior event's date. Without an anchor, relative expressions are
     unresolvable — flag them, don't guess.
5. **Build event records.** One record per event: `(date, granularity, label,
   surface_text, char_span, temporality, confidence, source_note_id)`.
6. **Sort and de-duplicate.** Sort by `(date, granularity)`; merge repeated
   mentions of the same event across notes (same label + overlapping date).
7. **Emit.** A sorted list for a UI, or FHIR resources (see hand-off).

## Worked example: events → sorted timeline

```python
def to_timeline(events, *, anchor, note_id):
    """events: list of {text,label,start,end,confidence}. anchor: date.
    Returns sorted [(date, granularity, label, text, confidence)]."""
    timeline = []
    for e in events:
        date, gran = resolve_event_date(e, note=note, anchor=anchor)  # your resolver
        if date is None:
            continue  # undated/unresolvable: route to an "undated" bucket, don't drop silently
        timeline.append((date, gran, e["label"], e["text"], e["confidence"]))
    # year-only ('Y') sorts before month ('M') before day ('D') on ties
    order = {"Y": 0, "M": 1, "D": 2}
    return sorted(timeline, key=lambda r: (r[0], order[r[1]]))

# resolve_event_date handles: ISO dates, "March 2019" (gran='M'),
# "yesterday"/"two days after admission" (relative to anchor/admission), POD-n, etc.
```

## Hand-off to / from OpenMed

- **From OpenMed:** `analyze_text` entities (`extracting-clinical-entities`) and
  `clinical` context tags (`resolving-clinical-context`) are the inputs. Run
  `deidentify` upstream when notes carry PHI.
- **To OpenMed / interop:** feed the sorted, dated events into
  `exporting-to-fhir` (`openmed.interop`). Map an admission/discharge event to a
  FHIR `Encounter`, a diagnosis date to `Condition.onsetDateTime`, a med-start
  to `MedicationStatement.effectiveDateTime`, a procedure to
  `Procedure.performedDateTime`.
- **Downstream:** the same timeline feeds `etl-to-omop-cdm` (start/end dates on
  `condition_occurrence` / `drug_exposure`) and clinical-summary cards.

## Edge cases & gotchas

- **No anchor → no relative dates.** "Two days later", "POD 2", "yesterday" are
  meaningless without a reference date. Parse the document date / admission date
  first; if absent, keep the event in an *undated* bucket rather than inventing
  a date.
- **Preserve granularity.** Don't coerce "2019" to `2019-01-01` and then sort it
  as if it were a precise day — it'll outrank real January events. Carry a
  granularity flag and sort coarse dates conservatively.
- **Drop the wrong people and tenses.** Negated ("no prior MI"), hypothetical
  ("would consider surgery if…"), and family-history mentions must not land on
  the patient's timeline. That's what the temporality pass is for.
- **Time zones and 2-digit years** are ambiguous — normalize to dates (not
  datetimes) for clinical timelines unless you genuinely have timestamps, and
  resolve `dd/mm` vs `mm/dd` from the document locale, not a guess.
- **Future/scheduled events** (follow-up appointments) are real but belong on a
  separate "planned" lane, not interleaved with what already happened.
- **No raw PHI in logs.** Log timeline events by label + offset + note id, never
  the patient's name or the raw note text.

## Standards & references

- FHIR R4 Encounter: https://www.hl7.org/fhir/encounter.html
- FHIR R4 Condition (`onsetDateTime`, `recordedDate`):
  https://www.hl7.org/fhir/condition.html
- ISO 8601 date/time representation:
  https://www.iso.org/iso-8601-date-and-time-format.html
- Background on clinical temporal expression normalization (TimeML / i2b2 2012
  temporal relations task): https://www.i2b2.org/NLP/TemporalRelations/
- OpenMed source: `openmed/processing/` (`analyze_text` output), `openmed.clinical`
  (temporality), `openmed.interop` (FHIR export).
