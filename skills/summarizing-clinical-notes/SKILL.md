---
name: summarizing-clinical-notes
description: "Produces structured, citation-anchored summaries of clinical notes — one-liner, hospital course, and problem-oriented views — where every claim cites a source span so nothing is hallucinated. Use after de-identifying notes when the user wants a discharge summary draft, handoff/SBAR, problem list, or chart-abstraction summary. De-identify FIRST with openmed.deidentify, then anchor summary claims to entity spans from openmed.analyze_text. Trigger keywords: summarize note, discharge summary, hospital course, problem-oriented, one-liner, SOAP, SBAR, handoff, chart abstraction."
license: Apache-2.0
metadata:
  project: OpenMed
  category: clinical-nlp
  pairs: after
  version: "1.0"
---

# Summarizing Clinical Notes with Span Citations

A clinical summary is only useful if it is *faithful*: every statement must
trace back to something the chart actually says. The failure mode for
note summarization is the confident hallucination — an invented dose, a
fabricated allergy, a discharge diagnosis that was never made. This skill
produces summaries where **each line cites the source span** that supports it,
so a clinician can verify in one glance and catch any fabrication.

> **Not a medical device.** OpenMed and this skill assist documentation; they
> do not diagnose, triage, or make autonomous clinical decisions. Every summary
> is a *draft for clinician review and editing*. Surface that disclaimer in any
> UI that renders these summaries.

## When to use

- Drafting a discharge summary, transfer note, or SBAR/handoff from a long
  encounter.
- Building a problem-oriented view (problem list with supporting evidence).
- Generating a "one-liner" (the single-sentence patient summary) for rounds.
- Chart abstraction where reviewers need quick, verifiable evidence pointers.

## Quick start

De-identify before anything else, extract entities to anchor against, then
compose the summary with citations:

```python
import openmed

note = """\
HPI: 68M with HTN, T2DM presents with 3 days of productive cough and fever to
38.9C. CXR shows RLL infiltrate. Started on ceftriaxone and azithromycin.
Hospital course: improved on IV antibiotics, transitioned to PO. Discharged on
amoxicillin-clavulanate. Follow up with PCP in 1 week.
"""

# 1) ALWAYS de-identify before summarizing or sending text anywhere.
deid = openmed.deidentify(note, method="replace", policy="hipaa_safe_harbor")

# 2) Extract entities; their offsets become your citation anchors.
ner = openmed.analyze_text(deid.text, output_format="dict")
spans = {
    (e["start"], e["end"]): e["text"]
    for e in ner["entities"]
}

# 3) Compose the summary. Every bullet references a (start, end) span so a
#    reviewer can click back to the exact evidence.
def cite(start, end):
    return f"[{start}:{end}] {deid.text[start:end]!r}"

# Example problem-oriented line, grounded in detected spans:
# "Community-acquired pneumonia (RLL infiltrate) — treated with ceftriaxone +
#  azithromycin." with cite(...) anchors for each entity.
```

`analyze_text` returns entities as
`{"text", "label", "confidence", "start", "end", "metadata"}`; the
`start`/`end` offsets index the de-identified text, giving you exact,
verifiable citation anchors.

## Workflow

1. **De-identify** with `openmed.deidentify`. Summaries are often shared or
   logged; PHI must be gone before this stage. Keep the mapping
   (`keep_mapping=True`) only if a downstream clinician must re-identify in a
   controlled context — never persist the mapping with the summary.
2. **Extract grounding spans** with `openmed.analyze_text` (problems, meds,
   labs, procedures). These define the *allowed evidence set*: a summary claim
   that cannot point at a span is unsupported.
3. **Resolve context** with `openmed.clinical` (negation, temporality, subject)
   so "no chest pain" and "father had MI" are not summarized as active patient
   problems. See `resolving-clinical-context`.
4. **Compose by view:**
   - **One-liner:** age/sex + key chronic problems + reason for encounter.
   - **Hospital course:** ordered problems → intervention → response, each line
     citing the spans it summarizes.
   - **Problem-oriented:** group entities into problems; attach supporting
     med/lab/procedure spans under each.
5. **Enforce citation coverage.** Reject or flag any output sentence with zero
   span citations. This is the anti-hallucination gate — keep it strict.
6. **Mark it a draft.** Render the medical-device disclaimer and require human
   sign-off before the summary enters the record.

## Hand-off to / from OpenMed

- **From OpenMed:** consumes `openmed.deidentify(...)` output (de-identified
  text + entity spans) and `openmed.analyze_text(...)` (`PredictionResult`
  dict). Entity `start`/`end` offsets are the citation anchors.
- **To OpenMed:** the summary text itself can be re-run through
  `openmed.analyze_text` for a coded problem list, or through `openmed.eval`
  leakage gates to confirm no PHI leaked into the generated summary.
- **Citation rendering:** `analyze_text(..., output_format="html")` produces a
  span-highlighted view of the source — handy for a click-to-evidence UI.

## Edge cases & gotchas

- **Hallucination is the failure mode.** If your summary backbone is an LLM,
  constrain it to the entity/span set and require a citation per sentence; do
  not let it introduce facts (doses, diagnoses, dates) absent from the spans.
- **Negation & family history.** Always run context resolution first; "denies",
  "ruled out", "FH of" must not become patient problems.
- **Copy-forward / note bloat.** EHR notes carry stale copy-pasted blocks. Cite
  the most recent supporting span and prefer the current encounter's text.
- **Conflicting statements.** When the chart contradicts itself (two different
  discharge diagnoses), surface both with citations rather than silently
  picking one.
- **No autonomous action.** Never auto-finalize, auto-sign, or auto-route a
  summary; it is decision support, not a clinical decision.
- **PHI in the summary.** A summary can re-introduce identifiers the model
  missed in the source. Run the *output* through `openmed.extract_pii` or an
  `openmed.eval` leakage gate before display or storage.

## Standards & references

- HL7 C-CDA Discharge Summary / Continuity of Care Document section structure:
  https://www.hl7.org/ccdasearch/
- Joint Commission discharge summary required elements (CAMH / record of care):
  https://www.jointcommission.org/
- SBAR handoff communication (IHI):
  https://www.ihi.org/resources/tools/sbar-tool-situation-background-assessment-recommendation
- Weed LL, problem-oriented medical record (POMR) — the origin of
  problem-oriented summaries: N Engl J Med, 1968.
- FDA Clinical Decision Support Software guidance (device vs. non-device CDS):
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
