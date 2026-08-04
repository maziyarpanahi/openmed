---
name: reconciling-problem-lists
description: "Deduplicate and reconcile OpenMed-extracted conditions into one clean active problem list with clinical status (active / resolved / historical). Use after NER and context resolution when the user wants a problem list, condition reconciliation, dedup of synonymous diagnosis mentions, or active-vs-resolved status from a note. Covers clustering synonymous mentions into one concept, excluding negated mentions, applying clinical context (historical / hypothetical / recent) to set status, and emitting a USCDI-Problem-shaped list. SNOMED CT concept grounding is user-supplied and out-of-process. Hand-off: consume openmed.analyze_text Disease entities plus resolving-clinical-context axes. Pairs after extracting-clinical-entities."
license: Apache-2.0
metadata:
  project: OpenMed
  category: clinical-nlp
  pairs: after
  version: "1.0"
---

# Reconciling problem lists

A single note mentions the same condition many ways — "DM2," "type 2 diabetes,"
"diabetes mellitus" — across PMH, HPI, and A&P, some negated, some historical.
A usable **problem list** collapses those mentions into one concept per problem,
drops what the patient does not have, and assigns a clinical status (active /
resolved / historical). This skill turns OpenMed's per-mention entity stream plus
ConText axes into that reconciled, de-duplicated list, shaped for USCDI "Problem"
exchange.

## When to use

- After `extracting-clinical-entities` and `resolving-clinical-context`, when the
  user wants a clean problem list, condition reconciliation, or dedup of repeated
  diagnosis mentions.
- You need active-vs-resolved-vs-historical status per problem, not just raw
  mentions.
- You are assembling a FHIR Condition list or a USCDI Problem element and need
  one entry per concept.

## Quick start

```python
import openmed
from openmed.clinical import resolve_span_context, NEGATED, HISTORICAL, HYPOTHETICAL

note = ("PMH: type 2 diabetes, prior MI 2019 (resolved). "
        "A&P: poorly controlled DM2; denies chest pain.")

ents = openmed.analyze_text(note, model_name="disease_detection_superclinical",
                            output_format="dict")

def normalize(surface: str) -> str:
    # Cheap synonym folding; replace with SNOMED grounding (out-of-process).
    s = surface.lower().strip()
    return {"dm2": "type 2 diabetes", "diabetes mellitus": "type 2 diabetes"}.get(s, s)

problems = {}  # concept -> reconciled record
for e in ents:
    surface = e["word"]
    ctx = resolve_span_context(surface, note)
    if ctx.negation == NEGATED:
        continue                                   # patient does NOT have it -> exclude
    concept = normalize(surface)
    status = ("resolved" if ctx.temporality == HISTORICAL else
              "active")
    if ctx.temporality == HYPOTHETICAL:
        continue                                   # not asserted as present
    rec = problems.setdefault(concept, {"concept": concept, "status": status,
                                        "mentions": 0})
    rec["mentions"] += 1
    # Active anywhere wins over a historical mention of the same concept.
    if status == "active":
        rec["status"] = "active"

problem_list = list(problems.values())
# -> [{"concept": "type 2 diabetes", "status": "active", "mentions": 2}, ...]
# "chest pain" excluded (negated); "MI" -> historical/resolved.
```

## Workflow

1. **Collect Disease/Condition entities** from `analyze_text` across the whole
   note (or per section if you ran `segmenting-clinical-sections`).
2. **Attach clinical context** per mention with `resolve_span_context` (or the
   axes from `resolving-clinical-context`): negation, temporality, uncertainty.
3. **Exclude what isn't a problem.** Drop `NEGATED` mentions (patient denies /
   no evidence of) and `HYPOTHETICAL` mentions (conditional, not asserted). These
   must never land on the active list.
4. **Cluster synonymous mentions into one concept.** Fold surface variants
   (abbreviations, word order, lexical synonyms) to a single canonical key.
   Cheap normalization gets you started; **SNOMED CT concept grounding** is the
   robust path — run it out-of-process with the user's own license and key on the
   concept code, not the surface string.
5. **Assign status by aggregating context.** A concept that is `RECENT`/active
   anywhere (typically A&P) is **active**; one seen only as `HISTORICAL`
   ("history of," "resolved," PMH-only) is **resolved/historical**. Active wins
   over historical when the same concept appears both ways.
6. **Emit the reconciled list** — one record per concept with status, mention
   count, and provenance offsets — shaped for USCDI Problem / FHIR Condition.

## Hand-off to / from OpenMed

- **From** `extracting-clinical-entities`: consumes `analyze_text` Disease
  entities. Run on a sectioned note (`segmenting-clinical-sections`) for best
  active-vs-historical signal.
- **From** `resolving-clinical-context`: this skill *depends* on the negation /
  temporality / uncertainty axes — reconciliation without them would put "denies
  chest pain" on the active list.
- **OpenMed calls:** `from openmed import analyze_text` and
  `from openmed.clinical import resolve_span_context, NEGATED, HISTORICAL,
  HYPOTHETICAL`.
- **To FHIR / USCDI:** each reconciled problem becomes a Condition with
  `clinicalStatus` active/resolved (from temporality) and `verificationStatus`
  refuted/provisional (from negation/uncertainty). SNOMED CT codes are
  user-supplied and grounded out-of-process — OpenMed produces the dedup'd
  concept and status, not the terminology binding.

## Edge cases & gotchas

- **Surface dedup is lossy.** "MI" and "myocardial infarction" only fold if your
  normalizer knows the synonym. Lexical folding handles the easy cases; lean on
  SNOMED CT grounding for real reconciliation, and never bundle SNOMED — call it
  out-of-process with the user's credentials.
- **Active beats historical for the same concept.** "History of asthma" in PMH
  plus "asthma exacerbation" in A&P is one **active** problem, not two entries.
  Aggregate before assigning status.
- **Don't resurrect resolved problems.** A concept seen only as `HISTORICAL` /
  "resolved" stays resolved; don't promote it to active just because it appears.
- **Negated and hypothetical are exclusions, not statuses.** They never become
  problem-list entries. Keep them out entirely.
- **Carry provenance.** Keep offsets / source sections per problem so a reviewer
  can trace each entry back to the note text.
- **Local-first, advisory-only.** Runs on-device; the reconciled list is decision
  support for clinician review, not an autonomous diagnosis.

## Standards & references

- USCDI v3+ — Problems / Health Concerns data class:
  https://www.healthit.gov/isa/united-states-core-data-interoperability-uscdi
- HL7 FHIR R4 Condition — `clinicalStatus` (active/resolved) and
  `verificationStatus`: https://hl7.org/fhir/R4/condition.html
- SNOMED CT — clinical concept reference terminology (user-supplied license):
  https://www.snomed.org/
