---
name: segmenting-clinical-sections
description: "Split a clinical note into canonical sections (Chief Complaint, HPI, PMH, Medications, Allergies, Assessment & Plan, etc.) before running OpenMed NER or de-identification, so section context sharpens downstream precision. Use when the user has a free-text note or discharge summary and wants section-aware processing, header detection, mapping headers to LOINC document-section codes, or per-section NER/de-id. Covers heuristic header detection, normalization to canonical section labels, LOINC/SecTag framing, and why a finding in PMH is historical while the same finding in A&P is active. Hand-off: feed each sectioned chunk into openmed.analyze_text / openmed.deidentify. Pairs before extracting-clinical-entities."
license: Apache-2.0
metadata:
  project: OpenMed
  category: clinical-nlp
  pairs: before
  version: "1.0"
---

# Segmenting clinical sections

A clinical note is not flat text — it is a sequence of named sections (Chief
Complaint, HPI, Past Medical History, Medications, Allergies, Assessment & Plan).
The **same phrase means different things in different sections**: "diabetes" in
PMH is historical context, "diabetes" in Assessment & Plan is an active problem,
and "penicillin" under Allergies is an adverse-reaction flag, not a current
medication. Splitting the note into canonical sections **before** NER or
de-identification gives every downstream OpenMed step the context it needs to be
more precise — and lets you process sensitive sections under stricter policies.

## When to use

- You have a free-text note, H&P, progress note, or discharge summary and are
  about to run NER (`extracting-clinical-entities`) or de-identification.
- The user wants section detection, header parsing, LOINC section mapping, or
  per-section processing (e.g. "redact the Social History section harder").
- Downstream NER is over- or under-firing because it can't tell historical PMH
  mentions from active A&P problems.

## Quick start

```python
import re
import openmed

# Synthetic note.
note = """CHIEF COMPLAINT: chest pain.
HPI: 54M with 2 hours of substernal pressure.
PAST MEDICAL HISTORY: type 2 diabetes, prior MI 2019.
MEDICATIONS: metformin 500 mg BID.
ALLERGIES: penicillin (rash).
ASSESSMENT AND PLAN: acute coronary syndrome; start aspirin, admit."""

# Map common header variants -> canonical section + LOINC document-section code.
SECTION_MAP = {
    "chief complaint": ("Chief Complaint", "10154-3"),
    "hpi": ("History of Present Illness", "10164-2"),
    "history of present illness": ("History of Present Illness", "10164-2"),
    "past medical history": ("Past Medical History", "11348-0"),
    "medications": ("Medications", "10160-0"),
    "allergies": ("Allergies", "48765-2"),
    "assessment and plan": ("Assessment and Plan", "51847-2"),
}

HEADER_RE = re.compile(r"^(?P<h>[A-Z][A-Za-z /&]+):", re.MULTILINE)

# Split note into (canonical_label, loinc, body) chunks at each header.
chunks, matches = [], list(HEADER_RE.finditer(note))
for i, m in enumerate(matches):
    raw = m.group("h").strip().lower()
    label, loinc = SECTION_MAP.get(raw, (m.group("h").strip(), None))
    body_start = m.end()
    body_end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
    chunks.append({"section": label, "loinc": loinc,
                   "text": note[body_start:body_end].strip()})

# Run NER per section — pass the section label downstream as context.
for c in chunks:
    ents = openmed.analyze_text(c["text"], model_name="disease_detection_superclinical",
                                output_format="dict")
    c["entities"] = ents
```

Each chunk now carries its canonical section label and LOINC code, so downstream
context resolution can treat PMH findings as historical and A&P findings as
active.

## Workflow

1. **Detect section headers.** Use header heuristics: a line that is a known
   header phrase, often uppercase, ending in a colon, at line start. Maintain a
   synonym map (HPI ↔ History of Present Illness, PMH ↔ Past Medical History,
   A&P ↔ Assessment and Plan) so variants normalize to one canonical label.
2. **Normalize to canonical labels and LOINC codes.** Map each detected header to
   a canonical section name and a LOINC document-section code (e.g. HPI →
   `10164-2`, PMH → `11348-0`, Medications → `10160-0`, Allergies → `48765-2`,
   A&P → `51847-2`). Unknown headers keep their literal text and a null code.
3. **Chunk the note** into `(section, loinc, body)` spans between consecutive
   headers, preserving original character offsets if you need to map results back.
4. **Process per section.** Run `analyze_text` / `deidentify` on each chunk and
   carry the section label forward. This is where precision is won: section-aware
   negation (PMH = historical) and section-specific de-id policy
   (Social History / Family History often warrant stricter redaction).
5. **Reassemble with provenance.** Tag each downstream entity with its source
   section so the problem-list and context layers can use it.

## Hand-off to / from OpenMed

- **To** `extracting-clinical-entities`: feed each section chunk into
  `openmed.analyze_text` and attach the section label to every entity — section
  context measurably sharpens entity precision and downstream status assignment.
- **To** `deidentifying-clinical-text`: run `openmed.deidentify` per section so
  high-risk sections (Social/Family History) can use a stricter policy profile
  than the body.
- **To** `resolving-clinical-context`: the section label is a strong prior — PMH
  biases temporality toward historical, A&P toward recent/active. Pass it as part
  of the modifier window.
- **To** `reconciling-problem-lists`: section provenance (PMH vs. A&P) is a key
  signal for active-vs-resolved reconciliation.

## Edge cases & gotchas

- **Header variants are endless.** "PMHx," "Past Med Hx," "PMH/PSH," inline
  headers without a colon, and run-on notes all appear. Keep the synonym map
  data-driven and fall back gracefully to the literal header for unknowns.
- **Don't drop unsectioned text.** Notes often start with un-headed preamble or
  have free text between sections. Capture it as an "unknown/other" chunk rather
  than discarding it, or you lose entities.
- **LOINC is a binding, not a parser.** LOINC document-section codes label the
  section; they do not detect it. Mapping is your responsibility and is
  user-supplied terminology — do not bundle LOINC content; reference codes only.
- **Preserve offsets** if you will re-merge entities into the original note for
  de-id; chunking loses position unless you track it.
- **Local-first.** All segmentation and per-section processing runs on-device.

## Standards & references

- LOINC document-section codes (clinical document ontology):
  https://loinc.org/
- Denny et al., *SecTag* — a clinical note section tagger:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2655775/
- HL7 C-CDA section templates (canonical clinical document sections):
  https://www.hl7.org/ccdasearch/
