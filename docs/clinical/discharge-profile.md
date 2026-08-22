# Structured discharge-summary profile

\`openmed.clinical.discharge_profile\` exposes a deterministic, section-scoped
view of a discharge summary. It extracts only evidence written in the source
document:

- diagnoses;
- procedures;
- discharge medications and locally parsed sig attributes;
- follow-up entries; and
- separately stated discharge instructions.

Each item keeps its half-open source span and section header. The profile also
attaches the existing local ConText axes (\`certainty\`, \`negation\`, and
\`temporality\`). An uncertain or hypothetical statement is retained and marked;
it is never silently promoted to a confirmed clinical fact.

\`\`\`python
from openmed.clinical import extract_discharge_profile

note = (
    "Discharge Diagnoses:\n"
    "- possible synthetic infection\n"
    "Discharge Medications:\n"
    "- SyntheticDrug 10 mg PO daily\n"
    "Follow-Up:\n"
    "- Primary care in 7 days.\n"
    "Instructions:\n"
    "- Return if fever.\n"
)

profile = extract_discharge_profile(note)

assert profile.diagnoses[0].certainty == "uncertain"
assert profile.medications[0].sig["frequency_per_day"] == 1.0
assert note[
    profile.diagnoses[0].start : profile.diagnoses[0].end
] == profile.diagnoses[0].text
\`\`\`

\`DischargeSummaryProfile.to_dict()\` and \`.to_json()\` provide stable,
JSON-compatible output. The report contains the extracted evidence and its
offsets; callers should keep the source document in their own protected
store. The implementation does not log source text, make mandatory network
requests, call a terminology service, or create recommendations. It is
assistive review tooling and is not a diagnosis, medication-reconciliation
decision, compliance certification, or medical device.

Caller-provided spans may include an \`assertion\` mapping. Its valid assertion
axes take precedence over local cue classification:

\`\`\`python
profile = extract_discharge_profile(
    note,
    spans=[
        {
            "label": "CONDITION",
            "start": note.index("possible synthetic infection"),
            "end": note.index("possible synthetic infection")
            + len("possible synthetic infection"),
            "assertion": {
                "certainty": "uncertain",
                "negation": "affirmed",
                "temporality": "recent",
            },
        }
    ],
)
\`\`\`

Use synthetic notes in tests and keep protected source text out of logs,
exceptions, committed fixtures, and generated reports that are not explicitly
intended to carry evidence text.
