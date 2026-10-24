# Synthetic patient-intake conversations

`generate_intake_conversations` writes redistributable intake dialogues for
conversational extraction and ambient-scribe evaluation. Each conversation has
several speakers, caregiver or interpreter speech, hesitations, self-repairs,
later corrections, explicit uncertainty and recording-consent events. Gold
labels are attached per turn so a streaming evaluator can score turn `t` before
it reads turn `t + 1`.

```python
from openmed.training.synthetic.intake_dialogue import (
    build_intake_dialogue_manifest,
    generate_intake_conversations,
)

conversations = generate_intake_conversations(6, seed=7)
for record in conversations[0].iter_stream_records():
    gold = record["gold"]
    print(record["turn_index"], record["speaker_role"], len(gold["phi_spans"]))

manifest = build_intake_dialogue_manifest(conversations)
```

The generator uses repository-authored templates and fictional value pools.
Phone numbers come from the reserved 555-01XX range, e-mail addresses use the
reserved `example.org` domain, and ages are computed against a fixed reference
date. It reads no dataset, makes no network call, and returns identical text,
offsets and labels for the same seed, conversation index and scenario.

## Conversation content

Conversation `i` uses `scenarios[i % len(scenarios)]`, which defaults to all
three scenarios:

| Scenario | Speakers | Characteristic turns |
| --- | --- | --- |
| `patient_only` | patient, clinician | patient self-repairs a birth date, then corrects an earlier phone number |
| `caregiver` | patient, caregiver, clinician | caregiver gives an uncertain birth date the patient corrects, gives the caregiver's own phone and e-mail, corrects the medication, and reports a symptom the patient denies |
| `interpreter` | patient, interpreter, clinician | interpreter speaks for the patient in the first person, makes an "interpreter correction", gives the interpreter's own name and ID, and corrects an earlier phone number |

Every conversation contains a consent request and response, at least one
hesitation, an explicit uncertainty cue, a within-turn self-repair and a later
turn that corrects an earlier fact. Other phenomena rotate by conversation
index: address answers are interrupted and completed on even indices, consent
is declined when `index % 5 == 3`, and on odd indices the clinician mishears
the birth date in a read-back and, unless consent was declined, the patient
pauses the recording to share a private number.

## Turn records

`IntakeConversation.iter_stream_records()` yields one JSON-compatible record
per turn. Inputs sit at the top level and every label sits under `gold`. In
this abridged patient turn, the speaker gives a birth date and then repairs it
after "no, sorry"; the other spans and the utterance are elided:

```json
{
  "schema_version": "openmed.training.synthetic_intake_dialogue.v1",
  "record_type": "intake_turn",
  "conversation_id": "synthetic-intake-7-0000",
  "turn_index": 4,
  "speaker_role": "patient",
  "content": "<synthetic utterance>",
  "stream_offset": 269,
  "gold": {
    "act": "identity_answer",
    "consent_state": "granted",
    "capture_permitted": true,
    "phi_spans": [
      {
        "span_id": "t04.s02",
        "start": 63,
        "end": 78,
        "stream_start": 332,
        "stream_end": 347,
        "label": "DATE_OF_BIRTH",
        "kind": "phi",
        "subject_role": "patient",
        "attribute": "date_of_birth",
        "slot": "patient.date_of_birth",
        "assertion": "affirmed",
        "revision": "self_repair",
        "supersedes": "t04.s01",
        "partial": false,
        "needs_adjudication": false,
        "conflicts_with": [],
        "text_sha256": "<SHA-256 of the span text>"
      }
    ],
    "clinical_spans": [],
    "events": [
      {
        "event_id": "t04.e00",
        "type": "self_repair",
        "start": 52,
        "end": 61,
        "stream_start": 321,
        "stream_end": 330,
        "attributes": {"repair": "t04.s02", "reparandum": "t04.s01"}
      }
    ],
    "fact_updates": {
      "patient.date_of_birth": "t04.s02",
      "patient.name": "t04.s00"
    }
  },
  "provenance": {"synthetic": true, "contains_real_phi": false, "seed": 7}
}
```

The `content` field follows the one-turn-per-line convention of
`openmed.multimodal.chatlog_jsonl`, so the JSONL chat-log redactor can process
these records and leave `gold` unchanged.

Each span carries:

- `start` and `end`: half-open offsets into the turn's `content`.
- `stream_start` and `stream_end`: offsets into `IntakeConversation.stream_text`,
  which is every turn followed by a newline (`iter_stream_chunks()`). Feeding
  those chunks in order to an incremental de-identifier such as
  `openmed.core.streaming.StreamingDeidentifier` produces spans in the same
  coordinates.
- `label`: a canonical label. PHI spans use `PERSON`, `DATE_OF_BIRTH`, `AGE`,
  `PHONE`, `EMAIL`, `STREET_ADDRESS`, `LOCATION`, `ID_NUM` and `DATE`;
  clinical intake facts use `CONDITION` and `MEDICATION`.
- `speaker_role` (on the turn) and `subject_role` (on the span): who spoke and
  whose fact it is. A caregiver can state the patient's birth date, and an
  interpreter's "my phone number" belongs to the patient.
- `slot`: `"<subject_role>.<attribute>"`, for example `patient.phone` or
  `caregiver.phone`.
- `assertion`: `affirmed`, `negated` or `uncertain`, the same strings used by
  `openmed.clinical.context`.
- `revision` and `supersedes`: `self_repair`, `correction` or `completion`,
  with the identifier of the earlier span this one replaces.
- `partial`: the speaker was cut off before finishing the value.
- `needs_adjudication` and `conflicts_with`: another speaker stated the
  same fact differently. The disagreement is flagged for review, never
  resolved by the generator.
- `text_sha256`: a digest that binds the span to its characters. Gold labels
  never repeat surface text.

Events (`hesitation`, `uncertainty`, `self_repair`, `correction`,
`interruption`, `consent`) are anchored to their cue words. Consent events
carry `action` (`request`, `grant`, `decline`, `pause`, `resume`), `scope` and
`subject_role`.

## Causal gold

Labels in turn `t` only refer to turn `t` or earlier turns. `supersedes`,
`conflicts_with` and event links always point backwards, and the constructors
reject records that point forwards. Applying each turn's `fact_updates` in
order reproduces `fact_timeline()`, the fact state after every turn. The
end-of-conversation views are `final_facts()`, `supersessions()` and
`adjudication_items()`, which `to_dict()` also exports.

`consent_state` is the state in effect when a turn starts. It is `pending`
until consent is given, then `granted`, `declined` or `paused`; a consent event
changes the state from the next turn on. `capture_permitted` is true only in
the `granted` state. An identifier spoken before consent, after a decline,
while recording is paused, or later superseded is still PHI: every PHI span
remains a redaction target whatever its assertion, revision or consent state.

`gold_openmed_spans(secret=...)` returns the PHI gold as `OpenMedSpan` records
in stream coordinates, with keyed `hmac-sha256` text hashes instead of text.

## Lineage and privacy

Each record's `provenance` states `synthetic: true`, `contains_real_phi:
false`, `restricted_data: false`, the source `openmed.synthetic.intake_dialogue`,
the `Apache-2.0` license, the seed, the conversation index and the scenario.
`IntakeConversation.report()` and `build_intake_dialogue_manifest()` contain
only counts, schema versions, seeds and SHA-256 digests. Validation errors
describe the broken rule without echoing any value.

## Limits

The templates are short, English-only scripted dialogue acts. They are not
transcribed speech, do not model recognition errors, and are not reviewed by
clinicians, interpreters or linguists. Interpreted turns render the patient in
English only. Symptoms,
allergies and medications are fictional software stimuli, not clinical
guidance. Using these conversations does not certify de-identification
quality, consent compliance or any clinical decision.
