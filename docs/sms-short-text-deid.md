# SMS short-text de-identification

OpenMed's `short_text` preset is for clinical messages up to SMS scale,
including MomConnect-style helpdesk conversations and RapidPro flow/message
exports. It runs locally and combines the normal PII pipeline with deterministic
rules for message formats where sentence context is sparse or absent.

RapidPro represents SMS addresses as contact URNs such as
`tel:+250788123123`, and a logical message can span more than one physical SMS.
The adapter therefore treats the message record—not a sentence—as the unit of
work. See RapidPro's documentation for its
[contact URN](https://rapidpro.github.io/rapidpro/docs/contacts/) and
[message](https://rapidpro.github.io/rapidpro/docs/msgs/) models.

## What the preset changes

The preset is named `short_text` and uses a lower high-recall threshold than
the default clinical-note path. It adds context-independent recognition for:

- Kenyan, Ugandan, and South African E.164-style MSISDNs (`+254`, `+256`,
  `+27`), spaced or punctuated variants, local `07xx` numbers, and 4–6 digit
  shortcodes;
- contextual and common-shape national/medical identifiers;
- names following code-mixed maternal-health honorifics such as `mama`,
  `maama`, `sisi`, `dada`, `mme`, and `ndugu`, including all-caps messages;
- common SMS and clinical abbreviations (`ANC`, `EDD`, `LMP`, `CHW`, `pls`,
  `appt`, and similar) as allowed terms so a sparse context window does not turn
  them into identifiers.

Detection may normalize whitespace internally, but replacement offsets are
mapped to the original string. The preset never truncates, wraps, or otherwise
reflows source text.

## One message

Install the model-backed PII dependencies once, then keep the model cached for
offline use:

```bash
uv pip install -e ".[hf]"
```

```python
from openmed.multimodal import deidentify_short_text

message = "  HABARI MAMA Amina Njeri call +254712345678 ANC  "
result = deidentify_short_text(message)

assert result.original_text == message
print(result.deidentified_text)
# "  HABARI [PERSON] call [PHONE] ANC  "
```

`deidentify_short_text` defaults to mask replacements and accepts the normal
OpenMed model, language, policy, loader, and mapping options. Code-mixed rules
remain language-independent; `lang` is still passed to the model-backed and
standard locale recognizers.

## RapidPro-shaped JSON

The JSON adapter accepts a top-level list, a single message record, or an object
whose record list is under `messages`, `results`, `runs`, `records`, or nested
`data`. Every string field named `text`, including flow-result `input.text`, is
de-identified. The surrounding object, row order, flow UUIDs, directions, and
other non-sensitive fields are retained.

```python
import os

from openmed.multimodal import redact_sms_json

result = redact_sms_json(
    "rapidpro-message-export.json",
    "rapidpro-message-export.redacted.json",
    batch_size=512,
)
print(result.summary.to_dict())
```

Contact `urn`, `urns`, `contact_urn`, and string-valued `contact` fields are
replaced with HMAC-SHA256 pseudonyms. Nested contact `name`, `urn`, and
`address` values are pseudonymized too; contact UUIDs and flow UUIDs remain
intact. Repeated source values receive the same pseudonym within one run.

By default a fresh random HMAC key is generated per run, so outputs from
different runs cannot be linked. Supply a secret key only when an authorized
workflow needs stable cross-run linkage:

```python
result = redact_sms_json(
    "messages.json",
    "messages.redacted.json",
    contact_hash_key=os.environ["OPENMED_SMS_CONTACT_HASH_KEY"],
)
```

Keep that key outside source control and logs. The summary contains only counts,
never contact values or message text.

## Generic CSV and high-volume logs

Generic CSV input must include `text` and at least one of `urn` or `contact`.
The expected RapidPro-compatible core columns are:

```text
urn,contact,direction,text,sent_on
```

Extra columns are preserved in their original order. The CSV path reads and
writes incrementally, while `iter_redacted_sms_records` accepts any iterable of
mapping records. Both retain at most one source batch in memory and reuse
OpenMed's `BatchProcessor` for batched model inference.

```python
from openmed.multimodal import redact_sms_csv

result = redact_sms_csv(
    "helpdesk.csv",
    "helpdesk.redacted.csv",
    batch_size=512,
)
assert result.summary.row_count > 0
```

For database cursors or other record streams:

```python
from openmed.multimodal import iter_redacted_sms_records

for redacted_record in iter_redacted_sms_records(database_cursor, batch_size=512):
    write_record(redacted_record)
```

The outer record batch may be larger than 100 rows; the existing batch utility
automatically uses its validated inference chunk size inside that bound.

## Timestamp handling and verification

ISO-like values in `sent_on`, `received_on`, `delivered_on`, `created_on`,
`modified_on`, and `timestamp` are coarsened to `YYYY-MM-DD`. Invalid or custom
timestamp strings are retained rather than guessed.

Before an export crosses a trust boundary, verify known synthetic or
run-scoped source identifiers with `assert_redacted`:

```python
from openmed.interop import assert_redacted

assert_redacted(redacted_text, replacement_to_original_mapping)
```

Use only synthetic fixtures in tests and benchmarks. Do not log raw input rows,
model entities, or failed records; the adapter's error and summary paths are
designed to report counts and error types without message content.

## Boundaries

This adapter does not call a live RapidPro API, serve webhooks, classify message
intent, or process WhatsApp media. It handles exported text payloads only. JSON
documents are materialized to preserve arbitrary wrapper schemas; for exports
with millions of rows, prefer streaming CSV or `iter_redacted_sms_records`.
