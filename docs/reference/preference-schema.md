# Preference-pair schema adapter

`openmed.traces.schemas.preference` provides a local-first adapter for
preference-training records with three required content fields:
`prompt`, `chosen`, and `rejected`.

```json
{
  "pair_id": "synthetic-pair-01",
  "prompt": "Which follow-up is appropriate for Ada Example?",
  "chosen": "Arrange the local follow-up.",
  "rejected": "Publish the source note.",
  "scores": {"chosen": 0.91, "rejected": 0.09},
  "metadata": {"synthetic": true, "source": "offline-fixture"}
}
```

The adapter copies the record and walks the three branches without changing
pair membership, score fields, IDs, roles, or non-content metadata. Strings in
message-style values such as `[{"role": "assistant", "content": "..."}]` are
redacted through their content fields. A single
`PreferenceRedactionState` is shared across all three branches, so an
identical `(label, sensitive value)` pair gets one deterministic surrogate.

```python
from openmed.traces.schemas.preference import PreferencePairAdapter

adapter = PreferencePairAdapter(seed=17)
redacted = adapter.redact(record)
result = adapter.redact_with_report(record)
assert result.to_mapping() == redacted
```

The default detector is deterministic and local. It covers common emails,
phone numbers, dates, network addresses, common identifiers, and conservative
Latin-script names; it does not load a model or make a network call. For a
richer detector that is already available locally, inject a span detector or a
text redactor:

```python
from openmed.traces.schemas.preference import PreferencePairAdapter, SensitiveSpan

def detector(text):
    start = text.find("SYNTH-42")
    if start < 0:
        return ()
    return (SensitiveSpan(start, start + len("SYNTH-42"), "ID_NUM"),)

adapter = PreferencePairAdapter(span_detector=detector, seed=17)
```

A `text_redactor` may accept either `text` or `text, state`. The latter form
can call `state.redact_spans(...)` to reuse the adapter’s pseudonym state.
Reports contain only schema versions and aggregate counts; source surfaces are
not copied into logs, exceptions, or reports. Metadata is intentionally
preserved as non-content data, so callers should keep metadata itself free of
raw sensitive values.
