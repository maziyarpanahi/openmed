# No-Raw-PHI Logging Policy

OpenMed code must not write raw protected health information (PHI), patient
text, source documents, or extracted cleartext spans to logs at any level.
Logs are operational telemetry only.

## Allowed log content

- Counts, durations, thresholds, model identifiers, backend names, and status
  transitions.
- Span metadata such as label, start offset, end offset, confidence bucket, and
  validity flags.
- A precomputed keyed `text_hash` when one already exists for the span or
  document. Do not create ad hoc hashes in log statements.
- Exception class names and high-level failure categories.

## Disallowed log content

- Input document text, cleaned text, truncated text snippets, prompts, sentences,
  or source surfaces that may contain clinical content.
- Entity text, original PII values, redacted-to-original mappings, or replacement
  mapping payloads.
- Exception messages that may include user input, source paths, request bodies,
  or downstream library payloads.
- File paths or item identifiers derived from patient, chart, or encounter data.

## Engineering requirements

- Prefer structured fields over formatted prose when adding logs.
- Use lengths, counts, labels, offsets, and safe identifiers instead of text.
- Keep request and response bodies out of service logs.
- When adding or changing a PII, de-identification, text-processing, or batch
  path, run the no-raw-PHI logging guard:

```bash
.venv/bin/python -m pytest tests/unit/test_no_raw_text_logging.py -q
```

The full suite must also pass before release or pull request review:

```bash
.venv/bin/python -m pytest tests/ -q
```
