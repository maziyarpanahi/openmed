# OpenMed Privacy Filter Stream

Streaming web demo for agent traces. The left pane receives raw trace chunks
immediately; the right pane receives a delayed share-safe copy where sensitive
values are masked or replaced with deterministic Faker surrogates.

The demo is intentionally cache-free and model-free so it opens immediately. It
uses a streaming detector for common trace secrets, user identifiers, clinical
IDs, contact details, and scenario-specific names/addresses, then reuses
OpenMed's `Anonymizer` to generate realistic fake values.

## Run

From the repository root:

```bash
pip install -e ".[service]"
python -m uvicorn examples.privacy_filter_stream.app:app --reload --port 8771
```

Open <http://127.0.0.1:8771>.

## What to look for

- Raw agent/tool trace chunks arrive on the left as soon as they are produced.
- The shareable trace on the right lags by the configured delay.
- Repeated sensitive values map to the same surrogate during a stream.
- Toggle **Mask** for bracketed labels or **Fake** for format-preserving
  replacement values.
