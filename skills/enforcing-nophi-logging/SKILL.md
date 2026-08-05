---
name: enforcing-nophi-logging
description: "Add a logging and telemetry guard that scrubs or blocks PHI from logs, traces, and error reports around an OpenMed deployment. Use when the user wants a Python logging.Filter that redacts protected health information before records are emitted, wants to keep PHI out of OpenTelemetry spans or error trackers, needs structured no-PHI log fields, or is worried that logs and stack traces are leaking patient data. Trigger on \"scrub logs\", \"redact PHI from logs\", \"no-PHI logging\", \"logging filter\", \"telemetry redaction\", \"logs leaking patient data\", or \"OpenTelemetry redaction\" in an OpenMed deployment."
license: Apache-2.0
metadata:
  project: OpenMed
  category: deployment-ops
  pairs: adjacent
  version: "1.0"
---

# Enforcing No-PHI Logging

Logs are a top breach vector: a clinical string lands in a log line, gets shipped
to a centralized log store and an error tracker, and is now PHI sitting outside
the de-id boundary. OpenMed's local-first stance says *no raw PHI in logs, caches,
or error reports* — this skill enforces it with a redaction guard that runs
**before** any record is emitted.

## When to use this skill

- An OpenMed service logs request text, model output, or exception messages.
- You ship logs/traces to a centralized store or error tracker (Sentry, ELK).
- You need a `logging.Filter` (or OTel processor) that redacts PHI pre-emit.
- You want structured, no-PHI log fields (offsets, hashes, counts) for debugging.

## Quick start — a redacting logging.Filter

```python
import logging
import re
import openmed

# Cheap regex pre-filter for the highest-risk structured identifiers. This runs
# on every record, so keep it fast; the model is the fallback for free-text PHI.
_FAST_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d{16}\b"), "[CARD]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?\d[\d().\-\s]{7,}\d)\b"), "[PHONE]"),
]

class NoPHIFilter(logging.Filter):
    """Redact PHI from a log record before it is emitted. Fail closed."""

    def __init__(self, model_name: str | None = None, use_model: bool = True):
        super().__init__()
        self.model_name = model_name
        self.use_model = use_model

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            record.msg = self._scrub(message)
            record.args = ()                 # message already rendered & scrubbed
        except Exception:
            # Never let the logger leak on error — drop the message, keep the level.
            record.msg = "[REDACTED: scrub error]"
            record.args = ()
        return True                          # keep the (now-clean) record

    def _scrub(self, text: str) -> str:
        for pattern, tag in _FAST_PATTERNS:
            text = pattern.sub(tag, text)
        if not self.use_model:
            return text
        # Model fallback for free-text PHI (names, locations, dates). Replace by
        # offset, right-to-left, so earlier offsets stay valid.
        spans = openmed.extract_pii(text, model_name=self.model_name) \
            if self.model_name else openmed.extract_pii(text)
        for e in sorted(spans.entities, key=lambda s: s.start, reverse=True):
            text = text[:e.start] + f"[{e.label}]" + text[e.end:]
        return text

# Attach to every handler that might emit clinical text.
handler = logging.StreamHandler()
handler.addFilter(NoPHIFilter(model_name="OpenMed/Privacy-PII-Detection"))
logging.getLogger("openmed.service").addHandler(handler)
```

## Prefer structured, no-PHI fields

Don't log the note and scrub it — log *about* it without the text in the first
place:

```python
logger.info(
    "deidentified note",
    extra={
        "doc_id": doc_id,                       # opaque id, not the text
        "phi_entity_count": len(result.entities),
        "phi_labels": sorted({e.label for e in result.entities}),
        "char_len": len(text),
        # offsets/hashes for debugging; never the plaintext span
        "phi_offsets": [(e.start, e.end) for e in result.entities],
    },
)
```

Redaction is the safety net; not logging PHI is the actual fix.

## OpenTelemetry / error trackers

- **Spans:** add a `SpanProcessor.on_end` (or attribute hook) that runs the same
  `_scrub` over string span attributes and events before export.
- **Error trackers:** register a `before_send` hook (e.g. Sentry) that scrubs
  exception messages, breadcrumbs, and request bodies. Stack traces often embed
  the offending input — scrub the message, not just the frames.

## Workflow

1. **Inventory sinks.** List every place a clinical string can reach: app logs,
   access logs, OTel spans, error tracker, crash reports, request/response dumps.
2. **Install the regex pre-filter** for structured identifiers (SSN, card, email,
   phone) — fast, runs on every record.
3. **Add the model fallback** (`openmed.extract_pii`) for free-text PHI on the
   sinks that carry clinical narrative; skip it on hot paths where regex suffices.
4. **Switch to structured fields.** Replace "log the text" with "log counts,
   labels, offsets, ids".
5. **Fail closed.** On any scrub error, drop the message content, not the
   redaction.
6. **Test it.** Unit-test that known PHI strings never survive a round trip
   through the filter, including in exception messages.

## Hand-off to / from OpenMed

- **Uses** `openmed.extract_pii` (and optionally the regex pre-filter) as the PHI
  detector — the same engine documented in `extracting-pii-entities`.
- **From** `building-with-openmed`: this is the runtime guard for the local-first,
  no-PHI-in-artifacts rule.
- **Pairs with** `gating-deid-leakage`: the gate proves the *model* doesn't leak;
  this guard proves your *logs and traces* don't leak.
- **To** `auditing-deidentification-runs`: route audit output through the same
  no-PHI discipline (offsets/hashes, never plaintext).

## Edge cases & gotchas

- **`record.args` must be cleared after scrubbing.** If you rewrite `record.msg`
  but leave `%s` args, the formatter re-injects raw PHI downstream.
- **Scrub before fan-out.** Filters on one handler don't protect others — attach
  to every handler, or scrub at the record/formatter layer.
- **Latency budget.** The model fallback costs inference per record; gate it
  behind a level threshold or reserve it for narrative-bearing sinks.
- **Regex alone is not de-id.** It catches structured identifiers; names,
  locations, and dates need the model. Use both, model last.
- **Exception messages are PHI carriers.** `f"failed on {note}"` leaks; scrub
  exception text and error-tracker payloads, not just `logger.info` calls.
- **Fail closed, never open.** A scrub error must redact the content, never emit
  the unscrubbed original.
- **No raw PHI even in DEBUG.** "It's only debug logs" is how breaches happen;
  the guard applies at every level.

## Standards & references

- HIPAA Security Rule — audit controls & log protection (45 CFR 164.312):
  https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/
- OWASP Logging Cheat Sheet (what not to log):
  https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
- OpenTelemetry span processors (attribute redaction on export):
  https://opentelemetry.io/docs/specs/otel/trace/sdk/#span-processor
- Python `logging.Filter` API:
  https://docs.python.org/3/library/logging.html#filter-objects
- OpenMed PHI detector: `openmed.extract_pii` (`openmed/core/pii.py`).
