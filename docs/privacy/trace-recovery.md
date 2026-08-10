# Transactional trace redaction recovery

`openmed.multimodal.trace_recovery` provides a local, crash-safe boundary for
redacting a UTF-8 trace file in place. It is useful when a process may stop
after producing redacted output but before the destination replacement has
completed.

```python
from openmed.multimodal import redact_trace_file


def redact(text: str) -> str:
    return text.replace("SYNTHETIC-PATIENT-771", "[PATIENT]")


result = redact_trace_file("trace.jsonl", redact)
```

The operation computes the input and output SHA-256 fingerprints in memory,
writes the output to a same-directory staging artifact, and atomically replaces
the source only after checking both fingerprints. The source is never copied
into the journal. The default sidecar journal name is derived from a
fingerprint of the target path, so it does not disclose the target filename.
Pass `journal_path` to place the journal at a controlled local path.

## Recovery

The journal is atomically written JSON with a bounded size and at most three
recovery attempts by default. It records only:

- a target identity fingerprint;
- input, output, and staging fingerprints;
- the transaction phase (`prepared`, `staged`, `committing`, or terminal);
- the recovery decision and attempt count.

If an interruption leaves a verified staging artifact, recovery can resume it
without calling the redactor:

```python
from openmed.multimodal import recover_trace_redaction

recover_trace_redaction("trace.jsonl", decision="resume")
```

To preserve the original file and remove the transaction's owned staging
artifact instead:

```python
recover_trace_redaction("trace.jsonl", decision="rollback")
```

`redact_trace_file` automatically resumes a pending transaction with the
provided redactor and returns a completed transaction without invoking that
redactor again. Repeated recovery is therefore idempotent. If the target or
staging fingerprint does not match the journal, recovery fails closed without
replacing the target or touching an unknown artifact. A target mismatch marks
the journal blocked; an operator can explicitly roll back a regular partial
staging file owned by the journal before retrying.

`TraceRecoveryError` messages contain only stable reason codes. Audit reports
from `TraceRedactionResult.to_audit_report()` and
`TraceRecoveryJournal.to_audit_report()` contain fingerprints and bounded
metadata, never source text, redacted text, paths, or exception details.

This is a crash-recovery primitive, not a compliance certification or a
clinical decision guarantee. Keep the supplied redactor local and ensure that
its output is safe for the trace's intended audience.
