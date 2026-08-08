# Session-end trace hook

The session-end hook scrubs one completed local trace after the host has
finished writing it. It reads only the explicit path supplied by the host; it
does not discover trace stores, load a model, or make a network request.

The hook accepts a JSON document or a JSONL/NDJSON file and replaces the input
only after redaction and structural validation succeed. Replacement is atomic,
so a malformed or concurrently changed trace remains untouched.

## Run the hook

Pass the completed trace path positionally or with `--trace`:

```bash
python -m openmed.guard.session_hook /path/to/completed-trace.json
python -m openmed.guard.session_hook --trace /path/to/completed-trace.ndjson
```

Success is quiet and returns exit code `0`. Failures return exit code `1` and
write only a stable category such as `invalid_trace` or `write_failed` to
standard error. No input path, trace field, exception detail, or source value
is included in the default output.

For a machine-readable value-free summary, opt into `--json`:

```bash
python -m openmed.guard.session_hook --json /path/to/completed-trace.json
```

The summary contains the format, redaction count, byte counts, and SHA-256
digests of the bytes before and after replacement. It never contains source
values or the input path.

## What is scrubbed

The hook uses deterministic local rules. Sensitive fields such as email,
phone, name, address, patient or member identifiers, dates of birth, tokens,
and message or prompt content are replaced with category-only placeholders.
Structured values such as email addresses, phone numbers, IP addresses,
credentials, bearer tokens, common identifiers, and labeled names or birth
dates are also scrubbed when they appear in otherwise neutral fields.

Trace metadata such as the shape of objects, list lengths, numeric status
fields, and opaque trace identifiers is retained when it is not identified as
sensitive. The rules are a local privacy guard, not a compliance certification;
hosts should avoid placing raw sensitive content in traces in the first place.

## Host contract

Invoke the hook once the host has closed the completed trace file. The host
must provide exactly one path and should treat any non-zero exit code as a
failed cleanup. A host can use the quiet exit status, or request `--json` for
counts-only evidence. The hook does not remove or inspect unrelated files.

Example synthetic trace:

```json
{
  "trace_id": "trace-0001",
  "attributes": {
    "patient_email": "synthetic.person@example.test",
    "message": "Synthetic Person called +1 555 010 2020"
  }
}
```

After a successful run, the sensitive values are replaced with placeholders
while the JSON object shape remains valid. Use only synthetic data in local
tests and fixtures.
