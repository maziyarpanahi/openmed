# Streaming trace redaction

OpenMed can redact structured trace records without loading the complete trace
store into memory. The streaming API retains at most the configured record and
byte batch, emits records in their original order, and reports only aggregate
counters.

The default text redactor is dependency-light and does not make network calls.
It recognizes common structured values such as email addresses, phone numbers,
dates, IP addresses, UUIDs, record identifiers, and names following a
patient, subject, or user cue. For higher-recall clinical text detection,
inject a text redactor backed by a model that is already available locally.

## Structured records

    from openmed.traces.streaming import TraceRedactor

    def local_redactor(text: str) -> str:
        # Call a preloaded local redaction pipeline here.
        return text.replace("Synthetic Person", "[NAME]")

    runner = TraceRedactor(
        record_batch_size=256,
        byte_batch_size=2 * 1024 * 1024,
        text_fields=(
            "message",
            "exception.message",
            "attributes.user.email",
            "events.*.attributes.exception.message",
        ),
        text_redactor=local_redactor,
    )

    for redacted_record in runner.iter_records(trace_records):
        write_record(redacted_record)

    report = runner.report

The input may be any iterable of mappings. Dotted paths address nested
mappings, and an asterisk addresses every item in a list or every value in a
mapping. A dotted key is also resolved literally, so an OpenTelemetry
attribute such as user.email is supported.

Both limits are enforced independently. A batch is flushed when either the
record count or estimated UTF-8 byte size reaches its limit. A single record
larger than the byte limit raises TraceRecordTooLargeError rather than
silently exceeding the caller's bound.

## Stable pseudonyms and progress

Use method replace with a caller-controlled HMAC secret when a stable,
value-free pseudonym is needed across batches:

    runner = TraceRedactor(
        record_batch_size=32,
        method="replace",
        hmac_secret=trace_linkage_secret,
    )

The default replacement is derived directly from the secret, method seed,
label, and source value. The source value is not put in the report, progress
snapshot, exception text, or pseudonym context representation. Do not use a
secret that is committed to source control for production linkage.

Progress callbacks receive TraceProgress objects containing only integer
counters and a cancellation flag. TraceRedactionReport.to_dict() has the same
value-free shape:

    def observe(progress):
        metrics.record(
            records=progress.records_emitted,
            bytes_written=progress.bytes_emitted,
        )

    runner = TraceRedactor(on_progress=observe)

Cancellation is cooperative and checked at batch boundaries. A batch is
either fully emitted or not started. Pass a CancellationToken and call its
cancel method from a progress callback or another owner:

    token = CancellationToken()
    runner = TraceRedactor(
        record_batch_size=128,
        cancellation=token,
        on_progress=lambda progress: progress.records_emitted >= 10_000,
    )

The final report marks cancelled as true and contains only the counters for
records that were observed or emitted.

## NDJSON

For line-oriented trace stores, redact_ndjson_stream reads and writes one JSON
object at a time:

    from openmed.traces.streaming import redact_ndjson_stream

    report = redact_ndjson_stream(
        input_stream,
        output_stream,
        record_batch_size=128,
        byte_batch_size=1024 * 1024,
        text_fields=("message", "attributes.exception.message"),
    )

Malformed input and redactor failures raise TraceRedactionError with
value-free diagnostics. Configure text fields deliberately, keep raw input
out of application logs, and use synthetic fixtures for tests.
