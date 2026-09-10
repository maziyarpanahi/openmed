# Email Ingestion and PHI Redaction

OpenMed can parse RFC 5322/MIME `.eml` messages, map decoded header and body
text back to source segments, and emit a clean EML with PHI removed from
headers, plain text, HTML, attachment names, and supported attachment content.
EML parsing uses the Python standard library and is available in the base
installation.

## Extract normalized text

```python
from openmed.multimodal import extract_email

message = extract_email("synthetic-message.eml")
print(len(message.text), message.metadata["body_part_count"])

for span in message.spans:
    print(span.start, span.end, span.metadata["block_type"])
```

The normalized document includes decoded `From`, `To`, `Cc`, `Bcc`,
`Reply-To`, `Sender`, `Subject`, routing/reference, and date headers plus
`text/plain` bodies and visible `text/html` content. HTML spans include local
source offsets. Scripts, styles, and templates are excluded from visible-text
extraction.

## Emit a redacted EML

Install the multimodal extra when messages can contain PDF, DOCX, PPTX, or
raster-image attachments:

```bash
uv pip install "openmed[multimodal]"
```

Then supply the same local OpenMed PII detector used by your text workflow:

```python
from openmed import extract_pii
from openmed.multimodal import redact_email

result = redact_email(
    "synthetic-message.eml",
    output_path="synthetic-message.clean.eml",
    models={"detector": extract_pii},
    lang="en",
)

print(result.header_redaction_count)
print(result.body_redaction_count)
print([attachment.to_dict() for attachment in result.attachments])
```

The supplied detector is followed by OpenMed's deterministic safety sweep.
HTML tags are retained while visible text, comments, and safe attributes are
redacted; active script/style/template content and event/style attributes are
removed. Remote image sources, unrecognized Content-ID references, and unsafe
link schemes are also removed. MIME boundaries, filenames, content IDs,
content locations, and authentication signatures are removed or regenerated.
Attachment reports contain only indexes, normalized types, counts, and hashes.

Attachments are dispatched in memory through `redact_document`. Raw message
or attachment bytes are never written to temporary files. PDF attachments are
emitted as fresh image-only PDFs with opaque boxes burned into page pixels, so
the original searchable text layer and source metadata cannot survive.
Unsupported attachments fail closed instead of being copied to the clean
message.

The generic dispatcher also recognizes email extensions:

```python
from openmed.multimodal import redact_document

document = redact_document("synthetic-message.eml")
```

## Optional Outlook MSG bridge

Outlook `.msg` parsing is opt-in because `extract-msg` is GPL-3.0. Install the
explicit bridge extra only when it is acceptable for your deployment:

```bash
uv pip install "openmed[email-msg-gpl,multimodal]"
```

OpenMed does not import `extract-msg` into its process. It invokes the parser in
an isolated Python subprocess, sends MSG bytes over standard input, receives a
normalized EML over standard output, and creates no PHI-bearing temporary
files. MSG input is always emitted as EML because the optional parser is
read-only. Without the extra, `.msg` input raises an actionable
`MissingDependencyError`.

All processing remains local. OpenMed performs no telemetry or network calls;
model artifacts must already be available when running offline.
