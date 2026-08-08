# Dataset upload privacy guard

`openmed.guard` provides an explicit local wrapper for dataset upload call
sites. It scans only the files selected by the caller, makes no network call,
and returns counts plus stable file identifiers. Matched text, source paths,
and scanner exceptions are not copied into guard reports or guard exceptions.

The built-in scanner is a deterministic safety net for common email, phone,
SSN-like, and Luhn-valid card-number patterns. It is not a compliance
certification or a complete clinical privacy detector. Pass a local scanner for
domain-specific identifiers.

## Block an upload

Block mode leaves the source files unchanged and does not invoke the upload
callable when a finding is present:

```python
from openmed.guard import DatasetUploadBlockedError, DatasetUploadGuard


def upload(files):
    return client.upload_dataset(files)


guarded_upload = DatasetUploadGuard(upload, mode="block_only")

try:
    result = guarded_upload(["training.csv"])
except DatasetUploadBlockedError as error:
    # Counts and hashes are safe for structured logging.
    safe_report = error.report.to_dict()
else:
    upload_response = result.upload_result
```

`"block"` is accepted as a short alias. A blocked call raises
`DatasetUploadBlockedError`; the wrapped upload function is never called.

## Redact to a staging directory

Redaction mode writes new UTF-8 files under the configured staging directory,
uses generated hash-based names, and passes those files to the upload callable.
The source files are not modified:

```python
from openmed.guard import DatasetUploadGuard

guarded_upload = DatasetUploadGuard(
    upload,
    mode="redact_to_staging",
    staging_dir=".openmed-staging",
)
result = guarded_upload(["training.csv"])
```

Use `"redact"` as a short mode alias. Replacements use stable tokens such as
`[OPENMED_REDACTED_EMAIL]`. `result.report.to_dict()` contains only the mode,
allow/deny state, counts, finding categories, and `sha256:` file identifiers;
it does not contain the original or staged path.

## Configure a local scanner

A scanner receives text and returns `DatasetFinding` objects (or mappings with
`start`, `end`, and an optional `label`). Offsets are character offsets, and
the matched surface is never part of a finding:

```python
from openmed.guard import DatasetFinding, DatasetUploadGuard


def scan_local_identifier(text):
    marker = "SYNTHETIC_IDENTIFIER"
    start = text.find(marker)
    if start < 0:
        return ()
    return (DatasetFinding("local_id", start, start + len(marker)),)


guarded_upload = DatasetUploadGuard(
    upload,
    scanner=scan_local_identifier,
    mode="redact_to_staging",
    staging_dir=".openmed-staging",
)
```

Scanner failures, unreadable files, invalid offsets, and overlapping findings
fail closed with generic exceptions. Keep reports and logs to
`result.report.to_dict()` or `error.report.to_dict()` so arbitrary upload-client
responses are not accidentally serialized as audit output.
