# Self-hosted redaction service

`openmed.service.redaction_app` provides a small local FastAPI surface for
redacting text and explicit UTF-8 files. The default redactor is deterministic
and uses only local standard-library patterns; importing or starting the app
does not download a model or make a network request.

The review page at `/review` displays policy, aggregate entity counts, and
artifact status only. It never renders source text, redacted text, file paths,
or entity surfaces. The service also retains only that aggregate state in
memory. Errors use content-free messages, and the file endpoint writes the
redacted artifact to the caller-selected output path.

## Start locally

Install the service extra and run it on loopback:

```bash
uv pip install -e ".[service]"
uvicorn openmed.service.redaction_app:app --host 127.0.0.1 --port 8080
```

The default baseline recognizes common email, phone, date, IP address, SSN,
name-after-a-label, and record-ID shapes. It is intentionally conservative and
does not replace a clinical privacy review or a broader locally hosted model.

## Redact text

Text can be returned directly, or written to an explicit output location. The
example uses synthetic values only:

```bash
curl -sS http://127.0.0.1:8080/redact/text \
  -H 'content-type: application/json' \
  -d '{
    "text": "Patient Synthetic Person, email synthetic@example.test.",
    "policy": "strict_no_leak"
  }'
```

The response contains the redacted text and aggregate counts. When
`output_path` is supplied, the same artifact is written there and the response
reports a content hash and `written` status.

## Redact an explicit file

The file endpoint accepts existing UTF-8 input and a separate output path. The
output directory is created when needed:

```bash
curl -sS http://127.0.0.1:8080/redact/file \
  -H 'content-type: application/json' \
  -d '{
    "input_path": "/work/synthetic-note.txt",
    "output_path": "/work/out/redacted-note.txt",
    "policy": "strict_no_leak"
  }'
```

The file response does not echo either file path or source content. The caller
already knows the explicit output location and can verify the returned SHA-256
hash. Input and output paths must differ.

## Use an already-provisioned local model

For broader detection, inject a model that is already available on disk. The
service does not provision or download that model:

```python
from openmed.service.redaction_app import create_app

app = create_app(local_model_path="/models/pii-model")
```

Alternatively, pass a callable with `redactor=`. It may return a core
de-identification result, a `RedactionResult`, a mapping containing
`redacted_text`, or a string. Only redacted text and aggregate labels are
retained by the service adapter.

This service is a local workflow component, not a compliance certification or a
clinical decision system. Keep source artifacts in controlled storage and use
synthetic data for development and tests.
