# Secret-detector negative controls

`negative_controls.jsonl` holds synthetic high-entropy strings that the secret
detector must not report. One JSON object per line:

```json
{"kind": "uuid", "text": "00000000-0000-0000-0000-000000000000"}
```

## Rules for adding a control

- Use synthetic values only. Never paste a real credential, token, production
  hostname, or personal data.
- Every `kind` needs at least two lines, so a single accidental match cannot
  hide a regression in that shape.
- Add the line, then run the detector's test:

  ```sh
  .venv/bin/python -m pytest tests/unit/core/test_secrets.py -q
  ```

  On Windows use `.venv\Scripts\python.exe -m pytest ...` because the repo's
  POSIX venv layout and `make` targets are unavailable.

- If a new control is flagged, report the kind and line number in the PR or
  issue instead of weakening the detector or deleting the control. The test
  reports only the kind and the 1-based line number, never the control text.

## Covered kinds

`uuid`, `git_commit_sha`, `sha256_fingerprint`, `oci_digest`, `sri_integrity`,
`base64_image_fragment`, `base64_document_fragment`, `phi_placeholder`,
`template_placeholder`, `public_key_block`, `certificate_block`, `ulid`,
`nanoid`, and `etag`.
