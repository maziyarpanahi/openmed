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

## How the current values were generated

Every value is a pure function of a short label, so it can be rebuilt or
extended without inventing new entropy, and a diff stays readable because only
the label changes.

- UUIDs are UUIDv5 in the URL namespace over
  `https://synthetic.openmed.invalid/<label>`, except the nil UUID, which is
  written out literally.
- Git object IDs are the SHA-1 hex digest of a fixed label such as
  `first synthetic commit`, `synthetic tree` or `synthetic merge base`. The
  third one is truncated to 12 characters so abbreviated IDs are covered too.
- `sha256:` fingerprints and OCI layer digests are the SHA-256 hex digest of a
  fixed label.
- SRI integrity values are the base64 encoding of a SHA-512 or SHA-384 digest
  of a fixed label.
- The base64 image control is a one-pixel PNG. Its second form is the first 44
  characters of the same base64 string followed by an ellipsis.
- The base64 document controls are the base64 encoding of fixed literal JSON
  and YAML bytes.
- The `<<OPENMED_PHI_...>>` values, the template placeholders and the nil UUID
  are literal documented placeholder forms.
- Public-key and certificate blocks are base64 of a fixed label inside the
  matching PEM header and footer lines. They are not real key material.
- ULIDs are the first 16 bytes of a SHA-256 digest mapped onto the Crockford
  base32 alphabet in the 26 character form. Nanoids are the digest mapped onto
  the nanoid alphabet in the 21 character form.
- ETag values are the quoted SHA-1 hex digest of a fixed label.

To add a control for an existing kind, pick a new label and compute the value
the same way, so the file stays reproducible from this file alone.
