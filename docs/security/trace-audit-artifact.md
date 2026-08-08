# Counts-Only Trace Privacy Audit Artifact

`openmed.guard.audit` provides a small local artifact for recording evidence
that an explicitly selected trace store was scanned. It is an evidence
summary, not a copy of the trace store and not a compliance certification.

## Recorded fields

The fixed artifact schema contains only:

- `scanner_version`: a bounded scanner identifier;
- `policy_hash`: a SHA-256 reference to the policy used by the scanner;
- `file_fingerprints`: sorted SHA-256 content fingerprints, never paths or
  file names;
- `category_counts`: sorted category-to-count pairs; and
- `disposition`: a bounded operational result such as `clean`, `redacted`, or
  `quarantined`.

There is deliberately no generic metadata or payload field. Source values,
replacement mappings, prompts, tool outputs, trace bodies, and finding details
cannot be represented by the artifact contract. Unknown fields supplied by a
scanner summary are ignored when the summary is converted to an artifact.

## Local-first deterministic usage

The helper performs no network calls. Policy content is consumed only while
calculating its digest, and `fingerprint_file` reads only the explicitly
supplied local file. Neither helper stores the input content or path:

```python
from openmed.guard.audit import TraceAuditArtifact, hash_policy

artifact = TraceAuditArtifact.from_files(
    scanner_version="trace-scanner/1.0",
    policy_hash=hash_policy("synthetic policy configuration"),
    files=["trace-store.jsonl"],
    category_counts={"NAME": 2, "PHONE": 1},
    disposition="redacted",
)

artifact.write_json("evidence/trace-audit.json")
artifact.write_markdown("evidence/trace-audit.md")
```

`to_json()` sorts keys and `to_markdown()` sorts fingerprints and categories.
Neither rendering includes timestamps, host details, paths, source values,
replacement mappings, prompts, or tool outputs, so repeated runs over the same
summary produce byte-stable output.

## Operational boundaries

Only pass hashes and aggregate counts from a scanner. Keep raw traces and any
reversible mapping in their separately controlled local stores. The artifact
proves what the scanner reported; it does not prove that the scanner detected
every sensitive value or that a particular disposition satisfies a legal or
regulatory requirement.
