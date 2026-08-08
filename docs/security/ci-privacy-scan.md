# CI privacy scan

The repository includes a dependency-free composite action for checking
committed fixtures, traces, and other explicitly selected files for likely
sensitive values. The scan is local and deterministic: it uses fixed regular
expression rules, reads only the paths supplied to the action, and performs no
network calls.

The scan is a defensive CI gate, not a compliance certification or a guarantee
that a document is free of sensitive data. Keep committed examples synthetic
and use the existing [no-raw-PHI logging policy](no-raw-phi-logging.md) for
runtime logging.

## Using the action

Set up Python 3.10 or newer, check out the repository, and pass the paths to
scan explicitly. Paths and glob patterns are newline-delimited. The action
fails when it finds a likely value, emits annotations that contain counts only,
and uploads `privacy-scan-report.json` as a machine-readable artifact.

```yaml
jobs:
  privacy-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.10"
      - uses: ./.github/actions/privacy-scan
        with:
          paths: |
            tests/fixtures
            traces
          policy: default
          synthetic-fixture-allowlist: |
            tests/fixtures/synthetic/**
```

The action does not infer a repository-wide path. This keeps the scan scope
reviewable and prevents an invocation from unexpectedly reading caches,
generated environments, or unrelated mounted data. Directory scans skip common
generated environments such as `.git`, `.venv`, `node_modules`, and
`__pycache__`; symbolic links are not followed.

## Policies

The built-in `default` policy checks for credentials, private keys, database
URLs, JWT-shaped values, email addresses, phone numbers, US Social Security
numbers, payment-card-shaped values that pass Luhn validation, IPv4 addresses,
and values assigned to common sensitive fields such as `patient_id`, `mrn`,
`dob`, `email`, or `token`.

The `strict` policy adds UUIDs and long numeric identifiers. The `minimal`
policy keeps credential, email, SSN, and labeled-field checks. The
`credentials` policy checks only credential-like values, private keys, database
URLs, and JWTs.

For a custom policy, pass a JSON file containing an enabled rule list. Rule
names are the names in the report's `rules` field:

```json
{
  "name": "fixture-policy",
  "rules": ["email", "phone", "credential", "labeled_sensitive"]
}
```

The scanner accepts `rules` as a list, a comma-separated string, or a mapping
of rule names to booleans. Unknown rules fail the action without echoing the
configuration contents.

## Synthetic fixture allowlists

Use `synthetic-fixture-allowlist` only for reviewed, committed synthetic data
that is intentionally shaped like a sensitive value. Its value is a newline-
delimited list of repository-relative paths or glob patterns:

```yaml
synthetic-fixture-allowlist: |
  tests/fixtures/synthetic/**
  examples/trace-canaries/*.json
```

For a larger list, prefix a JSON path-list file with `@`:

```json
["tests/fixtures/synthetic/**", "examples/trace-canaries/*.json"]
```

```yaml
synthetic-fixture-allowlist: "@.github/privacy-scan-allowlist.json"
```

Allowlisting skips the matching file, so it must not be used for real data or
as a way to suppress an unexpected finding. Non-allowlisted files are always
scanned.

## Report and privacy behavior

The JSON artifact contains the policy name, scanned-file counts, per-file
finding counts, and counts grouped by rule. It never stores matched text,
snippets, offsets, hashes of matched values, exception details, or fixture
contents. Console annotations and failure messages likewise contain only
counts and safe repository-relative paths. Configuration and read failures use
stable error categories and do not echo paths or input values.

Run the scanner locally with an output path outside the repository if you do
not want a report left in the worktree:

```bash
.venv/bin/python scripts/privacy_scan.py \
  --paths tests/fixtures traces \
  --policy default \
  --output /tmp/openmed-privacy-scan.json
```
