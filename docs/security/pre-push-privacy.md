# Pre-push privacy scanner

OpenMed includes a deterministic, local-only privacy scanner for files that a
developer is about to push. It is a development guard, not a compliance
certification or a guarantee that a repository contains no sensitive data.

## Install

After installing the repository's development dependencies, install the hook
from the repository root:

```bash
.venv/bin/python scripts/install_privacy_hook.py
```

The installer preserves an existing `pre-push` hook as
`pre-push.openmed-original` and runs it only after the privacy scan passes.
Running the installer again is safe. Set `OPENMED_PYTHON` when the hook should
use a different local interpreter.

## What is scanned

Git supplies the hook with the local and remote object IDs for each ref update.
The scanner resolves those objects with local Git commands and selects only
added (`A`) and modified (`M`) paths in each pushed range, then scans the blob
at the pushed head commit. Deleted files, unchanged files, unrelated
repository history, and unstaged working-tree changes are not scanned. No
network call is required by the scanner or the installed hook.

UTF-8 text candidates are checked for high-confidence categories:

- email addresses, telephone numbers, government-identifier-shaped values,
  payment-card-shaped values, and public IP addresses;
- private-key headers, provider token shapes, bearer tokens, JWTs, URL
  credentials, and non-placeholder secret assignments;
- structured fields carrying names, record identifiers, dates of birth,
  addresses, or raw text/prompt content.

Binary files are reported as skipped because this scanner does not interpret
binary formats. Oversized or unreadable candidates block the push instead of
being silently ignored.

## Reports and exit status

A passing scan exits `0`. A finding exits `1` and prints only a file/category
summary, for example:

```text
privacy scan failed: 1 file(s) scanned, 2 finding(s)
- tests/fixtures/synthetic_note.json: email (1), raw_text (1)
```

Reports never include matched values, snippets, exception payloads, or source
text. `--json` is available for local automation and contains the same
value-free fields. The detector order, path order, category order, and report
shape are stable.

## Versioned synthetic-fixture allowlist

The built-in allowlist is version `1`. It permits only narrowly scoped,
documented false positives:

- RFC-reserved documentation mailboxes and IP ranges;
- reserved `555-01xx` test telephone numbers and the all-zero synthetic
  government identifier;
- the committed scanner canary at
  `tests/fixtures/secret_scan_canary.txt`.

Additional reviewed synthetic fixtures may be supplied with a JSON extension:

```json
{
  "version": 1,
  "entries": [
    {
      "path": "tests/fixtures/example.txt",
      "category": "email",
      "pattern": "fixture@clinic\\.local",
      "reason": "synthetic fixture approved for the local test"
    }
  ]
}
```

Every entry requires a path glob, category, regular expression, and reason;
path-only exemptions are rejected. The extension is additive to the built-in
rules and is passed explicitly to the scanner with `--allowlist`. Keep the
file synthetic, narrow, and reviewed. Increment the version and update the
tests and this document if the allowlist contract changes.

## Local commands

Inspect explicit candidates without GitHub or another service:

```bash
.venv/bin/python -m openmed.guard.git_hook \
  --repo . \
  --path tests/fixtures/example.txt
```

The normal hook mode reads Git's pre-push update records from standard input.
The `--range BASE..HEAD` mode is useful for a deterministic local check of a
known commit range.
