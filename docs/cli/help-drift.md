# CLI help-surface drift

`openmed.cli.help_drift` compares synthetic command-help records without
running a command or making a network request. It is intended for offline CI
checks that keep generated help and documentation aligned.

## Input records

The checker accepts a JSON list of command records, or an object containing a
`commands` or `records` list:

```json
[
  {
    "command": ["reports", "inspect"],
    "options": [
      {"flags": ["--input", "-i"], "required": true},
      {"flags": ["--json"], "action": "store_true"}
    ]
  }
]
```

`command` may also be a whitespace-separated string. Option records support
`flags` (or `option_strings`, `names`, `flag`, or `option`) and the shape fields
`required`, `nargs`, `action`, `takes_value`, and `repeatable`.

## Value-free signatures

`normalize_help_records()` returns a `HelpSurfaceSignature`. Its canonical
representation contains only sorted command paths and option shape:

- option flags and aliases;
- whether an option is required;
- value arity (`none`, `one`, `optional`, `zero_or_more`, `one_or_more`, or
  `fixed:N`); and
- whether the option is repeatable.

Defaults, choices, help text, metavars, destinations, and runtime values are
discarded. The signature can be serialized with `to_json()` or identified by
its deterministic SHA-256 `digest`. Invalid input raises `HelpDriftError`
without including input values in the exception message.

## Drift categories and exit codes

`compare_help_surfaces(baseline, candidate)` returns a `HelpDriftReport` with
sorted `added`, `removed`, and `changed` option records. It also records empty
command additions/removals, which cannot be represented as option changes.
Option identity prefers a long flag, so adding or removing an alias is a
change to the same option; renaming a flag is an addition plus a removal.

| Exit code | Category | Meaning |
| ---: | --- | --- |
| 0 | `clean` | No command or option drift |
| 1 | `added` | Only commands/options were added |
| 2 | `removed` | Only commands/options were removed |
| 3 | `changed` | Existing option shape changed |
| 4 | `mixed` | More than one drift category is present |
| 5 | invalid input | A local record could not be normalized |

The JSON report contains only normalized flags, command paths, shape metadata,
category counts, and surface digests. It does not include discarded values.

## Local JSON CLI

Compare two local files with:

```bash
python -m openmed.cli.help_drift baseline.json candidate.json
python -m openmed.cli.help_drift baseline.json candidate.json --format text
```

The process prints a deterministic JSON report by default and returns the
category's exit code. No credentials, remote services, or mandatory network
calls are involved.
