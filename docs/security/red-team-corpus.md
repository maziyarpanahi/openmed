# Adversarial PHI red-team corpus

OpenMed ships a synthetic adversarial-PHI corpus and an offline harness that
measures whether protected identifiers survive the de-identification pipeline.
It turns the abuse cases in the [redactor threat model](threat-model.md) into a
repeatable, leakage-first evaluation gate.

The corpus is
[`eval/redteam/corpus/adversarial_phi.jsonl`](https://github.com/maziyarpanahi/openmed/blob/master/eval/redteam/corpus/adversarial_phi.jsonl).
Every row is explicitly marked `synthetic: true`; contains a stable
`abuse_case_id`, an attack type, and one or more `expected_protected`
assertions; and uses documentation-only identifiers. Restricted clinical
datasets and real PHI must never be added.

## Run offline

The default runner calls `openmed.deidentify` with `OpenMedConfig(local_only=True)`
and the deterministic safety sweep enabled. This blocks network fallback. The
selected model must already be cached, or `--model-name` must point to a local
model directory. A model-loading or pipeline error is scored as a bypass rather
than skipped.

```bash
OPENMED_OFFLINE=1 .venv/bin/python -m openmed.eval.redteam \
  --max-bypass-rate 0 \
  --output artifacts/redteam-report.json
```

Use a different cached model or local directory when needed:

```bash
.venv/bin/python -m openmed.eval.redteam \
  --model-name /path/to/local/pii-model \
  --max-bypass-rate 0.01 \
  --output artifacts/redteam-report.json
```

The threshold is inclusive: a measured rate equal to the configured maximum
passes, while a higher rate exits with status 1. CI can set the same optional
gate without changing the command:

```bash
export OPENMED_REDTEAM_MAX_BYPASS_RATE=0
.venv/bin/python -m openmed.eval.redteam \
  --output artifacts/redteam-report.json
```

Without the CLI flag or environment variable, the harness still emits the
report with decision `MEASURED` but does not enforce a rate.

## Scoring and report safety

Each assertion declares a protected surface and its comparison strategy:

```json
{
  "label": "email",
  "value": "synthetic.patient@example.com",
  "match": "normalized"
}
```

Supported matching modes are:

- `exact`: literal substring comparison.
- `casefold`: case-insensitive Unicode comparison.
- `normalized`: the same adversarial-Unicode folding used by PII detection.
- `alnum`: normalized comparison with separators removed, suitable for
  structured identifiers.

A case is a bypass if any expected-protected assertion survives or if the
pipeline fails to return de-identified text. The overall bypass rate is
`bypassed cases / total cases`; the report also groups the same rate by attack
type and retains the mapped abuse-case ids.

Reports never include input text, de-identified output, protected values, or
exception messages. Case results contain only ids, counts, exception types,
and SHA-256 hashes of leaked synthetic assertions.

## Corpus coverage

The committed pack covers:

| Attack type | Threat-model mapping |
|---|---|
| Zero-width and whitespace splits | AC-01 |
| Homoglyph and mixed-script email obfuscation | AC-03 |
| Full-width structured identifiers | AC-04 |
| Combining-mark obfuscation | AC-05 |
| Checksum-valid identifiers mixed with invalid decoys | AC-07 |
| Locale-specific date formats | AC-08 |
| Role-played leakage requests | AC-11 |

AC-02 remains a documented known gap. Do not publish an actionable reproducer
for an unmitigated bypass in this corpus; report it privately through
[`SECURITY.md`](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md)
and add the regression case with its fix.

## Add a case

Every new JSONL row must:

1. contain a unique `id`, a catalogued `AC-xx` id, and a stable `attack_type`;
2. set `synthetic` to the JSON boolean `true`;
3. include at least one `expected_protected` assertion whose value appears in
   the case text; and
4. use only generated, documentation-only fixtures with no real PHI.

The loader rejects missing assertions, unknown abuse-case-id shapes, duplicate
ids, malformed JSON, and any row not explicitly marked synthetic.
