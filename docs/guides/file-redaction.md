# File redaction

`openmed redact-files` redacts one local text or line-delimited file into a
separate output path. It is intended for ordinary text, logs, and support
exports; it does not infer a clinical-note schema or overwrite the source.

The command is local-only. It never downloads a model or makes a mandatory
network call, so the selected model must already be cached or supplied as a
local model path.

## Text files

Pass both paths explicitly. The command prints a JSON summary containing only
document and span counts, labels, and offsets in the input text. It never prints
detected source values.

```bash
openmed redact-files support-export.txt support-export.redacted.txt \
  --policy hipaa_safe_harbor \
  --lang en \
  --method mask
```

Use `--report` to write the same PHI-free summary to a separate JSON file:

```bash
openmed redact-files \
  --input support-export.txt \
  --output support-export.redacted.txt \
  --report support-export.redaction-report.json
```

The input and output paths must be different. Output is written atomically
after processing succeeds.

## Line-delimited files

Use `--format lines` for newline-delimited text or `--format jsonl` (also
`ndjson`) for JSON Lines. Each non-empty physical line is processed as one text
unit, blank lines and exact `LF`/`CRLF` endings are preserved, and offsets are
relative to that line. The format is inferred as `jsonl` for `.jsonl` and
`.ndjson` inputs when `--format` is omitted.

```bash
openmed redact-files raw-events.ndjson redacted-events.ndjson \
  --format jsonl \
  --method replace \
  --seed 42 \
  --locale en_US
```

Replacement and date-shifting runs use a fixed seed and consistent surrogates
by default. Use the existing `--policy`, `--lang`, `--method`,
`--confidence-threshold`, `--keep-year`, `--model`, `--locale`, and
`--no-safety-sweep` controls to match the library de-identification behavior.

The command is a redaction utility, not a compliance certification or a
guarantee of zero residual risk. Validate synthetic fixtures and review the
PHI-free summary before sharing any output.
