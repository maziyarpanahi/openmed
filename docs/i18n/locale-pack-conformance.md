# Offline locale-pack conformance checks

`openmed i18n check <language>` inspects one locale pack and reports one
independent result per component, so a missing registry entry, national-ID
validator, surrogate locale, synthetic fixture, span payload or evidence
reference can never mask another gap. The command is import-light and fully
offline: it needs no model weights, network access, credentials or real clinical
text.

```bash
openmed i18n check zh
openmed i18n check zh --fixture-root fixtures/zh --spans spans/zh.json --json
```

```python
from openmed.cli.i18n_check import (
    run_locale_pack_conformance,
    format_conformance_report,
)

report = run_locale_pack_conformance(
    "zh",
    fixture_roots=["fixtures/zh"],
    span_payloads=["spans/zh.json"],
    evidence_paths=["evidence/zh-review.json"],
    repository_root=".",
)
assert report.ok
print(format_conformance_report(report))
```

## Components and statuses

Components are reported in a fixed order: `metadata`, `registry`, `validator`,
`surrogate`, `fixtures`, `span_integrity` and `evidence`. Each component is
`pass`, `fail` or `skipped`; a component whose inputs were not supplied is
`skipped` instead of silently passing, and one component failing never stops the
others from running.

- `metadata` checks that the pack is registered, declares the requested language,
  and carries non-empty scripts and a default model.
- `registry` checks that the pack's `segmenter_id` is a registered segmenter and
  that its recognizers are valid names.
- `validator` resolves every `national_id_providers` entry against the
  `validate_<provider>` callables exported by `openmed.core.pii_i18n`.
- `surrogate` checks that the declared `surrogate_locale` resolves in the locale
  catalog.
- `fixtures` scans each `--fixture-root` directory for `.json`/`.jsonl` records
  and requires them to target the language, declare
  `"safety": "verified_synthetic"`, and carry no text-bearing keys.
- `span_integrity` loads each `--spans` payload
  (`{"text": ..., "spans": [{"start", "end", "label", "text"}]}`) and validates
  the offsets with `validate_span_integrity`.
- `evidence` requires each `--evidence` path to exist and be unique, and reports
  a `<scope>:sha256:<hex>` reference per file.

A check that finds no gap reports `fail` with a stable reason code; the closed
set lives in `REASON_CODES`, so a report can never carry an undocumented code.
Reason codes are grouped by component in the source constants.

## Output

The text render is one line per component followed by a summary, for example:

```text
locale pack conformance: zh [openmed.i18n.locale_pack_conformance.v1]
metadata pass pack_declared: pack 'zh' declares scripts Han
registry pass registry_wired: pack 'zh' resolves segmenter 'jieba'
validator pass national_id_providers_resolved: resolved 1 national-ID validator(s)
surrogate pass surrogate_locale_resolved: surrogate locale 'zh_CN' resolves
fixtures skipped no_fixture_roots: no fixture roots were supplied
span_integrity skipped no_span_payloads: no span payloads were supplied
evidence skipped no_evidence_references: no evidence references were supplied
summary pass=4 fail=0 skipped=3 ok=true
```

`--json` prints the standard command envelope
(`{"ok": true, "command": "i18n check", "data": ...}`) whose payload carries
`language`, `schema_version`, `findings`, `ok` and a `summary` of per-status
counts. The command exits `0` when no component failed, `1` when a component
failed, and `2` for a usage error such as an invalid language code.

## Determinism and privacy

Findings keep the fixed component order, JSON is written with sorted keys and
compact separators, and the payload contains no timestamps, absolute paths,
locale-dependent formatting or environment values: the same inputs render the
same bytes on every machine. Evidence references are derived from the path
relative to the repository root, and a file outside the root is reported by its
base name only, so no host path leaks into a report.

## Failures

`run_locale_pack_conformance` raises `ValueError` for a language code that is not
a lowercase ISO 639-1 code such as `en` or `zh`. Malformed component inputs are
never exceptions: an unreadable fixture root, a missing evidence file or a
malformed span payload is reported as a `fail` finding with a specific reason
code, so a single call always yields a complete report.

## Out of scope

The check verifies wiring, declared metadata and evidence shape. It does not run
models, download assets, grade clinical language quality, validate live PII
detection, or publish a pack. `skipped` means "not checked", not "passing".

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/cli/test_i18n_check.py -q
```

The focused suite is offline: it pins the golden text and the SHA-256 digest of
the canonical JSON, blocks socket calls, and exercises every component failing
independently, both identity and external evidence scopes, usage errors, and the
CLI help and JSON envelopes. Related references:
[Offline structural locale tags](locale-tags.md) and the
[clinical language-pack readiness matrix](language-readiness-matrix.md).
