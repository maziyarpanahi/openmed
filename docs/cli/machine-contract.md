# Machine-readable CLI contract

`--json` emits one document with stable command, exit-status, and error-code
fields.

## Contract fixtures

The offline gate covers the outcomes below. Fixtures omit inputs, paths,
exceptions, and record values.

| Outcome | Command path | Exit code | Error code |
| --- | --- | ---: | --- |
| Success | `models list` | `0` | — |
| Validation failure | `risk discover` | `2` | `invalid_discovery_config` |
| Offline failure | `models pull` | `1` | `offline_unavailable` |
| Privacy-policy failure | `risk assess` | `1` | `release_policy_failed` |

## JSON shape

Successful commands use the following top-level keys:

```json
{
  "ok": true,
  "command": "models list",
  "data": {"count": 0, "models": []}
}
```

Failures use the same command field and a stable error code/message pair:

```json
{
  "ok": false,
  "command": "models pull",
  "error": {
    "code": "offline_unavailable",
    "message": "The requested operation requires a local model cache."
  }
}
```

`message` never echoes input, identifiers, paths, exceptions, or model
responses. Change a fixture only for an intentional public-contract change.

## Exit codes

- `0` means the command completed successfully.
- `1` means a runtime, offline, or privacy-policy gate failed.
- `2` means command validation or usage failed.

Run the deterministic offline gate with
`.venv/bin/python -m pytest tests/unit/cli/test_contract.py -q`.

These fixtures are a scripting contract, not a compliance or clinical
guarantee.

## Guarded summary and NLI commands

`summarize` and `nli verify` use the same success/error envelopes above. Their
fixed error message is `Local clinical request could not complete.`; it never
includes clinical text, paths, backend names supplied by a caller or exception
details. Use `--json` on the leaf command for stable error codes.

| Outcome | Exit | Stable codes |
| --- | ---: | --- |
| Successful summary files or all claims entailed | `0` | None |
| Valid NLI contradiction, neutral or abstention | `1` | No error; `ok: true` with per-claim evidence |
| Backend/cache/dependency unavailable | `1` | `summary_backend_unavailable`, `nli_backend_unavailable` |
| Privacy or ordering gate refusal | `1` | `summary_leakage_rejected`, `summary_deidentification_rejected` |
| Output reservation/write failure | `1` | `summary_output_failed` |
| Invalid backend result or processing failure | `1` | `summary_result_invalid`, `summary_failed`, `nli_result_invalid`, `nli_failed` |
| Invalid command arguments | `2` | `summary_arguments_invalid`, `nli_arguments_invalid` |
| Unsupported mode or local alias | `2` | `summary_mode_unsupported`, `summary_model_invalid`, `nli_backend_invalid`, `nli_backend_factory_invalid` |
| Invalid, unavailable, special or oversized local input | `2` | `summary_input_invalid`, `summary_input_unavailable`, `summary_input_not_regular`, `summary_input_too_large`; corresponding `nli_input_*` codes |
| Invalid claim array or per-claim bounds | `2` | `nli_claims_invalid` |

The summary console/file metadata contains only `mode`, `backend_id`,
`template_digest`, `leakage_check`, `summary_characters` and
`human_review_required`. The protected summary goes to its separate explicit
new destination. NLI data contains only a `claims` array with each row's
`claim_index`, `label`, `score`, `backend_id`, `contradicted` and
`review_required`. Plain console output also omits clinical values.

Summary/source inputs are limited to 16 KiB, protected summary output to
8 KiB, and the claims file to 64 KiB with 1–128 claims of at most 4 KiB each.
These commands use existing native APIs with local-only defaults and never
download models. Their guards do not qualify a model artifact or validate a
clinical workflow. See [summarization](../clinical/summarization.md#guarded-local-cli)
and [NLI verification](../clinical/nli-verification.md#local-cli-verification)
for provisioning and trusted-code boundaries.

## Offline clinical extraction

`openmed clinical sdoh`, `openmed clinical relations` and
`openmed clinical timeline` consume existing analyzed spans. They run the local
Python extractors without loading NER models or downloading artifacts:

```bash
openmed clinical sdoh --note note.txt --spans analyzed.json --json
openmed clinical relations --note note.txt --spans analyzed.json --json
openmed clinical timeline --note note.txt --spans analyzed.json \
  --reference-time 2026-06-15 --json
```

`--note` is a UTF-8 regular local file limited to 16,384 bytes, the same note
limit as the brief CLI. `--spans` accepts the success envelope from
`openmed analyze --json`, or its unwrapped prediction-result `data` object.
Both shapes contain an `entities` array. Supplied document text and entity
surfaces must match the note exactly. Offsets are half-open Python character
offsets, including for Unicode text. Each entity needs integer `start` and
`end`, a registered NER label or supported clinical attribute label, and an
optional finite `confidence` in `[0, 1]`. The CLI maps `confidence` to the
Python extractor's `score`; the compatible `score` spelling is also accepted.
Conflicting scores are invalid. Empty notes and entity arrays are valid.
Existing model names, timestamps and arbitrary metadata are discarded.
ASCII labels allow letters, digits, underscores, hyphens and spaces; arbitrary
punctuation, controls and non-ASCII suffixes are refused before normalization.

Span and section JSON files are limited to 1 MiB. Inputs reject duplicate JSON
keys, non-finite numbers, nesting deeper than 32 levels, more than 65,536 JSON
values, more than 512 entities, and more than 256 sections. Symlinks and
non-regular input files are refused. Every span, section, supplied context axis
and explicit temporal link is validated before extraction. Malformed offsets
fail rather than being silently skipped.

`--sections sections.json` accepts a JSON array of `{start, end, label}` records
forming a complete contiguous partition of the note. Labels are canonical
section names or `unsectioned`. SDOH extraction retains only
[Social History findings](../clinical/sdoh-section-scope.md); English medication
relations and timeline context use the same section inputs as their Python
APIs. Supplied section headers and other metadata are omitted from output.

Languages are explicit: SDOH supports `en`; relations support the native `en`,
`hi` and `zh` APIs; timeline normalization supports native `en` and `de` rules.
The multilingual relation API does not accept sections. Unsupported languages
and option combinations fail with typed codes, without translation or English
fallback. Declaring a language does not validate the note's actual language.

The timeline command composes `assert_context`, `normalize_temporal` and
[the public `build_timeline` API](../clinical/timeline-buckets.md). It preserves
supplied axes under entity `metadata.clinical_context` (or compatible top-level
or `assertion` fields), then groups events into historical, recent and
hypothetical lanes. Conflicting supplied axes are invalid. `certainty` and
`uncertainty` accept `certain` or `uncertain`; negation accepts `affirmed` or
`negated`; experiencer accepts `patient` or `family`. Without supplied axes,
native deterministic ConText cues and section priors provide advisory assertions.

Temporal entities have `DATE`, `DATE_OF_BIRTH`, `TIME`, `TIMEX` or `DURATION`
labels, including registered aliases. `--reference-time` accepts an ASCII ISO
calendar date or timezone-aware ISO datetime with seconds. It never reads the
wall clock or analyzed result's timestamp. Relative expressions remain
`unanchored` without a reference. Partial and ambiguous expressions retain the
normalizer's offsets and granularity flags. Calendar values and reference dates
stay in memory and are omitted from JSON.

Temporal entities can order their own timeline events. Another event receives
a time only when its entity metadata contains an explicit
`clinical_time_span: [start, end]` pointing to a supplied temporal entity.
The CLI does not infer a nearest-date association. Only unambiguous normalized
calendar-day dates anchor events; other events remain unanchored. This is an
advisory span ordering, not a clinical chronology of record.

The shared `{ok, command, data}` envelope contains `schema_version: 1`,
language, input span count, SHA-256 digests of the input file bytes,
`review_required: true`, and the command's records:

- SDOH: trigger offsets, category, status, temporality, score and extent presence;
  trigger values, occupations and amounts are omitted.
- Relations: head/tail offsets and labels, canonical relation type, score,
  derived-tail flags and supported assertion status; source surfaces are omitted.
- Timeline: event offsets and supplied labels, kinds, assertions, time states,
  lane indices and temporal offsets/types/states/granularity flags; source
  surfaces, calendar values and raw identifiers are omitted.

All findings require review and do not authorize clinical action. Digests
support input binding; they are not an anonymization guarantee.

`--output result.json` reserves a new file exclusively with mode `0600` and
writes the same value-free envelope. Existing files and symlinks are refused;
a failed write removes the partial file only while its device and inode still
match the file reserved by this invocation. A replacement is preserved. The
complete envelope must serialize as finite JSON within 1 MiB before any file
creation or stdout emission; larger results fail without truncation. Stdout still contains
the value-free result with `output_written: true`, and never the output path.
Without `--json`, stdout contains the value-free data object. Errors use fixed
messages without note text, span surfaces, private paths or exception details,
including retained exception context. Projection codes and scores must be exact
built-in scalar types; custom scalar subclasses are refused.
Processor writes to Python stdout and stderr are discarded. Registered custom
extractors execute as trusted local code; this is not an execution sandbox.

| Failure | Exit | Stable code |
| --- | ---: | --- |
| Invalid clinical command arguments | 2 | `clinical_arguments_invalid` |
| Unsupported language / options | 2 | `clinical_language_unsupported` / `clinical_options_unsupported` |
| Invalid reference | 2 | `clinical_reference_invalid` |
| Missing, unreadable or symlink input | 2 | `clinical_input_unavailable` |
| Non-regular or oversized input | 2 | `clinical_input_not_regular` / `clinical_input_too_large` |
| Invalid UTF-8 / JSON | 2 | `clinical_input_encoding` / `clinical_json_invalid` |
| Invalid span document / entity / count | 2 | `clinical_spans_invalid` / `clinical_span_invalid` / `clinical_spans_too_many` |
| Unknown label or mismatched text | 2 | `clinical_label_unsupported` / `clinical_source_mismatch` |
| Invalid section partition / temporal link | 2 | `clinical_sections_invalid` / `clinical_time_link_invalid` |
| Processing or output failure | 1 | `clinical_processing_failed` / `clinical_output_unavailable` |

Supported clinical attribute labels include `DOSE`, `DOSAGE`, `DURATION`,
`FREQUENCY`, `ROUTE`, `STRENGTH`, `FORM`, `STATUS`, `INDICATION`, `SEVERITY`,
`TIMEX`, `DRUG`, `MED`, `MEDICINE`, `RX`, `DISEASE`, `SYMPTOM`, `DIAGNOSIS`,
`EVENT`, `FINDING` and `OBSERVATION`. Chinese CMeIE entity types are also
accepted. Unknown labels are refused rather than copied into evidence. Custom
SDOH extractors must emit built-in controlled categories, statuses and
temporality states for this CLI projection; unknown output codes fail privately.

Validate parity and process behavior offline with
`.venv/bin/python -m pytest tests/unit/cli/test_clinical_cli.py tests/integration/test_clinical_cli_offline.py -q`.
