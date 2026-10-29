# Synthetic script tokenization audit

`openmed.eval.script_tokenization.audit_script_tokenization` measures token
fragmentation, unknown-token share, grapheme splits, and source-offset recovery.
It complements the existing `openmed.eval.coverage` UNK/byte-fallback inventory:
it does not replace its manifest updates or claim that good tokenization implies
good entity recall.

Supply explicitly synthetic `ScriptTerm` records and a local tokenizer adapter
returning `TokenizationObservation(token_ids, offsets)`. Offsets are half-open
source **code-point** coordinates and must omit special tokens. Byte offsets must
be converted by the adapter. The shared extended-grapheme iterator handles
combining marks, Indic conjuncts and emoji joins consistently with other OpenMed
span utilities. Offsets that overlap, go backwards, are empty, or leave the source
bounds are scored as invalid; valid offsets contribute non-whitespace source
coverage. A grapheme-internal endpoint is scored separately even if all code
points are covered. No automatic correction hides an adapter failure.

Pin a SHA-256 model identity and `ScriptTokenizationLimits`. Each threshold is
checked independently on unrounded pooled counts; exact equality passes.
Empty tokenization and invalid offsets always fail qualification. The result uses
`qualified_tokenization` or `unsupported`, never a supported-language claim.
Keep language/model qualification and clinical evaluation gates separate.

Reports contain only model digest, controlled script codes, counts, rates,
thresholds and controlled reason codes. Text and token identities are not
returned. Script codes must come from a caller-controlled vocabulary: syntax
validation is not a PHI filter. A synthetic declaration is a caller assertion,
not proof that data is synthetic. No data, model, or tokenizer is downloaded.
Each call is limited to 10,000 terms, 4,096 characters per term, and 8,192 tokens
per term; outputs are aggregated by script rather than retained individually.

```sh
.venv/bin/python -m pytest tests/unit/eval/test_script_tokenization.py -q
```
