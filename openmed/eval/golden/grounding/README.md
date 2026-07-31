# Grounding accuracy gold (synthetic)

Synthetic gold for the grounding-accuracy eval suite
(`openmed/eval/grounding_accuracy.py`). Each file scores whether the sparse
candidate generator maps a clinical mention to the correct coded concept for one
permissive vocabulary system.

## Files

| File | System | Concepts | Groundable mention -> code pairs |
| --- | --- | --- | --- |
| `rxnorm.jsonl` | RxNorm | 60 | 315 (195 en, 60 zh, 60 hi) |
| `loinc.jsonl` | LOINC | 60 | 315 (195 en, 60 zh, 60 hi) |
| `icd10cm.jsonl` | ICD-10-CM | 60 | 315 (195 en, 60 zh, 60 hi) |

Each JSONL row is one concept: an invented `code`, `preferred_term`, `synonyms`,
per-language `language_aliases` (`zh`/`hi`), and a list of evaluation `mentions`.
A groundable mention carries the concept's `expected_code`; not-groundable
mentions carry none and exist to characterise abstention.

## Provenance

All content is **fully synthetic and algorithmically generated**. Concept codes,
preferred terms, and alias surfaces are constructed from invented morphemes
combined with a per-concept index (for example `medor007printium`), so every
alias is unique within a system. There is **no UMLS, SNOMED CT, or any other
real or restricted terminology content**: preferred terms, synonyms, and codes
do not reproduce any licensed vocabulary. The `zh`/`hi` aliases are invented
surface forms built from the same unique stems, not translations of real
concept names.

- Codes: `rxnorm` uses numeric RxCUI-like ids, `loinc` uses `NNNNN-N` shapes,
  `icd10cm` uses letter+digit shapes. None correspond to real assignments.
- Every concept is marked `metadata.synthetic = true`.
- Licence: `CC0-1.0` (public domain dedication); permissive so it lives in-repo.

The `tests/unit/eval/test_grounding_accuracy.py` license-policy test asserts the
committed gold contains no restricted-vocabulary markers.

## Regeneration

The corpus is deterministic. Concepts and mentions are produced by a fixed
algorithm (unique per-concept stem + fixed dose/region tokens + a deterministic
one/two-character typo for fuzzy mentions), so re-running the generator yields
byte-identical files. Accuracy is scored offline against an in-memory alias
index built from these same concepts.
