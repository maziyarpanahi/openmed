# OpenMed v0.6.4 — Multilingual PII Accuracy & Span-Alignment Consistency

**Release date:** 2026-03-24

v0.6.4 fixes the issues surfaced by the v0.6.3 quality gates: tokenizer span extension now handles
Unicode combining marks and diacritics, pattern validation failures are properly penalized in
confidence scoring, and overly-permissive regex patterns for postal codes, phone numbers, and
national IDs have been tightened across French, German, Hindi, and Telugu.

---

## What's New

### Aadhaar National ID Support (Hindi & Telugu)

New Verhoeff checksum validator and pattern for Indian Aadhaar numbers:
- 12-digit format with optional 4-4-4 spacing
- Rejects invalid prefixes (0xxx, 1xxx) and checksum failures
- Context-aware scoring with Hindi (`आधार`) and Telugu (`ఆధార్`) keywords

### Unicode-Aware Span Extension

`_fix_entity_spans` now correctly handles non-ASCII text:
- Replaced `.isalnum()` with `unicodedata.category` check covering letters (L), combining marks (M), and numbers (N)
- Capped forward extension at 10 characters to prevent runaway spans
- Removed redundant `.strip()` that created text-mismatch false positives in the quality gate

### Relaxed Quality Gate Text-Mismatch

Whitespace-only differences between `text[start:end]` and `entity.text` (common after span normalization) are now downgraded from WARNING to INFO level. Genuine text mismatches remain WARNING + `SpanValidationWarning`.

### Validation-Aware Confidence Scoring

When a pattern's validator fails (e.g., SSN checksum, NIR key), merged confidence now uses a 90/10 model/pattern weight (instead of the normal 60/40). This prevents high-confidence scores on structurally-invalid entities.

---

## Pattern Tightening

| Pattern | Before | After | Impact |
|---------|--------|-------|--------|
| French postal code | `\d{5}` | `01-95 + 971-976` prefixes | Rejects medical codes, invalid dept prefixes |
| German Steuer-ID | `\d{11}` | `[1-9]\d{10}` | Rejects leading-zero sequences |
| German postal code | `\d{5}` | `01xxx-99xxx` | Rejects `00xxx` range |
| German phone | `\d{2,4}[\s/-]?\d{3,8}` | `\d{2,4}[\s/-]?\d{4,8}` | Rejects short (< 4 digit) suffixes |

## Confidence Calibration

| Pattern | Old base_score | New base_score | Rationale |
|---------|---------------|----------------|-----------|
| French NIR | 0.40 | 0.55 | High structural specificity + validator |
| German Steuer-ID | 0.20 | 0.35 | Tightened pattern + validator |
| French postal code | 0.30 | 0.25 | Still ambiguous even with prefix filter |
| German postal code | 0.30 | 0.25 | Same reasoning |

## Label Normalization Expansion

New `normalize_label()` mappings:
- `bsn`, `dni`, `nie`, `aadhaar` → `national_id`
- `medical_record_number`, `mrn` → `medical_record`
- `account_number` → `account`
- `credit_debit_card`, `credit_card`, `debit_card` → `payment_card`

---

## Test Summary

| Suite | Tests | Status |
|-------|-------|--------|
| Span-boundary guards | 21 | All pass |
| PII accuracy | 28 | All pass |
| Multilingual regression | 33 | All pass |
| Label-map consistency | 46 | All pass |
| **Full suite** | **660** | **All pass** |

---

## Files Added

- `tests/unit/test_pii_accuracy.py` — Confidence penalty, pattern tightening, and calibration tests

## Files Changed

- `openmed/processing/outputs.py` — Unicode-aware `_fix_entity_spans` with capped extension
- `openmed/core/quality_gates.py` — Relaxed text-mismatch with whitespace fallback
- `openmed/core/pii_entity_merger.py` — Validation flag in merging, expanded `normalize_label`
- `openmed/core/pii_i18n.py` — Aadhaar validator + patterns, tightened postal/phone/ID patterns, score calibration
- `openmed/__about__.py` — Version `0.6.3` → `0.6.4`
- `docs/website/index.html` — `softwareVersion` → `0.6.4`
- `CHANGELOG.md` — Added v0.6.4 section
- `README.md` — Updated version references
- `tests/unit/test_quality_gates.py` — Combining-mark and whitespace-mismatch tests
- `tests/unit/test_pii_multilingual_regression.py` — Aadhaar tests for Hindi/Telugu
- `tests/unit/ner/test_label_map_consistency.py` — Expanded normalize_label coverage
- `tests/unit/test_pii_entity_merger.py` — Updated for 6-element semantic unit tuples

---

**Full Changelog:** https://github.com/maziyarpanahi/openmed/compare/v0.6.3...v0.6.4
