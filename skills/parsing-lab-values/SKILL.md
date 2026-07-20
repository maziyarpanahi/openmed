---
name: parsing-lab-values
description: "Parse laboratory values and reference ranges from clinical text and flag results as low, normal, high, or critical with OpenMed. Use when the user needs to interpret lab results, compute abnormal flags, parse reference ranges like \"135-145\" or \"<5\", honor an originating-lab flag (H/L/critical), or turn extracted lab entities into structured high/low/critical signals. Covers openmed.clinical.parse_reference_range, derive_abnormal_flag, ReferenceRange, and AbnormalFlag, with UCUM/LOINC framing. Unit-agnostic — it does not convert units. Pairs after extracting-clinical-entities (lab entities from analyze_text)."
license: Apache-2.0
metadata:
  project: OpenMed
  category: clinical-nlp
  pairs: after
  version: "1.0"
---

# Parsing lab values

Lab results in clinical text arrive as a value, a unit, and a reference range
("Sodium 132 mmol/L (135–145)"). To act on them you need a structured
**abnormal flag** — is 132 low, normal, high, or critical? OpenMed's
`openmed.clinical` lab helpers parse the reference range deterministically and
derive the flag, honoring any explicit flag the originating lab already supplied.
The helpers are **unit-agnostic by design**: they compare numbers within a stated
range and never convert units, so a mmol/L value is never silently compared
against a mg/dL range.

## When to use

- After `extracting-clinical-entities` surfaces lab/measurement entities and you
  need to classify each as low / normal / high / critical.
- The user asks to parse reference ranges, flag abnormal labs, build a flagged
  labs table, or interpret values like `<5`, `>=10`, `0.5 - 1.2`.
- You have an originating-lab flag (`H`, `L`, `C`, `HH`) and want it honored over
  a derived comparison.

## Quick start

```python
from openmed.clinical import (
    parse_reference_range, derive_abnormal_flag, LAB_FLAG_ADVISORY,
)

# Closed range
rng = parse_reference_range("135-145")
# -> {"low": 135.0, "high": 145.0, "low_inclusive": True, "high_inclusive": True}

derive_abnormal_flag(132, rng)            # "low"
derive_abnormal_flag(140, "135-145")      # "normal"  (raw range string accepted)
derive_abnormal_flag(150, "135 to 145")   # "high"

# One-sided bounds
derive_abnormal_flag(7, parse_reference_range("<5"))    # "high" (above the cap)
derive_abnormal_flag(3, parse_reference_range(">=10"))  # "low"

# Honor the lab's own explicit flag (takes precedence over derived comparison)
derive_abnormal_flag(132, "135-145", explicit_flag="C")   # "critical"
derive_abnormal_flag(132, "135-145", explicit_flag="HH")  # "critical"

# Unparseable / non-numeric inputs fail safe rather than guessing
derive_abnormal_flag("pending", "135-145")  # "unknown"
derive_abnormal_flag(132, "see report")     # "unknown"

print(LAB_FLAG_ADVISORY)  # surface this disclaimer with derived flags
```

`AbnormalFlag` is one of `"low" | "normal" | "high" | "critical" | "unknown"`.
`ReferenceRange` is a typed mapping of `low`, `high`, `low_inclusive`,
`high_inclusive`.

## Workflow

1. **Get value + range + (optional) lab flag** from extracted lab entities. The
   value should be numeric; the range may be a raw string or a parsed mapping.
2. **Parse the reference range** with `parse_reference_range`. It handles closed
   ranges (`"135-145"`, `"0.5 - 1.2"`, `"135 to 145"`, en/em dashes) and
   one-sided bounds (`"<5"`, `"<=5"`, `">10"`, `">=10"`). Contradictory or
   unparseable ranges return **empty bounds** rather than a guess — by design.
3. **Derive the flag** with `derive_abnormal_flag(value, range, explicit_flag=)`.
   Resolution order: an explicit lab flag wins first (`H/HIGH`, `L/LOW`,
   `C/CRIT/CRITICAL`, `HH/LL` → critical, `N/NORMAL`); an *unknown* explicit flag
   returns `"unknown"` instead of being silently ignored. With no explicit flag,
   it compares the numeric value against the parsed bounds, respecting inclusive
   vs. exclusive edges.
4. **Handle `"unknown"` explicitly.** Non-numeric values, empty/unparseable
   ranges, or unrecognized explicit flags yield `"unknown"`. Treat it as
   "needs review," not "normal."
5. **Attach the advisory.** Surface `LAB_FLAG_ADVISORY` wherever derived flags
   are shown — derived flags are heuristic and do **not** replace the originating
   laboratory's own diagnostic flagging.

## Hand-off to / from OpenMed

- **From** `extracting-clinical-entities`: `analyze_text` lab/measurement
  entities give you the value text, unit, and often the reference range; this
  skill turns them into structured flags. Parse the numeric value out of the
  entity surface before calling `derive_abnormal_flag`.
- **OpenMed calls:** `from openmed.clinical import parse_reference_range,
  derive_abnormal_flag, ReferenceRange, AbnormalFlag, LAB_FLAG_ADVISORY`.
- **To** `reconciling-problem-lists` / FHIR grounding: a `critical`/`high`/`low`
  flag becomes a FHIR `Observation.interpretation` code (HL7 v3
  ObservationInterpretation: `H`, `L`, `HH`, `LL`, `N`). Ground the LOINC code and
  UCUM unit out-of-process; OpenMed emits the flag, not the terminology binding.

## Edge cases & gotchas

- **Unit-agnostic — convert before comparing.** The helpers ignore units
  entirely. If the value and the range are in different units (mg/dL vs mmol/L),
  the flag is wrong. Normalize units **before** calling, or only compare value
  and range that share a unit.
- **Inclusive vs. exclusive edges.** `"<5"` makes 5 the high bound *exclusive*;
  a value of exactly 5 flags `high`. `parse_reference_range` records
  `high_inclusive=False` for `<` and `True` for `<=` — respect it.
- **Critical needs an explicit flag.** Derived comparison yields only low/normal/
  high. `"critical"` comes from the lab's explicit flag (`C`, `HH`, `LL`); the
  helpers do not infer critical thresholds beyond the reference range.
- **Empty bounds are intentional.** A range with both bounds missing returns
  `"unknown"` from `derive_abnormal_flag`, not `"normal"`. Don't treat unknown as
  in-range.
- **Local-first, advisory-only.** Runs on-device; flags are decision support, not
  a diagnosis. Always carry `LAB_FLAG_ADVISORY`.

## Standards & references

- LOINC — universal codes for laboratory observations:
  https://loinc.org/
- UCUM — Unified Code for Units of Measure:
  https://ucum.org/
- HL7 v3 ObservationInterpretation (H/L/HH/LL/N abnormal flags):
  https://terminology.hl7.org/CodeSystem-v3-ObservationInterpretation.html
- HL7 FHIR R4 Observation — `referenceRange` and `interpretation`:
  https://hl7.org/fhir/R4/observation.html
