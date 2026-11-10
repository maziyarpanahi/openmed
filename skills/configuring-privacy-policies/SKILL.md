---
name: configuring-privacy-policies
description: "Select and customize OpenMed's seven bundled privacy policy profiles for de-identification, and build custom surrogate generators. Use when the user asks which policy fits HIPAA Safe Harbor vs Expert Determination vs GDPR vs PIPEDA vs a research limited dataset vs strict no-leak, wants to pass policy= to deidentify(), needs to keep quasi-identifiers for research, or must register a custom MRN/name/address surrogate provider. Covers the profile-to-use-case map, AnonymizerConfig/Anonymizer for fine control, and register_clinical_provider / register_label_generator. Pairs with OpenMed deidentifying-clinical-text and generating-synthetic-surrogates."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: adjacent
  version: "1.0"
---

# Configuring privacy policies

A **policy profile** is a named bundle of de-identification decisions: which
action (mask/redact/replace/keep) applies to each label, how aggressively
detectors arbitrate, whether the mandatory safety sweep runs, and whether a
reversible mapping is produced. OpenMed ships seven profiles. Pass one by name
to `deidentify(policy=...)` and you get a compliance-aligned default without
hand-wiring 50+ per-label actions. Everything runs **on-device**.

## When to use this skill

Use it to pick the right `policy=` for a regulatory context, to understand what
a profile actually changes, or to go beyond the bundle — keeping quasi-
identifiers for research, or registering a custom surrogate generator (e.g. your
own MRN format).

## Quick start

```python
import openmed

note = "Jane Roe, DOB 1979-04-11, lives in Cambridge MA 02139. SSN 123-45-6789."

# HIPAA Safe Harbor: mask every identifier class.
safe = openmed.deidentify(note, policy="hipaa_safe_harbor")

# GDPR pseudonymization: replace with fakes AND keep a reversible mapping.
gdpr = openmed.deidentify(note, policy="gdpr_pseudonymization")
mapping = gdpr.mapping          # present because the profile sets keep_mapping=True

# Research limited dataset: mask direct identifiers, KEEP quasi-identifiers
# (dates, age, ZIP, geography) so the data stays analytically useful.
lds = openmed.deidentify(note, policy="research_limited_dataset")
```

## The seven bundled profiles

Each profile lives in `openmed/core/policies/<name>.json`. Summary of what each
actually configures:

| Profile | Default action | Quasi-identifiers | Mapping | Safety sweep | Use case |
| --- | --- | --- | --- | --- | --- |
| `hipaa_safe_harbor` | **mask** all | masked | none | **mandatory** | HIPAA §164.514(b)(2) Safe Harbor — strip all 18 identifier classes |
| `hipaa_expert_review_assist` | **redact** | redacted; clinical concepts **kept** | none | optional | Assist Expert Determination (§164.514(b)(1)); keeps microbiology/clinical terms for a statistician to assess residual risk |
| `gdpr_pseudonymization` | **replace** | replaced; clinical kept | **kept + reversible** | mandatory | GDPR Art. 4(5) pseudonymization — reversible under controlled key |
| `canada_pipeda` | **replace** (IDs **masked**) | replaced | **kept + reversible** | mandatory | PIPEDA-aligned; like GDPR but masks ID_NUM/SSN outright |
| `research_limited_dataset` | mask direct ids | **keeps** dates, age, ZIP, geography, org, job | none | mandatory | HIPAA Limited Data Set (§164.514(e)) — usable for research with a DUA |
| `clinical_minimal_redaction` | mask direct ids | **keeps** quasi-identifiers | none | **optional** | Internal clinical use where readability matters; lighter cascade |
| `strict_no_leak` | **mask everything** | masked; even clinical concepts masked | none | mandatory | Maximum-recall, union arbitration, all cascade tiers — zero-leakage posture |

Key dimensions to reason about:

- **`default_action`** — `mask` (`[NAME]`), `redact`, `replace` (fake value), or
  `keep`. Set per label in the profile's `actions` map.
- **`policy_label_actions`** — coarse action by class:
  `DIRECT_IDENTIFIER` / `QUASI_IDENTIFIER` / `CLINICAL_CONCEPT`. Research and
  minimal-redaction profiles **keep** quasi-identifiers; strict-no-leak masks
  even clinical concepts.
- **`keep_mapping` / `reversible_id`** — only GDPR and PIPEDA produce a
  reversible mapping. Treat that mapping as PHI.
- **`safety_sweep_mandatory`** — deterministic structured-ID sweep (SSN, MRN-
  like, emails) that runs regardless of model confidence. Off only for the two
  "minimal/assist" profiles.
- **`arbitration_mode` / `forced_cascade_tiers`** — `strict_no_leak` uses
  `high_recall_union` across tiers R0–R3 (most aggressive); minimal redaction
  uses only R0–R1.

## Choosing: map regulation → profile

- **Publish or share data with no DUA, US** → `hipaa_safe_harbor`.
- **Statistician will certify low risk (keep clinical signal)** →
  `hipaa_expert_review_assist`, then human Expert Determination.
- **EU subjects, need reversibility under a key** → `gdpr_pseudonymization`.
- **Canadian subjects** → `canada_pipeda`.
- **Research cohort needing dates/age/geography** → `research_limited_dataset`
  (requires a Data Use Agreement).
- **Internal clinical workflow, readability first** → `clinical_minimal_redaction`.
- **Adversarial / zero-tolerance leakage** → `strict_no_leak`.

## Customizing beyond the bundle

When a profile is close but not exact, drive the engine directly with
`Anonymizer` / `AnonymizerConfig`, or register custom generators.

```python
from openmed import (
    Anonymizer, AnonymizerConfig,
    register_label_generator, register_clinical_provider,
)

# 1) Per-instance config (language, locale, deterministic surrogates):
anon = Anonymizer(AnonymizerConfig(lang="en", consistent=True, seed=7))
fake_name = anon.surrogate("John Doe", "PERSON")     # type-matched surrogate

# 2) Override the surrogate for one canonical label (e.g. your MRN format).
#    Generators take (faker, original, *, locale) and return a string.
def hospital_mrn(faker, original, *, locale):
    return f"H{faker.numerify('#######')}"

register_label_generator("ID_NUM", hospital_mrn)     # global, all new Anonymizers

# 3) Add a custom Faker provider (e.g. proprietary identifier formats).
register_clinical_provider(MyClinicalProvider)        # a faker BaseProvider class
```

Use `register_label_generator(canonical_label, fn)` to swap one label's
surrogate; use `register_clinical_provider(provider)` to add whole Faker
providers. For per-call scoping, pass providers via
`AnonymizerConfig.custom_providers` instead of the global registry. Validate
custom labels against `openmed.CANONICAL_LABELS`.

## Hand-off to / from OpenMed

- **Apply a policy:** `openmed.deidentify(text, policy="<name>")` — see
  `deidentifying-clinical-text`.
- **Surrogate strategy:** `generating-synthetic-surrogates` for `method="replace"`
  with `consistent`/`seed`/`locale` and custom providers.
- **Verify coverage:** `auditing-deidentification-runs` (`audit=True`) and
  `auditing-safe-harbor-checklist` (18 identifier categories).
- **Other surfaces:** MCP `openmed_deidentify` and REST `POST /pii/deidentify`
  accept the same `policy` argument.

## Edge cases & gotchas

- **Profiles are configuration, not a guarantee.** A profile that *keeps* quasi-
  identifiers (research/minimal) does **not** meet Safe Harbor — pair it with a
  Data Use Agreement or Expert Determination.
- **Reversible profiles produce a re-identifying mapping.** GDPR/PIPEDA mappings
  are as sensitive as the raw PHI; store them encrypted and separately.
- **`register_label_generator` is global and persists for the process.** It
  mutates a shared registry; prefer `AnonymizerConfig.custom_providers` for
  isolated, per-run behavior.
- **Surrogates must not collide with real values.** Keep generated identifiers
  out of the real ID space; see `generating-synthetic-surrogates`.
- **Permissive licensing only.** Do not bundle UMLS/SNOMED/CPT/MIMIC/i2b2/n2c2
  into custom providers; call restricted terminologies out-of-process.

## Standards & references

- HIPAA de-identification, 45 CFR 164.514(b): https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- HIPAA Limited Data Set, 45 CFR 164.514(e).
- GDPR pseudonymization, Regulation (EU) 2016/679 Art. 4(5) & Art. 25.
- PIPEDA (Canada): https://www.priv.gc.ca/en/privacy-topics/privacy-laws-in-canada/the-personal-information-protection-and-electronic-documents-act-pipeda/
- OpenMed source: `openmed/core/policies/*.json`, `openmed/core/anonymizer/`.
