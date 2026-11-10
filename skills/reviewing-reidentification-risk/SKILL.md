---
name: reviewing-reidentification-risk
description: "Run expert-determination-style quasi-identifier risk scoring (k-anonymity, l-diversity) plus OpenMed's empirical re-identification attack on a de-identified dataset, then document residual risk in a defensible memo. Use when the user needs HIPAA Expert Determination (45 CFR 164.514(b)(1)) support, asks whether a dataset is safe to release, worries about singling-out via age/ZIP/dates, or wants a statistical \"very small risk\" determination. Covers identifying quasi-identifiers, computing k-anonymity / l-diversity, running openmed.eval.attacks.reid (run_reid_attack / run_reid_benchmark) as the adversarial attack, and writing the risk memo. Pairs after deidentifying-clinical-text and auditing-deid-leakage."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: after
  version: "1.0"
---

# Reviewing re-identification risk

Removing direct identifiers is not enough. A record stripped of name, SSN, and
MRN can still be **singled out** by a combination of *quasi-identifiers* — age,
ZIP/region, admission date, sex, rare diagnosis. The HIPAA Expert Determination
pathway (45 CFR 164.514(b)(1)) requires a qualified person to apply statistical
methods and document that the risk of re-identification is **"very small."**
This skill produces that evidence: quasi-identifier risk metrics (k-anonymity,
l-diversity) plus OpenMed's empirical re-identification attack, written up as a
residual-risk memo.

## When to use

- After direct-identifier removal passes `auditing-deid-leakage` (no leaks) and
  you must decide whether the dataset is releasable.
- The user invokes Expert Determination, asks for a re-identification risk score,
  k-anonymity, l-diversity, or a "very small risk" determination memo.
- You need an adversarial linkage attack — modeling an attacker with auxiliary
  data — not just a structural metric.

## Quick start

```python
from openmed.eval.attacks.reid import run_reid_attack, run_reid_benchmark

# Synthetic de-identified records; each row is the released, de-id'd data.
deidentified = [
    {"record_id": "r1", "text": "[NAME], 47F, ZIP 021xx, admitted 2024-03."},
    {"record_id": "r2", "text": "[NAME], 47F, ZIP 021xx, admitted 2024-03."},
    {"record_id": "r3", "text": "[NAME], 88M, ZIP 597xx, admitted 2024-03."},  # singleton
]
# Auxiliary = what an attacker might already hold (e.g. a voter list).
auxiliary = [{"record_id": "v9", "text": "88M ZIP 597xx"}]

result = run_reid_attack(
    fixtures=[],                          # bring your own records below
    deidentified_records=deidentified,
    auxiliary_records=auxiliary,
)
metric = result.to_metric()
print(metric["aux_linkage_rate"],        # empirical linkage success
      metric["k_min"],                   # smallest equivalence-class size
      metric["singleton_count"],         # k=1 records (uniquely identifiable)
      metric["quasi_identifier_count"])
```

`k_min` is the population k-anonymity floor across the dataset; a `k_min` of 1
means at least one record is unique on its quasi-identifiers and is the highest
re-identification risk. `aux_linkage_rate` is the empirical attack: how often the
adversary's auxiliary data successfully links back to a released record.

To run against the bundled golden suite and emit a leaderboard-style report:

```python
report = run_reid_benchmark(
    suite="golden",
    deidentified_records=deidentified,
    auxiliary_records=auxiliary,
    output_markdown="reid_risk.md",
)
```

## Workflow

1. **Enumerate quasi-identifiers (QIs).** List every field an outsider could
   plausibly know and cross-reference: age/DOB, ZIP/region, dates of service,
   sex, race, rare conditions, provider. Direct identifiers should already be
   gone (verified by `auditing-deid-leakage`); QIs are what's left to worry about.
2. **Compute k-anonymity.** For each equivalence class (records sharing the same
   QI combination), the class size is *k*. `run_reid_attack` returns `k_min` and
   the list of `singleton_records` (k=1). A common Expert Determination target is
   k ≥ a documented threshold (e.g. k ≥ 5 or k ≥ 11) for every record.
3. **Check l-diversity on sensitive attributes.** k-anonymity hides *which*
   record, but if every record in a class shares the same sensitive value (e.g.
   all HIV-positive), the attribute leaks anyway. Require ≥ l distinct sensitive
   values per class; flag homogeneous classes.
4. **Run the empirical attack.** `run_reid_attack` / `run_reid_benchmark` model an
   adversary with `auxiliary_records` and measure actual linkage success
   (`aux_linkage_rate`), residual leakage (`leakage_rate`), surrogate-consistency
   leaks, and date-shift-inversion leaks. Structural metrics bound risk;
   the attack demonstrates it.
5. **Generalize or suppress, then re-score.** For singletons / low-k classes,
   coarsen QIs (age → age band, ZIP5 → ZIP3, exact date → month/quarter) or
   suppress the record, then re-run until `k_min` and linkage rate meet your
   documented threshold.
6. **Write the determination memo.** Record the QIs considered, methods applied,
   `k_min`, l-diversity, the attack's `aux_linkage_rate`, the assumptions about
   attacker capability, and the conclusion that residual risk is "very small."
   Cite the metrics — never paste raw records into the memo.

## Hand-off to / from OpenMed

- **From** `auditing-deid-leakage`: only score QI risk once direct-identifier
  leakage is zero. A leak short-circuits the whole determination.
- **OpenMed calls:** `from openmed.eval.attacks.reid import run_reid_attack,
  run_reid_benchmark, generate_reid_leaderboard`. The attack delegates to
  `openmed.risk.risk_report` for k-anonymity / linkage internals.
- **To** `evaluating-with-leakage-gates`: register `reid_leakage_rate` as a gate
  in the eval harness so re-identification risk regressions fail CI.
- **From** `pseudonymizing-for-gdpr`: pseudonymized output is still re-identifiable
  via QIs — run this attack before claiming a dataset is low-risk or anonymized.

## Edge cases & gotchas

- **Expert Determination is a human judgment.** OpenMed produces the statistics;
  a qualified expert signs the determination. The tool supports the memo, it is
  not the memo.
- **Auxiliary data assumptions drive the result.** Linkage rate is only as
  meaningful as the `auxiliary_records` you model. Document the assumed attacker
  (motivated insider vs. public voter list) — different aux sets, different risk.
- **Singletons are the headline.** A single k=1 record can sink a release; check
  `singleton_count` and `singleton_records` first.
- **Date-shift can be inverted.** Preserving intervals across a date shift lets an
  attacker re-anchor the timeline; the attack flags `date_shift_inversion_rate`.
  Watch it when de-id used `method="shift_dates"`.
- **No raw records in artifacts.** Reports carry counts, rates, and offsets. Keep
  the underlying dataset out of the memo and out of logs.
- **Local-first.** Run the attack on-device; never ship candidate-release data to
  a third party to "test" re-identifiability.

## Standards & references

- HIPAA Expert Determination — 45 CFR 164.514(b)(1):
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/#expert-determination
- Sweeney, *k-anonymity: a model for protecting privacy* (2002):
  https://dataprivacylab.org/dataprivacy/projects/kanonymity/
- Machanavajjhala et al., *l-diversity* (2007):
  https://dl.acm.org/doi/10.1145/1217299.1217302
- NIST SP 800-188, *De-Identifying Government Data Sets*:
  https://csrc.nist.gov/pubs/sp/800/188/final
