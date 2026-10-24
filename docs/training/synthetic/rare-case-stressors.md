# Rare-case privacy stressors

Average synthetic populations seldom contain the unusual attribute combinations
that make a record re-identifiable: a centenarian in a hamlet of a few hundred
people, a lighthouse keeper with a rare diagnosis, one of quadruplets with a
29 February birth date. `generate_rare_case_stress_set` builds those
combinations on purpose, from a seed, and labels every record with the direct
identifiers it carries and with how much each quasi-identifier contributes to
its re-identification risk. Privacy attack suites can then be scored against
known ground truth.

```python
from openmed.training.synthetic.rare_case_stress import (
    RareCaseStressConfig,
    generate_rare_case_stress_set,
)

stress_set = generate_rare_case_stress_set(
    30,
    seed=7,
    config=RareCaseStressConfig(background_classes=10),
)
print(stress_set.summary()["stress_singleton_count"])
record = stress_set.records[0]
print(record.profile_id, record.equivalence_class_size, record.dominant_dimension)
```

Generation is deterministic: the same seed and configuration always yield the
same records, note text, offsets, and dataset digest. It uses a local
`random.Random`, reads no files, writes no logs, and makes no network call.

## Dimensions and tiers

Each dimension draws from a small, repository-authored vocabulary split into
`common`, `uncommon`, and `rare` tiers. Every value carries a synthetic
frequency: an order-of-magnitude prior, not a census, registry, or prevalence
statistic. Place names are fictional.

| Dimension | Column | Label | Common | Uncommon | Rare |
| --- | --- | --- | --- | --- | --- |
| `geography` | `region` | `LOCATION` | regions of 640,000+ people | towns of 16,500-48,000 | settlements of 210-1,800 |
| `age` | `age` | `AGE` | 18-64 | 85-89 | 90-109 |
| `date` | `birth_date` | `DATE`, `DATE_OF_BIRTH` | birth year only | year and month | full date, including 29 February |
| `diagnosis` | `diagnosis` | `CONDITION` | for example asthma | for example Crohn disease | for example Gaucher disease |
| `occupation` | `occupation` | `OCCUPATION` | for example schoolteacher | for example pharmacist | for example lighthouse keeper |
| `family_relationship` | `family_relationship` | none | for example lives with spouse | for example identical twin | for example one of quadruplets |

Dates are released at a precision. Their frequency is the share of people of
the same released age who share the released value, so the birth year adds no
information beyond age and age and date are not double counted. Leap-day birth
dates are always consistent with the released age at the configured
`reference_date` (default 2025-06-30); a configuration that cannot place a
29 February birth date inside the drawn age band raises `ValueError`.

`rare_case_vocabulary()` returns every opaque code (for example
`occupation.rare.04`) with its tier and frequency, without the surface values.

## Configuring combinations

- `RareCaseProfile(rare=(...), uncommon=(...))` requests a tier per dimension;
  unnamed dimensions stay common. `RareCaseProfile.from_tiers({...})` accepts a
  `{dimension: tier}` mapping, and `profile_id` gives a stable, value-free name
  such as `rare:occupation;uncommon:age`.
- `default_rare_case_profiles(order)` returns every combination of `order`
  simultaneously rare dimensions. The default configuration uses the 15
  pairwise combinations.
- `RareCaseStressConfig.profiles` is cycled in order across stress records, so
  every configured combination is covered once `count` reaches the number of
  profiles.
- `background_classes` and `background_class_size` add all-common crowd records
  that share one quasi-identifier tuple per class, so stress records are rare
  relative to a k-anonymous background.
- `include_direct_identifiers=False` omits the synthetic name and medical
  record number.

## Risk labels

`record.risk_labels()` is value-free. For each attribute it reports the
dimension, column, canonical label, identifier class, re-identification
category, tier, opaque code, character offsets, and:

- `frequency` and `surprisal_bits` (`-log2(frequency)`), the information the
  released value carries;
- `contribution_share`, the attribute's share of the record's combined
  surprisal;
- `risk_level` (`low`, `medium`, `high` for common, uncommon, rare);
- `safe_harbor_flags`, descriptive flags such as `age_over_89`,
  `date_indicates_age_over_89`, `date_more_specific_than_year`, and
  `small_geographic_unit` (20,000 people or fewer);
- `sensitive_attribute` for diagnoses and `relative_role` for family
  relationships, using the roles from family-history attribution.

Record-level labels add `equivalence_class_size` (the dataset k computed over
the generated set), `dataset_singleton`, `combined_surprisal_bits`,
`expected_population_matches` against a fictional reference population of ten
million, a Poisson `population_unique_probability`, and the
`dominant_dimension`. The combined figures assume independent attributes, a
deliberate simplification.

Identifier classes follow the core label policy: the synthetic name, record
number, and a full birth date are `DIRECT_IDENTIFIER`. Every other dimension
is scored as a `QUASI_IDENTIFIER`, including diagnoses that core label
metadata treats as clinical concepts, because rare conditions enable linkage.
Categories reuse the re-identification profiler vocabulary (`geography`, `age`,
`date`, `rare_condition`) and add `occupation` and `family_relationship`.

## Using the set with privacy attack suites

`rows(include_direct_identifiers=False)` models a release with direct
identifiers removed and quasi-identifiers intact; `quasi_identifier_fields`
names the linkage key.

```python
from openmed.eval.attacks import linkage_attack
from openmed.risk.reid import risk_report

released = stress_set.rows(include_direct_identifiers=False)
fields = stress_set.quasi_identifier_fields

report = risk_report(released, quasi_identifier_fields=fields)
attack = linkage_attack(released, stress_set.rows(), quasi_identifiers=fields)

truth = {r.record_id for r in stress_set.records if r.is_dataset_singleton}
assert {item["record_id"] for item in report["singleton_records"]} == truth
assert attack.unique_matches == len(truth)
```

`to_dicts()` returns training records with note text, offset-aligned `spans`
in the shape OpenMed risk profilers read, typed `fields`, the value-free
`risk` labels, and synthetic provenance metadata.

## Privacy boundary

- Every value is synthetic. No restricted dataset, registry, or real record is
  read or bundled.
- `summary()`, `risk_labels()`, `rare_case_vocabulary()`, configuration
  summaries, and object representations contain codes, tiers, offsets, counts,
  and `sha256:` digests only. Generated values appear only in the records'
  `text`, `fields`, and rows.
- Validation errors never echo the rejected value.
- The labels describe synthetic risk for testing attack suites and
  de-identification pipelines. They are not an Expert Determination, a
  compliance certification, or a clinical statement.
