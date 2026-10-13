# Synthetic non-IID federation partitions

`simulate_non_iid_partitions` partitions explicitly synthetic fixture metadata
into controlled institution archetypes. Each profile declares an exact sample
quota and language, label-prevalence and specialty weights. Missing weights default
to one; zero excludes the category. Integer weights are preferences rather than
promised proportions. Reports record achieved language/label/specialty counts,
quota membership, unassigned fixtures, the seed and input-manifest digest.

Seed-bound weighted exponential ranks and deterministic augmenting paths assign
without replacement. Augmentation can relocate an earlier choice to preserve
feasible constrained quotas rather than letting a flexible profile consume a
restricted profile's only examples. Input ordering, profile ordering and mapping
ordering do not change results. Duplicate fixture identities and impossible
quotas fail; no source record, institution identity or patient count is simulated.

Inputs contain SHA-256 fixture references and approved controlled categories,
not clinical records. Synthetic provenance must be established separately;
reference hashing is not anonymization. A profile is an invented archetype, not
an actual site. Sample-count statistics describe synthetic fixtures only.

The simulator bounds fixtures, profiles, weights and augmenting-path work. It
does not train models, execute federation, simulate real institutions, or prove
convergence/fairness. Use the recorded exact partition statistics and independent
training/evaluation evidence for experiments. No mandatory network call occurs.
