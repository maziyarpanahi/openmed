# Ancestry-isolated synthetic splits

`openmed.training.synthetic.split_planner.plan_synthetic_splits` creates offline
partitions from explicitly synthetic artifact metadata. Supply
`SyntheticSplitRecord` objects containing SHA-256 references for each artifact,
synthetic trajectory, and derivation lineage, optional parent artifact references,
and controlled language or rare-case stratum codes.

A lineage digest identifies shared derivation ancestry, not a generator version:
assigning the same digest to all outputs of one generator intentionally places
all those outputs in one group. Parent references must resolve within the supplied
manifest. Duplicate artifact identities, duplicate parent references, self-links,
and ancestry cycles fail before planning. References must be generated for
synthetic artifacts; do not supply hashes of real patient identifiers. Hashing
does not prove anonymization, synthetic origin, or dataset quality.

The planner combines shared trajectory, shared lineage, and parent edges into
indivisible connected groups. All modalities and transitive descendants stay in
one partition. Seed-bound SHA-256 group ordering, sorted keys, and deterministic
ties make membership independent of record and mapping order. Fractions must be
positive, finite, and sum to one; they express target proportions rather than
exact quotas because groups cannot be divided. Keep the input manifest and seed
pinned when reproducing a split; adding records can change membership.

Use `required_strata={"train": ("hi", "rare"), "test": ("hi", "rare")}` to
request coverage. Required strata are assigned with scarcity-first backtracking before
proportion-based allocation. Search is limited to 128 candidate groups per unmet requirement,
128 required partition/stratum pairs and 4,096 search states. Successful search
returns `search_status="complete"`; a fully searched unsatisfied case returns
`"unsatisfied"`. A bounded search that cannot finish returns
`"budget_exhausted"`, which means inconclusive and requires review rather than
proving impossibility. Both unsuccessful outcomes return `unmet_strata`, preserve
group isolation, and refuse `to_manifest()` export. It never invents missing examples
or relaxes ancestry to fill a stratum. The caller can revise the recipe or inspect
the unmet controlled categories before trying again.

A successful manifest contains synthetic markers, opaque artifact assignments,
partition codes and a group count only. Planning performs no file writes,
networking, cross-hospital record matching, or medical plausibility validation.
It does not certify contamination freedom outside the supplied ancestry graph.

```sh
.venv/bin/python -m pytest tests/unit/training/synthetic/test_split_planner.py -q
```
