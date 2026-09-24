# Evidence-bound drug-safety signals

OpenMed can normalize caller-supplied public adverse-event rows and calculate
descriptive report-level disproportionality signals. These outputs are
hypothesis-generating only. They do not establish causality, estimate
incidence, or direct diagnosis, treatment, prescribing, or outreach.

No event dataset is bundled or downloaded automatically. The caller chooses an
appropriately licensed public source, supplies the file, records its version
and license, and pins its SHA-256 digest.

## Import a pinned local dataset

The generic CSV adapter expects one drug-event pair per row and groups rows by
the source report identifier. That identifier is used transiently for grouping
and is replaced by a derived opaque case identifier in the normalized dataset.
Persisted row evidence contains digests, never the source report identifier.

```python
from pathlib import Path

from openmed.clinical.drug_safety import import_open_event_csv
from openmed.clinical.journey_contracts import sha256_digest

path = Path("public-events-2026-09.csv")
source_digest = sha256_digest(path.read_bytes())

dataset = import_open_event_csv(
    path,
    dataset_id="public_event_export",
    version="2026-09",
    source_digest=source_digest,
    license_id="caller-verified-license",
)
```

The adapter normalizes drug and event text, seriousness, and optional exposure
day windows. Exact duplicate rows are counted and excluded. Conflicting
seriousness values within one source report fail closed. The normalized dataset
retains its source digest, adapter version, imported-row count, duplicate-row
count, and deterministic dataset digest.

## Calculate a descriptive signal

```python
from openmed.clinical.drug_safety import (
    SafetySeriousness,
    SignalFilter,
    SignalPolicy,
    compute_descriptive_signal,
)

signal = compute_descriptive_signal(
    dataset,
    drug="example drug",
    event="example event",
    signal_filter=SignalFilter(
        seriousness=(SafetySeriousness.SERIOUS,),
    ),
    policy=SignalPolicy(
        minimum_pair_count=3,
        minimum_cell_count=1,
    ),
)
```

The calculation builds a report-level 2x2 table and, when every denominator and
minimum-count rule is satisfied, reports the proportional reporting ratio
(PRR) and reporting odds ratio (ROR). It does not apply continuity corrections
or turn missing data into evidence.

Every result has one of three states:

- `computed`: both descriptive ratios are present;
- `suppressed`: a pair, cell, or zero-cell policy prevents ratio release; or
- `insufficient_data`: the filtered population or a required denominator is
  absent.

Non-computed results contain controlled reason codes and null ratios. Each
artifact carries the exact source and license, normalized dataset, versioned
filter, policy, and adapter provenance, plus the pinned
`report_level_prr_ror` method and its version. It also carries explicit caveats
for reporting bias, residual duplicates, confounding, non-causality, and
non-incidence.

## Keep patient suspicions separate

`SuspectedDrugEventRelation` represents a chart-level temporal suspicion using
only a named Journey snapshot and opaque fact/evidence identifiers. It always
requires review and cannot carry PRR, ROR, population counts, or a causal claim.
`DescriptiveSignal` is the distinct population-level type. The two artifacts
cannot be substituted for one another. Serialized suspected relations also
carry a schema version, compatibility policy, advisory, and complete artifact
digest.

## Evaluation and compatibility

`run_drug_safety_benchmark()` compares frozen synthetic or otherwise permitted
cases with hand-computed 2x2 tables, states, PRR, and ROR. Its report includes
the fixture, normalized dataset, suppression policy, adapter, and suite
versions.

Serialized descriptive signals use schema version `1.0.0` with `same_major`
compatibility and validate against the bundled
`drug_safety_signal.schema.json`. `DescriptiveSignal.from_dict()` verifies the
filter, policy, identifier, and complete artifact digests before accepting a
persisted result.
