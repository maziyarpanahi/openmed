# Privacy evidence freshness

OpenMed's privacy release evidence is only eligible while it is current for
the policy that produced it. The freshness gate is a local, deterministic
check; it is not a compliance certification or a clinical decision.

## Define a typed age policy

Each evidence kind gets a `datetime.timedelta` maximum age. Policy versions are
compared exactly. Age limits are intentionally typed so a caller cannot
silently change the unit from days to hours or seconds.

```python
from datetime import timedelta

from openmed.compliance import EvidenceFreshnessPolicy

policy = EvidenceFreshnessPolicy(
    policy_version="privacy-v2",
    age_limits={
        "release": timedelta(days=30),
        "calibration": timedelta(days=7),
    },
)
```

Use the wildcard key `"*"` when every otherwise-unlisted evidence kind shares
one limit. A specific kind always takes precedence over the wildcard.

## Evaluate with an injected clock

Supply an aware UTC `as_of`, `now`, or a callable/object clock. Exactly one is
required; the evaluator never calls `datetime.now()` and makes no network
request. This makes a release decision replayable from the same evidence,
policy, and clock value.

```python
from datetime import datetime, timezone

from openmed.compliance import EvidenceRecord, evaluate_evidence_freshness

as_of = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
evidence = [
    EvidenceRecord(
        evidence_id="release-2026-08-11",
        evidence_type="release",
        generated_at=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
        policy_version="privacy-v2",
    )
]

report = evaluate_evidence_freshness(evidence, policy, as_of=as_of)
if not report.passed:
    raise RuntimeError(report.failure_message())
```

The gate fails closed for an empty input, missing or invalid descriptors,
unknown evidence kinds, naive or malformed timestamps, timestamps in the
future, timestamps older than the typed limit, and policy-version mismatches.
An age exactly equal to the configured limit is current.

## Supersession

`superseded_by` means that an evidence record is no longer eligible. A
replacement can use `supersedes` to identify an earlier opaque reference. If
the earlier record is in the evaluated bundle, it is counted as superseded;
the replacement remains eligible when the earlier artifact is retained in an
external archive.

```python
replacement = EvidenceRecord(
    evidence_id="release-2026-08-12",
    evidence_type="release",
    generated_at=as_of,
    policy_version="privacy-v2",
    supersedes="release-2026-08-11",
)
```

References are compared only as opaque tokens. They are never emitted by the
freshness report.

## Privacy-safe diagnostics

`EvidenceFreshnessReport.to_dict()` and `.to_json()` contain the policy
version, aggregate totals, the number accepted/current, the number rejected,
and deterministic reason counts. They do not include evidence references,
timestamps, payload fields, source text, identifiers, or supersession values.
`assert_evidence_freshness()` raises `EvidenceFreshnessError` with the same
counts-only content.

The report is a technical gate for release evidence freshness. It does not
claim that the underlying evidence is accurate, complete, legally sufficient,
or clinically safe.
