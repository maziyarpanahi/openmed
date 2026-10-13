# Secure aggregation count preflight

Before enrollment, callers can check whether declared anonymous round counts
are internally consistent. This module does not enroll clients, create shares,
load keys, perform secure aggregation, choose production thresholds or certify
privacy. A declared dropout bound is an assumption, not a measured guarantee.

```python
from openmed.training.secure_aggregation_preflight import secure_aggregation_preflight

report = secure_aggregation_preflight(
    participant_count=10, quorum=6, share_threshold=4, dropout_bound=2,
    minimum_aggregation_group=3,
)
assert report.outcome == "accept"
assert report.worst_case_survivors == 8
print(report.to_json())
```

Inputs are anonymous declared round participation, surviving-participant quorum,
recovery-share threshold, tolerated dropout count and minimum aggregation group.
Every value must be an integer in `0..2**31-1`; booleans and non-integers reject.
Participation, quorum and recovery threshold must be positive. The minimum group
must be at least two, matching the existing OpenMed aggregate count-release
contract, but this check does not recommend a production group size.

The worst-case survivor count is participation minus the declared dropout bound.
It must cover quorum, recovery threshold and the declared minimum group. Recovery
threshold and minimum group cannot exceed quorum: reaching quorum must not imply
that recovery or the declared release group is impossible. Each requirement plus
dropout is checked for signed 32-bit overflow **before** addition.

- `accept`: declared relationships hold with positive survivor headroom.
- `review`, reason `zero_headroom`: one declared requirement is reached exactly;
  an operator should review assumptions before enrollment. This does not grant
  permission to enroll or automatically change a threshold.
- `reject`: malformed count, input or arithmetic overflow, dropout exceeding
  participation, under-quorum, unrecoverable dropout, unsafe-small-group, recovery
  threshold exceeding quorum, or minimum group exceeding quorum.

Reports contain only schema version, outcome, controlled reasons and an optional
anonymous worst-case survivor count. Malformed or overflowed inputs are never
copied into the report. JSON is deterministic; the immutable report validates
its fields even when constructed directly. No participant IDs, endpoints,
shares, keys, tensors, clinical examples or network calls are accepted.
