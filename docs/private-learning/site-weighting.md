# Transparent coarse site weighting

These policies operate on anonymous **positions**, declared coarse volume classes
and controlled protected-slice codes. They accept no site IDs, names, raw record
or patient counts, endpoints, local metrics or training examples. The caller
keeps any binding from positions to real participants private.

```python
from fractions import Fraction
from openmed.training.federated.site_weighting import (
    CoarseSiteMetadata, SiteWeightingPolicy, compute_site_weights,
    render_site_weighting_comparison,
)

groups = [CoarseSiteMetadata('very_large'),
          CoarseSiteMetadata('tiny', ('low_resource',))]
policy = SiteWeightingPolicy(
    'inverse_volume', caps=(Fraction(7, 10), Fraction(7, 10)),
    protected_minimums={'low_resource': Fraction(7, 10)},
)
assert compute_site_weights(groups, policy) == (Fraction(3, 10), Fraction(7, 10))
print(render_site_weighting_comparison(groups, policy))
```

The explicit strategies are `uniform`, `volume` and `inverse_volume`. Coarse
classes `tiny`, `small`, `medium`, `large`, `very_large` map to relative factors
1, 2, 4, 8, 16; they are declarations, not exact counts or measured site size.
Uniform scores are one, volume scores are those factors and inverse scores are
their reciprocals. No strategy is selected automatically.

Per-position floors and caps must allow weights summing to one. The allocator
starts at declared floors and distributes the remaining mass proportionally to
strategy scores. Capped positions saturate, and residual mass is redistributed
among remaining positions. Every cap/floor and normalization calculation uses
exact rational arithmetic. At most 128 positions, 16 slice codes per position
and 16 protected constraints are accepted; input ratio denominators are bounded
to one billion. One capped position retires per bounded allocation iteration.

Protected constraints require the sum over each declared slice to reach its
minimum. Overlapping slices are checked independently. A selected policy that
violates a constraint rejects export weights; it does **not** silently relax a
minimum or claim that every possible policy is infeasible. Operators can choose
another declared strategy or floors. This is a transparent policy gate, not a
clinical fairness optimizer or a measured quality guarantee.

The deterministic counterfactual report compares accepted selected weights with
unconstrained uniform and coarse-volume baselines, including exact selected-minus-
baseline deltas and protected coverage. Fractions serialize as numerator and
denominator pairs, and governing floors/caps are included. Failed protected
constraints produce `hold`, suppress selected export weights and deltas, and
leave clearly labeled counterfactual baselines for review. No training, secure
aggregation, network transport or autonomous clinical action is performed.
