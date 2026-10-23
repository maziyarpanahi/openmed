# Synthetic longitudinal trajectories

`openmed.training.synthetic.longitudinal` creates deterministic multi-encounter
synthetic fact graphs and observable notes using existing `ClinicalFact`,
`ClinicalArtifact` and `EvidenceLocator` Journey contracts. It introduces no
alternate runtime fact schema, clinical decision or private dataset dependency.

Declare bounded `TrajectoryEvent` items with controlled event and encounter codes,
effective and observed relative days, and explicit parent event codes. Encounters
must exist, temporal and observed order must permit each parent, and same-day
causal cycles reject. Delayed results preserve their effective day separately
from when the observation becomes available. No clinical progression is inferred;
causality and fictional event codes come from the declared synthetic recipe.

`generate_trajectory(events, seed=..., start_date=..., date_shift_days=...)` produces
a stable patient identity, runtime contract objects, exact note evidence spans,
content hashes and synthetic provenance. Uniform date shifting retains patient /
fact identities and parent relationships while moving dates, preserving intervals.
Only 1,000 events and 128 parents per event are accepted. No model/network/file
write is implicit. `validate_trajectory` independently checks the runtime graph,
causal time, subject/provenance, observable hashes and evidence bindings.

The `report()` projection contains counts, digests/opaque references and controlled
metadata, never document text. Fixture notes are fictional placeholders, not
physiological progression, clinical utility, compliance or diagnostic evidence.
Keep all synthetic text and recipes separate from protected evaluation data.
