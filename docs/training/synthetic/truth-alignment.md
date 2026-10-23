# Synthetic cross-artifact truth alignment

`validate_truth_alignment` compares artifact-declared facts, times, labels and
coordinates against shared `ClinicalFact` and `EvidenceLocator` records from the
existing Journey contracts. `SyntheticArtifactView` groups those records with
an existing `ClinicalArtifact` and a controlled note/FHIR/OMOP/image/waveform
modality; it does not introduce another clinical truth graph.

All records must explicitly declare synthetic origin. Each view declares the
same truth subset; use separate packages for intentionally partial views. The
validator checks medication/value and unit changes, effective times, fact labels,
missing/unexpected facts, evidence references, coordinate transforms and artifact
bindings. It never fabricates missing modalities. Reports use hashed artifact/fact
references and controlled mismatch categories, not submitted values.

Intentional corruptions require a named negative-control fixture, explicit
expected mismatch categories and the `negative_control` partition. Such fixtures
can pass their negative-control check but always remain `valid_for_training=false`.
A normal partition rejects intentional corruption metadata. The validator checks
declared consistency, not clinical plausibility, source-rights adequacy or privacy.
No generator, network, model, or publication is invoked. Synthetic flags and
opaque references are assertions requiring independent provenance review.

Shared fact IDs also retain their declared synthetic subject and encounter bindings. Changing either emits `fact_identity` without including identity values; an intentional identity corruption must use the named negative-control partition.
