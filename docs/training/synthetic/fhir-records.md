# Synthetic FHIR records

`openmed.training.synthetic.fhir_records.generate_fhir_records` consumes existing
`ClinicalFact` graph objects explicitly marked synthetic. It emits an offline R4
**collection** Bundle containing Patient, Encounter, Condition,
MedicationStatement, Observation, Procedure and source-bound Provenance fixtures.
No transaction requests, server calls, write-back, terminology download or patient
identifiers are involved. Local placeholder coding uses `urn:openmed:synthetic-facts`.

Facts must contain only a controlled `synthetic_code` value, valid effective and
observed dates, matching subject/encounter references, and existing causal
parents. Missing encounters reject rather than being fabricated. Facts are sorted
before assembly; patient/resource identities and Bundle full URLs are stable for
an identical graph and seed. The existing R4 Bundle assembler handles reference
rewriting. Every fact gets source provenance linking its opaque runtime identity.
Encounter class AMB is a declared synthetic fixture default; clinical statuses
are unknown where appropriate, not inferred from model confidence.

`validate_synthetic_fhir` independently checks referential closure, unique full
URLs, required fields/types in OpenMed's bundled R4 subset, provenance shape,
synthetic tags, time order and absence of write requests. Patient/Provenance
receive fixture checks because the bundled base validator warns for unsupported
resource types. Each encounter/clinical resource requires a distinct source-fact
provenance binding; subject references must resolve to Patient and encounter
references to Encounter with the same subject. Recorded provenance timestamps
must be timezone-aware and cannot precede their target's declared effective time.
For this fixture subset, date-only values represent UTC midnight when compared
with timezone-aware timestamp instants. This convention does not infer actual
clinical time zones or recover event times that were never declared. This is **not** full FHIR conformance, an implementation-guide
certification or clinical validity. Reference-server compatibility remains the
separate conformance workstream. The graph is capped at 1,000 facts and the
validator at 3,000 entries. All data must be synthetic, with no restricted
terminology or real-record import disguised as a fixture.
