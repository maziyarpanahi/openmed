# Versioned Journey resources

OpenMed exposes one read-only Journey resource contract through Python, REST,
GraphQL, and SQL projections. Every response carries `schema_version`,
`compatibility_policy`, a typed state, policy metadata, and a snapshot digest.
The surfaces never convert `partial`, `unknown`, `conflict`, `unsupported`,
`denied`, or `failure` into an apparent success.

## Resource families

The public contract covers artifacts, jobs, facts, conflicts, journeys,
cohorts, datasets, registries, measures, and trial-review records. Evidence,
current facts, journey events, mappings, cohort runs, and dataset manifests are
also available as read-optimized projections.

Records contain opaque identifiers and allowlisted structured fields. Raw
source text, credentials, vault material, and unrestricted metadata keys are
rejected by the Python contract before a record can enter a public catalog.

## REST

`GET /v1/journey/resources` accepts:

- `resource_type` (required)
- `namespace`, `purpose`, and `role` (policy inputs)
- `attributes`, a bounded comma-separated attribute set
- `consent_state`: `active`, `unknown`, or `withdrawn`
- `export_policy`, which defaults to `metadata_only`
- `first` from 1 through 100
- `after`, an opaque cursor tied to the query and immutable snapshot
- `fields`, a comma-separated minimum-necessary projection

An empty or denied result is a valid typed response with no resources. Invalid,
stale, or query-mismatched cursors fail closed and never restart at page one.
The opaque cursor is bound to the full access context as well as the selected
resource fields. It cannot be replayed with a different tenant namespace,
role, attribute assertion, consent state, or export policy.

## GraphQL

The `journeyResources` query uses the same catalog, policy, field projection,
cursor, and limit implementation as REST. It returns a
`JourneyResourceConnection`; no Journey mutation or subscription type exists.
GraphQL selection sets can further reduce the transport response, while the
`fields` argument controls the underlying minimum-necessary data projection.
The generated Python and TypeScript clients and the six read-only MCP workflow
tools expose the same access-context arguments and typed denials.

## Read-only SQL

[`journey-views.sql`](./journey-views.sql) is generated from the Python view
registry. It defines views for artifacts, facts, evidence, current facts,
journey events, mappings, cohort runs, and dataset manifests. The caller owns
the base table and grants analytics identities `SELECT` on views only.

`validate_journey_analytics_sql` requires a single bounded `SELECT`, rejects
wildcards and mutation or execution primitives, and restricts reads to the
credential's view allowlist. `query_journey_view` runs through the same catalog
and access policy as REST and GraphQL for local parity testing.

## Access-decision evidence

Every response policy block repeats the controlled namespace, purpose, role,
attribute set, consent state, export policy, selected fields, decision state,
reason code, opaque decision ID, request digest, and policy version. This is
sufficient to reconstruct why a read was allowed or denied without recording
resource values. Namespace, role,
missing-attribute, withdrawn or unknown consent, export-policy, and field
failures all return a typed `denied` page with an empty resource list.

## Compatibility and migration

Version `1.x` readers use `same_major` compatibility. Additive extension fields
are preserved during load and migration, but omitted from field-limited public
projections to prevent unselected data from leaking into responses. A different
major version is rejected
instead of being silently downgraded. Committed OpenAPI, GraphQL SDL, and SQL
view snapshots are drift-tested against their generators.

All committed examples and fixtures are synthetic. These interfaces authorize
read-only review and analytics; they do not authorize diagnosis, treatment,
enrollment, outreach, ordering, or another patient-care action.
