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
- `namespace`, `purpose`, and `role`, which select within the caller's grant
- `attributes`, a bounded comma-separated subset of the granted attributes
- `consent_state`: `active`, `unknown`, or `withdrawn`; it can only make the
  effective consent state more restrictive
- `export_policy`, which selects within the granted export postures
- `first` from 1 through 100
- `after`, an opaque cursor tied to the query and immutable snapshot
- `fields`, a comma-separated minimum-necessary projection

An empty or denied result is a valid typed response with no resources. Invalid,
stale, or query-mismatched cursors fail closed and never restart at page one.
The opaque cursor is bound to the full access context as well as the selected
resource fields. It cannot be replayed with a different tenant namespace,
role, attribute set, consent state, or export policy, including by another
principal whose grant binds a different context.

## Principal binding

Request parameters are requests, not authority. The role, attributes,
namespaces, purposes, export postures, and consent state come from the
authenticated principal (API key, JWT, or mTLS) through a
`JourneyAccessContextProvider`. Callers may only narrow that context. An
omitted parameter selects the granted `default`, `care_review`, `clinician`, or
`metadata_only` value when the grant includes it, otherwise the single granted
value; an ambiguous grant returns `<field>_selection_required`. Omitted
attributes mean every granted attribute.

The default provider reads an operator grant document from
`OPENMED_SERVICE_JOURNEY_GRANTS` (inline JSON) or
`OPENMED_SERVICE_JOURNEY_GRANTS_FILE`:

```json
{
  "version": "1.0.0",
  "jwt_claim": null,
  "grants": [
    {
      "credential_type": "api_key",
      "principal": "svc-analytics",
      "roles": ["researcher"],
      "attributes": [],
      "namespaces": ["default"],
      "purposes": ["analytics"],
      "export_policies": ["metadata_only"],
      "consent_state": "active",
      "expires_at": "2027-01-01T00:00:00Z"
    }
  ]
}
```

A grant matches only the exact subject for its `credential_type`, so a JWT
subject cannot borrow an API-key or mTLS grant with the same name. Omitted
`consent_state` is `unknown` and denies. When `jwt_claim` names a claim, a
verified JWT without an operator grant may carry a grant object in that signed
claim; the resulting context never outlives the token's `exp`. Embedders can
pass their own provider, or a `consent_resolver` hook, through
`create_app(journey_access_context_provider=...)`.

Anonymous Journey reads are denied with `anonymous_access_denied` unless
`OPENMED_PROFILE` is `dev` or `test`; the service defaults to `prod`. Under
those development profiles an anonymous caller receives the configured policy
envelope. Denial codes include `access_context_missing`,
`access_context_expired`, `access_context_unavailable`, `consent_unknown`,
`consent_withdrawn`, and `<field>_not_granted` for any widening attempt.

## GraphQL

The `journeyResources` query uses the same principal binding, catalog, policy,
field projection, cursor, and limit implementation as REST, so one principal
receives the same decision on both transports. It returns a
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
resource values. The service also appends a value-free audit record per
decision with the decision ID, request digest, policy and grant versions,
credential type, and a keyed principal digest; raw subjects, token claims, and
consent records are never recorded. Namespace, role,
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
