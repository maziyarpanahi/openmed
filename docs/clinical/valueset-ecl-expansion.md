# FHIR ValueSet and ECL expansion

OpenMed can expand an extensional FHIR `ValueSet` locally or delegate a
canonical ValueSet URL or Expression Constraint Language (ECL) expression to a
terminology server that you supply. No vocabulary or endpoint is configured by
default, and importing or using the local path does not create a network
client.

The result is a `ValueSetExpansion`: an immutable set of member codes with
system-aware `codings` and deterministic `provenance`. The provenance records
the ValueSet or response version, the expansion method, vocabulary release
pins, and request/response SHA-256 digests. It does not copy raw ECL into
reports or cache metadata.

## Expand a local ValueSet

Pass a FHIR JSON mapping, inline JSON, or local path. Whole-code-system and
code/display filter clauses require a caller-loaded free vocabulary. Explicit
`compose.include.concept` entries and complete `expansion.contains` entries can
be expanded directly.

```python
from openmed.clinical.grounding import (
    TerminologySnapshot,
    VocabConcept,
    VocabularyIndex,
    expand_valueset,
)

system = "http://human-phenotype-ontology.org"
index = VocabularyIndex(
    "hpo",
    [
        VocabConcept("hpo", "HP:0001250", "Synthetic example one"),
        VocabConcept("hpo", "HP:0001263", "Synthetic example two"),
    ],
)
snapshot = TerminologySnapshot(
    index=index,
    system_uri=system,
    release_version="2026.09",
    content_hash=index.content_hash,
)
value_set = {
    "resourceType": "ValueSet",
    "url": "https://example.org/fhir/ValueSet/synthetic-example",
    "version": "1.0.0",
    "compose": {"include": [{"system": system, "version": "2026.09"}]},
}

expanded = expand_valueset(value_set, vocabularies={system: snapshot})
assert expanded.members == frozenset({"HP:0001250", "HP:0001263"})
assert expanded.version == "1.0.0"
```

Local support is deliberately extensional. Nested ValueSet references,
hierarchy operators such as `is-a` and `descendent-of`, and terminology-specific
properties raise `ValueSetExpansionUnsupportedError` instead of guessing. Send
those definitions to a terminology server explicitly.

## Delegate FHIR `$expand` or ECL

Supplying `endpoint` opts into an out-of-process request. A canonical URL calls
`ValueSet/$expand` with FHIR's `url` parameter; any other non-path string is
syntax-checked as ECL and sent as a SNOMED CT implicit ValueSet URL of the form
`http://snomed.info/sct?fhir_vs=ecl/...`. Canonicals may carry the standard
`|version` suffix. For ECL, `version` is sent as `system-version`; pass
`ecl_system_uri` when the endpoint uses a specific SNOMED edition base URI.

```python
from openmed.clinical.grounding import expand_valueset

expanded = expand_valueset(
    "https://example.org/fhir/ValueSet/findings|2026.09",
    endpoint="https://terminology.example/fhir",
    bearer_token="caller-owned-secret",
)

descendants = expand_valueset(
    "*",  # Replace with an ECL expression licensed for your endpoint.
    version="2026.09",
    endpoint="https://terminology.example/fhir",
    bearer_token="caller-owned-secret",
)
```

Remote expansion sends the canonical URL or raw ECL because the terminology
server must evaluate it. Do not put patient text in either input. Credentials
are caller supplied, excluded from representations and errors, and never
cached. Redirects are rejected by the standard-library transport. Responses
must be FHIR ValueSets with `expansion`; `OperationOutcome`, incomplete local
pages, inconsistent remote pages, and configured size/page limit violations
fail closed.

ECL commonly targets SNOMED CT or another licensed terminology. OpenMed does
not ship, install, license, or silently fall back to such content. Operate the
endpoint and obtain terminology rights separately.

## Opt-in expansion cache

There is no default expansion cache. Provide a path or reuse a configured
`TerminologySnapshotCache` only when persistence is appropriate:

```python
from openmed.clinical.grounding import (
    ValueSetExpansionCache,
    ValueSetExpansionEngine,
)

cache = ValueSetExpansionCache("/caller/controlled/terminology-cache")
engine = ValueSetExpansionEngine(
    vocabularies={system: snapshot},
    cache=cache,
)
expanded = engine.expand_valueset(value_set)
```

Entries live in the snapshot root's `expansions/` namespace and are keyed by a
hash of source kind, canonical URL or ECL, and resolved version. Artifacts
contain only system/version/code members and raw-query-free provenance; they
do not contain displays, credentials, or ECL. Digests are verified on read.

Restricted member codes are not written with the default policy. Persist them
only with a cache you explicitly control and an additional policy opt-in:

```python
restricted_cache = ValueSetExpansionCache(
    "/caller/controlled/restricted-cache",
    allow_restricted=True,
)
```

Without that opt-in, remote restricted results remain in memory and repeated
calls contact the configured endpoint again. Cache lifecycle, access controls,
encryption, licensing, and deletion remain the caller's responsibility.

## Result contract

- `members`, `codes`, and `member_codes` are the immutable code-only set.
- `codings` is a stable tuple of `ValueSetMember(system, code, version)`.
- `version` is the requested or server-declared version. If neither exists,
  OpenMed uses a deterministic SHA-256 content stamp.
- `provenance.cache_hit` distinguishes verified cache reads from fresh work.
- `provenance.request_sha256` binds the input without exposing raw ECL.
- `provenance.response_sha256` binds sorted members and their version.

The engine constrains terminology for grounding and export workflows; it does
not make a clinical decision and does not replace terminology governance or
license review.

Local cache identities bind both the supplied ValueSet definition and its
resolved members; changing a loaded vocabulary or definition cannot reuse a
stale entry under the same URL and version. Remote cache identities also bind
the configured endpoint and ECL system. ValueSet versions remain independent
of code-system version pins. Remote pages must match their requested offsets
and declared totals.
