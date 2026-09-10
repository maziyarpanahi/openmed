# SNOMED CT Terminology Bridge

SNOMED CT is licensed terminology and is not bundled with OpenMed. Use
`SNOMEDTerminologyBridge` only with a FHIR terminology endpoint or Snowstorm
FHIR interface that your organization operates and is licensed to use.

The bridge is deliberately out of process. It does not download, persist, or
cache SNOMED content. A text lookup sends only the supplied span as the FHIR
`CodeSystem` `filter` parameter. An SCTID lookup sends only the supplied code
to `CodeSystem/$lookup`, along with the required SNOMED system URI. No note,
patient, assertion, or surrounding context is added to either request.

```python
from openmed.interop.bridges import SNOMEDTerminologyBridge

bridge = SNOMEDTerminologyBridge(
    endpoint="https://terminology.example/fhir",
    headers={"Authorization": "Bearer <caller-managed-token>"},
)

matches = bridge.lookup("synthetic finding alpha", limit=5)
for match in matches:
    print(match.code, match.display, match.score)
```

The result is a tuple of the same `ConceptMatch` records used by the shared
free-vocabulary matcher. An empty tuple is an ordinary terminology
abstention. Missing endpoint configuration raises
`SNOMEDTerminologyConfigurationError`; the bridge never silently falls back
to bundled or local SNOMED data.

For a local server that does not require authentication, omit `headers`. For
server-specific authentication, use explicit caller-managed headers or the
`bearer_token`, `api_key`, or basic-auth options. Credentials remain in memory
for the bridge lifetime and are excluded from configuration representations.
The terminology server remains responsible for access control, licensing, and
its own audit/logging policy.

Grounding output is assistive and requires qualified review. It must not be
used as an autonomous diagnosis, treatment, billing, or clinical decision.
