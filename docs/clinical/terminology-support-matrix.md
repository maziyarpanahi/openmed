# Terminology operation support matrix

This page tells an integrator which FHIR terminology operations OpenMed runs
offline, which it sends to a terminology server that you operate, and which it
does not provide. It describes the current behavior only. Operations that do
not exist yet are marked as not available rather than planned.

OpenMed bundles no terminology content. Everything that runs locally runs over
a ValueSet document or a vocabulary that you load yourself. Everything that
leaves the process needs an endpoint that you supply, and nothing is sent when
none is configured.

## Summary

| FHIR operation | Offline | Delegates to your server | Module |
| --- | --- | --- | --- |
| `$validate-code` | Yes, against a ValueSet you supply | No | `openmed/clinical/exporters/valueset.py` |
| `$expand` | Yes, for extensional ValueSets | Yes, for a canonical URL or ECL | `openmed/clinical/grounding/valueset.py` |
| `$lookup` | No | Yes, `CodeSystem/$lookup` for SNOMED CT concept ids | `openmed/interop/bridges/snomed_terminology_bridge.py` |
| `$translate` | Not available | Not available | none |
| `$subsumes` | Not available | Not available | none |

## `$validate-code`

`validate_code(system, code, valueset)` and `validate_codeable_concept(...)`
check membership against a ValueSet document that you pass in (a mapping,
inline JSON, or a local JSON path). They never contact a server.

| Parameter | Supported | Notes |
| --- | --- | --- |
| `system` | Required | Canonical code-system URI. A blank value returns `valid=False`. |
| `code` | Required | An empty value returns `valid=False`. |
| `valueset` | Required | Mapping, inline JSON object, or local file path. |

What counts as a member:

- an explicit `compose.include.concept` entry with the same code, unless it is
  marked `abstract`;
- a code listed in a complete `expansion.contains`, including nested entries;
- a code that satisfies every `filter` on an include clause (see the filter
  table below).

A whole-code-system include with no concepts and no filter, and any filter
that cannot be evaluated locally, are not treated as membership. The result is
`valid=False` with a diagnostic, so a locally incomplete answer is never a
successful validation. An `expansion` whose `total` is larger than the entries
present, or that has a non-zero `offset`, is treated as incomplete in the same
way.

A `compose.exclude` clause removes a code that its include clause would
otherwise match.

## `$expand`

`expand_valueset(...)` and `ValueSetExpansionEngine` return a
`ValueSetExpansion`. The input decides where the work happens:

| Input | Where it runs |
| --- | --- |
| A ValueSet mapping, inline JSON, or a local path | Offline. The endpoint is never used. |
| An absolute canonical URL, optionally with `\|version` | Your server, through `ValueSet/$expand`. |
| Any other non-path string | Treated as ECL after a syntax check, then sent to your server as a SNOMED CT implicit ValueSet URL. |

Remote input without an `endpoint` raises
`ValueSetExpansionConfigurationError`. There is no fallback from one path to
the other.

Offline expansion accepts:

- explicit `compose.include.concept` entries, with `abstract` concepts skipped;
- a complete `expansion.contains`;
- a whole-code-system include or filters, only when you loaded a free
  vocabulary for that system (`vocabularies=` or a `vocabulary_registry`).
  The free vocabularies are `rxnorm`, `icd10cm`, `loinc`, `hpo` and `mesh`;
- `compose.exclude`, evaluated the same way as the includes.

Offline expansion raises `ValueSetExpansionUnsupportedError` for:

- a nested `valueSet` reference;
- a whole-code-system include or a filter when no vocabulary is loaded for the
  system;
- a filter on any property other than `code`, `concept`, `system` or
  `display`, and a filter operator other than the ones listed below;
- an `expansion` that is paged or incomplete (for example, a `total` larger
  than the entries present).

A system on the restricted list (`snomed` and `umls`) raises
`ValueSetExpansionPolicyError` and must go through your endpoint. See
[FHIR ValueSet and ECL expansion](valueset-ecl-expansion.md) for the result
contract, caching, and examples.

Remote parameters sent to your server:

| Parameter | Sent |
| --- | --- |
| `url` | Always. For ECL it is the implicit ValueSet URL. |
| `valueSetVersion` | For a canonical URL when a version is given. |
| `system-version` | For ECL when a version is given. |
| `count` | Always, from `page_size`. |
| `offset` | On every page after the first. |

Limits you can configure: `page_size`, `max_pages`, `max_members` and
`timeout`. Exceeding the page or member limit raises an error instead of
returning a partial expansion.

## Filter operators

Two code paths evaluate `compose.include.filter` clauses, and they differ.

| Property | Operators accepted | `$validate-code` | Offline `$expand` |
| --- | --- | --- | --- |
| `code`, `concept` | `=`, `==`, `regex`, `exists` | Yes | Yes |
| `system` | `=`, `==`, `regex`, `exists` | Yes | Yes |
| `display` | `=`, `==`, `regex`, `exists` | Only when the same clause has a `concept` entry for the queried code with a `display` string; otherwise the filter cannot be evaluated | Yes, against the loaded vocabulary |
| Any other property, including hierarchy filters such as `is-a` and `descendent-of` | None | Cannot be evaluated (`valid=False` with a diagnostic) | `ValueSetExpansionUnsupportedError` |

`_local_filters_match()` in `openmed/clinical/grounding/valueset.py` is the
function behind the offline `$expand` column. It accepts exactly `=`, `==`,
`regex` and `exists`. Any other operator raises
`ValueSetExpansionUnsupportedError`, and so does an invalid regular
expression. `exists` is true for the values `true`, `1` and `yes` (compared
case-insensitively) and false otherwise.

When a clause has several filters, all of them must match. In
`$validate-code` an unsupported operator or an invalid regular expression makes
that filter unevaluable, which returns `valid=False` with a diagnostic rather
than raising.

## `$lookup`

`SNOMEDTerminologyBridge` sends requests to a caller-operated FHIR terminology
server. It never works offline and keeps no cache.

| Method | Server request | Used for |
| --- | --- | --- |
| `lookup_code(code)` | `GET CodeSystem/$lookup` with `system` and `code` | A numeric SNOMED CT concept id |
| `search(term)` | `GET CodeSystem` with `url` and `filter` | Free text |
| `lookup(term)` and `match(term)` | Either of the above | A numeric concept id of 6 to 18 digits goes to `$lookup`; anything else goes to `search` |

Optional parameters are `displayLanguage` (the `language` argument) and
`_count` (the `limit` argument). A `limit` above 50, or a non-positive one,
raises `ValueError`.

Only the SNOMED CT system is wired to `$lookup`. There is no `$lookup` for any
other code system.

## Not available

- `$translate`: no ConceptMap translation is implemented. For code
  crosswalks see [ICD-10-CM and SNOMED CT crosswalks](code-crosswalk.md).
- `$subsumes`: no subsumption test is implemented, and hierarchy filters are
  not evaluated offline.

## Which path to use

| You have | Use |
| --- | --- |
| A ValueSet document and a code to check | `validate_code`, offline |
| An extensional ValueSet or a loaded free vocabulary | Offline `$expand` |
| A canonical ValueSet URL, a hierarchy filter, or ECL | `$expand` through your endpoint |
| A SNOMED CT concept id to resolve | `SNOMEDTerminologyBridge.lookup_code` through your endpoint |

Remote requests carry the canonical URL, ECL expression, concept id or search
term to your server, so do not put patient text in any of them. Credentials are
supplied by the caller and are never cached. Obtaining the rights to the
terminology behind your endpoint is your responsibility; OpenMed does not ship,
install, or license it.

## Offline example

The codes below are invented and live under a `urn:openmed:example:` system.

```python
from openmed.clinical.exporters import validate_code

system = "urn:openmed:example:findings"
value_set = {
    "resourceType": "ValueSet",
    "url": "urn:openmed:example:valueset:findings",
    "compose": {
        "include": [
            {
                "system": system,
                "concept": [
                    {"code": "F-001", "display": "Example finding one"},
                    {"code": "F-002", "display": "Example finding two"},
                ],
            }
        ]
    },
}

assert validate_code(system, "F-001", value_set).valid
assert not validate_code(system, "F-999", value_set).valid
```
