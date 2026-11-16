# Generated Contract Types

OpenMed's record contracts are the versioned JSON Schemas in
[`openmed/core/schemas/json/`](https://github.com/maziyarpanahi/openmed/tree/master/openmed/core/schemas/json).
The SDK types for those records are generated from the same files, so a field
or label added in Python reaches the other platforms through one regeneration
step instead of a hand copy.

## Covered schemas

The generators read eight core schemas, in this order:

| Schema file | Title | Generated record |
| --- | --- | --- |
| `span.schema.json` | `OpenMedSpan` | `Span` |
| `entity.schema.json` | `OpenMedEntity` | `Entity` |
| `action.schema.json` | `OpenMedAction` | `Action` |
| `audit.schema.json` | `OpenMedAuditEvent` | `AuditEvent` |
| `attestation.schema.json` | `OpenMedDataResidencyAttestation` | `DataResidencyAttestation` |
| `relation.schema.json` | `OpenMedRelation` | `Relation` |
| `code.schema.json` | `OpenMedCode` | `Code` |
| `provenance.schema.json` | `OpenMedProvenance` | `Provenance` |

Each platform adds the `OpenMedContract` prefix, so the generated names never
clash with existing public types. Nested objects get their own record named
after the path to them, for example
`OpenMedContractDataResidencyAttestationTemplate`; array items add `Item`.

The canonical label list (`CANONICAL_LABELS`), the policy labels
(`POLICY_LABELS`) and the span actions (`SPAN_ACTIONS`) come from the
`canonical_label`, `policy_label` and `action` enums of `span.schema.json`.
Any other enum with exactly the same values, such as the attestation's
`policy.default_action`, reuses the shared type.

Generation is strict. A construct a generator cannot represent exactly
(`$ref`, `oneOf`, open records with `additionalProperties`, unions of two
non-null types, non-string enums) stops generation with a reason code and the
property path, never a schema value.

## TypeScript

`scripts/codegen/emit_ts_types.py` writes
`js/openmedkit-web/src/generated/contract-types.ts`. The module contains only
interfaces, type aliases and `as const` arrays, so it adds no runtime code
beyond the three label arrays and no dependency.

| JSON Schema | TypeScript |
| --- | --- |
| `string` | `string` |
| `string` with an anchored literal `pattern` prefix | template literal, for example `` `hmac-sha256:${string}` `` |
| `integer`, `number` | `number` |
| `boolean` | `boolean` |
| `const` | the literal value, for example `1` or `true` |
| `enum` | the shared named type, or a union of string literals |
| `["T", "null"]` | the type of `T`, unioned with `null` |
| object with `properties` | a generated `interface` |
| object without `properties` | `Record<string, unknown>` |
| `array` | `T[]` |
| property not in `required` | optional property (`name?: T`) |

The `openmed` web package exports every generated `OpenMedContract*` type, and
its `CANONICAL_LABELS`, `POLICY_LABELS` and `SPAN_ACTIONS` exports are the
generated arrays. The hand-written `OpenMedSpan` stays the public span type;
`types.ts` carries a compile-time guard that it is assignable to and from the
generated `OpenMedContractSpan`, so `npm run typecheck` fails when they
diverge.

```ts
import type { OpenMedContractProvenance } from "openmed";

const provenance: OpenMedContractProvenance = JSON.parse(body);
```

The types describe the wire shape only. They do not validate values at run
time; validate untrusted JSON against the schema before relying on it.

## Regenerating and drift checks

After changing a core schema, regenerate every committed contract type file:

```bash
make contract-types
```

Each generator also has a `--check` mode that rebuilds the output in memory and
exits non-zero when the committed file has drifted. CI runs every
`scripts/codegen/emit_*_types.py` generator with `--check` in the lint job, and
`make contract-types-check` runs the same loop locally:

```bash
python scripts/codegen/emit_ts_types.py --check
python -m pytest tests/unit/release/test_codegen_types.py -q
```

Output is deterministic: no timestamps, and the header records a SHA-256 digest
of the canonicalized source schemas.
