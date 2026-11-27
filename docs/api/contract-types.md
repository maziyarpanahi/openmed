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

## Swift

`scripts/codegen/emit_swift_types.py` writes
`swift/OpenMedKit/Sources/OpenMedKit/Generated/ContractTypes.swift`. Every
record is a public `Codable, Equatable, Sendable` struct with `let`
properties, `camelCase` names and explicit `CodingKeys` for the snake_case
JSON keys. The shared label, policy and action enums are `String`-backed,
`CaseIterable` enums (`OpenMedContractCanonicalLabel`,
`OpenMedContractPolicyLabel`, `OpenMedContractSpanAction`); other enums are
named after their record and property, for example
`OpenMedContractDataResidencyAttestationModelsItemKind`.

| JSON Schema | Swift |
| --- | --- |
| `string` | `String` |
| `integer` | `Int` |
| `number` | `Double` |
| `boolean` | `Bool` |
| `enum` | a generated `String`-backed enum |
| `["T", "null"]` | `T?` |
| object with `properties` | a generated struct |
| object without `properties` | `[String: OpenMedContractJSONValue]` |
| `array` | `[T]` |
| property not in `required` | `T?`, omitted when `nil` |

`OpenMedContractJSONValue` keeps integers, other numbers, booleans, strings,
arrays, objects and `null` distinct, so free-form `evidence`, `metadata` and
`parameters` objects survive a decode and encode round trip unchanged.

Decoding is exact and fails closed:

- A missing required key fails with `DecodingError.keyNotFound`, even when its
  value may be `null`. An explicit `null` is encoded back as `null`.
- A key the schema does not declare fails, and the error names the record
  type, not the key or its value.
- An unknown enum value fails with a message naming the enum type only.
- `const` values such as `schema_version` are checked on decode and filled in
  by the initializer, which does not take them as parameters.

```swift
let span = try JSONDecoder().decode(OpenMedContractSpan.self, from: data)
if span.policyLabel == .directIdentifier {
    // ...
}
```

These types are a wire format for apps that receive OpenMed JSON, for example
from a local OpenMed service. They do not replace the existing OpenMedKit
result types such as `DeidentificationResult` or `EntityPrediction`.

## Kotlin

`scripts/codegen/emit_kotlin_types.py` writes
`android/openmedkit/src/main/kotlin/com/openmed/openmedkit/contract/ContractTypes.kt`.
The generated types live in their own package, `com.openmed.openmedkit.contract`,
so the existing `OpenMedSpan`, `EntityPrediction` and `DeidentificationResult`
types are untouched. Every record is a `data class` with `camelCase`
properties, a `toJson()` method and `fromJson(JsonObject)` and
`fromJson(String)` companion functions built on kotlinx `JsonObject`, in the
same manual-parsing style as `ModelCatalog`. The build applies no serialization
compiler plugin, and the generated code needs none.

| JSON Schema | Kotlin |
| --- | --- |
| `string` | `String` |
| `integer` | `Int` |
| `number` | `Double` |
| `boolean` | `Boolean` |
| `enum` | a generated `enum class` with a `value` property and `fromValue` |
| `["T", "null"]` | `T?` |
| object with `properties` | a generated `data class` |
| object without `properties` | `JsonObject` |
| `array` | `List<T>` |
| property not in `required` | `T? = null`, omitted when `null` |

Decoding fails closed with `OpenMedContractException`, an
`IllegalArgumentException` whose message holds the record, the schema key and
a reason code, never a value:

| Failure | Message |
| --- | --- |
| A required key is missing, even if its value may be `null` | `OpenMedContractSpan.score: missing_required_key` |
| A value has the wrong JSON type, or an integer has a fraction | `OpenMedContractSpan.start: expected_integer` |
| An enum value is not in the schema | `OpenMedContractSpan.canonical_label: unknown_enum_value` |
| A `const` value differs | `OpenMedContractSpan.schema_version: constant_mismatch` |
| A key is not declared by the schema | `OpenMedContractSpan has an undeclared property` |
| The text is not JSON | `OpenMedContractSpan: malformed_json` |

`const` values such as `schema_version` are read-only properties, not
constructor parameters, and an explicit `null` is encoded back as `JsonNull`.

```kotlin
val span = OpenMedContractSpan.fromJson(body)
if (span.policyLabel == OpenMedContractPolicyLabel.DIRECT_IDENTIFIER) {
    // ...
}
val json: JsonObject = span.toJson()
```

`ContractTypesTest` decodes and re-encodes every record in
`tests/fixtures/contracts/core_contract_examples.json` and every span of the
Swift-Kotlin parity fixture `fixtures/parity/expected_spans.json` (turned into
an `OpenMedContractSpan` record) and requires equivalent JSON back.

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
python scripts/codegen/emit_swift_types.py --check
python scripts/codegen/emit_kotlin_types.py --check
python -m pytest tests/unit/release/test_codegen_types.py -q
python -m pytest tests/unit/release/test_codegen_swift_types.py -q
cd swift/OpenMedKit && swift test --filter ContractTypesTests
python -m pytest tests/unit/release/test_codegen_kotlin_types.py -q
cd android && ./gradlew :openmedkit:testDebugUnitTest --tests '*ContractTypesTest'
```

The generated Swift file is already in `swift format` style, so
`make format-swift` leaves it unchanged.

## Shared examples

`tests/fixtures/contracts/core_contract_examples.json` holds synthetic example
records for every core schema. The Python tests validate them against the
schemas, the TypeScript spec type-checks them against the generated
interfaces, and `ContractTypesTests` decodes and re-encodes every one of them
with the generated Swift types. The Swift test also turns each span of the
Swift-Kotlin parity fixture `fixtures/parity/expected_spans.json` into an
`OpenMedContractSpan` record and requires the same lossless round trip.

New generated Kotlin API also needs `./gradlew :openmedkit:apiDump` so
`android/openmedkit/api/openmedkit.api` records it.

Output is deterministic: no timestamps, and the header records a SHA-256 digest
of the canonicalized source schemas.
