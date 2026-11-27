package com.openmed.openmedkit

import com.openmed.openmedkit.contract.OpenMedContractAction
import com.openmed.openmedkit.contract.OpenMedContractAuditEvent
import com.openmed.openmedkit.contract.OpenMedContractCanonicalLabel
import com.openmed.openmedkit.contract.OpenMedContractCode
import com.openmed.openmedkit.contract.OpenMedContractDataResidencyAttestation
import com.openmed.openmedkit.contract.OpenMedContractEntity
import com.openmed.openmedkit.contract.OpenMedContractException
import com.openmed.openmedkit.contract.OpenMedContractPolicyLabel
import com.openmed.openmedkit.contract.OpenMedContractProvenance
import com.openmed.openmedkit.contract.OpenMedContractRelation
import com.openmed.openmedkit.contract.OpenMedContractSpan
import com.openmed.openmedkit.contract.OpenMedContractSpanAction
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Decodes the shared core-contract examples and the Swift-Kotlin span parity
 * fixture with the generated `com.openmed.openmedkit.contract` types from
 * `scripts/codegen/emit_kotlin_types.py`.
 *
 * Every record must decode and encode back to equivalent JSON. Failure
 * messages carry schema names, indices, keys and reason codes only.
 */
class ContractTypesTest {
    private val roundTrips: Map<String, (JsonObject) -> JsonObject> = mapOf(
        "span" to codec({ OpenMedContractSpan.fromJson(it) }, { it.toJson() }),
        "entity" to codec({ OpenMedContractEntity.fromJson(it) }, { it.toJson() }),
        "action" to codec({ OpenMedContractAction.fromJson(it) }, { it.toJson() }),
        "audit" to codec({ OpenMedContractAuditEvent.fromJson(it) }, { it.toJson() }),
        "attestation" to codec(
            { OpenMedContractDataResidencyAttestation.fromJson(it) },
            { it.toJson() },
        ),
        "relation" to codec({ OpenMedContractRelation.fromJson(it) }, { it.toJson() }),
        "code" to codec({ OpenMedContractCode.fromJson(it) }, { it.toJson() }),
        "provenance" to codec({ OpenMedContractProvenance.fromJson(it) }, { it.toJson() }),
    )

    @Test
    fun everySchemaExampleRoundTripsWithoutLoss() {
        val examples = examples()
        assertEquals(roundTrips.keys, examples.keys)

        for ((schema, records) in examples) {
            val roundTrip = roundTrips.getValue(schema)
            assertTrue(records.isNotEmpty(), schema)
            records.forEachIndexed { index, record ->
                assertTrue(
                    equivalent(record, roundTrip(record)),
                    "$schema[$index] changed in a decode and encode round trip",
                )
            }
        }
    }

    @Test
    fun paritySpanFixtureRoundTripsAsContractSpans() {
        val fixture = resourceJson("expected_spans.json")
        val taxonomy = resourceJson("policies/portable_policy_taxonomy.json")
        val aliases = taxonomy.getValue("aliases").jsonObject
        val labels = taxonomy.getValue("labels").jsonObject
        val cases = fixture.getValue("cases").jsonArray
        var decoded = 0
        var expected = 0

        for (caseElement in cases) {
            val fixtureCase = caseElement.jsonObject
            val caseId = fixtureCase.string("id")
            val actions = fixtureCase.getValue("actions").jsonArray.map { it.jsonObject }
            val spans = fixtureCase.getValue("spans").jsonArray
            expected += spans.size
            spans.forEachIndexed { index, spanElement ->
                val context = "case=$caseId span=$index"
                val span = spanElement.jsonObject
                val label = span.string("label")
                val start = span.int("start")
                val end = span.int("end")
                val canonical = aliases[taxonomyKey(label)]?.jsonPrimitive?.content
                    ?: label.uppercase()
                val policy = labels.getValue(canonical).jsonObject.string("policy_label")
                val action = actions.firstOrNull {
                    it.string("label") == label && it.int("start") == start &&
                        it.int("end") == end
                }
                if (action != null) {
                    assertEquals(canonical, action.string("canonical_label"), context)
                }
                val record = JsonObject(
                    linkedMapOf(
                        "schema_version" to JsonPrimitive(1),
                        "doc_id" to JsonPrimitive(caseId),
                        "start" to JsonPrimitive(start),
                        "end" to JsonPrimitive(end),
                        "text_hash" to JsonPrimitive("hmac-sha256:" + "0".repeat(64)),
                        "entity_type" to JsonPrimitive(label),
                        "canonical_label" to JsonPrimitive(canonical),
                        "policy_label" to JsonPrimitive(policy),
                        "regulatory_tags" to JsonArray(emptyList()),
                        "score" to JsonNull,
                        "detector" to JsonPrimitive("native_span_parity_fixture"),
                        "evidence" to JsonObject(
                            listOf("utf8_start", "utf8_end", "utf16_start", "utf16_end")
                                .associateWith { span.getValue(it) },
                        ),
                        "action" to (action?.getValue("action") ?: JsonPrimitive("keep")),
                        "replacement" to (action?.getValue("replacement") ?: JsonNull),
                        "reversible_id" to JsonNull,
                        "section" to JsonNull,
                        "metadata" to JsonObject(
                            mapOf("policy" to fixtureCase.getValue("policy")),
                        ),
                    ),
                )

                val contract = OpenMedContractSpan.fromJson(record)

                assertEquals(caseId, contract.docId, context)
                assertEquals(start, contract.start, context)
                assertEquals(end, contract.end, context)
                assertEquals(label, contract.entityType, context)
                assertEquals(canonical, contract.canonicalLabel.value, context)
                assertEquals(policy, contract.policyLabel.value, context)
                assertEquals(span.getValue("utf16_end"), contract.evidence["utf16_end"], context)
                assertNull(contract.score, context)
                assertTrue(equivalent(record, contract.toJson()), context)
                assertEquals(contract, OpenMedContractSpan.fromJson(contract.toJson()), context)
                decoded += 1
            }
        }
        assertEquals(expected, decoded)
        assertTrue(decoded > 0)
    }

    @Test
    fun missingRequiredKeyFailsClosedEvenWhenItsValueMayBeNull() {
        val record = JsonObject(firstExample("span") - "score")

        val error = assertFailsWith<OpenMedContractException> {
            OpenMedContractSpan.fromJson(record)
        }

        assertEquals("OpenMedContractSpan.score: missing_required_key", error.message)
    }

    @Test
    fun missingNestedRequiredKeyNamesTheNestedRecord() {
        val example = firstExample("attestation")
        val template = JsonObject(example.getValue("template").jsonObject - "hash")
        val record = JsonObject(example + ("template" to template))

        val error = assertFailsWith<OpenMedContractException> {
            OpenMedContractDataResidencyAttestation.fromJson(record)
        }

        assertEquals(
            "OpenMedContractDataResidencyAttestationTemplate.hash: missing_required_key",
            error.message,
        )
    }

    @Test
    fun undeclaredPropertyFailsWithoutNamingItOrItsValue() {
        val record = JsonObject(
            firstExample("span") +
                ("synthetic_undeclared" to JsonPrimitive("SYNTHETIC-SECRET-VALUE")),
        )

        val error = assertFailsWith<OpenMedContractException> {
            OpenMedContractSpan.fromJson(record)
        }

        assertEquals("OpenMedContractSpan has an undeclared property", error.message)
    }

    @Test
    fun mistypedUnknownAndConstantValuesFailWithValueFreeReasons() {
        val cases = listOf(
            "canonical_label" to JsonPrimitive("SYNTHETIC_UNKNOWN_LABEL") to
                "OpenMedContractSpan.canonical_label: unknown_enum_value",
            "schema_version" to JsonPrimitive(2) to
                "OpenMedContractSpan.schema_version: constant_mismatch",
            "start" to JsonPrimitive("SYNTHETIC-NOT-AN-OFFSET") to
                "OpenMedContractSpan.start: expected_integer",
            "start" to JsonPrimitive(1.5) to "OpenMedContractSpan.start: expected_integer",
            "score" to JsonPrimitive("0.5") to "OpenMedContractSpan.score: expected_number",
            "detector" to JsonPrimitive(true) to "OpenMedContractSpan.detector: expected_string",
            "evidence" to JsonArray(emptyList()) to
                "OpenMedContractSpan.evidence: expected_object",
            "regulatory_tags" to JsonArray(listOf(JsonPrimitive(7))) to
                "OpenMedContractSpan.regulatory_tags: expected_string",
        )

        for ((change, message) in cases) {
            val (key, value) = change
            val record = JsonObject(firstExample("span") + (key to value))
            val error = assertFailsWith<OpenMedContractException>(key) {
                OpenMedContractSpan.fromJson(record)
            }
            assertEquals(message, error.message, key)
        }
    }

    @Test
    fun malformedTextFailsWithoutEchoingIt() {
        val error = assertFailsWith<OpenMedContractException> {
            OpenMedContractSpan.fromJson("{\"doc_id\": \"SYNTHETIC-SECRET-VALUE\"")
        }

        assertEquals("OpenMedContractSpan: malformed_json", error.message)
        val array = assertFailsWith<OpenMedContractException> {
            OpenMedContractSpan.fromJson("[]")
        }
        assertEquals("OpenMedContractSpan: expected_object", array.message)
    }

    @Test
    fun constructorFillsConstantsAndEncodesExplicitNulls() {
        val span = OpenMedContractSpan(
            docId = "synthetic-doc",
            start = 0,
            end = 4,
            textHash = "hmac-sha256:" + "a".repeat(64),
            entityType = "first_name",
            canonicalLabel = OpenMedContractCanonicalLabel.FIRST_NAME,
            policyLabel = OpenMedContractPolicyLabel.SENSITIVE_ATTRIBUTE,
            regulatoryTags = emptyList(),
            score = null,
            detector = null,
            evidence = JsonObject(mapOf("flag" to JsonPrimitive(true))),
            action = OpenMedContractSpanAction.FORMAT_PRESERVE,
            replacement = null,
            reversibleId = null,
            section = null,
            metadata = JsonObject(emptyMap()),
        )
        val json = span.toJson()

        assertEquals(1, span.schemaVersion)
        assertEquals(JsonPrimitive(1), json["schema_version"])
        assertEquals("format_preserve", json.string("action"))
        for (key in listOf("score", "detector", "replacement", "reversible_id", "section")) {
            assertEquals(JsonNull, json[key], key)
        }
        assertEquals(span, OpenMedContractSpan.fromJson(json.toString()))
        assertEquals(
            listOf("DIRECT_IDENTIFIER", "QUASI_IDENTIFIER", "SENSITIVE_ATTRIBUTE", "CLINICAL_CONCEPT"),
            OpenMedContractPolicyLabel.entries.map { it.value },
        )
        assertNull(OpenMedContractCanonicalLabel.fromValue("SYNTHETIC_UNKNOWN_LABEL"))
    }

    // Helpers

    private fun <T : Any> codec(
        decode: (JsonObject) -> T,
        encode: (T) -> JsonObject,
    ): (JsonObject) -> JsonObject = { json ->
        val decoded = decode(json)
        val encoded = encode(decoded)
        assertEquals(decoded, decode(encoded))
        encoded
    }

    /** JSON equality where numbers compare by value and booleans stay distinct. */
    private fun equivalent(left: JsonElement, right: JsonElement): Boolean = when {
        left is JsonNull || right is JsonNull -> left is JsonNull && right is JsonNull
        left is JsonObject && right is JsonObject ->
            left.keys == right.keys && left.all { (key, value) -> equivalent(value, right.getValue(key)) }
        left is JsonArray && right is JsonArray ->
            left.size == right.size && left.indices.all { equivalent(left[it], right[it]) }
        left is JsonPrimitive && right is JsonPrimitive -> primitivesEquivalent(left, right)
        else -> false
    }

    private fun primitivesEquivalent(left: JsonPrimitive, right: JsonPrimitive): Boolean {
        if (left.isString || right.isString) {
            return left.isString == right.isString && left.content == right.content
        }
        val booleans = setOf("true", "false")
        if (left.content in booleans || right.content in booleans) {
            return left.content == right.content
        }
        val leftNumber = left.content.toBigDecimalOrNull() ?: return false
        val rightNumber = right.content.toBigDecimalOrNull() ?: return false
        return leftNumber.compareTo(rightNumber) == 0
    }

    private fun taxonomyKey(label: String): String =
        label.lowercase().filter { it in 'a'..'z' || it in '0'..'9' }

    private fun examples(): Map<String, List<JsonObject>> {
        val fixture = resourceJson("core_contract_examples.json")
        assertEquals("openmed.core_contract_examples", fixture.string("schema"))
        assertEquals(1, fixture.int("version"))
        return fixture.getValue("examples").jsonObject.mapValues { (_, records) ->
            records.jsonArray.map { it.jsonObject }
        }
    }

    private fun firstExample(schema: String): JsonObject = examples().getValue(schema).first()

    private fun resourceJson(name: String): JsonObject {
        val stream = requireNotNull(javaClass.classLoader?.getResourceAsStream(name)) {
            "$name is not registered as an OpenMedKit test resource"
        }
        return stream.use { Json.parseToJsonElement(it.readBytes().decodeToString()).jsonObject }
    }

    private fun JsonObject.string(key: String): String = getValue(key).jsonPrimitive.content

    private fun JsonObject.int(key: String): Int = getValue(key).jsonPrimitive.int
}
