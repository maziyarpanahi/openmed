package com.openmed.openmedkit.parity

import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.Description
import org.junit.runners.model.Statement
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Offline tests for the Kotlin contract-gate reporter over synthetic manifests
 * and fixtures, plus checks of the committed manifest's Kotlin cells.
 */
class ContractGateReporterTest {
    @get:Rule
    val temporary = TemporaryFolder()

    @Test
    fun caseIdsFollowTheGateRules() {
        val groups = listOf(
            ContractGateCatalog.Group("cases", "id"),
            ContractGateCatalog.Group("invalid", null),
            ContractGateCatalog.Group("segment", null),
        )
        val document = json(
            """{"cases": [{"id": "a-1"}, {"id": "b.2"}], "invalid": ["x", "y"], "segment": {"k": 1}}""",
        )

        assertEquals(
            listOf("cases/a-1", "cases/b.2", "invalid/0", "invalid/1", "segment"),
            ContractGateCatalog.deriveCaseIds(groups, document),
        )
        for ((broken, code) in listOf(
            """{"cases": [{"id": "a"}, {"id": "a"}], "invalid": [], "segment": {}}""" to "case_id_duplicate",
            """{"cases": [{"id": "bad id"}], "invalid": [], "segment": {}}""" to "case_id_invalid",
            """{"cases": [{"id": 7}], "invalid": [], "segment": {}}""" to "case_id_invalid",
            """{"cases": [], "segment": {}}""" to "case_group_missing",
            """[]""" to "case_group_missing",
        )) {
            val error = assertFailsWith<ContractGateException> {
                ContractGateCatalog.deriveCaseIds(groups, json(broken))
            }
            assertEquals(code, error.code)
            assertEquals("contract_gate $code", error.message)
        }
    }

    @Test
    fun wholeMethodOutcomesCoverEveryCaseAndCountTests() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()

        run(repo.rule(results), statement { })
        var document = read(results)
        assertEquals(manifestDigest(repo), document.string("manifest_sha256"))
        assertEquals(listOf("alpha", "beta"), document.fixtures().map { it.string("fixture_id") })
        val alpha = document.fixture("alpha")
        assertEquals(ALPHA_CASES, alpha.strings("passed"))
        assertEquals(counts(1, 0, 0), alpha["tests"])
        assertEquals(repo.alphaDigest, alpha.string("fixture_sha256"))
        val beta = document.fixture("beta")
        assertEquals(listOf("segment" to "consumer_not_collected"), beta.outcomes("failed"))
        assertFalse("tests" in beta)
        assertEquals(repo.betaDigest, beta.string("fixture_sha256"))

        val failure = AssertionError("synthetic failure")
        val thrown = assertFailsWith<AssertionError> { run(repo.rule(results), statement { throw failure }) }
        assertSame(failure, thrown)
        document = read(results)
        assertEquals(emptyList(), document.fixture("alpha").strings("passed"))
        assertEquals(
            ALPHA_CASES.map { it to "consumer_failed" },
            document.fixture("alpha").outcomes("failed"),
        )
        assertEquals(counts(1, 1, 0), document.fixture("alpha")["tests"])
        assertSchemaShape(document)
    }

    @Test
    fun perCaseRecordsKeepCasesApartAndRethrowTheFirstFailure() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()
        val rule = repo.rule(results)
        val seen = mutableListOf<String>()
        val first = AssertionError("first")

        val thrown = assertFailsWith<AssertionError> {
            run(
                rule,
                statement {
                    rule.cases(listOf("a-1", "b-2", "c-3"), { "cases/$it" }) { id ->
                        seen += id
                        when (id) {
                            "a-1" -> throw first
                            "b-2" -> throw IllegalStateException("second")
                        }
                    }
                },
            )
        }

        assertSame(first, thrown)
        assertEquals(listOf("a-1", "b-2", "c-3"), seen)
        val alpha = read(results).fixture("alpha")
        assertEquals(listOf("cases/c-3"), alpha.strings("passed"))
        assertEquals(
            listOf(
                "cases/a-1" to "consumer_failed",
                "cases/b-2" to "consumer_error",
                "invalid/0" to "case_not_exercised",
            ),
            alpha.outcomes("failed"),
        )
        assertEquals(counts(0, 1, 0), alpha["tests"])

        // A passing whole-method test later never clears a recorded failure.
        run(repo.rule(results), statement { })
        val merged = read(results).fixture("alpha")
        assertEquals(listOf("cases/c-3", "invalid/0"), merged.strings("passed"))
        assertEquals(
            listOf("cases/a-1" to "consumer_failed", "cases/b-2" to "consumer_error"),
            merged.outcomes("failed"),
        )
        assertEquals(counts(1, 1, 0), merged["tests"])
    }

    @Test
    fun failureOutsideTheCaseRecordsFailsEveryCase() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()
        val rule = repo.rule(results)

        assertFailsWith<AssertionError> {
            run(
                rule,
                statement {
                    rule.cases(listOf("a-1"), { "cases/$it" }) { }
                    throw AssertionError("after the loop")
                },
            )
        }

        assertEquals(
            ALPHA_CASES.map { it to "consumer_failed" },
            read(results).fixture("alpha").outcomes("failed"),
        )
    }

    @Test
    fun skippedTestsOnlyCountWhenNothingRan() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()

        assertFailsWith<org.junit.AssumptionViolatedException> {
            run(repo.rule(results), statement { assumeTrue("synthetic", false) })
        }
        assertEquals(
            ALPHA_CASES.map { it to "consumer_skipped" },
            read(results).fixture("alpha").outcomes("skipped"),
        )

        run(repo.rule(results), statement { })
        val alpha = read(results).fixture("alpha")
        assertEquals(ALPHA_CASES, alpha.strings("passed"))
        assertEquals(emptyList(), alpha.outcomes("skipped"))
        assertEquals(counts(1, 0, 1), alpha["tests"])
    }

    @Test
    fun digestRecordsTheFixtureBytesActuallyLoaded() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()
        val stale = repo.alphaBytes + " ".toByteArray()

        val rule = repo.rule(results, served = mapOf(ALPHA_PATH to stale))
        run(rule, statement { assertTrue(rule.fixtureBytes().contentEquals(stale)) })
        val loaded = read(results).fixture("alpha").string("fixture_sha256")
        assertEquals(contractGateDigest(stale), loaded)
        assertNotEquals(repo.alphaDigest, loaded)

        // A later read of the current bytes does not hide the stale copy.
        val current = repo.rule(results)
        run(current, statement { current.fixtureBytes() })
        assertEquals(loaded, read(results).fixture("alpha").string("fixture_sha256"))
    }

    @Test
    fun resultFilesCarryIdsDigestsAndCodesOnly() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()
        val rule = repo.rule(results)

        assertFailsWith<AssertionError> {
            run(
                rule,
                statement {
                    rule.fixtureText()
                    rule.cases(listOf("a-1"), { "cases/$it" }) { throw AssertionError(SENTINEL) }
                },
            )
        }

        val text = String(Files.readAllBytes(results.resolve("kotlin.json")), Charsets.UTF_8)
        assertFalse(SENTINEL in text)
        assertFalse(results.toString() in text)
        assertTrue(text.endsWith("}\n"))
        assertSchemaShape(json(text).jsonObject)
        assertEquals(setOf("kotlin.json", ".kotlin.json.lock"), results.toFile().list()!!.toSet())
    }

    @Test
    fun undeclaredConsumersUnknownFixturesAndUnknownCasesFailWithCodes() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()

        for ((rule, className, code) in listOf(
            Triple(repo.rule(results), "com.example.UndeclaredTest", "consumer_undeclared"),
            Triple(repo.rule(results, fixtureId = "gamma"), CONSUMER, "runtime_not_required"),
            Triple(repo.rule(results, fixtureId = "delta"), CONSUMER, "unknown_fixture"),
        )) {
            val error = assertFailsWith<ContractGateException> { run(rule, statement { }, className) }
            assertEquals(code, error.code)
        }
        assertFalse(Files.exists(results.resolve("kotlin.json")))

        val rule = repo.rule(results)
        val error = assertFailsWith<IllegalStateException> {
            run(rule, statement { rule.cases(listOf(SENTINEL), { "cases/$it" }) { } })
        }
        assertEquals("contract_gate unknown_case fixture=alpha", error.message)
        assertEquals(
            ALPHA_CASES.map { it to "consumer_error" },
            read(results).fixture("alpha").outcomes("failed"),
        )
    }

    @Test
    fun unreadableFixtureReportsAFixtureError() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()

        run(repo.rule(results, served = mapOf(BETA_PATH to null)), statement { })

        val beta = read(results).fixture("beta")
        assertEquals("0".repeat(64), beta.string("fixture_sha256"))
        assertEquals(listOf("fixture_unreadable"), beta.strings("errors"))
        assertEquals(emptyList(), beta.strings("passed"))
        assertSchemaShape(read(results))
    }

    @Test
    fun noResultsDirectoryWritesNothingAndStaleFilesAreReplaced() {
        val repo = SyntheticRepo()
        val results = temporary.newFolder("results").toPath()

        run(repo.rule(null), statement { })
        assertEquals(0, results.toFile().list()!!.size)

        val target = results.resolve("kotlin.json")
        for (stale in listOf("not json", """{"schema": "openmed.contract_gate.result", "runtime": "kotlin", "manifest_sha256": "${"1".repeat(64)}", "fixtures": []}""")) {
            Files.write(target, stale.toByteArray())
            run(repo.rule(results), statement { })
            assertEquals(counts(1, 0, 0), read(results).fixture("alpha")["tests"])
        }
    }

    @Test
    fun committedManifestDeclaresARuleInEveryKotlinConsumer() {
        val catalog = ContractGateCatalog.shared
        val root = File(requireNotNull(System.getProperty("openmedkit.moduleDirectory"))).parentFile.parentFile
        val required = catalog.required

        assertTrue(required.isNotEmpty())
        for (entry in required) {
            assertTrue(entry.consumers.isNotEmpty(), entry.id)
            assertEquals(entry.sha256, catalog.resourceDigest(entry.id), entry.id)
            assertNotNull(catalog.caseIds(entry.id), entry.id)
            for (consumer in entry.consumers) {
                assertTrue(root.resolve(consumer).isFile, consumer)
                assertTrue(consumer.startsWith(ContractGateCatalog.TEST_SOURCES), consumer)
                val className = consumer.removePrefix(ContractGateCatalog.TEST_SOURCES)
                    .removeSuffix(".kt").replace('/', '.')
                val type = Class.forName(className)
                assertTrue(
                    type.declaredFields.any { it.type == ContractGateRule::class.java },
                    "$className declares no ContractGateRule",
                )
                catalog.requireConsumer(entry.id, className)
            }
        }
        assertEquals(
            ContractGateCatalog.Companion::class.java.classLoader!!
                .getResourceAsStream(ContractGateCatalog.MANIFEST_RESOURCE)!!.use { contractGateDigest(it.readBytes()) },
            catalog.manifestSha256,
        )
    }

    // Helpers.

    private class SyntheticRepo {
        val alphaBytes = """{"version": 1, "cases": [{"id": "a-1", "text": "$SENTINEL"}, {"id": "b-2"}, {"id": "c-3"}], "invalid": ["$SENTINEL"]}"""
            .toByteArray()
        val betaBytes = """{"segment": {"text": "$SENTINEL"}}""".toByteArray()
        val alphaDigest = contractGateDigest(alphaBytes)
        val betaDigest = contractGateDigest(betaBytes)
        val manifest = """
            {
              "schema": "openmed.contract_gate.manifest",
              "version": 1,
              "algorithm": "sha256",
              "roots": ["fixtures/parity", "tests/fixtures/parity"],
              "runtimes": {
                "python": {"reporter": "active", "command": "python"},
                "kotlin": {"reporter": "active", "command": "gradle"}
              },
              "fixtures": [
                {"id": "alpha", "path": "$ALPHA_PATH", "sha256": "$alphaDigest", "version": 1,
                 "case_groups": [{"key": "cases", "id_field": "id"}, {"key": "invalid"}], "companions": [],
                 "runtimes": {"python": {"status": "required", "consumers": ["a.py"]},
                              "kotlin": {"status": "required", "consumers": ["$CONSUMER_PATH"]}}},
                {"id": "beta", "path": "$BETA_PATH", "sha256": "$betaDigest", "version": 1,
                 "case_groups": [{"key": "segment"}], "companions": [],
                 "runtimes": {"python": {"status": "required", "consumers": ["b.py"]},
                              "kotlin": {"status": "required", "consumers": ["$OTHER_PATH"]}}},
                {"id": "gamma", "path": "$BETA_PATH", "sha256": "$betaDigest", "version": 1,
                 "case_groups": [{"key": "segment"}], "companions": [],
                 "runtimes": {"python": {"status": "required", "consumers": ["c.py"]},
                              "kotlin": {"status": "not_applicable", "reason": "Synthetic."}}}
              ]
            }
        """.trimIndent().toByteArray()

        fun rule(
            results: Path?,
            served: Map<String, ByteArray?> = emptyMap(),
            fixtureId: String = "alpha",
        ): ContractGateRule {
            val files = mapOf(ALPHA_PATH to alphaBytes, BETA_PATH to betaBytes) + served
            val catalog = ContractGateCatalog(manifest) { files[it] }
            return ContractGateRule(fixtureId, { catalog }, { results })
        }
    }

    private fun run(rule: ContractGateRule, statement: Statement, className: String = CONSUMER) {
        rule.apply(statement, Description.createTestDescription(className, "synthetic")).evaluate()
    }

    private fun statement(block: () -> Unit): Statement = object : Statement() {
        override fun evaluate() = block()
    }

    private fun manifestDigest(repo: SyntheticRepo): String = contractGateDigest(repo.manifest)

    private fun read(results: Path): JsonObject =
        json(String(Files.readAllBytes(results.resolve("kotlin.json")), Charsets.UTF_8)).jsonObject

    private fun json(text: String): JsonElement = Json.parseToJsonElement(text)

    private fun counts(passed: Int, failed: Int, skipped: Int): JsonObject =
        json("""{"passed": $passed, "failed": $failed, "skipped": $skipped}""").jsonObject

    private fun JsonObject.string(key: String): String? = (this[key] as? JsonPrimitive)?.content

    private fun JsonObject.fixtures(): List<JsonObject> = getValue("fixtures").jsonArray.map { it.jsonObject }

    private fun JsonObject.fixture(id: String): JsonObject = fixtures().single { it.string("fixture_id") == id }

    private fun JsonObject.strings(key: String): List<String> =
        (this[key] as? JsonArray).orEmpty().map { it.jsonPrimitive.content }

    private fun JsonObject.outcomes(key: String): List<Pair<String, String>> =
        getValue(key).jsonArray.map { it.jsonObject }.map {
            it.getValue("case_id").jsonPrimitive.content to it.getValue("reason").jsonPrimitive.content
        }

    /** Checks keys, types and patterns against the committed result JSON Schema. */
    private fun assertSchemaShape(document: JsonObject) {
        val schema = json(
            requireNotNull(javaClass.classLoader?.getResourceAsStream("contract_gate_result.schema.json"))
                .use { String(it.readBytes(), Charsets.UTF_8) },
        ).jsonObject
        val definitions = schema.getValue("\$defs").jsonObject
        fun properties(node: JsonObject) = node.getValue("properties").jsonObject.keys
        fun required(node: JsonObject) = node.getValue("required").jsonArray.map { it.jsonPrimitive.content }
        fun pattern(name: String) = Regex(definitions.getValue(name).jsonObject.getValue("pattern").jsonPrimitive.content)
        val fixtureSchema = definitions.getValue("fixture_result").jsonObject

        assertTrue(document.keys.containsAll(required(schema)) && properties(schema).containsAll(document.keys))
        assertEquals("openmed.contract_gate.result", document.string("schema"))
        assertEquals(1, document.getValue("schema_version").jsonPrimitive.int)
        assertTrue(pattern("sha256").matches(document.string("manifest_sha256")!!))
        for (fixture in document.fixtures()) {
            assertTrue(fixture.keys.containsAll(required(fixtureSchema)), fixture.keys.toString())
            assertTrue(properties(fixtureSchema).containsAll(fixture.keys), fixture.keys.toString())
            assertTrue(pattern("sha256").matches(fixture.string("fixture_sha256")!!))
            val reported = fixture.strings("passed") +
                fixture.outcomes("failed").map { it.first } +
                fixture.outcomes("skipped").map { it.first }
            assertEquals(reported.size, reported.toSet().size)
            reported.forEach { assertTrue(pattern("case_id").matches(it), it) }
            (fixture.outcomes("failed") + fixture.outcomes("skipped") + fixture.strings("errors").map { "x" to it })
                .forEach { assertTrue(pattern("reason").matches(it.second), it.second) }
            (fixture["tests"] as? JsonObject)?.let { tests ->
                assertEquals(setOf("passed", "failed", "skipped"), tests.keys)
            }
        }
    }

    private companion object {
        const val SENTINEL = "SYNTH_PHI_SENTINEL Jane Roe"
        const val ALPHA_PATH = "tests/fixtures/parity/alpha.json"
        const val BETA_PATH = "fixtures/parity/beta.json"
        const val CONSUMER = "com.example.SyntheticConsumerTest"
        const val CONSUMER_PATH = "android/openmedkit/src/test/kotlin/com/example/SyntheticConsumerTest.kt"
        const val OTHER_PATH = "android/openmedkit/src/test/kotlin/com/example/OtherConsumerTest.kt"
        val ALPHA_CASES = listOf("cases/a-1", "cases/b-2", "cases/c-3", "invalid/0")
    }
}
