package com.openmed.openmedkit.parity

import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.internal.AssumptionViolatedException
import org.junit.rules.TestRule
import org.junit.runner.Description
import org.junit.runners.model.Statement

/**
 * Kotlin reporter for the cross-runtime contract gate (`docs/contract-gate.md`).
 *
 * A consumer of a shared parity fixture declares one rule per fixture it reads:
 *
 * ```
 * @get:Rule
 * val contractGate = ContractGateRule("offset_contract")
 * ```
 *
 * The rule checks that the test class is a declared Kotlin consumer of the
 * fixture in `contract_gate.json`, loads the fixture through [fixtureBytes] so
 * the digest of the bytes actually read is recorded, and maps each test method
 * onto the fixture's case ids. As in the Python runner, a test method covers
 * every case of the fixture unless it records per-case outcomes with [cases]. A
 * case passes when at least one covering test passed and none failed.
 *
 * When the `openmed.contractGate.resultsDir` system property names a directory,
 * every method's outcome is merged into `kotlin.json` there. Without it the rule
 * only loads the fixture and checks the declaration. The result file carries
 * fixture ids, digests, case ids, counts and reason codes only; assertion
 * messages and fixture values never reach it.
 *
 * @property fixtureId Manifest id of the fixture the test class consumes.
 */
class ContractGateRule internal constructor(
    val fixtureId: String,
    private val catalog: () -> ContractGateCatalog,
    private val resultsDirectory: () -> Path?,
) : TestRule {
    /** Rule for [fixtureId] backed by the shared test resources. */
    constructor(fixtureId: String) : this(
        fixtureId,
        { ContractGateCatalog.shared },
        { configuredResultsDirectory() },
    )

    private var loadedDigest: String? = null
    private val recorded = linkedMapOf<String, ContractGateOutcome>()
    private var caseFailure: Throwable? = null

    /**
     * Reads the fixture resource and records the SHA-256 of the bytes read.
     *
     * @return The fixture bytes as served from the test classpath.
     */
    fun fixtureBytes(): ByteArray {
        val entry = catalog().fixture(fixtureId)
        val bytes = catalog().read(entry.path)
            ?: throw IllegalStateException("contract_gate fixture_unreadable fixture=$fixtureId")
        val digest = contractGateDigest(bytes)
        if (loadedDigest == null || loadedDigest == entry.sha256) {
            loadedDigest = digest
        }
        return bytes
    }

    /** Reads the fixture resource as UTF-8 text; see [fixtureBytes]. */
    fun fixtureText(): String = String(fixtureBytes(), Charsets.UTF_8)

    /**
     * Runs [check] for every item and records each item's outcome under its
     * case id.
     *
     * Every item runs even after a failure, so each case gets its own outcome.
     * The first failure is rethrown once all items ran, so the test still fails
     * with the original assertion.
     *
     * @param items Fixture cases, in fixture order.
     * @param caseId Maps an item to its contract-gate case id, for example
     *   `cases/<id>`. Unknown ids fail the test with `unknown_case`.
     * @param check Assertions for one item.
     */
    fun <T> cases(items: Iterable<T>, caseId: (T) -> String, check: (T) -> Unit) {
        val expected = catalog().caseIds(fixtureId)?.toSet()
            ?: throw IllegalStateException("contract_gate fixture_unreadable fixture=$fixtureId")
        var first: Throwable? = null
        for (item in items) {
            val id = caseId(item)
            if (id !in expected) {
                throw IllegalStateException("contract_gate unknown_case fixture=$fixtureId")
            }
            val outcome = try {
                check(item)
                ContractGateOutcome.PASSED
            } catch (error: AssumptionViolatedException) {
                ContractGateOutcome.SKIPPED
            } catch (error: AssertionError) {
                first = first ?: error
                ContractGateOutcome.FAILED
            } catch (error: Exception) {
                first = first ?: error
                ContractGateOutcome.ERROR
            }
            recorded[id] = maxOf(recorded[id] ?: outcome, outcome)
        }
        first?.let {
            caseFailure = it
            throw it
        }
    }

    override fun apply(base: Statement, description: Description): Statement =
        object : Statement() {
            override fun evaluate() {
                val gate = catalog()
                gate.requireConsumer(fixtureId, description.className)
                var thrown: Throwable? = null
                val outcome = try {
                    base.evaluate()
                    ContractGateOutcome.PASSED
                } catch (error: AssumptionViolatedException) {
                    thrown = error
                    ContractGateOutcome.SKIPPED
                } catch (error: AssertionError) {
                    thrown = error
                    ContractGateOutcome.FAILED
                } catch (error: Throwable) {
                    thrown = error
                    ContractGateOutcome.ERROR
                }
                resultsDirectory()?.let { directory ->
                    ContractGateResultFile.merge(directory, gate, observation(gate, outcome, thrown))
                }
                thrown?.let { throw it }
            }
        }

    private fun observation(
        gate: ContractGateCatalog,
        outcome: ContractGateOutcome,
        thrown: Throwable?,
    ): ContractGateObservation {
        val cases = LinkedHashMap(recorded)
        // A method without per-case records, or one that failed outside them,
        // covers every case of the fixture with its own outcome.
        if (recorded.isEmpty() || (thrown != null && thrown !== caseFailure)) {
            for (id in gate.caseIds(fixtureId).orEmpty()) {
                cases[id] = maxOf(cases[id] ?: outcome, outcome)
            }
        }
        return ContractGateObservation(fixtureId, loadedDigest, outcome, cases)
    }

    internal companion object {
        /** System property naming the directory that receives `kotlin.json`. */
        const val RESULTS_DIR_PROPERTY = "openmed.contractGate.resultsDir"

        fun configuredResultsDirectory(): Path? =
            System.getProperty(RESULTS_DIR_PROPERTY)?.takeIf { it.isNotBlank() }?.let(Paths::get)
    }
}

/** Value-free failure of the reporter itself, identified by [code]. */
internal class ContractGateException(val code: String) : IllegalStateException("contract_gate $code")

/** Outcome of a test or a case; a later constant wins when outcomes merge. */
internal enum class ContractGateOutcome(val reason: String?) {
    SKIPPED("consumer_skipped"),
    PASSED(null),
    ERROR("consumer_error"),
    FAILED("consumer_failed"),
}

/** One test method's contribution to a fixture's result. */
internal class ContractGateObservation(
    val fixtureId: String,
    val digest: String?,
    val test: ContractGateOutcome,
    val cases: Map<String, ContractGateOutcome>,
)

/**
 * The Kotlin view of `contract_gate.json` plus the fixture resources it names.
 *
 * @param manifestBytes The manifest bytes as read.
 * @param read Returns a resource's bytes for a repository-relative manifest
 *   path, or `null` when the resource is missing.
 */
internal class ContractGateCatalog(manifestBytes: ByteArray, private val read: (String) -> ByteArray?) {
    class Group(val key: String, val idField: String?)

    class Fixture(
        val id: String,
        val path: String,
        val sha256: String,
        val groups: List<Group>,
        val required: Boolean,
        val consumers: List<String>,
    )

    val manifestSha256: String = contractGateDigest(manifestBytes)
    val fixtures: List<Fixture>
    private val caseIdCache = mutableMapOf<String, List<String>?>()
    private val digestCache = mutableMapOf<String, String?>()

    init {
        val manifest = try {
            Json.parseToJsonElement(String(manifestBytes, Charsets.UTF_8)) as JsonObject
        } catch (error: Exception) {
            throw ContractGateException("manifest_unreadable")
        }
        if (manifest.string("schema") != MANIFEST_SCHEMA || manifest["version"] != JsonPrimitive(1)) {
            throw ContractGateException("manifest_unsupported")
        }
        val runtimes = manifest["runtimes"] as? JsonObject
        if (runtimes?.get(RUNTIME) !is JsonObject) {
            throw ContractGateException("runtime_cell_missing")
        }
        val entries = manifest["fixtures"] as? JsonArray ?: throw ContractGateException("manifest_invalid")
        fixtures = entries.map { raw ->
            val entry = raw as? JsonObject ?: throw ContractGateException("manifest_invalid")
            val cell = (entry["runtimes"] as? JsonObject)?.get(RUNTIME) as? JsonObject
                ?: throw ContractGateException("runtime_cell_missing")
            Fixture(
                id = entry.string("id") ?: throw ContractGateException("manifest_invalid"),
                path = entry.string("path") ?: throw ContractGateException("manifest_invalid"),
                sha256 = entry.string("sha256") ?: throw ContractGateException("manifest_invalid"),
                groups = (entry["case_groups"] as? JsonArray ?: throw ContractGateException("manifest_invalid"))
                    .map { group ->
                        val value = group as? JsonObject ?: throw ContractGateException("manifest_invalid")
                        Group(
                            key = value.string("key") ?: throw ContractGateException("manifest_invalid"),
                            idField = value.string("id_field"),
                        )
                    },
                required = cell.string("status") == "required",
                consumers = (cell["consumers"] as? JsonArray).orEmpty().map {
                    (it as? JsonPrimitive)?.content ?: throw ContractGateException("manifest_invalid")
                },
            )
        }
    }

    /** Fixtures whose Kotlin cell is `required`, in manifest order. */
    val required: List<Fixture> get() = fixtures.filter { it.required }

    fun fixture(id: String): Fixture =
        fixtures.firstOrNull { it.id == id } ?: throw ContractGateException("unknown_fixture")

    /** Fails unless [className] is a declared Kotlin consumer of a required fixture. */
    fun requireConsumer(fixtureId: String, className: String) {
        val entry = fixture(fixtureId)
        if (!entry.required) {
            throw ContractGateException("runtime_not_required")
        }
        if (consumerPath(className) !in entry.consumers) {
            throw ContractGateException("consumer_undeclared")
        }
    }

    /** Bytes of the resource behind a manifest path, or `null` when missing. */
    fun read(path: String): ByteArray? = read.invoke(path)

    /** SHA-256 of the fixture resource, or `null` when it cannot be read. */
    fun resourceDigest(fixtureId: String): String? = digestCache.getOrPut(fixtureId) {
        read(fixture(fixtureId).path)?.let(::contractGateDigest)
    }

    /** Case ids derived from the fixture resource, or `null` when it is unreadable. */
    fun caseIds(fixtureId: String): List<String>? = caseIdCache.getOrPut(fixtureId) {
        val entry = fixture(fixtureId)
        val bytes = read(entry.path) ?: return@getOrPut null
        try {
            deriveCaseIds(entry.groups, Json.parseToJsonElement(String(bytes, Charsets.UTF_8)))
        } catch (error: Exception) {
            null
        }
    }

    companion object {
        const val RUNTIME = "kotlin"
        const val MANIFEST_RESOURCE = "contract_gate.json"
        const val MANIFEST_SCHEMA = "openmed.contract_gate.manifest"

        /** Kotlin test sources, relative to the repository root. */
        const val TEST_SOURCES = "android/openmedkit/src/test/kotlin/"

        /** Repository directories the Gradle build adds as test resource roots. */
        val RESOURCE_ROOTS = listOf("tests/fixtures/parity/", "fixtures/parity/")

        private val CASE_PART = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,95}")

        /** The catalog over this classloader's test resources. */
        val shared: ContractGateCatalog by lazy {
            val loader = ContractGateCatalog::class.java.classLoader
            val readResource = { name: String -> loader?.getResourceAsStream(name)?.use { it.readBytes() } }
            val manifest = readResource(MANIFEST_RESOURCE) ?: throw ContractGateException("manifest_unreadable")
            ContractGateCatalog(manifest) { path -> resourceName(path)?.let(readResource) }
        }

        /** Classpath resource name of a repository-relative fixture path. */
        fun resourceName(path: String): String? =
            RESOURCE_ROOTS.firstOrNull(path::startsWith)?.let { path.removePrefix(it) }

        /** Repository-relative source path of a top-level test class. */
        fun consumerPath(className: String): String =
            TEST_SOURCES + className.substringBefore('$').replace('.', '/') + ".kt"

        /**
         * Derives case ids with the gate's rules: `<key>/<id field>` for a list
         * group with an id field, `<key>/<index>` for one without, `<key>` for
         * an object group.
         */
        fun deriveCaseIds(groups: List<Group>, document: JsonElement): List<String> {
            val root = document as? JsonObject ?: throw ContractGateException("case_group_missing")
            val ids = mutableListOf<String>()
            for (group in groups) {
                when (val value = root[group.key]) {
                    is JsonObject -> ids += group.key
                    is JsonArray -> value.forEachIndexed { index, item ->
                        val part = if (group.idField == null) {
                            index.toString()
                        } else {
                            val raw = ((item as? JsonObject)?.get(group.idField) as? JsonPrimitive)
                                ?.takeIf { it.isString }?.content
                            if (raw == null || !CASE_PART.matches(raw)) {
                                throw ContractGateException("case_id_invalid")
                            }
                            raw
                        }
                        ids += "${group.key}/$part"
                    }
                    else -> throw ContractGateException("case_group_missing")
                }
            }
            if (ids.size != ids.toSet().size) {
                throw ContractGateException("case_id_duplicate")
            }
            return ids
        }
    }
}

/** One runtime result document under construction; see `contract_gate_result.schema.json`. */
internal class ContractGateResult private constructor(
    private val catalog: ContractGateCatalog,
    private val fixtures: Map<String, FixtureRecord>,
) {
    private class FixtureRecord(
        var digest: String,
        val caseIds: List<String>?,
        val cases: MutableMap<String, ContractGateOutcome> = linkedMapOf(),
        val tests: IntArray = IntArray(3),
    )

    /** Merges one test method's outcomes; the worse outcome of a case wins. */
    fun merge(observation: ContractGateObservation) {
        val record = fixtures[observation.fixtureId] ?: throw ContractGateException("unknown_fixture")
        val expected = catalog.fixture(observation.fixtureId).sha256
        val digest = observation.digest
        if (digest != null && record.digest == expected) {
            record.digest = digest
        }
        val known = record.caseIds?.toSet().orEmpty()
        for ((id, outcome) in observation.cases) {
            if (id in known) {
                record.cases[id] = maxOf(record.cases[id] ?: outcome, outcome)
            }
        }
        record.tests[testSlot(observation.test)] += 1
    }

    /** Pretty-printed JSON with a trailing newline, fixtures in manifest order. */
    fun toJson(): String = PRETTY.encodeToString(JsonElement.serializer(), toElement()) + "\n"

    private fun toElement(): JsonObject = buildJsonObject {
        put("schema", RESULT_SCHEMA)
        put("schema_version", 1)
        put("runtime", ContractGateCatalog.RUNTIME)
        put("manifest_sha256", catalog.manifestSha256)
        put(
            "fixtures",
            buildJsonArray {
                for (entry in catalog.required) {
                    add(fixtureElement(entry.id, fixtures.getValue(entry.id)))
                }
            },
        )
    }

    private fun fixtureElement(id: String, record: FixtureRecord): JsonObject = buildJsonObject {
        put("fixture_id", id)
        put("fixture_sha256", record.digest)
        val caseIds = record.caseIds
        val ran = record.tests.sum() > 0
        val placeholder = if (ran) NOT_EXERCISED else NOT_COLLECTED
        val outcomes = caseIds.orEmpty().map { it to record.cases[it] }
        put("passed", strings(outcomes.filter { it.second == ContractGateOutcome.PASSED }.map { it.first }))
        put(
            "failed",
            outcomes(
                outcomes.mapNotNull { (case, outcome) ->
                    when (outcome) {
                        null -> case to placeholder
                        ContractGateOutcome.FAILED, ContractGateOutcome.ERROR -> case to outcome.reason!!
                        else -> null
                    }
                },
            ),
        )
        put(
            "skipped",
            outcomes(
                outcomes.filter { it.second == ContractGateOutcome.SKIPPED }
                    .map { it.first to ContractGateOutcome.SKIPPED.reason!! },
            ),
        )
        if (caseIds == null) {
            put("errors", strings(listOf(FIXTURE_UNREADABLE)))
        }
        if (ran) {
            put(
                "tests",
                buildJsonObject {
                    put("passed", record.tests[0])
                    put("failed", record.tests[1])
                    put("skipped", record.tests[2])
                },
            )
        }
    }

    companion object {
        const val RESULT_SCHEMA = "openmed.contract_gate.result"
        const val NOT_COLLECTED = "consumer_not_collected"
        const val NOT_EXERCISED = "case_not_exercised"
        const val FIXTURE_UNREADABLE = "fixture_unreadable"
        private val ZERO_DIGEST = "0".repeat(64)
        private val PRETTY = Json { prettyPrint = true }

        /** A result where no consumer has reported yet. */
        fun empty(catalog: ContractGateCatalog): ContractGateResult = ContractGateResult(
            catalog,
            catalog.required.associate { entry ->
                entry.id to FixtureRecord(
                    digest = catalog.resourceDigest(entry.id) ?: ZERO_DIGEST,
                    caseIds = catalog.caseIds(entry.id),
                )
            },
        )

        /**
         * Rebuilds a result this reporter wrote earlier in the same run.
         *
         * @return `null` when [text] is malformed, belongs to another runtime or
         *   another manifest, so the caller starts over.
         */
        fun parse(catalog: ContractGateCatalog, text: String): ContractGateResult? {
            val document = try {
                Json.parseToJsonElement(text) as? JsonObject
            } catch (error: Exception) {
                null
            } ?: return null
            if (document.string("schema") != RESULT_SCHEMA ||
                document.string("runtime") != ContractGateCatalog.RUNTIME ||
                document.string("manifest_sha256") != catalog.manifestSha256
            ) {
                return null
            }
            val result = empty(catalog)
            val rows = document["fixtures"] as? JsonArray ?: return null
            for (row in rows) {
                val item = row as? JsonObject ?: return null
                val record = result.fixtures[item.string("fixture_id")] ?: return null
                item.string("fixture_sha256")?.let { record.digest = it }
                item.strings("passed").forEach { record.cases[it] = ContractGateOutcome.PASSED }
                for ((case, reason) in item.outcomes("failed") + item.outcomes("skipped")) {
                    ContractGateOutcome.values().firstOrNull { it.reason == reason }
                        ?.let { record.cases[case] = it }
                }
                (item["tests"] as? JsonObject)?.let { tests ->
                    listOf("passed", "failed", "skipped").forEachIndexed { slot, key ->
                        record.tests[slot] = (tests[key] as? JsonPrimitive)?.content?.toIntOrNull() ?: 0
                    }
                }
            }
            return result
        }

        private fun testSlot(outcome: ContractGateOutcome): Int = when (outcome) {
            ContractGateOutcome.PASSED -> 0
            ContractGateOutcome.FAILED, ContractGateOutcome.ERROR -> 1
            ContractGateOutcome.SKIPPED -> 2
        }

        private fun strings(values: List<String>): JsonArray = buildJsonArray {
            values.forEach { add(JsonPrimitive(it)) }
        }

        private fun outcomes(values: List<Pair<String, String>>): JsonArray = buildJsonArray {
            for ((case, reason) in values) {
                add(
                    buildJsonObject {
                        put("case_id", case)
                        put("reason", reason)
                    },
                )
            }
        }

        private fun JsonObject.strings(key: String): List<String> =
            (this[key] as? JsonArray).orEmpty().mapNotNull { (it as? JsonPrimitive)?.content }

        private fun JsonObject.outcomes(key: String): List<Pair<String, String>> =
            (this[key] as? JsonArray).orEmpty().mapNotNull { row ->
                val item = row as? JsonObject ?: return@mapNotNull null
                val case = item.string("case_id") ?: return@mapNotNull null
                val reason = item.string("reason") ?: return@mapNotNull null
                case to reason
            }
    }
}

/** Merges observations into `kotlin.json` under a file lock, replacing it atomically. */
internal object ContractGateResultFile {
    const val FILE_NAME = "kotlin.json"
    private const val LOCK_NAME = ".kotlin.json.lock"

    fun merge(directory: Path, catalog: ContractGateCatalog, observation: ContractGateObservation) {
        Files.createDirectories(directory)
        val target = directory.resolve(FILE_NAME)
        synchronized(this) {
            FileChannel.open(
                directory.resolve(LOCK_NAME),
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
            ).use { channel ->
                channel.lock().use {
                    val current = if (Files.isRegularFile(target)) {
                        ContractGateResult.parse(catalog, String(Files.readAllBytes(target), Charsets.UTF_8))
                    } else {
                        null
                    }
                    val result = current ?: ContractGateResult.empty(catalog)
                    result.merge(observation)
                    val temporary = Files.createTempFile(directory, ".kotlin", ".tmp")
                    try {
                        Files.write(temporary, result.toJson().toByteArray(Charsets.UTF_8))
                        Files.move(
                            temporary,
                            target,
                            StandardCopyOption.REPLACE_EXISTING,
                            StandardCopyOption.ATOMIC_MOVE,
                        )
                    } finally {
                        Files.deleteIfExists(temporary)
                    }
                }
            }
        }
    }
}

private fun JsonObject.string(key: String): String? =
    (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content

internal fun contractGateDigest(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 0xff) }
