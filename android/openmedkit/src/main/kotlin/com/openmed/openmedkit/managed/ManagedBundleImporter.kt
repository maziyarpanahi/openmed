package com.openmed.openmedkit.managed

import com.openmed.openmedkit.bundle.Ed25519SignatureVerifier
import com.openmed.openmedkit.bundle.Json
import com.openmed.openmedkit.bundle.PortableBundle
import com.openmed.openmedkit.bundle.PortableBundleContext
import com.openmed.openmedkit.bundle.PortableBundleException
import com.openmed.openmedkit.bundle.PortableBundleReason
import com.openmed.openmedkit.bundle.PortableBundleTrustPolicy
import com.openmed.openmedkit.bundle.PortableBundleTrustedKey
import com.openmed.openmedkit.bundle.Rfc8032Ed25519Verifier
import com.openmed.openmedkit.cache.BundleAcquireRequest
import com.openmed.openmedkit.cache.BundleCacheCrash
import com.openmed.openmedkit.cache.BundleCacheException
import com.openmed.openmedkit.cache.BundleCacheReason
import com.openmed.openmedkit.cache.BundleGenerationCache
import com.openmed.openmedkit.cache.BundleLease
import com.openmed.openmedkit.cache.BundleRangeSource
import com.openmed.openmedkit.cache.BundleRecoveryReport
import com.openmed.openmedkit.cache.DirectoryBundleRangeSource
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.URI
import java.net.URISyntaxException
import java.nio.charset.StandardCharsets
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.locks.ReentrantLock
import java.util.stream.Collectors
import kotlin.concurrent.withLock

/** Outcome of one managed import (`docs/mobile/managed-model-bundles.md`). */
enum class ManagedImportOutcome(val code: String) {
    /** Verified and committed as a new generation; not activated (stage policy). */
    IMPORTED("imported"),

    /** The generation was already committed (and active, when activation was requested). */
    ALREADY_PRESENT("already_present"),

    /** The generation is now active. */
    ACTIVATED("activated"),

    /** Nothing to import yet: no managed configuration, or the delivery is pending. */
    DEFERRED("deferred"),

    /** The activation check failed; the prior last-green generation is active again. */
    ROLLED_BACK("rolled_back"),

    /** The configuration version, handoff kind, or archive format is not supported. */
    UNSUPPORTED("unsupported"),

    /** Refused before activation; the cache is unchanged except for quarantine. */
    REJECTED("rejected"),
}

/** Stable, value-free reason codes of a managed import. */
enum class ManagedImportReason(val code: String) {
    MANAGED_CONFIG_ABSENT("managed_config_absent"),
    HANDOFF_PENDING("handoff_pending"),
    CONFIG_VERSION_UNSUPPORTED("config_version_unsupported"),
    HANDOFF_UNSUPPORTED("handoff_unsupported"),
    ARCHIVE_FORMAT_UNSUPPORTED("archive_format_unsupported"),
    CONFIG_INVALID("config_invalid"),
    TRUST_POLICY_MISSING("trust_policy_missing"),
    TRUST_POLICY_INVALID("trust_policy_invalid"),
    TRUST_ANCHOR_UNPINNED("trust_anchor_unpinned"),
    LOCATION_UNAPPROVED("location_unapproved"),
    HANDOFF_UNREADABLE("handoff_unreadable"),
    BUDGET_ARCHIVE_BYTES("budget_archive_bytes"),
    BUDGET_FILE_COUNT("budget_file_count"),
    BUDGET_TOTAL_BYTES("budget_total_bytes"),
    BUDGET_COMPRESSION_RATIO("budget_compression_ratio"),
    ARCHIVE_MALFORMED("archive_malformed"),
    ENTRY_UNSAFE("entry_unsafe"),
    ENTRY_DUPLICATE("entry_duplicate"),
    BUNDLE_REJECTED("bundle_rejected"),
    BUNDLE_NOT_EXPECTED("bundle_not_expected"),
    VERIFICATION_FAILED("verification_failed"),
    STORAGE_PRESSURE("storage_pressure"),
    QUOTA_EXCEEDED("quota_exceeded"),
    TRANSFER_INTERRUPTED("transfer_interrupted"),
    CANCELLED("cancelled"),
    LOCK_TIMEOUT("lock_timeout"),
    STORAGE_IO("storage_io"),
    ACTIVATION_CHECK_FAILED("activation_check_failed"),
    ;

    /** Outcome a refusal with this reason reports. */
    val refusalOutcome: ManagedImportOutcome
        get() = when (this) {
            MANAGED_CONFIG_ABSENT, HANDOFF_PENDING -> ManagedImportOutcome.DEFERRED
            CONFIG_VERSION_UNSUPPORTED, HANDOFF_UNSUPPORTED, ARCHIVE_FORMAT_UNSUPPORTED ->
                ManagedImportOutcome.UNSUPPORTED
            else -> ManagedImportOutcome.REJECTED
        }

    companion object {
        fun fromCode(code: String?): ManagedImportReason? = values().firstOrNull { it.code == code }
    }
}

/** How an imported generation is activated. */
enum class ManagedActivationPolicy(val code: String) {
    /** Activate and, after an optional activation check, mark last green. */
    ACTIVATE("activate"),

    /** Activate on probation; the host calls [ManagedBundleImporter.confirm] later. */
    PROBATION("probation"),

    /** Commit only; the active generation does not change. */
    STAGE("stage"),
}

/** Kind of local platform handoff that delivered the bundle. */
enum class ManagedHandoffKind(val code: String) {
    FILE("file"),
    DIRECTORY("directory"),
    STREAM("stream"),
    CONTENT_URI("content_uri"),
}

/** An approved local handle to a bundle. The importer never opens a network connection. */
sealed class ManagedBundleHandoff {
    /** A ZIP archive file or an extracted bundle directory in app-readable storage. */
    class LocalFile(val file: File) : ManagedBundleHandoff()

    /** Archive bytes; spooled into the importer's private area first. */
    class Stream(val open: () -> InputStream) : ManagedBundleHandoff()

    /** A `content://` URI opened through the importer's [ManagedContentStreamOpener]. */
    class ContentUri(val uri: String) : ManagedBundleHandoff()

    override fun toString(): String = javaClass.simpleName
}

/** Resource budgets enforced before any byte reaches the cache. */
data class ManagedImportLimits(
    /** Largest archive file or stream, in bytes. */
    val maxArchiveBytes: Long = 4_294_967_295L,
    /** Largest number of archive entries (files and directories). */
    val maxEntries: Int = 4_096,
    /** Largest sum of uncompressed entry sizes. */
    val maxTotalBytes: Long = 4_294_967_295L,
    /** Largest uncompressed-to-compressed ratio of one DEFLATE entry. */
    val maxCompressionRatio: Long = 200L,
) {
    init {
        require(maxArchiveBytes >= 0 && maxEntries > 0 && maxTotalBytes >= 0 && maxCompressionRatio > 0) {
            "limits must be positive"
        }
    }
}

/** Error carrying only a reason code and, where relevant, the verifier reason. */
class ManagedImportException(
    val reason: ManagedImportReason,
    val bundleReason: PortableBundleReason? = null,
) : Exception(reason.code) {
    val outcome: ManagedImportOutcome get() = reason.refusalOutcome
}

/**
 * A parsed managed configuration (Android app restrictions or Apple managed
 * app configuration). Keys are shared by both platforms.
 */
class ManagedBundleConfiguration(
    /** Trust roots delivered by the administrator. Public keys only. */
    val trustPolicy: PortableBundleTrustPolicy,
    /** `sha256:` fingerprints of raw Ed25519 public keys; empty means no pinning. */
    val keyPins: Set<String> = emptySet(),
    /** Where the delivered bundle is found: `content://`, `file:`, or a relative path. */
    val location: String? = null,
    val expectedBundleId: String? = null,
    val expectedContentId: String? = null,
    val minimumSequence: Long? = null,
    val activation: ManagedActivationPolicy? = null,
) {
    override fun toString(): String = "ManagedBundleConfiguration(keys=${trustPolicy.keys.size})"

    companion object {
        const val VERSION = 1
        const val VERSION_KEY = "openmed_bundle_config_version"
        const val TRUST_POLICY_KEY = "openmed_bundle_trust_policy"
        const val KEY_PINS_KEY = "openmed_bundle_key_pins"
        const val LOCATION_KEY = "openmed_bundle_location"
        const val EXPECTED_ID_KEY = "openmed_bundle_expected_id"
        const val EXPECTED_CONTENT_ID_KEY = "openmed_bundle_expected_content_id"
        const val MIN_SEQUENCE_KEY = "openmed_bundle_min_sequence"
        const val ACTIVATION_KEY = "openmed_bundle_activation"

        /** Every key the importer reads; other `openmed_bundle_` keys are refused. */
        val KEYS: List<String> = listOf(
            VERSION_KEY, TRUST_POLICY_KEY, KEY_PINS_KEY, LOCATION_KEY, EXPECTED_ID_KEY, EXPECTED_CONTENT_ID_KEY,
            MIN_SEQUENCE_KEY, ACTIVATION_KEY,
        )
        private const val PREFIX = "openmed_bundle_"
        private val SHA256 = Regex("sha256:[0-9a-f]{64}")
        private val BUNDLE_ID = Regex("[a-z0-9][a-z0-9._-]{0,127}")

        /**
         * Parses managed values. Keys without the `openmed_bundle_` prefix are
         * ignored; the checks run in the documented order.
         *
         * @throws ManagedImportException with `config_invalid`,
         *   `config_version_unsupported`, `trust_policy_missing`, or
         *   `trust_policy_invalid`.
         */
        fun fromManagedValues(values: Map<String, Any?>): ManagedBundleConfiguration {
            val invalid = { ManagedImportException(ManagedImportReason.CONFIG_INVALID) }
            val version = integer(values[VERSION_KEY] ?: throw invalid()) ?: throw invalid()
            if (version != VERSION.toLong()) throw ManagedImportException(ManagedImportReason.CONFIG_VERSION_UNSUPPORTED)
            if (values.keys.any { it.startsWith(PREFIX) && it !in KEYS }) throw invalid()
            val rawPolicy = values[TRUST_POLICY_KEY] ?: throw ManagedImportException(
                ManagedImportReason.TRUST_POLICY_MISSING,
            )
            val policy = (rawPolicy as? String)
                ?.takeIf { it.toByteArray(StandardCharsets.UTF_8).size <= 64 shl 10 }
                ?.let { text ->
                    try {
                        PortableBundleTrustPolicy.fromJson(text)
                    } catch (error: PortableBundleException) {
                        null
                    }
                }
                ?: throw ManagedImportException(ManagedImportReason.TRUST_POLICY_INVALID)
            val pins = LinkedHashSet<String>()
            values[KEY_PINS_KEY]?.let { raw ->
                val text = raw as? String ?: throw invalid()
                for (token in text.split(Regex("[\\s,]+")).filter { it.isNotEmpty() }) {
                    if (!SHA256.matches(token)) throw invalid()
                    pins += token
                }
            }
            fun optional(key: String, valid: (String) -> Boolean): String? {
                val raw = values[key] ?: return null
                val text = raw as? String ?: throw invalid()
                if (text.isEmpty()) return null
                if (!valid(text)) throw invalid()
                return text
            }
            val location = optional(LOCATION_KEY) { it.toByteArray(StandardCharsets.UTF_8).size <= 1024 }
            val expectedId = optional(EXPECTED_ID_KEY) { BUNDLE_ID.matches(it) }
            val expectedContent = optional(EXPECTED_CONTENT_ID_KEY) { SHA256.matches(it) }
            val minimum = values[MIN_SEQUENCE_KEY]?.let { raw ->
                integer(raw)?.takeIf { it in 0..PortableBundle.MAX_SAFE_INTEGER } ?: throw invalid()
            }
            val activation = optional(ACTIVATION_KEY) { text -> ManagedActivationPolicy.values().any { it.code == text } }
                ?.let { text -> ManagedActivationPolicy.values().first { it.code == text } }
            return ManagedBundleConfiguration(policy, pins, location, expectedId, expectedContent, minimum, activation)
        }

        /** The pin of a trust-policy key: `sha256:` of its raw 32-byte public key. */
        fun pin(key: PortableBundleTrustedKey): String =
            "sha256:" + MessageDigest.getInstance("SHA-256").digest(key.publicKey).joinToString("") {
                "%02x".format(it)
            }

        private fun integer(value: Any): Long? = when (value) {
            is Int -> value.toLong()
            is Long -> value
            is Short -> value.toLong()
            is Byte -> value.toLong()
            else -> null
        }
    }
}

/** Supplies the managed configuration, or `null` when none is installed. */
fun interface ManagedConfigurationSource {
    fun managedConfiguration(): Map<String, Any?>?
}

/** Opens a `content://` URI as a stream. Throws [FileNotFoundException] when not yet delivered. */
fun interface ManagedContentStreamOpener {
    fun open(uri: String): InputStream
}

/** One managed import. */
class ManagedImportRequest(
    val handoff: ManagedBundleHandoff,
    val configuration: ManagedBundleConfiguration,
    /** Verification instant, device target, caller floors, and manifest limits. */
    val context: PortableBundleContext,
    /** Pins compiled into the app; each non-empty pin set must cover every trust root. */
    val keyPins: Set<String> = emptySet(),
    val limits: ManagedImportLimits = ManagedImportLimits(),
    /** Used when the configuration does not set an activation policy. */
    val activation: ManagedActivationPolicy = ManagedActivationPolicy.ACTIVATE,
    /** Optional check on the newly active generation; `false` rolls back to last green. */
    val activationCheck: ((BundleLease) -> Boolean)? = null,
    /** Checked while spooling and after every committed chunk. */
    val isCancelled: () -> Boolean = { false },
    val verifier: Ed25519SignatureVerifier = Rfc8032Ed25519Verifier,
)

/**
 * Bounded, value-free receipt of one managed import
 * (`openmed.managed_import.receipt` v1). It holds opaque identities, a sequence
 * number, counts, a digest, and reason codes only.
 */
data class ManagedImportReceipt(
    val outcome: ManagedImportOutcome,
    val reason: ManagedImportReason?,
    val bundleReason: PortableBundleReason?,
    val handoff: ManagedHandoffKind?,
    /** Opaque bundle key; present once the manifest authenticated. */
    val bundleKey: String?,
    val generation: String?,
    val sequence: Long?,
    val previousActive: String?,
    val active: String?,
    val lastGreen: String?,
    val filesVerified: Int,
    val bytesStaged: Long,
    val resumed: Boolean,
    /** Binding digest of the trust policy used ([BundleGenerationCache.trustDigest]). */
    val trustDigest: String?,
    val evicted: List<String>,
) {
    /** Canonical JSON of the receipt, safe to persist or report as managed feedback. */
    fun toJson(): String = String(Json.encode(json()), StandardCharsets.UTF_8)

    internal fun json(): Json {
        fun text(value: String?): Json = value?.let { Json.Str(it) } ?: Json.Null
        return Json.Obj(
            listOf(
                "active" to text(active),
                "bundle_key" to text(bundleKey),
                "bundle_reason" to text(bundleReason?.code),
                "bytes_staged" to Json.Num(bytesStaged),
                "evicted" to Json.Arr(evicted.map { Json.Str(it) }),
                "files_verified" to Json.Num(filesVerified.toLong()),
                "generation" to text(generation),
                "handoff" to text(handoff?.code),
                "last_green" to text(lastGreen),
                "outcome" to Json.Str(outcome.code),
                "previous_active" to text(previousActive),
                "reason" to text(reason?.code),
                "resumed" to Json.Bool(resumed),
                "schema" to Json.Str(SCHEMA),
                "schema_version" to Json.Num(SCHEMA_VERSION.toLong()),
                "sequence" to (sequence?.let { Json.Num(it) } ?: Json.Null),
                "trust_digest" to text(trustDigest),
            ),
        )
    }

    companion object {
        const val SCHEMA = "openmed.managed_import.receipt"
        const val SCHEMA_VERSION = 1

        internal fun fromJson(value: Json): ManagedImportReceipt? {
            val members = value as? Json.Obj ?: return null
            fun text(key: String): String? = (members[key] as? Json.Str)?.value
            fun number(key: String): Long? = (members[key] as? Json.Num)?.value
            if (text("schema") != SCHEMA) return null
            val outcome = ManagedImportOutcome.values().firstOrNull { it.code == text("outcome") } ?: return null
            return ManagedImportReceipt(
                outcome = outcome,
                reason = ManagedImportReason.fromCode(text("reason")),
                bundleReason = PortableBundleReason.values().firstOrNull { it.code == text("bundle_reason") },
                handoff = ManagedHandoffKind.values().firstOrNull { it.code == text("handoff") },
                bundleKey = text("bundle_key"),
                generation = text("generation"),
                sequence = number("sequence"),
                previousActive = text("previous_active"),
                active = text("active"),
                lastGreen = text("last_green"),
                filesVerified = (number("files_verified") ?: 0L).toInt(),
                bytesStaged = number("bytes_staged") ?: 0L,
                resumed = (members["resumed"] as? Json.Bool)?.value ?: false,
                trustDigest = text("trust_digest"),
                evicted = (members["evicted"] as? Json.Arr)?.items?.mapNotNull { (it as? Json.Str)?.value }
                    ?: emptyList(),
            )
        }
    }
}

/** Result of [ManagedBundleImporter.recover]. */
data class ManagedImportRecovery(
    val cache: BundleRecoveryReport,
    /** Spooled stream copies left by an interrupted import and now deleted. */
    val spoolPurged: Int,
)

/**
 * Imports signed portable bundles delivered through managed, local platform
 * handoffs, verifies them fully offline against administrator trust roots, and
 * activates or rolls them back atomically through [BundleGenerationCache].
 *
 * Accepted handoffs are local files and directories, streams, and `content://`
 * URIs opened through [contentOpener]. The importer never opens a network
 * connection, never executes bundled content, and never extracts an archive by
 * its own entry names. It keeps a small state file with per-bundle sequence
 * floors and the last receipts, all value-free. Imports through one state
 * directory are serialized. Mirrors `OpenMedManagedBundleImporter` in
 * OpenMedKit for Swift.
 *
 * @param cache the bundle generation cache that stages and activates.
 * @param stateDirectory app-private directory for the state file and the stream spool.
 * @param contentOpener opens `content://` handoffs, for example [ContentResolverStreamOpener].
 */
class ManagedBundleImporter(
    private val cache: BundleGenerationCache,
    stateDirectory: File,
    private val contentOpener: ManagedContentStreamOpener? = null,
) {
    private val stateDirectory: File = stateDirectory.also { it.mkdirs() }.canonicalFile
    private val mutex: ReentrantLock = MUTEXES.computeIfAbsent(this.stateDirectory.path) { ReentrantLock() }

    /** Test seam: called at the importer's named crash checkpoints. */
    internal var checkpointHook: ((String) -> Unit)? = null

    private val spoolDirectory: File get() = File(stateDirectory, "spool")
    private val stateFile: File get() = File(stateDirectory, "managed-import.json")

    // region Public API

    /**
     * Reads the managed configuration and imports the bundle it names, or the
     * explicit [handoff] (for example a stream from a managed file provider).
     *
     * A configured `content://` location must use one of [approvedAuthorities].
     * A relative path resolves against [approvedRoots] in order, and a `file:`
     * URI must lie inside one of them. Any other scheme is `unsupported`
     * (`handoff_unsupported`).
     */
    fun importManaged(
        source: ManagedConfigurationSource,
        context: PortableBundleContext,
        approvedRoots: List<File> = emptyList(),
        approvedAuthorities: Set<String> = emptySet(),
        handoff: ManagedBundleHandoff? = null,
        keyPins: Set<String> = emptySet(),
        limits: ManagedImportLimits = ManagedImportLimits(),
        activation: ManagedActivationPolicy = ManagedActivationPolicy.ACTIVATE,
        activationCheck: ((BundleLease) -> Boolean)? = null,
        isCancelled: () -> Boolean = { false },
        verifier: Ed25519SignatureVerifier = Rfc8032Ed25519Verifier,
    ): ManagedImportReceipt {
        val values = source.managedConfiguration() ?: return record(refusal(ManagedImportReason.MANAGED_CONFIG_ABSENT))
        val configuration = try {
            ManagedBundleConfiguration.fromManagedValues(values)
        } catch (error: ManagedImportException) {
            return record(refusal(error.reason))
        }
        val resolved = handoff ?: try {
            resolve(configuration.location, approvedRoots, approvedAuthorities)
        } catch (error: ManagedImportException) {
            return record(refusal(error.reason, kindOf(configuration.location)))
        }
        return importBundle(
            ManagedImportRequest(
                resolved, configuration, context, keyPins, limits, activation, activationCheck, isCancelled, verifier,
            ),
        )
    }

    /** Imports one bundle delivered through an approved local handoff. */
    fun importBundle(request: ManagedImportRequest): ManagedImportReceipt = mutex.withLock {
        val draft = Draft(handoff = kind(request.handoff))
        val spooled = arrayOfNulls<File>(1)
        try {
            importLocked(request, draft, spooled)
        } catch (crash: BundleCacheCrash) {
            // A simulated process death leaves the spool for recover().
            throw crash
        } catch (error: ManagedImportException) {
            draft.outcome = error.outcome
            draft.reason = error.reason
            draft.bundleReason = error.bundleReason
        } catch (error: BundleCacheException) {
            draft.outcome = ManagedImportOutcome.REJECTED
            draft.reason = reason(error.reason)
        } catch (error: Exception) {
            draft.outcome = ManagedImportOutcome.REJECTED
            draft.reason = ManagedImportReason.STORAGE_IO
        }
        spooled[0]?.delete()
        recordLocked(draft.receipt())
    }

    /**
     * Confirms a probation activation: the active generation becomes last green
     * and its sequence becomes the bundle's floor.
     */
    fun confirm(bundleId: String) = mutex.withLock {
        cache.confirm(bundleId)
        val active = cache.snapshot(bundleId).active ?: return@withLock
        val sequence = cache.sequence(bundleId, active) ?: return@withLock
        val state = readState()
        val key = BundleGenerationCache.bundleKey(bundleId)
        state.floors[key] = maxOf(state.floors[key] ?: 0L, sequence)
        writeState(state)
    }

    /**
     * Crash recovery: runs [BundleGenerationCache.recover] and deletes stream
     * spools left by an interrupted import. Idempotent; call at startup.
     */
    fun recover(): ManagedImportRecovery = mutex.withLock {
        val report = cache.recover()
        var purged = 0
        spoolDirectory.listFiles()?.sortedBy { it.name }?.forEach { if (it.delete()) purged++ }
        ManagedImportRecovery(report, purged)
    }

    /** Recorded receipts, oldest first (at most [RECEIPT_CAPACITY]). */
    fun receipts(): List<ManagedImportReceipt> = mutex.withLock { readState().receipts.toList() }

    /** Highest sequence accepted as last green for a bundle, if any. */
    fun sequenceFloor(bundleId: String): Long? =
        mutex.withLock { readState().floors[BundleGenerationCache.bundleKey(bundleId)] }

    override fun toString(): String = "ManagedBundleImporter"

    // endregion

    // region Locations

    private fun kindOf(location: String?): ManagedHandoffKind =
        if (location?.startsWith("content:", ignoreCase = true) == true) {
            ManagedHandoffKind.CONTENT_URI
        } else {
            ManagedHandoffKind.FILE
        }

    private fun resolve(location: String?, roots: List<File>, authorities: Set<String>): ManagedBundleHandoff {
        location ?: throw ManagedImportException(ManagedImportReason.HANDOFF_PENDING)
        val canonicalRoots = roots.map { it.canonicalFile }
        val scheme = location.substringBefore(':', missingDelimiterValue = "")
        if (scheme.isNotEmpty() && scheme.all { it.isLetterOrDigit() || it in "+.-" }) {
            val uri = try {
                URI(location)
            } catch (error: URISyntaxException) {
                throw ManagedImportException(ManagedImportReason.HANDOFF_UNSUPPORTED)
            }
            when (scheme.lowercase()) {
                "content" -> {
                    if (uri.authority == null || uri.authority !in authorities) {
                        throw ManagedImportException(ManagedImportReason.LOCATION_UNAPPROVED)
                    }
                    return ManagedBundleHandoff.ContentUri(location)
                }
                "file" -> {
                    val file = try {
                        File(uri).canonicalFile
                    } catch (error: IllegalArgumentException) {
                        throw ManagedImportException(ManagedImportReason.LOCATION_UNAPPROVED)
                    }
                    if (canonicalRoots.none { file.path.startsWith(it.path + File.separator) }) {
                        throw ManagedImportException(ManagedImportReason.LOCATION_UNAPPROVED)
                    }
                    if (!file.exists()) throw ManagedImportException(ManagedImportReason.HANDOFF_PENDING)
                    return ManagedBundleHandoff.LocalFile(file)
                }
                else -> throw ManagedImportException(ManagedImportReason.HANDOFF_UNSUPPORTED)
            }
        }
        if (!BundleArchive.isSafePath(location)) throw ManagedImportException(ManagedImportReason.LOCATION_UNAPPROVED)
        for (root in canonicalRoots) {
            val candidate = File(root, location)
            if (!candidate.exists()) continue
            // A symbolic link inside an approved directory must not lead out of it.
            val resolved = candidate.canonicalFile
            if (!resolved.path.startsWith(root.path + File.separator)) {
                throw ManagedImportException(ManagedImportReason.LOCATION_UNAPPROVED)
            }
            return ManagedBundleHandoff.LocalFile(resolved)
        }
        throw ManagedImportException(ManagedImportReason.HANDOFF_PENDING)
    }

    // endregion

    // region Import

    private fun importLocked(request: ManagedImportRequest, draft: Draft, spooled: Array<File?>) {
        val configuration = request.configuration
        val policy = configuration.trustPolicy
        // Every delivered trust root must appear in every non-empty pin set.
        for (pins in listOf(configuration.keyPins, request.keyPins)) {
            if (pins.isNotEmpty() && !policy.keys.all { ManagedBundleConfiguration.pin(it) in pins }) {
                throw ManagedImportException(ManagedImportReason.TRUST_ANCHOR_UNPINNED)
            }
        }
        draft.trustDigest = BundleGenerationCache.trustDigest(policy)
        if (request.isCancelled()) throw ManagedImportException(ManagedImportReason.CANCELLED)

        val file: File = when (val handoff = request.handoff) {
            is ManagedBundleHandoff.LocalFile -> handoff.file
            is ManagedBundleHandoff.Stream -> {
                val stream = try {
                    handoff.open()
                } catch (error: IOException) {
                    throw ManagedImportException(ManagedImportReason.HANDOFF_UNREADABLE)
                }
                spool(stream, request, spooled)
            }
            is ManagedBundleHandoff.ContentUri -> {
                if (!handoff.uri.startsWith("content:", ignoreCase = true)) {
                    throw ManagedImportException(ManagedImportReason.HANDOFF_UNSUPPORTED)
                }
                val opener = contentOpener ?: throw ManagedImportException(ManagedImportReason.HANDOFF_UNSUPPORTED)
                val stream = try {
                    opener.open(handoff.uri)
                } catch (error: FileNotFoundException) {
                    throw ManagedImportException(ManagedImportReason.HANDOFF_PENDING)
                } catch (error: IOException) {
                    throw ManagedImportException(ManagedImportReason.HANDOFF_UNREADABLE)
                } catch (error: SecurityException) {
                    throw ManagedImportException(ManagedImportReason.HANDOFF_UNREADABLE)
                }
                spool(stream, request, spooled)
            }
        }
        if (!file.exists()) throw ManagedImportException(ManagedImportReason.HANDOFF_PENDING)
        val delivery: Delivery = if (file.isDirectory) {
            if (request.handoff is ManagedBundleHandoff.LocalFile) draft.handoff = ManagedHandoffKind.DIRECTORY
            DirectoryDelivery(file.canonicalFile, request.context)
        } else {
            ArchiveDelivery(file, request.limits, request.context)
        }
        delivery.use { located -> importDelivery(request, located, draft) }
    }

    private fun importDelivery(request: ManagedImportRequest, delivery: Delivery, draft: Draft) {
        val configuration = request.configuration
        val policy = configuration.trustPolicy
        // Authenticate the manifest before trusting anything it declares.
        val header = BundleGenerationCache.ManifestHeader.parse(delivery.manifest)
        var context = request.context
        if (header != null) {
            val key = BundleGenerationCache.bundleKey(header.bundleId)
            val floor = maxOf(
                context.sequenceFloors[header.bundleId] ?: 0L,
                configuration.minimumSequence ?: 0L,
                readState().floors[key] ?: 0L,
            )
            if (floor > 0) context = context.copy(sequenceFloors = context.sequenceFloors + (header.bundleId to floor))
        }
        val verification = PortableBundle.verify(
            delivery.manifest, delivery.envelope, policy, context, null, request.verifier,
        )
        val contentId = verification.contentId
        if (!verification.ok || contentId == null || header == null) {
            throw ManagedImportException(
                ManagedImportReason.BUNDLE_REJECTED,
                if (verification.ok) PortableBundleReason.MANIFEST_MALFORMED else verification.reason,
            )
        }
        val generation = BundleGenerationCache.generationId(contentId)
        draft.bundleKey = BundleGenerationCache.bundleKey(header.bundleId)
        draft.generation = generation
        draft.sequence = header.sequence
        val before = pointer(header.bundleId)
        draft.previousActive = before.first
        draft.active = before.first
        draft.lastGreen = before.second

        if (configuration.expectedBundleId != null && configuration.expectedBundleId != header.bundleId) {
            throw ManagedImportException(ManagedImportReason.BUNDLE_NOT_EXPECTED)
        }
        if (configuration.expectedContentId != null && configuration.expectedContentId != contentId) {
            throw ManagedImportException(ManagedImportReason.BUNDLE_NOT_EXPECTED)
        }
        delivery.inventory(header.files)?.let { reason ->
            throw ManagedImportException(ManagedImportReason.VERIFICATION_FAILED, reason)
        }

        val activation = configuration.activation ?: request.activation
        val source = delivery.source("openmed.managed_import:$contentId")
        val result = cache.acquire(
            BundleAcquireRequest(
                manifest = delivery.manifest,
                envelope = delivery.envelope,
                source = source,
                trustPolicy = policy,
                context = context,
                activate = activation != ManagedActivationPolicy.STAGE,
                confirm = activation == ManagedActivationPolicy.ACTIVATE && request.activationCheck == null,
                isCancelled = request.isCancelled,
                verifier = request.verifier,
            ),
        )
        if (!result.ok) {
            if ((source as? ArchiveBundleRangeSource)?.streamFailed == true) {
                throw ManagedImportException(ManagedImportReason.ARCHIVE_MALFORMED)
            }
            throw ManagedImportException(reason(result.reason), result.bundleReason)
        }
        draft.filesVerified = result.filesVerified
        draft.bytesStaged = result.bytesTransferred
        draft.resumed = result.resumed
        draft.evicted = result.evicted

        when (activation) {
            ManagedActivationPolicy.STAGE ->
                draft.outcome = if (result.fromCache) ManagedImportOutcome.ALREADY_PRESENT else ManagedImportOutcome.IMPORTED
            ManagedActivationPolicy.ACTIVATE, ManagedActivationPolicy.PROBATION -> {
                val unchanged = result.fromCache && before.first == generation &&
                    (before.second == generation || activation == ManagedActivationPolicy.PROBATION)
                val check = request.activationCheck
                if (unchanged) {
                    draft.outcome = ManagedImportOutcome.ALREADY_PRESENT
                } else if (check != null) {
                    val healthy = cache.lease(header.bundleId, generation).use { lease -> check(lease) }
                    if (!healthy) {
                        cache.withdraw(header.bundleId, generation)
                        val after = pointer(header.bundleId)
                        draft.active = after.first
                        draft.lastGreen = after.second
                        draft.outcome =
                            if (after.first == null) ManagedImportOutcome.REJECTED else ManagedImportOutcome.ROLLED_BACK
                        draft.reason = ManagedImportReason.ACTIVATION_CHECK_FAILED
                        return
                    }
                    if (activation == ManagedActivationPolicy.ACTIVATE) cache.confirm(header.bundleId)
                    draft.outcome = ManagedImportOutcome.ACTIVATED
                } else {
                    draft.outcome = ManagedImportOutcome.ACTIVATED
                }
            }
        }
        val after = pointer(header.bundleId)
        draft.active = after.first
        draft.lastGreen = after.second
        if (after.second == generation) {
            val state = readState()
            val key = BundleGenerationCache.bundleKey(header.bundleId)
            state.floors[key] = maxOf(state.floors[key] ?: 0L, header.sequence)
            writeState(state)
        }
    }

    private fun pointer(bundleId: String): Pair<String?, String?> = try {
        val snapshot = cache.snapshot(bundleId)
        snapshot.active to snapshot.lastGreen
    } catch (error: BundleCacheException) {
        throw ManagedImportException(reason(error.reason))
    }

    private fun spool(stream: InputStream, request: ManagedImportRequest, spooled: Array<File?>): File {
        spoolDirectory.mkdirs()
        val file = File(spoolDirectory, UUID.randomUUID().toString() + ".zip")
        spooled[0] = file
        stream.use { input ->
            try {
                FileOutputStream(file).use { output ->
                    val buffer = ByteArray(1 shl 20)
                    var total = 0L
                    while (true) {
                        if (request.isCancelled()) throw ManagedImportException(ManagedImportReason.CANCELLED)
                        val count = try {
                            input.read(buffer)
                        } catch (error: IOException) {
                            throw ManagedImportException(ManagedImportReason.TRANSFER_INTERRUPTED)
                        }
                        if (count < 0) break
                        total += count
                        if (total > request.limits.maxArchiveBytes) {
                            throw ManagedImportException(ManagedImportReason.BUDGET_ARCHIVE_BYTES)
                        }
                        output.write(buffer, 0, count)
                    }
                    output.fd.sync()
                }
            } catch (error: IOException) {
                throw ManagedImportException(ManagedImportReason.STORAGE_IO)
            }
        }
        checkpointHook?.invoke("archive_spooled")
        return file
    }

    private fun kind(handoff: ManagedBundleHandoff): ManagedHandoffKind = when (handoff) {
        is ManagedBundleHandoff.LocalFile -> ManagedHandoffKind.FILE
        is ManagedBundleHandoff.Stream -> ManagedHandoffKind.STREAM
        is ManagedBundleHandoff.ContentUri -> ManagedHandoffKind.CONTENT_URI
    }

    private fun refusal(reason: ManagedImportReason, handoff: ManagedHandoffKind? = null): ManagedImportReceipt =
        Draft(handoff = handoff, outcome = reason.refusalOutcome, reason = reason).receipt()

    // endregion

    // region State

    private class State(val floors: MutableMap<String, Long>, val receipts: MutableList<ManagedImportReceipt>)

    private fun readState(): State {
        val empty = State(mutableMapOf(), mutableListOf())
        val root = try {
            Json.parse(stateFile.readBytes()) as? Json.Obj
        } catch (error: IOException) {
            null
        } catch (error: Json.Invalid) {
            null
        } ?: return empty
        if (root["schema"] != Json.Str(STATE_SCHEMA)) return empty
        val floors = mutableMapOf<String, Long>()
        (root["sequence_floors"] as? Json.Obj)?.members?.forEach { (key, value) ->
            (value as? Json.Num)?.let { floors[key] = it.value }
        }
        val receipts = (root["receipts"] as? Json.Arr)?.items?.mapNotNull { ManagedImportReceipt.fromJson(it) }
            ?: emptyList()
        return State(floors, receipts.toMutableList())
    }

    private fun writeState(state: State) {
        val value = Json.Obj(
            listOf(
                "receipts" to Json.Arr(state.receipts.takeLast(RECEIPT_CAPACITY).map { it.json() }),
                "schema" to Json.Str(STATE_SCHEMA),
                "schema_version" to Json.Num(1),
                "sequence_floors" to Json.Obj(state.floors.toSortedMap().map { (key, floor) -> key to Json.Num(floor) }),
            ),
        )
        val temporary = File(stateDirectory, ".managed-import.json." + UUID.randomUUID())
        try {
            FileOutputStream(temporary).use { output ->
                output.write(Json.encode(value))
                output.fd.sync()
            }
            try {
                Files.move(
                    temporary.toPath(), stateFile.toPath(), StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (error: AtomicMoveNotSupportedException) {
                Files.move(temporary.toPath(), stateFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            temporary.delete()
        }
    }

    private fun record(receipt: ManagedImportReceipt): ManagedImportReceipt = mutex.withLock { recordLocked(receipt) }

    private fun recordLocked(receipt: ManagedImportReceipt): ManagedImportReceipt {
        val state = readState()
        state.receipts += receipt
        while (state.receipts.size > RECEIPT_CAPACITY) state.receipts.removeAt(0)
        try {
            writeState(state)
        } catch (error: IOException) {
            // The receipt is still returned; only the persisted copy is lost.
        }
        return receipt
    }

    private class Draft(
        var handoff: ManagedHandoffKind? = null,
        var outcome: ManagedImportOutcome = ManagedImportOutcome.REJECTED,
        var reason: ManagedImportReason? = null,
        var bundleReason: PortableBundleReason? = null,
        var bundleKey: String? = null,
        var generation: String? = null,
        var sequence: Long? = null,
        var previousActive: String? = null,
        var active: String? = null,
        var lastGreen: String? = null,
        var filesVerified: Int = 0,
        var bytesStaged: Long = 0L,
        var resumed: Boolean = false,
        var trustDigest: String? = null,
        var evicted: List<String> = emptyList(),
    ) {
        fun receipt() = ManagedImportReceipt(
            outcome, reason, bundleReason, handoff, bundleKey, generation, sequence, previousActive, active,
            lastGreen, filesVerified, bytesStaged, resumed, trustDigest, evicted,
        )
    }

    // endregion

    companion object {
        /** Number of receipts kept in the state file. */
        const val RECEIPT_CAPACITY = 16
        private const val STATE_SCHEMA = "openmed.managed_import.state"
        private val MUTEXES = ConcurrentHashMap<String, ReentrantLock>()

        private fun reason(reason: BundleCacheReason?): ManagedImportReason = when (reason) {
            BundleCacheReason.BUNDLE_REJECTED -> ManagedImportReason.BUNDLE_REJECTED
            BundleCacheReason.LOCK_TIMEOUT -> ManagedImportReason.LOCK_TIMEOUT
            BundleCacheReason.QUOTA_EXCEEDED -> ManagedImportReason.QUOTA_EXCEEDED
            BundleCacheReason.STORAGE_PRESSURE -> ManagedImportReason.STORAGE_PRESSURE
            BundleCacheReason.TRANSFER_INTERRUPTED, BundleCacheReason.RANGE_MISMATCH ->
                ManagedImportReason.TRANSFER_INTERRUPTED
            BundleCacheReason.CANCELLED -> ManagedImportReason.CANCELLED
            BundleCacheReason.VERIFICATION_FAILED, BundleCacheReason.GENERATION_CORRUPT ->
                ManagedImportReason.VERIFICATION_FAILED
            else -> ManagedImportReason.STORAGE_IO
        }
    }
}

// region Deliveries

/** A located bundle: its manifest, envelope, inventory, and a range source. */
private interface Delivery : java.io.Closeable {
    val manifest: ByteArray
    val envelope: ByteArray

    /** First inventory refusal against the declared files, in manifest order. */
    fun inventory(declared: List<BundleGenerationCache.DeclaredFile>): PortableBundleReason?

    fun source(identity: String): BundleRangeSource

    override fun close() = Unit
}

private fun inventoryReason(
    declared: List<BundleGenerationCache.DeclaredFile>,
    sizes: Map<String, Long>,
    order: List<String>,
): PortableBundleReason? {
    for (file in declared) {
        val size = sizes[file.path] ?: return PortableBundleReason.FILE_MISSING
        if (size != file.size) return PortableBundleReason.FILE_SIZE_MISMATCH
    }
    val known = declared.map { it.path }.toSet() + PortableBundle.MANIFEST_FILENAME + PortableBundle.ENVELOPE_FILENAME
    return if (order.any { it !in known }) PortableBundleReason.FILE_UNDECLARED else null
}

private class ArchiveDelivery(file: File, limits: ManagedImportLimits, context: PortableBundleContext) : Delivery {
    private val archive: BundleArchive = try {
        BundleArchive(file, limits)
    } catch (failure: ArchiveFailure) {
        throw ManagedImportException(failure.reason)
    }
    override val manifest: ByteArray
    override val envelope: ByteArray

    init {
        try {
            manifest = document(
                PortableBundle.MANIFEST_FILENAME, context.maxManifestBytes, PortableBundleReason.MANIFEST_TOO_LARGE,
            )
            envelope = document(
                PortableBundle.ENVELOPE_FILENAME, context.maxEnvelopeBytes, PortableBundleReason.ENVELOPE_TOO_LARGE,
            )
        } catch (error: Throwable) {
            archive.close()
            throw error
        }
    }

    private fun document(name: String, limit: Int, tooLarge: PortableBundleReason): ByteArray {
        val entry = archive.files[name]
            ?: throw ManagedImportException(ManagedImportReason.BUNDLE_REJECTED, PortableBundleReason.FILE_MISSING)
        if (entry.size > limit) throw ManagedImportException(ManagedImportReason.BUNDLE_REJECTED, tooLarge)
        return try {
            archive.read(entry)
        } catch (failure: ArchiveFailure) {
            throw ManagedImportException(failure.reason)
        }
    }

    override fun inventory(declared: List<BundleGenerationCache.DeclaredFile>): PortableBundleReason? {
        val order = archive.entries.filter { !it.isDirectory }.map { it.name.substring(archive.root.length) }
        return inventoryReason(declared, archive.files.mapValues { it.value.size }, order)
    }

    override fun source(identity: String): BundleRangeSource = ArchiveBundleRangeSource(archive, identity)

    override fun close() = archive.close()
}

private class DirectoryDelivery(private val root: File, context: PortableBundleContext) : Delivery {
    private val sizes = LinkedHashMap<String, Long>()
    override val manifest: ByteArray
    override val envelope: ByteArray

    init {
        val base = root.toPath()
        val paths = try {
            Files.walk(base).use { stream -> stream.sorted().collect(Collectors.toList()) }
        } catch (error: IOException) {
            throw ManagedImportException(ManagedImportReason.HANDOFF_UNREADABLE)
        }
        for (path in paths) {
            if (path == base) continue
            val relative = base.relativize(path).toString().replace(File.separatorChar, '/')
            when {
                Files.isSymbolicLink(path) -> throw ManagedImportException(ManagedImportReason.ENTRY_UNSAFE)
                Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS) -> Unit
                Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) -> sizes[relative] = Files.size(path)
                else -> throw ManagedImportException(ManagedImportReason.ENTRY_UNSAFE)
            }
        }
        manifest = document(
            PortableBundle.MANIFEST_FILENAME, context.maxManifestBytes, PortableBundleReason.MANIFEST_TOO_LARGE,
        )
        envelope = document(
            PortableBundle.ENVELOPE_FILENAME, context.maxEnvelopeBytes, PortableBundleReason.ENVELOPE_TOO_LARGE,
        )
    }

    private fun document(name: String, limit: Int, tooLarge: PortableBundleReason): ByteArray {
        val size = sizes[name]
            ?: throw ManagedImportException(ManagedImportReason.BUNDLE_REJECTED, PortableBundleReason.FILE_MISSING)
        if (size > limit) throw ManagedImportException(ManagedImportReason.BUNDLE_REJECTED, tooLarge)
        return try {
            File(root, name).readBytes()
        } catch (error: IOException) {
            throw ManagedImportException(ManagedImportReason.HANDOFF_UNREADABLE)
        }
    }

    override fun inventory(declared: List<BundleGenerationCache.DeclaredFile>): PortableBundleReason? =
        inventoryReason(declared, sizes, sizes.keys.sorted())

    override fun source(identity: String): BundleRangeSource = DirectoryBundleRangeSource(root, identity)
}

// endregion
