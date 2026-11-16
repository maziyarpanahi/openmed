package com.openmed.openmedkit.deid

import java.math.BigInteger
import java.security.MessageDigest
import java.time.DateTimeException
import java.time.LocalDate
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * Patient-keyed and seeded date-shift offsets that match Python
 * `openmed.core.date_shift` exactly, plus ISO date shifting.
 *
 * An offset is a non-zero whole number of days in `[-maxDays, maxDays]`. The
 * digest (HMAC-SHA256 of the patient key under the caller's secret, or SHA-256
 * of the domain-separated seed) is read as an unsigned big-endian integer,
 * reduced modulo `2 * maxDays`, and mapped onto that range without zero. The
 * patient key is used only as the HMAC message; nothing is stored or logged,
 * and errors never contain keys, secrets, seeds or dates.
 */
object DateShift {
    /** Default bound, as Python `DEFAULT_DATE_SHIFT_MAX_DAYS`. */
    const val DEFAULT_MAX_DAYS: Int = 365

    private val SEED_DOMAIN = "openmed-date-shift-seed-v1\u0000".toByteArray(Charsets.US_ASCII)
    private val ISO_DATE = Regex("[0-9]{4}-[0-9]{2}-[0-9]{2}")

    /**
     * Stable offset for [patientKey] (UTF-8) under the HMAC [secret].
     *
     * @throws IllegalArgumentException when either value is empty or [maxDays]
     *   is not positive.
     */
    fun stableOffset(patientKey: String, secret: ByteArray, maxDays: Int = DEFAULT_MAX_DAYS): Int {
        requireMaxDays(maxDays)
        return offset(patientKeyDigest(patientKey, secret), maxDays)
    }

    /** Stable offset for a request [seed]; see [stableOffsetFromSeed] for any integer. */
    fun stableOffsetFromSeed(seed: Long, maxDays: Int = DEFAULT_MAX_DAYS): Int =
        stableOffsetFromSeed(BigInteger.valueOf(seed), maxDays)

    /**
     * Stable offset for a request [seed] of any size, as Python hashes the
     * seed's decimal text.
     *
     * @throws IllegalArgumentException when [maxDays] is not positive.
     */
    fun stableOffsetFromSeed(seed: BigInteger, maxDays: Int = DEFAULT_MAX_DAYS): Int {
        requireMaxDays(maxDays)
        return offset(seedDigest(seed), maxDays)
    }

    /**
     * Shifts an ISO `YYYY-MM-DD` date by [days] in the proleptic Gregorian calendar.
     *
     * @throws IllegalArgumentException when [isoDate] is not a valid date or the
     *   result leaves years 1 to 9999, the range Python dates support.
     */
    fun shift(isoDate: String, days: Int): String {
        val date = try {
            if (!ISO_DATE.matches(isoDate)) null else LocalDate.parse(isoDate)
        } catch (error: DateTimeException) {
            null
        } ?: throw IllegalArgumentException("isoDate must be a YYYY-MM-DD calendar date")
        val shifted = date.plusDays(days.toLong())
        require(date.year >= 1 && shifted.year in 1..9999) { "shifted date must stay within years 1 to 9999" }
        return shifted.toString()
    }

    internal fun patientKeyDigest(patientKey: String, secret: ByteArray): ByteArray {
        val message = patientKey.toByteArray(Charsets.UTF_8)
        require(message.isNotEmpty()) { "patientKey must be non-empty" }
        require(secret.isNotEmpty()) { "secret must be non-empty" }
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret, "HmacSHA256"))
        return mac.doFinal(message)
    }

    internal fun seedDigest(seed: BigInteger): ByteArray =
        MessageDigest.getInstance("SHA-256").run {
            update(SEED_DOMAIN)
            digest(seed.toString().toByteArray(Charsets.US_ASCII))
        }

    private fun offset(digest: ByteArray, maxDays: Int): Int {
        val bucket = BigInteger(1, digest).mod(BigInteger.valueOf(2L * maxDays)).toInt()
        return if (bucket < maxDays) bucket - maxDays else bucket - maxDays + 1
    }

    private fun requireMaxDays(maxDays: Int) {
        require(maxDays > 0) { "maxDays must be positive" }
    }
}
