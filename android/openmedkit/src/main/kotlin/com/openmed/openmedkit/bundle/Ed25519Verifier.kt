package com.openmed.openmedkit.bundle

import java.math.BigInteger
import java.security.MessageDigest

/**
 * Verifies one Ed25519 signature. Implementations must not perform I/O.
 *
 * Apps that already ship a platform or library Ed25519 implementation (for
 * example Tink, BouncyCastle, or the JDK 15+ `Ed25519` provider) may inject it
 * into [PortableBundle.verify]; the default is [Rfc8032Ed25519Verifier].
 */
fun interface Ed25519SignatureVerifier {
    fun verify(publicKey: ByteArray, signature: ByteArray, message: ByteArray): Boolean
}

/**
 * Verify-only Ed25519 from RFC 8032 section 5.1.7, using only
 * [java.math.BigInteger] and SHA-512.
 *
 * Android API 26 and JDK 11 ship no Ed25519 provider, so this keeps the kit
 * dependency-free. It handles only public inputs (keys, signatures, and
 * manifests), so constant-time arithmetic is not required. Non-canonical
 * encodings are rejected: `S >= L`, `y >= p`, and `x = 0` with the sign bit set.
 * It never signs and never holds private keys.
 */
object Rfc8032Ed25519Verifier : Ed25519SignatureVerifier {
    private val P: BigInteger = BigInteger.ONE.shiftLeft(255).subtract(BigInteger.valueOf(19))
    private val L: BigInteger = BigInteger.ONE.shiftLeft(252)
        .add(BigInteger("27742317777372353535851937790883648493"))
    private val D: BigInteger = BigInteger.valueOf(-121665)
        .multiply(BigInteger.valueOf(121666).modInverse(P)).mod(P)
    private val SQRT_M1: BigInteger = BigInteger.valueOf(2).modPow(P.subtract(BigInteger.ONE).shiftRight(2), P)
    private val TWO: BigInteger = BigInteger.valueOf(2)
    private val BASE: Point = run {
        val y = BigInteger.valueOf(4).multiply(BigInteger.valueOf(5).modInverse(P)).mod(P)
        val x = requireNotNull(recoverX(y, 0))
        Point(x, y, BigInteger.ONE, x.multiply(y).mod(P))
    }
    private val IDENTITY = Point(BigInteger.ZERO, BigInteger.ONE, BigInteger.ONE, BigInteger.ZERO)

    private class Point(val x: BigInteger, val y: BigInteger, val z: BigInteger, val t: BigInteger)

    override fun verify(publicKey: ByteArray, signature: ByteArray, message: ByteArray): Boolean {
        if (publicKey.size != 32 || signature.size != 64) return false
        val a = decompress(publicKey) ?: return false
        val rBytes = signature.copyOfRange(0, 32)
        val r = decompress(rBytes) ?: return false
        val s = littleEndian(signature.copyOfRange(32, 64))
        if (s >= L) return false
        val digest = MessageDigest.getInstance("SHA-512").run {
            update(rBytes)
            update(publicKey)
            update(message)
            digest()
        }
        val h = littleEndian(digest).mod(L)
        return equal(multiply(s, BASE), add(r, multiply(h, a)))
    }

    private fun littleEndian(bytes: ByteArray): BigInteger = BigInteger(1, bytes.reversedArray())

    private fun add(p: Point, q: Point): Point {
        val a = p.y.subtract(p.x).multiply(q.y.subtract(q.x)).mod(P)
        val b = p.y.add(p.x).multiply(q.y.add(q.x)).mod(P)
        val c = TWO.multiply(p.t).multiply(q.t).multiply(D).mod(P)
        val d = TWO.multiply(p.z).multiply(q.z).mod(P)
        val e = b.subtract(a)
        val f = d.subtract(c)
        val g = d.add(c)
        val h = b.add(a)
        return Point(
            e.multiply(f).mod(P),
            g.multiply(h).mod(P),
            f.multiply(g).mod(P),
            e.multiply(h).mod(P),
        )
    }

    private fun multiply(scalar: BigInteger, point: Point): Point {
        var result = IDENTITY
        var addend = point
        var remaining = scalar
        while (remaining.signum() > 0) {
            if (remaining.testBit(0)) result = add(result, addend)
            addend = add(addend, addend)
            remaining = remaining.shiftRight(1)
        }
        return result
    }

    private fun equal(p: Point, q: Point): Boolean =
        p.x.multiply(q.z).subtract(q.x.multiply(p.z)).mod(P).signum() == 0 &&
            p.y.multiply(q.z).subtract(q.y.multiply(p.z)).mod(P).signum() == 0

    private fun recoverX(y: BigInteger, sign: Int): BigInteger? {
        if (y >= P) return null
        val y2 = y.multiply(y)
        val x2 = y2.subtract(BigInteger.ONE)
            .multiply(D.multiply(y2).add(BigInteger.ONE).modInverse(P)).mod(P)
        if (x2.signum() == 0) return if (sign == 0) BigInteger.ZERO else null
        var x = x2.modPow(P.add(BigInteger.valueOf(3)).shiftRight(3), P)
        if (x.multiply(x).subtract(x2).mod(P).signum() != 0) x = x.multiply(SQRT_M1).mod(P)
        if (x.multiply(x).subtract(x2).mod(P).signum() != 0) return null
        if (x.testBit(0) != (sign == 1)) x = P.subtract(x)
        return x
    }

    private fun decompress(encoded: ByteArray): Point? {
        if (encoded.size != 32) return null
        val value = littleEndian(encoded)
        val sign = if (value.testBit(255)) 1 else 0
        val y = value.clearBit(255)
        val x = recoverX(y, sign) ?: return null
        return Point(x, y, BigInteger.ONE, x.multiply(y).mod(P))
    }
}
