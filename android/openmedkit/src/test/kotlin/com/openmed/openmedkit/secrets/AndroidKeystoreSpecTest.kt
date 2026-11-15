package com.openmed.openmedkit.secrets

import android.security.keystore.KeyProperties
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class AndroidKeystoreSpecTest {
    private val keystore = AndroidKeystore(apiLevel = 33)

    @Test
    fun hmacGenerationSpecIsExplicitAndCarriesNoOptionalProtection() {
        val spec = keystore.keyGenParameterSpec(spec(OpenMedSecretAlgorithm.HMAC_SHA256))

        assertEquals(ALIAS, spec.keystoreAlias)
        assertEquals(KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY, spec.purposes)
        assertArrayEquals(arrayOf(KeyProperties.DIGEST_SHA256), spec.digests)
        assertFalse(spec.isUserAuthenticationRequired)
        assertFalse(spec.isStrongBoxBacked)
        assertFalse(spec.isUnlockedDeviceRequired)
    }

    @Test
    fun aesGenerationSpecUsesRandomizedGcmWithoutPadding() {
        val spec = keystore.keyGenParameterSpec(spec(OpenMedSecretAlgorithm.AES_256_GCM))

        assertEquals(KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT, spec.purposes)
        assertEquals(256, spec.keySize)
        assertArrayEquals(arrayOf(KeyProperties.BLOCK_MODE_GCM), spec.blockModes)
        assertArrayEquals(arrayOf(KeyProperties.ENCRYPTION_PADDING_NONE), spec.encryptionPaddings)
        assertTrue(spec.isRandomizedEncryptionRequired)
    }

    @Test
    fun authenticationStrongBoxAndUnlockRequestsAreApplied() {
        val spec = keystore.keyGenParameterSpec(
            spec(
                OpenMedSecretAlgorithm.HMAC_SHA256,
                strongBox = true,
                auth = OpenMedUserAuthentication(30, allowStrongBiometric = false),
                unlocked = true,
            ),
        )

        assertTrue(spec.isUserAuthenticationRequired)
        assertEquals(30, spec.userAuthenticationValidityDurationSeconds)
        assertEquals(KeyProperties.AUTH_DEVICE_CREDENTIAL, spec.userAuthenticationType)
        assertTrue(spec.isInvalidatedByBiometricEnrollment)
        assertTrue(spec.isStrongBoxBacked)
        assertTrue(spec.isUnlockedDeviceRequired)
    }

    @Test
    fun importProtectionMirrorsTheGenerationSpec() {
        val hmac = keystore.keyProtection(
            spec(
                OpenMedSecretAlgorithm.HMAC_SHA256,
                auth = OpenMedUserAuthentication(45),
                unlocked = true,
            ),
        )
        val aes = keystore.keyProtection(spec(OpenMedSecretAlgorithm.AES_256_GCM))

        assertEquals(KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY, hmac.purposes)
        assertArrayEquals(arrayOf(KeyProperties.DIGEST_SHA256), hmac.digests)
        assertTrue(hmac.isUserAuthenticationRequired)
        assertEquals(45, hmac.userAuthenticationValidityDurationSeconds)
        assertEquals(
            KeyProperties.AUTH_DEVICE_CREDENTIAL or KeyProperties.AUTH_BIOMETRIC_STRONG,
            hmac.userAuthenticationType,
        )
        assertTrue(hmac.isUnlockedDeviceRequired)
        assertEquals(KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT, aes.purposes)
        assertArrayEquals(arrayOf(KeyProperties.BLOCK_MODE_GCM), aes.blockModes)
        assertTrue(aes.isRandomizedEncryptionRequired)
        assertFalse(aes.isUserAuthenticationRequired)
    }

    @Test
    fun platformSecurityLevelAndOriginCodesMapToReportedProperties() {
        assertEquals(
            OpenMedSecurityLevel.STRONGBOX,
            securityLevelFromCode(KeyProperties.SECURITY_LEVEL_STRONGBOX),
        )
        assertEquals(
            OpenMedSecurityLevel.TRUSTED_ENVIRONMENT,
            securityLevelFromCode(KeyProperties.SECURITY_LEVEL_TRUSTED_ENVIRONMENT),
        )
        assertEquals(
            OpenMedSecurityLevel.SECURE_HARDWARE,
            securityLevelFromCode(KeyProperties.SECURITY_LEVEL_UNKNOWN_SECURE),
        )
        assertEquals(
            OpenMedSecurityLevel.SOFTWARE,
            securityLevelFromCode(KeyProperties.SECURITY_LEVEL_SOFTWARE),
        )
        assertEquals(
            OpenMedSecurityLevel.UNKNOWN,
            securityLevelFromCode(KeyProperties.SECURITY_LEVEL_UNKNOWN),
        )
        assertEquals(OpenMedSecurityLevel.SECURE_HARDWARE, securityLevelFromHardwareFlag(true))
        assertEquals(OpenMedSecurityLevel.SOFTWARE, securityLevelFromHardwareFlag(false))
        assertEquals(OpenMedSecretOrigin.GENERATED, originFromCode(KeyProperties.ORIGIN_GENERATED))
        assertEquals(OpenMedSecretOrigin.IMPORTED, originFromCode(KeyProperties.ORIGIN_IMPORTED))
        assertEquals(
            OpenMedSecretOrigin.IMPORTED,
            originFromCode(KeyProperties.ORIGIN_SECURELY_IMPORTED),
        )
        assertEquals(OpenMedSecretOrigin.UNKNOWN, originFromCode(KeyProperties.ORIGIN_UNKNOWN))
    }

    companion object {
        const val ALIAS = "openmed.deid-hash.default.v1"

        fun spec(
            algorithm: OpenMedSecretAlgorithm,
            strongBox: Boolean = false,
            auth: OpenMedUserAuthentication? = null,
            unlocked: Boolean = false,
        ) = OpenMedKeystoreKeySpec(
            alias = ALIAS,
            algorithm = algorithm,
            strongBoxBacked = strongBox,
            userAuthentication = auth,
            invalidatedByBiometricEnrollment = true,
            unlockedDeviceRequired = unlocked,
        )
    }
}

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class AndroidKeystoreLegacySpecTest {
    @Test
    fun preApi30AuthenticationUsesTheValidityDuration() {
        val spec = AndroidKeystore(apiLevel = 26).keyGenParameterSpec(
            AndroidKeystoreSpecTest.spec(
                OpenMedSecretAlgorithm.HMAC_SHA256,
                auth = OpenMedUserAuthentication(45),
            ),
        )

        assertTrue(spec.isUserAuthenticationRequired)
        assertEquals(45, spec.userAuthenticationValidityDurationSeconds)
        assertTrue(spec.isInvalidatedByBiometricEnrollment)
    }
}
