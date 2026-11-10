package com.openmed.openmedkit.merge

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PiiValidatorsTest {
    @Test
    fun validatesSsnRules() {
        assertTrue(PiiValidators.validateSSN("123-45-6789"))
        assertTrue(PiiValidators.validateSSN("123 45 6789"))

        assertFalse(PiiValidators.validateSSN("000-45-6789"))
        assertFalse(PiiValidators.validateSSN("666-45-6789"))
        assertFalse(PiiValidators.validateSSN("900-45-6789"))
        assertFalse(PiiValidators.validateSSN("123-00-6789"))
        assertFalse(PiiValidators.validateSSN("123-45-0000"))
        assertFalse(PiiValidators.validateSSN("123-45-678"))
    }

    @Test
    fun validatesLuhnNumbers() {
        assertTrue(PiiValidators.validateLuhn("4111 1111 1111 1111"))
        assertTrue(PiiValidators.validateLuhn("4012-8888-8888-1881"))

        assertFalse(PiiValidators.validateLuhn("4111 1111 1111 1112"))
        assertFalse(PiiValidators.validateLuhn("123456789012"))
    }

    @Test
    fun validatesNpiWithPrefixChecksum() {
        assertTrue(PiiValidators.validateNPI("1234567893"))

        assertFalse(PiiValidators.validateNPI("1234567890"))
        assertFalse(PiiValidators.validateNPI("123456789"))
    }

    @Test
    fun validatesUsPhoneNumbers() {
        assertTrue(PiiValidators.validatePhoneUS("(415) 555-1234"))
        assertTrue(PiiValidators.validatePhoneUS("1-415-555-1234"))

        assertFalse(PiiValidators.validatePhoneUS("015-555-1234"))
        assertFalse(PiiValidators.validatePhoneUS("415-055-1234"))
        assertFalse(PiiValidators.validatePhoneUS("555-1234"))
    }
}
