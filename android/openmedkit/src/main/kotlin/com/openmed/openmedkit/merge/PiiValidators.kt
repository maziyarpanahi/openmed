package com.openmed.openmedkit.merge

/**
 * Deterministic validators used by semantic PII patterns.
 */
object PiiValidators {
    @JvmStatic
    fun validateSSN(text: String): Boolean {
        val digits = digitsOnly(text)
        if (digits.length != 9) {
            return false
        }

        val area = digits.substring(0, 3)
        val group = digits.substring(3, 5)
        val serial = digits.substring(5, 9)

        if (area == "000" || area == "666" || area.startsWith("9")) {
            return false
        }
        if (group == "00") {
            return false
        }
        if (serial == "0000") {
            return false
        }

        return true
    }

    @JvmStatic
    fun validateLuhn(text: String): Boolean {
        val digits = digitsOnly(text)
        if (digits.length < 13) {
            return false
        }
        return luhnChecksum(digits) == 0
    }

    @JvmStatic
    fun validateNPI(text: String): Boolean {
        val digits = digitsOnly(text)
        if (digits.length != 10) {
            return false
        }
        return luhnChecksum("80840$digits") == 0
    }

    @JvmStatic
    fun validatePhoneUS(text: String): Boolean {
        val digits = digitsOnly(text)

        if (digits.length == 10) {
            val firstArea = digits[0]
            val firstExchange = digits[3]

            if (firstArea == '0' || firstArea == '1') {
                return false
            }
            if (firstExchange == '0') {
                return false
            }

            return true
        }

        if (digits.length == 11 && digits.first() == '1') {
            return validatePhoneUS(digits.drop(1))
        }

        return false
    }

    private fun luhnChecksum(digits: String): Int {
        var checksum = 0

        digits.reversed().forEachIndexed { index, char ->
            val value = char.digitToIntOrNull() ?: return@forEachIndexed
            checksum += if (index % 2 == 0) {
                value
            } else {
                val doubled = value * 2
                if (doubled > 9) doubled - 9 else doubled
            }
        }

        return checksum % 10
    }

    private fun digitsOnly(text: String): String = text.filter { it.isDigit() }
}
