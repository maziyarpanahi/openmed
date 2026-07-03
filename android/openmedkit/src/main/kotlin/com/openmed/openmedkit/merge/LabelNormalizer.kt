package com.openmed.openmedkit.merge

/**
 * Label family normalization and specificity rules shared by merger decisions.
 */
object LabelNormalizer {
    private val nationalIdLabels = setOf(
        "national_id",
        "nir",
        "insee",
        "steuer_id",
        "steuernummer",
        "codice_fiscale",
        "bsn",
        "dni",
        "nie",
        "aadhaar",
    )

    @JvmStatic
    fun normalizeLabel(label: String): String {
        val normalized = label.lowercase()

        if ("date" in normalized) {
            return "date"
        }
        if ("phone" in normalized || "fax" in normalized) {
            return "phone"
        }
        if ("address" in normalized) {
            return "address"
        }
        if (normalized in setOf("ssn", "social_security", "social_security_number")) {
            return "ssn"
        }
        if (normalized in nationalIdLabels) {
            return "national_id"
        }
        if (normalized in setOf("postcode", "zipcode", "zip", "postal_code")) {
            return "postcode"
        }
        if (normalized in setOf("medical_record_number", "mrn", "medical_record")) {
            return "medical_record"
        }
        if (normalized in setOf("account_number", "account")) {
            return "account"
        }
        if (normalized in setOf("encounter_number", "encounter")) {
            return "encounter"
        }
        if (normalized in setOf("document_id", "document_number", "document")) {
            return "document"
        }
        if (normalized in setOf("npi", "provider_id", "provider_identifier")) {
            return "provider_identifier"
        }
        if (normalized in setOf("insurance_id", "member_id", "policy_number", "policy")) {
            return "insurance_id"
        }
        if (
            normalized in setOf(
                "credit_debit_card",
                "credit_card",
                "debit_card",
                "payment_card",
            )
        ) {
            return "payment_card"
        }

        return normalized
    }

    @JvmStatic
    fun isMoreSpecific(label: String, than: String): Boolean {
        val lhs = label.lowercase()
        val rhs = than.lowercase()

        if (rhs != lhs && lhs.contains(rhs)) {
            return true
        }

        val specificityHierarchy = mapOf(
            "date" to setOf("date_of_birth", "date_time"),
            "name" to setOf("first_name", "last_name", "full_name"),
            "phone" to setOf("phone_number", "fax_number", "mobile_number"),
            "address" to setOf("street_address", "home_address", "billing_address"),
            "id" to setOf(
                "ssn",
                "medical_record_number",
                "account_number",
                "employee_id",
                "encounter_number",
                "document_id",
                "npi",
                "insurance_id",
            ),
            "national_id" to setOf(
                "nir",
                "insee",
                "steuer_id",
                "steuernummer",
                "codice_fiscale",
            ),
        )

        return specificityHierarchy.any { (general, specificLabels) ->
            normalizeLabel(than) == general && lhs in specificLabels
        }
    }
}
