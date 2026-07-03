package com.openmed.openmedkit.policy

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Supported de-identification policy actions.
 */
enum class PolicyAction(val wireName: String) {
    KEEP("keep"),
    REDACT("redact"),
    REPLACE("replace"),
    MASK("mask"),
    REMOVE("remove"),
    HASH("hash"),
    ;

    val redactionEquivalent: PolicyAction
        get() = if (this == REDACT) MASK else this

    companion object {
        fun fromWireName(value: String): PolicyAction =
            values().firstOrNull { it.wireName == value }
                ?: throw IllegalArgumentException("unknown policy action '$value'")
    }
}

/**
 * Loaded OpenMed de-identification policy profile.
 */
data class PolicyProfile(
    val schemaVersion: Int,
    val name: String,
    val posture: String,
    val thresholdProfile: String,
    val defaultAction: PolicyAction,
    val defaultActionBias: String,
    val arbitrationMode: String,
    val strictNoLeak: Boolean,
    val safetySweepMandatory: Boolean,
    val keepMapping: Boolean,
    val reversibleId: Boolean,
    val forcedCascadeTiers: List<String>,
    val policyLabelActions: Map<String, PolicyAction>,
    val actions: Map<String, PolicyAction>,
) {
    fun actionFor(label: String): PolicyAction {
        val canonical = canonicalLabel(label)
        val action = actions[canonical]
            ?: policyLabelActions[policyLabel(canonical)]
            ?: defaultAction
        return if (strictNoLeak && action == PolicyAction.KEEP) {
            PolicyAction.MASK
        } else {
            action
        }
    }

    fun canonicalLabel(label: String): String = PolicyProfiles.canonicalLabel(label)

    private fun policyLabel(canonicalLabel: String): String = when (canonicalLabel) {
        in CLINICAL_CONCEPT_LABELS -> "CLINICAL_CONCEPT"
        in QUASI_IDENTIFIER_LABELS -> "QUASI_IDENTIFIER"
        else -> "DIRECT_IDENTIFIER"
    }
}

/**
 * Loader for the six OM-031a bundled policy profiles.
 */
object PolicyProfiles {
    const val DEFAULT_PROFILE = "hipaa_safe_harbor"

    val bundledProfileNames: List<String> = listOf(
        "hipaa_safe_harbor",
        "hipaa_expert_review_assist",
        "gdpr_pseudonymization",
        "research_limited_dataset",
        "strict_no_leak",
        "clinical_minimal_redaction",
    )

    private val aliases: Map<String, String> = mapOf(
        "gdpr" to "gdpr_pseudonymization",
    )

    fun load(name: String = DEFAULT_PROFILE): PolicyProfile {
        val canonicalName = canonicalProfileName(name)
        val resourceName = "policies/$canonicalName.json"
        val stream = javaClass.classLoader.getResourceAsStream(resourceName)
            ?: throw IllegalArgumentException("bundled policy profile '$canonicalName' was not found")
        val payload = stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        val root = Json.parseToJsonElement(payload).jsonObject
        val schemaVersion = root.requiredInt("schema_version")
        require(schemaVersion == 1) {
            "policy profile schema_version $schemaVersion is not supported"
        }

        return PolicyProfile(
            schemaVersion = schemaVersion,
            name = canonicalProfileName(root.requiredString("name")),
            posture = root.requiredString("posture"),
            thresholdProfile = root.requiredString("threshold_profile"),
            defaultAction = PolicyAction.fromWireName(root.requiredString("default_action")),
            defaultActionBias = root.requiredString("default_action_bias"),
            arbitrationMode = root.requiredString("arbitration_mode"),
            strictNoLeak = root.requiredBoolean("strict_no_leak"),
            safetySweepMandatory = root.requiredBoolean("safety_sweep_mandatory"),
            keepMapping = root.requiredBoolean("keep_mapping"),
            reversibleId = root.requiredBoolean("reversible_id"),
            forcedCascadeTiers = root.requiredStringList("forced_cascade_tiers"),
            policyLabelActions = root.requiredActionMap("policy_label_actions"),
            actions = root.requiredActionMap("actions"),
        )
    }

    fun canonicalProfileName(name: String): String {
        val normalized = name.trim().lowercase().replace('-', '_')
        require(normalized.isNotEmpty()) {
            "policy profile name must not be blank"
        }
        val canonical = aliases[normalized] ?: normalized
        require(canonical in bundledProfileNames) {
            "unknown policy profile '$name'; expected one of: ${
                (bundledProfileNames + aliases.keys).sorted().joinToString(", ")
            }"
        }
        return canonical
    }

    fun canonicalLabel(label: String): String {
        val key = label.labelKey()
        LABEL_ALIASES[key]?.let { return it }

        val upper = label
            .uppercase()
            .replace('-', '_')
            .replace(' ', '_')
            .filter { it.isLetterOrDigit() || it == '_' }
        return if (upper in CANONICAL_LABELS) upper else "OTHER"
    }
}

private fun Map<String, kotlinx.serialization.json.JsonElement>.requiredString(key: String): String =
    getValue(key).jsonPrimitive.content

private fun Map<String, kotlinx.serialization.json.JsonElement>.requiredInt(key: String): Int =
    getValue(key).jsonPrimitive.int

private fun Map<String, kotlinx.serialization.json.JsonElement>.requiredBoolean(key: String): Boolean =
    getValue(key).jsonPrimitive.boolean

private fun Map<String, kotlinx.serialization.json.JsonElement>.requiredStringList(key: String): List<String> =
    getValue(key).jsonArray.map { it.jsonPrimitive.content }

private fun Map<String, kotlinx.serialization.json.JsonElement>.requiredActionMap(
    key: String,
): Map<String, PolicyAction> =
    getValue(key).jsonObject.mapValues { (_, value) ->
        PolicyAction.fromWireName(value.jsonPrimitive.content)
    }

private fun String.labelKey(): String {
    val trimmed = trim()
    val withoutBio = if (trimmed.length > 2 && trimmed[1] == '-' && trimmed[0] in "BIES") {
        trimmed.drop(2)
    } else {
        trimmed
    }
    return withoutBio.lowercase().filter { it.isLetterOrDigit() }
}

private val CLINICAL_CONCEPT_LABELS = setOf(
    "MICROORGANISM",
    "ANTIBIOTIC",
    "SUSCEPTIBILITY",
    "CONDITION",
    "MEDICATION",
    "LAB_TEST",
    "PROCEDURE",
    "BODY_SITE",
    "DIET_TYPE",
    "NUTRITION_TARGET",
    "FEEDING_ROUTE",
    "NUTRITIONAL_STATUS",
    "OTHER",
)

private val QUASI_IDENTIFIER_LABELS = setOf(
    "LOCATION",
    "ZIPCODE",
    "ORDINAL_DIRECTION",
    "DATE",
    "TIME",
    "AGE",
    "CREDIT_CARD_ISSUER",
    "AMOUNT",
    "CURRENCY",
    "GENDER",
    "EYE_COLOR",
    "HEIGHT",
    "ORGANIZATION",
    "JOB_TITLE",
    "JOB_DEPARTMENT",
    "OCCUPATION",
)

private val CANONICAL_LABELS = setOf(
    "PERSON",
    "FIRST_NAME",
    "LAST_NAME",
    "MIDDLE_NAME",
    "PREFIX",
    "USERNAME",
    "EMAIL",
    "PHONE",
    "URL",
    "LOCATION",
    "STREET_ADDRESS",
    "BUILDING_NUMBER",
    "ZIPCODE",
    "GPS_COORDINATES",
    "ORDINAL_DIRECTION",
    "DATE",
    "DATE_OF_BIRTH",
    "TIME",
    "AGE",
    "ID_NUM",
    "SSN",
    "ACCOUNT_NUMBER",
    "PASSWORD",
    "PIN",
    "API_KEY",
    "CREDIT_CARD",
    "CREDIT_CARD_ISSUER",
    "CVV",
    "IBAN",
    "BIC",
    "AMOUNT",
    "CURRENCY",
    "BITCOIN_ADDRESS",
    "ETHEREUM_ADDRESS",
    "LITECOIN_ADDRESS",
    "MASKED_NUMBER",
    "GENDER",
    "EYE_COLOR",
    "HEIGHT",
    "ORGANIZATION",
    "JOB_TITLE",
    "JOB_DEPARTMENT",
    "OCCUPATION",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "USER_AGENT",
    "VIN",
    "VEHICLE_REGISTRATION",
    "IMEI",
    "MICROORGANISM",
    "ANTIBIOTIC",
    "SUSCEPTIBILITY",
    "CONDITION",
    "MEDICATION",
    "LAB_TEST",
    "PROCEDURE",
    "BODY_SITE",
    "DIET_TYPE",
    "NUTRITION_TARGET",
    "FEEDING_ROUTE",
    "NUTRITIONAL_STATUS",
    "OTHER",
)

private val LABEL_ALIASES = mapOf(
    "name" to "PERSON",
    "person" to "PERSON",
    "patient" to "PERSON",
    "doctor" to "PERSON",
    "fullname" to "PERSON",
    "firstname" to "FIRST_NAME",
    "givenname" to "FIRST_NAME",
    "lastname" to "LAST_NAME",
    "surname" to "LAST_NAME",
    "email" to "EMAIL",
    "emailaddress" to "EMAIL",
    "phone" to "PHONE",
    "phonenumber" to "PHONE",
    "telephone" to "PHONE",
    "address" to "STREET_ADDRESS",
    "zipcode" to "ZIPCODE",
    "zip" to "ZIPCODE",
    "date" to "DATE",
    "dateofbirth" to "DATE_OF_BIRTH",
    "dob" to "DATE_OF_BIRTH",
    "time" to "TIME",
    "age" to "AGE",
    "mrn" to "ID_NUM",
    "idnum" to "ID_NUM",
    "id" to "ID_NUM",
    "ssn" to "SSN",
    "socialsecuritynumber" to "SSN",
    "accountnumber" to "ACCOUNT_NUMBER",
    "creditcard" to "CREDIT_CARD",
    "creditcardnumber" to "CREDIT_CARD",
    "ipaddress" to "IP_ADDRESS",
    "condition" to "CONDITION",
    "diagnosis" to "CONDITION",
    "medication" to "MEDICATION",
    "drug" to "MEDICATION",
    "labtest" to "LAB_TEST",
    "procedure" to "PROCEDURE",
)
