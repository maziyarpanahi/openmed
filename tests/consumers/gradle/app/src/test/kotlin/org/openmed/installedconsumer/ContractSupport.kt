package org.openmed.installedconsumer

import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Loads the synthetic contract the verifier staged beside this consumer. */
internal object ConsumerContract {
    val root: JsonObject by lazy {
        val path = System.getProperty("openmed.consumer.contract")
            ?: error("contract_property_missing")
        Json.parseToJsonElement(File(path).readText()).jsonObject
    }
}

internal fun JsonObject.string(key: String): String = getValue(key).jsonPrimitive.content

internal fun JsonObject.int(key: String): Int = getValue(key).jsonPrimitive.int

internal fun JsonObject.obj(key: String): JsonObject = getValue(key).jsonObject

internal fun JsonObject.array(key: String): List<JsonObject> =
    getValue(key).jsonArray.map { it.jsonObject }
