package org.openmed.installedconsumer

import com.openmed.openmedkit.DeidentifyEngine
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.TokenClassificationPrediction
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.policy.PolicyProfiles
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonPrimitive
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

/**
 * Android-runtime checks: the facade's grapheme snapping uses android.icu and
 * the catalog loads through AssetManager from the installed AAR's assets.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class InstalledFacadeTest {
    private val contract = ConsumerContract.root

    @Test
    fun facadeRedactsSyntheticSessionUnderEachPolicy() = runBlocking {
        for (redactionCase in contract.array("redaction_cases")) {
            val id = redactionCase.string("id")
            val text = redactionCase.string("text")
            val profile = redactionCase.string("profile")
            val predictions = redactionCase.array("entities").map { entity ->
                val start = entity.int("start")
                val end = entity.int("end")
                TokenClassificationPrediction(
                    label = entity.string("label"),
                    text = UnicodeOffsetContract.substring(text, start, end),
                    confidence = entity.getValue("confidence").jsonPrimitive.double.toFloat(),
                    start = start,
                    end = end,
                )
            }
            val kit = OpenMedKit(classifier = FixtureClassifier(predictions))
            val result = if (profile == PolicyProfiles.DEFAULT_PROFILE) {
                MigrationExamples.redact(kit, text)
            } else {
                kit.deidentify(text, policy = profile)
            }
            assertEquals(profile, result.policyName, "$id:policy")
            assertEquals(redactionCase.string("expected_redacted"), result.redactedText, "$id:redaction")
            assertEquals(predictions.size, result.actions.size, "$id:action_count")

            val engineResult = DeidentifyEngine().deidentify(
                text,
                predictions.map { EntityPrediction(it.label, it.text, it.confidence, it.start, it.end) },
                PolicyProfiles.load(profile),
            )
            assertEquals(result.redactedText, engineResult.redactedText, "$id:engine_parity")
        }
    }

    @Test
    fun packagedModelCatalogLoadsThroughAssetManager() {
        val catalog = ModelCatalog.load(RuntimeEnvironment.getApplication())
        assertTrue(catalog.entries.isNotEmpty(), "catalog_empty")
        assertTrue(
            catalog.entries.all { it.reproducibilityHash.startsWith("sha256:") },
            "catalog_hash",
        )
        assertTrue(catalog.filter(format = "onnx").isNotEmpty(), "catalog_onnx")
    }

    private class FixtureClassifier(
        private val predictions: List<TokenClassificationPrediction>,
    ) : OnnxTokenClassifier {
        override suspend fun predict(text: String): List<TokenClassificationPrediction> =
            predictions
    }
}
