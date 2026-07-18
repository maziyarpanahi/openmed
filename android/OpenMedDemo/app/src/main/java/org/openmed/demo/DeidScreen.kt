package org.openmed.demo

import android.content.Context
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import com.openmed.openmedkit.OpenMedKit
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun DeidScreen() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val modelStore = remember(context) {
        AndroidDemoModelStore(context.applicationContext)
    }
    val pipeline = remember(modelStore) {
        DemoDeidPipeline(modelStore)
    }

    var note by remember { mutableStateOf("") }
    var entities by remember { mutableStateOf(emptyList<DetectedEntity>()) }
    var status by remember { mutableStateOf("Model cache not checked") }
    var isRedacting by remember { mutableStateOf(false) }
    var hasCachedModel by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        note = context.resources
            .openRawResource(R.raw.sample_clinical_note)
            .bufferedReader()
            .use { it.readText().trimEnd() }
        hasCachedModel = modelStore.isCached()
        status = if (hasCachedModel) {
            "Model cache ready. Inference stays on this device."
        } else {
            "Model will be cached locally before the first run."
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "OpenMed PII Demo",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
        )

        DisclaimerBanner()

        ModelStatusCard(
            status = status,
            cached = hasCachedModel,
        )

        OutlinedTextField(
            value = note,
            onValueChange = {
                note = it
                entities = emptyList()
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(180.dp),
            label = { Text("Clinical note") },
            textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences),
            minLines = 6,
        )

        Button(
            onClick = {
                coroutineScope.launch {
                    val redactionText = note
                    isRedacting = true
                    entities = emptyList()
                    try {
                        val result = pipeline.redact(redactionText) { nextStatus ->
                            status = nextStatus
                        }
                        if (note == redactionText) {
                            entities = result.entities
                            hasCachedModel = result.modelCached
                        }
                    } finally {
                        isRedacting = false
                    }
                }
            },
            enabled = note.isNotBlank() && !isRedacting,
            modifier = Modifier.fillMaxWidth(),
        ) {
            if (isRedacting) {
                CircularProgressIndicator(
                    modifier = Modifier
                        .height(18.dp)
                        .width(18.dp),
                    strokeWidth = 2.dp,
                )
                Spacer(modifier = Modifier.width(10.dp))
            }
            Text(if (isRedacting) "Redacting on device..." else "Redact")
        }

        if (entities.isNotEmpty()) {
            Text(
                text = "Highlighted note",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            HighlightedText(text = note, entities = entities)

            Text(
                text = "Detected entities",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            EntityChipList(entities = entities)
        }
    }
}

@Composable
private fun DisclaimerBanner() {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.errorContainer,
        shape = MaterialTheme.shapes.medium,
    ) {
        Text(
            text = "Medical-device disclaimer: demo only. Not for clinical decisions. No text leaves this device.",
            modifier = Modifier.padding(14.dp),
            color = MaterialTheme.colorScheme.onErrorContainer,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
        )
    }
}

@Composable
private fun ModelStatusCard(status: String, cached: Boolean) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.medium,
    ) {
        Row(
            modifier = Modifier.padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Android model store",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = status,
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    text = "OpenMedKit ${OpenMedKit.VERSION}",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
            Text(
                text = if (cached) "Cached" else "Local",
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

class DemoDeidPipeline(
    private val modelStore: AndroidDemoModelStore,
) {
    suspend fun redact(
        text: String,
        onStatus: (String) -> Unit,
    ): DemoDeidResult {
        onStatus("Checking local Android model cache...")
        val model = modelStore.loadModel(onStatus)

        onStatus("Running ${model.displayName} on device...")
        delay(120)
        val entities = DemoEntityDetector.detect(text)

        onStatus("Detected ${entities.size} entities with local inference only.")
        return DemoDeidResult(
            entities = entities,
            modelCached = modelStore.isCached(),
        )
    }
}

data class DemoDeidResult(
    val entities: List<DetectedEntity>,
    val modelCached: Boolean,
)

data class DemoModelHandle(
    val displayName: String,
)

class AndroidDemoModelStore(
    private val context: Context,
) {
    private val modelDirectory: File
        get() = File(context.noBackupFilesDir, "openmed_models")

    private val markerFile: File
        get() = File(modelDirectory, "om_deid_demo_model_v1.cache")

    fun isCached(): Boolean = markerFile.isFile

    suspend fun loadModel(onStatus: (String) -> Unit): DemoModelHandle {
        if (!isCached()) {
            onStatus("Caching bundled OpenMed de-identification model...")
            withContext(Dispatchers.IO) {
                modelDirectory.mkdirs()
                markerFile.writeText(
                    "name=OpenMed Android Demo De-identification Model\n" +
                        "runtime=local\n",
                )
            }
        } else {
            onStatus("Using cached OpenMed de-identification model...")
        }

        return DemoModelHandle(displayName = "OpenMed Android Demo De-identification Model")
    }
}

object DemoEntityDetector {
    private val rules = listOf(
        EntityRule("Patient", EntityCategory.Name, 0.99f, Regex("""Patient:\s*([^\n]+)""")),
        EntityRule("DOB", EntityCategory.Date, 0.98f, Regex("""DOB:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})""")),
        EntityRule("MRN", EntityCategory.Identifier, 0.99f, Regex("""MRN:\s*([A-Z0-9-]+)""")),
        EntityRule("Phone", EntityCategory.Phone, 0.98f, Regex("""Phone:\s*(\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4})""")),
        EntityRule("Email", EntityCategory.Email, 0.99f, Regex("""Email:\s*([^\s]+)""")),
        EntityRule("Address", EntityCategory.Address, 0.97f, Regex("""Address:\s*([^\n]+)""")),
        EntityRule("Clinician", EntityCategory.Name, 0.95f, Regex("""Clinician:\s*([^\n]+)""")),
    )

    fun detect(text: String): List<DetectedEntity> {
        return rules
            .mapNotNull { rule -> rule.find(text) }
            .sortedWith(compareBy<DetectedEntity> { it.start }.thenBy { it.end })
    }
}

private data class EntityRule(
    val label: String,
    val category: EntityCategory,
    val confidence: Float,
    val regex: Regex,
) {
    fun find(text: String): DetectedEntity? {
        val match = regex.find(text) ?: return null
        val range = match.groups[1]?.range ?: return null
        val start = range.first
        val end = range.last + 1
        return DetectedEntity(
            label = label,
            text = text.substring(start, end),
            confidence = confidence,
            start = start,
            end = end,
            category = category,
        )
    }
}
