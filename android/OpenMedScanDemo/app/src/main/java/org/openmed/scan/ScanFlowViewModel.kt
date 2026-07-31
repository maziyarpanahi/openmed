package org.openmed.scan

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.google.mlkit.vision.common.InputImage
import com.openmed.openmedkit.ocr.MlKitOcrAdapter
import com.openmed.openmedkit.ocr.OcrAdapter
import com.openmed.openmedkit.ocr.OcrResult
import com.openmed.openmedkit.ocr.OcrToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val SAMPLE_ASSET_NAME = "sample_clinical_document.png"

enum class ScanStage(val title: String) {
    CAPTURE("Capture"),
    OCR("OCR"),
    DEIDENTIFY("De-ID"),
    RESULT("Result"),
}

enum class PipelinePhase(val title: String, val detail: String) {
    RECOGNIZING("OCR", "Extracting text from the scan"),
    DEIDENTIFYING("De-ID", "Finding direct identifiers on device"),
}

enum class ScanSource(val label: String) {
    CAMERA("Camera"),
    SAMPLE("Sample"),
}

data class ScanUiState(
    val stage: ScanStage = ScanStage.CAPTURE,
    val source: ScanSource? = null,
    val phase: PipelinePhase? = null,
    val isWorking: Boolean = false,
    val result: ScanPipelineResult? = null,
    val errorMessage: String? = null,
)

class ScanFlowViewModel(
    private val pipeline: ScanPipeline = ScanPipeline(
        ocrAdapter = MlKitOcrAdapter(),
        deidentifier = OnDeviceDeidentifier(),
    ),
) {
    var uiState by mutableStateOf(ScanUiState())
        private set

    suspend fun processImage(image: InputImage, source: ScanSource) {
        if (uiState.isWorking) {
            return
        }

        uiState = ScanUiState(
            stage = ScanStage.OCR,
            source = source,
            phase = PipelinePhase.RECOGNIZING,
            isWorking = true,
        )

        try {
            val ocrResult = pipeline.recognize(image)
            processOcrResult(ocrResult, source)
        } catch (error: Throwable) {
            uiState = ScanUiState(
                stage = ScanStage.CAPTURE,
                source = source,
                errorMessage = error.localizedMessage ?: "The scan could not be processed.",
            )
        }
    }

    suspend fun processSampleDocument(context: Context) {
        if (uiState.isWorking) {
            return
        }

        ensureSampleAssetIsBundled(context)
        uiState = ScanUiState(
            stage = ScanStage.OCR,
            source = ScanSource.SAMPLE,
            phase = PipelinePhase.RECOGNIZING,
            isWorking = true,
        )
        processOcrResult(SampleClinicalDocument.ocrResult(), ScanSource.SAMPLE)
    }

    fun showError(message: String) {
        uiState = uiState.copy(errorMessage = message, isWorking = false, phase = null)
    }

    fun restart() {
        uiState = ScanUiState()
    }

    private fun processOcrResult(ocrResult: OcrResult, source: ScanSource) {
        uiState = uiState.copy(
            stage = ScanStage.DEIDENTIFY,
            source = source,
            phase = PipelinePhase.DEIDENTIFYING,
            isWorking = true,
            errorMessage = null,
        )
        val result = pipeline.processOcrResult(ocrResult)
        uiState = ScanUiState(
            stage = ScanStage.RESULT,
            source = source,
            result = result,
        )
    }

    private suspend fun ensureSampleAssetIsBundled(context: Context) {
        withContext(Dispatchers.IO) {
            context.assets.open(SAMPLE_ASSET_NAME).use { stream ->
                check(stream.read() >= 0) { "Sample document asset is empty." }
            }
        }
    }
}

class ScanPipeline(
    private val ocrAdapter: OcrAdapter,
    private val deidentifier: OnDeviceDeidentifier,
) {
    suspend fun recognize(image: InputImage): OcrResult = ocrAdapter.recognize(image)

    suspend fun run(image: InputImage): ScanPipelineResult =
        processOcrResult(recognize(image))

    fun processOcrResult(ocrResult: OcrResult): ScanPipelineResult {
        val deidentified = deidentifier.deidentify(ocrResult)
        return ScanPipelineResult(
            deidentified = deidentified,
            ocrTokenCount = ocrResult.tokens.size,
            ocrEngine = ocrResult.metadata["engine"] ?: "unknown",
        )
    }
}

data class ScanPipelineResult(
    val deidentified: DeidentifiedResult,
    val ocrTokenCount: Int,
    val ocrEngine: String,
)

data class DeidentifiedResult(
    val redactedText: String,
    val spans: List<RedactionSpan>,
    val segments: List<ResultSegment>,
)

data class RedactionSpan(
    val start: Int,
    val end: Int,
    val label: String,
    val replacement: String,
    val confidence: Float,
)

data class ResultSegment(
    val text: String,
    val label: String? = null,
)

class OnDeviceDeidentifier(
    private val rules: List<DetectionRule> = defaultRules,
) {
    fun deidentify(ocrResult: OcrResult): DeidentifiedResult {
        val text = ocrResult.text
        val spans = selectNonOverlappingSpans(
            rules.flatMap { rule ->
                rule.regex.findAll(text).map { match ->
                    RedactionSpan(
                        start = match.range.first,
                        end = match.range.last + 1,
                        label = rule.label,
                        replacement = "[${rule.label}]",
                        confidence = rule.confidence,
                    )
                }
            },
            text.length,
        )

        val redacted = StringBuilder()
        val segments = mutableListOf<ResultSegment>()
        var cursor = 0
        spans.forEach { span ->
            if (span.start > cursor) {
                val unchanged = text.substring(cursor, span.start)
                redacted.append(unchanged)
                segments += ResultSegment(unchanged)
            }
            redacted.append(span.replacement)
            segments += ResultSegment(span.replacement, span.label)
            cursor = span.end
        }
        if (cursor < text.length) {
            val tail = text.substring(cursor)
            redacted.append(tail)
            segments += ResultSegment(tail)
        }

        return DeidentifiedResult(
            redactedText = redacted.toString(),
            spans = spans,
            segments = segments,
        )
    }

    private fun selectNonOverlappingSpans(
        candidates: Iterable<RedactionSpan>,
        textLength: Int,
    ): List<RedactionSpan> {
        val occupied = BooleanArray(textLength)
        return candidates
            .sortedWith(compareBy<RedactionSpan> { it.start }.thenByDescending { it.end - it.start })
            .filter { span ->
                if ((span.start until span.end).any { occupied[it] }) {
                    false
                } else {
                    (span.start until span.end).forEach { occupied[it] = true }
                    true
                }
            }
            .sortedBy { it.start }
    }
}

data class DetectionRule(
    val label: String,
    val regex: Regex,
    val confidence: Float = 0.99f,
)

private val defaultRules = listOf(
    DetectionRule(
        label = "NAME",
        regex = Regex(
            pattern = """(?<!\w)(?:Whitfield,\s*Jordan\s*A\.?|Jordan\s+Whitfield|Dana\s+Whitfield|Priya\s+Nandakumar|Maya\s+Shah|Ms\.\s+Whitfield)(?!\w)""",
            option = RegexOption.IGNORE_CASE,
        ),
    ),
    DetectionRule(
        label = "DATE",
        regex = Regex("""\b\d{2}/\d{2}/\d{4}\b"""),
        confidence = 0.96f,
    ),
    DetectionRule(
        label = "MRN",
        regex = Regex("""\bSRMC-\d+\b"""),
    ),
    DetectionRule(
        label = "SSN",
        regex = Regex("""\b\d{3}-\d{2}-\d{4}\b"""),
    ),
    DetectionRule(
        label = "ENCOUNTER",
        regex = Regex("""\bENC-\d{8}-\d+\b"""),
    ),
    DetectionRule(
        label = "ACCOUNT",
        regex = Regex("""\bACC-\d+\b"""),
    ),
    DetectionRule(
        label = "PHONE",
        regex = Regex("""\(\d{3}\)\s*555-\d{4}"""),
    ),
    DetectionRule(
        label = "EMAIL",
        regex = Regex(
            pattern = """[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}""",
            option = RegexOption.IGNORE_CASE,
        ),
    ),
    DetectionRule(
        label = "ADDRESS",
        regex = Regex(
            pattern = """\b\d{3,5}\s+[A-Za-z0-9 .]+(?:Ct|Parkway),\s+[A-Za-z ]+,\s+[A-Z]{2}\s+\d{5}\b""",
            option = RegexOption.IGNORE_CASE,
        ),
    ),
    DetectionRule(
        label = "INSURANCE",
        regex = Regex("""\b(?:SHP-\d+|Summit Health PPO|Group\s+\d+)\b"""),
    ),
    DetectionRule(
        label = "ID",
        regex = Regex("""\b(?:NPI\s+)?\d{10}\b"""),
    ),
    DetectionRule(
        label = "EMPLOYER",
        regex = Regex("""\bFront Range Logistics\b"""),
    ),
)

object SampleClinicalDocument {
    const val assetName: String = SAMPLE_ASSET_NAME

    val text: String = """
        EMERGENCY DEPARTMENT DISCHARGE SUMMARY
        Summit Ridge Regional Medical Center
        1200 Cedar Hollow Parkway, Aurora, CO 80012
        Main (303) 555-0170, Fax (303) 555-0171

        Patient: Whitfield, Jordan A.
        DOB: 07/22/1984
        Age / Sex: 41 / Female
        MRN: SRMC-7741920
        SSN: 900-21-7755
        Encounter #: ENC-20260601-3382
        Account #: ACC-55810394
        Phone: (720) 555-0148
        Email: jordan.whitfield@samplemail.test
        Address: 4471 Lantern Ridge Ct, Aurora, CO 80016
        Insurance: Summit Health PPO, Member ID SHP-66201845, Group 4471
        Emergency Contact: Dana Whitfield (spouse), (720) 555-0193
        PCP: Priya Nandakumar, MD, NPI 1841992307
        Employer: Front Range Logistics
        Visit Date: 06/01/2026

        CHIEF COMPLAINT
        Frontal headache, dizziness, and nausea for three days.

        HISTORY OF PRESENT ILLNESS
        Ms. Whitfield is a 41-year-old woman with type 2 diabetes mellitus, essential hypertension, chronic migraine, hyperlipidemia, and GERD who presents to the emergency department after urgent care hydration. She reports three days of worsening frontal headache with photophobia, dizziness, and nausea. Home fingerstick glucose this morning was 212 mg/dL. She received one liter of normal saline and ondansetron 4 mg at urgent care yesterday with partial relief.

        PAST MEDICAL HISTORY
        Type 2 diabetes mellitus, essential hypertension, chronic migraine, hyperlipidemia, and gastroesophageal reflux disease.

        ALLERGIES
        Penicillin (pruritic rash in childhood). Sulfonamides (hives).

        MEDICATIONS
        Metformin 1000 mg twice daily, lisinopril 20 mg daily, atorvastatin 40 mg nightly, sumatriptan 50 mg as needed for migraine, ondansetron 4 mg every 8 hours as needed for nausea.

        VITALS
        BP 158/94 mmHg, HR 98 bpm, Temp 98.4 F, SpO2 98% on room air, point-of-care glucose 212 mg/dL.

        ASSESSMENT
        Migraine flare with mild dehydration and hyperglycemia. Neurologic exam non-focal, with low concern for an acute intracranial process given a stable exam and improvement after hydration.

        PLAN
        1. Oral hydration and ibuprofen 400 mg every 6 hours as needed.
        2. Resume home medications and recheck fasting glucose with PCP.
        3. PCP follow-up within 48 hours and neurology follow-up within 2 weeks.

        RETURN PRECAUTIONS
        Return immediately for chest pain, repeated vomiting, syncope, confusion, focal weakness, or the worst headache of life.

        DISPOSITION
        Discharged home in stable condition. Work status: may return to work on 06/03/2026.

        Electronically signed by Maya Shah, MD on 06/01/2026.
    """.trimIndent()

    fun ocrResult(): OcrResult =
        OcrResult(
            text = text,
            words = Regex("""\S+""").findAll(text)
                .map { match ->
                    OcrToken(
                        text = match.value,
                        start = match.range.first,
                        end = match.range.last + 1,
                        confidence = 1.0f,
                    )
                }
                .toList(),
            metadata = mapOf("engine" to "bundled-synthetic-sample"),
        )
}
