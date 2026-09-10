package org.openmed.maple

import android.content.Context
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.io.Closeable
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class MapleDemoUiState(
    val selectedTask: MapleTask = MapleTask.REDACT,
    val clinicalText: String = SyntheticClinicalNote.text,
    val question: String = "What facts support the documented assessment, and what remains uncertain?",
    val bundle: InstalledMapleBundle? = null,
    val isLoadingBundle: Boolean = true,
    val isImporting: Boolean = false,
    val importProgress: Float = 0f,
    val importStatus: String? = null,
    val isGenerating: Boolean = false,
    val generationStatus: String? = null,
    val partialOutput: String = "",
    val presentation: MaplePresentation? = null,
    val isSyntheticPreview: Boolean = false,
    val generatedTokens: Int = 0,
    val tokensPerSecond: Double = 0.0,
    val errorMessage: String? = null,
)

class MapleDemoViewModel(context: Context) : Closeable {
    private val applicationContext = context.applicationContext
    private val installer = MapleBundleInstaller(applicationContext)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var engine: MapleOnnxEngine? = null
    private var generationJob: Job? = null

    var uiState by mutableStateOf(MapleDemoUiState())
        private set

    init {
        scope.launch {
            val active = withContext(Dispatchers.IO) { installer.activeBundle() }
            uiState = uiState.copy(bundle = active, isLoadingBundle = false)
        }
    }

    fun selectTask(task: MapleTask) {
        if (uiState.isGenerating) return
        uiState = uiState.copy(
            selectedTask = task,
            partialOutput = "",
            presentation = null,
            isSyntheticPreview = false,
            errorMessage = null,
        )
    }

    fun updateClinicalText(value: String) {
        uiState = uiState.copy(clinicalText = value, presentation = null, errorMessage = null)
    }

    fun updateQuestion(value: String) {
        uiState = uiState.copy(question = value, presentation = null, errorMessage = null)
    }

    fun loadSyntheticNote() {
        uiState = uiState.copy(
            clinicalText = SyntheticClinicalNote.text,
            presentation = null,
            errorMessage = null,
        )
    }

    fun importBundle(uri: Uri) {
        if (uiState.isImporting || uiState.isGenerating) return
        scope.launch {
            uiState = uiState.copy(
                isImporting = true,
                importProgress = 0f,
                importStatus = "Reading export metadata",
                errorMessage = null,
            )
            try {
                val installed = withContext(Dispatchers.IO) {
                    val source = applicationContext.contentResolver.openInputStream(uri)
                        ?: throw MapleBundleException("The selected bundle could not be opened")
                    source.use { input ->
                        installer.install(input) { progress ->
                            scope.launch {
                                uiState = uiState.copy(
                                    importProgress = progress.fraction,
                                    importStatus = "Verifying ${progress.fileName.substringAfterLast('/')}",
                                )
                            }
                        }
                    }
                }
                withContext(Dispatchers.IO) {
                    engine?.close()
                    engine = null
                }
                uiState = uiState.copy(
                    bundle = installed,
                    isImporting = false,
                    importProgress = 1f,
                    importStatus = "Integrity verified; ready to load offline",
                )
            } catch (error: CancellationException) {
                throw error
            } catch (error: Throwable) {
                uiState = uiState.copy(
                    isImporting = false,
                    importStatus = null,
                    errorMessage = safeMessage(error, "The Maple bundle could not be installed"),
                )
            }
        }
    }

    fun runSelectedTask() {
        if (uiState.isGenerating || uiState.clinicalText.isBlank()) return
        if (uiState.bundle == null) {
            if (uiState.clinicalText.trim() != SyntheticClinicalNote.text.trim()) {
                uiState = uiState.copy(
                    errorMessage = "Synthetic preview only supports the bundled demo note. " +
                        "Import a verified Maple bundle to process other text.",
                )
                return
            }
            uiState = uiState.copy(
                presentation = SyntheticPreviewResults.forTask(uiState.selectedTask),
                isSyntheticPreview = true,
                partialOutput = "",
                generatedTokens = 0,
                tokensPerSecond = 0.0,
                errorMessage = null,
            )
            return
        }

        generationJob = scope.launch {
            val task = uiState.selectedTask
            val sourceText = uiState.clinicalText
            val prompt = runCatching {
                MaplePromptFactory.build(task, sourceText, uiState.question)
            }.getOrElse { error ->
                uiState = uiState.copy(errorMessage = safeMessage(error, "Input is not ready"))
                return@launch
            }
            uiState = uiState.copy(
                isGenerating = true,
                generationStatus = if (engine == null) {
                    "Loading the verified model into ONNX Runtime"
                } else {
                    "Maple is reasoning on device"
                },
                partialOutput = "",
                presentation = null,
                isSyntheticPreview = false,
                generatedTokens = 0,
                tokensPerSecond = 0.0,
                errorMessage = null,
            )
            try {
                val activeEngine = engine ?: MapleOnnxEngine.open(uiState.bundle!!).also {
                    engine = it
                }
                uiState = uiState.copy(generationStatus = "Maple is reasoning on device")
                val result = activeEngine.generate(
                    MapleGenerationRequest(
                        prompt = prompt,
                        maxNewTokens = task.maxNewTokens,
                        temperature = task.temperature,
                    ),
                ) { partial, count ->
                    if (count == 1 || count % PARTIAL_UPDATE_INTERVAL == 0) {
                        withContext(Dispatchers.Main.immediate) {
                            uiState = uiState.copy(
                                partialOutput = MapleOutputParser.visibleText(partial),
                                generatedTokens = count,
                            )
                        }
                    }
                }
                uiState = uiState.copy(
                    isGenerating = false,
                    generationStatus = null,
                    partialOutput = "",
                    presentation = MapleOutputParser.parse(task, result.text, sourceText),
                    generatedTokens = result.generatedTokens,
                    tokensPerSecond = result.tokensPerSecond,
                )
            } catch (_: CancellationException) {
                uiState = uiState.copy(
                    isGenerating = false,
                    generationStatus = null,
                    partialOutput = "",
                )
            } catch (error: Throwable) {
                uiState = uiState.copy(
                    isGenerating = false,
                    generationStatus = null,
                    partialOutput = "",
                    presentation = null,
                    errorMessage = safeMessage(error, "On-device Maple inference failed"),
                )
            }
        }
    }

    fun cancelGeneration() {
        generationJob?.cancel()
    }

    fun dismissError() {
        uiState = uiState.copy(errorMessage = null)
    }

    override fun close() {
        generationJob?.cancel()
        val currentEngine = engine
        engine = null
        scope.cancel()
        if (currentEngine != null) {
            CoroutineScope(Dispatchers.IO).launch { currentEngine.close() }
        }
    }

    private fun safeMessage(error: Throwable, fallback: String): String {
        val message = error.message?.takeIf { it.isNotBlank() } ?: fallback
        return message.take(MAX_ERROR_LENGTH)
    }

    private companion object {
        const val PARTIAL_UPDATE_INTERVAL = 4
        const val MAX_ERROR_LENGTH = 240
    }
}
