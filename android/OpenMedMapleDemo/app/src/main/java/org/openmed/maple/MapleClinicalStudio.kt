@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package org.openmed.maple

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.util.Locale

private val Ink = Color(0xFF071A1C)
private val Panel = Color(0xFF10272A)
private val PanelRaised = Color(0xFF173236)
private val Mint = Color(0xFF47E3C2)
private val PaleMint = Color(0xFFC7FFF3)
private val Sky = Color(0xFF6DB7FF)
private val Warm = Color(0xFFFFC76D)
private val Muted = Color(0xFFA7C1C3)
private val Danger = Color(0xFFFF8D8D)

@Composable
fun MapleClinicalStudioApp() {
    val context = LocalContext.current
    val viewModel = remember { MapleDemoViewModel(context) }
    DisposableEffect(viewModel) {
        onDispose(viewModel::close)
    }
    MapleStudioTheme {
        MapleStudioScreen(viewModel)
    }
}

@Composable
private fun MapleStudioTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Mint,
            onPrimary = Ink,
            secondary = Sky,
            tertiary = Warm,
            background = Ink,
            surface = Panel,
            onSurface = Color(0xFFE9FFFB),
            onBackground = Color(0xFFE9FFFB),
            error = Danger,
        ),
        content = content,
    )
}

@Composable
private fun MapleStudioScreen(viewModel: MapleDemoViewModel) {
    val state = viewModel.uiState
    val bundlePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        uri?.let(viewModel::importBundle)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.linearGradient(
                    listOf(Ink, Color(0xFF0A2425), Color(0xFF071A1C)),
                ),
            ),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp, vertical = 22.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            StudioHeader()
            PrivacyRibbon()
            ModelCard(
                state = state,
                onImport = {
                    bundlePicker.launch(
                        arrayOf("application/zip", "application/octet-stream"),
                    )
                },
            )
            TaskRail(
                selected = state.selectedTask,
                enabled = !state.isGenerating && !state.isImporting,
                onSelect = viewModel::selectTask,
            )
            Workspace(
                state = state,
                onClinicalTextChange = viewModel::updateClinicalText,
                onQuestionChange = viewModel::updateQuestion,
                onLoadSample = viewModel::loadSyntheticNote,
                onRun = viewModel::runSelectedTask,
                onCancel = viewModel::cancelGeneration,
            )
            MedicalDisclaimer()
            Spacer(Modifier.height(8.dp))
        }
        state.errorMessage?.let { message ->
            ErrorToast(
                message = message,
                onDismiss = viewModel::dismissError,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(18.dp),
            )
        }
    }
}

@Composable
private fun StudioHeader() {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Top,
    ) {
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(5.dp),
        ) {
            Text(
                text = "OPENMEDKIT / MAPLE",
                color = Mint,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.6.sp,
            )
            Text(
                text = "Clinical Studio",
                color = PaleMint,
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Private document intelligence, right here on your device.",
                color = Muted,
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        Spacer(Modifier.size(12.dp))
        Surface(
            modifier = Modifier.size(48.dp),
            color = Mint.copy(alpha = 0.13f),
            shape = CircleShape,
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(
                    text = "M",
                    color = Mint,
                    fontWeight = FontWeight.Black,
                    fontSize = 22.sp,
                )
            }
        }
    }
}

@Composable
private fun PrivacyRibbon() {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = Mint.copy(alpha = 0.09f),
        shape = RoundedCornerShape(14.dp),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 11.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            StatusDot(Mint)
            Text(
                text = "DEVICE-ONLY",
                color = Mint,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "No internet permission • no telemetry • no clinical text in logs",
                color = PaleMint.copy(alpha = 0.82f),
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun ModelCard(state: MapleDemoUiState, onImport: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Panel),
        shape = RoundedCornerShape(20.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(
                    modifier = Modifier.weight(1f),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(11.dp),
                ) {
                    StatusDot(
                        when {
                            state.isLoadingBundle || state.isImporting -> Warm
                            state.bundle != null -> Mint
                            else -> Muted
                        },
                    )
                    Column {
                        Text(
                            text = when {
                                state.isLoadingBundle -> "Checking local model"
                                state.bundle != null -> "Maple bundle verified"
                                else -> "Synthetic preview mode"
                            },
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            text = state.bundle?.let {
                                "${it.manifest.quantization} • ${it.manifest.sourceRevision.take(10)} • verified"
                            } ?: "Import a user-exported .zip bundle to run real inference",
                            color = Muted,
                            style = MaterialTheme.typography.bodySmall,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
                OutlinedButton(
                    enabled = !state.isLoadingBundle && !state.isImporting && !state.isGenerating,
                    onClick = onImport,
                ) {
                    Text(if (state.bundle == null) "Import model" else "Replace")
                }
            }
            if (state.isImporting) {
                LinearProgressIndicator(
                    progress = state.importProgress,
                    modifier = Modifier.fillMaxWidth(),
                    color = Mint,
                    trackColor = PanelRaised,
                )
            }
            state.importStatus?.let {
                Text(text = it, color = Muted, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun TaskRail(
    selected: MapleTask,
    enabled: Boolean,
    onSelect: (MapleTask) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(9.dp),
    ) {
        MapleTask.values().forEach { task ->
            FilterChip(
                selected = task == selected,
                enabled = enabled,
                onClick = { onSelect(task) },
                label = {
                    Column(modifier = Modifier.padding(vertical = 3.dp)) {
                        Text(task.eyebrow.uppercase(Locale.ROOT), fontSize = 10.sp)
                        Text(task.title, fontWeight = FontWeight.SemiBold)
                    }
                },
                colors = FilterChipDefaults.filterChipColors(
                    containerColor = Panel,
                    labelColor = Muted,
                    selectedContainerColor = Mint,
                    selectedLabelColor = Ink,
                ),
                shape = RoundedCornerShape(14.dp),
            )
        }
    }
}

@Composable
private fun Workspace(
    state: MapleDemoUiState,
    onClinicalTextChange: (String) -> Unit,
    onQuestionChange: (String) -> Unit,
    onLoadSample: () -> Unit,
    onRun: () -> Unit,
    onCancel: () -> Unit,
) {
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val wide = maxWidth >= 760.dp
        if (wide) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.Top,
            ) {
                InputPanel(
                    state,
                    onClinicalTextChange,
                    onQuestionChange,
                    onLoadSample,
                    onRun,
                    onCancel,
                    Modifier.weight(1f),
                )
                OutputPanel(state, Modifier.weight(1f))
            }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                InputPanel(
                    state,
                    onClinicalTextChange,
                    onQuestionChange,
                    onLoadSample,
                    onRun,
                    onCancel,
                    Modifier.fillMaxWidth(),
                )
                OutputPanel(state, Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun InputPanel(
    state: MapleDemoUiState,
    onClinicalTextChange: (String) -> Unit,
    onQuestionChange: (String) -> Unit,
    onLoadSample: () -> Unit,
    onRun: () -> Unit,
    onCancel: () -> Unit,
    modifier: Modifier,
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = Panel),
        shape = RoundedCornerShape(20.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(13.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text("Clinical context", fontWeight = FontWeight.SemiBold)
                    Text("Text never leaves this app", color = Muted, fontSize = 12.sp)
                }
                TextButton(enabled = !state.isGenerating, onClick = onLoadSample) {
                    Text("Load synthetic note")
                }
            }
            OutlinedTextField(
                value = state.clinicalText,
                enabled = !state.isGenerating,
                onValueChange = onClinicalTextChange,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(245.dp),
                label = { Text("Clinical note") },
                textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
                shape = RoundedCornerShape(14.dp),
            )
            if (state.selectedTask == MapleTask.CHAT) {
                OutlinedTextField(
                    value = state.question,
                    enabled = !state.isGenerating,
                    onValueChange = onQuestionChange,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Question for Maple") },
                    minLines = 2,
                    shape = RoundedCornerShape(14.dp),
                )
            }
            if (state.isGenerating) {
                OutlinedButton(onClick = onCancel, modifier = Modifier.fillMaxWidth()) {
                    Text("Cancel generation")
                }
            } else {
                Button(
                    enabled = state.clinicalText.isNotBlank() && !state.isImporting,
                    onClick = onRun,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = Mint, contentColor = Ink),
                    shape = RoundedCornerShape(14.dp),
                ) {
                    Text(
                        if (state.bundle == null) {
                            "Preview ${state.selectedTask.title}"
                        } else {
                            state.selectedTask.action
                        },
                        modifier = Modifier.padding(vertical = 5.dp),
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }
    }
}

@Composable
private fun OutputPanel(state: MapleDemoUiState, modifier: Modifier) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = Panel),
        shape = RoundedCornerShape(20.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(13.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text("Maple output", fontWeight = FontWeight.SemiBold)
                    Text(
                        when {
                            state.isGenerating -> state.generationStatus ?: "Running locally"
                            state.isSyntheticPreview -> "Synthetic UI preview • no inference"
                            state.presentation != null -> "Review required before any use"
                            else -> "Results appear here"
                        },
                        color = if (state.isSyntheticPreview) Warm else Muted,
                        fontSize = 12.sp,
                    )
                }
                if (state.isGenerating) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        color = Mint,
                        strokeWidth = 2.dp,
                    )
                } else if (state.generatedTokens > 0) {
                    Surface(color = Mint.copy(alpha = 0.12f), shape = RoundedCornerShape(10.dp)) {
                        Text(
                            text = String.format(
                                Locale.ROOT,
                                "%d tok • %.1f tok/s",
                                state.generatedTokens,
                                state.tokensPerSecond,
                            ),
                            modifier = Modifier.padding(horizontal = 9.dp, vertical = 6.dp),
                            color = Mint,
                            fontSize = 11.sp,
                        )
                    }
                }
            }
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(245.dp),
                color = PanelRaised,
                shape = RoundedCornerShape(14.dp),
            ) {
                val presentation = state.presentation
                when {
                    presentation != null -> ResultContent(presentation)
                    state.partialOutput.isNotBlank() -> SelectionContainer {
                        Text(
                            text = state.partialOutput,
                            modifier = Modifier
                                .verticalScroll(rememberScrollState())
                                .padding(14.dp),
                            color = PaleMint,
                        )
                    }
                    else -> EmptyOutput()
                }
            }
        }
    }
}

@Composable
private fun ResultContent(presentation: MaplePresentation) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        SelectionContainer {
            Text(
                text = presentation.body.ifBlank { "Maple returned no visible final answer." },
                color = PaleMint,
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        presentation.rows.forEach { row -> ResultRow(row) }
    }
}

@Composable
private fun ResultRow(row: MapleResultRow) {
    Surface(color = Ink.copy(alpha = 0.48f), shape = RoundedCornerShape(12.dp)) {
        Row(
            modifier = Modifier.padding(11.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Surface(color = Sky.copy(alpha = 0.16f), shape = RoundedCornerShape(7.dp)) {
                Text(
                    text = row.badge,
                    modifier = Modifier.padding(horizontal = 7.dp, vertical = 4.dp),
                    color = Sky,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                )
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(row.headline, fontWeight = FontWeight.SemiBold, color = PaleMint)
                Text(row.supporting, color = Muted, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun EmptyOutput() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("✦", color = Mint, fontSize = 26.sp)
            Spacer(Modifier.height(7.dp))
            Text("Choose a task and run the note", color = Muted)
        }
    }
}

@Composable
private fun MedicalDisclaimer() {
    Text(
        text = "Research preview — not a medical device. Maple can omit, misclassify, or invent information. A qualified professional must compare every output with the source record before clinical, privacy, or disclosure decisions.",
        modifier = Modifier.padding(horizontal = 4.dp),
        color = Muted.copy(alpha = 0.82f),
        style = MaterialTheme.typography.bodySmall,
    )
}

@Composable
private fun ErrorToast(message: String, onDismiss: () -> Unit, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        color = Color(0xFF5A2428),
        shape = RoundedCornerShape(14.dp),
        shadowElevation = 8.dp,
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(message, modifier = Modifier.weight(1f), color = Color.White)
            TextButton(onClick = onDismiss) { Text("Dismiss", color = Color.White) }
        }
    }
}

@Composable
private fun StatusDot(color: Color) {
    Box(
        modifier = Modifier
            .size(9.dp)
            .background(color, CircleShape),
    )
}

@Preview(widthDp = 412, heightDp = 900)
@Composable
private fun StudioPreview() {
    MapleStudioTheme {
        Box(modifier = Modifier.background(Ink).padding(18.dp)) {
            Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                StudioHeader()
                PrivacyRibbon()
                OutputPanel(
                    state = MapleDemoUiState(
                        isLoadingBundle = false,
                        selectedTask = MapleTask.RELATIONS,
                        presentation = SyntheticPreviewResults.forTask(MapleTask.RELATIONS),
                        isSyntheticPreview = true,
                    ),
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}
