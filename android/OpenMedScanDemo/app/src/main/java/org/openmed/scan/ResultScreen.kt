package org.openmed.scan

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp

@Composable
fun ResultScreen(
    result: ScanPipelineResult,
    source: ScanSource?,
    onCaptureAnother: () -> Unit,
) {
    Card(
        modifier = Modifier.widthIn(max = 860.dp),
        shape = RoundedCornerShape(8.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text(
                        text = "Highlighted Redaction",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = "${source?.label ?: "Scan"} source, ${result.ocrTokenCount} OCR tokens, ${result.deidentified.spans.size} spans",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color(0xFF64748B),
                    )
                }
                Button(onClick = onCaptureAnother) {
                    Text("New Scan")
                }
            }
            Column(
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                spanCounts(result.deidentified.spans).forEach { (label, count) ->
                    Surface(
                        color = Color(0xFFE0F2FE),
                        contentColor = Color(0xFF075985),
                        shape = MaterialTheme.shapes.small,
                    ) {
                        Text(
                            text = "$label $count",
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                }
            }
            SelectionContainer {
                Surface(
                    color = Color(0xFFF8FAFC),
                    shape = MaterialTheme.shapes.medium,
                ) {
                    Text(
                        text = highlightedText(result.deidentified.segments),
                        modifier = Modifier.padding(14.dp),
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontFamily = FontFamily.Monospace,
                        ),
                    )
                }
            }
        }
    }
}

private fun spanCounts(spans: List<RedactionSpan>): List<Pair<String, Int>> =
    spans.groupingBy { it.label }
        .eachCount()
        .toList()
        .sortedBy { it.first }

private fun highlightedText(segments: List<ResultSegment>) =
    buildAnnotatedString {
        segments.forEach { segment ->
            if (segment.label == null) {
                append(segment.text)
            } else {
                withStyle(
                    SpanStyle(
                        background = Color(0xFFFFF3B0),
                        color = Color(0xFF713F12),
                        fontWeight = FontWeight.Bold,
                    ),
                ) {
                    append(segment.text)
                }
            }
        }
    }
