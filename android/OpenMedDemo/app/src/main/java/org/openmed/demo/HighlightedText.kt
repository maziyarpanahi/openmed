package org.openmed.demo

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import java.util.Locale

data class DetectedEntity(
    val label: String,
    val text: String,
    val confidence: Float,
    val start: Int,
    val end: Int,
    val category: EntityCategory,
)

enum class EntityCategory(val displayName: String, val color: Color) {
    Name("Name", Color(0xFF1D4ED8)),
    Date("Date", Color(0xFF7E22CE)),
    Phone("Phone", Color(0xFF047857)),
    Email("Email", Color(0xFF0F766E)),
    Address("Address", Color(0xFFC2410C)),
    Identifier("ID", Color(0xFFBE123C)),
    Organization("Org", Color(0xFF4338CA)),
    Other("Other", Color(0xFF4B5563)),
}

data class HighlightSegment(
    val text: String,
    val entity: DetectedEntity?,
    val start: Int,
    val end: Int,
)

fun mapHighlightSegments(text: String, entities: List<DetectedEntity>): List<HighlightSegment> {
    val normalized = entities
        .filter { it.start >= 0 && it.end <= text.length && it.start < it.end }
        .sortedWith(compareBy<DetectedEntity> { it.start }.thenBy { it.end })

    val segments = mutableListOf<HighlightSegment>()
    var cursor = 0

    for (entity in normalized) {
        if (entity.start < cursor) {
            continue
        }

        if (cursor < entity.start) {
            segments += HighlightSegment(
                text = text.substring(cursor, entity.start),
                entity = null,
                start = cursor,
                end = entity.start,
            )
        }

        segments += HighlightSegment(
            text = text.substring(entity.start, entity.end),
            entity = entity,
            start = entity.start,
            end = entity.end,
        )
        cursor = entity.end
    }

    if (cursor < text.length) {
        segments += HighlightSegment(
            text = text.substring(cursor),
            entity = null,
            start = cursor,
            end = text.length,
        )
    }

    return segments
}

@Composable
fun HighlightedText(
    text: String,
    entities: List<DetectedEntity>,
    modifier: Modifier = Modifier,
) {
    val annotated = buildAnnotatedString {
        for (segment in mapHighlightSegments(text, entities)) {
            val entity = segment.entity
            if (entity == null) {
                append(segment.text)
            } else {
                pushStyle(
                    SpanStyle(
                        background = entity.category.color.copy(alpha = 0.16f),
                        color = MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.SemiBold,
                    )
                )
                append(segment.text)
                pop()
            }
        }
    }

    Text(
        text = annotated,
        modifier = modifier
            .fillMaxWidth()
            .wrapContentHeight()
            .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp))
            .padding(12.dp),
        fontFamily = FontFamily.Monospace,
        style = MaterialTheme.typography.bodyMedium,
    )
}

@Composable
fun EntityChipList(
    entities: List<DetectedEntity>,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        for (entity in entities) {
            EntityChip(entity = entity)
        }
    }
}

@Composable
private fun EntityChip(entity: DetectedEntity) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = entity.category.color.copy(alpha = 0.10f),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = entity.label,
                    color = entity.category.color,
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.labelLarge,
                )
                Text(
                    text = entity.text,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
            Text(
                text = "${String.format(Locale.US, "%.0f", entity.confidence * 100)}%",
                color = entity.category.color,
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.labelLarge,
            )
        }
    }
}
