"""Advanced NER processing with proven filtering techniques from OpenMed Gradio app."""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntitySpan:
    """Represents a single entity span with position information."""

    text: str
    label: str
    start: int
    end: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }


class AdvancedNERProcessor:
    """
    Advanced NER processor implementing proven filtering techniques.

    Based on the OpenMed Gradio app's successful approach:
    - Smart BIO token grouping
    - Confidence threshold filtering
    - Content validation
    - Edge case handling
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        min_length: int = 1,
        remove_punctuation: bool = True,
        strip_edges: bool = True,
        merge_adjacent: bool = True,
        max_merge_gap: int = 10,
    ):
        """
        Initialize the advanced NER processor.

        Args:
            min_confidence: Minimum confidence threshold for entities
            min_length: Minimum length for valid entities
            remove_punctuation: Whether to filter punctuation-only entities
            strip_edges: Whether to strip punctuation from entity edges
            merge_adjacent: Whether to merge adjacent entities of same type
            max_merge_gap: Maximum character gap for merging adjacent entities
        """
        self.min_confidence = min_confidence
        self.min_length = min_length
        self.remove_punctuation = remove_punctuation
        self.strip_edges = strip_edges
        self.merge_adjacent = merge_adjacent
        self.max_merge_gap = max_merge_gap

        # Regex for content detection
        self.has_content = (
            re.compile(r"[A-Za-z0-9]") if remove_punctuation else re.compile(r".")
        )

        # Patterns to exclude (known false positives)
        self.exclude_patterns = [
            r"^[\s\-.,!?;:()[\]{}\"'_]+$",  # Only punctuation/whitespace
            r"^\d{1,2}$",  # Single/double digits only
            r"^[.,!?;:]+$",  # Only punctuation
        ]

    def ner_filtered(
        self, text: str, pipeline_result: List[Dict[str, Any]]
    ) -> List[EntitySpan]:
        """
        Apply confidence and punctuation filtering to NER pipeline results.
        This is the proven filtering approach that eliminates spurious predictions.

        Args:
            text: Original input text
            pipeline_result: Raw output from HuggingFace NER pipeline

        Returns:
            List of filtered EntitySpan objects
        """
        logger.debug(f"Processing {len(pipeline_result)} raw entities")

        filtered_entities = []

        for entity in pipeline_result:
            # Confidence filter
            if entity.get("score", 0) < self.min_confidence:
                continue

            word = entity.get("word", "")

            # Length filter
            if len(word.strip()) < self.min_length:
                continue

            # Content filter - must have actual content
            if self.remove_punctuation and not self.has_content.search(word):
                continue

            # Exclude pattern filter
            if any(re.match(pattern, word) for pattern in self.exclude_patterns):
                continue

            # Create EntitySpan
            span = EntitySpan(
                text=word,
                label=entity.get("entity", "").replace("B-", "").replace("I-", ""),
                start=entity.get("start", 0),
                end=entity.get("end", 0),
                score=entity.get("score", 0.0),
            )

            filtered_entities.append(span)

        logger.debug(f"After filtering: {len(filtered_entities)} entities")
        return filtered_entities

    def smart_group_entities(
        self, tokens: List[Dict[str, Any]], text: str
    ) -> List[EntitySpan]:
        """
        Smart entity grouping that properly merges sub-tokens into complete entities.

        This fixes the issue where aggregation_strategy="simple" creates overlapping spans
        by implementing proper BIO tag handling.

        Args:
            tokens: Raw token-level predictions from transformer model
            text: Original input text

        Returns:
            List of properly grouped EntitySpan objects
        """
        if not tokens:
            return []

        entities = []
        current_entity = None

        for token in tokens:
            label = token.get("entity", "O")
            score = token.get("score", 0.0)
            start = token.get("start", 0)
            end = token.get("end", 0)

            # Skip O (Outside) tags
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            # Clean the label (remove B- and I- prefixes)
            clean_label = label.replace("B-", "").replace("I-", "")

            # Start new entity (B- tag or different entity type)
            if label.startswith("B-") or (
                current_entity and current_entity.label != clean_label
            ):
                if current_entity:
                    entities.append(current_entity)

                current_entity = EntitySpan(
                    text=text[start:end],
                    label=clean_label,
                    start=start,
                    end=end,
                    score=score,
                )

            # Continue current entity (I- tag)
            elif current_entity and clean_label == current_entity.label:
                # Extend the current entity
                current_entity.end = end
                current_entity.text = text[current_entity.start : end]
                current_entity.score = (
                    current_entity.score + score
                ) / 2  # Average scores

        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)

        logger.debug(
            f"Smart grouping created {len(entities)} entities from {len(tokens)} tokens"
        )
        return entities

    def advanced_filter(
        self, entities: List[EntitySpan], text: str
    ) -> List[EntitySpan]:
        """
        Advanced filtering with edge stripping and additional quality checks.

        Args:
            entities: List of EntitySpan objects to filter
            text: Original input text

        Returns:
            List of filtered EntitySpan objects
        """
        filtered = []

        for entity in entities:
            # Skip if below confidence threshold
            if entity.score < self.min_confidence:
                continue

            original_text = entity.text

            # Strip punctuation from edges if enabled
            if self.strip_edges:
                stripped = original_text.strip(".,!?;:()[]{}\"'-_")
                if not stripped:
                    continue

                # Update entity text and positions if stripped
                if stripped != original_text:
                    # Find new start/end positions
                    start_offset = original_text.find(stripped)
                    entity.text = stripped
                    entity.start += start_offset
                    entity.end = entity.start + len(stripped)

            # Final content validation
            if not re.search(r"[A-Za-z0-9]", entity.text):
                continue

            # Length check after stripping
            if len(entity.text.strip()) < self.min_length:
                continue

            filtered.append(entity)

        return filtered

    def merge_adjacent_entities(
        self, entities: List[EntitySpan], original_text: str
    ) -> List[EntitySpan]:
        """
        Merge adjacent entities of the same type that are separated by small gaps.

        Useful for handling cases like "BRCA1 and BRCA2" or "HER2-positive".

        Args:
            entities: List of EntitySpan objects to potentially merge
            original_text: Original input text

        Returns:
            List of EntitySpan objects with adjacent entities merged
        """
        if len(entities) < 2:
            return entities

        merged = []
        current = entities[0]

        for next_entity in entities[1:]:
            # Check if same entity type and close proximity
            if (
                current.label == next_entity.label
                and next_entity.start - current.end <= self.max_merge_gap
            ):

                # Check what's between them
                gap_text = original_text[current.end : next_entity.start]

                # Merge if gap contains only connecting words/punctuation
                if re.match(r"^[\s\-,/and]*$", gap_text.lower()):
                    # Extend current entity to include the next one
                    current.text = original_text[current.start : next_entity.end]
                    current.end = next_entity.end
                    current.score = (current.score + next_entity.score) / 2
                    continue

            # No merge, add current and move to next
            merged.append(current)
            current = next_entity

        # Don't forget the last entity
        merged.append(current)

        logger.debug(f"Merge process: {len(entities)} -> {len(merged)} entities")
        return merged

    def process_pipeline_output(
        self,
        text: str,
        pipeline_output: List[Dict[str, Any]],
        use_smart_grouping: bool = True,
    ) -> List[EntitySpan]:
        """
        Complete processing pipeline for NER output.

        Args:
            text: Original input text
            pipeline_output: Raw output from HuggingFace pipeline
            use_smart_grouping: Whether to use smart BIO token grouping

        Returns:
            List of processed EntitySpan objects
        """
        logger.info(
            f"Processing pipeline output with {len(pipeline_output)} raw predictions"
        )

        # Step 1: Smart grouping if requested and we have token-level output
        if use_smart_grouping and pipeline_output:
            # Check if we have token-level data (no aggregation_strategy used)
            first_item = pipeline_output[0]
            if "entity" in first_item and not "entity_group" in first_item:
                entities = self.smart_group_entities(pipeline_output, text)
            else:
                # Already grouped, convert to EntitySpan format
                entities = []
                for item in pipeline_output:
                    span = EntitySpan(
                        text=item.get("word", ""),
                        label=item.get("entity_group", item.get("entity", "")),
                        start=item.get("start", 0),
                        end=item.get("end", 0),
                        score=item.get("score", 0.0),
                    )
                    entities.append(span)
        else:
            # Use basic filtering approach
            entities = self.ner_filtered(text, pipeline_output)

        # Step 2: Advanced filtering
        entities = self.advanced_filter(entities, text)

        # Step 3: Merge adjacent entities if enabled
        if self.merge_adjacent:
            entities = self.merge_adjacent_entities(entities, text)

        logger.info(f"Final result: {len(entities)} high-quality entities")
        return entities

    def create_entity_summary(self, entities: List[EntitySpan]) -> Dict[str, Any]:
        """
        Create a summary of detected entities.

        Args:
            entities: List of EntitySpan objects

        Returns:
            Dictionary with entity statistics and examples
        """
        if not entities:
            return {"total": 0, "by_type": {}, "confidence_stats": {}}

        by_type = {}
        scores = []

        for entity in entities:
            label = entity.label
            if label not in by_type:
                by_type[label] = {
                    "count": 0,
                    "examples": [],
                    "avg_confidence": 0.0,
                    "scores": [],
                }

            by_type[label]["count"] += 1
            by_type[label]["scores"].append(entity.score)
            scores.append(entity.score)

            # Keep unique examples (up to 5)
            if (
                entity.text not in by_type[label]["examples"]
                and len(by_type[label]["examples"]) < 5
            ):
                by_type[label]["examples"].append(entity.text)

        # Calculate average confidences
        for label in by_type:
            by_type[label]["avg_confidence"] = sum(by_type[label]["scores"]) / len(
                by_type[label]["scores"]
            )
            del by_type[label]["scores"]  # Remove raw scores from output

        confidence_stats = {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
        }

        return {
            "total": len(entities),
            "by_type": by_type,
            "confidence_stats": confidence_stats,
            "filter_settings": {
                "min_confidence": self.min_confidence,
                "min_length": self.min_length,
                "remove_punctuation": self.remove_punctuation,
                "strip_edges": self.strip_edges,
                "merge_adjacent": self.merge_adjacent,
            },
        }


def create_advanced_processor(
    confidence_threshold: float = 0.60, **kwargs
) -> AdvancedNERProcessor:
    """
    Convenience function to create an AdvancedNERProcessor with recommended settings.

    Args:
        confidence_threshold: Minimum confidence for entity predictions
        **kwargs: Additional arguments for AdvancedNERProcessor

    Returns:
        Configured AdvancedNERProcessor instance
    """
    return AdvancedNERProcessor(
        min_confidence=confidence_threshold,
        min_length=1,
        remove_punctuation=True,
        strip_edges=True,
        merge_adjacent=True,
        max_merge_gap=10,
        **kwargs,
    )
