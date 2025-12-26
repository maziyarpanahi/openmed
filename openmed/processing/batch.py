"""Batch processing utilities for OpenMed.

This module provides efficient batch processing capabilities for analyzing
multiple texts or files with progress reporting and result aggregation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

from .outputs import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represents a single item in a batch processing job."""

    id: str
    text: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchItemResult:
    """Result for a single batch item."""

    id: str
    result: Optional[PredictionResult] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    source: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if the item was processed successfully."""
        return self.error is None and self.result is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "success": self.success,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "processing_time": self.processing_time,
            "source": self.source,
        }


@dataclass
class BatchResult:
    """Aggregate result for a batch processing job."""

    items: List[BatchItemResult] = field(default_factory=list)
    total_processing_time: float = 0.0
    model_name: str = ""
    started_at: str = ""
    completed_at: str = ""

    @property
    def total_items(self) -> int:
        """Total number of items in the batch."""
        return len(self.items)

    @property
    def successful_items(self) -> int:
        """Number of successfully processed items."""
        return sum(1 for item in self.items if item.success)

    @property
    def failed_items(self) -> int:
        """Number of failed items."""
        return sum(1 for item in self.items if not item.success)

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if not self.items:
            return 0.0
        return (self.successful_items / self.total_items) * 100

    @property
    def average_processing_time(self) -> float:
        """Average processing time per item."""
        if not self.items:
            return 0.0
        return sum(item.processing_time for item in self.items) / len(self.items)

    def get_successful_results(self) -> List[BatchItemResult]:
        """Get only successful results."""
        return [item for item in self.items if item.success]

    def get_failed_results(self) -> List[BatchItemResult]:
        """Get only failed results."""
        return [item for item in self.items if not item.success]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "items": [item.to_dict() for item in self.items],
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the batch results."""
        lines = [
            f"Batch Processing Summary",
            f"========================",
            f"Model: {self.model_name}",
            f"Total items: {self.total_items}",
            f"Successful: {self.successful_items}",
            f"Failed: {self.failed_items}",
            f"Success rate: {self.success_rate:.1f}%",
            f"Total time: {self.total_processing_time:.2f}s",
            f"Average time per item: {self.average_processing_time:.3f}s",
        ]
        return "\n".join(lines)


# Type alias for progress callback
ProgressCallback = Callable[[int, int, Optional[BatchItemResult]], None]


class BatchProcessor:
    """Process multiple texts efficiently with progress tracking.

    Example usage:
        >>> from openmed import BatchProcessor, OpenMedConfig
        >>> processor = BatchProcessor(model_name="disease_detection_superclinical")
        >>> texts = ["Patient has diabetes.", "No significant findings."]
        >>> result = processor.process_texts(texts)
        >>> print(result.summary())
    """

    def __init__(
        self,
        model_name: str = "disease_detection_superclinical",
        config: Optional[Any] = None,
        loader: Optional[Any] = None,
        aggregation_strategy: Optional[str] = "simple",
        confidence_threshold: float = 0.0,
        group_entities: bool = False,
        continue_on_error: bool = True,
        **analyze_kwargs: Any,
    ):
        """Initialize batch processor.

        Args:
            model_name: Model registry key or HuggingFace identifier.
            config: Optional OpenMedConfig instance.
            loader: Optional ModelLoader instance to reuse.
            aggregation_strategy: HuggingFace aggregation strategy.
            confidence_threshold: Minimum confidence for entities.
            group_entities: Whether to group adjacent entities.
            continue_on_error: Continue processing on individual item errors.
            **analyze_kwargs: Additional arguments passed to analyze_text.
        """
        self.model_name = model_name
        self.config = config
        self.loader = loader
        self.aggregation_strategy = aggregation_strategy
        self.confidence_threshold = confidence_threshold
        self.group_entities = group_entities
        self.continue_on_error = continue_on_error
        self.analyze_kwargs = analyze_kwargs

        self._analyze_text = None

    def _get_analyze_text(self) -> Callable:
        """Lazily import and cache analyze_text function."""
        if self._analyze_text is None:
            from openmed import analyze_text
            self._analyze_text = analyze_text
        return self._analyze_text

    def _create_batch_items(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
    ) -> List[BatchItem]:
        """Create BatchItem objects from raw texts."""
        items = []
        for i, text in enumerate(texts):
            item_id = ids[i] if ids and i < len(ids) else f"item_{i}"
            items.append(BatchItem(id=item_id, text=text))
        return items

    def process_texts(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """Process multiple texts.

        Args:
            texts: Sequence of texts to analyze.
            ids: Optional identifiers for each text.
            progress_callback: Optional callback for progress updates.
                Signature: callback(current_index, total_count, result)

        Returns:
            BatchResult with all processing results.
        """
        items = self._create_batch_items(texts, ids)
        return self._process_items(items, progress_callback)

    def process_files(
        self,
        file_paths: Sequence[Union[str, Path]],
        encoding: str = "utf-8",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """Process multiple files.

        Args:
            file_paths: Paths to text files.
            encoding: File encoding.
            progress_callback: Optional callback for progress updates.

        Returns:
            BatchResult with all processing results.
        """
        items = []
        for path in file_paths:
            path = Path(path)
            try:
                text = path.read_text(encoding=encoding)
                items.append(
                    BatchItem(
                        id=path.name,
                        text=text,
                        source=str(path),
                    )
                )
            except (OSError, IOError) as e:
                logger.warning(f"Failed to read file {path}: {e}")
                if not self.continue_on_error:
                    raise
                items.append(
                    BatchItem(
                        id=path.name,
                        text="",
                        source=str(path),
                        metadata={"read_error": str(e)},
                    )
                )
        return self._process_items(items, progress_callback)

    def process_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.txt",
        recursive: bool = False,
        encoding: str = "utf-8",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """Process all matching files in a directory.

        Args:
            directory: Directory path.
            pattern: Glob pattern for file matching.
            recursive: Whether to search recursively.
            encoding: File encoding.
            progress_callback: Optional callback for progress updates.

        Returns:
            BatchResult with all processing results.
        """
        directory = Path(directory)
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        files.sort()
        return self.process_files(files, encoding, progress_callback)

    def process_items(
        self,
        items: Sequence[BatchItem],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """Process a sequence of BatchItem objects.

        Args:
            items: Sequence of BatchItem objects.
            progress_callback: Optional callback for progress updates.

        Returns:
            BatchResult with all processing results.
        """
        return self._process_items(list(items), progress_callback)

    def _process_items(
        self,
        items: List[BatchItem],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> BatchResult:
        """Internal method to process batch items."""
        from datetime import datetime

        analyze_text = self._get_analyze_text()

        batch_result = BatchResult(
            model_name=self.model_name,
            started_at=datetime.now().isoformat(),
        )

        total = len(items)
        batch_start = time.time()

        for i, item in enumerate(items):
            item_result = self._process_single_item(item, analyze_text)
            batch_result.items.append(item_result)

            if progress_callback:
                try:
                    progress_callback(i + 1, total, item_result)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        batch_result.total_processing_time = time.time() - batch_start
        batch_result.completed_at = datetime.now().isoformat()

        return batch_result

    def _process_single_item(
        self,
        item: BatchItem,
        analyze_text: Callable,
    ) -> BatchItemResult:
        """Process a single batch item."""
        if not item.text:
            read_error = (item.metadata or {}).get("read_error")
            return BatchItemResult(
                id=item.id,
                error=read_error or "Empty text",
                source=item.source,
            )

        start_time = time.time()
        try:
            result = analyze_text(
                item.text,
                model_name=self.model_name,
                config=self.config,
                loader=self.loader,
                aggregation_strategy=self.aggregation_strategy,
                confidence_threshold=self.confidence_threshold,
                group_entities=self.group_entities,
                output_format="dict",
                metadata=item.metadata,
                **self.analyze_kwargs,
            )
            processing_time = time.time() - start_time

            return BatchItemResult(
                id=item.id,
                result=result,
                processing_time=processing_time,
                source=item.source,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"Error processing item {item.id}: {e}")

            if not self.continue_on_error:
                raise

            return BatchItemResult(
                id=item.id,
                error=str(e),
                processing_time=processing_time,
                source=item.source,
            )

    def iter_process(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
    ) -> Iterator[BatchItemResult]:
        """Process texts as an iterator, yielding results one at a time.

        This is useful for streaming results or processing very large batches
        where you don't want to hold all results in memory.

        Args:
            texts: Sequence of texts to analyze.
            ids: Optional identifiers for each text.

        Yields:
            BatchItemResult for each processed text.
        """
        analyze_text = self._get_analyze_text()
        items = self._create_batch_items(texts, ids)

        for item in items:
            yield self._process_single_item(item, analyze_text)


def process_batch(
    texts: Sequence[str],
    model_name: str = "disease_detection_superclinical",
    ids: Optional[Sequence[str]] = None,
    config: Optional[Any] = None,
    progress_callback: Optional[ProgressCallback] = None,
    **kwargs: Any,
) -> BatchResult:
    """Convenience function for batch processing texts.

    Args:
        texts: Sequence of texts to analyze.
        model_name: Model registry key or HuggingFace identifier.
        ids: Optional identifiers for each text.
        config: Optional OpenMedConfig instance.
        progress_callback: Optional callback for progress updates.
        **kwargs: Additional arguments passed to BatchProcessor.

    Returns:
        BatchResult with all processing results.

    Example:
        >>> from openmed import process_batch
        >>> texts = ["Patient has diabetes.", "Normal findings."]
        >>> result = process_batch(texts, model_name="disease_detection_superclinical")
        >>> print(f"Processed {result.successful_items}/{result.total_items} texts")
    """
    processor = BatchProcessor(model_name=model_name, config=config, **kwargs)
    return processor.process_texts(texts, ids, progress_callback)
