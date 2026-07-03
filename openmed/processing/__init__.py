"""Text processing utilities for OpenMed."""

from . import sentences
from .advanced_ner import (
    StreamingReplayResult,
    StreamingTokenClassifier,
    replay_token_classifier,
    stream_token_classifier,
)
from .batch import (
    BatchItem,
    BatchItemResult,
    BatchProcessor,
    BatchProgress,
    BatchResult,
    DatasetRedactionResult,
    DatasetRedactionSummary,
    process_batch,
    redact_dataset,
)
from .kafka_connector import (
    ConsumerProtocol,
    KafkaClientPair,
    ProducerProtocol,
    create_confluent_kafka_clients,
    deidentify_stream,
)
from .outputs import OutputFormatter, format_predictions
from .text import TextProcessor, postprocess_text, preprocess_text
from .tokenization import TokenizationHelper, infer_tokenizer_max_length
from .tokenizer_cache import clear_tokenizer_cache, get_tokenizer

__all__ = [
    "TextProcessor",
    "preprocess_text",
    "postprocess_text",
    "TokenizationHelper",
    "infer_tokenizer_max_length",
    "get_tokenizer",
    "clear_tokenizer_cache",
    "OutputFormatter",
    "format_predictions",
    "BatchProcessor",
    "BatchItem",
    "BatchItemResult",
    "BatchProgress",
    "BatchResult",
    "DatasetRedactionResult",
    "DatasetRedactionSummary",
    "process_batch",
    "redact_dataset",
    "StreamingReplayResult",
    "StreamingTokenClassifier",
    "replay_token_classifier",
    "stream_token_classifier",
    "ConsumerProtocol",
    "ProducerProtocol",
    "KafkaClientPair",
    "create_confluent_kafka_clients",
    "deidentify_stream",
    "sentences",
]
