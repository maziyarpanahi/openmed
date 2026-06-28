"""Text processing utilities for OpenMed."""

from . import sentences
from .batch import (
    BatchItem,
    BatchItemResult,
    BatchProcessor,
    BatchProgress,
    BatchResult,
    process_batch,
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

__all__ = [
    "TextProcessor",
    "preprocess_text",
    "postprocess_text",
    "TokenizationHelper",
    "infer_tokenizer_max_length",
    "OutputFormatter",
    "format_predictions",
    "BatchProcessor",
    "BatchItem",
    "BatchItemResult",
    "BatchProgress",
    "BatchResult",
    "process_batch",
    "ConsumerProtocol",
    "ProducerProtocol",
    "KafkaClientPair",
    "create_confluent_kafka_clients",
    "deidentify_stream",
    "sentences",
]
