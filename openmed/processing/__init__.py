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
from .checkpoint import (
    DEDUPE_HEADER,
    CheckpointRecord,
    InMemoryCheckpointStore,
    LocalFileCheckpointStore,
    OutputPosition,
    SourcePosition,
    StreamFingerprint,
    build_stream_fingerprint,
    dedupe_key_for_source,
)
from .display import NormalizedSpan, render_spans_html, show
from .distributed import (
    DocumentIdExtractor,
    DocumentShard,
    DuplicateDocumentIDError,
    MissingDocumentIDError,
    ShardingError,
    ShardPlan,
    assign_document_shard,
    plan_document_shards,
    stable_document_hash,
)
from .kafka_connector import (
    CheckpointFingerprintError,
    ConsumerProtocol,
    KafkaClientPair,
    ProducerProtocol,
    create_confluent_kafka_clients,
    deidentify_stream,
    replay,
)
from .object_storage import (
    ObjectProgressCallback,
    ObjectStorageBatchResult,
    ObjectStorageItemResult,
    deidentify_bucket,
)
from .outputs import OutputFormatter, format_predictions
from .pulsar_connector import PulsarClientPair, create_pulsar_clients
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
    "render_spans_html",
    "show",
    "NormalizedSpan",
    "BatchProcessor",
    "BatchItem",
    "BatchItemResult",
    "BatchProgress",
    "BatchResult",
    "DatasetRedactionResult",
    "DatasetRedactionSummary",
    "process_batch",
    "redact_dataset",
    "DEDUPE_HEADER",
    "CheckpointRecord",
    "InMemoryCheckpointStore",
    "LocalFileCheckpointStore",
    "OutputPosition",
    "SourcePosition",
    "StreamFingerprint",
    "build_stream_fingerprint",
    "dedupe_key_for_source",
    "CheckpointFingerprintError",
    "DocumentIdExtractor",
    "DocumentShard",
    "ShardPlan",
    "ShardingError",
    "MissingDocumentIDError",
    "DuplicateDocumentIDError",
    "stable_document_hash",
    "assign_document_shard",
    "plan_document_shards",
    "ObjectStorageBatchResult",
    "ObjectStorageItemResult",
    "ObjectProgressCallback",
    "deidentify_bucket",
    "StreamingReplayResult",
    "StreamingTokenClassifier",
    "replay_token_classifier",
    "stream_token_classifier",
    "ConsumerProtocol",
    "ProducerProtocol",
    "KafkaClientPair",
    "create_confluent_kafka_clients",
    "PulsarClientPair",
    "create_pulsar_clients",
    "deidentify_stream",
    "replay",
    "sentences",
]
