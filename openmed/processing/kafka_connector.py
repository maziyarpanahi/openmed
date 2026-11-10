"""Kafka streaming connector for OpenMed de-identification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from openmed.__about__ import __version__
from openmed.core.policy import PolicyName, canonical_policy_name
from openmed.processing.checkpoint import (
    DEDUPE_HEADER,
    CheckpointRecord,
    CheckpointStore,
    OutputPosition,
    SourcePosition,
    StreamFingerprint,
    build_stream_fingerprint,
    checkpoint_for_delivery,
    dedupe_key_for_source,
)

DEFAULT_POLL_TIMEOUT = 1.0
DEFAULT_POLICY = "hipaa_safe_harbor"
PROVENANCE_FIELD = "deid_provenance"

logger = logging.getLogger(__name__)


class ConsumerProtocol(Protocol):
    """Minimal consumer surface required by :func:`deidentify_stream`."""

    def poll(self, timeout: float | None = None) -> Any | None:
        """Return the next message, or ``None`` when no message is available."""

    def commit(self, message: Any | None = None) -> Any:
        """Commit the consumed message offset after successful production."""

    def begin_transaction(self) -> Any:
        """Optional hook called before a transactional stream unit starts."""

    def commit_transaction(self) -> Any:
        """Optional hook called after output, checkpoint, and source commit."""

    def abort_transaction(self) -> Any:
        """Optional hook called when a transactional stream unit fails."""


class ProducerProtocol(Protocol):
    """Minimal producer surface required by :func:`deidentify_stream`."""

    def produce(
        self,
        topic: str,
        value: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Produce a redacted record to ``topic``."""

    def begin_transaction(self) -> Any:
        """Optional hook called before a transactional stream unit starts."""

    def commit_transaction(self) -> Any:
        """Optional hook called after output, checkpoint, and source commit."""

    def abort_transaction(self) -> Any:
        """Optional hook called when a transactional stream unit fails."""


@dataclass(frozen=True)
class KafkaClientPair:
    """Consumer/producer pair returned by concrete Kafka factory helpers."""

    consumer: ConsumerProtocol
    producer: ProducerProtocol


class KafkaConnectorError(RuntimeError):
    """Base exception for Kafka connector runtime failures."""


class KafkaMessageError(KafkaConnectorError):
    """Raised when a consumed Kafka message carries a client error."""


class CheckpointFingerprintError(KafkaConnectorError):
    """Raised when checkpoint fingerprints do not match the current run."""


def deidentify_stream(
    consumer: ConsumerProtocol,
    producer: ProducerProtocol,
    *,
    in_topic: str,
    out_topic: str,
    text_field: str = "text",
    policy: str | PolicyName = DEFAULT_POLICY,
    poll_timeout: float | None = DEFAULT_POLL_TIMEOUT,
    max_messages: int | None = None,
    idle_polls: int | None = 1,
    provenance_field: str = PROVENANCE_FIELD,
    checkpoint_store: CheckpointStore | None = None,
    dedupe_header: str = DEDUPE_HEADER,
    **deidentify_kwargs: Any,
) -> int:
    """Consume records, de-identify their text field, and produce redacted output.

    Offsets are committed only after ``producer.produce`` returns successfully.
    When ``checkpoint_store`` is supplied, each consume -> de-identify ->
    produce -> checkpoint -> commit cycle is wrapped in optional transaction
    hooks exposed by the consumer and producer. Checkpoints are persisted only
    after the redacted output is acknowledged, and dedupe headers are derived
    from source position metadata rather than record text.

    Args:
        consumer: Object exposing ``poll`` and ``commit``.
        producer: Object exposing ``produce``.
        in_topic: Topic to subscribe to when the consumer supports ``subscribe``.
        out_topic: Topic receiving redacted records.
        text_field: Record field containing the note text.
        policy: De-identification policy profile name.
        poll_timeout: Timeout passed to ``consumer.poll``.
        max_messages: Optional cap for bounded runs.
        idle_polls: Stop after this many empty polls. Use ``None`` for an
            unbounded stream that keeps polling during idle periods.
        provenance_field: Output field receiving OpenMed provenance metadata.
        checkpoint_store: Optional durable checkpoint store for exactly-once
            recovery.
        dedupe_header: Header name receiving the source-position dedupe key.
        **deidentify_kwargs: Extra keyword arguments forwarded to
            :func:`openmed.deidentify`.

    Returns:
        Number of messages handled. Recovered messages already covered by a
        checkpoint are counted after their source offset is committed.
    """

    _validate_stream_args(
        in_topic=in_topic,
        out_topic=out_topic,
        text_field=text_field,
        max_messages=max_messages,
        idle_polls=idle_polls,
        provenance_field=provenance_field,
    )
    if not isinstance(dedupe_header, str) or not dedupe_header.strip():
        raise ValueError("dedupe_header must be a non-empty string")
    policy_name = canonical_policy_name(policy)
    fingerprint = build_stream_fingerprint(
        policy_name=policy_name,
        deidentify_kwargs=deidentify_kwargs,
    )
    if checkpoint_store is not None:
        _ensure_idempotent_contract(producer)
    _subscribe_if_supported(consumer, in_topic)

    processed = 0
    idle_count = 0
    while max_messages is None or processed < max_messages:
        message = consumer.poll(poll_timeout)
        if message is None:
            idle_count += 1
            if idle_polls is not None and idle_count >= idle_polls:
                break
            continue

        idle_count = 0
        _raise_for_message_error(message)
        source_position = (
            _source_position_from_message(message, fallback_topic=in_topic)
            if checkpoint_store is not None
            else None
        )
        if source_position is not None and _commit_if_checkpointed(
            consumer=consumer,
            message=message,
            source_position=source_position,
            checkpoint_store=checkpoint_store,
            fingerprint=fingerprint,
        ):
            processed += 1
            continue

        record = _decode_record(message)
        redacted_record = _deidentify_record(
            record,
            text_field=text_field,
            policy_name=policy_name,
            provenance_field=provenance_field,
            deidentify_kwargs=deidentify_kwargs,
        )

        dedupe_key = (
            dedupe_key_for_source(source_position)
            if source_position is not None
            else None
        )
        headers = {dedupe_header: dedupe_key} if dedupe_key is not None else None

        if checkpoint_store is not None:
            _begin_transactions(consumer, producer)
        try:
            delivery = _produce_redacted_record(
                producer,
                out_topic,
                redacted_record,
                headers=headers,
            )
            if checkpoint_store is not None and source_position is not None:
                if dedupe_key is None:
                    raise KafkaConnectorError("dedupe key was not created")
                checkpoint = checkpoint_for_delivery(
                    source=source_position,
                    redacted_output=_output_position_from_delivery(
                        delivery,
                        topic=out_topic,
                        fallback_partition=source_position.partition,
                        fallback_offset=source_position.offset,
                    ),
                    fingerprint=fingerprint,
                    dedupe_key=dedupe_key,
                )
                checkpoint_store.save(checkpoint)
                _log_checkpoint_saved(checkpoint)
            consumer.commit(message)
            if checkpoint_store is not None:
                _commit_transactions(consumer, producer)
        except BaseException:
            if checkpoint_store is not None:
                _abort_transactions(consumer, producer)
            raise
        processed += 1

    return processed


def replay(
    consumer: ConsumerProtocol,
    producer: ProducerProtocol,
    *,
    from_checkpoint: CheckpointRecord,
    to_position: SourcePosition,
    out_topic: str,
    text_field: str = "text",
    policy: str | PolicyName = DEFAULT_POLICY,
    poll_timeout: float | None = DEFAULT_POLL_TIMEOUT,
    idle_polls: int | None = 1,
    provenance_field: str = PROVENANCE_FIELD,
    dedupe_header: str = DEDUPE_HEADER,
    **deidentify_kwargs: Any,
) -> int:
    """Replay a bounded source window with pinned policy/model fingerprints.

    The consumer may expose ``seek(SourcePosition)``. If it does not, it must
    already be positioned at or before ``from_checkpoint.source``. Replay does
    not commit source offsets or write checkpoints; it only re-emits redacted
    output records with deterministic dedupe headers.
    """

    _validate_topic(out_topic, "out_topic")
    if not isinstance(dedupe_header, str) or not dedupe_header.strip():
        raise ValueError("dedupe_header must be a non-empty string")
    if to_position.topic != from_checkpoint.source.topic:
        raise ValueError("to_position topic must match from_checkpoint source")
    if to_position.partition != from_checkpoint.source.partition:
        raise ValueError("to_position partition must match from_checkpoint source")
    if to_position.offset < from_checkpoint.source.offset:
        raise ValueError("to_position offset must be at or after from_checkpoint")

    policy_name = canonical_policy_name(policy)
    fingerprint = build_stream_fingerprint(
        policy_name=policy_name,
        deidentify_kwargs=deidentify_kwargs,
    )
    _ensure_checkpoint_fingerprint(from_checkpoint, fingerprint)
    _seek_if_supported(consumer, from_checkpoint.source)

    replayed = 0
    idle_count = 0
    while True:
        message = consumer.poll(poll_timeout)
        if message is None:
            idle_count += 1
            if idle_polls is not None and idle_count >= idle_polls:
                break
            continue

        idle_count = 0
        _raise_for_message_error(message)
        source_position = _source_position_from_message(
            message,
            fallback_topic=from_checkpoint.source.topic,
        )
        if source_position.topic != from_checkpoint.source.topic:
            continue
        if source_position.partition != from_checkpoint.source.partition:
            continue
        if source_position.offset < from_checkpoint.source.offset:
            continue
        if source_position.offset > to_position.offset:
            break

        record = _decode_record(message)
        redacted_record = _deidentify_record(
            record,
            text_field=text_field,
            policy_name=policy_name,
            provenance_field=provenance_field,
            deidentify_kwargs=deidentify_kwargs,
        )
        dedupe_key = dedupe_key_for_source(source_position)
        _produce_redacted_record(
            producer,
            out_topic,
            redacted_record,
            headers={
                dedupe_header: dedupe_key,
                "openmed-replay": "true",
            },
        )
        replayed += 1
        logger.info(
            "stream_replay_record",
            extra={
                "source_topic": source_position.topic,
                "source_partition": source_position.partition,
                "source_offset": source_position.offset,
                "replayed_count": replayed,
            },
        )
        if source_position.offset == to_position.offset:
            break

    return replayed


def create_confluent_kafka_clients(
    *,
    consumer_config: Mapping[str, Any],
    producer_config: Mapping[str, Any],
    in_topic: str,
    delivery_timeout: float | None = 30.0,
) -> KafkaClientPair:
    """Create JSON record adapters backed by ``confluent-kafka``.

    ``confluent-kafka`` is imported only inside this helper so importing
    ``openmed`` or ``openmed.processing`` does not require a Kafka client. The
    producer adapter waits for delivery before returning so callers can commit
    offsets after ``produce`` succeeds.
    """

    _validate_topic(in_topic, "in_topic")
    try:
        from confluent_kafka import Consumer, Producer
    except ImportError as exc:
        raise ImportError(
            "Kafka support requires the optional kafka extra. "
            "Install with `pip install openmed[kafka]`."
        ) from exc

    resolved_producer_config = dict(producer_config)
    resolved_producer_config.setdefault("enable.idempotence", True)
    consumer = _ConfluentJsonConsumer(Consumer(dict(consumer_config)))
    producer = _ConfluentJsonProducer(
        Producer(resolved_producer_config),
        delivery_timeout=delivery_timeout,
        transactional=bool(resolved_producer_config.get("transactional.id")),
    )
    consumer.subscribe([in_topic])
    return KafkaClientPair(consumer=consumer, producer=producer)


class _ConfluentJsonConsumer:
    def __init__(self, consumer: Any) -> None:
        self._consumer = consumer

    def subscribe(self, topics: list[str]) -> Any:
        return self._consumer.subscribe(topics)

    def poll(self, timeout: float | None = DEFAULT_POLL_TIMEOUT) -> Any | None:
        return self._consumer.poll(timeout if timeout is not None else 0.0)

    def commit(self, message: Any | None = None) -> Any:
        return self._consumer.commit(message=message)


class _ConfluentJsonProducer:
    supports_idempotent_produce = True

    def __init__(
        self,
        producer: Any,
        *,
        delivery_timeout: float | None = 30.0,
        transactional: bool = False,
    ) -> None:
        self._producer = producer
        self._delivery_timeout = delivery_timeout
        self._transactional = transactional
        if self._transactional:
            init_transactions = getattr(self._producer, "init_transactions", None)
            if callable(init_transactions):
                init_transactions()

    def begin_transaction(self) -> Any:
        if not self._transactional:
            return None
        return self._producer.begin_transaction()

    def commit_transaction(self) -> Any:
        if not self._transactional:
            return None
        return self._producer.commit_transaction()

    def abort_transaction(self) -> Any:
        if not self._transactional:
            return None
        return self._producer.abort_transaction()

    def produce(
        self,
        topic: str,
        value: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        payload = json.dumps(
            dict(value),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        delivery_errors: list[Any] = []
        delivered_messages: list[Any] = []

        def on_delivery(error: Any, message: Any) -> None:
            if error is not None:
                delivery_errors.append(error)
            else:
                delivered_messages.append(message)

        produce_kwargs: dict[str, Any] = {
            "value": payload,
            "on_delivery": on_delivery,
        }
        if headers is not None:
            produce_kwargs["headers"] = dict(headers)
        result = self._producer.produce(topic, **produce_kwargs)
        remaining = (
            self._producer.flush()
            if self._delivery_timeout is None
            else self._producer.flush(self._delivery_timeout)
        )
        if remaining:
            raise KafkaConnectorError("Timed out waiting for Kafka producer delivery")
        if delivery_errors:
            raise KafkaConnectorError(
                f"Kafka producer delivery failed: {delivery_errors[0]}"
            )
        return delivered_messages[0] if delivered_messages else result


def _validate_stream_args(
    *,
    in_topic: str,
    out_topic: str,
    text_field: str,
    max_messages: int | None,
    idle_polls: int | None,
    provenance_field: str,
) -> None:
    _validate_topic(in_topic, "in_topic")
    _validate_topic(out_topic, "out_topic")
    if not isinstance(text_field, str) or not text_field.strip():
        raise ValueError("text_field must be a non-empty string")
    if max_messages is not None and max_messages < 0:
        raise ValueError("max_messages must be non-negative or None")
    if idle_polls is not None and idle_polls < 1:
        raise ValueError("idle_polls must be positive or None")
    if not isinstance(provenance_field, str) or not provenance_field.strip():
        raise ValueError("provenance_field must be a non-empty string")


def _validate_topic(topic: str, field_name: str) -> None:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _subscribe_if_supported(consumer: ConsumerProtocol, in_topic: str) -> None:
    subscribe = getattr(consumer, "subscribe", None)
    if callable(subscribe):
        subscribe([in_topic])


def _source_position_from_message(
    message: Any,
    *,
    fallback_topic: str,
) -> SourcePosition:
    embedded = _message_field(message, "source_position")
    if isinstance(embedded, SourcePosition):
        return embedded
    if isinstance(embedded, Mapping):
        return SourcePosition.from_dict(embedded)

    topic = _message_field(message, "source_topic", "topic")
    partition = _message_field(message, "source_partition", "partition")
    offset = _message_field(message, "source_offset", "offset")
    if topic is None:
        topic = fallback_topic
    if partition is None or offset is None:
        raise KafkaMessageError(
            "checkpointed streams require source partition and offset metadata"
        )
    try:
        return SourcePosition(
            topic=str(topic),
            partition=str(partition),
            offset=int(offset),
        )
    except (TypeError, ValueError) as exc:
        raise KafkaMessageError("invalid source partition or offset metadata") from exc


def _message_field(message: Any, *names: str) -> Any:
    if isinstance(message, Mapping):
        for name in names:
            if name in message:
                return message[name]
    for name in names:
        value = getattr(message, name, None)
        if callable(value):
            return value()
        if value is not None:
            return value
    return None


def _commit_if_checkpointed(
    *,
    consumer: ConsumerProtocol,
    message: Any,
    source_position: SourcePosition,
    checkpoint_store: CheckpointStore,
    fingerprint: StreamFingerprint,
) -> bool:
    checkpoint = checkpoint_store.load(
        source_position.topic,
        source_position.partition,
    )
    if checkpoint is None:
        return False
    _ensure_checkpoint_fingerprint(checkpoint, fingerprint)
    if checkpoint.source.offset < source_position.offset:
        return False
    consumer.commit(message)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "stream_checkpoint_recovered",
            extra={
                "source_topic": source_position.topic,
                "source_partition": source_position.partition,
                "source_offset": source_position.offset,
                "checkpoint_offset": checkpoint.source.offset,
            },
        )
    return True


def _ensure_checkpoint_fingerprint(
    checkpoint: CheckpointRecord,
    fingerprint: StreamFingerprint,
) -> None:
    if checkpoint.policy_fingerprint != fingerprint.policy:
        raise CheckpointFingerprintError("checkpoint policy fingerprint differs")
    if checkpoint.model_fingerprint != fingerprint.model:
        raise CheckpointFingerprintError("checkpoint model fingerprint differs")


def _ensure_idempotent_contract(producer: ProducerProtocol) -> None:
    supports = getattr(producer, "supports_idempotent_produce", None)
    if callable(supports):
        supports = supports()
    if supports is False:
        raise KafkaConnectorError(
            "checkpointed streams require idempotent produce support"
        )


def _produce_redacted_record(
    producer: ProducerProtocol,
    topic: str,
    value: Mapping[str, Any],
    *,
    headers: Mapping[str, str] | None,
) -> Any:
    if headers is None:
        return producer.produce(topic, value=value)
    try:
        return producer.produce(topic, value=value, headers=headers)
    except TypeError as exc:
        raise KafkaConnectorError(
            "producer must accept headers for checkpointed stream output"
        ) from exc


def _output_position_from_delivery(
    delivery: Any,
    *,
    topic: str,
    fallback_partition: str,
    fallback_offset: int,
) -> OutputPosition:
    if isinstance(delivery, OutputPosition):
        return delivery

    delivery_topic = _message_field(delivery, "topic") if delivery is not None else None
    delivery_partition = (
        _message_field(delivery, "partition") if delivery is not None else None
    )
    delivery_offset = (
        _message_field(delivery, "offset") if delivery is not None else None
    )
    return OutputPosition(
        topic=str(delivery_topic or topic),
        partition=str(
            fallback_partition if delivery_partition is None else delivery_partition
        ),
        offset=fallback_offset if delivery_offset is None else delivery_offset,
    )


def _begin_transactions(
    consumer: ConsumerProtocol,
    producer: ProducerProtocol,
) -> None:
    _call_optional_transaction_hook(producer, "begin_transaction")
    _call_optional_transaction_hook(consumer, "begin_transaction")


def _commit_transactions(
    consumer: ConsumerProtocol,
    producer: ProducerProtocol,
) -> None:
    _call_optional_transaction_hook(consumer, "commit_transaction")
    _call_optional_transaction_hook(producer, "commit_transaction")


def _abort_transactions(
    consumer: ConsumerProtocol,
    producer: ProducerProtocol,
) -> None:
    for participant in (producer, consumer):
        try:
            _call_optional_transaction_hook(participant, "abort_transaction")
        except Exception as exc:  # pragma: no cover - defensive logging branch
            logger.warning(
                "stream_transaction_abort_failed",
                extra={"error_type": type(exc).__name__},
            )


def _call_optional_transaction_hook(participant: Any, name: str) -> Any:
    hook = getattr(participant, name, None)
    if callable(hook):
        return hook()
    return None


def _seek_if_supported(consumer: ConsumerProtocol, position: SourcePosition) -> None:
    seek = getattr(consumer, "seek", None)
    if callable(seek):
        seek(position)


def _log_checkpoint_saved(checkpoint: CheckpointRecord) -> None:
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "stream_checkpoint_saved",
            extra={
                "source_topic": checkpoint.source.topic,
                "source_partition": checkpoint.source.partition,
                "source_offset": checkpoint.source.offset,
                "output_topic": checkpoint.redacted_output.topic,
                "output_partition": checkpoint.redacted_output.partition,
                "output_offset": checkpoint.redacted_output.offset,
                "policy_fingerprint": checkpoint.policy_fingerprint,
                "model_fingerprint": checkpoint.model_fingerprint,
            },
        )


def _raise_for_message_error(message: Any) -> None:
    error = getattr(message, "error", None)
    if not callable(error):
        return
    if error() is not None:
        raise KafkaMessageError("Kafka consumer returned an errored message")


def _decode_record(message: Any) -> dict[str, Any]:
    value = _message_value(message)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Kafka message value must be a JSON object") from exc
    if not isinstance(value, Mapping):
        raise TypeError("Kafka message value must be a mapping or JSON object")
    return dict(value)


def _message_value(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message

    value = getattr(message, "value", None)
    if callable(value):
        return value()
    if value is not None:
        return value
    raise TypeError("Kafka message must be a mapping or expose a value")


def _deidentify_record(
    record: Mapping[str, Any],
    *,
    text_field: str,
    policy_name: str,
    provenance_field: str,
    deidentify_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if text_field not in record:
        raise KeyError(f"record is missing text field {text_field!r}")
    text = record[text_field]
    if not isinstance(text, str):
        raise TypeError(f"record field {text_field!r} must contain text")

    from openmed import deidentify

    result = deidentify(text, policy=policy_name, **deidentify_kwargs)
    output = dict(record)
    output[text_field] = result.deidentified_text
    output[provenance_field] = {
        "policy": policy_name,
        "openmed_version": __version__,
    }
    return output
