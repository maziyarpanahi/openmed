"""Kafka streaming connector for OpenMed de-identification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from openmed.__about__ import __version__
from openmed.core.policy import PolicyName, canonical_policy_name

DEFAULT_POLL_TIMEOUT = 1.0
DEFAULT_POLICY = "hipaa_safe_harbor"
PROVENANCE_FIELD = "deid_provenance"


class ConsumerProtocol(Protocol):
    """Minimal consumer surface required by :func:`deidentify_stream`."""

    def poll(self, timeout: float | None = None) -> Any | None:
        """Return the next message, or ``None`` when no message is available."""

    def commit(self, message: Any | None = None) -> Any:
        """Commit the consumed message offset after successful production."""


class ProducerProtocol(Protocol):
    """Minimal producer surface required by :func:`deidentify_stream`."""

    def produce(self, topic: str, value: Mapping[str, Any]) -> Any:
        """Produce a redacted record to ``topic``."""


@dataclass(frozen=True)
class KafkaClientPair:
    """Consumer/producer pair returned by concrete Kafka factory helpers."""

    consumer: ConsumerProtocol
    producer: ProducerProtocol


class KafkaConnectorError(RuntimeError):
    """Base exception for Kafka connector runtime failures."""


class KafkaMessageError(KafkaConnectorError):
    """Raised when a consumed Kafka message carries a client error."""


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
    **deidentify_kwargs: Any,
) -> int:
    """Consume records, de-identify their text field, and produce redacted output.

    Offsets are committed only after ``producer.produce`` returns successfully,
    giving the connector at-least-once behavior for the consumed messages. The
    consumer and producer are protocol-based so tests and non-confluent clients
    can provide small adapters.

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
        **deidentify_kwargs: Extra keyword arguments forwarded to
            :func:`openmed.deidentify`.

    Returns:
        Number of messages successfully produced and committed.
    """

    _validate_stream_args(
        in_topic=in_topic,
        out_topic=out_topic,
        text_field=text_field,
        max_messages=max_messages,
        idle_polls=idle_polls,
        provenance_field=provenance_field,
    )
    policy_name = canonical_policy_name(policy)
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
        record = _decode_record(message)
        redacted_record = _deidentify_record(
            record,
            text_field=text_field,
            policy_name=policy_name,
            provenance_field=provenance_field,
            deidentify_kwargs=deidentify_kwargs,
        )

        producer.produce(out_topic, value=redacted_record)
        consumer.commit(message)
        processed += 1

    return processed


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

    consumer = _ConfluentJsonConsumer(Consumer(dict(consumer_config)))
    producer = _ConfluentJsonProducer(
        Producer(dict(producer_config)),
        delivery_timeout=delivery_timeout,
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
    def __init__(
        self,
        producer: Any,
        *,
        delivery_timeout: float | None = 30.0,
    ) -> None:
        self._producer = producer
        self._delivery_timeout = delivery_timeout

    def produce(self, topic: str, value: Mapping[str, Any]) -> Any:
        payload = json.dumps(
            dict(value),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        delivery_errors: list[Any] = []

        def on_delivery(error: Any, _message: Any) -> None:
            if error is not None:
                delivery_errors.append(error)

        result = self._producer.produce(
            topic,
            value=payload,
            on_delivery=on_delivery,
        )
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
        return result


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
