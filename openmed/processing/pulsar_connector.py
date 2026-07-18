"""Pulsar streaming connector adapters for OpenMed de-identification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from openmed.processing.kafka_connector import (
    DEFAULT_POLL_TIMEOUT,
    ConsumerProtocol,
    KafkaConnectorError,
    ProducerProtocol,
)


@dataclass(frozen=True)
class PulsarClientPair:
    """Consumer/producer pair returned by concrete Pulsar factory helpers."""

    consumer: ConsumerProtocol
    producer: ProducerProtocol
    client: Any


def create_pulsar_clients(
    *,
    service_url: str,
    in_topic: str,
    subscription_name: str,
    client_config: Mapping[str, Any] | None = None,
    consumer_config: Mapping[str, Any] | None = None,
    producer_config: Mapping[str, Any] | None = None,
) -> PulsarClientPair:
    """Create JSON record adapters backed by ``pulsar-client``.

    ``pulsar`` is imported only inside this helper so importing
    ``openmed`` or ``openmed.processing`` does not require a Pulsar client.
    """

    _validate_non_empty(service_url, "service_url")
    _validate_non_empty(in_topic, "in_topic")
    _validate_non_empty(subscription_name, "subscription_name")
    try:
        import pulsar
    except ImportError as exc:
        raise ImportError(
            "Pulsar support requires the optional pulsar-client package. "
            "Install with `pip install pulsar-client`."
        ) from exc

    client = pulsar.Client(service_url, **dict(client_config or {}))
    consumer = client.subscribe(
        in_topic,
        subscription_name,
        **dict(consumer_config or {}),
    )
    return PulsarClientPair(
        consumer=_PulsarJsonConsumer(
            consumer,
            timeout_error=getattr(pulsar, "Timeout", TimeoutError),
        ),
        producer=_PulsarJsonProducer(
            client, producer_config=dict(producer_config or {})
        ),
        client=client,
    )


class _PulsarJsonConsumer:
    def __init__(self, consumer: Any, *, timeout_error: type[BaseException]) -> None:
        self._consumer = consumer
        self._timeout_error = timeout_error

    def poll(self, timeout: float | None = DEFAULT_POLL_TIMEOUT) -> Any | None:
        try:
            if timeout is None:
                return self._consumer.receive()
            return self._consumer.receive(timeout_millis=max(0, int(timeout * 1000)))
        except self._timeout_error:
            return None

    def commit(self, message: Any | None = None) -> Any:
        if message is None:
            return None
        return self._consumer.acknowledge(message)


class _PulsarJsonProducer:
    supports_idempotent_produce = True

    def __init__(
        self,
        client: Any,
        *,
        producer_config: Mapping[str, Any],
    ) -> None:
        self._client = client
        self._producer_config = dict(producer_config)
        self._producers: dict[str, Any] = {}

    def begin_transaction(self) -> None:
        return None

    def commit_transaction(self) -> None:
        return None

    def abort_transaction(self) -> None:
        return None

    def produce(
        self,
        topic: str,
        value: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        _validate_non_empty(topic, "topic")
        payload = json.dumps(
            dict(value),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        producer = self._producer_for(topic)
        try:
            message_id = producer.send(payload, properties=dict(headers or {}))
        except Exception as exc:  # pragma: no cover - exercised by real client users
            raise KafkaConnectorError("Pulsar producer delivery failed") from exc
        return {
            "topic": topic,
            "partition": "0",
            "offset": str(message_id),
        }

    def _producer_for(self, topic: str) -> Any:
        producer = self._producers.get(topic)
        if producer is None:
            producer = self._client.create_producer(topic, **self._producer_config)
            self._producers[topic] = producer
        return producer


def _validate_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


__all__ = [
    "PulsarClientPair",
    "create_pulsar_clients",
]
