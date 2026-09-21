"""Resource ownership when Pulsar adapter construction fails."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openmed.processing import pulsar_connector


@pytest.fixture
def clients(monkeypatch):
    client = Mock(spec=["subscribe", "create_producer", "close"])
    client.close.return_value = None
    factory = Mock(return_value=client)
    monkeypatch.setitem(
        sys.modules, "pulsar", SimpleNamespace(Client=factory, Timeout=TimeoutError)
    )
    return factory, client


def create(**kwargs):
    options = {
        "service_url": "pulsar://synthetic.invalid:6650",
        "in_topic": "synthetic-in",
        "subscription_name": "synthetic-subscription",
    }
    options.update(kwargs)
    return pulsar_connector.create_pulsar_clients(**options)


@pytest.mark.parametrize(
    "error_type", [OSError, ValueError, KeyboardInterrupt, asyncio.CancelledError]
)
def test_subscription_failure_closes_client_and_propagates(clients, error_type):
    _, client = clients
    failure = error_type("synthetic subscription failure")
    client.subscribe.side_effect = failure
    with pytest.raises(error_type) as caught:
        create()
    assert caught.value is failure
    client.close.assert_called_once_with()


def test_close_failure_does_not_replace_subscription_error(clients):
    _, client = clients
    failure = ValueError("synthetic subscription failure")
    client.subscribe.side_effect = failure
    client.close.side_effect = RuntimeError("synthetic close failure")
    with pytest.raises(ValueError) as caught:
        create()
    assert caught.value is failure
    client.close.assert_called_once_with()


@pytest.mark.parametrize("adapter", ["_PulsarJsonConsumer", "_PulsarJsonProducer"])
def test_adapter_failure_also_closes_client(monkeypatch, clients, adapter):
    _, client = clients
    failure = RuntimeError("synthetic adapter failure")
    monkeypatch.setattr(pulsar_connector, adapter, Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        create()
    assert caught.value is failure
    client.close.assert_called_once_with()


@pytest.mark.parametrize("config_name", ["consumer_config", "producer_config"])
def test_configuration_copy_failure_closes_created_client(clients, config_name):
    _, client = clients
    failure = ValueError("synthetic configuration failure")

    class BrokenMapping(Mapping):
        def __len__(self):
            return 1

        def __iter__(self):
            raise failure

        def __getitem__(self, key):
            raise KeyError(key)

    with pytest.raises(ValueError) as caught:
        create(**{config_name: BrokenMapping()})
    assert caught.value is failure
    client.close.assert_called_once_with()


def test_failed_client_constructor_is_not_closed(clients):
    factory, client = clients
    failure = RuntimeError("synthetic client construction failure")
    factory.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        create()
    assert caught.value is failure
    client.close.assert_not_called()


@pytest.mark.parametrize("field", ["service_url", "in_topic", "subscription_name"])
def test_invalid_required_field_does_not_create_client(clients, field):
    factory, client = clients
    with pytest.raises(ValueError):
        create(**{field: " "})
    factory.assert_not_called()
    client.close.assert_not_called()


def test_success_transfers_ownership_and_preserves_configuration(clients):
    factory, client = clients
    client_config = {"io_threads": 1}
    consumer_config = {"receiver_queue_size": 10}
    producer_config = {"batching_enabled": False}
    pair = create(
        client_config=client_config,
        consumer_config=consumer_config,
        producer_config=producer_config,
    )
    assert pair.client is client
    factory.assert_called_once_with("pulsar://synthetic.invalid:6650", io_threads=1)
    client.subscribe.assert_called_once_with(
        "synthetic-in", "synthetic-subscription", receiver_queue_size=10
    )
    client.close.assert_not_called()
    client.create_producer.assert_not_called()
    client.create_producer.return_value.send.return_value = "synthetic-id"
    result = pair.producer.produce("synthetic-out", {"redacted": True})
    assert result == {
        "topic": "synthetic-out",
        "partition": "0",
        "offset": "synthetic-id",
    }
    client.create_producer.assert_called_once_with(
        "synthetic-out", batching_enabled=False
    )
    assert client_config == {"io_threads": 1}
    assert consumer_config == {"receiver_queue_size": 10}
    assert producer_config == {"batching_enabled": False}
    pair.client.close()
    client.close.assert_called_once_with()
