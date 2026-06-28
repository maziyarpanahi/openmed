from __future__ import annotations

import builtins
import importlib
import sys
from datetime import datetime
from types import ModuleType
from typing import Any

import pytest

from openmed.__about__ import __version__
from openmed.processing.kafka_connector import (
    KafkaConnectorError,
    create_confluent_kafka_clients,
    deidentify_stream,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult


class FakeConsumer:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        events: list[tuple[str, Any]] | None = None,
    ) -> None:
        self.messages = list(messages)
        self.events = events if events is not None else []
        self.committed: list[dict[str, Any]] = []
        self.subscriptions: list[list[str]] = []

    def subscribe(self, topics: list[str]) -> None:
        self.subscriptions.append(topics)

    def poll(self, timeout: float | None = None) -> dict[str, Any] | None:
        self.events.append(("poll", timeout))
        if not self.messages:
            return None
        return self.messages.pop(0)

    def commit(self, message: dict[str, Any] | None = None) -> None:
        self.events.append(("commit", message))
        if message is not None:
            self.committed.append(message)


class FakeProducer:
    def __init__(
        self,
        events: list[tuple[str, Any]] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.events = events if events is not None else []
        self.fail = fail
        self.produced: list[tuple[str, dict[str, Any]]] = []

    def produce(self, topic: str, value: dict[str, Any]) -> None:
        self.events.append(("produce", topic))
        if self.fail:
            raise RuntimeError("produce failed")
        self.produced.append((topic, dict(value)))


def _patch_extract_pii(monkeypatch: pytest.MonkeyPatch) -> None:
    from openmed.core import pii

    def fake_extract_pii(text: str, *args: Any, **kwargs: Any) -> PredictionResult:
        surface = "Jane Roe"
        start = text.index(surface)
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text=surface,
                    label="NAME",
                    start=start,
                    end=start + len(surface),
                    confidence=0.99,
                )
            ],
            model_name="stub",
            timestamp=datetime.now().isoformat(),
        )

    monkeypatch.setattr(pii, "extract_pii", fake_extract_pii)


def test_deidentify_stream_produces_redacted_record_with_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extract_pii(monkeypatch)
    message = {
        "event_id": "evt-1",
        "note": "Patient Jane Roe checked in",
        "encounter": {"id": "enc-7"},
    }
    events: list[tuple[str, Any]] = []
    consumer = FakeConsumer([message], events)
    producer = FakeProducer(events)

    processed = deidentify_stream(
        consumer,
        producer,
        in_topic="raw-notes",
        out_topic="redacted-notes",
        text_field="note",
        policy="hipaa_safe_harbor",
        use_safety_sweep=False,
    )

    assert processed == 1
    assert consumer.subscriptions == [["raw-notes"]]
    assert consumer.committed == [message]
    assert len(producer.produced) == 1

    topic, output = producer.produced[0]
    assert topic == "redacted-notes"
    assert output["note"] == "Patient [NAME] checked in"
    assert output["event_id"] == "evt-1"
    assert output["encounter"] == {"id": "enc-7"}
    assert output["deid_provenance"] == {
        "policy": "hipaa_safe_harbor",
        "openmed_version": __version__,
    }


def test_deidentify_stream_commits_after_successful_produce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extract_pii(monkeypatch)
    message = {"text": "Patient Jane Roe arrived"}
    events: list[tuple[str, Any]] = []
    consumer = FakeConsumer([message], events)
    producer = FakeProducer(events)

    deidentify_stream(
        consumer,
        producer,
        in_topic="raw-notes",
        out_topic="redacted-notes",
        use_safety_sweep=False,
    )

    produce_index = next(
        index for index, event in enumerate(events) if event[0] == "produce"
    )
    commit_index = next(
        index for index, event in enumerate(events) if event[0] == "commit"
    )
    assert produce_index < commit_index
    assert consumer.committed == [message]


def test_deidentify_stream_does_not_commit_when_produce_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extract_pii(monkeypatch)
    message = {"text": "Patient Jane Roe arrived"}
    events: list[tuple[str, Any]] = []
    consumer = FakeConsumer([message], events)
    producer = FakeProducer(events, fail=True)

    with pytest.raises(RuntimeError, match="produce failed"):
        deidentify_stream(
            consumer,
            producer,
            in_topic="raw-notes",
            out_topic="redacted-notes",
            use_safety_sweep=False,
        )

    assert producer.produced == []
    assert consumer.committed == []
    assert all(event[0] != "commit" for event in events)


def test_importing_openmed_processing_does_not_import_kafka_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("confluent_kafka", None)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "confluent_kafka" or name.startswith("confluent_kafka."):
            raise AssertionError("confluent_kafka must be imported lazily")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import openmed
    import openmed.processing

    processing = importlib.reload(openmed.processing)
    package = importlib.reload(openmed)

    assert hasattr(package, "__version__")
    assert hasattr(processing, "deidentify_stream")


def test_confluent_factory_serializes_json_and_waits_for_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: dict[str, Any] = {}

    class FakeConfluentConsumer:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config
            self.subscriptions: list[list[str]] = []
            instances["consumer"] = self

        def subscribe(self, topics: list[str]) -> None:
            self.subscriptions.append(topics)

    class FakeConfluentProducer:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config
            self.produced: list[tuple[str, bytes]] = []
            self.flush_timeouts: list[float | None] = []
            instances["producer"] = self

        def produce(
            self,
            topic: str,
            *,
            value: bytes,
            on_delivery: Any,
        ) -> None:
            self.produced.append((topic, value))
            on_delivery(None, object())

        def flush(self, timeout: float | None = None) -> int:
            self.flush_timeouts.append(timeout)
            return 0

    module = ModuleType("confluent_kafka")
    module.Consumer = FakeConfluentConsumer
    module.Producer = FakeConfluentProducer
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)

    pair = create_confluent_kafka_clients(
        consumer_config={"group.id": "openmed"},
        producer_config={"bootstrap.servers": "localhost:9092"},
        in_topic="raw-notes",
        delivery_timeout=2.5,
    )

    pair.producer.produce("redacted-notes", {"text": "safe", "event_id": "1"})

    assert instances["consumer"].subscriptions == [["raw-notes"]]
    assert instances["producer"].produced == [
        ("redacted-notes", b'{"event_id":"1","text":"safe"}')
    ]
    assert instances["producer"].flush_timeouts == [2.5]


def test_confluent_factory_raises_on_delivery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfluentConsumer:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def subscribe(self, topics: list[str]) -> None:
            self.topics = topics

    class FakeConfluentProducer:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def produce(
            self,
            topic: str,
            *,
            value: bytes,
            on_delivery: Any,
        ) -> None:
            on_delivery(RuntimeError("broker rejected message"), object())

        def flush(self, timeout: float | None = None) -> int:
            return 0

    module = ModuleType("confluent_kafka")
    module.Consumer = FakeConfluentConsumer
    module.Producer = FakeConfluentProducer
    monkeypatch.setitem(sys.modules, "confluent_kafka", module)

    pair = create_confluent_kafka_clients(
        consumer_config={"group.id": "openmed"},
        producer_config={"bootstrap.servers": "localhost:9092"},
        in_topic="raw-notes",
    )

    with pytest.raises(KafkaConnectorError, match="broker rejected message"):
        pair.producer.produce("redacted-notes", {"text": "safe"})
