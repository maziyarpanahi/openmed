from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from openmed.__about__ import __version__
from openmed.processing.checkpoint import (
    DEDUPE_HEADER,
    InMemoryCheckpointStore,
    LocalFileCheckpointStore,
    OutputPosition,
    SourcePosition,
    build_stream_fingerprint,
    checkpoint_for_delivery,
    dedupe_key_for_source,
)
from openmed.processing.kafka_connector import (
    CheckpointFingerprintError,
    KafkaConnectorError,
    create_confluent_kafka_clients,
    deidentify_stream,
    replay,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.processing.pulsar_connector import create_pulsar_clients


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


class FakeMessage:
    def __init__(
        self,
        value: dict[str, Any],
        *,
        topic: str = "raw-notes",
        partition: int = 0,
        offset: int = 0,
    ) -> None:
        self._value = value
        self._topic = topic
        self._partition = partition
        self._offset = offset

    def value(self) -> dict[str, Any]:
        return self._value

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset


class ReplayableConsumer:
    def __init__(
        self,
        messages: list[FakeMessage],
        events: list[tuple[str, Any]] | None = None,
        *,
        fail_commit_offsets: set[int] | None = None,
    ) -> None:
        self.messages = list(messages)
        self.events = events if events is not None else []
        self.fail_commit_offsets = set(fail_commit_offsets or set())
        self.index = 0
        self.committed: list[FakeMessage] = []
        self.subscriptions: list[list[str]] = []

    def subscribe(self, topics: list[str]) -> None:
        self.subscriptions.append(topics)

    def poll(self, timeout: float | None = None) -> FakeMessage | None:
        self.events.append(("poll", timeout))
        if self.index >= len(self.messages):
            return None
        message = self.messages[self.index]
        self.index += 1
        return message

    def commit(self, message: FakeMessage | None = None) -> None:
        offset = message.offset() if message is not None else None
        self.events.append(("commit", offset))
        if offset in self.fail_commit_offsets:
            raise RuntimeError("commit failed")
        if message is not None:
            self.committed.append(message)

    def seek(self, position: SourcePosition) -> None:
        for index, message in enumerate(self.messages):
            if (
                message.topic() == position.topic
                and str(message.partition()) == position.partition
                and message.offset() >= position.offset
            ):
                self.index = index
                self.events.append(("seek", position.offset))
                return
        self.index = len(self.messages)

    def begin_transaction(self) -> None:
        self.events.append(("consumer_begin_transaction", None))

    def commit_transaction(self) -> None:
        self.events.append(("consumer_commit_transaction", None))

    def abort_transaction(self) -> None:
        self.events.append(("consumer_abort_transaction", None))


class IdempotentProducer:
    supports_idempotent_produce = True

    def __init__(
        self,
        events: list[tuple[str, Any]] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.events = events if events is not None else []
        self.fail = fail
        self.produced: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self._dedupe_keys: set[str] = set()

    def begin_transaction(self) -> None:
        self.events.append(("producer_begin_transaction", None))

    def commit_transaction(self) -> None:
        self.events.append(("producer_commit_transaction", None))

    def abort_transaction(self) -> None:
        self.events.append(("producer_abort_transaction", None))

    def produce(
        self,
        topic: str,
        value: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        resolved_headers = dict(headers or {})
        self.events.append(("produce", resolved_headers))
        if self.fail:
            raise RuntimeError("produce failed")
        dedupe_key = resolved_headers.get(DEDUPE_HEADER)
        if dedupe_key is not None and dedupe_key in self._dedupe_keys:
            return {
                "topic": topic,
                "partition": "0",
                "offset": max(len(self.produced) - 1, 0),
            }
        if dedupe_key is not None:
            self._dedupe_keys.add(dedupe_key)
        self.produced.append((topic, dict(value), resolved_headers))
        return {
            "topic": topic,
            "partition": "0",
            "offset": len(self.produced) - 1,
        }


class FakeKafkaConsumer(ReplayableConsumer):
    pass


class FakeKafkaProducer(IdempotentProducer):
    pass


class FakePulsarConsumer(ReplayableConsumer):
    pass


class FakePulsarProducer(IdempotentProducer):
    pass


def _stream_messages(count: int = 3) -> list[FakeMessage]:
    return [
        FakeMessage(
            {"event_id": f"evt-{index}", "note": f"Patient Jane Roe visit {index}"},
            offset=index,
        )
        for index in range(count)
    ]


def _produced_dedupe_keys(producer: IdempotentProducer) -> list[str]:
    return [headers[DEDUPE_HEADER] for _, _, headers in producer.produced]


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


def test_checkpoint_resume_after_crash_between_produce_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extract_pii(monkeypatch)
    messages = _stream_messages(3)
    events: list[tuple[str, Any]] = []
    store = InMemoryCheckpointStore()
    producer = IdempotentProducer(events)
    first_consumer = ReplayableConsumer(
        messages,
        events,
        fail_commit_offsets={0},
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        deidentify_stream(
            first_consumer,
            producer,
            in_topic="raw-notes",
            out_topic="redacted-notes",
            text_field="note",
            checkpoint_store=store,
            use_safety_sweep=False,
        )

    assert len(producer.produced) == 1
    assert first_consumer.committed == []
    assert store.load("raw-notes", "0") is not None

    restart_consumer = ReplayableConsumer(messages, events)
    processed = deidentify_stream(
        restart_consumer,
        producer,
        in_topic="raw-notes",
        out_topic="redacted-notes",
        text_field="note",
        checkpoint_store=store,
        use_safety_sweep=False,
    )

    expected_keys = {
        dedupe_key_for_source(SourcePosition("raw-notes", "0", offset))
        for offset in range(3)
    }
    assert processed == 3
    assert len(producer.produced) == 3
    assert set(_produced_dedupe_keys(producer)) == expected_keys
    assert len(_produced_dedupe_keys(producer)) == len(expected_keys)
    assert [message.offset() for message in restart_consumer.committed] == [0, 1, 2]
    assert ("producer_abort_transaction", None) in events


def test_checkpointed_dispatch_failure_aborts_without_commit_or_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extract_pii(monkeypatch)
    events: list[tuple[str, Any]] = []
    store = InMemoryCheckpointStore()
    consumer = ReplayableConsumer(_stream_messages(1), events)
    producer = IdempotentProducer(events, fail=True)

    with pytest.raises(RuntimeError, match="produce failed"):
        deidentify_stream(
            consumer,
            producer,
            in_topic="raw-notes",
            out_topic="redacted-notes",
            text_field="note",
            checkpoint_store=store,
            use_safety_sweep=False,
        )

    assert store.all() == {}
    assert consumer.committed == []
    assert producer.produced == []
    assert ("producer_abort_transaction", None) in events


def test_replay_is_deterministic_and_refuses_fingerprint_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_extract_pii(monkeypatch)
    fingerprint = build_stream_fingerprint(
        policy_name="hipaa_safe_harbor",
        deidentify_kwargs={"use_safety_sweep": False},
    )
    source = SourcePosition("raw-notes", "0", 0)
    checkpoint = checkpoint_for_delivery(
        source=source,
        redacted_output=OutputPosition("redacted-notes", "0", 0),
        fingerprint=fingerprint,
        dedupe_key=dedupe_key_for_source(source),
        created_at=1.0,
    )
    to_position = SourcePosition("raw-notes", "0", 1)

    first_producer = IdempotentProducer()
    second_producer = IdempotentProducer()
    first_count = replay(
        ReplayableConsumer(_stream_messages(3)),
        first_producer,
        from_checkpoint=checkpoint,
        to_position=to_position,
        out_topic="quarantine-notes",
        text_field="note",
        use_safety_sweep=False,
    )
    second_count = replay(
        ReplayableConsumer(_stream_messages(3)),
        second_producer,
        from_checkpoint=checkpoint,
        to_position=to_position,
        out_topic="quarantine-notes",
        text_field="note",
        use_safety_sweep=False,
    )

    def canonical(producer: IdempotentProducer) -> list[str]:
        return [
            json.dumps(
                {"topic": topic, "value": value, "headers": headers},
                sort_keys=True,
                separators=(",", ":"),
            )
            for topic, value, headers in producer.produced
        ]

    assert first_count == second_count == 2
    assert canonical(first_producer) == canonical(second_producer)

    bad_checkpoint = replace(checkpoint, model_fingerprint="different")
    with pytest.raises(CheckpointFingerprintError, match="model fingerprint"):
        replay(
            ReplayableConsumer(_stream_messages(1)),
            IdempotentProducer(),
            from_checkpoint=bad_checkpoint,
            to_position=source,
            out_topic="quarantine-notes",
            text_field="note",
            use_safety_sweep=False,
        )


def test_checkpoint_file_logs_and_dedupe_key_do_not_contain_phi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_extract_pii(monkeypatch)
    raw_phi = "Patient Jane Roe visit 0"
    checkpoint_path = tmp_path / "checkpoints.json"
    store = LocalFileCheckpointStore(checkpoint_path, fsync=False)
    producer = IdempotentProducer()
    consumer = ReplayableConsumer(
        [FakeMessage({"note": raw_phi, "event_id": "evt-0"}, offset=0)]
    )

    with caplog.at_level(logging.INFO, logger="openmed.processing.kafka_connector"):
        deidentify_stream(
            consumer,
            producer,
            in_topic="raw-notes",
            out_topic="redacted-notes",
            text_field="note",
            checkpoint_store=store,
            use_safety_sweep=False,
        )

    checkpoint_text = checkpoint_path.read_text(encoding="utf-8")
    dedupe_key = producer.produced[0][2][DEDUPE_HEADER]
    for leak in (raw_phi, "Jane Roe"):
        assert leak not in checkpoint_text
        assert leak not in caplog.text
        assert leak not in dedupe_key
    assert dedupe_key.startswith("sha256:")
    assert store.load("raw-notes", "0") is not None


@pytest.mark.parametrize(
    ("consumer_cls", "producer_cls"),
    [
        (FakeKafkaConsumer, FakeKafkaProducer),
        (FakePulsarConsumer, FakePulsarProducer),
    ],
)
def test_kafka_and_pulsar_protocol_conformance(
    monkeypatch: pytest.MonkeyPatch,
    consumer_cls: type[ReplayableConsumer],
    producer_cls: type[IdempotentProducer],
) -> None:
    _patch_extract_pii(monkeypatch)
    store = InMemoryCheckpointStore()
    consumer = consumer_cls(_stream_messages(2))
    producer = producer_cls()

    processed = deidentify_stream(
        consumer,
        producer,
        in_topic="raw-notes",
        out_topic="redacted-notes",
        text_field="note",
        checkpoint_store=store,
        use_safety_sweep=False,
    )
    checkpoint = store.load("raw-notes", "0")

    assert processed == 2
    assert checkpoint is not None
    assert checkpoint.source.offset == 1
    assert len(producer.produced) == 2
    assert all(DEDUPE_HEADER in headers for _, _, headers in producer.produced)


def test_checkpointed_stream_throughput_overhead_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openmed

    def fake_deidentify(text: str, *args: Any, **kwargs: Any) -> Any:
        payload = text.encode("utf-8")
        digest = b""
        # Keep the synthetic pipeline cost high enough that CI timer noise does
        # not dominate the checkpoint bookkeeping overhead measurement.
        for _ in range(300):
            digest = hashlib.sha256(payload + digest).digest()
        return SimpleNamespace(deidentified_text=text.replace("Jane Roe", "[NAME]"))

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    count = 2400

    def measure(checkpointed: bool) -> float:
        start = time.perf_counter()
        deidentify_stream(
            ReplayableConsumer(_stream_messages(count)),
            IdempotentProducer(),
            in_topic="raw-notes",
            out_topic="redacted-notes",
            text_field="note",
            checkpoint_store=InMemoryCheckpointStore() if checkpointed else None,
            max_messages=count,
            use_safety_sweep=False,
        )
        return time.perf_counter() - start

    baseline_elapsed = min(measure(False) for _ in range(2))
    checkpoint_elapsed = min(measure(True) for _ in range(2))
    checkpoint_rate = count / checkpoint_elapsed
    overhead = (checkpoint_elapsed - baseline_elapsed) / baseline_elapsed

    assert checkpoint_rate >= 2_000
    assert overhead <= 0.10


def test_importing_openmed_processing_does_not_import_kafka_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("confluent_kafka", None)
    sys.modules.pop("pulsar", None)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "confluent_kafka" or name.startswith("confluent_kafka."):
            raise AssertionError("confluent_kafka must be imported lazily")
        if name == "pulsar" or name.startswith("pulsar."):
            raise AssertionError("pulsar must be imported lazily")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    import openmed
    import openmed.processing

    processing = importlib.reload(openmed.processing)
    package = importlib.reload(openmed)

    assert hasattr(package, "__version__")
    assert hasattr(processing, "deidentify_stream")
    assert hasattr(processing, "create_pulsar_clients")


def test_pulsar_factory_serializes_json_and_uses_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: dict[str, Any] = {}

    class FakePulsarTimeout(Exception):
        pass

    class FakePulsarConsumerClient:
        def __init__(self) -> None:
            self.acknowledged: list[Any] = []

        def receive(self, timeout_millis: int | None = None) -> Any:
            raise FakePulsarTimeout()

        def acknowledge(self, message: Any) -> None:
            self.acknowledged.append(message)

    class FakePulsarProducerClient:
        def __init__(self) -> None:
            self.sent: list[tuple[bytes, dict[str, str]]] = []

        def send(
            self,
            payload: bytes,
            *,
            properties: dict[str, str],
        ) -> str:
            self.sent.append((payload, properties))
            return "ledger:entry"

    class FakePulsarClient:
        def __init__(self, service_url: str, **config: Any) -> None:
            self.service_url = service_url
            self.config = config
            self.consumer = FakePulsarConsumerClient()
            self.producer = FakePulsarProducerClient()
            instances["client"] = self

        def subscribe(
            self,
            topic: str,
            subscription_name: str,
            **config: Any,
        ) -> FakePulsarConsumerClient:
            self.subscription = (topic, subscription_name, config)
            return self.consumer

        def create_producer(
            self,
            topic: str,
            **config: Any,
        ) -> FakePulsarProducerClient:
            self.producer_topic = (topic, config)
            return self.producer

    module = ModuleType("pulsar")
    module.Client = FakePulsarClient
    module.Timeout = FakePulsarTimeout
    monkeypatch.setitem(sys.modules, "pulsar", module)

    pair = create_pulsar_clients(
        service_url="pulsar://localhost:6650",
        in_topic="raw-notes",
        subscription_name="openmed",
        client_config={"operation_timeout_seconds": 2},
        consumer_config={"consumer_name": "deid"},
        producer_config={"producer_name": "redacted"},
    )
    result = pair.producer.produce(
        "redacted-notes",
        {"event_id": "1", "text": "safe"},
        headers={DEDUPE_HEADER: "sha256:test"},
    )

    client = instances["client"]
    assert client.service_url == "pulsar://localhost:6650"
    assert client.subscription == (
        "raw-notes",
        "openmed",
        {"consumer_name": "deid"},
    )
    assert client.producer_topic == (
        "redacted-notes",
        {"producer_name": "redacted"},
    )
    assert client.producer.sent == [
        (
            b'{"event_id":"1","text":"safe"}',
            {DEDUPE_HEADER: "sha256:test"},
        )
    ]
    assert result == {
        "topic": "redacted-notes",
        "partition": "0",
        "offset": "ledger:entry",
    }


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
