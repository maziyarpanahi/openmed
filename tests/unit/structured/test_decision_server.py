"""Loopback decision-server backend against in-process fake servers.

Every server here is an in-process ``http.server`` bound to loopback or a Unix
socket; no test reaches another host. Fixture text is synthetic.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import socket
import socketserver
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from openmed.structured.decision import (
    DecisionBackend,
    DecisionBackendIdentity,
    DecisionBackendKind,
    DecisionError,
    DecisionMode,
    DecisionQuestion,
    DecisionQuestionSetBackend,
    DecisionRequest,
    DecisionState,
    DecisionUnsupportedError,
    decide,
)
from openmed.structured.decision_interop import (
    to_openai_decisions_request,
    to_system_one_request,
)
from openmed.structured.decision_server import (
    DecisionServerBackend,
    DecisionServerConflictError,
    DecisionServerEndpointError,
    DecisionServerError,
    DecisionServerScope,
    DecisionServerTimeoutError,
    DecisionServerUnsupportedError,
)

MODEL = "synthetic-org/decider-4b"
REVISION = "0123456789abcdef0123456789abcdef01234567"
INPUT_TEXT = "Synthetic ticket ZX-4471: checkout fails for every synthetic customer."
INSTRUCTIONS = "Which synthetic queue should review ticket ZX-4471?"
OPTION_TEXT = "synthetic-payments-queue"
QUERY_TOKEN = "synthetic-query-token-5521"
BEARER_TOKEN = "synthetic-bearer-key-0042"
PUBLIC_ADDRESS = "93.184.216.34"
LEVELS = ("Low", "Medium", "High")

SYSTEM_ONE_RESPONSE: dict[str, Any] = {
    "model": MODEL,
    "answers": {
        "queue": {
            "type": "choice",
            "choice": OPTION_TEXT,
            "confidence": 0.81,
            "probabilities": {OPTION_TEXT: 0.9, "web": 0.05, "support": 0.05},
        },
        "outage": {"type": "noul", "noul": 0.9},
        "urgency": {
            "type": "score",
            "legend": {"0": "Low", "1": "Medium", "2": "High"},
            "score": 1.8,
            "confidence": 0.85,
            "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85},
        },
    },
    "probability_source": "native",
    "effort_used": {"passes_per_decision": 1, "tier": "fast"},
    "usage": {"input_tokens": 231, "output_tokens": 3, "latency_ms": 74.9},
}
OPENAI_RESPONSE: dict[str, Any] = {
    "model": MODEL,
    "answers": [
        {
            "type": "choice",
            "name": "queue",
            "choice": OPTION_TEXT,
            "confidence": 0.81,
            "probabilities": [
                {"value": OPTION_TEXT, "probability": 0.9},
                {"value": "web", "probability": 0.05},
                {"value": "support", "probability": 0.05},
            ],
        },
        {"type": "predicate", "name": "outage", "probability": 0.9},
        {
            "type": "score",
            "name": "urgency",
            "score": 1.8,
            "confidence": 0.85,
            "probabilities": [
                {"label": "Low", "value": 0, "probability": 0.05},
                {"label": "Medium", "value": 1, "probability": 0.1},
                {"label": "High", "value": 2, "probability": 0.85},
            ],
        },
    ],
    "usage": {"input_tokens": 231, "output_tokens": 3, "total_tokens": 234},
}


# ---------------------------------------------------------------------------
# In-process fake decision servers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Received:
    """One request as the fake server saw it."""

    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def document(self) -> Any:
        return json.loads(self.body)


Action = Callable[[BaseHTTPRequestHandler], None]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        owner: FakeServer = self.server.owner  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        owner.received.append(
            Received(
                self.path,
                {key.lower(): value for key, value in self.headers.items()},
                body,
            )
        )
        owner.next_action()(self)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None


class _TCPServer(ThreadingHTTPServer):
    daemon_threads = True


class _IPv6TCPServer(_TCPServer):
    address_family = socket.AF_INET6


if hasattr(socket, "AF_UNIX"):

    class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True


class FakeServer:
    """Programmable decision server; actions are consumed in order."""

    def __init__(self, server: socketserver.BaseServer, origin: str) -> None:
        self.server = server
        self.origin = origin
        self.received: list[Received] = []
        self.actions: list[Action] = []
        self.default: Action = reply_json(SYSTEM_ONE_RESPONSE)
        server.owner = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self._thread.start()

    def queue(self, *actions: Action) -> None:
        self.actions.extend(actions)

    def next_action(self) -> Action:
        return self.actions.pop(0) if self.actions else self.default

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def reply_raw(
    body: bytes,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    content_length: bool = True,
) -> Action:
    def act(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(status)
        for key, value in (headers or {"Content-Type": "application/json"}).items():
            handler.send_header(key, value)
        if content_length:
            handler.send_header("Content-Length", str(len(body)))
        else:
            handler.close_connection = True
        handler.end_headers()
        handler.wfile.write(body)

    return act


def reply_json(document: Any, *, status: int = 200) -> Action:
    return reply_raw(json.dumps(document).encode("utf-8"), status=status)


def redirect(location: str, *, status: int = 307) -> Action:
    return reply_raw(b"", status=status, headers={"Location": location})


def drop() -> Action:
    def act(handler: BaseHTTPRequestHandler) -> None:
        handler.close_connection = True

    return act


def sleep_then(seconds: float, then: Action) -> Action:
    def act(handler: BaseHTTPRequestHandler) -> None:
        time.sleep(seconds)
        try:
            then(handler)
        except OSError:
            pass

    return act


def drip(interval: float, count: int) -> Action:
    """Send a status line, then one header byte per interval."""

    def act(handler: BaseHTTPRequestHandler) -> None:
        try:
            handler.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
            handler.wfile.flush()
            for _ in range(count):
                time.sleep(interval)
                handler.wfile.write(b"a")
                handler.wfile.flush()
        except OSError:
            pass
        handler.close_connection = True

    return act


@pytest.fixture
def start_server() -> Iterator[Callable[..., FakeServer]]:
    servers: list[FakeServer] = []

    def start(*, ipv6: bool = False) -> FakeServer:
        if ipv6:
            server: socketserver.BaseServer = _IPv6TCPServer(("::1", 0), _Handler)
            port = server.server_address[1]
            fake = FakeServer(server, f"http://[::1]:{port}")
        else:
            server = _TCPServer(("127.0.0.1", 0), _Handler)
            port = server.server_address[1]
            fake = FakeServer(server, f"http://127.0.0.1:{port}")
        servers.append(fake)
        return fake

    yield start
    for fake in servers:
        fake.close()


@pytest.fixture
def unix_server() -> Iterator[tuple[FakeServer, str]]:
    if not hasattr(socket, "AF_UNIX") or sys.platform == "win32":
        pytest.skip("Unix domain sockets are unavailable on this platform")
    directory = tempfile.mkdtemp(prefix="omds")
    path = os.path.join(directory, "decider.sock")
    fake = FakeServer(_UnixServer(path, _Handler), "unix")
    yield fake, path
    fake.close()
    if os.path.exists(path):
        os.unlink(path)
    os.rmdir(directory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _question_set(*, timeout_ms: int = 5000) -> DecisionRequest:
    return DecisionRequest(
        mode=DecisionMode.QUESTION_SET,
        input_text=INPUT_TEXT,
        timeout_ms=timeout_ms,
        questions=(
            DecisionQuestion(
                question_id="queue",
                mode=DecisionMode.FIXED_CHOICE,
                options=(OPTION_TEXT, "web", "support"),
                option_descriptions=("Synthetic payments", None, "Support"),
                instructions=INSTRUCTIONS,
            ),
            DecisionQuestion(
                question_id="outage",
                mode=DecisionMode.BOOLEAN_CHOICE,
                options=("true", "false"),
                instructions="Is this a synthetic full outage?",
            ),
            DecisionQuestion(
                question_id="urgency",
                mode=DecisionMode.SCALAR_SCORE,
                levels=LEVELS,
                instructions="How urgent is the synthetic ticket?",
            ),
        ),
    )


def _backend(endpoint: str, **overrides: Any) -> DecisionServerBackend:
    options: dict[str, Any] = {
        "endpoint": endpoint,
        "model_id": MODEL,
        "revision": REVISION,
        "license_id": "Apache-2.0",
    }
    options.update(overrides)
    return DecisionServerBackend(**options)


def _states(result: Any) -> list[tuple[str, DecisionState, str | None]]:
    return [(item.question_id, item.state, item.code) for item in result.questions]


def _assert_no_answers(result: Any) -> None:
    assert result.choice is None
    assert all(item.choice is None for item in result.questions)
    assert all(item.option_scores == () for item in result.questions)
    assert all(item.expected_level is None for item in result.questions)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _system_one(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    response = copy.deepcopy(SYSTEM_ONE_RESPONSE)
    change(response)
    return response


# ---------------------------------------------------------------------------
# Choice, yes/no and score answers; multi-question requests
# ---------------------------------------------------------------------------


def test_question_set_is_sent_once_and_decided_per_question(start_server) -> None:
    server = start_server()
    backend = _backend(f"{server.origin}")
    request = _question_set()

    assert isinstance(backend, DecisionBackend)
    assert isinstance(backend, DecisionQuestionSetBackend)
    result = decide(request, backend=backend)

    assert result.state is DecisionState.SUCCESS
    queue, outage, urgency = result.questions
    assert queue.choice == OPTION_TEXT
    assert outage.choice == "true"
    assert urgency.expected_level == pytest.approx(1.8)
    assert [item["score"] for item in urgency.level_scores] == pytest.approx(
        [0.05, 0.1, 0.85]
    )
    assert len(server.received) == 1
    sent = server.received[0]
    assert sent.path == "/v1/systemone"
    assert sent.document == to_system_one_request(request)
    assert "model" not in sent.document
    assert sent.headers["content-type"] == "application/json"
    assert sent.headers["host"] == server.origin.removeprefix("http://")
    assert "authorization" not in sent.headers
    assert result.backend == {
        "backend_id": "openmed.decision_server.system_one",
        "kind": "specialist",
        "runtime": "system_one_server:loopback",
        "model_id": MODEL,
        "revision": REVISION,
        "license_id": "Apache-2.0",
        "local": True,
        "deterministic": False,
    }
    assert result.review["required"] is True


@pytest.mark.parametrize(
    ("request_fields", "answer", "expected_choice"),
    [
        (
            {"mode": "fixed_choice", "options": ["alpha", "beta"]},
            {
                "type": "choice",
                "choice": "beta",
                "probabilities": {"alpha": 0.1, "beta": 0.9},
            },
            "beta",
        ),
        (
            {"mode": "boolean_choice", "options": ["true", "false"]},
            {"type": "noul", "noul": 0.08},
            "false",
        ),
        (
            {"mode": "boolean_choice", "options": ["yes", "no"]},
            {"type": "choice", "probabilities": {"yes": 0.95, "no": 0.05}},
            "yes",
        ),
    ],
)
def test_single_question_requests_use_score(
    start_server,
    request_fields: dict[str, Any],
    answer: dict[str, Any],
    expected_choice: str,
) -> None:
    server = start_server()
    server.default = reply_json({"answers": {"decision": answer}})
    request = DecisionRequest.from_dict({"input_text": INPUT_TEXT, **request_fields})

    result = decide(request, backend=_backend(server.origin))

    assert result.state is DecisionState.SUCCESS
    assert result.choice == expected_choice
    assert list(server.received[0].document["questions"]) == ["decision"]


def test_single_score_question_reports_the_expected_level(start_server) -> None:
    server = start_server()
    server.default = reply_json(
        {"answers": {"triage": SYSTEM_ONE_RESPONSE["answers"]["urgency"]}}
    )
    request = DecisionRequest(
        mode=DecisionMode.SCALAR_SCORE,
        input_text=INPUT_TEXT,
        question_id="triage",
        levels=LEVELS,
    )

    output = _backend(server.origin).score(request)
    result = decide(request, backend=_backend(server.origin))

    assert output.option_scores == pytest.approx((0.05, 0.1, 0.85))
    assert result.state is DecisionState.SUCCESS
    assert result.expected_level == pytest.approx(1.8)


def test_openai_decisions_protocol_round_trips(start_server) -> None:
    server = start_server()
    server.default = reply_json(OPENAI_RESPONSE)
    request = _question_set()
    backend = _backend(
        server.origin,
        protocol="openai_decisions",
        served_model="decider-4b",
    )

    result = decide(request, backend=backend)

    assert result.state is DecisionState.SUCCESS
    assert [item.choice for item in result.questions[:2]] == [OPTION_TEXT, "true"]
    assert result.questions[2].expected_level == pytest.approx(1.8)
    assert result.backend["runtime"] == "openai_decisions_server:loopback"
    sent = server.received[0]
    assert sent.path == "/v1/decisions"
    assert sent.document == to_openai_decisions_request(request, model="decider-4b")


def test_served_model_is_sent_to_system_one_servers(start_server) -> None:
    server = start_server()

    decide(_question_set(), backend=_backend(server.origin, served_model=MODEL))

    assert server.received[0].document["model"] == MODEL


# ---------------------------------------------------------------------------
# Documented response variants
# ---------------------------------------------------------------------------


def _mica(response: dict[str, Any]) -> None:
    # Mica: no legend or score; ``level`` and ``expected_level`` instead.
    response["model"] = "decider-4b"
    response["usage"] = {"input_tokens": 212, "output_tokens": 0}
    del response["probability_source"], response["effort_used"]
    response["answers"]["urgency"] = {
        "type": "score",
        "level": 2,
        "expected_level": 1.8,
        "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85},
    }
    response["answers"]["outage"]["abstain"] = False


def _mica_label_keyed(response: dict[str, Any]) -> None:
    _mica(response)
    response["answers"]["urgency"]["probabilities"] = {
        "Low": 0.05,
        "Medium": 0.1,
        "High": 0.85,
    }
    response["answers"]["urgency"]["level"] = "High"


def _mica_listed(response: dict[str, Any]) -> None:
    _mica(response)
    response["answers"]["urgency"]["probabilities"] = [0.05, 0.1, 0.85]


def _elephant(response: dict[str, Any]) -> None:
    # elephant-1: choice without confidence, score without legend/confidence.
    del response["answers"]["queue"]["confidence"]
    response["answers"]["urgency"] = {
        "type": "score",
        "score": 1.8,
        "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85},
    }
    response.pop("model")


def _label_list_legend(response: dict[str, Any]) -> None:
    # pplx-decider and Clef document a legend without fixing its shape.
    response["answers"]["urgency"]["legend"] = list(LEVELS)


def _d3(response: dict[str, Any]) -> None:
    # vllm-sr d3: {"model", "answers", "usage"} with an index-keyed legend.
    response["usage"] = {"input_tokens": 120, "output_tokens": 0}
    del response["probability_source"], response["effort_used"]


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(lambda response: None, id="h2o-lightning-system-one"),
        pytest.param(_d3, id="vllm-sr-d3"),
        pytest.param(_mica, id="mica-index-keyed"),
        pytest.param(_mica_label_keyed, id="mica-label-keyed"),
        pytest.param(_mica_listed, id="mica-listed"),
        pytest.param(_elephant, id="elephant-1"),
        pytest.param(_label_list_legend, id="pplx-clef-list-legend"),
    ],
)
def test_documented_system_one_variants_decide_identically(
    start_server,
    variant: Callable[[dict[str, Any]], None],
) -> None:
    server = start_server()
    server.default = reply_json(_system_one(variant))

    result = decide(_question_set(), backend=_backend(server.origin))

    assert result.state is DecisionState.SUCCESS
    queue, outage, urgency = result.questions
    assert (queue.choice, outage.choice) == (OPTION_TEXT, "true")
    assert urgency.expected_level == pytest.approx(1.8)
    assert [item["score"] for item in urgency.level_scores] == pytest.approx(
        [0.05, 0.1, 0.85]
    )


def test_score_answers_with_only_an_expected_level_are_scalar(start_server) -> None:
    server = start_server()
    answer = {"type": "score", "level": 2, "expected_level": 1.5}
    server.queue(
        reply_json(_system_one(lambda r: r["answers"].update(urgency=answer))),
        reply_json(
            _system_one(
                lambda r: r["answers"].update(urgency={**answer, "confidence": 0.9})
            )
        ),
    )
    backend = _backend(server.origin)

    without_confidence = decide(_question_set(), backend=backend).questions[2]
    with_confidence = decide(_question_set(), backend=backend).questions[2]

    # Without a confidence the level is not trusted: abstain, never fabricate.
    assert without_confidence.state is DecisionState.ABSTAINED
    assert without_confidence.expected_level is None
    assert with_confidence.state is DecisionState.SUCCESS
    assert with_confidence.expected_level == pytest.approx(1.5)
    assert with_confidence.level_scores == ()


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


def test_a_server_slower_than_the_timeout_fails_without_an_answer(
    start_server,
) -> None:
    server = start_server()
    server.default = sleep_then(1.5, reply_json(SYSTEM_ONE_RESPONSE))
    started = time.monotonic()

    result = decide(_question_set(timeout_ms=250), backend=_backend(server.origin))

    assert time.monotonic() - started < 1.2
    assert result.state is DecisionState.FAILURE
    assert result.code == "backend_timeout"
    _assert_no_answers(result)


def test_a_trickling_server_is_cut_off_at_the_deadline(start_server) -> None:
    server = start_server()
    server.default = drip(0.05, 60)
    backend = _backend(server.origin)
    started = time.monotonic()

    with pytest.raises(DecisionServerTimeoutError):
        backend.score_questions(_question_set(timeout_ms=300))

    assert time.monotonic() - started < 1.5


# ---------------------------------------------------------------------------
# Malformed bodies and status codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(reply_raw(b"not json"), id="not-json"),
        pytest.param(reply_raw(b""), id="empty"),
        pytest.param(reply_raw(b"\xff\xfe\x00"), id="not-utf8"),
        pytest.param(reply_json([SYSTEM_ONE_RESPONSE]), id="array"),
        pytest.param(reply_raw(b'{"answers": NaN}'), id="nan-constant"),
        pytest.param(reply_raw(b'{"answers": {}, "answers": {}}'), id="duplicate-keys"),
        pytest.param(reply_json({"model": MODEL}), id="missing-answers"),
        pytest.param(
            reply_json(_system_one(lambda r: r["answers"].pop("urgency"))),
            id="missing-question",
        ),
        pytest.param(
            reply_json(_system_one(lambda r: r["answers"]["outage"].update(noul=1.5))),
            id="probability-out-of-range",
        ),
        pytest.param(
            reply_json(
                _system_one(lambda r: r["answers"]["queue"].update(choice="other"))
            ),
            id="undeclared-choice",
        ),
        pytest.param(
            reply_json(
                _system_one(
                    lambda r: r["answers"]["urgency"].update(legend=["Low", "High"])
                )
            ),
            id="legend-mismatch",
        ),
        pytest.param(
            reply_raw(b'{"answers": {', headers={"Content-Length": "4096"}),
            id="truncated",
        ),
    ],
)
def test_malformed_bodies_fail_without_a_fabricated_answer(
    start_server,
    action: Action,
) -> None:
    server = start_server()
    server.default = action

    result = decide(_question_set(), backend=_backend(server.origin))

    assert result.state is DecisionState.FAILURE
    assert result.code == "decision_server_response_invalid"
    assert {item.code for item in result.questions} == {
        "decision_server_response_invalid"
    }
    _assert_no_answers(result)


@pytest.mark.parametrize("content_length", [True, False])
def test_oversized_bodies_are_refused(start_server, content_length: bool) -> None:
    server = start_server()
    server.default = reply_raw(
        json.dumps(SYSTEM_ONE_RESPONSE).encode("utf-8"),
        content_length=content_length,
    )

    result = decide(
        _question_set(),
        backend=_backend(server.origin, max_response_bytes=128),
    )

    assert result.state is DecisionState.FAILURE
    assert result.code == "decision_server_response_too_large"
    _assert_no_answers(result)


@pytest.mark.parametrize(
    ("status", "state", "code"),
    [
        (400, DecisionState.UNSUPPORTED, "decision_server_input_rejected"),
        (413, DecisionState.UNSUPPORTED, "decision_server_input_rejected"),
        (422, DecisionState.UNSUPPORTED, "decision_server_input_rejected"),
        (404, DecisionState.UNSUPPORTED, "decision_server_route_unsupported"),
        (405, DecisionState.UNSUPPORTED, "decision_server_route_unsupported"),
        (401, DecisionState.FAILURE, "decision_server_unauthorized"),
        (403, DecisionState.FAILURE, "decision_server_unauthorized"),
        (429, DecisionState.FAILURE, "decision_server_unavailable"),
        (500, DecisionState.FAILURE, "decision_server_unavailable"),
        (502, DecisionState.FAILURE, "decision_server_unavailable"),
        (503, DecisionState.FAILURE, "decision_server_unavailable"),
        (504, DecisionState.FAILURE, "backend_timeout"),
        (201, DecisionState.FAILURE, "decision_server_response_invalid"),
        (302, DecisionState.FAILURE, "decision_server_redirect_refused"),
        (303, DecisionState.FAILURE, "decision_server_redirect_refused"),
    ],
)
def test_status_codes_map_to_typed_states(
    start_server,
    status: int,
    state: DecisionState,
    code: str,
) -> None:
    server = start_server()
    # Error bodies may echo the request; they are never read into results.
    server.default = reply_json(
        {"detail": f"cannot decide {INPUT_TEXT}"}, status=status
    )

    result = decide(_question_set(), backend=_backend(server.origin))

    assert result.state is state
    assert result.code == code
    assert len(server.received) == 1
    assert INPUT_TEXT not in json.dumps(result.to_dict())
    _assert_no_answers(result)


def test_modes_without_a_wire_shape_are_unsupported_before_any_call(
    start_server,
) -> None:
    server = start_server()
    backend = _backend(server.origin)
    multi_label = DecisionRequest(
        mode=DecisionMode.MULTI_LABEL,
        input_text=INPUT_TEXT,
        options=("alpha", "beta"),
    )
    mixed = DecisionRequest(
        mode=DecisionMode.QUESTION_SET,
        input_text=INPUT_TEXT,
        questions=(
            DecisionQuestion(
                question_id="order",
                mode=DecisionMode.ORDERED_PREFERENCE,
                options=("alpha", "beta"),
            ),
            DecisionQuestion(
                question_id="flag",
                mode=DecisionMode.BOOLEAN_CHOICE,
                options=("true", "false"),
            ),
        ),
    )

    for request in (multi_label, mixed):
        result = decide(request, backend=backend)
        assert result.state is DecisionState.UNSUPPORTED
        assert result.code == "backend_unsupported"
    with pytest.raises(DecisionUnsupportedError):
        backend.score(_question_set())
    assert server.received == []


# ---------------------------------------------------------------------------
# Redirects and retries
# ---------------------------------------------------------------------------


def test_same_origin_redirects_are_followed_with_the_same_body(start_server) -> None:
    server = start_server()
    server.queue(
        redirect("/v1/systemone"),
        redirect(f"{server.origin}/v1/systemone?step=2", status=308),
    )

    result = decide(
        _question_set(),
        backend=_backend(f"{server.origin}/v1/systemone/"),
    )

    assert result.state is DecisionState.SUCCESS
    assert [item.path for item in server.received] == [
        "/v1/systemone/",
        "/v1/systemone",
        "/v1/systemone?step=2",
    ]
    assert len({item.body for item in server.received}) == 1


def test_cross_origin_redirects_never_resend_the_body(start_server) -> None:
    server = start_server()
    other = start_server()
    port = server.origin.rsplit(":", 1)[1]
    targets = (
        f"{other.origin}/v1/systemone",
        f"http://localhost:{port}/v1/systemone",
        f"http://{PUBLIC_ADDRESS}/v1/systemone",
        f"https://127.0.0.1:{port}/v1/systemone",
        "http://user@127.0.0.1/v1/systemone",
    )

    for location in targets:
        server.queue(redirect(location))
        result = decide(
            _question_set(),
            backend=_backend(server.origin, resolver=lambda host, port: ["127.0.0.1"]),
        )
        assert result.state is DecisionState.FAILURE
        assert result.code == "decision_server_redirect_refused"
        _assert_no_answers(result)

    assert len(server.received) == len(targets)
    assert other.received == []


@pytest.mark.parametrize("max_redirects", [0, 2])
def test_redirect_hops_are_bounded(start_server, max_redirects: int) -> None:
    server = start_server()
    server.default = redirect("/v1/systemone")

    result = decide(
        _question_set(),
        backend=_backend(server.origin, max_redirects=max_redirects),
    )

    assert result.code == "decision_server_redirect_refused"
    assert len(server.received) == max_redirects + 1


def test_retries_reuse_the_approved_address_without_new_lookups(
    start_server,
) -> None:
    server = start_server()
    port = server.origin.rsplit(":", 1)[1]
    lookups: list[str] = []

    def resolver(host: str, port: int) -> list[str]:
        # Later answers would point elsewhere; a retry must not ask again.
        lookups.append(host)
        return ["127.0.0.1"] if len(lookups) <= 2 else [PUBLIC_ADDRESS]

    backend = _backend(f"http://localhost:{port}", resolver=resolver)
    server.queue(drop())

    result = decide(_question_set(), backend=backend)

    assert result.state is DecisionState.SUCCESS
    assert lookups == ["localhost", "localhost"]
    assert len(server.received) == 2
    assert server.received[0].body == server.received[1].body


def test_retries_are_bounded_and_end_in_failure(start_server) -> None:
    server = start_server()
    server.default = drop()

    no_retry = decide(_question_set(), backend=_backend(server.origin, max_retries=0))
    assert no_retry.code == "decision_server_unreachable"
    assert len(server.received) == 1

    retried = decide(_question_set(), backend=_backend(server.origin, max_retries=2))
    assert retried.state is DecisionState.FAILURE
    assert retried.code == "decision_server_unreachable"
    assert len(server.received) == 1 + 3
    _assert_no_answers(retried)


def test_a_closed_port_is_unreachable() -> None:
    backend = _backend(f"http://127.0.0.1:{_free_port()}")

    result = decide(_question_set(), backend=backend)

    assert result.state is DecisionState.FAILURE
    assert result.code == "decision_server_unreachable"


def test_https_endpoints_require_a_working_tls_handshake(start_server) -> None:
    server = start_server()
    port = server.origin.rsplit(":", 1)[1]

    result = decide(_question_set(), backend=_backend(f"https://127.0.0.1:{port}"))

    assert result.state is DecisionState.FAILURE
    assert result.code == "decision_server_unreachable"
    assert server.received == []


def test_ipv6_loopback_is_accepted(start_server) -> None:
    if not socket.has_ipv6:
        pytest.skip("IPv6 is unavailable")
    try:
        server = start_server(ipv6=True)
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")

    result = decide(_question_set(), backend=_backend(server.origin))

    assert result.state is DecisionState.SUCCESS


# ---------------------------------------------------------------------------
# Endpoint policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint", "reason_code"),
    [
        ("http://10.0.0.5:8000/v1/systemone", "non_https_origin"),
        ("http://192.168.1.10/v1/systemone", "non_https_origin"),
        (f"http://{PUBLIC_ADDRESS}/v1/systemone", "non_https_origin"),
        (f"https://{PUBLIC_ADDRESS}/v1/systemone", "origin_not_allowed"),
        ("https://decider.example.test/v1/systemone", "origin_not_allowed"),
        ("http://169.254.169.254/latest", "non_https_origin"),
        ("http://0.0.0.0:8000/v1/systemone", "non_https_origin"),
        ("ftp://127.0.0.1/v1/systemone", "unsupported_scheme"),
        ("http://user:secret@127.0.0.1/v1", "userinfo_not_allowed"),
        ("127.0.0.1:8000", "missing_scheme"),
    ],
)
def test_non_loopback_endpoints_are_refused_by_default(
    endpoint: str,
    reason_code: str,
) -> None:
    with pytest.raises(DecisionServerEndpointError) as raised:
        _backend(f"{endpoint}?token={QUERY_TOKEN}")

    assert raised.value.reason_code == reason_code
    assert raised.value.code == "decision_server_endpoint_refused"
    assert QUERY_TOKEN not in json.dumps(raised.value.to_dict())
    assert QUERY_TOKEN not in repr(raised.value)


def test_localhost_must_resolve_to_loopback_only() -> None:
    with pytest.raises(DecisionServerEndpointError) as raised:
        _backend(
            "http://localhost:8000",
            resolver=lambda host, port: ["127.0.0.1", PUBLIC_ADDRESS],
        )

    assert raised.value.reason_code == "loopback_only_violation"


def test_endpoints_are_revalidated_on_every_call(start_server) -> None:
    server = start_server()
    port = server.origin.rsplit(":", 1)[1]
    answers = [["127.0.0.1"], [PUBLIC_ADDRESS]]
    backend = _backend(
        f"http://localhost:{port}",
        resolver=lambda host, port: answers.pop(0) if len(answers) > 1 else answers[0],
    )

    result = decide(_question_set(), backend=backend)

    assert result.state is DecisionState.FAILURE
    assert result.code == "decision_server_endpoint_refused"
    assert server.received == []


def test_private_network_hosts_need_operator_approval_and_no_phi(
    start_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = "http://10.24.8.9:8000"
    endpoint = f"{origin}/v1/systemone"
    with pytest.raises(DecisionServerEndpointError) as raised:
        _backend(endpoint, contains_phi=False)
    assert raised.value.reason_code == "non_https_origin"
    with pytest.raises(DecisionServerEndpointError) as raised:
        _backend(endpoint, allowed_private_origins=[origin])
    assert raised.value.reason_code == "phi_requires_loopback"
    with pytest.raises(DecisionServerEndpointError) as raised:
        _backend(endpoint, allowed_private_origins=["http://127.0.0.1:8000"])
    assert raised.value.reason_code == "invalid_private_origin"

    server = start_server()
    fake_port = int(server.origin.rsplit(":", 1)[1])
    dialed: list[tuple[str, int]] = []
    real_create_connection = socket.create_connection

    def create_connection(address: tuple[str, int], *args: Any, **kwargs: Any):
        # Stand-in for the private network: dial the in-process server instead.
        dialed.append(address)
        return real_create_connection(("127.0.0.1", fake_port), *args, **kwargs)

    backend = _backend(endpoint, allowed_private_origins=[origin], contains_phi=False)
    assert backend.scope is DecisionServerScope.PRIVATE_NETWORK
    assert backend.identity.runtime == "system_one_server:private_network"
    monkeypatch.setattr(socket, "create_connection", create_connection)

    result = decide(_question_set(), backend=backend)

    assert result.state is DecisionState.SUCCESS
    assert dialed == [("10.24.8.9", 8000)]
    assert server.received[0].headers["host"] == "10.24.8.9:8000"

    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    offline = decide(_question_set(), backend=backend)
    assert offline.code == "decision_server_endpoint_refused"
    assert len(server.received) == 1
    with pytest.raises(DecisionServerEndpointError) as raised:
        _backend(endpoint, allowed_private_origins=[origin], contains_phi=False)
    assert raised.value.reason_code == "offline_mode"


def test_offline_mode_keeps_loopback_available(
    start_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    server = start_server()

    result = decide(_question_set(), backend=_backend(server.origin))

    assert result.state is DecisionState.SUCCESS


def test_unix_socket_servers_are_local(unix_server) -> None:
    server, path = unix_server
    backend = DecisionServerBackend(
        unix_socket=path,
        model_id=MODEL,
        revision=REVISION,
        license_id="MIT",
    )
    server.queue(redirect("/v1/systemone?hop=1"))

    result = decide(_question_set(), backend=backend)

    assert result.state is DecisionState.SUCCESS
    assert backend.scope is DecisionServerScope.UNIX_SOCKET
    assert result.backend["runtime"] == "system_one_server:unix_socket"
    assert [item.path for item in server.received] == [
        "/v1/systemone",
        "/v1/systemone?hop=1",
    ]
    assert server.received[0].headers["host"] == "localhost"
    assert path not in json.dumps(backend.safe_metadata())

    server.queue(redirect("http://127.0.0.1:9/v1/systemone"))
    refused = decide(_question_set(), backend=backend)
    assert refused.code == "decision_server_redirect_refused"


def test_unix_socket_paths_are_validated() -> None:
    if not hasattr(socket, "AF_UNIX") or sys.platform == "win32":
        pytest.skip("Unix domain sockets are unavailable on this platform")
    directory = tempfile.mkdtemp(prefix="omds")
    for path in ("relative/decider.sock", "/" + "d" * 120 + ".sock"):
        with pytest.raises(DecisionServerEndpointError) as raised:
            DecisionServerBackend(
                unix_socket=path,
                model_id=MODEL,
                revision=REVISION,
                license_id="MIT",
            )
        assert raised.value.reason_code == "invalid_unix_socket"
    with pytest.raises(DecisionError):
        _backend("http://127.0.0.1:8000", path="/v1/systemone")
    with pytest.raises(DecisionError):
        DecisionServerBackend(
            endpoint="http://127.0.0.1:8000",
            unix_socket=os.path.join(directory, "decider.sock"),
            model_id=MODEL,
            revision=REVISION,
            license_id="MIT",
        )
    regular_file = os.path.join(directory, "not-a-socket")
    with open(regular_file, "w", encoding="utf-8") as handle:
        handle.write("synthetic")

    try:
        for path in (os.path.join(directory, "missing.sock"), regular_file):
            backend = DecisionServerBackend(
                unix_socket=path,
                model_id=MODEL,
                revision=REVISION,
                license_id="MIT",
            )
            result = decide(_question_set(), backend=backend)
            assert result.code == "decision_server_unreachable"
    finally:
        os.unlink(regular_file)
        os.rmdir(directory)


# ---------------------------------------------------------------------------
# Identity, license and served model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("license_id", ["CC-BY-NC-4.0", "GPL-3.0-only", "", None])
def test_undeclared_or_non_permissive_licenses_are_rejected(license_id: Any) -> None:
    with pytest.raises(DecisionError):
        _backend("http://127.0.0.1:8000", license_id=license_id)


def test_reported_models_must_match_the_declaration(start_server) -> None:
    server = start_server()
    backend = _backend(server.origin, served_model="decider-served")

    for reported in (MODEL, "DECIDER-4B", "local/decider-4b", "decider-served"):
        server.queue(
            reply_json(_system_one(lambda r, name=reported: r.update(model=name)))
        )
        assert decide(_question_set(), backend=backend).state is (DecisionState.SUCCESS)

    server.queue(reply_json(_system_one(lambda r: r.update(model="other-model-9b"))))
    mismatch = decide(_question_set(), backend=backend)
    assert mismatch.state is DecisionState.CONFLICT
    assert mismatch.code == "decision_server_model_mismatch"
    _assert_no_answers(mismatch)

    server.queue(reply_json(_system_one(lambda r: r.update(model="other-model-9b"))))
    unverified = _backend(server.origin, verify_served_model=False)
    assert decide(_question_set(), backend=unverified).state is DecisionState.SUCCESS


def test_bearer_tokens_are_sent_but_never_exposed(start_server) -> None:
    server = start_server()
    backend = _backend(server.origin, bearer_token=BEARER_TOKEN)

    decide(_question_set(), backend=backend)

    assert server.received[0].headers["authorization"] == f"Bearer {BEARER_TOKEN}"
    assert BEARER_TOKEN not in repr(backend)
    assert BEARER_TOKEN not in json.dumps(backend.safe_metadata())
    assert backend.safe_metadata()["authenticated"] is True
    with pytest.raises(DecisionError):
        _backend(server.origin, bearer_token="has space")


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_retries", -1),
        ("max_retries", 4),
        ("max_redirects", 6),
        ("max_response_bytes", 0),
        ("max_retries", True),
        ("protocol", "grpc"),
    ],
)
def test_bounds_fail_closed(option: str, value: Any) -> None:
    with pytest.raises((DecisionError, ValueError)):
        _backend("http://127.0.0.1:8000", **{option: value})


# ---------------------------------------------------------------------------
# Privacy of errors, results and logs
# ---------------------------------------------------------------------------


def test_errors_results_and_logs_never_carry_query_input_or_instructions(
    start_server,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = start_server()
    echo = f"{INPUT_TEXT} {INSTRUCTIONS} {OPTION_TEXT} {QUERY_TOKEN}"
    endpoint = f"{server.origin}/v1/systemone?token={QUERY_TOKEN}"
    backend = _backend(endpoint, bearer_token=BEARER_TOKEN)
    server.queue(
        reply_json({"detail": echo}, status=422),
        reply_raw(echo.encode("utf-8")),
        redirect(f"http://{PUBLIC_ADDRESS}/?q={QUERY_TOKEN}"),
        reply_json(_system_one(lambda r: r.update(model=echo))),
        sleep_then(1.0, reply_json(SYSTEM_ONE_RESPONSE)),
    )
    errors: list[DecisionServerError] = []
    results = []

    with caplog.at_level(logging.DEBUG):
        for timeout_ms in (5000, 5000, 5000, 5000, 200):
            request = _question_set(timeout_ms=timeout_ms)
            with pytest.raises(DecisionServerError) as raised:
                backend.score_questions(request)
            errors.append(raised.value)
        server.queue(reply_json({"detail": echo}, status=422))
        results.append(decide(_question_set(), backend=backend))

    assert [type(error) for error in errors] == [
        DecisionServerUnsupportedError,
        DecisionServerError,
        DecisionServerError,
        DecisionServerConflictError,
        DecisionServerTimeoutError,
    ]
    assert server.received[0].path == f"/v1/systemone?token={QUERY_TOKEN}"
    exposed = "\n".join(
        [
            caplog.text,
            *(record.getMessage() for record in caplog.records),
            *(str(error) for error in errors),
            *(repr(error) for error in errors),
            *(json.dumps(error.to_dict()) for error in errors),
            *(json.dumps(result.to_dict()) for result in results),
            repr(backend),
            json.dumps(backend.safe_metadata()),
        ]
    )
    for secret in (INPUT_TEXT, INSTRUCTIONS, OPTION_TEXT, QUERY_TOKEN, BEARER_TOKEN):
        assert secret not in exposed
    assert endpoint not in exposed
    assert any(
        "decision_server_input_rejected" in r.getMessage() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Result codes refined by backend exceptions
# ---------------------------------------------------------------------------


class _RaisingBackend:
    def __init__(self, error: Exception) -> None:
        self.identity = DecisionBackendIdentity(
            backend_id="synthetic.raising.v1",
            kind=DecisionBackendKind.SPECIALIST,
            runtime="python",
            model_id=MODEL,
            revision=REVISION,
            license_id="Apache-2.0",
        )
        self._error = error

    def score(self, request: DecisionRequest) -> Any:
        raise self._error


def _with_code(base: type[Exception], code: Any) -> Exception:
    error = base("synthetic")
    error.decision_code = code  # type: ignore[attr-defined]
    return error


@pytest.mark.parametrize(
    ("error", "state", "code"),
    [
        (
            _with_code(DecisionError, "synthetic_detail"),
            DecisionState.FAILURE,
            "synthetic_detail",
        ),
        (
            _with_code(DecisionUnsupportedError, "synthetic_refusal"),
            DecisionState.UNSUPPORTED,
            "synthetic_refusal",
        ),
        (
            _with_code(DecisionError, "Not A Code! " + INPUT_TEXT),
            DecisionState.FAILURE,
            "backend_failure",
        ),
        (_with_code(RuntimeError, 42), DecisionState.FAILURE, "backend_failure"),
        (DecisionServerTimeoutError(), DecisionState.FAILURE, "backend_timeout"),
    ],
)
def test_backend_exceptions_may_refine_the_result_code(
    error: Exception,
    state: DecisionState,
    code: str,
) -> None:
    request = DecisionRequest(
        mode=DecisionMode.FIXED_CHOICE,
        input_text=INPUT_TEXT,
        options=("alpha", "beta"),
    )

    result = decide(request, backend=_RaisingBackend(error))

    assert (result.state, result.code) == (state, code)
    assert INPUT_TEXT not in json.dumps(result.to_dict())
