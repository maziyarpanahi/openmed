"""Recipe: one decision model behind its own local server.

This test is skipped unless ``OPENMED_DECISION_SERVER_ENDPOINT`` names a running
loopback server, so it never runs in CI. See "Local decision servers" in
``docs/api/fixed-option-decisions.md`` for the serving commands. The request is
synthetic.

Environment:
    OPENMED_DECISION_SERVER_ENDPOINT: Loopback URL, for example
        ``http://127.0.0.1:8000/v1/systemone``.
    OPENMED_DECISION_SERVER_MODEL: Declared model ID, for example
        ``vllm-sr/d3-mini``.
    OPENMED_DECISION_SERVER_REVISION: Immutable revision of the served weights.
    OPENMED_DECISION_SERVER_LICENSE: SPDX license of the weights
        (``Apache-2.0`` by default).
    OPENMED_DECISION_SERVER_PROTOCOL: ``system_one`` (default) or
        ``openai_decisions``.
    OPENMED_DECISION_SERVER_SERVED_MODEL: Optional name the server uses.
    OPENMED_DECISION_SERVER_TOKEN: Optional bearer key for servers that need one.
"""

from __future__ import annotations

import json
import os

import pytest

from openmed.structured.decision import (
    DecisionMode,
    DecisionQuestion,
    DecisionRequest,
    DecisionState,
    decide,
)
from openmed.structured.decision_server import DecisionServerBackend

ENDPOINT_ENV = "OPENMED_DECISION_SERVER_ENDPOINT"
SYNTHETIC_TICKET = (
    "Synthetic ticket 9001: since 09:10 the checkout page returns HTTP 500 for "
    "every customer and no orders have gone through in 40 minutes."
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get(ENDPOINT_ENV),
        reason=f"set {ENDPOINT_ENV} to a running local decision server",
    ),
]


def test_local_decision_server_answers_a_synthetic_question_set() -> None:
    backend = DecisionServerBackend(
        endpoint=os.environ[ENDPOINT_ENV],
        protocol=os.environ.get("OPENMED_DECISION_SERVER_PROTOCOL", "system_one"),
        model_id=os.environ["OPENMED_DECISION_SERVER_MODEL"],
        revision=os.environ["OPENMED_DECISION_SERVER_REVISION"],
        license_id=os.environ.get("OPENMED_DECISION_SERVER_LICENSE", "Apache-2.0"),
        served_model=os.environ.get("OPENMED_DECISION_SERVER_SERVED_MODEL"),
        bearer_token=os.environ.get("OPENMED_DECISION_SERVER_TOKEN"),
    )
    request = DecisionRequest(
        mode=DecisionMode.QUESTION_SET,
        input_text=SYNTHETIC_TICKET,
        timeout_ms=30000,
        questions=(
            DecisionQuestion(
                question_id="priority",
                mode=DecisionMode.FIXED_CHOICE,
                options=("p1", "p2", "p3"),
                option_descriptions=(
                    "Revenue path fully down",
                    "Degraded but working",
                    "Cosmetic",
                ),
                instructions="Which priority does the ticket need?",
            ),
            DecisionQuestion(
                question_id="all_users",
                mode=DecisionMode.BOOLEAN_CHOICE,
                options=("true", "false"),
                instructions="The problem affects every customer.",
            ),
            DecisionQuestion(
                question_id="urgency",
                mode=DecisionMode.SCALAR_SCORE,
                levels=("Low", "Medium", "High"),
                instructions="How urgent is this ticket?",
            ),
        ),
    )

    result = decide(request, backend=backend)

    # Real probabilities may fall below the calibration threshold; abstaining
    # is allowed, a transport or contract failure is not.
    assert {item.state for item in result.questions} <= {
        DecisionState.SUCCESS,
        DecisionState.ABSTAINED,
    }, [(item.question_id, item.state.value, item.code) for item in result.questions]
    assert result.backend["kind"] == "specialist"
    assert result.backend["runtime"].endswith(":loopback")
    assert SYNTHETIC_TICKET not in json.dumps(result.to_dict())
