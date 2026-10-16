"""Offline provider injection through actual Python, CLI, REST and MCP seams."""

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from openmed.cli.brief import handle_brief
from openmed.clinical.brief import build_clinical_brief
from openmed.service.app import create_app
from openmed.service.brief import brief_response
from openmed.service.logging import ACCESS_LOGGER_NAME
from tests.unit.clinical.test_brief_context import REVIEW_ID, adapter_fixture
from tests.unit.service.test_brief_surfaces import cli_args

pytestmark = pytest.mark.integration


def test_local_provider_has_identical_protected_output_on_every_seam(tmp_path, caplog):
    extraction, manual, provider, _, _ = adapter_fixture()
    text = extraction.artifact.original_text
    expected = build_clinical_brief(
        extraction.artifact, model="extractive", context=manual
    ).to_response()
    assert (
        brief_response(
            text, model="extractive", review_id=REVIEW_ID, context_provider=provider
        )
        == expected
    )
    app = create_app()
    app.state.brief_context_provider = provider
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/brief", json={"text": text, "model": "extractive", "review_id": REVIEW_ID}
        )
    assert response.status_code == 200
    assert response.json() == expected
    from openmed.mcp.server import openmed_brief

    assert (
        openmed_brief(
            text,
            model="extractive",
            review_id=REVIEW_ID,
            runtime_provider=lambda: SimpleNamespace(brief_context_provider=provider),
        )
        == expected
    )
    args = cli_args(tmp_path, text)
    assert handle_brief(args, context_provider=provider) == 0
    assert args.summary_output.read_text() == expected["summary"]
    audit = json.loads(args.review_output.read_text())
    assert audit == {key: value for key, value in expected.items() if key != "summary"}
    assert "dehydration" not in caplog.text
    assert "dehydration" not in json.dumps(audit)


@pytest.mark.parametrize("state", ["unreviewed", "changed", "unavailable"])
def test_provider_refusals_do_not_generate_or_log_content(state, caplog):
    extraction, _, provider, _, verifier = adapter_fixture()
    if state == "unreviewed":
        verifier.verify = lambda *args: None
    elif state == "changed":
        extraction.artifact.deidentified_text += " Changed."
    else:

        def broken(*args):
            raise RuntimeError("PRIVATE_CANARY")

        verifier.verify = broken
    app = create_app()
    app.state.brief_context_provider = provider
    caplog.set_level(logging.INFO)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/brief",
            json={
                "text": extraction.artifact.original_text,
                "model": "extractive",
                "review_id": REVIEW_ID,
            },
        )
    payload = response.json()
    assert payload["status"] == "refused"
    assert payload["summary"] == ""
    assert payload["stages"] == []
    assert "dehydration" not in response.text + caplog.text
    assert "PRIVATE_CANARY" not in response.text + caplog.text
