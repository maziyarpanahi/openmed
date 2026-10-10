"""Offline end-to-end multilingual packets across the existing public surfaces."""

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from openmed.cli.brief import handle_brief
from openmed.clinical import build_clinical_brief
from openmed.clinical.summarize_backends import ExtractiveSummarizerBackend
from openmed.mcp.server import openmed_brief
from openmed.service.app import create_app
from openmed.service.client import OpenMedClient
from openmed.service.logging import ACCESS_LOGGER_NAME
from tests.fixtures.clinical.multilingual_briefs import (
    REVIEW_ID,
    SCENARIOS,
    corpus,
    fixture_context,
    parity_report,
)
from tests.unit.service.test_brief_surfaces import cli_args
from tests.unit.service.test_rest_client import SyncASGITransport

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("case", corpus()["cases"], ids=lambda c: c["case_id"])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fixed_local_provider_full_pipeline_surface_parity(
    case, scenario, monkeypatch, tmp_path, capsys, caplog
):
    _, _, value, context, generated = fixture_context(case, scenario)
    admitted = " ".join(
        value.deidentified_text[r.start : r.end] for r in context.packet.references
    )
    calls = []

    def generate(_self, text, *, mode="bhc"):
        assert text == admitted and mode == "bhc"
        assert value.pii_entities[0].text not in text
        calls.append(1)
        return generated

    monkeypatch.setattr(ExtractiveSummarizerBackend, "summarize", generate)

    def provider(text, review_id):
        assert text == value.original_text and review_id == REVIEW_ID
        return value, context

    composer = build_clinical_brief(value, model="extractive", context=context)
    expected = composer.to_response()
    app = create_app()
    app.state.brief_context_provider = provider
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        rest = client.post(
            "/brief",
            json={
                "text": value.original_text,
                "model": "extractive",
                "review_id": REVIEW_ID,
            },
        )
    assert rest.status_code == 200
    client_app = create_app()
    client_app.state.brief_context_provider = provider
    with OpenMedClient(transport=SyncASGITransport(client_app)) as client:
        python = client.brief(
            value.original_text, model="extractive", review_id=REVIEW_ID
        )
    mcp = openmed_brief(
        value.original_text,
        model="extractive",
        review_id=REVIEW_ID,
        runtime_provider=lambda: SimpleNamespace(brief_context_provider=provider),
    )
    args = cli_args(tmp_path, value.original_text)
    assert handle_brief(args, context_provider=provider) == (
        0 if scenario == "preserved" else 1
    )
    cli = {
        **json.loads(args.review_output.read_text()),
        "summary": args.summary_output.read_text(),
    }
    responses = {
        "composer": expected,
        "rest": rest.json(),
        "python": python,
        "mcp": mcp,
        "cli": cli,
    }
    assert all(response == expected for response in responses.values())
    assert len(calls) == len(responses)
    report = parity_report(case, scenario, responses)
    assert report["contract_parity"]
    assert report["surface_count"] == 5
    assert report["model_quality"] == "not_evaluated"
    records = [
        r.openmed_access_log for r in caplog.records if r.name == ACCESS_LOGGER_NAME
    ]
    assert records
    safe = json.dumps(report) + json.dumps(records) + capsys.readouterr().out
    assert value.original_text not in safe
    assert value.pii_entities[0].text not in safe
    assert all(sentence not in safe for sentence in case["sentences"])
    assert REVIEW_ID not in safe
    if scenario == "preserved":
        assert expected["envelope"]["requires_human_review"]
        assert len(expected["citations"]) == 3
    else:
        assert expected["summary"] == "" and expected["citations"] == []
