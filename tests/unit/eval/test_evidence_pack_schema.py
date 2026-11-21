import json
from importlib.resources import files

import pytest
from jsonschema import Draft202012Validator

from openmed.core.intended_use import RegulatoryTier
from openmed.eval.evidence_pack import EvidencePackError, ValidationEvidencePack
from openmed.eval.journey_release import JOURNEY_RELEASE_STATES


def fixture():
    digest = "sha256:" + "a" * 64
    return dict(
        schema_version="1.0.0",
        capability="deidentify",
        version="4.3.0",
        generated_at="2026-01-01T00:00:00Z",
        intended_use_digest=digest,
        regulatory_tier="non_clinical",
        lanes=[
            dict(
                id="privacy",
                mandatory=True,
                state="success",
                report_digest=digest,
                required_metrics=["recall"],
                metrics={"recall": 1.0},
                claims=["synthetic_release_evidence"],
            )
        ],
        frozen_inputs=[dict(id="suite", origin="synthetic", digest=digest)],
        licenses=[],
        exceptions=[],
        claims=["synthetic_release_evidence"],
        limitations=["not_clinically_validated", "not_a_medical_device"],
        signature=None,
    )


def test_round_trip_digest_and_schema_vocabularies():
    schema = json.loads(
        files("openmed.core.schemas.json")
        .joinpath("validation_evidence_pack.schema.json")
        .read_text()
    )
    Draft202012Validator(schema).validate(fixture())
    assert set(schema["$defs"]["state"]["enum"]) == set(JOURNEY_RELEASE_STATES)
    assert set(schema["$defs"]["tier"]["enum"]) == {t.value for t in RegulatoryTier}
    pack = ValidationEvidencePack(fixture())
    assert ValidationEvidencePack.from_json(pack.to_json()).digest() == pack.digest()
    assert not pack.lane_blockers()


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(extra="private"),
        lambda p: p["frozen_inputs"][0].update(id="/private/note"),
        lambda p: p["limitations"].pop(),
        lambda p: p["limitations"].remove("not_clinically_validated"),
        lambda p: p["lanes"][0]["metrics"].update(recall=float("inf")),
        lambda p: p["lanes"].append(dict(p["lanes"][0], state="partial")),
        lambda p: p["claims"].append("local_first"),
    ],
)
def test_invalid_or_ambiguous_input(change):
    payload = fixture()
    change(payload)
    with pytest.raises(EvidencePackError):
        ValidationEvidencePack(payload)


def test_duplicate_json_keys_rejected():
    with pytest.raises(EvidencePackError, match="invalid_pack_json"):
        ValidationEvidencePack.from_json(
            json.dumps(fixture()).replace(
                '"recall": 1.0', '"recall": 1.0, "recall": 0.9'
            )
        )


@pytest.mark.parametrize("state", JOURNEY_RELEASE_STATES)
def test_mandatory_lane_and_missing_metric(state):
    data = fixture()
    lane = data["lanes"][0]
    lane.update(state=state, metrics={})
    pack = ValidationEvidencePack(data)
    assert "missing:privacy:recall" in pack.lane_blockers()
    assert (f"state:privacy:{state}" in pack.lane_blockers()) == (state != "success")
    assert pack.to_dict()["lanes"][0]["metrics"] == {}
