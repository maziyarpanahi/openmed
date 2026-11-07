"""Local fixture-to-evaluation-to-review workflow, without model or network."""

import hashlib
import json
from pathlib import Path

import pytest

from openmed.eval.ambient_drafts import (
    REVIEW_SCHEMA_VERSION,
    AmbientDraft,
    AmbientFact,
    evaluate_ambient_drafts,
    import_ambient_reviews,
)


@pytest.mark.integration
def test_hand_authored_encounters_and_blinded_reviews():
    fixture = Path(__file__).resolve().parents[1] / "fixtures/eval/ambient_drafts.json"
    source = json.loads(fixture.read_text())
    assert source["synthetic"]
    drafts = []
    for row in source["encounters"]:
        assert {t["role"] for t in row["transcript"]} <= set(row["roles"])

        def records(key):
            return tuple(
                AmbientFact(**{**f, "citations": tuple(f["citations"])})
                for f in row[key]
            )

        drafts.append(
            AmbientDraft(
                row["blinded_case_id"],
                row["text_digest"],
                records("truth"),
                records("statements"),
            )
        )
    report = evaluate_ambient_drafts(drafts)
    assert report["slices"]["overall"]["misattribution"]["suppressed"]
    payload = {"schema_version": REVIEW_SCHEMA_VERSION, "rows": []}
    for draft in drafts:
        for reviewer in ("one", "two"):
            payload["rows"].append(
                {
                    "blinded_case_id": draft.blinded_case_id,
                    "draft_revision": draft.revision,
                    "reviewer_id": hashlib.sha256(reviewer.encode()).hexdigest(),
                    "decision": "revise",
                    "reason": None,
                    "adjudicated": False,
                }
            )
    reviews = import_ambient_reviews(payload, drafts)
    assert reviews["rates"]["agreement"]["rate"] == 1
    assert reviews["reviewer_confirmation_required"]
    assert "synthetic-med" not in json.dumps([report, reviews])
    stale = dict(payload)
    stale["rows"] = [dict(payload["rows"][0], draft_revision="0" * 64)]
    with pytest.raises(ValueError, match="stale_or_unknown_review"):
        import_ambient_reviews(stale, drafts)
