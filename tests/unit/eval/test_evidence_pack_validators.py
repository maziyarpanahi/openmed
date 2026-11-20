import hashlib
import json
from datetime import datetime, timezone

import pytest

from openmed.eval.evidence_pack.validators import (
    JourneyReleaseError,
    canonical_json,
    claim_and_limitation_blockers,
    evaluate_exceptions,
    evaluate_license_catalog,
    reject_duplicate_keys,
    verify_frozen_inputs,
)


def test_frozen_inputs_check_content_and_reject_traversal(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    item = {
        "id": "fixture",
        "origin": "synthetic",
        "path": "input",
        "sha256": "sha256:" + hashlib.sha256(b"safe").hexdigest(),
    }
    (root / "input").write_bytes(b"safe")
    (root / "link").symlink_to(root / "input")
    (tmp_path / "outside").write_bytes(b"safe")
    results, blockers = verify_frozen_inputs(root, [item])
    assert not blockers and results[0]["verified"]
    for path in ("link", "../outside", "sub/../input", str(root / "input")):
        results, blockers = verify_frozen_inputs(root, [{**item, "path": path}])
        assert blockers == ("frozen_input_mismatch:fixture",)
        assert not results[0]["verified"]


def test_license_catalog():
    assets = [
        {
            "asset_id": "fixture",
            "asset_type": "dataset",
            "redistributable": False,
            "distribution": "bundled",
            "use": "train",
        }
    ]
    summary, blockers = evaluate_license_catalog(assets)
    assert summary["restricted_asset_count"] == 1
    assert set(blockers) == {
        "restricted_asset_bundled:fixture",
        "restricted_training_asset:fixture",
        "model_license_inventory_empty",
    }


def test_claims_and_limitations_follow_profile():
    assert claim_and_limitation_blockers(
        ["b"], [], required_claims={"b", "a"}, required_limitations={"x"}
    ) == ("missing_claim:a", "missing_limitation:x")


def test_exceptions_at_expiry_boundary():
    record = {
        "code": "fixture",
        "expires_at": "2025-01-01T00:00:00Z",
        "severity": "high",
        "disposition": "pending",
    }
    records, blockers = evaluate_exceptions(
        [record], evaluated_at=datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    assert records == (record,)
    assert blockers == (
        "expired_exception:fixture",
        "nonwaivable_exception:fixture",
        "unaccepted_exception:fixture",
    )


def test_canonical_json():
    assert canonical_json({"b": "é", "a": 2}) == '{"a":2,"b":"\\u00e9"}'
    with pytest.raises(ValueError):
        canonical_json(float("nan"))


def test_duplicate_keys_hook():
    assert json.loads('{"a":1}', object_pairs_hook=reject_duplicate_keys) == {"a": 1}
    with pytest.raises(JourneyReleaseError, match="duplicate JSON key"):
        json.loads('{"outer":{"a":1,"a":2}}', object_pairs_hook=reject_duplicate_keys)
