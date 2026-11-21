import json
from dataclasses import asdict, replace
from datetime import datetime

import pytest

from openmed.clinical.use_errors import UseErrorCode
from openmed.core.disclaimer import OpenMedDisclaimer, notice_catalog
from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from scripts.android import generate_parity_fixtures as generator


def test_all_golden_vectors_match_runtime_behavior():
    vectors = json.loads(generator.SAFETY_FIXTURE.read_bytes())
    assert set(vectors["use_error_codes"]) == {c.value for c in UseErrorCode}
    tiers = vectors["tiers"]
    assert len(tiers) == len(RegulatoryTier)
    assert {row["tier"] for row in tiers} == {t.value for t in RegulatoryTier}
    for row in tiers:
        tier = RegulatoryTier(row["tier"])
        assert sorted(tier.obligations) == row["obligations"]
        assert OpenMedDisclaimer.for_tier(tier).notice_id == row["notice_id"]
    catalog = notice_catalog()
    assert {v["id"] for v in vectors["notices"]} == catalog.keys()
    instant = datetime.fromisoformat(vectors["acknowledged_at"])
    for row in vectors["notices"]:
        notice = catalog[row["id"]]
        assert row["text"] == notice.text and row["locale"] == notice.locale
        assert (
            row["version"] == notice.version
            and row["version_digest"] == notice.version_digest
        )
        assert row["tier"] == notice.tier.value
        receipt = notice.acknowledge(at=instant)
        assert (asdict(receipt) if receipt else None) == row["acknowledgement"]
        if row["requires_acknowledgement"]:
            with pytest.raises(ValueError, match="^acknowledgement_required$"):
                notice.require_acknowledgement(None)
        notice.require_acknowledgement(receipt)
    registry = load_intended_use_registry()
    capabilities = vectors["capability_notices"]
    assert capabilities.keys() == registry.declarations.keys()
    for code in capabilities:
        assert OpenMedDisclaimer.for_capability(code).notice_id == capabilities[code]
    for name, code in registry.entry_points.items():
        expected = vectors["entry_point_overrides"].get(name, capabilities[code])
        assert OpenMedDisclaimer.for_capability(name).notice_id == expected


def test_generator_and_read_only_staleness_check(tmp_path, monkeypatch):
    out = tmp_path / "fixture.json"
    args = ["--safety-primitives", "--output", str(out)]
    assert generator.main([*args, "--check"]) == 1
    assert not out.exists()
    assert generator.main(args) == 0
    original = out.read_bytes()
    assert original == generator.SAFETY_FIXTURE.read_bytes()
    assert generator.main([*args, "--check"]) == 0

    import openmed.core.disclaimer as source

    original_catalog = source.notice_catalog()
    key, notice = next(iter(original_catalog.items()))
    monkeypatch.setattr(
        source,
        "notice_catalog",
        lambda: {
            **original_catalog,
            key: replace(notice, version=notice.version + ".1"),
        },
    )
    assert generator.main([*args, "--check"]) == 1
    assert out.read_bytes() == original
