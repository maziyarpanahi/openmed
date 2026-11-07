"""Source-backed projections and the exact fixture shared with Swift."""

import json
from pathlib import Path

import pytest

from openmed.multimodal.apple_contract import (
    AppleContractError,
    ContractRefusal,
    MultimodalPacket,
    ProjectionKind,
    run_local_adapter,
)
from tests.fixtures.multimodal.apple_contract import fixture

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures/parity/multimodal_contract_v1.json"
)


def corpus():
    return json.loads(FIXTURE.read_text())


def packets():
    return {
        row["id"]: MultimodalPacket(row["wire"].encode()) for row in corpus()["cases"]
    }


def test_actual_source_projections_match_shared_vectors():
    assert fixture() == corpus()
    for row in corpus()["cases"]:
        packet = MultimodalPacket(row["wire"].encode())
        assert packet.evidence_digest == row["digest"]
        assert packet.kind.value == row["expected"]["kind"]
        payload = packet.payload()
        payload.clear()
        assert packet.payload()


@pytest.mark.parametrize("row", corpus()["negative_cases"], ids=lambda row: row["id"])
def test_rejection_codes_match_shared_vectors(row):
    with pytest.raises(AppleContractError) as error:
        MultimodalPacket(row["wire"].encode())
    assert str(error.value) == row["reason"]


def test_scalar_sample_and_clock_boundaries_preserve_independent_expectations():
    rows = packets()
    scalar = rows["transcript_partial"].payload()
    assert scalar["current_span"] == [1, 3]
    assert "A😀éB"[1:3] == "😀é"
    assert scalar["sample_bounds"] == [132, 309]  # floor(3*44.1), ceil(7*44.1)
    assert rows["transcript_shifted"].payload()["current_span"] == [3, 5]
    assert rows["transcript_final"].payload()["finalized"] is True
    assert rows["transcript_invalidated"].payload()["current_span"] is None
    assert rows["clock_aligned"].payload()["guaranteed_overlap_ns"] == [
        3_000_000_000,
        4_000_000_000,
    ]
    assert rows["clock_hold"].payload()["target_start_ns"] is None
    assert (
        rows["clock_large_integer"].payload()["target_start_ns"]["lower"]
        == 4_000_000_000_000_000_001
    )


def test_local_adapter_capability_binding_and_exception_boundaries():
    rows = packets()

    class Adapter:
        local_only = True
        capabilities = frozenset({ProjectionKind.ASSET})
        calls = 0
        response = rows["provider_success_none"]

        def run(self, request):
            self.calls += 1
            return self.response

    adapter = Adapter()
    assert run_local_adapter(adapter, rows["asset_image"]) == adapter.response
    adapter.capabilities = frozenset()
    with pytest.raises(AppleContractError, match=ContractRefusal.CAPABILITY.value):
        run_local_adapter(adapter, rows["asset_image"])
    assert adapter.calls == 1
    adapter.capabilities = frozenset({ProjectionKind.ASSET})
    adapter.local_only = False
    with pytest.raises(AppleContractError, match=ContractRefusal.CAPABILITY.value):
        run_local_adapter(adapter, rows["asset_image"])
    assert adapter.calls == 1
    adapter.local_only = True
    with pytest.raises(AppleContractError, match=ContractRefusal.PROVENANCE.value):
        run_local_adapter(adapter, rows["asset_audio"])
    adapter.response = rows["asset_image"]
    with pytest.raises(AppleContractError, match=ContractRefusal.INVALID.value):
        run_local_adapter(adapter, rows["asset_image"])

    class Failed(Adapter):
        def run(self, request):
            raise RuntimeError("synthetic-private-diagnostic")

    with pytest.raises(AppleContractError) as error:
        run_local_adapter(Failed(), rows["asset_image"])
    assert str(error.value) == ContractRefusal.INVALID.value
