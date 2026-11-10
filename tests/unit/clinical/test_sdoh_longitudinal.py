"""Synthetic offline tests for longitudinal SDOH status resolution."""

from __future__ import annotations

import socket
from datetime import datetime, timezone

import pytest

from openmed.clinical.sdoh_deduplicate import SDOHSourceReference
from openmed.clinical.sdoh_longitudinal import (
    LATEST_EPISODE_UNRESOLVED,
    NO_CURRENT_SUPPORT,
    SAME_TIME_STATUS_CONFLICT,
    SDOHStatusObservation,
    resolve_sdoh_longitudinal_status,
)


def _observation(
    observation_id: str,
    status: str,
    day: int,
    *,
    supports_current: bool = False,
) -> SDOHStatusObservation:
    return SDOHStatusObservation(
        observation_id=observation_id,
        category="food_access",
        status=status,
        effective_at=datetime(2026, 1, day, tzinfo=timezone.utc),
        source=SDOHSourceReference(
            source_id="document-local-1",
            version_id=f"v{day}",
            start=day,
            end=day + 3,
        ),
        supports_current=supports_current,
    )


def test_conflicting_observations_are_retained_in_unresolved_episode() -> None:
    result = resolve_sdoh_longitudinal_status(
        (
            _observation("observation-1", "secure", 2, supports_current=True),
            _observation("observation-2", "insecure", 2, supports_current=True),
        )
    )[0]

    assert result.current_status is None
    assert result.conflict_codes == (
        LATEST_EPISODE_UNRESOLVED,
        SAME_TIME_STATUS_CONFLICT,
    )
    episode = result.episodes[0]
    assert episode.resolved_status is None
    assert [item.observation_id for item in episode.observations] == [
        "observation-1",
        "observation-2",
    ]
    assert len(episode.source_references) == 2


def test_latest_explicit_current_evidence_resolves_current_status() -> None:
    result = resolve_sdoh_longitudinal_status(
        (
            _observation("observation-old", "insecure", 1),
            _observation("observation-current", "secure", 5, supports_current=True),
        )
    )[0]

    assert [episode.resolved_status for episode in result.episodes] == [
        "insecure",
        "secure",
    ]
    assert result.current_status == "secure"
    assert [item.version_id for item in result.current_source_references] == ["v5"]
    assert result.conflict_codes == ()


def test_latest_historical_evidence_does_not_predict_current_status() -> None:
    result = resolve_sdoh_longitudinal_status(
        (_observation("observation-old", "insecure", 1),)
    )[0]

    assert result.current_status is None
    assert result.current_source_references == ()
    assert result.conflict_codes == (NO_CURRENT_SUPPORT,)


def test_longitudinal_resolution_is_deterministic() -> None:
    observations = (
        _observation("observation-current", "secure", 5, supports_current=True),
        _observation("observation-old", "insecure", 1),
    )

    first = resolve_sdoh_longitudinal_status(observations)[0].to_dict()
    second = resolve_sdoh_longitudinal_status(reversed(observations))[0].to_dict()

    assert first == second


def test_resolution_performs_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    assert (
        resolve_sdoh_longitudinal_status(
            (_observation("observation-current", "secure", 5, supports_current=True),)
        )[0].current_status
        == "secure"
    )
