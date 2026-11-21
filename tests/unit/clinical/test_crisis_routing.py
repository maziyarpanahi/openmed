import json
import socket
from dataclasses import asdict

import pytest

from openmed.clinical.crisis_routing import (
    GENERIC_NOTICE,
    CrisisRouting,
    CrisisRoutingError,
)


def route(router, locale, **kwargs):
    return router.route(
        **{"capability": "mental_health", "locale": locale, "signal": True, **kwargs}
    )


def test_locale_vectors_are_offline_and_fail_closed(monkeypatch):
    monkeypatch.setattr(
        socket.socket, "connect", lambda *args: pytest.fail("network access")
    )
    router = CrisisRouting()
    for locale, key in [
        ("en-AU", "en-AU"),
        ("en_latn_AU", "en-AU"),
        ("EN_gb", "en-GB"),
        ("en-US", None),
        ("en", None),
        ("unknown", None),
    ]:
        notice = route(router, locale)
        assert notice.mandatory and notice.text == GENERIC_NOTICE
        assert notice.matched_locale == key and bool(notice.resources) == bool(key)
    assert route(router, "en-AU", signal=False) is None
    for capability in ["mental_health", "ecg", "unregistered"]:
        with pytest.raises(CrisisRoutingError, match="^crisis_notice_required$"):
            route(router, "en-AU", capability=capability, suppress_notice=True)
    assert route(router, "en-AU", capability="deidentify", suppress_notice=True) is None


def test_overrides_only_replace_their_locales(tmp_path):
    original = CrisisRouting()
    resource = asdict(route(original, "en-AU").resources[0])
    resource.update(
        id="site_resource", label="Site service", url="https://example.org/emergency"
    )
    p = tmp_path / "override.json"
    p.write_text(
        json.dumps(
            {"schema_version": 1, "locales": {"en-AU": [resource], "fr": [resource]}}
        )
    )
    router = CrisisRouting(override_file=p)
    assert route(router, "en-AU").resources[0].id == "site_resource"
    assert route(router, "fr-CA").matched_locale == "fr"
    assert route(router, "en-GB") == route(original, "en-GB")
    assert route(router, "en-US").resources == ()


@pytest.mark.parametrize(
    "bad",
    [
        {"source_url": None},
        {"verified_on": ""},
        {"verified_on": "2025-02-31"},
        {"url": "javascript:private_marker"},
        {"url": "https://example.org:invalid/private_marker"},
        {"url": "https://user:private_marker@example.org/"},
    ],
)
def test_invalid_override_redacts_diagnostic(tmp_path, bad):
    resource = asdict(route(CrisisRouting(), "en-GB").resources[0])
    p = tmp_path / "private_marker"
    p.write_text(
        json.dumps({"schema_version": 1, "locales": {"en-GB": [{**resource, **bad}]}})
    )
    with pytest.raises(CrisisRoutingError) as exc:
        CrisisRouting(override_file=p)
    assert "private_marker" not in str(exc.value)
