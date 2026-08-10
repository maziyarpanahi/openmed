"""Focused tests for the local optional-integration capability report."""

from __future__ import annotations

import json

from openmed.integrations.capability_probe import (
    CapabilityAdapter,
    CapabilityCheck,
    probe_capabilities,
    provider_fingerprint,
)


def test_probe_is_deterministic_and_reports_stable_counts() -> None:
    adapters = [
        CapabilityAdapter(
            name="remote-looking-name",
            provider="Local Adapter",
            extra="synthetic-extra",
            probe=lambda: True,
        ),
        CapabilityAdapter(
            name="missing-backend",
            provider="Local Adapter",
            extra="missing-extra",
            probe=lambda: False,
        ),
        CapabilityAdapter(
            name="configured-backend",
            provider="Other Adapter",
            probe=lambda: CapabilityCheck(True, "ignored"),
        ),
    ]

    first = probe_capabilities(adapters)
    second = probe_capabilities(reversed(adapters))

    assert first.as_dict() == second.as_dict()
    assert first.counts == {"total": 3, "available": 2, "unavailable": 1}
    assert [entry.name for entry in first.capabilities] == [
        "configured-backend",
        "missing-backend",
        "remote-looking-name",
    ]
    assert first.unavailable_count == 1
    assert first.available_count == 2
    assert first.provider_fingerprints == (
        provider_fingerprint("Local Adapter"),
        provider_fingerprint("Other Adapter"),
    )


def test_missing_extra_and_probe_errors_are_classified_without_exception_text() -> None:
    secret = "synthetic-secret-value"

    def failed_probe() -> bool:
        raise RuntimeError(secret)

    report = probe_capabilities(
        [
            CapabilityAdapter(
                name="missing-package",
                provider="synthetic-provider",
                extra="optional-pack",
                probe=lambda: (_ for _ in ()).throw(ImportError(secret)),
            ),
            CapabilityAdapter(
                name="failed-package",
                provider=secret,
                probe=failed_probe,
            ),
        ]
    )

    payload = report.to_json()
    assert secret not in payload
    assert [entry.reason for entry in report.capabilities] == [
        "probe_error",
        "missing_extra",
    ]
    assert report.capabilities[1].extra == "optional-pack"
    assert report.capabilities[1].provider_fingerprint is not None


def test_mapping_declarations_are_local_and_invalid_results_are_safe() -> None:
    calls: list[str] = []

    report = probe_capabilities(
        {
            "local-ready": lambda: calls.append("local-ready") or True,
            "local-missing": {"available": False, "extra": "optional-pack"},
            "local-invalid": {"probe": lambda: {"unexpected": "value"}},
        }
    )

    assert calls == ["local-ready"]
    assert report.counts == {"total": 3, "available": 1, "unavailable": 2}
    assert {entry.name: entry.reason for entry in report.capabilities} == {
        "local-ready": "available",
        "local-missing": "missing_extra",
        "local-invalid": "invalid_result",
    }


def test_report_json_is_json_safe_and_provider_fingerprint_is_one_way() -> None:
    report = probe_capabilities(
        [CapabilityAdapter(name="local", provider="Provider Name", probe=lambda: True)]
    )

    parsed = json.loads(report.to_json(indent=None))
    assert parsed["fingerprint"] == report.fingerprint
    assert parsed["capabilities"][0]["provider_fingerprint"].startswith("sha256:")
    assert provider_fingerprint(" Provider Name ") == provider_fingerprint(
        "provider name"
    )
    assert "Provider Name" not in report.to_json()
