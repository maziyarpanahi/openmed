"""Focused tests for the local optional-integration capability report."""

from __future__ import annotations

import json
import traceback
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from openmed.integrations.capability_probe import (
    MAX_CAPABILITY_ADAPTERS,
    CapabilityAdapter,
    CapabilityCheck,
    CapabilityProbeError,
    CapabilityProbeReport,
    CapabilityStatus,
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
    assert first.provider_fingerprints == tuple(
        sorted(
            (
                provider_fingerprint("Local Adapter"),
                provider_fingerprint("Other Adapter"),
            )
        )
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


def test_hostile_declaration_iteration_failure_is_value_free() -> None:
    secret = "synthetic-sensitive-iterator-value"

    class BrokenDeclarations:
        def __iter__(self) -> Iterator[Any]:
            raise RuntimeError(secret)

    with pytest.raises(CapabilityProbeError) as error:
        probe_capabilities(BrokenDeclarations())

    rendered = "".join(
        traceback.format_exception(
            type(error.value), error.value, error.value.__traceback__
        )
    )
    assert secret not in rendered


def test_base_exception_failures_are_contained_without_value_leakage() -> None:
    secret = "synthetic-sensitive-base-exception-value"

    class ProbeFailure(BaseException):
        pass

    def failed_probe() -> bool:
        raise ProbeFailure(secret)

    report = probe_capabilities(
        [CapabilityAdapter(name="local-failure", probe=failed_probe)]
    )

    assert report.capabilities[0].reason == "probe_error"
    assert secret not in report.to_json()


def test_registry_name_overlapping_a_declaration_field_is_not_ambiguous() -> None:
    report = probe_capabilities({"available": lambda: True})

    assert report.capabilities[0].name == "available"
    assert report.capabilities[0].available is True


def test_hostile_probe_result_is_safely_classified() -> None:
    secret = "synthetic-sensitive-result-value"

    class HostileResult(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError(f"{secret}:{key}")

        def __iter__(self) -> Iterator[str]:
            return iter(("available",))

        def __len__(self) -> int:
            return 1

    report = probe_capabilities(
        [CapabilityAdapter(name="local-hostile", probe=HostileResult)]
    )

    assert report.capabilities[0].reason == "invalid_result"
    assert secret not in report.to_json()


def test_identifier_shaped_names_are_fingerprinted_before_reporting() -> None:
    secret = "patient-482901"

    report = probe_capabilities([CapabilityAdapter(name=secret, probe=lambda: True)])

    assert report.capabilities[0].name.startswith("capability-")
    assert secret not in report.to_json()


def test_raw_declaration_repr_is_value_free() -> None:
    secret = "synthetic-provider-secret-482901"
    adapter = CapabilityAdapter(
        name="local-adapter",
        provider=secret,
        version=secret,
        extra="optional-pack",
        probe=lambda: CapabilityCheck(False, secret),
    )

    assert secret not in repr(adapter)
    assert secret not in repr(adapter.probe())


def test_provider_fingerprint_has_unambiguous_structured_components() -> None:
    assert provider_fingerprint("a\x00b", version="c") != provider_fingerprint(
        "a", version="b\x00c"
    )


def test_duplicate_capability_names_fail_closed() -> None:
    with pytest.raises(CapabilityProbeError, match="unique names"):
        probe_capabilities(
            [
                CapabilityAdapter(name="duplicate", probe=lambda: True),
                CapabilityAdapter(name="duplicate", probe=lambda: False),
            ]
        )


def test_non_callable_probe_is_classified_as_an_invalid_result() -> None:
    report = probe_capabilities([{"name": "local-invalid", "probe": None}])

    assert report.capabilities[0].reason == "invalid_result"


def test_capability_declaration_count_is_bounded() -> None:
    declarations = (
        CapabilityAdapter(name=f"adapter-{index}", probe=lambda: True)
        for index in range(MAX_CAPABILITY_ADAPTERS + 1)
    )

    with pytest.raises(CapabilityProbeError, match="limit of 10000"):
        probe_capabilities(declarations)


def test_public_report_types_revalidate_sensitive_or_tampered_entries() -> None:
    with pytest.raises(ValueError, match="safe identifier"):
        CapabilityStatus(
            name="patient-482901",
            available=True,
            reason="available",
            extra=None,
            provider_fingerprint=None,
        )

    status = CapabilityStatus(
        name="local-safe",
        available=True,
        reason="available",
        extra=None,
        provider_fingerprint=None,
    )
    object.__setattr__(status, "name", "patient-482901")
    with pytest.raises(ValueError, match="safe identifier"):
        CapabilityProbeReport((status,))

    clean_status = CapabilityStatus(
        name="local-safe",
        available=True,
        reason="available",
        extra=None,
        provider_fingerprint=None,
    )
    report = CapabilityProbeReport((clean_status,))
    object.__setattr__(report.capabilities[0], "name", "patient-482901")

    with pytest.raises(ValueError, match="safe identifier") as exc_info:
        report.to_json()

    assert "patient-482901" not in str(exc_info.value)


def test_tampered_structured_probe_result_is_invalid() -> None:
    check = CapabilityCheck(available=True)
    object.__setattr__(check, "available", "yes")

    report = probe_capabilities(
        [CapabilityAdapter(name="local-tampered", probe=lambda: check)]
    )

    assert report.capabilities[0].available is False
    assert report.capabilities[0].reason == "invalid_result"
