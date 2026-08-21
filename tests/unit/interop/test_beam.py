from __future__ import annotations

import json
import pickle
import traceback
from collections.abc import Iterator, Mapping
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.interop.beam import (
    BeamRedactionError,
    BeamRedactionSpec,
    BeamRedactionTransform,
    run_synthetic_harness,
    serialize_records,
)

beam_adapter = import_module("openmed.interop.beam")
_SENSITIVE = "synthetic-person-001@example.test"


def _fake_deidentifier(text: str, **kwargs):
    assert kwargs["policy"] == "hipaa_safe_harbor"
    assert kwargs["method"] == "mask"
    return SimpleNamespace(
        deidentified_text=text.replace(_SENSITIVE, "[EMAIL]"),
        pii_entities=[object()] if _SENSITIVE in text else [],
    )


def test_spec_is_explicit_bounded_and_value_free():
    spec = BeamRedactionSpec(
        text_field="note",
        max_records=2,
        max_input_bytes=512,
        max_record_bytes=256,
        max_attempts=2,
        extra_kwargs={"seed": 7, "secret": _SENSITIVE},
    )

    metadata = json.dumps(spec.to_dict(), sort_keys=True)

    assert spec.input_schema == "string_or_mapping"
    assert spec.output_schema == "same_as_input"
    assert spec.to_dict()["max_records"] == 2
    assert spec.to_dict()["extra_key_count"] == 2
    assert spec.to_dict()["extra_keys_fingerprint"].startswith("sha256:")
    assert "extra_keys" not in spec.to_dict()
    assert _SENSITIVE not in metadata
    assert _SENSITIVE not in repr(spec)
    assert spec.fingerprint().startswith("sha256:")


def test_direct_harness_serializes_retries_and_reports_only_aggregates():
    attempts = 0

    def flaky_deidentifier(text: str, **kwargs):
        nonlocal attempts
        del kwargs
        attempts += 1
        if attempts == 1:
            raise RuntimeError(f"driver detail: {text}")
        return text.replace(_SENSITIVE, "[EMAIL]")

    result = run_synthetic_harness(
        [
            {"record_id": "synthetic-1", "note": _SENSITIVE},
            {"record_id": "synthetic-2", "note": "synthetic note"},
        ],
        spec=BeamRedactionSpec(text_field="note", max_attempts=2),
        deidentifier=flaky_deidentifier,
    )

    report = json.dumps(result.report(), sort_keys=True)

    assert result.redacted_records == (
        {"record_id": "synthetic-1", "note": "[EMAIL]"},
        {"record_id": "synthetic-2", "note": "synthetic note"},
    )
    assert result.counters.to_dict() == {
        "attempts": 3,
        "input_bytes": 121,
        "output_bytes": 95,
        "records_changed": 1,
        "records_failed": 0,
        "records_processed": 2,
        "retries": 1,
        "spans_redacted": 1,
    }
    assert result.serialized_output == serialize_records(result.redacted_records)
    assert _SENSITIVE not in report
    assert _SENSITIVE not in repr(result)


def test_harness_bounds_state_and_sanitizes_failures():
    with pytest.raises(BeamRedactionError, match="configured limit") as limit_error:
        run_synthetic_harness(
            ["one", "two"],
            spec=BeamRedactionSpec(max_records=1),
            deidentifier=lambda text, **_: text,
        )
    assert _SENSITIVE not in str(limit_error.value)

    def broken_deidentifier(text: str, **kwargs):
        del kwargs
        raise RuntimeError(f"raw failure: {text}")

    with pytest.raises(BeamRedactionError) as failure_error:
        run_synthetic_harness(
            [_SENSITIVE],
            spec=BeamRedactionSpec(max_attempts=2),
            deidentifier=broken_deidentifier,
        )
    assert _SENSITIVE not in str(failure_error.value)


def test_transform_requires_optional_beam_only_when_expanded(monkeypatch):
    transform = BeamRedactionTransform(deidentifier=_fake_deidentifier)
    monkeypatch.setattr(beam_adapter, "_beam", None)

    with pytest.raises(ImportError, match=r"openmed\[beam\]"):
        transform.expand(object())


def test_spec_snapshots_bounded_options_and_reserves_safety_controls():
    nested = {"labels": ["synthetic"]}
    spec = BeamRedactionSpec(extra_kwargs={"nested": nested})
    nested["labels"].append("mutated")

    assert spec.to_deidentify_kwargs()["nested"] == {"labels": ["synthetic"]}
    kwargs = spec.to_deidentify_kwargs()
    kwargs["nested"]["labels"].append("worker-mutation")
    assert spec.to_deidentify_kwargs()["nested"] == {"labels": ["synthetic"]}
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.to_deidentify_kwargs()["nested"] == {"labels": ["synthetic"]}

    for key in (
        "audit",
        "config",
        "keep_mapping",
        "loader",
        "method",
        "policy",
        "use_safety_sweep",
    ):
        with pytest.raises(ValueError, match="reserved worker options"):
            BeamRedactionSpec(extra_kwargs={key: object()})

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="unsupported or unbounded"):
        BeamRedactionSpec(extra_kwargs={"cyclic": cyclic})


def test_spec_rejects_unbounded_limits_and_non_finite_backoff():
    with pytest.raises(ValueError, match="bounded maximum"):
        BeamRedactionSpec(max_records=beam_adapter._MAX_RECORDS + 1)
    with pytest.raises(ValueError, match="bounded range"):
        BeamRedactionSpec(retry_backoff_seconds=float("nan"))
    with pytest.raises(ValueError, match="bounded text"):
        BeamRedactionSpec(text_field="x" * 257)
    with pytest.raises(ValueError, match="safe field identifier"):
        BeamRedactionSpec(text_field="patient-123456")
    with pytest.raises(ValueError, match="not supported"):
        BeamRedactionSpec(method="synthetic")


def test_mutated_spec_and_result_are_revalidated_before_publication():
    spec = BeamRedactionSpec(extra_kwargs={_SENSITIVE: "synthetic"})
    metadata = json.dumps(spec.to_dict(), sort_keys=True)

    assert _SENSITIVE not in metadata
    assert _SENSITIVE not in repr(spec)

    object.__setattr__(spec, "method", _SENSITIVE)
    with pytest.raises(ValueError, match="not supported"):
        spec.to_dict()
    with pytest.raises(ValueError, match="not supported"):
        run_synthetic_harness([], spec=spec, deidentifier=_fake_deidentifier)
    assert repr(spec) == "BeamRedactionSpec(<invalid>)"

    result = run_synthetic_harness(
        ["synthetic"],
        deidentifier=lambda text, **kwargs: text,
    )
    object.__setattr__(result, "output_fingerprint", "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="output_fingerprint"):
        result.report()
    assert repr(result) == "BeamRedactionResult(<invalid>)"


def test_default_worker_options_cannot_weaken_offline_safety(monkeypatch):
    captured: dict[str, object] = {}

    def fake_default(text: str, **kwargs):
        captured.update(kwargs)
        return text.replace(_SENSITIVE, "[EMAIL]")

    monkeypatch.setattr(beam_adapter, "_default_deidentifier", fake_default)

    result = run_synthetic_harness(
        [_SENSITIVE],
        loader_factory=lambda: object(),
    )

    assert result.redacted_records == ("[EMAIL]",)
    assert captured["audit"] is False
    assert captured["keep_mapping"] is False
    assert captured["use_safety_sweep"] is True
    assert getattr(captured["config"], "local_only") is True
    assert getattr(captured["config"], "hf_token") == ""


def test_hostile_record_boundaries_are_value_free():
    class SensitiveAbort(BaseException):
        pass

    class HostileRecord(dict):
        def items(self):
            raise SensitiveAbort(_SENSITIVE)

    with pytest.raises(BeamRedactionError) as mapping_error:
        run_synthetic_harness(
            [HostileRecord(text=_SENSITIVE)],
            deidentifier=_fake_deidentifier,
        )
    assert str(mapping_error.value) == "record could not be inspected"
    assert _SENSITIVE not in str(mapping_error.value)

    def hostile_records():
        yield "synthetic"
        raise SensitiveAbort(_SENSITIVE)

    with pytest.raises(BeamRedactionError) as iterator_error:
        run_synthetic_harness(
            hostile_records(),
            deidentifier=_fake_deidentifier,
        )
    assert str(iterator_error.value) == "record source could not be read"
    assert _SENSITIVE not in str(iterator_error.value)


def test_hostile_redactor_failures_are_value_free():
    class SensitiveAbort(BaseException):
        pass

    def aborting_redactor(text: str, **kwargs):
        del kwargs
        raise SensitiveAbort(text)

    with pytest.raises(BeamRedactionError) as error:
        run_synthetic_harness(
            [_SENSITIVE],
            spec=BeamRedactionSpec(max_attempts=1),
            deidentifier=aborting_redactor,
        )

    assert "record_fingerprint=sha256:" in str(error.value)
    assert _SENSITIVE not in str(error.value)


@pytest.mark.parametrize("fatal_error", [KeyboardInterrupt, SystemExit])
def test_harness_preserves_interpreter_control_exceptions(fatal_error):
    def aborting_redactor(text: str, **kwargs):
        del text, kwargs
        raise fatal_error

    with pytest.raises(fatal_error):
        run_synthetic_harness(
            ["synthetic"],
            deidentifier=aborting_redactor,
        )


def test_redactor_output_and_span_counts_are_bounded(monkeypatch):
    monkeypatch.setattr(beam_adapter, "_MIN_OUTPUT_CHARS", 4)

    with pytest.raises(BeamRedactionError, match="redaction failed after"):
        run_synthetic_harness(
            ["x"],
            spec=BeamRedactionSpec(max_attempts=1),
            deidentifier=lambda text, **kwargs: "x" * 9,
        )

    monkeypatch.setattr(beam_adapter, "_MAX_SPANS_PER_RECORD", 2)
    changed = run_synthetic_harness(
        ["x"],
        deidentifier=lambda text, **kwargs: SimpleNamespace(
            deidentified_text="[R]",
            pii_entities=[object(), object(), object()],
        ),
    )
    assert changed.counters.spans_redacted == 2

    unchanged = run_synthetic_harness(
        ["x"],
        deidentifier=lambda text, **kwargs: SimpleNamespace(
            deidentified_text=text,
            pii_entities=[object(), object(), object()],
        ),
    )
    assert unchanged.counters.records_changed == 0
    assert unchanged.counters.spans_redacted == 0


def test_output_batch_bytes_are_bounded():
    with pytest.raises(BeamRedactionError, match="output byte limit"):
        run_synthetic_harness(
            ["x", "x"],
            spec=BeamRedactionSpec(
                max_records=2,
                max_input_bytes=20,
                max_record_bytes=20,
            ),
            deidentifier=lambda text, **kwargs: "r" * 100,
        )


def test_serialize_records_rejects_an_unbounded_source(monkeypatch):
    monkeypatch.setattr(beam_adapter, "_DEFAULT_MAX_RECORDS", 2)

    def endless_records():
        while True:
            yield "synthetic"

    with pytest.raises(BeamRedactionError, match="configured limit"):
        serialize_records(endless_records())


def test_optional_entity_metadata_failure_is_contained():
    class MetadataFailure(BaseException):
        pass

    class Result:
        deidentified_text = "[REDACTED]"

        @property
        def pii_entities(self):
            raise MetadataFailure(_SENSITIVE)

    result = run_synthetic_harness(
        [_SENSITIVE],
        deidentifier=lambda text, **kwargs: Result(),
    )

    assert result.redacted_records == ("[REDACTED]",)
    assert result.counters.spans_redacted == 1


def test_callable_contracts_are_validated():
    with pytest.raises(TypeError, match="deidentifier must be callable"):
        BeamRedactionTransform(deidentifier=object())
    with pytest.raises(TypeError, match="loader_factory must be callable"):
        run_synthetic_harness([], loader_factory=object())
    with pytest.raises(BeamRedactionError, match="worker-local model setup failed"):
        run_synthetic_harness(["synthetic"], loader_factory=lambda: None)


def test_output_growth_is_bounded_before_result_publication():
    with pytest.raises(BeamRedactionError, match="output"):
        run_synthetic_harness(
            ["synthetic"],
            spec=BeamRedactionSpec(
                max_output_bytes=8,
                max_record_bytes=64,
            ),
            deidentifier=lambda text, **_: text * 8,
        )


def test_reserved_worker_options_cannot_disable_offline_configuration():
    for key in ("config", "loader", "method", "policy"):
        with pytest.raises(ValueError, match="reserved worker options"):
            BeamRedactionSpec(extra_kwargs={key: object()})


def test_reported_spec_metadata_rejects_identifier_shaped_values():
    with pytest.raises(ValueError, match="invalid"):
        BeamRedactionSpec(policy="patient-482901")


def test_mutated_spec_options_are_revalidated_before_execution():
    spec = BeamRedactionSpec()
    object.__setattr__(spec, "extra_kwargs", {"config": object()})

    with pytest.raises(ValueError, match="reserved worker options"):
        run_synthetic_harness(
            ["synthetic"],
            spec=spec,
            deidentifier=lambda text, **_: text,
        )


def test_hostile_record_iteration_failure_is_value_free():
    secret = "synthetic-sensitive-iterator-value"

    class BrokenRecords:
        def __iter__(self) -> Iterator[Any]:
            raise RuntimeError(secret)

    with pytest.raises(BeamRedactionError) as error:
        run_synthetic_harness(BrokenRecords(), deidentifier=lambda text, **_: text)

    rendered = "".join(
        traceback.format_exception(
            type(error.value), error.value, error.value.__traceback__
        )
    )
    assert secret not in rendered


def test_hostile_record_mapping_failure_is_value_free():
    secret = "synthetic-sensitive-mapping-value"

    class HostileRecord(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError(f"{secret}:{key}")

        def __iter__(self) -> Iterator[str]:
            return iter(("text",))

        def __len__(self) -> int:
            return 1

    with pytest.raises(BeamRedactionError) as error:
        run_synthetic_harness(
            [HostileRecord()],
            deidentifier=lambda text, **_: text,
        )

    rendered = "".join(
        traceback.format_exception(
            type(error.value), error.value, error.value.__traceback__
        )
    )
    assert secret not in rendered


def test_loader_initialization_failure_is_value_free():
    secret = "synthetic-sensitive-loader-value"

    def broken_loader():
        raise RuntimeError(secret)

    with pytest.raises(BeamRedactionError) as error:
        run_synthetic_harness(["synthetic"], loader_factory=broken_loader)

    rendered = "".join(
        traceback.format_exception(
            type(error.value), error.value, error.value.__traceback__
        )
    )
    assert secret not in rendered
