"""Synthetic offline controls for exact FHIR transaction assembly."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmed.interop.fhir.transactions import (
    TransactionApproval,
    TransactionAssemblyError,
    TransactionLimits,
    TransactionReviewerRole,
    assemble_transaction,
)
from openmed.interop.fhir.validation import validation_result

INSTANT = datetime(2026, 1, 2, 3, 4, 5, 123456, timezone.utc)
APPROVAL = TransactionApproval(
    "sha256:" + "a" * 64,
    "sha256:" + "b" * 64,
    TransactionReviewerRole.CLINICAL_REVIEWER,
    ("urn:sha256:" + "c" * 64, "urn:uuid:4c545501-b3a7-4e73-88db-779b7b5b6d73"),
)
LIMITS = TransactionLimits(10, 100_000)


@dataclass
class SyntheticEntry:
    resource: dict[str, Any]
    kind: str = "create"
    conditional_predicate: str | None = None
    expected_version: str | None = None


def observation(**updates: Any) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "status": "preliminary",
        "code": {"coding": [{"system": "urn:synthetic", "code": "measurement"}]},
        **updates,
    }


def assemble(entries=None, **kwargs):
    return assemble_transaction(
        [SyntheticEntry(observation())] if entries is None else entries,
        approval=kwargs.pop("approval", APPROVAL),
        limits=kwargs.pop("limits", LIMITS),
        clock=kwargs.pop("clock", lambda: INSTANT),
        **kwargs,
    )


def test_canonical_bytes_and_detached_payload():
    resource = observation(valueString="synthetic 日本語")
    entry = SyntheticEntry(resource)
    result = assemble([entry])
    reversed_entry = SyntheticEntry(dict(reversed(list(resource.items()))))
    repeated = assemble([reversed_entry])
    assert result == repeated
    assert result.serialized == json.dumps(
        result.bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert (
        result.bundle_digest
        == "sha256:" + hashlib.sha256(result.serialized).hexdigest()
    )
    assert validation_result(result.bundle, "R4").valid
    assert result.entry_count == 2
    resource["status"] = "final"
    detached = result.bundle
    detached["entry"].clear()
    assert result == repeated
    assert len(result.bundle["entry"]) == 2
    assert "日本語" not in repr(result)


@pytest.mark.parametrize(
    "entry,expected_request",
    [
        (SyntheticEntry(observation()), {"method": "POST", "url": "Observation"}),
        (
            SyntheticEntry(
                observation(), conditional_predicate="identifier=urn%3Asynthetic%7Cabc"
            ),
            {
                "method": "POST",
                "url": "Observation",
                "ifNoneExist": "identifier=urn%3Asynthetic%7Cabc",
            },
        ),
        (
            SyntheticEntry(
                observation(id="synthetic-1"), "update", expected_version="7"
            ),
            {"method": "PUT", "url": "Observation/synthetic-1", "ifMatch": 'W/"7"'},
        ),
        (
            SyntheticEntry(observation(id="synthetic-1"), "update"),
            {"method": "PUT", "url": "Observation/synthetic-1"},
        ),
        (
            SyntheticEntry(
                observation(), "update", "identifier=urn%3Asynthetic%7Cabc", "7"
            ),
            {
                "method": "PUT",
                "url": "Observation?identifier=urn%3Asynthetic%7Cabc",
                "ifMatch": 'W/"7"',
            },
        ),
    ],
)
def test_request_shapes_are_preserved(entry, expected_request):
    original = copy.deepcopy(entry)
    result = assemble([entry])
    assert result.bundle["entry"][0]["request"] == expected_request
    assert entry == original


def test_provenance_targets_all_writes_without_copying_source_metadata():
    entries = [
        SyntheticEntry(observation(valueString="SYNTHETIC SOURCE TEXT")),
        SyntheticEntry({"resourceType": "Patient", "id": "synthetic"}),
    ]
    # Planner fields outside the minimal protocol must never enter provenance.
    entries[0].reviewer_name = "SYNTHETIC REVIEWER"
    entries[0].credential = "SYNTHETIC CREDENTIAL"
    bundle = assemble(entries).bundle
    provenance = bundle["entry"][-1]
    assert provenance["request"] == {"method": "POST", "url": "Provenance"}
    resource = provenance["resource"]
    assert resource["target"] == [
        {"reference": entry["fullUrl"]} for entry in bundle["entry"][:-1]
    ]
    assert resource["recorded"] == "2026-01-02T03:04:05.123456Z"
    assert resource["extension"][0]["valueCode"] == "clinical-reviewer"
    assert resource["agent"][0]["who"]["identifier"]["value"] == "openmed"
    assert [entity["what"]["identifier"]["value"] for entity in resource["entity"]] == [
        APPROVAL.receipt_digest,
        APPROVAL.action_digest,
        *APPROVAL.evidence_references,
    ]
    assert "SYNTHETIC" not in json.dumps(resource)
    assert validation_result(bundle, "R4").valid


def test_internal_references_are_rewritten_without_mutating_inputs():
    patient = SyntheticEntry({"resourceType": "Patient", "id": "synthetic-patient"})
    obs = SyntheticEntry(
        observation(subject={"reference": "Patient/synthetic-patient"})
    )
    result = assemble([patient, obs])
    assert (
        result.bundle["entry"][1]["resource"]["subject"]["reference"]
        == result.bundle["entry"][0]["fullUrl"]
    )
    assert obs.resource["subject"]["reference"] == "Patient/synthetic-patient"
    assert validation_result(result.bundle).valid


@pytest.mark.parametrize(
    "mutation",
    [
        "resource",
        "predicate",
        "version",
        "kind",
        "receipt",
        "action",
        "evidence",
        "role",
        "clock",
        "order",
    ],
)
def test_all_reviewed_inputs_change_the_digest(mutation):
    entries = [
        SyntheticEntry(observation(id="one"), "update", "identifier=synthetic", "1"),
        SyntheticEntry({"resourceType": "Patient", "id": "two"}),
    ]
    original = assemble(entries)
    approval = APPROVAL
    clock = lambda: INSTANT
    if mutation == "resource":
        entries[0].resource["status"] = "final"
    elif mutation == "predicate":
        entries[0].conditional_predicate = "identifier=changed"
    elif mutation == "version":
        entries[0].expected_version = "2"
    elif mutation == "kind":
        entries[0].kind = "create"
        entries[0].expected_version = None
    elif mutation == "receipt":
        approval = replace(approval, receipt_digest="sha256:" + "d" * 64)
    elif mutation == "action":
        approval = replace(approval, action_digest="sha256:" + "d" * 64)
    elif mutation == "evidence":
        approval = replace(approval, evidence_references=("urn:sha256:" + "d" * 64,))
    elif mutation == "role":
        approval = replace(approval, reviewer_role=TransactionReviewerRole.DATA_STEWARD)
    elif mutation == "clock":
        clock = lambda: INSTANT + timedelta(microseconds=1)
    elif mutation == "order":
        entries.reverse()
    assert (
        assemble(entries, approval=approval, clock=clock).bundle_digest
        != original.bundle_digest
    )


def test_limits_are_inclusive_and_count_provenance_and_utf8_bytes():
    entries = [SyntheticEntry(observation(valueString="日本語"))]
    result = assemble(entries)
    size = len(result.serialized)
    assert assemble(entries, limits=TransactionLimits(2, size)) == result
    for limits, code in [
        (TransactionLimits(1, size), "entry_limit_exceeded"),
        (TransactionLimits(2, size - 1), "size_limit_exceeded"),
    ]:
        with pytest.raises(TransactionAssemblyError, match=f"^{code}$"):
            assemble(entries, limits=limits)


def test_entry_limit_refuses_before_clock_or_entry_access():
    with pytest.raises(TransactionAssemblyError, match="^entry_limit_exceeded$"):
        assemble(
            [object(), object()],
            limits=TransactionLimits(2, 1000),
            clock=lambda: pytest.fail("clock must not run"),
        )


@pytest.mark.parametrize(
    "entry,code",
    [
        (SyntheticEntry(observation(), "delete"), "unsupported_write_kind"),
        (SyntheticEntry(observation(), "update"), "update_target_required"),
        (
            SyntheticEntry(observation(), expected_version="1"),
            "create_version_not_supported",
        ),
        (
            SyntheticEntry(observation(), "update", "identifier=x", 'W/"1"'),
            "invalid_expected_version",
        ),
        (
            SyntheticEntry(observation(status="synthetic-invalid")),
            "invalid_r4_structure",
        ),
        (SyntheticEntry({"resourceType": "Observation"}), "invalid_r4_structure"),
        (
            SyntheticEntry(observation(unexpected="SYNTHETIC PHI")),
            "invalid_r4_structure",
        ),
        (
            SyntheticEntry({"resourceType": "Bundle", "type": "transaction"}),
            "unsupported_resource_type",
        ),
        (SyntheticEntry({"resourceType": "Provenance"}), "unsupported_resource_type"),
        (SyntheticEntry(observation(id="/private/synthetic")), "invalid_resource_id"),
        (SyntheticEntry(observation(id=None)), "invalid_resource_id"),
        (SyntheticEntry(observation(id=True)), "invalid_resource_id"),
        (
            SyntheticEntry(
                observation(
                    subject={
                        "reference": "urn:uuid:00000000-0000-0000-0000-000000000000"
                    }
                )
            ),
            "invalid_r4_structure",
        ),
        (object(), "invalid_entry"),
    ],
)
def test_invalid_entries_have_fixed_errors(entry, code):
    with pytest.raises(TransactionAssemblyError, match=f"^{code}$"):
        assemble([entry])


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), {1: "secret"}, ("tuple",), object(), "\ud800"]
)
def test_non_json_resources_refused_without_echo(value):
    with pytest.raises(TransactionAssemblyError, match="^invalid_resource_json$"):
        assemble([SyntheticEntry(observation(valueString=value))])


@pytest.mark.parametrize(
    "predicate",
    [
        "",
        "?identifier=x",
        "identifier",
        "identifier=",
        "identifier=x#fragment",
        "identifier=%0Asecret",
        "identifier=%GG",
        "identifier=%FF",
        "identifier=x\nAuthorization:secret",
        "https://synthetic.invalid/?identifier=x",
        "identifier=x&",
        "x=" + "a" * 4096,
        "&".join(["x=a"] * 65),
    ],
)
def test_unsafe_predicates_refused(predicate):
    with pytest.raises(
        TransactionAssemblyError, match="^invalid_conditional_predicate$"
    ):
        assemble([SyntheticEntry(observation(), conditional_predicate=predicate)])


@pytest.mark.parametrize(
    "evidence",
    [
        "synthetic source text",
        "/private/synthetic",
        "https://user:secret@example.invalid/source",
        "urn:uuid:not-a-uuid",
        "urn:sha256:secret",
        "urn:uuid:4C545501-B3A7-4E73-88DB-779B7B5B6D73",
        3,
    ],
)
def test_private_evidence_is_rejected(evidence):
    with pytest.raises(TransactionAssemblyError, match="^invalid_evidence_references$"):
        replace(APPROVAL, evidence_references=(evidence,))


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"action_digest": "secret"}, "invalid_approval_digest"),
        ({"receipt_digest": "secret"}, "invalid_approval_digest"),
        ({"reviewer_role": "SYNTHETIC NAME"}, "invalid_reviewer_role"),
        ({"evidence_references": ()}, "invalid_evidence_references"),
        (
            {"evidence_references": (APPROVAL.evidence_references[0],) * 2},
            "invalid_evidence_references",
        ),
    ],
)
def test_invalid_approval_metadata(kwargs, code):
    with pytest.raises(TransactionAssemblyError, match=f"^{code}$"):
        replace(APPROVAL, **kwargs)


@pytest.mark.parametrize(
    "values", [(True, 100), (1, False), (0, 100), (2, 0), (2.0, 100), (-1, 100)]
)
def test_invalid_declared_limits(values):
    with pytest.raises(TransactionAssemblyError, match="^invalid_limits$"):
        TransactionLimits(*values)


def test_duplicate_targets_refused():
    for entries, code in [
        (
            [
                SyntheticEntry(observation(id="same")),
                SyntheticEntry(observation(id="same")),
            ],
            "duplicate_resource_target",
        ),
        (
            [
                SyntheticEntry(observation(), "update", "identifier=x", "1"),
                SyntheticEntry(observation(), "update", "identifier=x", "2"),
            ],
            "duplicate_request_target",
        ),
        (
            [SyntheticEntry(observation(), conditional_predicate="identifier=x")] * 2,
            "duplicate_request_target",
        ),
    ]:
        with pytest.raises(TransactionAssemblyError, match=f"^{code}$"):
            assemble(entries)


@pytest.mark.parametrize(
    "clock", [None, lambda: "secret", lambda: datetime(2026, 1, 1)]
)
def test_invalid_clocks_refused(clock):
    with pytest.raises(TransactionAssemblyError, match="^invalid_clock$"):
        assemble(clock=clock)


def test_clock_failure_does_not_disclose_exception_text():
    def clock():
        raise ValueError("SYNTHETIC CREDENTIAL")

    with pytest.raises(TransactionAssemblyError, match="^invalid_clock$") as error:
        assemble(clock=clock)
    assert error.value.__suppress_context__


def test_equivalent_timezone_instants_produce_identical_bytes():
    offset = timezone(timedelta(hours=2))
    assert assemble(clock=lambda: INSTANT.astimezone(offset)) == assemble()


@pytest.mark.parametrize("entries", [[], "secret", None])
def test_bad_entry_sequences(entries):
    with pytest.raises(TransactionAssemblyError, match="^invalid_entries$"):
        assemble_transaction(
            entries, approval=APPROVAL, limits=LIMITS, clock=lambda: INSTANT
        )


@pytest.mark.parametrize(
    "sentinel",
    [
        "SYNTHETIC_NAME_123",
        "患者番号１２３",
        "patient@example.invalid",
        "/private/synthetic?token=secret",
    ],
)
def test_metadata_and_diagnostic_leakage_sweep(sentinel, caplog):
    for constructor in (
        lambda: replace(APPROVAL, evidence_references=(sentinel,)),
        lambda: replace(APPROVAL, reviewer_role=sentinel),
        lambda: assemble([SyntheticEntry(observation(unexpected=sentinel))]),
        lambda: assemble(
            [SyntheticEntry(observation(), "update", "identifier=x", sentinel)]
        ),
    ):
        with pytest.raises(TransactionAssemblyError) as error:
            constructor()
        assert sentinel not in str(error.value)
        assert sentinel not in repr(error.value)
        assert sentinel not in caplog.text


def test_failing_mapping_does_not_echo_exception():
    class BrokenMapping(Mapping):
        def __iter__(self):
            raise RuntimeError("SYNTHETIC CREDENTIAL")

        def __len__(self):
            return 1

        def __getitem__(self, key):
            raise RuntimeError("SYNTHETIC CREDENTIAL")

    with pytest.raises(
        TransactionAssemblyError, match="^invalid_resource_json$"
    ) as error:
        assemble([SyntheticEntry(BrokenMapping())])
    assert error.value.__suppress_context__


def test_deep_and_cyclic_resources_are_refused():
    cyclic = observation()
    cyclic["extension"] = cyclic
    deep = observation()
    pointer = deep
    for _ in range(65):
        pointer["nested"] = {}
        pointer = pointer["nested"]
    for resource in (deep, cyclic):
        with pytest.raises(TransactionAssemblyError, match="^invalid_resource_json$"):
            assemble([SyntheticEntry(resource)])


def test_repeated_equal_creates_have_distinct_stable_fullurls():
    entries = [SyntheticEntry(observation())] * 2
    result = assemble(entries)
    urls = [entry["fullUrl"] for entry in result.bundle["entry"]]
    assert len(set(urls)) == 3
    assert result == assemble(entries)


def test_public_facade_exports_assembler():
    from openmed.interop.fhir import assemble_transaction as public_assembler

    assert public_assembler is assemble_transaction


@pytest.mark.parametrize("kwargs", [{"approval": None}, {"limits": None}])
def test_invalid_assembly_metadata(kwargs):
    with pytest.raises(TransactionAssemblyError, match="^invalid_assembly_metadata$"):
        assemble(**kwargs)


def test_protocol_properties_are_read_once_and_clock_is_read_once():
    calls = []

    class Entry:
        @property
        def resource(self):
            calls.append("resource")
            return observation()

        @property
        def kind(self):
            calls.append("kind")
            return "create"

        @property
        def conditional_predicate(self):
            calls.append("predicate")
            return None

        @property
        def expected_version(self):
            calls.append("version")
            return None

    def clock():
        calls.append("clock")
        return INSTANT

    assemble([Entry()], clock=clock)
    assert calls == ["resource", "kind", "predicate", "version", "clock"]
