"""Integration coverage for hostile inputs at the agent dispatch boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from openmed.agent.permissions import (
    CapabilityGrantConstraint,
    CapabilityGrantRequest,
    CapabilityGrantSigner,
    CapabilityGrantVerifier,
)
from openmed.agent.security import (
    ADVERSARIAL_CAPABILITY,
    ADVERSARIAL_POLICY_PROFILE,
    AdversarialAttempt,
    AdversarialReasonCode,
    AttackClass,
    BoundaryVerdict,
    InjectionGuard,
    PromptInjectionDetected,
    assert_adversarial_suite,
)
from openmed.agent.tool_catalog_diff import diff_tool_catalogs
from openmed.agent.tool_inventory import (
    SideEffectClass,
    ToolInventory,
    ToolInventoryRecord,
)
from openmed.agent.tools import plan_data_projection

_KEY = b"synthetic-agent-boundary-key-32-bytes"
_NOW = 2_000_000_000
_PURPOSE = "purpose:org.openmed/adversarial-assurance@1.0.0"
_DATA_CLASS = "data:org.openmed/synthetic-summary@1.0.0"
_TOOL = "tool:org.openmed/adversarial-boundary@1.0.0"
_RESOURCE = "resource:org.openmed/synthetic-envelope@1.0.0"
_ACTION = "action:org.openmed/execute@1.0.0"
_SCHEMA = {
    "type": "object",
    "x-openmed-purpose": _PURPOSE,
    "properties": {
        "summary": {
            "type": "string",
            "x-openmed-purpose": _PURPOSE,
            "x-openmed-minimum-data": "required",
            "x-openmed-data-class": _DATA_CLASS,
        }
    },
    "required": ["summary"],
    "additionalProperties": False,
}


def _catalog(version: str, schema_digest: str) -> ToolInventory:
    return ToolInventory.from_records(
        (
            ToolInventoryRecord(
                tool_id="tool:org.openmed/synthetic-summary",
                version=version,
                capability_class=ADVERSARIAL_CAPABILITY,
                side_effect_class=SideEffectClass.NONE,
                schema_digest=schema_digest,
            ),
        )
    )


def test_hostile_boundaries_fail_before_host_effects_while_control_runs(
    tmp_path: Path,
) -> None:
    constraint = CapabilityGrantConstraint(
        tool=_TOOL,
        resource=_RESOURCE,
        action=_ACTION,
        policy_profile=ADVERSARIAL_POLICY_PROFILE,
    )
    manifest = CapabilityGrantSigner(_KEY).issue((constraint,), expires_at=_NOW + 60)
    verifier = CapabilityGrantVerifier(_KEY)
    request = CapabilityGrantRequest(**constraint.to_dict())
    baseline = _catalog("1.0.0", "sha256:" + "0" * 64)
    injection_guard = InjectionGuard(mode="strict")
    policy_checks: list[str] = []
    host_effects: list[str] = []
    allowed_output = tmp_path / "allowed-control.txt"

    def boundary(
        attempt: AdversarialAttempt,
        dispatch: Callable[[], object],
    ) -> BoundaryVerdict:
        assert attempt.capability == ADVERSARIAL_CAPABILITY
        verifier.verify(manifest, request, now=_NOW)
        projection = plan_data_projection(
            _SCHEMA,
            workflow_purpose=_PURPOSE,
            granted_data_classes=(_DATA_CLASS,),
        )
        assert projection.field_paths == ("/summary",)
        policy_checks.append(attempt.case_id)

        if attempt.attack_class in {
            AttackClass.INSTRUCTION_INJECTION,
            AttackClass.HOSTILE_TOOL_RESULT,
        }:
            try:
                injection_guard.guard_text(str(attempt.payload["content"]))
            except PromptInjectionDetected:
                reason = (
                    AdversarialReasonCode.UNTRUSTED_INSTRUCTION
                    if attempt.attack_class is AttackClass.INSTRUCTION_INJECTION
                    else AdversarialReasonCode.HOSTILE_TOOL_RESULT
                )
                return BoundaryVerdict.deny(reason)

        if attempt.attack_class is AttackClass.CONFUSED_DEPUTY_DELEGATION:
            if attempt.payload["requested_scope"] != attempt.payload["delegated_scope"]:
                return BoundaryVerdict.deny(
                    AdversarialReasonCode.DELEGATION_SCOPE_AMPLIFIED
                )

        if attempt.attack_class is AttackClass.TOOL_CATALOG_SUBSTITUTION:
            candidate = _catalog(
                str(attempt.payload["version"]),
                str(attempt.payload["schema_digest"]),
            )
            if not diff_tool_catalogs(baseline, candidate).is_empty:
                return BoundaryVerdict.deny(AdversarialReasonCode.CATALOG_SUBSTITUTION)

        if attempt.attack_class is AttackClass.CREDENTIAL_LEAKAGE:
            if "credential" in attempt.payload:
                return BoundaryVerdict.deny(AdversarialReasonCode.CREDENTIAL_EXPOSURE)

        if attempt.attack_class is AttackClass.ENDPOINT_LEAKAGE:
            if "endpoint" in attempt.payload:
                return BoundaryVerdict.deny(AdversarialReasonCode.ENDPOINT_EXPOSURE)

        if attempt.attack_class is AttackClass.PATH_TRAVERSAL:
            path = PurePosixPath(str(attempt.payload["path"]))
            if ".." in path.parts:
                return BoundaryVerdict.deny(AdversarialReasonCode.PATH_ESCAPE)

        if attempt.attack_class is AttackClass.URL_ABUSE:
            if urlsplit(str(attempt.payload["url"])).scheme != "https":
                return BoundaryVerdict.deny(AdversarialReasonCode.URL_SCHEME_DENIED)

        if attempt.attack_class is AttackClass.FILESYSTEM_ESCAPE:
            return BoundaryVerdict.deny(AdversarialReasonCode.FILESYSTEM_ACCESS_DENIED)

        if attempt.attack_class is AttackClass.NETWORK_ESCAPE:
            return BoundaryVerdict.deny(AdversarialReasonCode.NETWORK_ACCESS_DENIED)

        if attempt.attack_class is AttackClass.BENIGN_CONTROL:
            allowed_output.write_text("synthetic-control-ok", encoding="utf-8")
            host_effects.append("local-control-write")
            dispatch()
            return BoundaryVerdict.allow()

        raise AssertionError("unhandled adversarial attack class")

    report = assert_adversarial_suite(boundary)

    assert report.passed is True
    assert len(policy_checks) == len(report.cases)
    assert set(policy_checks) == {case.case_id for case in report.cases}
    assert host_effects == ["local-control-write"]
    assert allowed_output.read_text(encoding="utf-8") == "synthetic-control-ok"
    assert all(
        case.dispatch_count
        == (1 if case.attack_class is AttackClass.BENIGN_CONTROL else 0)
        for case in report.cases
    )
