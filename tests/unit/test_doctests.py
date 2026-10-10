"""Offline execution coverage for core and governed-agent documentation."""

from __future__ import annotations

import doctest
import hashlib
import http.client
import importlib
import os
import re
import socket
import subprocess
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_TARGET_MODULES = (
    "openmed",
    "openmed.core.pii",
)


@pytest.mark.doctest_examples
@pytest.mark.parametrize("module_name", _TARGET_MODULES)
def test_public_core_doctests(module_name: str) -> None:
    """Run doctests for the targeted public modules."""
    module = importlib.import_module(module_name)
    result = doctest.testmod(
        module,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.failed == 0


_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_EXTRA_GOVERNED_PAGES = (
    "interop/fhir-write-preflight.md",
    "interop/fhir-omop-write-lineage.md",
    "interop/omop-staged-mutations.md",
    "interop/omop-vocabulary-write-gates.md",
    "interop/omop-rollback-manifests.md",
    "interop/omop-rollback-schema.md",
    "evaluation/comparator-harness.md",
    "evaluation/comparator-reports.md",
    "evaluation/governed-agent-fixtures.md",
    "evaluation/sealed-workflow-manifests.md",
    "evaluation/v3.1-agent-gates.md",
    "evaluation/blinded-adjudication.md",
    "evaluation/feedback-budgets.md",
    "evaluation/holdout-commitments.md",
    "evaluation/overlap-forensics.md",
    "security/agent-threat-model.md",
    "security/key-custody-metadata.md",
    "compliance/v3.1-agent-assurance.md",
)

# Each exception binds one reviewed adapter sketch to its exact content.
# A new marker, changed sketch or orphaned entry must be reviewed explicitly.
_FRAGMENTS: dict[str, tuple[str, str, str]] = {
    "workflow-recovery-adapters": (
        "agent/workflow-recovery.md",
        "6223e4d27fb39b2a8fe1de8b831e6542a9c9e43ac5cf89259788dad510605785",
        "Application journal, effect-sink inspection and dispatch adapters.",
    ),
    "omop-database-transaction": (
        "interop/omop-staged-mutations.md",
        "089fc1b11fff64e3d703437fdc785b68e59c97a8b5b01485f9e882468e2d2099",
        "Application transaction and reference-snapshot implementation.",
    ),
    "application-adversarial-boundary": (
        "security/agent-threat-model.md",
        "0333862dff19c5a8229abc59f412df6801059bacff7e6ce3da4705c4fb68d867",
        "Application authority checks and adversarial boundary implementation.",
    ),
}
_MARKER = re.compile(r"<!-- openmed-docs-fragment: ([a-z][a-z0-9-]+) -->")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


class _SnippetError(AssertionError):
    """Report only a controlled reason and documentation location."""

    def __init__(self, reason: str, page: str, line: int) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {page}:{line}")


@dataclass(frozen=True)
class _DocBlock:
    page: str
    line: int
    source: str
    fragment: str | None = None


def _python_blocks(text: str, page: str) -> tuple[_DocBlock, ...]:
    """Collect Python fences in order, including explicitly marked fragments."""
    lines = text.splitlines(keepends=True)
    blocks = []
    consumed_markers: set[int] = set()
    index = 0
    while index < len(lines):
        opening = _FENCE.match(lines[index].rstrip("\r\n"))
        if opening is None:
            index += 1
            continue
        fence, info = opening.groups()
        start = index
        index += 1
        while index < len(lines):
            closing = _FENCE.match(lines[index].rstrip("\r\n"))
            if (
                closing is not None
                and closing[1][0] == fence[0]
                and len(closing[1]) >= len(fence)
                and not closing[2].strip()
            ):
                break
            index += 1
        if index == len(lines):
            raise _SnippetError("unterminated_fence", page, start + 1)
        language = info.strip().split(maxsplit=1)[0] if info.strip() else ""
        if language in {"python", "python3", "py"}:
            preceding = start - 1
            while preceding >= 0 and not lines[preceding].strip():
                preceding -= 1
            marker = (
                _MARKER.fullmatch(lines[preceding].strip()) if preceding >= 0 else None
            )
            fragment = marker[1] if marker is not None else None
            if marker is not None:
                consumed_markers.add(preceding)
            blocks.append(
                _DocBlock(
                    page,
                    start + 2,
                    textwrap.dedent("".join(lines[start + 1 : index])),
                    fragment,
                )
            )
        index += 1
    declared = {
        index for index, line in enumerate(lines) if "openmed-docs-fragment:" in line
    }
    if declared != consumed_markers:
        line = min(declared - consumed_markers) + 1
        raise _SnippetError("invalid_fragment_marker", page, line)
    return tuple(blocks)


def _governed_pages() -> tuple[str, ...]:
    """Discover agent pages; include the reviewed external governance guides."""
    agent = tuple(
        str(path.relative_to(_DOCS)).replace(os.sep, "/")
        for path in sorted((_DOCS / "agent").rglob("*.md"))
    )
    return (*agent, *_EXTRA_GOVERNED_PAGES)


def _check_fragment(block: _DocBlock) -> None:
    entry = _FRAGMENTS.get(block.fragment or "")
    digest = hashlib.sha256(block.source.encode("utf-8")).hexdigest()
    if entry is None or entry[0] != block.page or entry[1] != digest or not entry[2]:
        raise _SnippetError("unreviewed_fragment", block.page, block.line)


def _execute_block(
    block: _DocBlock, namespace: dict[str, Any], blocked_effects: list[str]
) -> None:
    """Execute unchanged code against actual APIs, without exposing error text."""
    try:
        code = compile("\n" * (block.line - 1) + block.source, block.page, "exec")
        if block.fragment is not None:
            _check_fragment(block)
            return
        exec(code, namespace)
    except KeyboardInterrupt:
        raise
    except _SnippetError:
        raise
    except BaseException:
        reason = "external_effect_blocked" if blocked_effects else "snippet_failed"
        raise _SnippetError(reason, block.page, block.line) from None
    if blocked_effects:
        # A broad exception handler in a snippet must not hide an I/O attempt.
        raise _SnippetError("external_effect_blocked", block.page, block.line)


@pytest.fixture
def _offline_doc_effects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Block normal network/process entry points for trusted repository examples."""
    attempts: list[str] = []

    def deny(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        attempts.append("external_effect")
        raise RuntimeError("docs_external_effect_blocked")

    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):
            monkeypatch.setattr(socket.socket, name, deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(urllib.request, "urlopen", deny)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", deny)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", deny)
    monkeypatch.setattr(subprocess, "Popen", deny)
    monkeypatch.setattr(os, "system", deny)
    for name in ("posix_spawn", "posix_spawnp", "execv", "execve", "execvp", "execvpe"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, deny)
    monkeypatch.setattr(time, "sleep", deny)
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    return attempts


class _SyntheticCommitter:
    """Record an approved call in memory; never open a database."""

    def __init__(self) -> None:
        self.mutation_counts: list[int] = []

    def commit_batch(self, mutations: Any, *, batch_digest: str, approval: Any) -> None:
        assert batch_digest == approval.batch_digest
        self.mutation_counts.append(len(mutations))


def _synthetic_batch(concept_id: int = 3303) -> Any:
    from openmed.interop.omop import OmopMutation, OmopMutationBatch

    return OmopMutationBatch(
        (
            OmopMutation.insert("person", {"person_id": 1101}),
            OmopMutation.insert(
                "measurement",
                {
                    "measurement_id": 2202,
                    "person_id": 1101,
                    "measurement_concept_id": concept_id,
                },
            ),
        )
    )


def _synthetic_seals() -> dict[str, Any]:
    from openmed.eval.governance import commit_holdout_manifests
    from openmed.eval.workflows import (
        SEALED_MANIFEST_COMPONENTS,
        seal_workflow_manifest,
    )

    return {
        "holdout_commitment": commit_holdout_manifests(
            "synthetic-docs-v1",
            {
                kind: ("sha256:" + character * 64,)
                for kind, character in zip(
                    ("case", "label", "template", "randomization"), "abcd", strict=True
                )
            },
        ),
        "sealed_manifest": seal_workflow_manifest(
            {
                component: "sha256:" + "e" * 64
                for component in SEALED_MANIFEST_COMPONENTS
            }
        ),
    }


def _synthetic_prelude(page: str) -> dict[str, Any]:
    """Supply only the page's documented application-owned synthetic context."""
    callbacks: list[Any] = []
    namespace: dict[str, Any] = {
        "__name__": "__openmed_doc_example__",
        "send_to_local_review": callbacks.append,
        "send_for_review": callbacks.append,
        "handle_closed_reason": callbacks.append,
        "_review_calls": callbacks,
    }
    if page == "agent/error-envelopes.md":
        from openmed.agent import RunId

        def run_tool() -> None:
            raise ValueError("synthetic-docs-failure")

        namespace.update(run_tool=run_tool, run_id=RunId.generate().serialize())
    elif page == "agent/quality-measure-evidence.md":
        namespace["load_local_signing_key"] = lambda: (
            b"synthetic-docs-key-32-bytes-or-more"
        )
    elif page == "agent/tool-argument-classification.md":
        from openmed.agent.permissions import (
            CapabilityGrantConstraint,
            CapabilityGrantRequest,
            CapabilityGrantSigner,
            CapabilityGrantVerifier,
        )

        key = b"synthetic-docs-key-32-bytes-or-more"
        constraint = CapabilityGrantConstraint(
            tool="tool:org.example/redact@1.0.0",
            resource="resource:org.example/clinical-document@1.0.0",
            action="action:org.example/read@1.0.0",
            policy_profile="policy:org.example/minimum-necessary@1.0.0",
        )
        namespace.update(
            manifest=CapabilityGrantSigner(key).issue(
                [constraint], expires_at=2_000_000_000
            ),
            request=CapabilityGrantRequest(
                tool=constraint.tool,
                resource=constraint.resource,
                action=constraint.action,
                policy_profile=constraint.policy_profile,
            ),
            verifier=CapabilityGrantVerifier(key),
            local_tool=lambda **arguments: arguments,
        )
    elif page == "agent/trial-eligibility-review.md":
        from openmed.agent.workflows import (
            CohortCriterion,
            CohortDefinition,
            CohortRecordEvidence,
            CriterionEvidence,
            CriterionKind,
            EvidenceAssertion,
            explain_criterion_membership,
        )

        definition = CohortDefinition(
            "trial.synthetic_protocol",
            1,
            (CohortCriterion("trial.confirmed_condition", CriterionKind.INCLUSION),),
        )
        record = CohortRecordEvidence(
            "sha256:" + "d" * 64,
            definition.definition_id,
            definition.version,
            (
                CriterionEvidence(
                    "trial.confirmed_condition",
                    EvidenceAssertion.MET,
                    "sha256:" + "e" * 64,
                ),
            ),
        )
        namespace["rule_explanation"] = explain_criterion_membership(record, definition)
    elif page == "interop/fhir-write-preflight.md":
        # Deliberately incompatible: exercise the documented refusal callback.
        namespace["cached_capability_statement"] = {
            "resourceType": "CapabilityStatement",
            "fhirVersion": "4.0.1",
            "rest": [{"mode": "server", "resource": []}],
        }
    elif page in {
        "interop/fhir-omop-write-lineage.md",
        "interop/omop-vocabulary-write-gates.md",
    }:
        from openmed.interop.lineage import FhirElementReference
        from openmed.interop.omop import (
            OmopRowKey,
            VocabularyConcept,
            VocabularyMappingProvenance,
            VocabularySnapshot,
        )

        concept_id = 123456 if page.endswith("omop-vocabulary-write-gates.md") else 3303
        batch = _synthetic_batch(concept_id)
        preview = batch.preview(
            existing_rows=(OmopRowKey("concept", {"concept_id": concept_id}),)
        )
        namespace.update(
            batch=batch,
            preview=preview,
            reviewed_preview_digest=preview.preview_digest,
            receipt_digest="sha256:" + "d" * 64,
            approval=batch.bind_approval(
                preview,
                approved_preview_digest=preview.preview_digest,
                approval_receipt_digest="sha256:" + "d" * 64,
            ),
            local_committer=_SyntheticCommitter(),
            target_snapshot=VocabularySnapshot(
                {"LOINC": "synthetic-release"},
                (VocabularyConcept(3303, "LOINC", standard_concept="S"),),
            ),
            mapping_provenance=VocabularyMappingProvenance(
                3303, "LOINC", "synthetic-release"
            ),
            unsupported_element=FhirElementReference(
                "sha256:" + "c" * 64, "Observation.valueString"
            ),
        )
    elif page == "interop/omop-rollback-manifests.md":
        from openmed.interop.omop import OmopRowKey

        namespace.update(
            locally_prepared_person_row={"person_id": 101},
            locally_selected_visit_key={"visit_occurrence_id": 201},
            locally_prepared_changes={"visit_source_value": "synthetic-revised"},
            local_versions={"SYNTHETIC": "synthetic-release"},
            local_concepts=(),
            insert_rollback_digest="sha256:" + "a" * 64,
            update_rollback_digest="sha256:" + "b" * 64,
            local_row_snapshot=(
                OmopRowKey("visit_occurrence", {"visit_occurrence_id": 201}),
            ),
        )
    elif page == "interop/omop-rollback-schema.md":
        from openmed.interop.omop import VocabularySnapshot
        from openmed.interop.omop_rollback_manifest import (
            OmopRollbackInstruction,
            RollbackStrategy,
            build_omop_rollback_manifest,
        )

        namespace["manifest"] = build_omop_rollback_manifest(
            _synthetic_batch(),
            VocabularySnapshot({"SYNTHETIC": "synthetic-release"}, ()),
            tuple(
                OmopRollbackInstruction(
                    i, RollbackStrategy.DELETE_INSERTED_ROW, "sha256:" + "a" * 64
                )
                for i in range(2)
            ),
        )
    elif page == "evaluation/comparator-reports.md":
        namespace["matrix"] = {
            "suite": "synthetic-docs",
            "model_name": "synthetic-rule",
            "device": "cpu",
            "fixture_count": 1,
            "rows": [
                {"system": "synthetic-local", "status": "scored", "fixture_count": 1}
            ],
        }
    elif page in {"evaluation/feedback-budgets.md", "evaluation/overlap-forensics.md"}:
        namespace.update(_synthetic_seals())
        namespace.update(
            normalized_score=0.8,
            failure_evidence_digest="sha256:" + "f" * 64,
            operator_approval_record_digest="sha256:" + "1" * 64,
            submission_text=("Synthetic submitted artifact.",),
            public_benchmark_text=("Synthetic public artifact.",),
            private_shadow_text=("Synthetic shadow artifact.",),
            private_canary_markers=("synthetic-docs-canary",),
        )
    elif page in {
        "evaluation/v3.1-agent-gates.md",
        "compliance/v3.1-agent-assurance.md",
    }:
        from openmed.eval.suites.agent_release import evaluate_agent_release_gates

        namespace.update(
            evidence=(),
            candidate_digest="sha256:" + "a" * 64,
            source_revision="b" * 40,
            artifact_digest="sha256:" + "c" * 64,
            gate_report=evaluate_agent_release_gates(
                (), candidate_digest="sha256:" + "a" * 64
            ),
        )
    return namespace


def _prepare_followup(page: str, namespace: dict[str, Any]) -> None:
    if page == "agent/quality-measure-evidence.md" and "packet" in namespace:
        namespace.update(
            previous_packet=namespace["packet"], current_packet=namespace["packet"]
        )


@pytest.mark.doctest_examples
@pytest.mark.parametrize("page", _governed_pages())
def test_governed_documentation_python_blocks(
    page: str, _offline_doc_effects: list[str]
) -> None:
    """Run each non-fragment block in one isolated, synthetic namespace per page."""
    blocks = _python_blocks((_DOCS / page).read_text(encoding="utf-8"), page)
    namespace = _synthetic_prelude(page)
    for block in blocks:
        _prepare_followup(page, namespace)
        _execute_block(block, namespace, _offline_doc_effects)
    assert not _offline_doc_effects
    if page == "interop/fhir-write-preflight.md":
        assert len(namespace["_review_calls"]) == 1
    if page == "agent/chart-abstraction-evidence.md":
        assert len(namespace["_review_calls"]) == 1
    if page == "agent/tool-argument-classification.md":
        assert (
            namespace["result"].value["documents"][0]["text"] == "synthetic-redaction"
        )
    if page in {
        "interop/fhir-omop-write-lineage.md",
        "interop/omop-vocabulary-write-gates.md",
    }:
        assert namespace["local_committer"].mutation_counts == [2]
        assert namespace["result"].status.value == "committed"
    if page == "evaluation/v3.1-agent-gates.md":
        assert namespace["report"].decision == "NOT_READY"


def test_documentation_fragment_catalog_is_closed() -> None:
    found = []
    for page in _governed_pages():
        for block in _python_blocks((_DOCS / page).read_text(encoding="utf-8"), page):
            if block.fragment is not None:
                _check_fragment(block)
                found.append(block.fragment)
    assert len(found) == len(set(found))
    assert set(found) == set(_FRAGMENTS)


def test_documented_api_signature_drift_fails_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmed.agent import tool_contract_lint

    def changed_signature(schema: Any, *, new_required_argument: str) -> None:
        del schema, new_required_argument

    monkeypatch.setattr(tool_contract_lint, "lint_tool_contract", changed_signature)
    page = "agent/minimum-data-contracts.md"
    block = _python_blocks((_DOCS / page).read_text(encoding="utf-8"), page)[0]
    with pytest.raises(_SnippetError, match="snippet_failed"):
        _execute_block(block, _synthetic_prelude(page), [])


@pytest.mark.parametrize(
    "source",
    [
        "import socket\nsocket.create_connection(('synthetic.invalid', 443))\n",
        "import urllib.request\nurllib.request.urlopen('https://synthetic.invalid')\n",
        "import subprocess\nsubprocess.run(['synthetic-command'])\n",
        "import os\nos.system('synthetic-command')\n",
        "import time\ntime.sleep(1)\n",
        "import socket\ntry:\n socket.create_connection(('synthetic.invalid', 443))\nexcept Exception:\n pass\n",
    ],
)
def test_documentation_external_effects_fail_closed(
    source: str, _offline_doc_effects: list[str]
) -> None:
    block = _DocBlock("synthetic.md", 1, source)
    with pytest.raises(_SnippetError, match="external_effect_blocked"):
        _execute_block(block, {}, _offline_doc_effects)


def test_snippet_errors_do_not_echo_exception_or_source() -> None:
    sentinel = "synthetic-private-value"
    block = _DocBlock("synthetic.md", 7, f"raise ValueError('{sentinel}')")
    with pytest.raises(_SnippetError) as error:
        _execute_block(block, {}, [])
    assert str(error.value) == "snippet_failed: synthetic.md:7"
    assert sentinel not in str(error.value)


@pytest.mark.parametrize(
    "text,reason",
    [
        ("```python\nx = 1\n", "unterminated_fence"),
        (
            "<!-- openmed-docs-fragment: unreviewed -->\nprose\n",
            "invalid_fragment_marker",
        ),
    ],
)
def test_malformed_documentation_fences_fail(text: str, reason: str) -> None:
    with pytest.raises(_SnippetError, match=reason):
        _python_blocks(text, "synthetic.md")


def test_new_python_blocks_are_discovered_in_order() -> None:
    text = (
        "```text\n```python\n````\n~~~py\ncount = 1\n~~~\n```python\ncount += 1\n```\n"
    )
    blocks = _python_blocks(text, "synthetic.md")
    namespace: dict[str, Any] = {}
    for block in blocks:
        _execute_block(block, namespace, [])
    assert len(blocks) == 2
    assert namespace["count"] == 2


@pytest.mark.parametrize(
    "page,fragment",
    [
        ("synthetic.md", "new-exception"),
        ("agent/workflow-recovery.md", "workflow-recovery-adapters"),
        ("synthetic.md", "workflow-recovery-adapters"),
    ],
)
def test_unreviewed_fragment_cannot_skip_a_new_block(page: str, fragment: str) -> None:
    block = _DocBlock(page, 1, "raise RuntimeError()\n", fragment)
    with pytest.raises(_SnippetError, match="unreviewed_fragment"):
        _execute_block(block, {}, [])
