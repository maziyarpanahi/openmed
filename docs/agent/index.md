# Governed agents

OpenMed provides local contracts for permissioned agent workflows: bounded
authority, minimum data, human review, evidence and recovery. The application
composes these contracts around its own dispatch boundary. Installing OpenMed,
validating a schema or recording a digest does not install that boundary or
authorize a tool call.

**OpenMed makes no autonomous clinical decisions.** These contracts do not
authorize diagnosis, treatment, prescribing, enrollment, outreach or clinical
orders. A human-review packet requests review; it cannot supply approval.
Production effects require the application's policy, valid authority, required
human approval and a bounded executor. Synthetic examples demonstrate contract
behavior, not clinical validation or permission to act on real records.

## Lifecycle

| Stage | Application responsibility before advancing |
| --- | --- |
| Authority | Verify signed capability constraints, audience, expiry and any delegation ancestry for the exact tool, resource, action and policy |
| Minimum data | Verify the run- and purpose-bound access ticket; plan the permitted fields before materializing records or tool arguments |
| Preview | Validate the action graph, tool catalog, argument policy, target capabilities and staged changes; retain a value-free preview of the proposed effect |
| Approval | Obtain review through the trusted reviewer boundary; verify the approval against the exact action or preview digest, reviewer role and exclusive expiry |
| Execution | Recheck the approved context at dispatch; keep credentials and clinical values in trusted local code and delegate only the bounded approved effect |
| Evidence | Record controlled outcomes, opaque correlations, counts, offsets and digests; preserve the distinction between success, abstention, review, denial and failure |
| Recovery | Reconcile durable commit evidence before a retry; repeat only a proven-absent effect under still-valid approval, and send ambiguity to human review |
| Evaluation | Bind the evaluated submission and evidence, apply every release gate and disclose limitations before deployment review |

These stages explain composition; they are not an automatic end-to-end
orchestrator. Individual helpers validate only their documented boundary.
Capability lifetime checks do not verify a signature, a cached FHIR capability
check does not validate a clinical resource on a server, and a batch's approval
binding does not independently authenticate its reviewer. The deployment must
supply each missing integration before dispatch.

For a synthetic example, suppose a run has permission to read declared evidence
and proposes an OMOP update. The read grant cannot authorize that update, so
dispatch stops before a clinical write. A separately authorized write still
requires minimum-data planning, a reviewed preview, matching approval and a
trusted committer. Copying the preview digest into a receipt is not review.

## Authority and minimum data

Choose exact scopes before loading data. Tool schemas and catalogs describe
the reviewed surface; they do not grant authority by themselves.

- [Signed capability grants](capability-grants.md): offline signature verification over exact capability constraints.
- [Capability validity](capability-validity.md): caller-supplied audience and lifetime checks.
- [Purpose-bound access tickets](purpose-bound-access.md): bind a run, purpose, data classes, selectors and tool/action pairs.
- [Non-amplifying delegation](non-amplifying-delegation.md): restrict child authority to the active parent's permitted intersection.
- [Governance identifiers](governance-identifiers.md): canonical developer-authored capability, purpose, policy, workflow and tool names.
- [Minimum-data tool contracts](minimum-data-contracts.md): lint declared input schemas before registration or review.
- [Minimum-data projections](minimum-data-projections.md): plan field materialization before reading a record.
- [Tool argument classification](tool-argument-classification.md): apply trusted path rules to structured arguments before invocation.
- [Tool inventory](tool-inventory.md): retain canonical tool metadata and closed side-effect classes.
- [Tool inventory JSON Schema](tool-inventory-schema.md): validate the portable inventory structure locally.
- [Tool catalog diffs](tool-catalog-diffs.md): review content-free changes to the registered tool surface.

## Approvals and previews

Keep reviewer handoffs separate from approval. A changed action, expired approval
or consumed nonce cannot silently acquire permission for another dispatch.

- [Reviewer handoff packets](reviewer-handoffs.md): request a bounded human decision without authorizing an action.
- [Policy decision matrices](policy-decision-matrices.md): compare caller-supplied decisions for review without clinical content.
- [Single-use approval tokens](human-approval-tokens.md): verify signed action, role, expiry and nonce claims before a high-impact action.
- [Approval receipt failures](human-approval-receipts.md): exercise expiry, replay and changed-review-context refusals.

## Write-back and execution

Preflight and staging precede write dispatch. Check cached FHIR capabilities
before obtaining write credentials or materializing a resource. Local FHIR
capability metadata establishes declared interaction support; SMART scope
comparison is an offline audit. Neither performs OAuth, obtains credentials or
sends a resource. Legacy direct FHIR/OpenMRS writers remain outside the complete
governed lifecycle. Applications must compose the reviewed authority and
approval boundary around any production writer.

- [Action graphs](action-graphs.md): validate dependencies and deterministic ordering without scheduling or execution.
- [Action execution phases](action-phases.md): validate metadata-only lifecycle transitions without running the action.
- [FHIR write capability preflight](../interop/fhir-write-preflight.md): check a plan against a cached R4 CapabilityStatement.
- [SMART scope audits](../interop/smart-scope-audit.md): compare synthetic workflow requirements with declared resource scopes.
- [Staged OMOP mutations](../interop/omop-staged-mutations.md): separate trusted row values from review metadata and delegate atomic commit to an application-owned committer.
- [OMOP vocabulary write gates](../interop/omop-vocabulary-write-gates.md): verify mapping provenance against the target snapshot before commit.
- [FHIR-to-OMOP write lineage](../interop/fhir-omop-write-lineage.md): bind source-element and target-row digests before batch approval.

## Evidence

Evidence records what was checked or observed. Structural completeness is not
proof of clinical truth, a digest is not anonymity, and an outcome report cannot
authorize a subsequent action. Apply access controls and retention limits to
opaque references, offsets and digests as well as to protected source artifacts.

- [Artifact references](artifact-references.md): point to an artifact without serializing its content, filename or location.
- [Event correlation](event-correlation.md): correlate runs and actions with opaque identifiers.
- [Event attributes](event-attributes.md): constrain event metadata to a closed allowlist.
- [Event sequences](event-sequences.md): validate append-only replay order without repairing or storing events.
- [Outcome reason codes](outcome-reasons.md): retain distinct success, abstention, review, denial and failure outcomes.
- [Error envelopes](error-envelopes.md): produce controlled failures without copying source exception text.
- [Timing metadata](timing-metadata.md): retain caller-supplied monotonic boundaries and exact durations.
- [Run summaries](run-summaries.md): serialize validated aggregate run metadata without raw traces.
- [Run commitments](run-commitments.md): bind evidence to one exact validated summary.
- [Run diffs](run-diffs.md): compare summary metadata without reopening raw events.
- [Workflow rollups](workflow-rollups.md): aggregate validated runs by workflow.
- [Schema compatibility](schema-compatibility.md): reject unsupported artifact versions explicitly.
- [Evidence-grounded query planning](evidence-query-planner.md): plan bounded read-only evidence retrieval without generating clinical facts.
- [Chart-abstraction evidence](chart-abstraction-evidence.md): retain field-level provenance, uncertainty and review state.
- [Prior-authorization completeness](prior-auth-completeness.md): report structural evidence gaps without deciding coverage or clinical truth.
- [Cohort explanations](cohort-explanations.md): distinguish met, unmet, unknown and conflicting evidence without enrollment or contact.
- [Quality-measure evidence](quality-measure-evidence.md): bind calculation provenance and aggregate results for reproduction and review.
- [Trial-eligibility disagreement review](trial-eligibility-review.md): compare evidence-backed assessments and rules without initiating enrollment or outreach.

## Recovery

Reconcile each effect using the same durable idempotency key. A committed effect
must not be repeated; ambiguous, missing, conflicting or changed evidence needs
review. Rollback manifests describe structural coverage and protected artifact
bindings. They neither establish semantic reversibility nor execute a rollback.
Compensating clinical writes remain proposals requiring separate review.

- [Durable workflow recovery](workflow-recovery.md): reconcile absent, committed and ambiguous effects without replaying bearer approval tokens.
- [OMOP rollback manifests](../interop/omop-rollback-manifests.md): bind operation-compatible rollback strategies to the exact batch and vocabulary snapshot.
- [OMOP rollback JSON Schema](../interop/omop-rollback-schema.md): validate the portable structural contract before semantic binding checks.

## Evaluation

Keep fixtures synthetic and evaluated surfaces reproducible. Passing synthetic
checks is regression evidence, not clinical validation. A failed required gate
cannot be offset by another score or turned into permission to deploy or write.

- [Synthetic governed-agent trace fixtures](../evaluation/governed-agent-fixtures.md): test allow, abstention, denial, review and bounded failure outcomes offline.
- [Synthetic FHIR capability fixtures](../interop/fhir-capability-fixtures.md): construct independent cached-capability inputs for preflight checks.
- [Sealed workflow manifests](../evaluation/sealed-workflow-manifests.md): bind the submission surface and verify it again at evaluation start.
- [v3.1 agent release gates](../evaluation/v3.1-agent-gates.md): evaluate every non-compensable gate using aggregate evidence and explicit limitations.

## Assurance

The deployment owner verifies that the composed boundary actually guards every
effect. Conformance tests and assurance packs provide technical evidence;
they do not provide certification or autonomous clinical authority.

- [Agent boundary threat model](../security/agent-threat-model.md): exercise a caller-composed policy boundary against synthetic adversarial attempts.
- [Agent deployment assurance](../compliance/v3.1-agent-assurance.md): bind revision, deployment, candidate and gate-report evidence in a bounded adoption pack.

## Maintaining this section

Add a new governance guide to its relevant navigation subsection when the guide
lands, link it once from this overview and update the publication classification.
Keep its existing file path when regrouping pages so public URLs remain stable.
Describe the helper's actual boundary and required application integration;
do not present a schema, projection, preview or evidence artifact as an executor.
