# Independent review of OpenMed 4.3 assurance evidence

Use this checklist to review a specific release candidate. Record its commit,
package version, reviewer role, review date and the digest of each artifact you
inspect. Mark each item supported, failed, unavailable or outside the declared
deployment scope. Record a reason for each exclusion.

The artifacts establish technical behavior under the recorded test conditions.
Clinical validation, certification and deployment authorization require
separate decisions. The deploying site is responsible for local validation,
oversight, escalation procedures and the population and languages it supports.
See the [clinical validation protocol](../../clinical-validation-protocol.md)
and [clinician review protocol](../v3.1-clinician-review.md).

## Review checklist

Run commands from an offline checkout of the candidate with development
dependencies installed. For a feature still under review, inspect its linked
change and mark the item unavailable until it is present in the candidate.

| Technical claim to check | Offline evidence and reviewer action |
| --- | --- |
| Each public capability has a declared purpose and exclusions. | Compare the entry points in `openmed/core/data/intended_use.json` with the installed surface. Run `python -m pytest tests/unit/core/test_intended_use_registry.py -q`. |
| Tier-specific obligations match the declared use. | Inspect `openmed/core/intended_use.py` and run `python -m pytest tests/unit/core/test_intended_use.py -q`. Record the tier and declaration digest for the capability under review. |
| The notice shown to a reviewer is bound to the acknowledged version. | On a candidate containing the [tiered notice implementation](https://github.com/maziyarpanahi/openmed/pull/5732), run `python -m pytest tests/unit/core/test_disclaimer.py -q`. Check the notice and acknowledgement against the [human review sequence](../v3.1-clinician-review.md). |
| A restricted workflow enforces its required clinician scope and crisis routing. | Compare the restricted obligations in `openmed/core/intended_use.py` with the [clinician scope](https://github.com/maziyarpanahi/openmed/issues/5212) and [routing](https://github.com/maziyarpanahi/openmed/issues/5214) contracts. For each kit under review, record a local test exercising refusal and one exercising the permitted path. An absent primitive leaves the claim open. |
| The pack contains the mandatory lanes and metrics for the selected capability and tier. | Run `python -m pytest tests/unit/eval/test_evidence_pack_lanes.py -q`. Compare the resulting profile with the candidate pack. See the [capability pack format and commands](../../release/v3.0-journey-release-gate.md#capability-validation-evidence-packs). |
| Pack authentication and freshness are reproducible. | Run `openmed evidence verify --help`, then verify the candidate pack using the trusted key, an explicit timestamp and the site's age limit. Preserve the exit status, reason codes and digest. See the [freshness policy](../evidence-freshness.md). |
| Corrective actions affecting this capability and version are traceable. | Inspect `openmed/core/schemas/json/corrective_action.schema.json` from the [corrective action change](https://github.com/maziyarpanahi/openmed/pull/5727). Reconcile open records with capability, version and evidence digest before sign-off. See the [waiver ledger](../waiver-ledger.md). |
| A proposed LTS patch follows the supported backport policy. | Read `docs/release/lts-backport-policy.md` in the candidate and compare the change with the eligibility rules, review and evidence requirements in the [4.3 backport policy](https://github.com/maziyarpanahi/openmed/pull/5711) and [release channels](../../release/semver-and-channels.md). |
| The deployment's trust boundaries and data flows match those assessed. | Start with `docs/security/threat-model.md`. Compare the [service](../../security/service-threat-model.md), [agent](../../security/agent-threat-model.md), [Journey](../../security/journey-threat-model.md), [clinical brief](../../security/clinical-brief-threat-model.md), [MCP](../../security/mcp-threat-model.md) and [on-device kit](../../security/openmedkit-threat-model.md) models with the components the site actually enables. |
| The technical release evidence covers each deployed surface. | Compare the [agent assurance pack](../v3.1-agent-assurance.md), [service gate](../../release/v4.0-service-release-gate.md) and [kit gate](../../release/v4.0-kit-release-gate.md) with the deployment inventory. Run `python -m pytest tests/unit/compliance/test_agent_assurance_pack.py -q` for the agent pack. |

Follow the [offline build, signing and verification example](../../release/validation-evidence-packs.md) to reproduce the complete pack workflow on synthetic inputs.

## Recording the review

Keep the marked checklist with the candidate's evidence digests, typed failure
reasons and corrective-action identifiers. Keep source records, private paths,
credentials and reviewer identities out of shared evidence. Revisit affected
claims when the intended use, model, policy, pack or deployment changes.
The [4.3 milestone](https://github.com/maziyarpanahi/openmed/milestone/23)
tracks assurance work still under review.
