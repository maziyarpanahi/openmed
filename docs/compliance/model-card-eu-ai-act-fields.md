# EU AI Act and GDPR Model-Card Field Specification

> **Template — requires legal review. Not legal advice.** Completing this block
> does not classify a system, demonstrate conformity, complete a GDPR DPIA, or
> replace the technical documentation required for a particular deployment.

This specification defines the EU AI Act / GDPR block for OpenMed model cards.
It gives model-card generation a stable target for evidence from `models.jsonl`,
`BenchmarkReport`, release-gate artifacts, and deployer-supplied governance
fields.

The provider and deployer remain responsible for deciding whether a system is
high-risk, determining their legal roles and obligations, performing conformity
assessment where required, and keeping the full technical documentation current.

## Rendering Rules

1. Render every field in the block. Use **Not evaluated**, **Not provided**, or
   **Not applicable — [reason]** rather than omitting a field.
2. Copy measured values from immutable evidence. Do not infer, average, round
   away a gate failure, or convert a missing value into zero.
3. Include the model, dataset or fixture, device, language, format, policy,
   threshold profile, timestamp, and artifact hash needed to interpret a metric.
4. Keep provider evidence separate from deployment decisions. Manifest and
   `BenchmarkReport` fields may be generated; intended workflow, legal basis,
   oversight, and key custody require deployer review.
5. Publish only aggregate, privacy-safe evidence. Never include raw clinical
   text, identifiers, prompts, cleartext spans, or reversible mappings.
6. Update the block after a material model, dataset, threshold, policy, format,
   intended-purpose, oversight, or robustness change.

## Model-Card Template Block

Copy this section into a model-card template. Values in `{{ double braces }}`
identify source fields; values in `[square brackets]` require an accountable
human owner.

```markdown
## EU AI Act / GDPR Compliance Fields

> **Template — requires legal review. Not legal advice.** This section is
> evidence for deployment review, not a conformity assessment or legal opinion.

### System and regulatory context

| Field | Value |
|---|---|
| Model and version | `{{ manifest.repo_id }}`; released `{{ manifest.released }}` |
| Artifact formats and hashes | {{ manifest.formats }}; {{ manifest.reproducibility_hash }} |
| Provider/deployer roles | [Provider, deployer, controller, and processor allocation] |
| AI Act classification | [High-risk / not high-risk / undecided, with rationale and owner] |
| GDPR processing role | [Controller / joint controller / processor / not applicable, with rationale] |
| Technical-documentation owner | [Role and record location] |
| Review date and status | [Date; draft / approved / rejected] |

### Intended purpose

| Field | Value |
|---|---|
| Model task and family | {{ manifest.task }}; {{ manifest.family }} |
| Intended clinical or operational purpose | [Specific workflow and expected output] |
| Intended users and affected persons | [Qualified users and populations] |
| Intended languages and environments | {{ manifest.languages }}; {{ manifest.tier }}; {{ benchmark.device }} |
| Integration and deployment mode | [On-device, local service, API, dependencies, and interfaces] |
| Permitted decisions | [How output may inform a decision] |
| Prohibited uses | [Unsupported, autonomous, or safety-critical uses] |

### Known limitations and reasonably foreseeable misuse

| Field | Value |
|---|---|
| Unsupported populations, languages, or inputs | [Measured gaps and exclusions] |
| Data and evaluation limitations | [Representativeness, sample size, labels, and known gaps] |
| Known failure modes | [False negatives, false positives, boundary errors, drift, and other failures] |
| Release-gate failures or blocked formats | [Gate, reason, affected format, and disposition] |
| Foreseeable misuse | [Scenarios and preventive controls] |
| Required local validation | [Deployment-specific tests and acceptance thresholds] |

### Accuracy and leakage metrics

| Metric | Value | Evidence context |
|---|---:|---|
| Exact-span F1 | {{ benchmark.metrics.exact_span_f1.f1 }} | {{ benchmark.suite }}; {{ benchmark.generated_at }} |
| Relaxed-span F1 | {{ benchmark.metrics.relaxed_span_f1.f1 }} | {{ benchmark.suite }}; {{ benchmark.generated_at }} |
| Character recall | {{ benchmark.metrics.character_recall.rate }} | {{ benchmark.fixture_count }} fixtures; {{ benchmark.device }} |
| Direct-identifier recall by label/language | {{ benchmark.metrics.recall_slices }} | [Critical labels and minimum approved threshold] |
| Character leakage | {{ benchmark.metrics.leakage.overall }} | [Overall, critical labels, and confidence interval] |
| Critical leakage count | {{ gate_report.critical_leakage_count }} | {{ gate_report.leakage_fixture_hash }} |
| Residual leakage rate and target | {{ gate_report.residual_leakage_rate }} / {{ gate_report.target_leakage_rate }} | [Gate decision and blocked formats] |
| Quantization recall delta | {{ gate_report.quant_recall_delta }} | [Format and reference precision] |

Metrics are estimates on the named fixtures, not guarantees of zero leakage or
clinical performance. Report **Not evaluated** when the applicable evidence is
absent.

### Human oversight

| Field | Value |
|---|---|
| Oversight owner and qualifications | [Role, training, and authority] |
| Review point | [Before use, before disclosure, sampled review, or other] |
| Information shown to reviewer | [Output, confidence, provenance, limitations, and safe audit evidence] |
| Escalation and abstention triggers | [Thresholds, unsupported inputs, drift, and gate failures] |
| Override and stop authority | [Who can reject output or suspend the system] |
| Automation restriction | [Decisions the model must not make without human review] |
| Oversight effectiveness review | [Metric, cadence, and owner] |

### Robustness, cybersecurity, and monitoring evidence

| Field | Value |
|---|---|
| Robustness suites | [OCR noise, typos, casing, multilingual, adversarial, or other suites] |
| Robustness results and failures | [Aggregate results, thresholds, and unresolved failures] |
| Evaluation and fixture hashes | {{ benchmark.metadata }}; [artifact hashes] |
| Device/format performance envelope | [Latency, memory, device, runtime, and format] |
| Integrity and provenance | {{ manifest.reproducibility_hash }}; [signature and dependency evidence] |
| Cybersecurity and local-processing controls | [Egress, signing, access control, logging, and patching] |
| Drift and incident monitoring | [Signals, cadence, thresholds, and response owner] |

### GDPR and de-identification evidence

| Field | Value |
|---|---|
| Processing purpose and lawful basis | [Article 6 basis and Article 9 condition, where applicable] |
| OpenMed policy profile | `[hipaa_safe_harbor or gdpr_pseudonymization]` |
| Audit residual-risk score | `{{ audit.residual_risk.risk_report_record_score }}` |
| Other audit risk indicators | `{{ audit.residual_risk.projected_leakage }}`; `{{ audit.residual_risk.risk_report }}` |
| Pseudonymization and key custody | [Not used, or mapping/key owner, separation, access, rotation, and deletion] |
| DPIA and DPA records | [Approved record ids, dates, owners, and review triggers] |
| Retention and deletion | [Inputs, outputs, audit evidence, mappings, logs, caches, and backups] |
| Data-subject rights process | [Notice and request-handling reference] |

Pseudonymized data remains personal data. The `hipaa_safe_harbor` profile does
not by itself establish anonymization under the GDPR.
```

## Field Sources and Ownership

| Section | Source of generated evidence | Human-owned completion |
|---|---|---|
| System context | `models.jsonl`: `repo_id`, `released`, `formats`, `reproducibility_hash` | Legal roles, classification, documentation owner, approval |
| Intended purpose | Manifest: `task`, `family`, `languages`, `tier`; `BenchmarkReport.device` | Workflow, users, populations, decisions, integration, prohibited uses |
| Limitations | Failed release gates and documented artifact limitations | Foreseeable misuse, deployment gaps, local-validation requirements |
| Accuracy/leakage | `BenchmarkReport.suite`, `fixture_count`, `generated_at`, `device`, and `metrics`; signed release-gate report | Metric applicability, thresholds, interpretation, disposition of failures |
| Human oversight | No safe default | Reviewer competence, timing, escalation, override, stop authority, effectiveness |
| Robustness | Robustness `BenchmarkReport` payloads, artifact hashes, manifest provenance | Applicable threats, acceptance thresholds, monitoring and incident response |
| GDPR | `AuditReport.policy`, `resolved_profile`, and privacy-safe `residual_risk` fields | Lawful basis, DPIA/DPA, retention, rights, recipients, and key custody |

### Accuracy and leakage minimums

For de-identification model cards, include at least:

- exact-span and relaxed-span F1 where applicable;
- character recall and direct-identifier recall, sliced by critical label and
  supported language;
- overall and critical-label character leakage, with confidence intervals when
  emitted;
- critical leakage count, residual leakage rate, target leakage rate, and the
  release decision;
- quantized-versus-reference recall delta for each published quantized format;
- fixture count, dataset or suite name, device, model/format, policy, threshold
  profile, generated timestamp, and evidence hashes; and
- every failed, waived, or not-evaluated gate without concealment.

### Human-oversight minimums

The oversight description must identify a natural-person role with sufficient
competence, authority, and information to understand limitations, detect
automation bias, interpret or disregard output, escalate uncertainty, and stop
use. A generic statement such as "human in the loop" is insufficient.

### Robustness minimums

Describe the relevant perturbation and adversarial suites, test population,
acceptance thresholds, results, failures, artifact hashes, device/runtime, and
monitoring plan. Separate measured robustness from architectural claims such as
local-only execution or signed evidence.

## Relationship to Other Records

The model-card block is a summary and index. Keep detailed evidence in the
versioned manifest, `BenchmarkReport` files, signed gate reports, audit reports,
DPIA, DPA, risk-management record, incident process, and deployment technical
documentation. The deployment owner should link those records rather than copy
personal data into a public model card.

## Primary References

- [EU AI Act, including Articles 11, 14, 15 and Annex IV](https://eur-lex.europa.eu/eli/reg/2024/1689/oj?locale=en)
- [GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- [OpenMed compliance posture](../compliance.md)
- [OpenMed DPIA template](dpia-template.md)
- [OpenMed DPA template](dpa-template.md)
- [OpenMed model manifest](../model-manifest.md)
- [OpenMed evaluation harness](../eval-harness.md)
