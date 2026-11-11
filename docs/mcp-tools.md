# MCP tool reference

This page lists every built-in OpenMed MCP tool with its current schema version
and the behavior hints that the server advertises to MCP clients. To install
the server and connect a client, see [MCP client connections](mcp-clients.md).

The table is generated from `TOOL_REGISTRY` in `openmed/mcp/tool_registry.py`.
`tests/unit/mcp/test_tool_docs.py` renders it again from the registry and fails
when this page drifts, so update both together.

## Behavior hints

MCP clients use these hints to decide how to present a tool and when to ask
the user for confirmation. They describe intent; they are not a security
boundary.

| Column | MCP annotation | Meaning |
| --- | --- | --- |
| Read-only | `readOnlyHint` | The tool does not change runtime or durable state. |
| Destructive | `destructiveHint` | A state-changing call may remove or overwrite data. |
| Idempotent | `idempotentHint` | Repeating a call with the same arguments has no extra effect. |
| Open-world | `openWorldHint` | The tool may work with entities outside the request, such as an external terminology source. |

Tools that are not read-only are treated as state-changing. Their registry
schema document sets `authorization.consent_receipt_required` to `true`; see
[MCP consent receipts](security/mcp-consent-receipts.md).

## Tools

Optional tools contributed by SDK plugins are not listed here; they appear in
the MCP tool list only when their plugin is enabled.

<!-- mcp-tools-table:start -->
| Tool | Title | Version | Read-only | Destructive | Idempotent | Open-world |
| --- | --- | --- | :---: | :---: | :---: | :---: |
| `openmed_analyze_text` | Analyze Clinical Text | 1.0.0 | yes | no | yes | no |
| `openmed_brief` | Build Guarded Clinical Brief | 1.0.0 | yes | no | yes | no |
| `openmed_clinical_pipeline` | Plan Clinical Pipeline | 1.0.0 | yes | no | yes | yes |
| `openmed_decide` | Evaluate Fixed-Option Decision | 1.1.0 | yes | no | yes | no |
| `openmed_deidentify` | De-identify Clinical Text | 1.0.0 | yes | no | yes | no |
| `openmed_export_fhir` | Export Clinical Spans to FHIR | 1.0.0 | yes | no | yes | no |
| `openmed_extract_pii` | Extract PII and PHI | 1.0.0 | yes | no | yes | no |
| `openmed_fhir_bundle` | Assemble FHIR Bundle | 1.0.0 | yes | no | yes | no |
| `openmed_ground` | Ground Clinical Spans | 1.0.0 | yes | no | yes | yes |
| `openmed_ground_concepts` | Ground Clinical Concepts | 1.0.0 | yes | no | yes | yes |
| `openmed_list_models` | List Available Models | 1.0.0 | yes | no | yes | no |
| `openmed_list_pii_languages` | List PII Languages | 1.0.0 | yes | no | yes | no |
| `openmed_loaded_models` | List Loaded Models | 1.0.0 | yes | no | yes | no |
| `openmed_read_cohort` | Read Cohort | 1.0.0 | yes | no | yes | no |
| `openmed_read_dataset` | Read Dataset Snapshot | 1.0.0 | yes | no | yes | no |
| `openmed_read_journey` | Read Patient Journey | 1.0.0 | yes | no | yes | no |
| `openmed_read_measure` | Read Measure Results | 1.0.0 | yes | no | yes | no |
| `openmed_read_registry` | Read Clinical Registry | 1.0.0 | yes | no | yes | no |
| `openmed_read_trial_review` | Read Trial Review | 1.0.0 | yes | no | yes | no |
| `openmed_risk_report` | Score Re-identification Risk | 1.0.0 | yes | no | yes | no |
| `openmed_risk_score` | Score Clinical Re-identification Risk | 1.0.0 | yes | no | yes | no |
| `openmed_run_workflow` | Run Clinical Workflow | 1.0.0 | no | no | yes | no |
| `openmed_search_models` | Search Available Models | 1.0.0 | yes | no | yes | no |
| `openmed_signed_audit_report` | Generate Signed Audit Report | 1.0.0 | yes | no | yes | no |
| `openmed_unload_model` | Unload Model | 1.0.0 | no | no | yes | no |
| `openmed_workflow_preflight` | Governed Workflow Preflight | 1.0.0 | yes | no | yes | no |
| `openmed_workflow_preview` | Governed Workflow Preview | 1.0.0 | yes | no | yes | no |
| `openmed_workflow_request_review` | Governed Workflow Request Review | 1.0.0 | no | no | no | no |
| `openmed_workflow_status` | Governed Workflow Status | 1.0.0 | yes | no | yes | no |
<!-- mcp-tools-table:end -->
