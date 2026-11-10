# Evidence-grounded query planner

OpenMed can turn a bounded patient or cohort evidence request into a closed set
of read-only Journey, cohort, measure, registry, or SQL calls. Values must come
from those calls. The planner does not generate clinical facts, and its answer
composer does not accept free-form model output.

The result is an evidence report, not clinical advice. It cannot authorize a
diagnosis, treatment, prescription, enrollment, outreach, order, write, or
other state change.

## Build a bounded request

```python
from openmed.agent.workflows import (
    BoundedEvidenceQuery,
    BoundedQueryOperation,
    EvidenceTool,
    QueryIntent,
    QueryScope,
    plan_evidence_query,
)
from openmed.clinical.journey_contracts import derived_opaque_id

query = BoundedEvidenceQuery(
    query_id=derived_opaque_id("query", "synthetic-example"),
    query_text="Show the current condition evidence.",
    intent=QueryIntent.EVIDENCE_RETRIEVAL,
    scope=QueryScope.PATIENT,
    namespace="clinical",
    purpose="care_review",
    subject_id=derived_opaque_id("patient", "synthetic-subject"),
    operations=(
        BoundedQueryOperation(
            operation_id="read_journey",
            tool=EvidenceTool.JOURNEY,
            resource_id="current",
            fields=("condition",),
            limit=5,
        ),
    ),
    max_results=20,
)

plan = plan_evidence_query(query)
```

Query text is transient and absent from the serialized plan. SQL is also
transient; a serialized SQL call carries only its digest. Every emitted call
has `read_only=true`, a maximum result count, selected fields, namespace,
purpose, and—when applicable—an opaque subject identifier.

The planner refuses explicitly declared state changes and clinical advice. A
small fail-closed lexical guard also catches direct requests to diagnose,
prescribe, recommend treatment, contact a patient, or place an order. Refused
plans contain no tool calls.

## Bounded SQL

`validate_bounded_read_only_sql()` accepts a single bounded `SELECT` from one
explicitly allowlisted view, with plain identifier projections. A tool call
must project exactly its declared fields from its declared resource. It rejects:

- insert, update, delete, definition, privilege, execution, copy, and locking
  operations;
- multiple statements and wildcard selection;
- views outside the allowlist, joins, comma-separated relations, subqueries,
  CTEs, quoted identifiers, and functions; and
- missing, non-literal, zero, or excessive outer `LIMIT` values and excessive
  offsets.

Allowlisted views may be unqualified or use the `openmed` schema. Other schema
qualifiers are rejected. Functions are excluded altogether: a blacklist cannot
prove that installed database functions are free of side effects.

String literals and comments are structurally removed before the statement is
classified, so SQL-looking text inside a literal remains data. The validated
SQL is available only on the in-memory tool call; it is not published in plan
JSON.

## Compose an evidence-only answer

Tool adapters return `EvidenceToolResult` objects. Every result includes a
versioned access decision and one explicit state: `success`, `partial`,
`empty`, `unknown`, `denied`, `conflict`, `unsupported`, or `failure`.
Successful and partial facts must reference citations present in the same tool
result. A patient-level citation must carry an exact Journey snapshot ID and
digest before its value can appear in an answer.

`compose_evidence_answer()` returns `answered` only when every planned call is
present, authorized, successful, mutually consistent, and cited. It otherwise
returns one of `insufficient_data`, `refused`, `access_denied`, `conflict`, or
`failure`, with no patient statements attached. Partial, empty, unknown,
unsupported, conflicting, denied, and failed tool states are never converted
into an apparent success.

Answer statements are deterministic renderings of structured tool values. Each
value is quoted as single-line JSON and marked with the `quoted_tool_value`
data boundary. HTML/XML and Markdown delimiters inside a value are Unicode
escaped. Text such as role markers, tool syntax, or “ignore previous
instructions” inside source artifacts remains quoted data and cannot create a
new call or statement.

## Optional explanation layer

`answer.to_explanation_context()` exposes only already-grounded statements,
their citation IDs, and an instruction to treat quoted values as untrusted
data. A generative layer may explain those statements but must not introduce a
new patient fact or remove its citations. OpenMed’s core answer remains the
deterministic cited artifact.

## Provenance and compatibility

Plans and answers use schema version `1.0.0` with `same_major` compatibility.
The bundled `evidence_query_plan.schema.json` and
`evidence_query_answer.schema.json` schemas cover the persisted surfaces.
Plans, calls, access decisions, citations, statements, tool results, and
answers have deterministic identifiers or digests. `EvidenceQueryAnswer` can
reload a serialized artifact and verifies nested access, citation, statement,
identifier, and complete-answer custody before accepting it.
