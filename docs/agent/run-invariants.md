# End-of-run safety invariants

`openmed.agent.run_invariants.check_run_invariants` checks typed, synthetic
metadata for a completed run. It performs no execution, approval, artifact
read, clinical correctness check, or network I/O. A report passes only when
its `findings` list is empty. Each failure remains visible independently;
passing one check never cancels another finding.

## Input contract

Build a `CompletedRun` from existing metadata contracts:

- an opaque `RunId`, append-ordered `EventReference` records, and the terminal
  event's sequence number;
- an `ActionNode` dependency graph and one `RunAction` per planned action;
- `RunTiming` and one `ActionTiming` per recorded action;
- a final `WorkflowOutcome` and, for failed or policy-denied outcomes, a
  categorical `ErrorEnvelope`; and
- optional content-free `ArtifactReference` declarations, linked by opaque
  artifact IDs to actions that produced them.

Each `RunAction` carries an `ActionCorrelation`, final phase, outcome, optional
error, produced artifact IDs, and a boolean review declaration. The phase
vocabulary is `queued`, `preflight`,
`ready`, `running`, `waiting-review`, `completed`, and `aborted`. Only the last
two are acceptable at the end of a run. A successful run requires all actions
to be completed. Abstention, reviewer handoff, policy denial, and execution
failure may end with aborted actions. The checker accepts successful actions
whose final phase is completed after external review, but the declaration and
final metadata alone do not prove the review occurred.

```python
from openmed.agent.run_invariants import check_run_invariants

report = check_run_invariants(completed_run)
if not report.is_valid:
    handle_incomplete_run(report.to_dict()["findings"])
```

`check_run_invariants` composes the existing event-sequence and action-graph
validators. It also checks run and action correlation, orphan and duplicate
actions, final phases, missing outcomes, error/outcome consistency, monotonic
timing, and whether every declared artifact has one successful producer.
Results are sorted stable reason codes in
`openmed.agent.run_invariants.v1` JSON. Reports contain no run or action IDs,
paths, prompts, arguments, tool output, clinical text, credentials, exception
messages, or tracebacks. Structural input failures raise `RunInvariantError`
with a fixed code. The module never logs or persists evidence.

The phase transition table and reviewed resume rule belong to the separate
action-phase contract in issue #2998. This checker validates the final phase
only; the execution adapter must validate each transition and review evidence
when that contract is available. This checker also does not replace the signed
action ledger or sealed evaluation contracts.

Run the focused tests with:

```bash
.venv/bin/python -m pytest tests/unit/agent/test_run_invariants.py -q
```
