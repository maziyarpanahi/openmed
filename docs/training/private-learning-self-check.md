# Offline private-learning self-check

Run `openmed training self-check` or `openmed training self-check --json` before
working with private-learning metadata in an air-gapped install. The command
runs synthetic fixtures only; it does not start a round, train, aggregate, charge
privacy budget, publish artifacts or promote a model.

`run_private_learning_check()` covers all five existing JSON Schemas with actual
local validation, ordered schedules and half-open phases, lifecycle transitions,
update allowlists/shape budgets/clipping declarations, suppressed aggregate
metadata, forbidden fields and parameter-schema fingerprint invariance. Checks
are independent: one exception or mismatch does not prevent remaining checks.
Schema reference retrieval is disabled. The existing `jsonschema` core dependency
is used; no training library, model, credential, dataset, network or sleep is
required. Synthetic timestamps and metadata are fixed rather than wall-clock based.

Output contains only runner/contract versions, closed check categories/statuses, synthetic fixture case
counts and deterministic receipt digests. It never copies model/client IDs,
patient counts, tensor/gradient values, examples, metrics, paths, endpoints or
runtime exception text. Fixture case counts describe the fixed planned probes, not the executed prefix
when a check fails and never patient
or participant populations. JSON uses the existing CLI envelope; exit status is
zero for all checks passing, one for a negative/unavailable check and two for
invalid CLI usage. Failed checks retain a `check_failed` outcome with every
independent result available. Text output contains the same closed statuses.

The report is a bounded metadata smoke check, not certification, sealed
scientific evaluation, a privacy guarantee or a substitute for the full suite.
Passing confirms the installed contracts agree with bundled synthetic cases;
operators still review policy, rollout evidence and real deployment conditions.
