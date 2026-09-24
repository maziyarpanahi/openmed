# Governed clinical reference workflow

This repository-only example exercises the v3.1 authority, review, write,
evidence, replay, and recovery contracts together. Every record, identifier,
clock, key, and adapter response is synthetic. It runs offline and does not
contact a FHIR server, commit to a production OMOP database, or make a clinical
decision.

From the repository root:

```bash
.venv/bin/python -m pytest tests/integration/agent/test_governed_clinical_workflow.py -q
.venv/bin/python -m examples.agent.governed_clinical_workflow
```

The example prints only digests, counts, categorical outcomes, and a
content-free review packet. It deliberately uses the test fixture adapter;
it is not a deployable write client.

## Contract exercised

1. A signed, time-limited capability grant authorizes one exact tool request.
   Minimum-data planning selects only the reviewed `status` field.
2. A keyed side-effect preview covers the intended FHIR and staged OMOP
   mutations. A single-use approval token binds to that exact preview before
   tool dispatch. Expired or repeated tokens fail before dispatch.
3. The FHIR conditional-write plan requires a complete, unique match and a
   fresh version precondition. The OMOP batch validates its references and
   binds its staged preview to the approval receipt.
4. A content-free action ledger and signed frozen replay manifest bind the
   same grant and tool contract to the final effect evidence. Replay does not
   call the tools again.
5. A durable checkpoint identifies local, FHIR, and OMOP effects by stable
   idempotency keys. An injected restart before or after any effect produces
   the same committed effects and evidence without a duplicate commit.
6. A rollback manifest is produced for review. It is a proposal, **not** an
   automatic clinical rollback. A partial FHIR batch likewise yields a
   content-free compensation packet requiring human review.

The test injects approval expiry, duplicate delivery, stale FHIR evidence,
partial batch failure, and interruption at every effect boundary. Unsafe
mutations stop with categorical reason codes. No raw clinical value is written
to the ledger, checkpoints, review packet, or example output.

This proves the contracts with synthetic adapters only. Production EHR
credentials, patient data, autonomous clinical decisions, and unreviewed
compensating writes remain out of scope.
