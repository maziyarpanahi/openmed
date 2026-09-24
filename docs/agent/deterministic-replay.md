# Deterministic Agent Replay Verification

`openmed.agent.audit.replay_verifier` checks whether frozen evidence still
reconstructs the signed semantic artifact chain of an agent run. It does not
invoke tools, models, adapters, clocks, or network services. The verifier uses
the append-only action ledger from #2766 and signed capability grants from
#2761. End-of-run invariant checks remain the separate scope of #3001.

At capture time, retain a private copy of each tool contract, policy snapshot,
tool response, and model configuration. Pass these as `FrozenReplayEvidence`
objects in proposal order to `capture_replay_step`, supplying the same local
key used to sign the manifest as `commitment_key`. Each step commits the signed
grant and tool contract with SHA-256, and private policy, response, and model
bytes with domain-separated HMAC-SHA-256. It also binds a capture timestamp,
the opaque action ID, and the preceding semantic artifact digest. Sign the ordered step
list and the ledger head with `SignedReplayManifest.sign`, using an
application-owned local key of at least 32 bytes. Anchor the signed manifest
and ledger head in trusted storage; a local hash chain by itself cannot detect
whole-chain replacement or truncation.

At verification time, load and verify the complete action ledger, retrieve the
frozen byte inputs in the same order, and call `verify_replay` with the local
manifest key and a `CapabilityGrantVerifier` configured with the grant signing
key. The capture timestamp is supplied explicitly to grant verification so
expiry checks are deterministic. The verifier first authenticates the manifest,
ledger, and all grants. It refuses missing evidence, a changed ledger anchor,
or a mismatched run or plan. Its result is either a match or the first
divergence, with zero-based step, artifact field, and expected/actual hashes.

The frozen inputs stay in caller memory. Do not log, persist in audit reports,
or commit them as fixtures if they contain clinical content. The module's
errors use fixed codes, and reports contain hashes only. Keep
private evidence in operator-controlled encrypted storage and restrict access
to the signing keys. A response digest proves byte equality with a captured
response; replay does not independently re-execute a nondeterministic model or
prove that the original tool result was clinically correct. It also does not
establish end-of-run safety invariants or a compliance certification.

Run the focused offline check with:

```bash
.venv/bin/python -m pytest tests/unit/agent/audit/test_replay_verifier.py -q
```
