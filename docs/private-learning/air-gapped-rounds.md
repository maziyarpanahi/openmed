# Signed air-gapped rounds

Use `build_round_bundle` to create a deterministic Ed25519-signed offline transfer
containing explicitly reviewed code bytes, required synthetic self-test code IDs,
model digest references, an existing federated schedule and a dense update policy.
Code identifiers are bounded metadata codes, never filesystem paths. No model
weights, datasets, cache directories or credentials are included by this contract.

`verify_round_bundle` requires the public trust anchor and independent expected
code digests, model references, update policy, round reference and self-test IDs.
It verifies the actual signature and exact policy, checks every content digest,
validates the existing schedule contract, and returns deeply immutable data.
The public key is supplied by the caller; a bundle cannot nominate its own trust
anchor. Verification does not execute the included code or synthetic self-tests.
An authorized isolated runner must execute the required tests before training.

`build_return_bundle` admits exactly one dense update and existing aggregate
metric envelopes. The return is signed and bound to the exact input bundle digest.
`verify_return_bundle` independently checks that binding, expected update policy
and allowed metric IDs. Actual little-endian tensor bytes are checked against the
declared digest, parameter ordering, dimensions and dtype byte widths. Only finite
float32 and float64 values are supported by this transfer version. Existing metric
suppression is retained, including null values for small cohorts. Extra fields,
unexpected records, caches and arbitrary files have no export field.

Each transfer is at most one MiB. Code or update content is at most 256 KiB, with
at most 32 code assets/model references/self-tests/aggregate metrics. JSON keys
cannot be duplicated; schema version, direction, canonical encoding and signature
are checked. All outward construction/verification errors use a fixed category
without retaining parser excerpts or callback exception context.

`transfer_receipt` returns only the byte count and SHA-256 digest. It does not
verify signatures or approve privacy. Treat full transfer data as private: code
and tensor contents are intentionally retrievable by the authorized receiver.
Signing proves integrity relative to the supplied key, not that code is safe,
self-tests have passed, clipping occurred, differential privacy is qualified or
an update is free of memorized sensitive content. Review source/self-tests and
perform the required independent privacy/accounting checks before transfer.
This format performs no downloads, extraction, filesystem traversal, network call,
automatic training, clinical action or trust-key management.

```sh
.venv/bin/python -m pytest tests/unit/training/federated/test_airgap_bundle.py -q
```
