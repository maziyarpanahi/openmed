# Reproducible synthetic release capsules

`build_release_capsule` bundles actual controlled generator configurations,
versions, seed commitments, provenance digests, privacy/utility evidence outcomes,
permissive SPDX declarations and artifact hashes/sizes in canonical JSON. Artifact
content, raw prompts, private paths and reference records are not embedded.
Equivalent generator/artifact ordering produces identical bytes.

`verify_release_capsule` requires the matching local artifact bytes, reconstructs
the strict canonical capsule and refuses missing, extra, changed or unlicensed
assets and altered metadata. Verification is offline. Failed or unevaluated
checks remain explicit and never become a passing qualification.

The verifier distinguishes `external_model_required` from a recipe whose
 dependencies are declared locally. It always reports `generation_reproduced=false`:
verifying commitments does not regenerate a corpus. Retain local seed material,
generator code, pinned runtime and authorized dependencies to perform independent
regeneration. Commitments are not signatures or proof that declarations are true.
Passing integrity checks does not authorize publication or establish privacy,
clinical utility, source rights or a release gate.

Each capsule has bounded generators, artifacts and scalar configuration options.
String options must be controlled codes; approved vocabularies and independent
provenance review remain caller responsibilities. Only explicitly synthetic
corpora belong in this workflow; metadata checks cannot identify real records.
