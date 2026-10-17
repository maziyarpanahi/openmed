# Synthetic image/text pair provenance

`SyntheticPairManifest` links image, text and label artifacts to generator
version/digest, source SPDX declarations, source digests, reproducible
transformation chains and an explicit validation outcome/digest. Missing or
nonpermissive declarations and disconnected chains are rejected before a manifest
can be constructed. Each transform records input/output, tool and configuration
digests; prompts, payloads and private paths are not stored.

`to_json()` is canonical and byte-stable. `manifest_digest` detects metadata
changes; it is not a signature, authorization, license determination or evidence
that the declarations are truthful. `require_publishable()` refuses a failed
validation outcome, and `verify_pair_artifacts()` checks the bytes of all three
members offline against their declared hashes. Passing either check does not
publish anything or replace owner review.

Supply synthetic assets only. SPDX normalization reuses the maintained synthetic
license-identifier policy. Provenance, source rights, validator adequacy and the
actual reproducibility environment must be established outside this manifest.
Artifact verification accepts at most ten megabytes per member and never returns
source content in diagnostic messages. No model downloads or network calls occur.
