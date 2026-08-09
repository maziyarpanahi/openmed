# Standalone local redactor manifest

OpenMed includes a small, platform-neutral manifest for the local redaction
surface. It records the component and license boundary for a constrained
installation without importing the full collection of model, service, or
interoperability integrations.

The manifest is intentionally descriptive. It does not install packages,
inspect the active environment, resolve entry points, load a model, or make a
network request. Print the canonical JSON from a checkout with:

```bash
python packaging/standalone_manifest.py > standalone-manifest.json
```

The output is deterministic, so the same source revision produces the same
bytes on every supported platform. It contains package metadata only; source
text, document identifiers, credentials, and model weights do not belong in
this file.

## Default boundary

The component plus the `dependencies.required` list form the complete default
install boundary:

- the local redactor component, licensed under Apache-2.0;
- `faker`, `jieba`, `pysbd`, and `pyyaml`, each recorded with its permissive
  license and bounded requirement.

The default entries are platform-neutral and declare no network egress. The
manifest does not turn model acquisition or remote inference on implicitly.
Any model or asset needed by a caller must be staged by that caller before an
offline run.

## Opt-in and excluded material

`dependencies.optional` describes integrations such as local model runtimes and
interoperability adapters. These entries are metadata only and are never
returned by the default dependency view. Selecting one is an explicit caller
choice; the Hugging Face client is marked as potentially network-capable and is
not part of the offline default.

`dependencies.restricted` and `restricted_assets` document material that the
standalone bundle must not install or ship. GPL bridges remain subprocess-only,
and restricted terminology or evaluation corpora remain user-supplied and
outside the repository. The manifest records their exclusion without copying,
embedding, or resolving them.

## Privacy-safe diagnostics

Manifest validation reports only structural field paths and fixed reasons. It
does not include values supplied by a caller in exceptions. Keep descriptions,
license notes, and any generated reports static and non-sensitive; do not pass
raw notes or identifiers as manifest metadata.

This manifest describes an installation boundary, not a compliance
certification or a clinical decision guarantee. Redaction remains a local
privacy-support operation and must be reviewed for the caller's use case.
