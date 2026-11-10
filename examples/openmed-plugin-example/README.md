# OpenMed example plugin

This copyable package shows the smallest complete OpenMed plugin distribution:
a deterministic toy recognizer, a privacy-safe exporter, Python entry-point
metadata, and an offline conformance test. It is an example for extension
authors, not a clinical model or medical device.

All runtime behavior is local. The components make no network calls, enable no
telemetry, and write no files. The recognizer responds only to the fictional
marker `OPENMED_SYNTHETIC_PERSON`; do not use real patient data in its tests.

## Self-certify offline

From the repository root, with OpenMed and its development dependencies already
installed:

```bash
PYTHONPATH="examples/openmed-plugin-example/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m openmed.plugins.conformance \
  openmed_example_plugin:plugin_components

PYTHONPATH="examples/openmed-plugin-example/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m pytest \
  examples/openmed-plugin-example/tests/test_conformance.py -q
```

Both commands use only committed source and synthetic values. They do not
enumerate installed plugins or access a package index. For an editable install
in a prepared offline environment, disable dependency resolution and build
isolation explicitly:

```bash
python -m pip install --no-index --no-deps --no-build-isolation -e \
  examples/openmed-plugin-example
```

## Entry-point contract

The distribution registers one zero-argument component factory in
`pyproject.toml`:

```toml
[project.entry-points."openmed.plugins"]
example = "openmed_example_plugin:plugin_components"
```

Each component exposes static metadata with SDK version `1.0.0`, an
`Apache-2.0` permissive-license declaration, `network_egress = false`, and the
canonical labels it handles. A recognizer returns `OpenMedSpan` objects whose
offsets refer to the supplied source text. The exporter serializes offsets,
hashes, labels, and safe provenance, never source surfaces.

## Adapt this package

1. Copy this directory and rename the project, import package, plugin id, and
   entry-point name.
2. Keep every `component_id` unique within the distribution.
3. Replace the toy methods while preserving the metadata and component method
   contracts.
4. Keep fixtures synthetic and ensure raw source surfaces never enter logs,
   span metadata, exporter artifacts, caches, or telemetry.
5. Run the conformance command before publishing.

`tests/fixtures/malformed_plugin.py` deliberately declares
`network_egress = "false"` as a string. The kit rejects it with the stable
reason `invalid_metadata` and the specific message
`network_egress must be a boolean`, illustrating the feedback plugin authors
receive.
