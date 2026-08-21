# Extension plugin SDK stability policy

OpenMed's extension SDK lets a separately installed Python distribution provide
recognizers, anonymizer providers, exporters, interop adapters, and language
packs without a core-code change. The public SDK lives in `openmed.plugins` and
uses canonical `OpenMedSpan` records at every span-producing boundary.

Importing `openmed` does not import `openmed.plugins`, enumerate entry points,
or import plugin dependencies. Discovery is local, lazy, and process-scoped.
Installing a plugin is a user decision; discovery never downloads or installs
packages.

## Package and entry-point contract

Plugin distributions publish one or more components through the stable
`openmed.plugins` entry-point group:

```toml
[project.entry-points."openmed.plugins"]
example = "example_openmed_plugin:plugin_components"
```

The referenced object may be one component, a zero-argument factory, or an
iterable of components. Each component exposes static `metadata` as a
`PluginComponentMetadata` instance or mapping. A minimal recognizer factory is:

```python
from openmed.core.schemas.span import OpenMedSpan


class ExampleRecognizer:
    metadata = {
        "plugin_id": "example-openmed-plugin",
        "component_id": "example-recognizer",
        "kind": "recognizer",
        "sdk_version": "1.0.0",
        "license": "Apache-2.0",
        "network_egress": False,
        "labels": ("PERSON",),
        "languages": ("en",),
        "metadata": {"stage": "fast_pii"},
    }

    def recognize(self, text: str, **kwargs) -> tuple[OpenMedSpan, ...]:
        return ()


def plugin_components() -> tuple[ExampleRecognizer, ...]:
    return (ExampleRecognizer(),)
```

### Stable metadata

| Field | Contract |
|---|---|
| `plugin_id` | Non-empty stable distribution identifier. It must not contain `:`. |
| `component_id` | Non-empty identifier unique within the plugin. It must not contain `:`. |
| `kind` | One supported component kind from the table below. |
| `sdk_version` | Semantic version of the SDK contract targeted by the component. |
| `license` | SPDX-like license expression. Unknown or restricted expressions require opt-in. |
| `network_egress` | Boolean declaring whether the component may make network calls. |
| `labels` | Canonical OpenMed labels emitted or handled by the component. Recognizers and anonymizer providers must declare at least one. |
| `languages` | Normalized language tags supported by the component, or `*`. |
| `name` and `description` | Optional human-readable component information. |
| `metadata` | Optional static, non-PHI mapping. Recognizers may select `deterministic`, `fast_pii`, or `clinical_phi` with `stage`. |

The stable qualified identifier is `plugin_id:component_id`. Registry reports,
detector provenance, interop registrations, and MCP tool documents retain this
identifier without including input text.

## Component protocols

The versioned protocols are defined in `openmed.plugins.protocols`.

| Kind | Required public behavior |
|---|---|
| `recognizer` | `recognize(text, **kwargs)` returns canonical `OpenMedSpan` values with offsets into `text`. |
| `anonymizer_provider` | `replacement_for(span, surface, **kwargs)` returns replacement text without retaining or logging `surface`. |
| `exporter` | `export(spans, **kwargs)` returns text, bytes, or structured records without adding source surfaces. |
| `interop_adapter` | `to_openmed_spans(payload, **kwargs)` and `from_openmed_spans(spans, **kwargs)` translate through canonical spans. |
| `language_pack` | `language_code()` and `canonical_labels()` declare routing and span capabilities. |

Recognizer and interop outputs must use valid character offsets, canonical
labels, finite scores, and privacy-safe evidence and metadata. Components must
not persist the source text or raw PHI. OpenMed rewrites document identity,
text hashes, and recognizer provenance before pipeline arbitration.

### Anonymizer provider example

An anonymizer provider is routed by its declared canonical labels and
languages. It receives the source span and surface only while replacement is
running. The configured Faker instance and resolved locale are available in
keyword arguments, so seeded anonymizers remain deterministic:

```toml
[project.entry-points."openmed.plugins"]
example = "example_openmed_plugin:plugin_components"
```

```python
class ExampleIdProvider:
    metadata = {
        "plugin_id": "example-openmed-plugin",
        "component_id": "example-id-provider",
        "kind": "anonymizer_provider",
        "sdk_version": "1.0.0",
        "license": "Apache-2.0",
        "network_egress": False,
        "labels": ("ID_NUM",),
        "languages": ("en",),
    }

    def replacement_for(self, span, surface, **kwargs):
        faker = kwargs["faker"]
        return f"SYN-{faker.random_int(min=100000, max=999999)}"


def plugin_components():
    return (ExampleIdProvider(),)
```

The replacement must be a non-empty string and must not contain the source
surface. Exceptions and invalid replacements produce a PHI-safe warning and
fall back to the built-in generator. When multiple accepted providers cover
the same label and locale, the lowest qualified component id wins.

## Discovery, quarantine, and policy opt-in

Call `openmed.plugins.discover_plugins()` to receive a
`PluginDiscoveryResult`. Accepted components appear as `PluginRegistration`
records. A broken or incompatible component is isolated as a
`PluginQuarantineRecord`; it does not crash discovery or prevent other plugins
from loading.

Quarantine records expose a stable `reason`, a safe `message`, entry-point and
component identifiers when available, and detached static metadata. Reasons
include invalid metadata or labels, unknown component kinds, duplicate ids,
load failures, SDK-major mismatches, and local-first policy rejections.

The default policy auto-loads only components that declare no network egress
and whose complete license expression is permissive. Opt-in is explicit and
local to the call:

```python
from openmed.plugins import discover_plugins

result = discover_plugins(
    opt_in_plugins=("example-openmed-plugin:remote-exporter",),
)
```

Callers may instead use `allow_network_egress=True` or
`allow_non_permissive_licenses=True` when they intentionally accept every
plugin in that policy class. These flags do not change the safe process
default.

## Runtime integration

Validated recognizers are adapted into `DetectorSpec` records on the first
detector lookup. Their spans run in the declared pipeline stage and participate
in the same arbitration, provenance rewriting, and privacy-safe metadata
filtering as first-party spans.

Validated anonymizer providers are adapted into the label-generator registry
on the first `Anonymizer.surrogate()` lookup. Discovery is process-scoped and
idempotent. A caller that intentionally enables a policy-restricted provider
can wire it before first use:

```python
from openmed import discover_anonymizer_provider_plugins

discover_anonymizer_provider_plugins(
    opt_in_plugins=("example-openmed-plugin:example-id-provider",),
)
```

Validated exporters and interop adapters are registered lazily with:

```python
from openmed.interop import available_adapters, get_adapter

names = available_adapters(include_plugins=True)
exporter = get_adapter("example-openmed-plugin:example-exporter")
```

A component may also expose `openmed_tools` declarations built from
`openmed.mcp.tool_registry.PluginTool`. Valid plugin tools appear beside
first-party tools on the MCP registry's first lookup, retain plugin provenance
in rendered tool documents, and route through the registered handler.

The older `openmed.detectors` / `DetectorSpec` entry point remains a supported
compatibility surface for detector-only packages. New multi-component packages
should use `openmed.plugins` so SDK-version, label, license, and network policy
validation happens before runtime registration.

## Local-first requirements

Every plugin must preserve OpenMed's privacy defaults:

- no telemetry or background network calls by default;
- no automatic model, package, or dataset download during import or discovery;
- no raw PHI in logs, exceptions, caches, temporary files, traces, or exports;
- remote services remain disabled until the caller explicitly opts in;
- credentials and restricted datasets remain user-supplied and are not bundled;
- optional dependencies fail clearly only when their component is selected.

Plugin code executes in the OpenMed process after installation and opt-in. The
registry validates compatibility and policy declarations; it is not a sandbox
for untrusted code.

## Conformance kit and example package

`examples/openmed-plugin-example` is a copyable Apache-2.0 distribution with a
deterministic toy recognizer, a privacy-safe exporter, installed entry-point
metadata, and synthetic self-certification tests. Run its conformance gate
offline from the repository root:

```bash
PYTHONPATH="examples/openmed-plugin-example/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m openmed.plugins.conformance \
  openmed_example_plugin:plugin_components

PYTHONPATH="examples/openmed-plugin-example/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python -m pytest \
  examples/openmed-plugin-example/tests/test_conformance.py -q
```

The example's malformed synthetic fixture fails with the stable reason
`invalid_metadata` and message `network_egress must be a boolean`. The
conformance kit exercises each component protocol with synthetic values and
does not enumerate installed packages, open sockets, or persist source text.

## Semantic-versioning policy

`PLUGIN_SDK_VERSION` follows Semantic Versioning. The registry accepts valid
plugin SDK versions with the supported major version and quarantines a
different major as `protocol_version_mismatch`.

The following changes require an SDK major-version increment unless a prior
deprecation path preserves compatibility:

- renaming or removing the `openmed.plugins` entry-point group;
- removing or renaming a stable metadata field or component kind;
- making an optional field required or changing its accepted value shape;
- changing a required component method, arguments, or return contract;
- changing `OpenMedSpan` offset or canonical-label semantics;
- rejecting a component shape or policy declaration that was previously valid;
- removing stable quarantine reason codes or plugin provenance fields;
- weakening local-first, no-telemetry, or no-raw-PHI defaults.

Additive component kinds, optional fields with compatible defaults, canonical
labels, safe return conveniences, and runtime bridges may ship in an SDK minor
release. Documentation corrections and stricter rejection of inputs that were
already invalid may ship in a patch release.

Deprecated fields remain available for at least two minor releases, emit a
`DeprecationWarning`, identify their replacement, and appear in the changelog
before removal.

Before widening an OpenMed dependency range, plugin authors should run the
conformance kit against the oldest and newest supported OpenMed releases, test
installed entry-point discovery, cover positive, negative, overlap, Unicode,
and invalid-span inputs, and confirm that offline execution emits no source
surface in logs or artifacts.
