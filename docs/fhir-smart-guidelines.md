# WHO SMART Guidelines Profile Checks

OpenMed can run a focused, offline conformance check over exported FHIR R4
Bundles before they are handed to an implementation-guide-aware server. The
checker reads profiles from a local FHIR IG npm package and validates only
resources that declare those profiles in `meta.profile`.

This is a targeted export safety check, not a replacement for a full FHIR
validator. It supports required and maximum cardinality, fixed values, locally
enumerable bindings, and mandatory `identifier`/`category` slices described by
fixed or pattern discriminators. FHIRPath invariants, remote terminology
expansion, and other unsupported constraints are reported as informational
`OperationOutcome` issues instead of failing or making a network call.

## Fetch an IG package

Choose the published version of the WHO SMART Guidelines implementation guide
used by the receiving system. From that guide's publication or version-history
page, copy the link to its `package.tgz` artifact. Keep the version and checksum
in deployment configuration so the conformance input is reproducible.

```bash
export SMART_IG_PACKAGE_URL="https://example.org/path/to/package.tgz"
export SMART_IG_DIR="./vendor/smart-ig"

mkdir -p "$SMART_IG_DIR"
curl --fail --location "$SMART_IG_PACKAGE_URL" --output /tmp/smart-ig-package.tgz
tar -xzf /tmp/smart-ig-package.tgz -C "$SMART_IG_DIR"
test -d "$SMART_IG_DIR/package"
```

The checker accepts either the directory containing `package/` or the
`package/` directory itself. It reads the local JSON resources only. OpenMed
does not bundle or redistribute the WHO implementation guides.

Prefer a package containing generated `StructureDefinition.snapshot` elements.
When only a differential is available, the checker evaluates only the
constraints explicitly present there; it does not fetch or merge the profile's
`baseDefinition`. Partial or non-enumerable value-set content is reported as
informational instead of being treated as proof that a code is invalid.

## Check a Bundle

`check_bundle()` returns a FHIR R4 `OperationOutcome`. A conformant Bundle has
the standard informational `No issues detected.` result. Violations use
FHIRPath-style expressions such as
`Bundle.entry[0].resource.name[0].family`.

```python
from openmed.clinical.exporters.fhir import check_bundle, to_bundle

bundle = to_bundle(resources, doc_id="encounter-123")
outcome = check_bundle(bundle, "./vendor/smart-ig")

blocking = [
    issue
    for issue in outcome["issue"]
    if issue["severity"] in {"fatal", "error"}
]
if blocking:
    raise ValueError("FHIR export does not satisfy its declared profiles")
```

The Bundle exporter also has an opt-in `profile_check=` callback. The callback
receives a deep copy of the completed Bundle, so it can collect an outcome or
raise a policy error without mutating the export. Its return value is ignored.

```python
outcomes = []


def profile_gate(candidate):
    outcome = check_bundle(candidate, "./vendor/smart-ig")
    outcomes.append(outcome)
    if any(issue["severity"] == "error" for issue in outcome["issue"]):
        raise ValueError("FHIR profile check failed")


bundle = to_bundle(
    resources,
    doc_id="encounter-123",
    profile_check=profile_gate,
)
```

Omitting `profile_check` preserves the existing Bundle export behavior.

## Audit de-identification effects

Pass the pre-de-identification Bundle as `original_bundle` to distinguish
violations introduced by de-identification from constraints the source export
already violated.

```python
from openmed.clinical.exporters.fhir import check_bundle
from openmed.interop.fhir_operations import de_identify_bundle

original = bundle
deidentified = de_identify_bundle(original, method="remove")
outcome = check_bundle(
    deidentified,
    "./vendor/smart-ig",
    original_bundle=original,
)
```

The diagnostics use one of these prefixes:

- `De-identification introduced profile violation` means the corresponding
  profile constraint passed before de-identification and fails afterward.
- `Pre-existing profile violation` means the same constraint failed in both
  Bundles.

If removal empties a profile-required element, adjust the de-identification
policy for that path. Prefer a policy-approved pseudonym or surrogate over
retaining the original value. For example, a required patient name can remain
structurally present as a consistent pseudonym while the raw identifier is
still removed. Re-run the post-de-identification check and the normal privacy
gates after changing policy.

## Privacy and interpretation

Checker diagnostics identify the constraint kind and structural expression;
they never quote the actual or expected element value. This keeps the
`OperationOutcome` safe to route through normal audit and observability paths,
subject to the same access controls as other export metadata.

An informational `not-supported` issue means the local checker deliberately did
not evaluate that constraint. Use the receiving program's full validator when
complete FHIRPath invariant, terminology, or slicing-discriminator coverage is
required.
