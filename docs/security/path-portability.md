# Local resource path portability audit

Offline manifests can be copied between Linux, macOS, and Windows hosts. A
path that is harmless on one host can be interpreted differently on another
when it contains traversal, an absolute root, a Windows-reserved component,
normalization differences, or a case-fold collision.

`openmed.core.path_portability.audit_resource_paths` provides a local-only
audit for manifest resource paths. It accepts path strings or path-like values,
normalizes separators and Unicode text in memory, and never resolves or opens a
path. It performs no network call. Each audit accepts at most 10,000 paths;
each path is limited to 4,096 characters and 256 components. Inputs outside
those bounds fail with a value-free error.

```python
from openmed.core.path_portability import audit_resource_paths

report = audit_resource_paths(
    [
        "models/weights.bin",
        "models/../config.json",
        "Models/Tokenizer.json",
        "models/tokenizer.JSON",
    ]
)

if not report.is_clean:
    for finding in report.findings:
        print(finding.to_dict())
```

The report contains only normalized path fingerprints (`sha256:<digest>`),
issue categories, and occurrence counts. It does not retain source path text.
The output is sorted by fingerprint and serialized with canonical JSON, so the
same synthetic input produces the same result regardless of input iteration
order.

The supported issue categories are:

- `traversal`: a `..` component is present.
- `absolute_root`: a POSIX root, UNC root, drive-qualified path, or `file:`
  root is present.
- `reserved_component`: a Windows device name, invalid character, control
  character, trailing dot, or trailing space is present.
- `normalization_drift`: separators, dot components, duplicate separators,
  Unicode compatibility forms, or trailing separators changed during
  normalization.
- `case_fold_collision`: two distinct normalized paths have the same Unicode
  case-folded identity.

This audit is a portability signal for release review. It is not a compliance
certification and does not make clinical or filesystem decisions.
