# Public API Deprecation Policy

OpenMed treats its public Python API as a contract. Downstream clinical
integrations pin the package and expect imports and call signatures to keep
working across patch and minor releases. This policy defines what "public"
means, how the surface is protected in CI, and the process for making a
backwards-incompatible change when one is genuinely required.

## What counts as public

The supported public surface is exactly what the checker captures:

- Every name exported from `openmed.__all__`.
- Every name exported from the `__all__` of the public subpackages listed in
  `scripts/release/check_public_api.py` (`PUBLIC_SUBPACKAGES`): `openmed.core`,
  `openmed.processing`, `openmed.utils`, `openmed.clinical`, `openmed.eval`,
  `openmed.risk`, `openmed.interop`, `openmed.structured`, `openmed.mlx`,
  `openmed.multimodal`, `openmed.ner`, `openmed.zero_shot`, and
  `openmed.compliance`.
- Every name exported from a nested `openmed.multimodal.*` package declaring
  `__all__` in its `__init__.py`. Discovery walks the checkout in sorted order
  without importing candidates. Any underscore-prefixed package component
  excludes the entire private subtree. Plain `.py` modules and packages without
  `__all__` are excluded; public children of an export-free namespace are still
  discovered. Optional-dependency shims that cannot import are skipped during
  capture, using the same partial-install behavior as the fixed package list.
- For every public callable (function or class), its parameter names, order,
  and which parameters are optional.

Anything not reachable through one of those `__all__` lists -- private modules,
underscore-prefixed names, backend shims, and internal helpers -- is not
covered and may change without a deprecation cycle. If you want a symbol to be
supported, add it to the relevant `__all__`.

## Compatible vs. breaking changes

The checker (`scripts/release/check_public_api.py`) classifies every diff
against the committed baseline (`scripts/release/public_api_baseline.json`).

**Backwards compatible (never fails CI):**

- Adding a new exported name.
- Adding a whole new public module.
- Appending a new positional parameter that has a default value, or adding an
  optional keyword-only parameter.
- Widening a callable with `*args` / `**kwargs`.
- Changing a parameter's default *value* (the baseline records only whether a
  default exists, not what it is).

**Breaking (fails CI unless announced):**

- Removing an exported name.
- Changing a member's kind (for example a function becoming a class).
- Removing or renaming a parameter.
- Reordering positional parameters.
- Making a positional-or-keyword parameter positional-only or keyword-only, or
  inserting an optional positional parameter before an existing one.
- Making a previously optional parameter required, or adding a new required
  parameter.

## Deprecation cycle

Backwards-incompatible changes are allowed, but only deliberately and with
warning. The required cycle is:

1. **Announce.** In the release that first deprecates a symbol, keep it working
   but emit a `DeprecationWarning` and document the replacement in the
   changelog and the affected docstring.
2. **Wait at least one full minor release and respect SemVer.** Give downstream
   users a complete minor-version warning window. A breaking removal or
   signature change may ship only in the next major release; the warning window
   is not permission to break a later minor release. See
   [Release Streams, SemVer, and Channels](../release/semver-and-channels.md).
3. **Record the intentional break.** When you finally remove or change the
   symbol, add an entry to `scripts/release/public_api_allowlist.json` under
   `announced_breaks`, keyed by the `module.name` location the checker reports,
   with a short justification (for example the deprecation version and the
   replacement). This is how you tell CI the break is sanctioned.
4. **Regenerate the baseline.** Run the update command below so the baseline
   reflects the new surface. Once the removed name is gone from the baseline,
   delete its now-stale allowlist entry in the same or a follow-up PR.

## Working with the checker

Run the check locally exactly as CI does:

```bash
python scripts/release/check_public_api.py
```

It exits non-zero and prints the offending symbols when it finds an unannounced
breaking change. Additions are reported for visibility but never fail.

When you have legitimately added public surface, or completed an announced
deprecation, regenerate the baseline and commit it:

```bash
python scripts/release/check_public_api.py --update
```

Review the resulting diff to `scripts/release/public_api_baseline.json` in your
PR: additions should look intentional, and any removals or signature changes
must correspond to an `announced_breaks` allowlist entry.

### Landing v3.2 multimodal subpackages

When a streaming, vision, or other nested multimodal package lands, declare its
supported exports in `__init__.py` (prefer a literal list or tuple). No addition
to `PUBLIC_SUBPACKAGES` or top-level re-export is required. Keep optional runtime
imports lazy so discovery and signature capture do not load models or require
network access. A dynamic top-level `__all__` assignment is also supported via
the imported package, while discovery itself remains import-free.

Run `python scripts/release/check_public_api.py --update` from that updated
checkout, review and commit the baseline diff with the package implementation,
then run:

```bash
python scripts/release/check_public_api.py
python -m pytest tests/unit/release/test_public_api_compat.py \
  tests/integration/test_public_api_multimodal.py -q
```

The coverage test requires every captured nested public package, including one
with an empty `__all__`, to appear in the committed baseline. Compatible
additions still pass the checker itself; this test prevents a new package from
remaining unprotected by an older baseline. Packages skipped for unavailable
optional dependencies do not create false coverage failures. Use an environment
with the relevant extra installed when intentionally baselining such a package;
review partial-install updates to avoid dropping existing optional surfaces.
Specific pending packages are added to the baseline when their implementations
land, rather than recording speculative APIs.

The checker is stdlib-only (`ast` / `inspect`). It reads each namespace's
literal `__all__` from source, then imports available objects to inspect their
callable signatures. Modules that cannot be imported in a given environment
(because an optional extra is missing) are skipped rather than reported as
removed. A declared export that is temporarily unavailable remains in the
snapshot but is not treated as a kind or signature change, so partial installs
never produce false breaks.
