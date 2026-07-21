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
- Adding a new parameter that has a default value.
- Widening a callable with `*args` / `**kwargs`.
- Changing a parameter's default *value* (the baseline records only whether a
  default exists, not what it is).

**Breaking (fails CI unless announced):**

- Removing an exported name.
- Changing a member's kind (for example a function becoming a class).
- Removing or renaming a parameter.
- Reordering positional parameters.
- Making a previously optional parameter required, or adding a new required
  parameter.

## Deprecation cycle

Backwards-incompatible changes are allowed, but only deliberately and with
warning. The required cycle is:

1. **Announce.** In the release that first deprecates a symbol, keep it working
   but emit a `DeprecationWarning` and document the replacement in the
   changelog and the affected docstring.
2. **Wait at least one minor release.** Give downstream users a full minor
   version to migrate before the symbol is removed or its signature changes.
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

The checker is stdlib-only (`ast` / `inspect`) and imports the public
namespaces directly. Modules that cannot be imported in a given environment
(because an optional extra is missing) are skipped rather than reported as
removed, so partial installs never produce false breaks.
