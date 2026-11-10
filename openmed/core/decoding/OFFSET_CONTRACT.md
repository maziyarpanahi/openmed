# Cross-runtime span offset contract

OpenMed exchanges entity spans between Python and Swift as half-open
`[start, end)` Unicode scalar (code point) offsets into the exact, unnormalized
source string.

- Python uses its native string indices.
- Swift uses `String.UnicodeScalarView` indices converted through
  `PostProcessing`.
- UTF-8 byte offsets and UTF-16 code-unit offsets are never accepted as entity
  coordinates.
- A non-empty span that starts or ends inside an extended grapheme cluster is
  expanded outward to the cluster boundaries.
- An empty span remains empty and moves to the preceding grapheme boundary.
- Out-of-range decoder offsets are clamped to the source before snapping.

For example, the Devanagari cluster `क्षि` contains four Unicode scalars but
one user-perceived character. A model span covering scalar offsets `[1, 3)` is
therefore emitted as `[0, 4)`. The same rule applies to combining marks,
joiner sequences, emoji ZWJ sequences, and all other extended grapheme
clusters.

Replacement must convert the scalar offsets back to native string indices and
apply multiple spans from highest start offset to lowest. This keeps earlier
source offsets stable when replacement lengths differ.

The shared executable examples live in
`tests/fixtures/parity/offset_contract.json` and are consumed by both pytest
and XCTest.
