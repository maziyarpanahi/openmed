# Brand ownership

| Surface | Responsible owner | Required review |
|---|---|---|
| Tokens and canonical identity assets | Repository owner | Brand + accessibility |
| Claims registry and refresh | Repository owner | Evidence owner for each claim |
| Website consumer and metadata | Website maintainer | Brand + accessibility + claims |
| Documentation consumer and locales | Documentation maintainer | Brand + localization + claims |
| Approved social exports and declared derivatives | Repository owner | Source-hash + crop/safe-zone validation |
| External profile uploads | Repository owner | Explicit per-platform authorization |

## Change protocol

1. Change canonical repository sources, never a generated consumer alone. For
   social pixels or baked image copy, change `OpenMed Social Cards.dc.html`,
   re-export, obtain owner review, and replace the approved export set and
   provenance hashes; never edit a PNG directly.
2. Refresh evidence only through `scripts/brand/update_claims.py`; network
   collection is a separate, explicit maintainer action.
3. Copy or derive social files into temporary output, validate approved-source
   hashes, exact aliases, dimensions, crops, and derivation rules, then update
   tracked distribution copies.
4. Update localized README alt text and claim wording in the same change.
5. Record an approved exception instead of silently adding another font,
   color, radius, breakpoint, motion pattern, or identity assignment.

External profile changes are not implied by repository changes. In particular,
never change the personal GitHub owner avatar, and never change a Hugging Face
Space's visibility.

## Review cadence

Review static identity guidance annually. Review dated claims by their
`review_by` field and immediately after release, runtime, license, or supported
language changes. A lapsed or unverified claim is removed from public copy
rather than carried forward. The baked-copy source exception is not silently
edited; resolving it requires an owner-reviewed source-DC re-export.
