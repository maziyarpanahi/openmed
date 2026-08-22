# OpenMed social and distribution art

The ten PNGs under `exports/` are the sole canonical social visuals. They are
byte-for-byte repository copies of the owner-approved July handoff directory
`social/exports/`, rendered from `OpenMed Social Cards.dc.html`. The repository
does not reconstruct, redraw, retokenize, or substitute those compositions.

The render script may only copy an approved @2x export exactly or produce the
declared size/crop derivative. The two renamed distribution masters below are
explicit aliases: `og-2x.png` copies `exports/og-website-2x.png`, and
`hf-card-2x.png` copies `exports/hf-org-2x.png`.

| Canonical export | Top-level @2x copy | Native output | Native size | Assignment |
|---|---|---|---:|---|
| `exports/og-website-2x.png` (2400×1260) | `og-2x.png` | `og.png` | 1200×630 | Website Open Graph |
| `exports/github-social-2x.png` (2560×1280) | `github-social-2x.png` | `github-social.png` | 1280×640 | GitHub repository social preview |
| `exports/x-header-2x.png` (3000×1000) | `x-header-2x.png` | `x-header.png` | 1500×500 | X profile header |
| `exports/hf-org-2x.png` (2400×1260) | `hf-card-2x.png` | `hf-card.png` | 1200×630 | Hugging Face organization card |
| `exports/readme-banner-2x.png` (2560×640) | `readme-banner-2x.png` | `readme-banner.png` | 1280×320 | README banner |
| `exports/linkedin-banner-2x.png` (2256×382) | `linkedin-banner-2x.png` | `linkedin-banner.png` | 1128×191 | LinkedIn company banner |
| `exports/avatar-cat-2x.png` (1024²) | `avatar-cat-2x.png` | `avatar-square-512.png` | 512² | Hugging Face `OpenMed` avatar |
| `exports/avatar-x-circle-2x.png` (800²) | `avatar-x-circle-2x.png` | `avatar-circle-400.png` | 400² | X avatar |
| `exports/avatar-linkedin-2x.png` (600²) | `avatar-linkedin-2x.png` | `avatar-linkedin-300.png` | 300² | LinkedIn company tile |
| `exports/favicon-2x.png` (128²) | `favicon-2x.png` | `favicon-64.png` | 64² | Raster favicon source |

`apple-touch-180.png`, the frames in `favicon.ico`, consumer copies, and
safe-zone previews are derivatives of these approved files. They are not
additional visual designs.

`readme-banner.png`, `og.png`, `favicon-64.png`, and `apple-touch-180.png`
are copied byte-for-byte to their repository consumers declared in the
manifest. External profile uploads remain separately authorized operations.
The personal GitHub owner avatar is explicitly out of scope.

## Sources

- `exports/`: immutable canonical @2x visual masters copied from the approved
  handoff.
- `_src/exports.json`: dimensions, roles, safe zones, exact aliases,
  derivative rules, and distribution targets for those approved masters.
- `_src/profile-copy.json`: profile text and alt-text templates only; it does
  not define or alter image pixels.
- `manifest.json`: canonical source and output hashes, dimensions, derivative
  rules, aliases, and distribution targets.
- `previews/`: reviewer overlays for X, LinkedIn, GitHub, and Hugging Face crop
  zones, derived from approved distribution art.
- `../system/evidence/social-visual-review.json`: hash-bound original-size,
  320 px thumbnail, safe-zone, and avatar review evidence for the repository
  candidate. It is not platform-owner approval.
- `PLATFORM_CUTOVER_RUNBOOK.md`: separately authorized upload, cache,
  evidence, and rollback procedure. It never authorizes external action.

## Regenerate

From the repository root, using the locked development environment:

```bash
UV_CACHE_DIR=/tmp/openmed-brand-uv-cache uv run --frozen --extra dev --extra docs python scripts/brand/render_social_assets.py --write
UV_CACHE_DIR=/tmp/openmed-brand-uv-cache uv run --frozen --extra dev --extra docs python scripts/brand/render_social_assets.py --check
UV_CACHE_DIR=/tmp/openmed-brand-uv-cache uv run --frozen --extra dev --extra docs python scripts/brand/validate_system.py
```

The check runs with socket creation blocked. It verifies that canonical export
hashes have not changed, every top-level @2x copy is byte-identical to its
declared export, each @1x file has the declared half-size derivation, and every
consumer copy, touch/icon derivative, and review preview is current.

## Baked-copy source exception

The approved handoff exports include baked model, download, install, and
license copy. The owner's 2026-07-29 direction makes the supplied visual bytes
authoritative as an explicit source exception; it does not independently
verify each embedded claim or make the pixels a projection of `claims.yml`.

Never edit those claims in a PNG, paint over them, or silently substitute
repository-generated copy. To change any baked claim, edit
`OpenMed Social Cards.dc.html`, re-export the affected approved asset set,
obtain owner review, and update the canonical exports and provenance hashes in
one change. External profile uploads remain separately authorized operations.
