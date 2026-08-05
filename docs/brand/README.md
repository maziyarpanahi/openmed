# OpenMed brand system

This directory is the repository-owned source of truth for OpenMed.life,
documentation, README, and social presentation. It implements the reviewed
July 2026 handoff without depending on the design-canvas runtime or remote
fonts.

## Authority

- `system/tokens/*.css` owns browser tokens; `system/tokens.json` is the
  validated platform-neutral projection.
- `system/claims.yml` owns publishable claims and their qualifications.
  Verified claims carry `as_of` and `review_by` dates; deliberately
  unpublished/unverified claims carry a separate `follow_up_by` deadline
  without pretending that they were verified.
- `assets/` owns the Open Cross, lowercase wordmark, cat crest, and self-hosted
  fonts.
- `social/exports/` owns the immutable, owner-approved social visuals.
  `scripts/brand/render_social_assets.py` may only exact-copy those @2x masters
  or make declared size/icon/consumer/review derivatives.
- `social/_src/exports.json` declares the approved export mappings, dimensions,
  safe zones, aliases, derivatives, and distribution targets.
- `social/_src/profile-copy.json` owns profile text and alt-text templates; it
  does not own image pixels.
- `system/iconography.md`, `voice.md`, `site-exceptions.md`, and
  `ownership.md` govern usage and review.
- `system/handoff-provenance.json` records exactly what was imported, excluded,
  or deliberately changed.
- `system/evidence/manual-accessibility-review.json` records the dated,
  repository-candidate manual accessibility scope, methods, results, and empty
  waiver set. It is evidence for the staged artifact, not user testing or
  live-platform approval.
- `system/asset-register.md` records the retained, superseded, and removed
  disposition of every repository-owned brand input.
- `system/version.json`, `CHANGELOG.md`, and `deprecation.md` own system
  versioning, migration history, and generated-consumer removal policy.

The older files under `assets/logo/` and `assets/brand/` are deliberately
retained as historical rollback and provenance inputs. They are superseded for
new use and have no current consumer; see the asset register for the exact
replacement of each file.

## Validate and regenerate

Run these commands from the repository root:

```bash
UV_CACHE_DIR=/tmp/openmed-brand-uv-cache uv run --frozen --extra dev --extra docs python scripts/brand/update_claims.py --check
UV_CACHE_DIR=/tmp/openmed-brand-uv-cache uv run --frozen --extra dev --extra docs python scripts/brand/sync_consumers.py --check
UV_CACHE_DIR=/tmp/openmed-brand-uv-cache uv run --frozen --extra dev --extra docs python scripts/brand/render_social_assets.py --check
UV_CACHE_DIR=/tmp/openmed-brand-uv-cache uv run --frozen --extra dev --extra docs python scripts/brand/validate_system.py
```

The `dev` extra pins the exact image and font tooling used for deterministic
derivatives; `--frozen` verifies the checked-in lockfile. Social generation
uses only the canonical local exports, never reconstructs their compositions,
and blocks network access.

The repository-star claim is also offline by default. A maintainer may refresh
its checked-in evidence deliberately with:

```bash
UV_CACHE_DIR=/tmp/openmed-brand-uv-cache uv run --frozen --extra dev --extra docs python scripts/brand/update_claims.py --refresh-github-stars
```

That command performs a network request and must never run in CI. Normal
`--write`, `--check`, rendering, and validation read only the checked-in
`system/evidence/github-repository.json` snapshot.

Newsreader web files are deterministic, explicitly ranged subsets of the
checked-in source fonts. IBM Plex remains full for the documentation and its
localized routes. Exact upstream provenance, embedded versions, license hashes,
subset settings, and fallback metrics are recorded in
`assets/fonts/manifest.json` and `system/tokens.json`.

Live profile edits, deployments, social posts, and cache refreshes are separate
maintainer-authorized operations. This source tree does not perform them.

The approved social exports contain baked handoff copy. That copy is an
explicit owner-approved source exception rather than a generated projection of
`system/claims.yml`. Change it only in `OpenMed Social Cards.dc.html`, then
re-export, review, and replace the canonical export set and provenance hashes;
never alter the PNG pixels directly.
