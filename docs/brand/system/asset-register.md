# Brand asset register

This register is the Phase 0 disposition record for every file under
`docs/brand/assets/` and every approved social export.
“Superseded” means the file remains tracked for provenance, rollback, or
historical comparison but must not be used by a new consumer.

| Asset | Disposition | Current authority or replacement | Reason |
|---|---|---|---|
| `assets/open-cross.svg` | Retained | Canonical light-surface Open Cross | Approved red-accent geometry for current consumers |
| `assets/open-cross-inverse.svg` | Retained | Canonical dark-surface Open Cross | Approved inverse variant |
| `assets/openmed-wordmark.svg` | Retained | Canonical lowercase wordmark | Current wordmark authority |
| `assets/cat-crest.png` | Retained | Canonical cat crest | Current community/avatar character asset |
| `assets/open-cross-handoff.svg` | Retained, provenance only | `assets/open-cross.svg` | Exact imported geometry is kept for hash verification; the current variant owns color |
| `assets/brand/cat-head.png` | Superseded | `assets/cat-crest.png` | Byte-identical historical path retained for rollback and old-link diagnosis |
| `assets/brand/openmed-mascot.png` | Superseded | `assets/cat-crest.png` plus the approved social exports | Old standalone mascot is not used by a current consumer |
| `assets/logo/openmed-favicon.svg` | Superseded | `assets/open-cross.svg` and approved-export-derived `social/favicon.*` | Old favicon geometry/color is no longer distributed |
| `assets/logo/openmed-glyph-reversed.svg` | Superseded | `assets/open-cross-inverse.svg` | Old reverse glyph is retained for visual history |
| `assets/logo/openmed-glyph.svg` | Superseded | `assets/open-cross.svg` | Old glyph is retained for visual history |
| `assets/logo/openmed-lockup-horizontal-dark.svg` | Superseded | `assets/open-cross-inverse.svg` plus `assets/openmed-wordmark.svg` | Consumers now compose canonical primitives |
| `assets/logo/openmed-lockup-horizontal.svg` | Superseded | `assets/open-cross.svg` plus `assets/openmed-wordmark.svg` | Consumers now compose canonical primitives |
| `assets/logo/openmed-lockup-stacked.svg` | Superseded | `assets/open-cross.svg` plus `assets/openmed-wordmark.svg` | Stacked lockup has no current approved role |
| `assets/logo/openmed-mark-light.svg` | Superseded | `assets/open-cross-inverse.svg` | Old light mark is retained for visual history |
| `assets/logo/openmed-mark.svg` | Superseded | `assets/open-cross.svg` | Old mark is retained for visual history |
| `assets/fonts/IBMPlexSans-Regular.ttf` | Retained | Source font input | Deterministic web-font and raster generation |
| `assets/fonts/IBMPlexSans-Medium.ttf` | Retained | Source font input | Deterministic web-font and raster generation |
| `assets/fonts/IBMPlexSans-SemiBold.ttf` | Retained | Source font input | Deterministic web-font and raster generation |
| `assets/fonts/IBMPlexSans-Bold.ttf` | Retained | Source font input | Deterministic web-font and raster generation |
| `assets/fonts/IBMPlexMono-Regular.ttf` | Retained | Source font input | Deterministic web-font and raster generation |
| `assets/fonts/IBMPlexMono-Medium.ttf` | Retained | Source font input | Deterministic web-font and raster generation |
| `assets/fonts/IBMPlexMono-SemiBold.ttf` | Retained | Source font input | Deterministic web-font and raster generation |
| `assets/fonts/Newsreader-Medium.ttf` | Retained | Source font input, approved exception | Website editorial and social roles only |
| `assets/fonts/Newsreader-MediumItalic.ttf` | Retained | Source font input, approved exception | Website editorial and social roles only |
| `assets/fonts/IBMPlexSans-Regular.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Published browser payload |
| `assets/fonts/IBMPlexSans-Medium.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Published browser payload |
| `assets/fonts/IBMPlexSans-SemiBold.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Published browser payload |
| `assets/fonts/IBMPlexSans-Bold.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Published browser payload |
| `assets/fonts/IBMPlexMono-Regular.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Published browser payload |
| `assets/fonts/IBMPlexMono-Medium.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Published browser payload |
| `assets/fonts/IBMPlexMono-SemiBold.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Published browser payload |
| `assets/fonts/Newsreader-Medium.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Website-only approved exception payload |
| `assets/fonts/Newsreader-MediumItalic.woff2` | Retained, generated | `scripts/brand/build_web_fonts.py` | Website-only approved exception payload |
| `assets/fonts/IBM-Plex-OFL.txt` | Retained | Upstream license notice | Required license provenance |
| `assets/fonts/Newsreader-OFL.txt` | Retained | Upstream license notice | Required license provenance |
| `assets/fonts/manifest.json` | Retained, generated | `scripts/brand/build_web_fonts.py` | Exact source/output hashes and tool version |

## Canonical social exports

These are immutable copies of the owner-approved handoff files under
`social/exports/`. They are the sole visual authority for social art.
Top-level @2x files are exact copies; native files, consumer copies, previews,
touch art, and ICO frames are declared derivatives. No repository-authored
artboard or source JSON may replace their pixels.

| Canonical export | Disposition | Distribution relationship |
|---|---|---|
| `social/exports/og-website-2x.png` | Retained, canonical | Exact alias `social/og-2x.png`; half-size `social/og.png`; website consumer copy |
| `social/exports/github-social-2x.png` | Retained, canonical | Exact `social/github-social-2x.png` copy; half-size repository-preview output |
| `social/exports/x-header-2x.png` | Retained, canonical | Exact `social/x-header-2x.png` copy; half-size X-header output |
| `social/exports/hf-org-2x.png` | Retained, canonical | Exact alias `social/hf-card-2x.png`; half-size Hugging Face card output |
| `social/exports/readme-banner-2x.png` | Retained, canonical | Exact `social/readme-banner-2x.png` copy; half-size README output and consumer copy |
| `social/exports/linkedin-banner-2x.png` | Retained, canonical | Exact `social/linkedin-banner-2x.png` copy; half-size LinkedIn output |
| `social/exports/avatar-cat-2x.png` | Retained, canonical | Exact top-level @2x copy; half-size Hugging Face avatar |
| `social/exports/avatar-x-circle-2x.png` | Retained, canonical | Exact top-level @2x copy; half-size X avatar |
| `social/exports/avatar-linkedin-2x.png` | Retained, canonical | Exact top-level @2x copy; half-size LinkedIn tile |
| `social/exports/favicon-2x.png` | Retained, canonical | Exact top-level @2x copy; raster favicon, touch, and ICO derivatives |

The baked copy in these files is an owner-approved source exception, not an
independently verified projection of `claims.yml`. Change it only in
`OpenMed Social Cards.dc.html`, followed by a reviewed re-export and provenance
update; never edit the PNGs directly.

## Retired consumer artifacts

| Former path | Disposition | Replacement | Reason |
|---|---|---|---|
| `docs/website/assets/openmed-tui-preview.png` | Removed | Semantic HTML/CSS terminal demonstrations in `docs/website/index.html` | The raster screenshot had no published consumer and duplicated a live, accessible surface |
| `docs/website/assets/welna-home.png` | Removed | Current text-led Welna card in `docs/website/index.html` | The stale product screenshot no longer represented the current surface |
| `docs/website/brand/cat-head.png` | Removed | `docs/brand/assets/cat-crest.png` | The canonical crest owns community and avatar character use |
| `docs/website/brand/openmed-favicon.svg` | Removed | Generated root favicon assets | The retired geometry and color no longer match the canonical Open Cross |
| `docs/website/brand/openmed-mark-light.svg` | Removed | `docs/website/logo-inverse.svg` | The inverse canonical lockup now owns dark-surface header use |
| `docs/website/brand/openmed-mark.svg` | Removed | `docs/website/logo.svg` | The canonical lockup now owns light-surface header use |
| `docs/website/brand/openmed-mascot.png` | Removed | `docs/brand/assets/cat-crest.png` and approved social exports | The old standalone mascot no longer has an approved website role |

No canonical input under `docs/brand/assets/` was removed. Historical logo and
mascot inputs remain because removal would discard rollback evidence without
reducing any published payload; validators prohibit them from becoming current
consumer dependencies.
