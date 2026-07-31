# Marks and iconography

OpenMed has two complementary identity registers. Neither is a universal
avatar.

| Context | Asset | Assignment |
|---|---|---|
| Website/docs chrome and favicon | `assets/open-cross.svg` + lowercase wordmark | Product identity |
| Website Open Graph | `social/exports/og-website-2x.png` | Approved social export |
| Hugging Face `OpenMed` organization avatar | `social/exports/avatar-cat-2x.png` | Approved social export |
| Future named OpenMed GitHub organization | `social/exports/avatar-cat-2x.png` | Only after a specific org is approved |
| Personal owner `maziyarpanahi` | None | Never change the maintainer's personal avatar |
| X avatar | `social/exports/avatar-x-circle-2x.png` | Approved social export |
| LinkedIn company tile | `social/exports/avatar-linkedin-2x.png` | Approved social export |
| README banner | `social/exports/readme-banner-2x.png` | Approved social export |
| GitHub repository card | `social/exports/github-social-2x.png` | Approved social export |
| Hugging Face organization card | `social/exports/hf-org-2x.png` | Approved social export |

For social use, these supplied exports—not the component descriptions below—
are the visual authority. The descriptions govern non-social identity use and
accessibility; they do not authorize rebuilding or modifying an export.

## Open Cross

- Minimum digital size: 24 CSS pixels; use the raster favicon below that.
- Clear space: at least one center-dot diameter on every side.
- Use `open-cross.svg` on light or neutral surfaces and
  `open-cross-inverse.svg` on dark surfaces.
- Preserve the shared 48-unit geometry. OpenMed.life uses the reviewed
  `#B0413E` center dot; do not restore the handoff root's tangerine dot.
- Do not recolor individual arms, add a shadow, animate, tilt, stretch, or use
  it as a medical-emergency or certification symbol.

## Wordmark

The visual wordmark is lowercase `openmed`. Do not rebuild it in Newsreader,
all caps, title case, or an icon font. In ordinary prose, keep the product name
`OpenMed`.

- In repository-owned product chrome, the live-text wordmark must not be set
  below the website's 15 CSS pixel floor. When
  `assets/openmed-wordmark.svg` is placed as artwork, render it at least 24 CSS
  pixels high and preserve its 178:48 view box.
- Keep clear space equal to at least one quarter of the rendered wordmark
  height on every side. In an Open Cross lockup, use whichever rule is larger:
  this wordmark clearance or the Cross's center-dot clearance.
- The canonical SVG has dark `#0E1116` ink and is for light, paper, or cream
  surfaces. On a dark repository-owned surface, use lowercase live text in IBM
  Plex Sans 600 with `--om-ink`; the dark token resolves to `#E6EBEE`. There is
  no canonical inverse wordmark SVG, so do not apply a CSS filter or edit the
  source fill.
- Do not crop the first or last letter, wrap the wordmark, distort its aspect
  ratio, or place it over a busy image. A lockup may pair it with the Open
  Cross, but it must not be fused into a new combined asset.

## Cat crest

Keep the original transparent pixels and cream crop background. Do not use the
cat for product chrome, favicon, X avatar, LinkedIn company tile, or a personal
maintainer account. For a social avatar, use the approved export and its
declared derivative without making a new crop.

- Outside the supplied social exports, render the crest no smaller than 96
  CSS pixels on its shortest side. Use
  `social/avatar-square-512.png`, rather than a new small crop, for the
  Hugging Face organization-avatar role.
- Preserve the full 502×462 `assets/cat-crest.png` canvas and its alpha
  channel. When the raw crest is placed in a separately approved non-social
  composition, keep
  clear space of at least one eighth of its rendered width; transparent source
  pixels count toward that clearance and must not be trimmed.
- The crest has one fixed palette. It has no inverse or monochrome variant.
  On light or dark interfaces, retain its colors and use the checked-in cream
  backing (`#FBF7EF`) instead of recoloring it for the surrounding theme.
- A face crop is permitted only for an approved small community role. Keep
  both ears, all whiskers, and the face center inside the crop. Do not place
  text or a platform mask over the face.

## Favicon and touch icon

Favicons use the Open Cross, not the cat or wordmark. Use the checked-in
outputs instead of making a new small raster:

| Role | Repository asset | Minimum rendered/export size |
|---|---|---:|
| Vector browser icon | `assets/open-cross.svg` through the distributed `favicon.svg` | 24×24 CSS pixels |
| Approved raster source | `social/exports/favicon-2x.png` | 128×128 pixels |
| Raster browser fallback | `social/favicon-64.png` | 64×64 pixels |
| Auto-discovered browser fallback | `social/favicon.ico` | Included 16, 32, 48, and 64 pixel frames |
| Apple touch icon | `social/apple-touch-180.png` | 180×180 pixels |

The raster fallback, ICO frames, and touch icon are declared derivatives of
the approved raster source. Preserve their complete canvases: do not crop, add
padding, round them again, add a background, substitute the inverse Cross, or
redraw them from a repository-authored artboard. These files are
theme-invariant; browser or operating-system chrome supplies the surrounding
light or dark context.

Favicons do not take HTML alt text. If a favicon or touch icon is shown as
content in documentation, use `OpenMed Open Cross` as the accessible name.

## Platform avatars

The dimensions below are OpenMed's repository export floors, not statements
about an external platform's limits. Upload the native output; its @2x source
is the immutable approved handoff export.

| Role | Canonical @2x export | Native output | Background and crop |
|---|---|---|---|
| Hugging Face `OpenMed` organization | `social/exports/avatar-cat-2x.png`, 1024² | `social/avatar-square-512.png`, 512² | Cat crest on fixed cream; preserve the square canvas |
| X `@OpenMed_AI` | `social/exports/avatar-x-circle-2x.png`, 800² | `social/avatar-circle-400.png`, 400² | Pre-cropped circle with transparent corners |
| LinkedIn `openmed-ai` company tile | `social/exports/avatar-linkedin-2x.png`, 600² | `social/avatar-linkedin-300.png`, 300² | Preserve the supplied square tile |

Do not crop, remove a background, add a new platform-shaped mask, or swap the
cat and Cross assignments. The avatar files are already composed for their
roles and remain the same in light and dark platform themes. Preview any
platform crop before a separately authorized upload. A future named OpenMed
GitHub organization may use the cat only after that organization and its exact
asset are approved; this register never authorizes a change to the personal
`maziyarpanahi` avatar.

## Accessibility and alt text

- When a mark and wordmark are inside a link already named `OpenMed home`, the
  images are decorative and use empty alt text. Do not make a screen reader
  announce `OpenMed` three times.
- When the wordmark is the only visible label, use `OpenMed` as its alt text,
  not the lowercase visual spelling. A standalone identity mark can use
  `OpenMed Open Cross`.
- When the cat is meaningful, use `OpenMed cat crest`; when it is decorative
  beside equivalent visible copy, use empty alt text.
- Use the checked-in avatar descriptions where the platform accepts alt text:
  `OpenMed cat crest on a cream background.` for Hugging Face,
  `OpenMed Open Cross in a circular crop.` for X, and
  `OpenMed Open Cross.` for LinkedIn.
- Describe any useful adjacent message after the identity, but do not narrate
  decorative geometry, duplicate nearby text, or put claims into an avatar's
  alt text. Localize prose alt text with the surrounding page.

## UI icons

Use simple SVG line or filled icons with accessible names. Emoji and icon fonts
are not brand chrome. Decorative icons are hidden from assistive technology;
interactive icons require a visible label or an accessible name.
