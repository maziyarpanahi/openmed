# OpenMed website, docs, and social brand rollout

| Field | Value |
|---|---|
| Audit date | 2026-07-29 |
| Branch | `design/website-docs-social-refresh` |
| Base commit | `fbe8d110757d0871715e6fe29d8f0ab330f06e6b` |
| Status | Repository implementation validated; external cutover not authorized |
| Publication | Internal repository plan; `docs/brand/**` is excluded from MkDocs output |

## Outcome

OpenMed should migrate the marketing site, documentation, repository
presentation, and social assets to the **2026-07-29 multi-site design handoff**.
This is a real visual-system migration, not a polish pass over the audit
baseline. At that baseline, the production marketing site substantially
implemented the older April editorial system; it did not yet implement the
July IBM Plex system, its single-token accent engine, or the new OpenMed.life
composition.

The target keeps the current site's useful product depth and the repository's
2.0 capability truth while adopting the new handoff's:

- cool near-white and blue-black surfaces;
- IBM Plex Sans and IBM Plex Mono shared foundations;
- one derived accent engine, set to OpenMed.life signal red `#B0413E`;
- tight radii, hairline/ledger structure, flat cards, and ink terminal bands;
- OpenMed.life-specific Newsreader metric treatment and rotating hardware word;
- final community-wall, left-terminal quickstart, and ownership-led hero
  variants;
- platform-specific social mark and mascot assignments.

The `.dc.html` exports are design documents, not deployable source. Production
must reimplement their approved composition in the repository's static
HTML/CSS/JavaScript and MkDocs structures. Do not copy their generated
`support.js`, runtime-loaded React/Babel, Google Fonts imports, GitHub API
fetches, stale examples, or unverified claims.

The ten reviewed files in `social/exports/` are the narrow exception: the owner
has designated their exact pixels as the sole canonical social visuals.
Repository tooling must preserve the @2x files byte-for-byte and may only
produce declared size, consumer, preview, touch, and icon derivatives. Their
baked copy is an explicit source exception, not an independently verified
projection of the claims registry. Any visual or baked-copy change must start
in `OpenMed Social Cards.dc.html`, be re-exported and owner-reviewed, and
update the canonical files and provenance hashes; never edit the PNG pixels.

Outside that explicit social-export exception, repository/runtime truth always
overrides handoff copy. The reviewed handoff
contains 1.8.x/1.9.1 examples, 17/27/29-language values, and 340M/30M/9.4M
metrics; the current repository is OpenMed 2.1.0 and documents 34 supported
PII language codes. At the audit baseline, public "model-backed" surfaces said
29, 30, and 32. The implementation resolves that definition
deterministically to 33 model-backed codes from the authoritative
language-pack/model catalog.

### Reviewed handoff identity

Record these files and hashes in repository provenance when implementation
begins:

| Handoff file | SHA-256 | Role |
|---|---|---|
| `SKILL.md` | `a0916f07025494456f0c83d51af70560c967233a081c91d30519d1b7099c7f58` | usage rules |
| `readme.md` | `253a2e9dfcbc2dc435f3c7c821117811b2bf10bb3626e55956c8c1ede896c4d4` | shared-system rationale |
| `tokens/colors.css` | `fc010d24c177cbeeeb95d6847c292814323cbc9aa2f2dba8a7322243e8bf4789` | themes and accent engine |
| `tokens/typography.css` | `e06a501ae0a754683b29348dab6b5a46e5c139dd994d884fb16479497a8a0778` | shared type scale |
| `tokens/surfaces.css` | `acce3d9551fa4cd045bf77e515c2bfcd714f4e3f8000ad96d4200544e15856a2` | layout, radii, rules, motion |
| `assets/logo.svg` | `43d553efb499c37f97e0752b0a9bed61b585edde983e4a8b35119442f8cbcf80` | shared brand mark |
| `openmed.life/design_handoff_landing_redesign/OpenMed.life Landing v2.dc.html` | `a2a1a321a25ef9154a8a6a275ba6d1c80037459501baf5b4c1dbcd4534df8a4b` | final OpenMed.life composition |
| `openmed.life/design_handoff_landing_redesign/README.md` | `be6d328d5a8c02de453b7203ae9bf88a45f12d56cbc6f01fb2bc68ee8e4e3716` | site implementation handoff |
| `openmed.life/docs/docs.css` | `15ccf320178ec4febdfc399a3e404cead97cbfcffc8114c8b17ae8459e194944` | docs visual reference |
| `openmed.life/docs/docs.js` | `048620241fa3f3b5032b0c929bfb51464584fa2efcd6504a96508f2168b748dc` | docs interaction reference |
| `OpenMed Social Cards.dc.html` | `a4d597baa83e71d8601d02d4ae9b2441b50cb69871887045d554c146384fe3d9` | social artboards |
| `social/README.md` | `e28ac10689d4ae392509bad8a7102fba63d3895ab1c6d862ff2e13bb5d5178a0` | export manifest |

The July bundle also embeds the old April `openmed-design-system` under `_ds`.
Its `colors_and_type.css` hash,
`fe2c5d97e2c2342cbbaf83d541f57e2faa5ab728ffa325933a3ed7b39886dca8`,
matches the separate April handoff exactly. Treat `_ds` as migration
provenance only; do not import it into the new canonical system.

## Authority and decisions

Use this precedence order during implementation:

1. Product capability, licensing, privacy, language, and release truth in the
   current repository.
2. The July root tokens, brand mark, component intent, and shared content
   rules.
3. The July `openmed.life/design_handoff_landing_redesign/` source and README
   for OpenMed.life-specific composition and exceptions.
4. The ten July `social/exports/` PNGs for social pixels, with the social
   artboard and export README governing their source, dimensions, and platform
   assignments.
5. Existing production behavior only where it preserves verified product
   content, accessibility, compatibility, or an intentional 2.0 capability.
6. The embedded April handoff only as migration evidence.

Lock these decisions in the first implementation pull request:

- OpenMed.life website, docs, and social use signal red `#B0413E` as
  `--om-accent-base`. The docs mock's blue `#4A92C8` override conflicts with
  the final landing and social direction and should not survive this unified
  rollout.
- IBM Plex Sans and IBM Plex Mono are the shared families. Newsreader is a
  deliberate **OpenMed.life artifact exception** for editorial numerals,
  selected community/FAQ display copy, and social headlines—not a third
  global UI family.
- The root system forbids gradients and broad animation, while the
  OpenMed.life source explicitly defines one rotating gradient hardware word.
  Keep that single site-specific exception, provide a solid-color fallback,
  and make it static under reduced motion.
- The final OpenMed.life source also explicitly overrides the shared
  no-pill/terminal-only-motion defaults for compact release/star metadata
  pills, the release-status pulse, the synthetic PHI sequence, and the FAQ
  disclosure transition. Register those exact selectors/states and the
  1080 px comparison-table overflow rule in `site-exceptions.md`; no other
  pill, gradient, animation, or breakpoint is implied.
- Use the Open Cross/rounded-tile mark in site chrome, favicon, X avatar, and
  LinkedIn tile. Use the cat crest for the Hugging Face organization avatar.
  The handoff also assigns it to a GitHub organization, but this repository is
  personally owned; do not change the maintainer's avatar. Apply that assignment
  only if a specific OpenMed GitHub organization is later named and approved.
  The mascot is not a substitute for the mark in unspecified contexts.
- Use only the ten supplied social exports for social visuals. Do not replace
  them with repository-reconstructed compositions. Preserve baked copy as the
  recorded source exception until an owner-reviewed source-DC re-export
  replaces it.
- Use lowercase `openmed` only in the visual wordmark. Product prose,
  metadata, accessibility text, and package naming retain `OpenMed`.
- A separate dark/mint app/demo system remains product-specific and does not
  override the OpenMed.life website/docs/social authority.

## Review coverage

| Surface | Reviewed | Audit baseline state |
|---|---:|---|
| July shared handoff, components, guidelines, and tokens | Yes | New authority; absent from the repository at audit time |
| OpenMed.life landing and docs handoff source | Yes | High-fidelity design evidence with stale product copy and prototype-only runtime |
| Marketing source and production site | Yes | Older April system; accessibility, motion, metadata, and network gaps remain |
| Material for MkDocs source and production docs | Yes | Minimal teal shim; new July foundations are not shared with the website |
| Website metadata, structured data, icons, OG image | Yes | Semantically rich, but claims and distributed OG art drift |
| Docs navigation, generated Pages artifact, and locales | Yes | False localized fallbacks, two broken links, shallow metadata, and custom-surface drift |
| Root README and all 14 translations | Yes | All use the old mascot lockup; none use the intended new README banner |
| Current tracked and new handoff social art | Yes | New @2x exports are dimensionally correct; they must replace reconstructed repository art while live state still differs |
| GitHub repository presentation | Yes | Live social preview and description do not match tracked assets/claims |
| Hugging Face organization presentation | Yes | Live organization art/copy differs from tracked art and contains stale claims |
| X organization presentation | Yes | Live header/avatar differ from the tracked documented files |
| LinkedIn organization presentation | Yes | Cover, avatar, and About copy do not form one OpenMed identity |
| Responsive light/dark website and docs | Yes | 390 px samples work; the marketing header overflows at 320 px and the 667×320 open menu is clipped |
| Keyboard and semantic interaction review | Yes | Several interactive controls lack the necessary HTML/ARIA contract |
| Third-party browser requests | Yes | Fonts, GitHub API, and analytics make external requests |
| Existing brand/content/build tests | Yes | Baseline focused tests and strict build passed; visual assets, localized output, profile state, and most claims were not governed |

Downstream product UI, app-store assets, slide templates, video templates, and
the Swift/Android design tokens were inventoried as consumers but are not in
the implementation scope of this website/docs/social rollout. The Agent site,
founder-personal banners, and unrelated application/demo themes are also
outside this rollout.

## Baseline evidence

The audit used source inspection, live Chromium checks, generated-site crawl,
asset/hash inspection, and repository tests. Current-state observations are
not target acceptance:

- Marketing and docs were checked in light/dark desktop and mobile states,
  including menu, FAQ, theme, navigation, and representative content.
  The 390 px layouts had no page overflow; dedicated 320×800 and 667×320
  checks exposed the narrower header/menu failures recorded below.
- The sampled **production-origin** marketing page produced no console error.
  Local/staged checks produced Cloudflare analytics CORS noise, which must not
  be mistaken for a production pass. The sampled production docs session
  requested two missing localized sitemap files.
- The generated docs crawl covered 495 HTML files and 87,095 internal
  references and found the two cross-locale broken links recorded below.
- All ten July social exports were inspected at @2x and match their handoff
  dimensions.
- `41` focused language/README tests passed; the README translation drift
  script passed.
- The strict MkDocs build passed for English, Chinese, and Hindi and repeated
  the orphan-page/cross-locale warnings.
- The final full unit suite passed: `8,835 passed, 70 skipped, 15 warnings`.
- Pre-commit passed on this plan after its trailing-whitespace hook normalized
  the file.

No website deployment, social-profile edit, social post, external asset
upload, or Hugging Face setting was changed during the audit.

## Current implementation map

The migration must account for these owners and consumers:

| Responsibility | Current repository path |
|---|---|
| Marketing HTML | `docs/website/index.html` |
| Marketing styles/interactions | `docs/website/assets/style.css`, `docs/website/assets/script.js` |
| Marketing icons/marks/OG | `docs/website/`, `docs/website/brand/` |
| Documentation configuration | `mkdocs.yml` |
| Material brand shim | `docs/stylesheets/openmed-brand.css` |
| Documentation corpus | `docs/**/*.md` |
| Generated benchmark UI | `openmed/eval/leaderboard.py` → `docs/eval/benchmark-leaderboard/` |
| Browser demo | `docs/demo/web/` |
| Social outputs/instructions | `docs/brand/social/` |
| README distribution art | `docs/brand/openmed-readme-banner.png`, root README files |
| Pages build/staging | `.github/workflows/pages.yml`, `Makefile` `docs-stage` |

The final validation target is the staged `site/` tree after MkDocs builds
`site/docs/` and `docs/website/` is copied over it, including generated
leaderboard artifacts. Testing only source files or a standalone MkDocs build
is incomplete.

## Findings

### P0 — establish one versioned source of truth

The July handoff is outside the repository. The website implements a
hand-maintained subset of the older April paper/ink system, the docs define
only a small Material palette shim, and the current social render sources were
deleted. No repository file is the complete current visual authority.

The July root system is materially different:

- IBM Plex Sans/Mono replaces the current Newsreader/Inter Tight/JetBrains Mono
  shared stack;
- cool near-white/blue-black semantic surfaces replace warm paper/ink;
- all tint roles derive from `--om-accent-base` with OKLCH relative color;
- cards use 10/6/4/3 px radii, hairlines, and no shadow except the terminal;
- the terminal and final CTA remain ink in both themes;
- one 900 px breakpoint governs shared behavior;
- emoji, icon fonts, decorative gradients, blur, texture, and lift effects are
  excluded from the shared system.

Import only the July root tokens/assets/guidelines, the final OpenMed.life
source, and the approved social exports. Explicitly exclude `_ds/`, generated
`support.js`, uploaded runtime libraries, design-canvas controls, alternate
review variants, and repository-authored social redraws. Future site-system
changes must be applied to a repository-owned canonical source and tested for
consumer drift; future social-pixel changes must be re-exported from the
reviewed source DC and imported with owner approval.

### P0 — prototype content and runtime cannot ship

The new design files intentionally prioritize visual review. They load fonts
from Google, use generated design-canvas infrastructure, pull React/ReactDOM
and Babel from public CDNs, and fetch GitHub state in the browser. Their copy
also contains obsolete or unverified 1.8.x/1.9.1 API, package, service,
language, performance, release-cadence, and model-count claims.

Rebuild the approved visual structure in production-native files. Every code
sample and product/link/license statement must be checked against the exact
2.0 repository before publication. A visual match is not acceptance if it
introduces a non-existent package, endpoint, runtime, model count, or
compliance claim.

The supplied social exports remain the recorded exception to this substitution
rule: their baked copy is preserved with the exact approved pixels, without
being relabeled as independently verified. A change requires source-DC
correction, re-export, owner review, and provenance update.

### P0 — public claims have no shared registry

The audited surfaces publish different values for model count, supported
languages, model-backed languages, model downloads, monthly downloads, and
package installs. These are not always the same metric, but the public copy
does not consistently explain the distinction.

Examples observed during the audit include:

- 2,000+ models on the website and README;
- 2,200+ models and 21 languages in the GitHub repository description;
- 1,500+ models in Hugging Face organization presentation;
- 1,000+ models in LinkedIn About copy;
- 34 supported PII language codes versus 29/30/32 "model-backed" values;
- 16 languages in the live Hugging Face organization card versus 17 in a
  tracked card;
- 9.4M installs on the website versus 10M+ in tracked social art.

The implementation must distinguish supported language codes, model-backed
routes, optional adapter routes, validation-only locale coverage, model
downloads, monthly downloads, and package installs. A rounded number is still
a dated claim and needs a definition, source, owner, and `as_of` date.

### P0 — tracked social assets do not describe live state

The repository's social README documents ten PNG deliverables and their
destinations, but several tracked files are not installed on the named
platforms:

- GitHub uses its generated repository preview rather than the tracked
  `github-social.png`.
- The live X header and avatar differ from the documented `x-header.png` and
  cat avatar.
- The Hugging Face organization uses an Open Cross avatar and a separate card
  with stale copy rather than the tracked `hf-card.png`.
- LinkedIn uses an OpenMed Agent cover and an older cyan-heart avatar.
- The production website uses `docs/website/og.png`, which differs from
  `docs/brand/social/og.png`.
- All 15 root README files use `openmed-mascot-lockup.png`; none use either
  documented README banner.

The live platform state, tracked assets, and documentation must agree after
cutover. Platform uploads remain manual/external actions and require separate
authorization.

### P0 — social renders are not reproducible

`docs/brand/social/README.md` says the PNGs are rendered from standalone HTML
under `docs/brand/social/_src/` and canonical files under
`docs/brand/assets/`. The canonical asset directory exists and contains logo,
cat, and mascot inputs; the `_src/` render directory is missing. The README
also says that two rendered files are copied to distribution paths, but both
pairs have different hashes:

- `docs/brand/social/og.png` and `docs/website/og.png`;
- `docs/brand/social/readme-banner.png` and
  `docs/brand/openmed-readme-banner.png`.

No current test detects these failures.

### P1 — marketing interactions need semantic and motion hardening

The production site works at the sampled 390 px and desktop widths, but its
header makes a 320 px document 352 px wide. At 667×320, the open mobile menu
extends below the viewport without a usable scroll strategy. The following
contracts are incomplete:

- The icon-only mobile GitHub link has no accessible name after its visible
  label is hidden.
- There is no skip link.
- FAQ rows are clickable `div` elements rather than buttons; they are absent
  from the keyboard tab order and expose no expanded state. The open
  `max-height` clips the longest current answer.
- Code tabs do not expose tab/list roles or `aria-selected`.
- Model filters do not expose pressed/selected state.
- Scrollspy does not expose `aria-current`; copy feedback has no live region.
- The animated PHI demo continues under `prefers-reduced-motion`.
- Light is forced on first visit rather than respecting the operating-system
  preference.
- The preview accent text on the ink panel measures about 3.86:1, below AA for
  normal text.

The July target should retain the current product-rich coverage while
recomposing it into the approved nav, ownership hero, community wall,
terminal-first quickstart, comparison matrix, runtimes, privacy, models,
products, research, FAQ, final CTA, and footer. Do not preserve the current
visual system merely because the content remains, and do not copy stale
prototype claims or obsolete API examples merely because the layout is
approved.

### P1 — metadata and claims are duplicated by hand

The website repeats titles, descriptions, FAQ copy, product naming, and claims
across visible HTML, Open Graph/X tags, JSON-LD, JavaScript, and PNG artwork.
The JSON-LD describes an "LLM Suite" while the core library primarily exposes
task-specific encoders. `sameAs` omits the GitHub identity, cache-busting query
keys are manually dated, and theme metadata/manifest coverage is incomplete.
Unused legacy assets and CSS also remain in the website tree.

Generate metadata and structured data from reviewed sources, crawl every
first-party URL, and inventory unused assets before cutover. Competitive,
performance, encryption, HIPAA/Safe Harbor, release-cadence, dataset-access,
non-profit, and product-medical claims require evidence or careful
qualification before publication.

### P1 — docs carry color, not the complete brand

`docs/stylesheets/openmed-brand.css` contains five Material theme declarations
covering clinical teal and the dark accent. It does not share the July
surface, type, spacing, radius, focus, accent-engine, or component tokens.
`mkdocs.yml` still configures Inter Tight and JetBrains Mono. The standalone
handoff docs kit demonstrates useful shell, sidebar, TOC, code, table,
admonition, tab, and pager patterns, but it is only a six-page mock with stale
1.8.2/17-language/1,500-model content and a conflicting blue accent. Adapt its
visual grammar into Material; do not replace the repository's 168-page
documentation corpus with it.

The docs also link the founder's X account while the website and brand metadata
link the OpenMed organization account. This should be an explicit information
architecture decision, not accidental drift.

Production docs emitted 404 requests for localized Hindi and Chinese sitemap
files during the browser audit. Responsive desktop/mobile layouts and theme
switching otherwise worked in the sampled checks. The current mobile drawer
control is not keyboard-focusable in the reviewed state, and the handoff mock's
tab behavior does not provide the full ARIA/keyboard contract; both must be
correct in the Material adaptation.

### P0 — generated localization currently misrepresents coverage

The English documentation has 168 Markdown pages, while only six translated
source pages exist. Material's `fallback_to_default` currently emits roughly
161 English pages under each `zh/` and `hi/` path with localized `lang`,
canonical, and hreflang metadata. That presents fallback English as translated
documentation. The single audited search artifact contained 1,297 records:
1,242 English, 33 Hindi, and 22 Chinese. The defect is disagreement among
emitted routes, declared page language, canonical/hreflang, switch targets,
and search coverage—not wholesale indexing of every fallback page.

The built Pages artifact also contains two broken cross-locale links:

- `docs/onboarding-china.md:3` emits `/docs/hi/onboarding-china/` with
  `../zh/onboarding-china/`, which the browser resolves to missing
  `/docs/hi/zh/onboarding-china/`;
- `docs/onboarding-india.md:3` emits `/docs/zh/onboarding-india/` with
  `../hi/onboarding-india/`, which the browser resolves to missing
  `/docs/zh/hi/onboarding-india/`.

Lock the route policy before visual cutover:

1. Disable `fallback_to_default`; publish `/zh/` and `/hi/` routes only for
   genuine translated source files.
2. Generate an exact route/translation-group manifest from English and
   localized sources.
3. Emit self-canonicals and hreflang only for real pages in a translation
   group, with English as `x-default`.
4. A language selector links to the exact counterpart when one exists.
   Otherwise it clearly labels the translation unavailable and links to that
   language's real landing page—never to a fabricated same-page route.
5. Generate and advertise locale sitemaps only when their exact files and
   genuine translated routes exist.
6. Index every real translated route exactly once; no English fallback is
   indexed under a localized URL; every language-switch target must exist.

### P1 — documentation IA and custom pages bypass shared governance

The current nav has 151 leaves/147 unique destinations, four duplicated
release-page entries, and 18 Markdown pages outside navigation. Of 166 audited
English pages, 164 reuse the generic site description and generated docs pages
have no explicit Open Graph metadata. Localized landing pages omit the English
ONNX/WebGPU destination; localized getting-started pages omit the low-bandwidth
install destination. `docs/index.md` lines 99–119 and localized equivalents
render as a flat 21-item ordered list rather than the intended nested
structure. Generated Chinese/Hindi pages also duplicate `_mkdocstrings.css`.

The generated benchmark leaderboard and browser demo are separate blue/green
HTML applications without the shared navigation, metadata, accessibility, or
token contract. The leaderboard uses click-only tabs; the demo lacks canonical
metadata and shared chrome. The leaderboard source of truth is
`openmed/eval/leaderboard.py`, not its generated HTML; the demo source is
`docs/demo/web/index.html`. Bring both into the same
design/metadata/test boundary and require clean regeneration.

### P1 — the local-first promise and browser network behavior diverge

The production website and docs request Google Fonts and GitHub repository
data client-side; the marketing page also loads Cloudflare browser analytics.
These requests do not transmit clinical content, but an unexplained
third-party request footprint weakens a privacy-first, local-first trust story.

Production should self-host the OFL font files and licenses. Replace live
client-side repository/release metrics on both website and docs with the
checked-in claims snapshot. Only an explicit maintainer refresh may use the
network; normal builds transform the snapshot offline. Disable the Material
repository integration if it still triggers browser API requests. Remove the
current Cloudflare browser analytics from this rollout. Any later analytics
proposal is a separate, explicitly reviewed change with minimization,
disclosure/consent, and request tests. This is a product trust requirement,
not legal advice.

### P1 — README localization tests do not govern claim freshness

Existing README drift checks guard structure and selected source relationships,
but several translations still publish older PII-language counts. Structural
parity is not claim parity. Generated claim fragments or explicit localized
claim fixtures should make numeric drift testable without rewriting prose.

### P2 — the identity register is platform-specific

The July handoff supplies a cross-in-rounded-square mark, lowercase Plex
wordmark, cat crest, and exact platform assignments. The current repository
and live profiles mix the cross, cat, older heart, old mascot lockups, and
product-specific artwork.

Do not flatten the new assignments into an inaccurate "official versus
community" rule. Record each use explicitly:

| Context | Approved handoff asset |
|---|---|
| Website/docs chrome and favicon | Open Cross tile + lowercase visual wordmark |
| Hugging Face `OpenMed` organization avatar | Cat crest |
| Future named OpenMed GitHub organization avatar | Cat crest; not applicable to personal owner |
| X avatar | Open Cross in circular crop |
| LinkedIn company logo tile | Open Cross |
| README and GitHub/Hugging Face cards | Cat crest + Open Cross lockup |
| Website OG and utility contexts | Open Cross |

Publish minimum-size, clear-space, light/dark, crop-safe, alt-text, and misuse
guidance for each role. Filenames and the render manifest should encode the
role rather than imply that one avatar works everywhere.

## Target source layout

Build on the existing `docs/brand/` location instead of introducing another
brand root:

```text
docs/brand/
├── README.md
├── system/
│   ├── tokens/
│   │   ├── colors.css
│   │   ├── typography.css
│   │   └── surfaces.css
│   ├── tokens.json
│   ├── voice.md
│   ├── iconography.md
│   ├── site-exceptions.md
│   ├── claims.yml
│   ├── publication.yml
│   └── handoff-provenance.json
├── assets/
│   ├── open-cross.svg
│   ├── open-cross-inverse.svg
│   ├── openmed-wordmark.svg
│   ├── cat-crest.*
│   └── fonts/
├── social/
│   ├── exports/
│   ├── _src/
│   │   ├── exports.json
│   │   └── profile-copy.json
│   ├── manifest.json
│   ├── README.md
│   └── exact master copies and declared PNG/icon derivatives
scripts/brand/
├── validate_system.py
├── update_claims.py
└── render_social_assets.*
scripts/docs/
└── stage_pages.py
tests/unit/
├── test_brand_system.py
└── test_pages_manifest.py
tests/browser/brand/
└── pinned three-engine flows and snapshots
```

The split CSS tokens are the browser source of truth and retain the handoff's
file boundaries. `tokens.json` contains equivalent platform-neutral roles for
generated consumers; it must be validated against the CSS, not maintained
independently. Website and docs styles may add surface-specific component
rules and documented exceptions, but may not redefine raw palette values or
derive private accent shades. Provide deterministic fallback values before
relative-OKLCH declarations so the site remains usable when relative-color
syntax is unavailable; verify both the fallback and enhanced paths.

`handoff-provenance.json` records the reviewed handoff filenames, hashes,
import date, excluded legacy/runtime paths, and approved deviations. It must
not depend on a contributor's Downloads directory or a private repository.
Its approved-export hash map and social source exception bind the repository
copies to the reviewed handoff without making a local Downloads path a runtime
dependency.
`publication.yml` classifies navigated/link-only/excluded docs, real
translation groups, expected routes, metadata policy, and approved exceptions;
the staged route manifest is generated from it rather than maintained by hand.

## Claims contract

`claims.yml` should contain, at minimum:

- package/release version;
- Hugging Face repositories owned by the `OpenMed` organization;
- broader curated/compatible catalog model count across owners;
- exact inclusion, exclusion, de-duplication, rounding, and `as_of` rules for
  each count;
- supported PII language-code count and exact codes;
- model-backed PII language count, exact codes, definition, and deterministic
  derivation from the language-pack/model catalog;
- optional user-configured adapter routes;
- model, PII-checkpoint, MLX-variant, dataset, and PII entity-type counts;
- cumulative model downloads;
- monthly model downloads;
- cumulative package installs;
- release cadence and community/founding timeline;
- benchmark/performance claims with hardware, method, result, and date;
- research/SOTA claims with paper citation and reproducibility evidence;
- local/offline/telemetry/network behavior by runtime;
- compliance and de-identification wording with evidence and qualification;
- competitive-matrix sources and `as_of` dates;
- product maturity and medical-decision disclaimers;
- dataset access/license restrictions and bundled-versus-user-supplied status;
- license by product surface;
- evidence link, approved public wording, required qualification, and owner;
- `as_of` plus expiry/review dates for every verified/publishable field, and a
  separate `follow_up_by` deadline for every deliberately unverified field.

Rules:

1. CI is offline and validates a committed snapshot. A network refresh is an
   explicit maintainer action.
2. Every numeric, performance, privacy, compliance, competitive, license, and
   availability claim in website metadata, visible website copy, every
   documentation page/code example, all content in all 15 READMEs, social
   source files, and profile-copy templates either comes from the registry or
   is allow-listed with a reason and expiry.
   The baked copy inside the ten owner-approved social exports is the explicit
   source exception: preserve the bytes, record that the claims are not
   independently verified by the registry, and require a source-DC re-export
   plus owner review for any change.
3. Dynamic metrics use conservative rounded display values and an `as_of`
   date where precision could imply live data.
4. OpenMed library/models/datasets and OpenMed Agent/Welna licensing are
   separate claims. Do not imply that closed products are Apache-2.0.
5. Supported, model-backed, optional, and validator-only language coverage
   remain distinct.
6. HIPAA/Safe Harbor, encryption, "no outbound calls," local/on-device,
   SOTA/accuracy, latency/speedup, and "ships weekly" copy is published only
   with scoped evidence; the library does not make a deployment compliant by
   itself.
7. Dataset names never imply bundled access when a DUA, credential, or
   user-supplied corpus is required. Founder/community facts never imply
   non-profit status without organizational evidence.
8. Training-corpus provenance, medical-device/product boundaries, and every
   product's license/maturity are governed text claims, not free marketing
   copy.
9. Unverified superlatives such as "#1 on Hugging Face" are prohibited, as is
   calling a population of models "open-source" without verified licenses for
   that exact population.

## Implementation sequence

### Phase 0 — lock the system

- Import the July root token files, logo, relevant guideline/component intent,
  OpenMed.life final source/README, social source/README, and all ten approved
  raster exports as the sole canonical social visuals.
- Explicitly exclude `_ds`, generated support/runtime files, uploads, alternate
  tweak variants, stale mock documentation copy, and repository-authored
  social redraws.
- Inventory the existing `docs/brand/assets/` inputs and map each to retained,
  superseded, or removed; do not discard the present canonical logo/cat assets
  merely because the render-source directory is missing.
- Record the OpenMed.life red accent, Newsreader/gradient exceptions,
  lowercase visual wordmark, and platform-specific cat/cross assignments.
- Mark older April website styling and separate app/demo themes as
  non-canonical for OpenMed.life.
- Create the claims registry and populate it from repository/runtime truth.
- Add ownership for design tokens, claims, social-export preservation and
  derivatives, and platform
  uploads.

**Exit gate:** one documented visual authority, one platform-aware asset
register, one reviewed claims snapshot, and a hash-complete provenance record.

### Phase 1 — make assets deterministic

- Copy the ten approved @2x PNGs into `docs/brand/social/exports/` without
  changing their bytes. Record the exact handoff hashes in provenance.
- Declare dimensions, platform roles, safe zones, exact aliases, derivatives,
  and consumer copies in `_src/exports.json`. This schema maps approved files;
  it does not define or reconstruct a visual composition.
- Store canonical Open Cross, wordmark, cat crest, and crop/safe-zone sources.
  Preserve the shared mark geometry but generate an OpenMed.life red-dot
  variant rather than shipping the root asset's tangerine default.
- Self-host IBM Plex Sans, IBM Plex Mono, and the OpenMed.life-only Newsreader
  subset with OFL license files and recorded font-file hashes.
- Make the asset tool exact-copy each approved @2x source into its top-level
  master name, with `og-2x.png` and `hf-card-2x.png` as explicit aliases of
  `og-website-2x.png` and `hf-org-2x.png`. Produce native outputs only as
  deterministic half-size LANCZOS derivatives.
- Add a derived 180×180 Apple touch icon resized from the approved
  `favicon-2x.png`. Store it at
  `docs/brand/social/apple-touch-180.png` and copy identical bytes to
  `docs/website/apple-touch-icon.png`.
- Generate favicon ICO frames, consumer copies, and crop-safe review overlays
  only from the approved masters or their declared native derivatives.
- Produce all derivatives in temporary output with networking disabled, then
  verify canonical source hashes, exact master aliases, dimensions, derivation
  rules, crop-safe overlays, and distribution-copy equality.
- Add a manifest with source, output, width, height, role, safe zone, and
  DPR, destination, expected approved-source hash, and copy/derivative rule for
  every asset.

**Exit gate:** a clean checkout verifies the immutable approved art and
reproduces every declared derivative without network access; source/master
equality, @2x/native mapping, and distribution-copy equality are tested.

### Phase 2 — harden and consolidate the website

- Reimplement the handoff's final ownership hero, community wall,
  terminal-left quickstart, comparison, runtime, privacy, model, product,
  research, FAQ, CTA, and footer composition in the current static site.
- Make the website consume the canonical split tokens with
  `--om-accent-base: #B0413E`; record only the approved Newsreader/gradient
  exceptions.
- Preserve current valid 2.x content and API examples while fitting it into
  the new composition. Delete obsolete hand-maintained legacy style/assets
  only after a reference audit proves they are unused.
- Replace duplicated metrics and metadata claims with generated fragments.
- Self-host fonts; replace client-side GitHub stats with a checked-in snapshot
  that normal offline builds only transform. A separate explicit maintainer
  refresh command may use the network to update evidence and the committed
  snapshot; normal CI and Pages builds may not. Remove the current Cloudflare
  browser analytics.
- Add accessible names, native FAQ buttons, tab semantics, selected filter
  state, keyboard behavior, focus order, live feedback, skip navigation, and
  reduced-motion behavior.
- Implement light/dark theme handling with an OS-resolved first visit,
  persistence, a scroll-safe mobile menu, copy feedback,
  no-horizontal-overflow down to 320 px, no-JavaScript
  fallback, and 400% zoom/320-CSS-pixel reflow plus text-spacing resilience.
- Generate canonical, Open Graph, X card, JSON-LD, favicon, Apple touch, and
  structured FAQ metadata from reviewed sources. Add validated
  `theme-color`/`color-scheme`, web manifest, `og:image:alt`, X image alt, and
  a required `sameAs` identity set for the GitHub repository, Hugging Face
  organization, X account, and LinkedIn company.
- Audit unused styles, images, scripts, event handlers, selectors, and
  generated fragments—including `initCopyInstall()`—before removing or
  carrying them forward.

**Exit gate:** zero unwaived accessibility violations, zero console errors,
zero unexpected third-party requests, and visual approval at all target
viewports/themes. Any waiver is owned, justified, time-bounded, and tested not
to hide keyboard or content access.

### Phase 3 — extend the system to documentation

- Map July semantic roles and the red accent engine into Material variables;
  do not copy the mock docs' blue override.
- Apply IBM Plex Sans to headings/prose and IBM Plex Mono to code/labels.
  Introduce Newsreader only where the approved OpenMed.life exception improves
  a display metric without reducing long-form readability.
- Share official marks, focus treatment, cool surfaces, borders, spacing,
  radii, and light/dark behavior while retaining Material's mature
  navigation/search semantics.
- Check navigation, search, code copy, tabs, admonitions, tables, API
  references, generated pages, print styles, forced colors, a named synthetic
  RTL fixture, and all real localized variants.
- Choose and document organization versus founder social links.
- Disable localized fallback emission and generate the exact route/translation
  manifest described above; fix the two nested cross-locale URLs and sitemap
  references.
- Add parity tests requiring real Chinese/Hindi translations to account for
  current release/version/claim/required-section changes, including
  ONNX/WebGPU and low-bandwidth-install destinations.
- Repair the landing-page nested-list DOM in English and localized sources.
- Reconcile the 151-leaf nav and four duplicate release entries. Classify
  every Markdown file as navigated, intentionally link-only, or excluded;
  require every link-only page to be reachable from an indexed page.
- Make the mobile drawer trigger focusable and named with `aria-expanded`;
  require keyboard open/close, Escape, visible focus, focus containment while
  open, and focus return.
- Update `openmed/eval/leaderboard.py` (not generated output) and
  `docs/demo/web/index.html` to implement shared chrome, metadata, keyboard,
  contrast, and responsive contracts; regeneration must leave the worktree
  clean.
- Give every indexable route a unique title, useful non-boilerplate
  description, canonical, and explicit OG/X policy or reviewed exception.
- Ensure each generated page includes each stylesheet/script once.

**Exit gate:** strict MkDocs build passes and representative landing,
guide, API, table, code-heavy, localized, generated, and synthetic RTL pages
pass browser and accessibility matrices; the build has zero unrecognized
relative-link diagnostics, non-navigation pages match an approved allowlist,
and the exact Pages artifact has zero internal broken links or
false-localization metadata.

### Phase 4 — distribute approved social presentation

- Import only the approved website OG, GitHub social, X header, Hugging Face
  card, LinkedIn cover, README banner, avatars, and favicon exports. Exact-copy
  their @2x masters and derive only the declared native, consumer, preview,
  Apple touch, and ICO outputs.
- Follow the handoff manifest exactly where the target exists: cat for the
  Hugging Face organization avatar; Open Cross for X circle, LinkedIn tile,
  favicon, and site chrome; combined cat/cross lockups for specified
  cards/banners. The handoff's "GitHub organization avatar" assignment is not
  applicable to the personal-owner repository `maziyarpanahi/openmed`; never
  change the maintainer's personal avatar. Revisit it only if a specific
  OpenMed GitHub organization login is identified and separately approved.
- Preserve the ten supplied @2x masters at 2400×1260 website/Hugging Face,
  2560×1280 GitHub, 3000×1000 X, 2560×640 README, 2256×382 LinkedIn,
  1024² cat, 800² X circle, 600² LinkedIn tile, and 128² favicon. Generate each
  native distribution file only as the declared half-size derivative, plus the
  favicon-derived 180² Apple touch icon and ICO frames.
- Update the English README first, then all 14 translations through the
  existing drift workflow. Each README must reference the canonical new banner
  exactly once, contain no old mascot-lockup reference, resolve to the approved
  output hash, and provide localized alt text according to the translation
  policy.
- Store platform-ready bio, tagline, About, pinned-post, and alt-text templates
  beside the export mapping. These text templates do not define image pixels.
- Include safe-zone previews for X, LinkedIn, GitHub, and Hugging Face crops.
- Require generated profile copy to say `Apache-2.0 SDK`; describe models as
  "open" rather than "open-source" unless each referenced license population
  is verified. Preserve the approved exports' baked copy under the recorded
  source exception; changing it requires a source-DC re-export and owner
  review, never a pixel edit.

**Exit gate:** all approved-source hashes, exact master aliases, derivative
dimensions, safe zones, distribution links, alt text, and generated profile
claim fragments pass offline validation.

### Phase 5 — authorized external cutover

After a separate explicit approval, update:

- GitHub repository `maziyarpanahi/openmed`: description, topics, website link,
  and social preview—never the personal-owner avatar;
- Hugging Face organization `OpenMed`: cat avatar plus the organization card
  stored in the existing Space repository `OpenMed/README`. Update only its
  `README.md` content/reference and native
  `openmed-social-card.png` distribution file from the approved
  `social/exports/hf-org-2x.png` master. Treat any auto-generated platform
  thumbnail as non-controllable unless a documented setting exists. Use a
  code/file push only—never read, change, or couple this work to the Space's
  visibility or settings;
- X `@OpenMed_AI`: Open Cross avatar, header, bio, link, and pinned post;
- LinkedIn company `openmed-ai`: Open Cross tile, cover, tagline, About, link,
  and featured content;
- any website deployment and cache purge required for OG/metadata changes.

The current Pages workflow validates pull requests without deploying them, but
deploys every push to `master`. Therefore merging this rollout branch into
`master` is itself the website-deployment authorization boundary. Keep the
branch or pull request unmerged until Phase 5 is explicitly authorized; a
routine code-review approval must not be interpreted as permission to trigger
the production Pages deployment.

For each surface, capture the uploaded source hash, resolved live CDN
URL/hash/dimensions, timestamped screenshot, and perceptual comparison.
Platforms may recompress images, so CDN byte equality is not required. Do not
change any Hugging Face Space visibility or couple organization branding
changes to Space settings.

**Exit gate:** live profiles match the manifest and claims registry after CDN
and social-card caches are refreshed.

### Phase 6 — final audit and release

- Run the complete test matrix at the exact final commit.
- Make `.github/workflows/pages.yml` and `Makefile` call one
  `scripts/docs/stage_pages.py` implementation. It must render the leaderboard
  with the resolved release tag, build MkDocs into `site/docs/`, verify feeds
  and real locale trees, reject any website-overlay collision, copy
  `docs/website/` to the artifact root, and emit an exact route/asset manifest.
- Browser-test and crawl that shared staged artifact. Its manifest must prove
  marketing owns `/`, MkDocs owns `/docs/`, generated leaderboard/demo/routes
  exist, all advertised feeds/sitemaps/assets exist, and the overlay did not
  replace docs output.
- Verify live metadata with social-card debuggers after deployment.
- After separately authorized deployment, crawl the live root, `/docs/`, every
  real locale root, leaderboard, demo, feeds, robots, canonical/OG assets, and
  every advertised sitemap; compare its route/metadata manifest with the exact
  staged commit.
- Re-run the surface inventory and account for every item below.
- Record exceptions with an owner and expiry date; an undocumented skip is a
  failed gate.

## Acceptance matrix

### Existing repository gates

```sh
uv run --extra dev --extra docs --frozen \
  pytest tests/unit/core/test_supported_languages_source_of_truth.py \
         tests/unit/test_docs_language_coherence.py \
         tests/unit/test_readme_i18n_drift.py \
         tests/unit/test_readme_sw_drift.py -q

uv run --extra dev --extra docs --frozen \
  python scripts/i18n/check_readme_drift.py

uv run --extra dev --extra docs --frozen mkdocs build --strict
```

Build the exact Pages artifact through the shared staging script, then crawl
that artifact rather than only MkDocs' default `site/`. When implementation
touches Python tooling, also run the normal formatter, lint, format-check, and
full unit suite. When it changes only static brand files, run pre-commit on the
exact tracked paths plus the brand tests.

### New automated gates to add

`tests/unit/test_brand_system.py` and `scripts/brand/validate_system.py` must
verify:

- handoff provenance and token CSS/JSON parity;
- required raw and semantic tokens;
- relative-OKLCH fallback values and derived-role contrast in both themes;
- consumer CSS imports canonical tokens and contains no unregistered raw
  accent shades, font families, or breakpoints; the docs blue override is
  absent; Newsreader, gradients, pills, animation, and non-900 px breakpoints
  occur only at their exact allow-listed OpenMed.life roles (including the
  1080 px comparison-overflow exception);
- brand chrome uses no emoji or icon fonts, and cards/controls use no
  unregistered shadow, blur, lift, or scale effect;
- canonical logo/wordmark/cat references and their platform assignments;
- all ten approved social exports, their exact provenance hashes, and canonical
  identity assets exist;
- exact source/master byte equality, @2x/native dimensions, DPR mapping, color
  mode, Apple touch/ICO derivation, and crop-safe overlays;
- the asset copy/derivative tool passes with network access disabled and never
  reconstructs a supplied composition;
- manifest output/destination pairs have identical bytes;
- all 15 READMEs use the canonical banner once, omit the old mascot lockup,
  resolve to the expected output hash, and contain policy-compliant alt text;
- numeric and non-numeric generated claims match the registry/allowlist;
  prohibited superlative, license, compliance, competitive, and availability
  phrases are absent outside the documented baked-export source exception;
- `Apache-2.0` modifies the SDK, and model populations do not inherit that
  license claim;
- structured data and visible website claims agree;
- organization/founder links match the recorded decision;
- remote font/icon/runtime imports and client-side GitHub fetches are absent
  from production;
- only real localized source routes are emitted; every translated route is
  indexed exactly once; every indexed language/URL, canonical, hreflang,
  sitemap, page language, and switch target agrees with the route manifest;
- translated pages pass required-section/link/release/claim parity, including
  ONNX/WebGPU and low-bandwidth-install coverage;
- nav destinations are unique unless allow-listed; duplicate release entries
  are absent; every source matches the navigated/link-only/excluded manifest,
  and every link-only page is reachable;
- the build emits zero unrecognized-relative-link diagnostics and its
  non-navigation diagnostics match the approved allowlist;
- generated pages include each stylesheet/script once;
- leaderboard generator and demo tests cover metadata, shared chrome, roles,
  keyboard tabs, and clean regeneration;
- internal links and every expected root/docs/locale/leaderboard/demo/feed/
  sitemap/asset route in the staged Pages manifest resolve; overlay ownership
  and collision checks pass;
- generated files are current and regeneration leaves the worktree clean.

### Browser matrix

Test the staged site, not only production:

| Surface | Viewports | Themes | Required flows |
|---|---|---|---|
| Website | 320×800, 390×844, 667×320, 768×1024, 1440×900, 1536×864 | light, dark; OS-resolved first visit | menu, theme persistence, anchors, code tabs/copy, filters, FAQ, external links |
| Website, JavaScript disabled | 390×844, 1440×900 | OS-resolved | meaningful content/code, PHI-demo fallback, visible FAQ answers, static year, install guidance |
| Docs landing | same | light, dark; OS-resolved first visit | nav drawer, search, locale links, theme, footer |
| Docs guide | 390×844, 1440×900 | light, dark | headings, admonitions, tables, code copy |
| Docs API page | 390×844, 1440×900 | light, dark | API nav, deep links, overflow, code |
| Real Chinese/Hindi page | 390×844, 1440×900 | light, dark | nav, code, exact translation switch, untranslated state |
| Synthetic RTL fixture | 390×844, 1440×900 | light, dark | direction, logical layout, nav, code |
| Benchmark leaderboard | 390×844, 1440×900 | light, dark | tabs, filters, table overflow, generated data |
| Browser demo | 390×844, 1440×900 | light, dark | model load/error, input/output, privacy copy |

For every row:

- no horizontal page overflow;
- no browser console error, failed first-party request, or sitemap 404;
- keyboard-only operation and visible focus;
- meaningful accessible names, roles, states, and heading hierarchy;
- `prefers-reduced-motion: reduce` disables non-essential animation;
- 400% zoom (or 320 CSS-pixel equivalent reflow) and WCAG text spacing do not
  hide content or require two-dimensional page scrolling;
- forced-colors/high-contrast mode retains structure, focus, and meaning;
- WCAG 2.2 AA contrast;
- automated and manual checks have zero unwaived accessibility violations;
- visual snapshots cover default, hover, focus, open, selected, copied, and
  error states where applicable.

Run at least Chromium, WebKit, and Firefox for the final candidate. A
Chromium-only pass is acceptable for iteration, not release. Pin the browser
toolchain/versions in a Pages or dedicated brand-preview CI job, define
reviewed snapshot tolerances, and retain screenshots, accessibility reports,
traces, console/network logs, and diffs as failure artifacts.

### Metadata, privacy, and performance gates

- Every indexable route has a unique title, useful non-boilerplate description,
  canonical, favicon, and an explicit Open Graph/X policy; applicable website
  routes include consistent JSON-LD. Website metadata includes validated
  theme colors, `color-scheme`, web manifest, image alt metadata, and the
  required identity set.
- OG images return 200 with correct MIME type and exact dimensions.
- Every sitemap/robots reference that is advertised returns 200; locale
  sitemaps contain only real translated routes.
- Every browser flow records requests and enforces an origin/method allowlist,
  not only first load. Any exception is documented and tested.
- No raw PHI or user-entered demo text appears in request URLs, headers,
  bodies, analytics, logs, storage, or audit artifacts. Demo fixtures remain
  synthetic.
- The rollout ships without browser analytics. Any separately approved future
  analytics change must make its disclosure/consent decision visible and
  testable; policy text alone is not sufficient.
- Font loading does not cause unusable layout shift; fallback metrics and
  preload policy are tested.
- Performance budgets are recorded before the visual cutover, then enforced
  for JavaScript, CSS, fonts, images, LCP, CLS, and INP.

### Social and profile gates

- Every canonical social master is byte-identical to its approved
  `social/exports/` source; every other raster is a declared derivative or
  distribution copy.
- Exact platform dimensions and crop-safe zones pass.
- Primary headlines and marks remain legible in platform thumbnail previews;
  supporting metadata is intentional fine print and is not claimed to be
  universally legible at thumbnail size.
- Each image has platform-appropriate alt text.
- Profile bio/About copy uses current claims and the correct product-license
  boundary.
- Baked export copy is reported as the owner-approved source exception and is
  changed only through a reviewed source-DC re-export.
- Published avatar, cover/card, description, link, and pinned content are
  captured after cache refresh with uploaded source hash, live CDN
  URL/hash/dimensions, screenshot, and perceptual comparison.
- Tracked live-state evidence contains no access token, cookie, or private
  account data.

## Repository status and recorded exceptions

The repository implementation covers phases 0–4 and the repository-owned
portion of phase 6. Phase 5 and every live-state check that depends on it
remain deliberately unexecuted because this plan does not authorize external
changes. An unchecked item below is not silently omitted; it is mapped to one
of these owned, time-bounded exceptions.

### Repository validation evidence

The final repository candidate passed:

- the full Python suite: `8,939 passed, 74 skipped, 15 warnings`;
- the macOS Chromium, Firefox, and WebKit browser matrix: `380 passed`,
  `46 skipped` by documented engine scope;
- the pinned Playwright 1.62 Linux Chromium, Firefox, and WebKit matrix:
  `380 passed`, `46 skipped` by documented engine scope; its Chromium subset
  passed all `142` tests, with `99` reviewed Chromium snapshots for each of
  macOS and Linux;
- strict English, Chinese, and Hindi MkDocs builds followed by exact Pages
  staging, crawler, and publication-manifest validation;
- deterministic checks for claims, web fonts, generated consumers, all
  `15` README files, approved social-export preservation and derivatives, plus
  the consolidated brand-system validator;
- hash-bound manual review of native social art, 320 px thumbnails, safe-zone
  overlays, and avatar crops, with primary headline/mark legibility separated
  from intentionally fine-print supporting metadata and recorded as
  repository-candidate evidence rather than platform-owner approval;
- `docs/brand/system/evidence/manual-accessibility-review.json`, a dated
  manual accessibility record for the website, documentation, leaderboard,
  browser demo, and synthetic RTL fixture across the governed viewport/theme
  and applicable forced-colors/print states, with keyboard, semantics, reflow,
  motion, contrast, axe, snapshot, and WebKit-tiling checks passing and no
  waivers; this is repository-candidate evidence, not user testing or
  live-platform approval;
- Ruff lint and format checks, JavaScript syntax checks, JSON and SVG
  validation, `actionlint`, `uv lock --check`, and resolution or pinning of
  all `153` remote GitHub Actions references; and
- a clean browser-test dependency install with no reported npm
  vulnerabilities.

No deployment, profile edit, cache refresh, social post, external upload,
Hugging Face setting change, or Hugging Face Space visibility operation was
performed as part of this repository validation.

| ID | Open item | Current control | Owner | Review by |
|---|---|---|---|---|
| E1 | Website deployment; GitHub, Hugging Face, X, and LinkedIn uploads/profile edits; cache refresh; live CDN and social-debugger evidence | Deterministic artifacts, profile-copy templates, crop previews, hashes, and the rollback/cutover runbook are ready in the repository. Execution requires explicit per-platform authorization. Because the Pages workflow deploys pushes to `master`, this branch must remain unmerged until website deployment is authorized. No deployment, profile, cache, social-post, external upload, or Hugging Face setting change was made. | Repository owner | 2026-10-29 |
| E2 | Exhaustive claim-to-registry substitution in every documentation prose fragment and code example | The registry governs the public website, structured data, generated social/profile copy, README facts, language/runtime/license boundaries, and prohibited-claim scans. The owner-approved exports' baked copy is the documented source exception and requires a source-DC re-export to change. Documentation examples remain source-authored and are checked for known stale versions, routes, links, and prohibited wording rather than generated field by field from `claims.yml`. | Documentation maintainer | 2026-10-29 |
| E3 | Field Interaction to Next Paint (INP) telemetry | The browser suite enforces a deterministic two-animation-frame interaction-latency budget, plus LCP and CLS lab budgets. No production analytics or real-user monitoring was added, so this proxy must not be reported as field INP. | Website maintainer | 2026-10-29 |

## Complete surface checklist

### Shared system

- [x] Palette, semantic roles, dark mode, typography, spacing, radius, border,
      elevation, motion, focus, and breakpoints
- [x] Cross, inverse cross, wordmark, cat crest, favicon, platform avatar
      crops, clear space, minimum sizes, and misuse guidance
- [x] July brand voice, capitalization, quantified-claim rules, product/license
      boundaries, and alt-text voice
- [x] Versioned provenance, ownership, changelog, and deprecation policy

### Marketing website

- [x] Header, desktop nav, mobile nav, theme control, release/stars snapshots
- [x] Ownership hero, PHI demo, community wall, terminal quickstart,
      comparison matrix, runtimes, privacy, models, products, research, FAQ,
      final CTA, and footer
- [x] Hover, focus, active, expanded, copied, reduced-motion, and applicable
      failure states; the static site has no runtime API request or loading
      state
- [x] Title, description, canonical, OG/X metadata, JSON-LD, FAQ schema,
      image alt metadata, theme metadata, manifest, identity set, favicon,
      touch icon, robots, sitemap, and `CNAME`
- [x] Desktop/tablet/mobile, light/dark with OS-resolved initial state, zoom,
      print, and no-JavaScript
      fallback

### Documentation

- [x] Landing page, primary/secondary navigation, search, language switch,
      version/release links, footer, social links
- [x] Guides, API reference, code blocks, tabs, copy controls, admonitions,
      tables, diagrams, benchmarks, generated pages, `llms.txt`, and print
- [x] Every real locale, including truthful translation state,
      canonical/hreflang/search/sitemap behavior
- [x] Named synthetic RTL fixture for logical layout validation
- [x] Benchmark leaderboard and browser demo shared chrome/metadata/states
- [x] Material theme tokens, shared fonts/assets, focus, contrast, and
      responsive overflow

### Repository presentation

- [x] English README and all 14 translations use the new banner exactly once
- [x] README banner hashes/paths, localized alt text, and old-lockup removal
- [x] GitHub social-preview artifact, profile-copy template, and
      default-generated fallback are ready in the repository
- [ ] Live GitHub description, topics, website link, and social-preview upload
      (E1)
- [x] Issue/PR templates reviewed; no user-visible brand migration was required

### Social and organization profiles

- [x] Website OG/X card and byte-identical website distribution copy
- [x] GitHub repository social-preview source; no personal-avatar change
- [x] Hugging Face cat/card sources, copy, safe zones, and explicit
      no-Space-setting policy
- [x] X Open Cross avatar/header sources, bio, link, pinned-post copy, and alt
      text
- [x] LinkedIn Open Cross tile/cover sources, tagline, About, link, and
      featured-content copy
- [x] Only owner-approved exports define social visuals; profile-copy templates
      remain text-only
- [x] Platform assignments match the approved manifest
- [ ] Live platform uploads, copy changes, cache refresh, and evidence capture
      (E1)

### Governance

- [x] Verified claims have source, definition, owner, `as_of`, and review
      cadence; unverified claims remain unpublished and have owned follow-up
      deadlines
- [x] Offline deterministic export preservation, derivation, and validation
- [x] Visual regression baselines and update review
- [x] Accessibility and privacy request budgets
- [x] Manual platform runbook, cache-refresh steps, rollback assets, and
      evidence requirements
- [x] No Hugging Face Space visibility operation

## Definition of done

This rollout is complete only when:

1. The handoff is versioned in the repository with approved deviations.
2. Website and docs consume the canonical tokens/assets/claims; README and
   social presentation consume only the approved exports and their declared
   derivatives; profile-copy templates consume the governed text claims.
3. Every approved social master matches its source hash, every derived asset is
   reproducible offline, and every distribution copy matches its declared
   source output.
4. Website and docs pass strict build, content, browser, accessibility,
   responsive, reduced-motion, metadata, privacy-request, and visual gates at
   the exact final commit.
5. All 15 README files pass structural and claim-parity checks.
6. Authorized live platform state matches the manifest after cache refresh,
   with evidence recorded.
7. Every checklist item is checked or has an explicit, owned, time-bounded
   exception.

The repository candidate can satisfy items 1–5 and 7 without performing
external work. The end-to-end live rollout remains open under E1 until item 6
is separately authorized and evidenced. E2 and E3 record the precise limits of
claim generation and performance measurement; neither may be presented as a
completed stronger guarantee.

## Safe rollout boundary

This plan does not authorize deployment, profile edits, social posts, external
asset uploads, or changes to any Hugging Face Space. Those are separate
production actions. The implementation should land as focused pull requests
in phase order. For this consolidated rollout, the pull request may be reviewed
and validated, but it must not merge to `master` until the website-deployment
portion of Phase 5 is explicitly authorized, because that merge triggers the
production Pages workflow.
