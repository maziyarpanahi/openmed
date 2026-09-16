# Social and profile cutover runbook

This is a repository-only procedure. It does not authorize deployment, profile
edits, posts, uploads, cache purges, or any other external change. Obtain
separate explicit approval for the exact platform and target before starting a
cutover.

## Canonical-source boundary

Only the owner-approved files under `exports/` are canonical social visuals.
Upload their declared native derivatives from this directory; do not redraw,
retokenize, paint over, or replace them with a repository-authored
composition. `og-2x.png` and `hf-card-2x.png` are exact aliases of
`exports/og-website-2x.png` and `exports/hf-org-2x.png`.

The exports include baked handoff copy as an explicit owner-approved source
exception. If that copy must change, stop the cutover. Update
`OpenMed Social Cards.dc.html`, re-export, obtain owner review, update the
canonical files and provenance hashes, regenerate derivatives, and repeat
preflight. Never correct a claim by editing PNG pixels.

## Preflight

1. Run the locked brand validator and record the exact commit.
2. Confirm every `exports/` hash matches the approved provenance, then confirm
   `manifest.json`, `exports.json`, `profile-copy.json`, the claims snapshot,
   and every distribution asset are current at that commit.
3. Resolve the exact platform target and compare it with the `target` and
   `link` fields in `profile-copy.json`.
4. Hash the source file to be uploaded. Never include tokens, cookies, private
   account data, or patient data in evidence.
5. Capture the current live copy and art before changing it so rollback remains
   possible.

## Authorized platform procedures

### GitHub repository

Target only `maziyarpanahi/openmed`. Update the repository description, topics,
website link, or social preview only when each exact field is authorized.
Never change the personal owner avatar. Upload `github-social.png` for the
repository preview; it is the declared half-size derivative of
`exports/github-social-2x.png`. Use its recorded alt template where the
platform supports alt text.

### Hugging Face organization and card repository

The organization target is `OpenMed`; the card repository target is
`OpenMed/README`. Use the cat crest for the organization avatar and
`hf-card.png` for the card distribution file. These are the declared native
derivatives of `exports/avatar-cat-2x.png` and `exports/hf-org-2x.png`.

**Hard boundary:** use a code/file push only. Never read, change, or couple this
work to any Hugging Face Space visibility or settings. Never call a visibility
or repository-settings API, and never use a UI visibility control. If a task
appears to require a Space setting, stop and ask the owner.

### X

Target only `@OpenMed_AI`. Apply `avatar-circle-400.png`, `x-header.png`, bio,
first-party link, pinned-post copy, and relevant alt templates as separately
authorized fields. The two PNGs derive only from the corresponding approved
exports. Confirm the avatar and header crop previews before upload.

### LinkedIn

Target only the OpenMed company page at `openmed-ai`. Apply
`avatar-linkedin-300.png`, `linkedin-banner.png`, tagline, About, first-party
link, featured copy, and alt templates as separately authorized fields. The
two PNGs derive only from the corresponding approved exports.

### Website

Deploy only a saved, reviewed artifact from the exact commit. Verify canonical,
Open Graph, X-card, favicon, touch-icon, manifest, robots, and sitemap URLs
after deployment. A website deployment does not authorize a social-profile
change.

## Evidence capture

Record one row per changed field or asset:

| Field | Required evidence |
|---|---|
| Commit | Exact commit SHA and validation command result |
| Source | Repository path, SHA-256, dimensions, color mode/profile |
| Target | Platform, account/repository, field name, and authorization reference |
| Live result | Resolved live/CDN URL, retrieval timestamp, dimensions, and downloaded SHA-256 when available |
| Visual proof | Timestamped screenshot with secrets and private account data excluded |
| Comparison | Perceptual comparison against the approved source, with crop/recompression notes |
| Copy proof | Exact published copy and character count |
| Reviewer | Reviewer identity and approval timestamp |

Platform recompression may change bytes, so record both the source hash and live
hash and use the screenshot/perceptual record to explain an accepted
difference. Evidence must bind the distribution file back to its canonical
`exports/` source hash and declared derivative rule. Store evidence only in the
separately approved release-evidence location; do not put credentials or
private session state in the repository.

## Cache refresh and verification

After an authorized change, refresh only the relevant platform cache or social
card debugger. Re-resolve the public URL rather than trusting the upload
dialog. Verify the target account, copy, asset dimensions, crop, alt text,
first-party link, and timestamp. A cache refresh must not trigger a deployment
or profile edit that was not separately authorized.

## Rollback

Keep the pre-change copy, screenshot, live URL/hash, and source asset before
cutover. If verification fails, restore that exact prior copy and art, refresh
the same cache, and capture a second evidence row. Repository rollback uses the
previous reviewed manifest, canonical export set, and derived assets; never
edit a canonical or derived PNG or consumer copy by hand.

## Completion

A cutover is complete only when the live target matches the authorized
manifest/copy after cache refresh, all evidence fields are recorded, and no
unapproved target changed. Repository validation alone is not proof of live
state.
