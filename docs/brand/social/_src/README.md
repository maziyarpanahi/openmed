# Supporting social records

The visual authority is the owner-approved handoff copied under `../exports/`.
Nothing in this directory may redraw, retokenize, or override those pixels.

`exports.json` records dimensions, roles, safe zones, exact top-level aliases,
declared derivatives, and distribution targets. `profile-copy.json` holds
profile text and alt-text templates that remain separate from baked image
copy. Neither file defines a new visual composition.

`scripts/brand/render_social_assets.py` performs only exact master copies and
declared size, icon, consumer, preview, and safe-zone derivatives. It does not
reconstruct the approved export art. Any visual or baked-copy change must be
made in the handoff's `OpenMed Social Cards.dc.html`, re-exported, reviewed by
the owner, and imported as a new canonical export set.
