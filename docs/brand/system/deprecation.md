# Brand system deprecation policy

Canonical sources under `docs/brand/system/`, `docs/brand/assets/`, and
`docs/brand/social/exports/`, plus `docs/brand/social/_src/exports.json` and the
text-only `docs/brand/social/_src/profile-copy.json`, are versioned together.
Generated consumer CSS, fonts, marks, README art, social distribution copies,
and social derivatives are never deprecated or edited in isolation.

For a breaking token, asset, or claim change:

1. Add the replacement to canonical source and document it in `CHANGELOG.md`.
2. Keep the old role for at least one reviewed migration unless retaining it
   would publish unsafe or false information.
3. Mark the old role with its replacement, owner, announcement date, and
   removal date. Removal dates must be time-bounded and must not precede the
   next scheduled review.
4. Regenerate every consumer and distribution copy in the same change.
5. Search the repository and staged site for the deprecated role or asset and
   test that no unapproved consumer remains.
6. Remove the old role only after all owned consumers migrate and rollback
   evidence is retained in the asset register or release record.

False, expired, unsafe, or license-misleading claims are removed immediately
rather than carried through a compatibility window. Historical inputs may stay
tracked for provenance, but a superseded input must not remain a current
consumer dependency.

The social exports are a deliberate source exception because their approved
pixels include baked handoff copy. Never deprecate or correct that copy by
editing, painting over, or reconstructing a PNG. Change it in
`OpenMed Social Cards.dc.html`, re-export the affected approved set, obtain
owner review, and replace the canonical exports, provenance hashes,
distribution copies, and derivatives together.
