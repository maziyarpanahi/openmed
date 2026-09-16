# OpenMed Mascot — Historical Designer Brief

> **Status: superseded for new brand work.** This brief is retained to document
> the origin of the cat crest. The current source of truth is
> [`docs/brand/README.md`](README.md), especially
> [`system/iconography.md`](system/iconography.md) and the canonical tokens under
> `system/tokens/`. The cat is a community character, not the universal product
> mark: current platform assignments use the Open Cross for website/docs chrome,
> favicons, X, and LinkedIn, and reserve the cat for approved README/social
> compositions and the Hugging Face organization avatar.

---

## 1. About OpenMed (context for the illustrator)

OpenMed is an **Apache-2.0-licensed, local-first clinical AI SDK**. It extracts
biomedical entities and supports de-identification workflows on hardware the
operator controls. Core inference can run locally after required artifacts are
present; model downloads and configured integrations may use a network. Model
and dataset terms vary. The audience is developers, clinicians, and researchers
who care about **privacy, sovereignty, and trust**.

The one-line promise the mascot must embody:

> **"Local-first clinical AI on hardware you control."**

Repository snapshot on 2026-07-29: 4,700+ GitHub stars, 1,520 unique committed
catalog entries, 34 supported PII routes, and 33 model-backed routes. These
figures are governed by `system/claims.yml`; do not copy them into new art
without the snapshot date and required qualifications.

---

## 2. The big idea — a Guardian of your data

Every concept below is a variation on one theme: **a guardian that protects the
user's most sensitive data and keeps processing close.** The data is precious
(treasure / a flock / a vault); the mascot watches over it, fast and vigilant,
while the surrounding copy states the actual runtime and network boundary.

This single idea ties the mascot to OpenMed's exact value: **privacy, local-first,
on-device, sovereign, trustworthy.**

---

## 3. Personality & tone

| Be… | Not… |
| --- | --- |
| Protective, vigilant, watchful | Aggressive, scary, fang-baring |
| Warm, friendly, approachable | Cutesy-childish, babyish |
| Capable, fast, alert | Slow, sleepy, lazy |
| Calm, trustworthy, grounded | Chaotic, hyper, gimmicky |
| Clean and modern | Sterile-corporate or "generic AI chrome robot" |

Emotional read in one phrase: **"I've got this. Your data is safe with me."**

---

## 4. The three directions to sample

Please sketch all three so we can compare. Each should clearly be *guarding the
user's data* and should feature the OpenMed cross (see §5 + §9 for the legal note).

### A. Guardian Dragon 🐉 *(boldest / most heroic)*
- **Story:** In myth, dragons guard the treasure. The user's private medical data is
  the treasure; OpenMed is the dragon coiled protectively around it — on the user's
  own ground, never the cloud.
- **Visual:** a small-to-medium, *friendly-not-fearsome* dragon coiled around a glowing
  data vault / record / heart marked with the cross. Soft snout, large kind eyes,
  wings tucked or half-spread. If it breathes anything, a tiny harmless wisp — never
  aggressive fire.
- **Mood:** powerful but benevolent, vigilant, warm. Best fit for the big hero banner.
- **Poses to explore:** coiled-guarding (hero) · alert mid-flight (speed) ·
  perched-watching · a cute hatchling (for the small icon).

### B. Guardian Saint Bernard 🐕 *(warmest / most universally loved)*
- **Story:** the classic **Alpine rescue dog**, reimagined as the gentle protector of
  your health data. Its iconic neck barrel becomes a **medkit** — the rescuer that keeps
  your records safe at home.
- **Visual:** a big, calm Saint Bernard sitting guard — broad head, soft floppy ears,
  gentle soulful eyes, white muzzle blaze — with the signature little **rescue barrel**
  (marked with the coral cross) on its collar. Signature patches recolored from brown to
  teal to stay on-brand.
- **Mood:** loyal, reassuring, trustworthy, gentle-giant.
- **Poses to explore:** sitting-guard with barrel (hero) · ears-up alert (watching) ·
  curled around records at home (local-first) · head-only (cute icon).

### C. Griffin 🦅🦁 *(most regal / distinctive)*
- **Story:** the mythological **guardian of treasure** — eagle vision + lion strength =
  vigilance plus protection. An ancient sentinel standing over the user's data.
- **Visual:** a stylized griffin (eagle head, lion body, wings) perched over or
  shielding a vault/record bearing the cross. Heraldic, but **softened and rounded** to
  stay friendly — sharp-eyed yet kind, never corporate-aggressive.
- **Mood:** regal, powerful, ancient-guardian. Striking and ownable; less "cuddly."
- **Poses to explore:** perched-guarding (hero) · wings spread, shielding (privacy) ·
  in flight (speed) · crest/head emblem (icon).

### D. Persian Cat — Avicenna's heir 🐱 *(personal front-runner)*
- **Story:** the maintainer's own fluffy white **Persian cat**, styled as a tiny
  **Avicenna (Ibn Sina)** — guardian-physician of your health data, in the lineage of the
  most famous physician in Persian history.
- **Symbol found:** the **Canon of Medicine** (*al-Qānūn*) as an open book bearing the
  coral cross; a small Persian scholar's **turban**; **Persian-turquoise arabesque**
  tilework accents *(OpenMed's teal ≈ Persian turquoise / fīrūza)*. Plus the **OpenMed**
  wordmark beneath the cat.
- **Mood:** warm, wise, calm, proud, scholarly.
- **Poses to explore:** sitting-with-book (hero) · head-only with turban (cute icon) ·
  curled around the book (local-first).
- Full prompt: [`mascot-image-prompts.md`](mascot-image-prompts.md).

---

## 5. Current color palette

The July 2026 system replaced the former teal-led palette. New compositions use
one signal-red accent engine with cool light and blue-black dark surfaces.
Preserve the checked-in cat crest as-is; do not recolor that historical raster.

| Role | Hex | Notes |
| --- | --- | --- |
| **Signal accent** | `#B0413E` | Sole base for derived accent roles |
| **Light background** | `#F4F7F8` | Default page canvas |
| **Light surface** | `#FFFFFF` | Cards and raised content planes |
| **Light ink** | `#0E1116` | Primary text and linework |
| **Dark background** | `#0B0E13` | Dark page canvas |
| **Dark surface** | `#10151C` | Dark cards and content planes |
| **Dark ink** | `#E6EBEE` | Primary dark-theme text |

Style of color: **flat fills**, minimal shading, and no decorative gradients.
The single rotating website word is a registered site exception, not a mascot
art direction.

---

## 6. Art style & references

- **Flat vector** with a **subtle paper-grain / editorial texture** to match OpenMed's
  "medical-journal" brand feel.
- **Rounded, friendly shapes**; confident clean linework in ink.
- Aim for the *quality bar and scalability* of the best open-source mascots —
  **Ferris the crab (Rust), the Go gopher, GitHub's Octocat** — simple, instantly
  recognizable, works tiny or huge. **Original, not derivative** of any of these.
- Should feel at home next to a **dark terminal UI** and on the current cool
  light surface equally well.

---

## 7. Deliverables

1. **Primary mascot** — friendly 3/4 view, "on guard" but welcoming.
2. **Head-only community portrait** — legible at **32px and 16px**, without
   implying approval for product chrome, favicons, personal avatars, or app
   icons.
3. **Pose set** (per chosen direction): guarding/hero · watching/alert ·
   at-home/curled (local-first) · holding-a-shield-or-lock (privacy).
4. **Monochrome / single-color** version (for stamps, watermarks, embroidery).
5. **Wide community lockup** — mascot + Open Cross + lowercase visual wordmark,
   for approved README, GitHub, and Hugging Face cards. Use the exact governed
   artboard dimensions instead of the historical approximate size.

**File formats:** layered editable source (**SVG + Figma or AI**), plus exported
**transparent PNGs @1×/2×/3×**.

---

## 8. Technical constraints

- Must remain **clear and recognizable at 16–32px** — avoid fine detail that vanishes.
- Must work as a **flat single color** (silhouette + cross should still read).
- Keep the **cross legible** at every size.
- Design the head/face so it can stand alone as a community portrait. Product,
  favicon, and app assignments require a separate explicit decision.

---

## 9. ⚠️ Legal / brand safety — please read

- **Do NOT use the official red-cross-on-white emblem.** The Red Cross / Red Crescent
  emblem is legally protected (Geneva Conventions + national trademark law). Use our
  **coral or teal "plus" sign** or a clearly stylized cross instead — never the exact
  red cross on a white field.
- Make the character **original** — not a recognizable redraw of an existing brand
  mascot, game character, or stock illustration.

---

## 10. Don'ts

- ❌ No syringes, needles, blood, or realistic anatomy (clinical-scary).
- ❌ No bared fangs, weapons, or menacing poses (we're a *friendly* guardian).
- ❌ No generic chrome/metallic "AI robot" clichés.
- ❌ No heavy photorealism or gradients that won't print or scale.
- ❌ No off-palette colors (keep to §5).

---

## 11. AI-prompt starters (optional — for quick reference thumbnails)

> **Historical reference only.** These condensed prompts preserve the original
> exploration and its superseded teal/coral palette. Retokenize any approved
> future exploration against `system/tokens.json`; do not use these as current
> production-art instructions.

**Dragon**
```
Friendly mascot of a small teal dragon coiled protectively around a glowing data
vault marked with a coral (#C5453A) medical plus; soft snout, large kind eyes, wings
half-spread; teal (#0D6E6E) scales, cream (#F7F4EC) belly, ink (#0E1116) linework;
flat vector with subtle paper grain, minimal shading, no gradients, rounded friendly
shapes, modern open-source mascot style, warm editorial palette, white background,
centered, scalable icon.
```

**Sheepdog**
```
Friendly mascot of a fluffy sheepdog sitting on guard beside a small shield, attentive
ears up, a coral (#C5453A) cross tag on a teal (#0D6E6E) collar; cream (#F7F4EC) fur,
ink (#0E1116) linework; flat vector with subtle paper grain, minimal shading, no
gradients, rounded shapes, modern open-source mascot style, warm editorial palette,
white background, centered, scalable icon.
```

**Griffin**
```
Friendly heraldic griffin mascot (eagle head, lion body, wings), softened and rounded,
sharp but kind eyes, perched and shielding a vault marked with a coral (#C5453A) cross;
teal (#0D6E6E) plumage, cream (#F7F4EC) face and feathers, ink (#0E1116) linework,
yellow (#F5E27A) beak highlight; flat vector with subtle paper grain, minimal shading,
no gradients, modern open-source mascot style, warm editorial palette, white
background, centered, scalable icon.
```

---

## 12. What to send back & how we'll choose

1. **Round 1 — thumbnails:** 2–3 rough sketches for *each* of the three directions
   (loose, grayscale or single-color is fine). We pick a winning direction.
2. **Round 2 — refinement:** flesh out the chosen direction in full color with the
   core poses.
3. **Round 3 — final:** the full deliverable set in §7.

**Acceptance criteria:** reads instantly as a *friendly guardian*; legible at 32px;
works in flat single-color; uses the then-approved exploration palette; clearly
communicates protective, local-first intent without an absolute privacy claim;
is original; and works on both light and dark backgrounds.

---

*Brand reference for the illustrator: see `docs/website/og.png` (hero style),
the dark theme in `docs/website/index.html`, and the canonical
`docs/brand/assets/open-cross.svg`, `open-cross-inverse.svg`, and
`cat-crest.png` sources. The former production copy at
`docs/website/assets/openmed-tui-preview.png` was retired during the July 2026
design-system rollout because no published website surface used it.*
