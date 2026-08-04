# OpenMed.life exceptions

The shared system uses IBM Plex Sans and Mono, a cool clinical surface, tight
radii, one red accent engine, no decorative gradients, and quiet state motion.
Only the following OpenMed.life roles may deviate.

| Exception | Exact role | Guardrail | Owner | Reviewed | Review by |
|---|---|---|---|---|---|
| Newsreader | Editorial numerals, selected community/FAQ display copy, and social headlines | Never body, navigation, controls, code, or docs prose | Brand | 2026-07-29 | 2026-10-29 |
| Rotating gradient word | Hardware word in the ownership hero | Solid red fallback; no other gradient text | Website | 2026-07-29 | 2026-10-29 |
| Release/star pills | Live release and star metadata only | Must have accessible labels and registry-backed values | Website | 2026-07-29 | 2026-10-29 |
| Release pulse | Release-status indicator | Disabled by reduced motion; no other pulsing chrome | Website | 2026-07-29 | 2026-10-29 |
| Synthetic PHI animation | Explicitly synthetic demo | Static meaningful fallback and reduced-motion state | Website | 2026-07-29 | 2026-10-29 |
| FAQ transition | Disclosure open/close | Content remains available without JavaScript | Website | 2026-07-29 | 2026-10-29 |
| Comparison overflow | Comparison table at or below the 1080 px viewport exception | Keep an 850 px intrinsic table inside a keyboard-focusable horizontal scroll region; all other layout breakpoints use 900 px | Website/docs | 2026-07-29 | 2026-10-29 |

No exception permits blur, glass, texture, hover lift, scale animation, generic
pill controls, decorative page gradients, or private accent shades. New
exceptions require a documented role, accessibility guardrail, owner, review
date, and automated test before use.
