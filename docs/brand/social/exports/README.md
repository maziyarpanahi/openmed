# OpenMed — Social Assets Handoff

Rendered from `../OpenMed Social Cards.dc.html` (current design system: cool near-white substrate, IBM Plex Sans/Mono + Newsreader display, accent engine base `#B0413E`). All PNGs in `exports/` are **@2x** — halve for @1x.

## Files → placement

| File | @2x px | Native | Goes to |
|---|---|---|---|
| og-website-2x.png | 2400×1260 | 1200×630 | openmed.life `og:image` |
| github-social-2x.png | 2560×1280 | 1280×640 | GitHub repo → Settings → Social preview |
| x-header-2x.png | 3000×1000 | 1500×500 | X profile header (avatar overlaps bottom-left; right-aligned type keeps clear) |
| hf-org-2x.png | 2400×1260 | 1200×630 | huggingface.co/OpenMed org card |
| readme-banner-2x.png | 2560×640 | 1280×320 | GitHub README hero (`<img width="1280">`) |
| linkedin-banner-2x.png | 2256×382 | 1128×191 | LinkedIn company banner (left 300px kept clear for logo tile) |
| avatar-cat-2x.png | 1024² | 512² | GitHub org + HF org avatar |
| avatar-x-circle-2x.png | 800² | 400² | X avatar (pre-cropped circle, transparent corners) |
| avatar-linkedin-2x.png | 600² | 300² | LinkedIn logo tile |
| favicon-2x.png | 128² | 64² | favicon / touch icon source |

## Notes
- Accent is tweakable on the source DC (`accentBase` prop: madder `#B0413E` / blue `#4A92C8` / teal `#2BA5A5` / tangerine `#FF823A`) — retints logo dots, links, stats. Re-export after changing.
- Cat crest asset: `../assets/brand/cat-head.png` on `#FBF7EF`.
- Stats baked into copy: 2,000+ models · 340M+ downloads · 10M+ installs · Apache-2.0. Update in the DC, not in the PNGs.
