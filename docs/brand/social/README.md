# OpenMed social / OG artboards

Pixel-exact renders of the OpenMed 2026 brand social assets, built to the locked
design system (see `docs/brand/` and the brand package). One register per surface:
the **Open Cross mark** for official/product surfaces, the **cat** for community/social.

| File | Size | Ships to |
|---|---|---|
| `og.png` | 1200×630 | `og:image` / `twitter:image` on openmed.life (also copied to `docs/website/og.png`) |
| `github-social.png` | 1280×640 | GitHub repo → Settings → Social preview |
| `x-header.png` | 1500×500 | X @OpenMed_AI header (60px safe margins) |
| `hf-card.png` | 1200×630 | Hugging Face org page |
| `readme-banner.png` | 1280×320 | Top of repo README (also copied to `docs/brand/openmed-readme-banner.png`) |
| `avatar-square-512.png` | 512×512 | GitHub + HF org avatar (cat-head, cream) |
| `avatar-circle-400.png` | 400×400 | X avatar (cat-head, circle, transparent corners) |
| `avatar-linkedin-300.png` | 300×300 | LinkedIn avatar (Open Cross mark — official register) |
| `favicon-64.png` / `apple-touch-180.png` | 64 / 180 | Raster favicon / apple-touch fallbacks |

## Regenerate

The `.png` files are rendered from the standalone HTML in `_src/` (each references the
canonical assets under `docs/brand/assets/` and pulls fonts from Google Fonts). To rebuild,
screenshot each source at its exact size, e.g. with headless Chrome:

```sh
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars \
  --user-data-dir="$(mktemp -d)" --force-device-scale-factor=1 \
  --window-size=1200,630 --virtual-time-budget=6000 \
  --screenshot=og.png "file://$PWD/_src/og.html"
```

Do not edit the PNGs by hand or regenerate the mark/cat — only the shipped assets in
`docs/brand/assets/` are canonical.
