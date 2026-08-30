# Brand browser validation

This package validates the exact artifact produced by
`scripts/docs/stage_pages.py`. It does not target the live site and does not
deploy anything.

The pinned Playwright suite covers Chromium, Firefox, and WebKit across the
marketing site, documentation, real Chinese and Hindi routes, generated
leaderboard, browser demo, and the non-indexed synthetic RTL fixture. The
matrix exercises light and dark across distinct mobile, landscape, tablet, and
desktop breakpoints. Focused cases cover both OS-resolved first-visit states,
print, a 400% zoom proxy, no-JavaScript, reduced-motion, text-spacing,
keyboard-focus, and forced-colors states.

Run the complete gate from the repository root:

```sh
make docs-browser-test
```

The test server accepts first-party requests only. The dedicated privacy spec
checks the raw URL, body, headers, and method of every request (including
same-origin requests) before routing. It disables trace, video, and screenshots,
stores only URL hashes, and recursively checks its JSON attachment before
writing it. A separate innocuous synthetic phrase drives the error-state visual
baseline. Optional GitHub metadata requests are fulfilled deterministically with
the site's offline response in tests, so initial page loads make no uncontrolled
external network requests.

Failure screenshots, videos, traces, console/network evidence, and the HTML
report are written below `output/playwright/` and uploaded by the Pages
workflow for 14 days. Reviewed baselines live under
`snapshots/<platform>/<project>/`. Representative renderer, breakpoint, and
interaction states have visual baselines; the broader matrix uses DOM,
accessibility, focus, overflow, privacy, and network assertions without
content-wide screenshots. The normal matrix uses reduced motion from page load
for deterministic accessibility scans and goldens; a separate clock-controlled
test advances every rotating word.
Refresh only after reviewing an intentional visual change:

```sh
npm --prefix tests/browser/brand test -- --project=chromium --update-snapshots
```

Linux CI baselines must be generated and reviewed in the pinned Playwright
container as a separate platform set:

```sh
docker run --rm --platform linux/amd64 --ipc=host \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work \
  mcr.microsoft.com/playwright:v1.62.0-noble@sha256:02bbb2155cd7109e3e9c741941097ed1608cf8b6fa44ee2595896da2bdc1f471 \
  /bin/bash -lc \
  'npm --prefix tests/browser/brand test -- --project=chromium --workers=3 --update-snapshots'
```

Run the same command without `--update-snapshots` after review. The screenshot
tolerance is `0.002` maximum differing pixels. `budgets.json` records per-file
caps, exact governed large payloads, total and content-hash-unique artifact
bytes, bounded duplicate bytes, zero source maps, representative first-party
route transfers, CLS/LCP, and a deterministic two-frame
interaction-latency proxy (not field INP).
