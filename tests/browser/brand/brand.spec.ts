import AxeBuilder from "@axe-core/playwright";
import {
  ConsoleMessage,
  Page,
  TestInfo,
  expect,
  test,
} from "@playwright/test";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../../..");
const artifactRoot = path.join(root, "site");
const MAX_FULL_PAGE_SCREENSHOT_DIMENSION = 30_000;

type Theme = "light" | "dark" | "system-light" | "system-dark";
type ThemeEngine = "material" | "standalone" | "website";
type Viewport = { name: string; width: number; height: number };
type Surface = {
  engine: ThemeEngine;
  name: string;
  path: string;
  themes: Theme[];
  viewports: Viewport[];
};

const fullViewports: Viewport[] = [
  { name: "mobile-320", width: 320, height: 800 },
  { name: "mobile-390", width: 390, height: 844 },
  { name: "landscape-667", width: 667, height: 320 },
  { name: "tablet-768", width: 768, height: 1024 },
  { name: "desktop-1440", width: 1440, height: 900 },
  { name: "desktop-1536", width: 1536, height: 864 },
];
const focusedViewports: Viewport[] = [
  { name: "mobile-390", width: 390, height: 844 },
  { name: "desktop-1440", width: 1440, height: 900 },
];

const surfaces: Surface[] = [
  {
    engine: "website",
    name: "website",
    path: "/",
    themes: ["light", "dark", "system-light", "system-dark"],
    viewports: fullViewports,
  },
  {
    engine: "material",
    name: "docs-landing",
    path: "/docs/",
    themes: ["light", "dark", "system-light", "system-dark"],
    viewports: fullViewports,
  },
  {
    engine: "material",
    name: "docs-guide",
    path: "/docs/getting-started/",
    themes: ["light", "dark"],
    viewports: focusedViewports,
  },
  {
    engine: "material",
    name: "docs-api",
    path: "/docs/api-reference/",
    themes: ["light", "dark"],
    viewports: focusedViewports,
  },
  {
    engine: "material",
    name: "docs-chinese",
    path: "/docs/zh/",
    themes: ["light", "dark"],
    viewports: focusedViewports,
  },
  {
    engine: "material",
    name: "docs-hindi",
    path: "/docs/hi/",
    themes: ["light", "dark"],
    viewports: focusedViewports,
  },
  {
    engine: "standalone",
    name: "leaderboard",
    path: "/docs/eval/benchmark-leaderboard/",
    themes: ["light", "dark"],
    viewports: focusedViewports,
  },
  {
    engine: "standalone",
    name: "browser-demo",
    path: "/docs/demo/web/",
    themes: ["light", "dark"],
    viewports: focusedViewports,
  },
  {
    engine: "standalone",
    name: "rtl-fixture",
    path: "/docs/demo/rtl/",
    themes: ["light", "dark"],
    viewports: focusedViewports,
  },
];

type Audit = {
  consoleErrors: string[];
  externalRequests: string[];
  failedRequests: string[];
  pageErrors: string[];
  responseErrors: string[];
  unexpectedMethods: string[];
  webSockets: string[];
};

function redactMarker(value: string, marker?: string): string {
  if (!marker || !value.includes(marker)) return value;
  const digest = crypto.createHash("sha256").update(marker).digest("hex");
  return value.split(marker).join(`[redacted-sha256:${digest}]`);
}

function isFirefoxSearchWorkerNavigationAbort(
  message: ConsoleMessage,
): boolean {
  const source = message.location().url;
  return (
    message.type() === "error"
    && message.text().includes('"NS_BINDING_ABORTED"')
    && /\/docs\/assets\/javascripts\/workers\/search\.[a-f0-9]+\.min\.js$/u.test(
      source,
    )
  );
}

function monitorPage(
  page: Page,
  baseURL: string | undefined,
  forbiddenMarker?: string,
): Audit {
  const expectedOrigin = new URL(
    baseURL ?? "http://127.0.0.1:4173",
  ).origin;
  const audit: Audit = {
    consoleErrors: [],
    externalRequests: [],
    failedRequests: [],
    pageErrors: [],
    responseErrors: [],
    unexpectedMethods: [],
    webSockets: [],
  };
  page.on("console", (message) => {
    if (
      message.type() === "error"
      && !isFirefoxSearchWorkerNavigationAbort(message)
    ) {
      audit.consoleErrors.push(redactMarker(message.text(), forbiddenMarker));
    }
  });
  page.on("pageerror", (error) =>
    audit.pageErrors.push(redactMarker(error.message, forbiddenMarker)),
  );
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (!["GET", "HEAD"].includes(request.method())) {
      audit.unexpectedMethods.push(
        redactMarker(`${request.method()} ${url.href}`, forbiddenMarker),
      );
    }
    if (
      (url.protocol === "http:" || url.protocol === "https:") &&
      url.origin !== expectedOrigin
    ) {
      audit.externalRequests.push(
        redactMarker(`${request.method()} ${url.href}`, forbiddenMarker),
      );
    }
  });
  page.on("requestfailed", (request) => {
    audit.failedRequests.push(
      redactMarker(
        `${request.method()} ${request.url()} ${
          request.failure()?.errorText ?? ""
        }`,
        forbiddenMarker,
      ),
    );
  });
  page.on("response", (response) => {
    if (response.status() >= 400 && response.url().startsWith(expectedOrigin)) {
      audit.responseErrors.push(
        redactMarker(
          `${response.status()} ${response.url()}`,
          forbiddenMarker,
        ),
      );
    }
  });
  page.on("websocket", (socket) => {
    const url = new URL(socket.url());
    if (url.origin !== expectedOrigin) {
      audit.webSockets.push(redactMarker(url.href, forbiddenMarker));
    }
  });
  return audit;
}

async function prepareTheme(page: Page, theme: Theme): Promise<void> {
  const mode = theme.startsWith("system") ? "system" : theme;
  const resolved = theme === "dark" || theme === "system-dark" ? "dark" : "light";
  await page.emulateMedia({
    colorScheme: resolved,
    reducedMotion: "no-preference",
  });
  await page.addInitScript(({ selectedMode, selectedTheme }) => {
    try {
      const markerKey = "__openmed_brand_theme_prepared";
      if (sessionStorage.getItem(markerKey) === "true") {
        return;
      }
      sessionStorage.setItem(markerKey, "true");
      if (selectedMode === "system") {
        localStorage.removeItem("openmed-theme");
        localStorage.removeItem("/docs/.__palette");
      } else {
        localStorage.setItem("openmed-theme", selectedMode);
        const dark = selectedTheme === "dark";
        localStorage.setItem(
          "/docs/.__palette",
          JSON.stringify({
            color: {
              accent: "custom",
              media: dark
                ? "(prefers-color-scheme: dark)"
                : "(prefers-color-scheme: light)",
              primary: "custom",
              scheme: dark ? "slate" : "default",
            },
            index: dark ? 2 : 1,
          }),
        );
      }
    } catch {
      // Storage may be unavailable in hardened browser contexts.
    }
  }, { selectedMode: mode, selectedTheme: resolved });
}

async function expectThemeInitialized(
  page: Page,
  surface: Surface,
  theme: Theme,
): Promise<void> {
  const mode = theme.startsWith("system") ? "system" : theme;
  const resolved = theme === "dark" || theme === "system-dark" ? "dark" : "light";
  if (surface.engine === "material") {
    await expect(page.locator("body")).toHaveAttribute(
      "data-md-color-scheme",
      resolved === "dark" ? "slate" : "default",
    );
    if ((page.viewportSize()?.width ?? 0) > 1220) {
      await expect(
        page.locator(".md-header .om-docs-mark:visible"),
      ).toHaveCount(1);
    }
    const mark =
      resolved === "dark"
        ? page.locator(".md-logo .om-docs-mark--inverse:visible").first()
        : page.locator(".md-logo .om-docs-mark--default:visible").first();
    await expect(mark).toHaveAttribute(
      "src",
      resolved === "dark"
        ? /openmed-mark-inverse\.svg$/
        : /openmed-mark\.svg$/,
    );
  } else if (surface.engine === "standalone") {
    await expect(page.locator("html")).toHaveAttribute("data-theme", resolved);
    await expect(page.locator("html")).toHaveAttribute("data-theme-mode", mode);
  } else if (mode === "system") {
    await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.+/);
    await expect(page.locator("html")).toHaveAttribute(
      "data-theme-preference",
      "system",
    );
  } else {
    await expect(page.locator("html")).toHaveAttribute("data-theme", mode);
    await expect(page.locator("html")).toHaveAttribute(
      "data-theme-preference",
      mode,
    );
  }

  const colors = await page.evaluate(() => ({
    background: getComputedStyle(document.body).backgroundColor,
    colorScheme: getComputedStyle(document.body).colorScheme,
  }));
  if (surface.engine === "website") {
    expect(colors.background).toBe(
      resolved === "dark" ? "rgb(11, 14, 19)" : "rgb(244, 247, 248)",
    );
  }
  expect(colors.colorScheme).toContain(resolved);
}

async function expectNoPageOverflow(page: Page): Promise<void> {
  const overflow = await page.evaluate(() => ({
    body: document.body.scrollWidth - document.body.clientWidth,
    document:
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  }));
  expect(overflow.body, JSON.stringify(overflow)).toBeLessThanOrEqual(1);
  expect(overflow.document, JSON.stringify(overflow)).toBeLessThanOrEqual(1);
}

function forwardLinkTabKey(browserName: string): "Alt+Tab" | "Tab" {
  return browserName === "webkit" && process.platform === "darwin"
    ? "Alt+Tab"
    : "Tab";
}

async function expectVisibleKeyboardFocus(page: Page): Promise<void> {
  for (let attempt = 0; attempt < 8; attempt += 1) {
    await page.keyboard.press("Tab");
    const focus = await page.evaluate(() => {
      const element = document.activeElement;
      if (!(element instanceof HTMLElement) || element === document.body) {
        return null;
      }
      const style = getComputedStyle(element);
      return {
        boxShadow: style.boxShadow,
        outlineStyle: style.outlineStyle,
        outlineWidth: Number.parseFloat(style.outlineWidth),
        tag: element.tagName,
      };
    });
    if (focus) {
      const visible =
        (focus.outlineStyle !== "none" && focus.outlineWidth > 0) ||
        focus.boxShadow !== "none";
      expect(visible, JSON.stringify(focus)).toBe(true);
      return;
    }
  }
  throw new Error("No keyboard-focusable element was reachable");
}

async function expectTextSpacingReflow(page: Page): Promise<void> {
  await page.addStyleTag({
    content: `
      * {
        letter-spacing: 0.12em !important;
        line-height: 1.5 !important;
        word-spacing: 0.16em !important;
      }
      p { margin-bottom: 2em !important; }
    `,
  });
  await expectNoPageOverflow(page);
}

function formatViolations(
  violations: Array<{
    help: string;
    id: string;
    nodes: Array<{ target: string[] }>;
  }>,
): string {
  return violations
    .map(
      (violation) =>
        `${violation.id}: ${violation.help}\n` +
        violation.nodes.map((node) => `  ${node.target.join(" ")}`).join("\n"),
    )
    .join("\n");
}

async function expectAccessible(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page })
    .withTags([
      "wcag2a",
      "wcag2aa",
      "wcag21a",
      "wcag21aa",
      "wcag22aa",
    ])
    .analyze();
  expect(results.violations, formatViolations(results.violations)).toEqual([]);
}

async function attachVisual(
  page: Page,
  testInfo: TestInfo,
  name: string,
): Promise<void> {
  const { devicePixelRatio, pageHeight } = await page.evaluate(() => ({
    devicePixelRatio: window.devicePixelRatio,
    pageHeight: Math.max(
      document.body.scrollHeight,
      document.documentElement.scrollHeight,
    ),
  }));
  if (
    pageHeight * devicePixelRatio <= MAX_FULL_PAGE_SCREENSHOT_DIMENSION
  ) {
    await testInfo.attach(name, {
      body: await page.screenshot({
        animations: "disabled",
        fullPage: true,
      }),
      contentType: "image/png",
    });
    return;
  }

  const viewportHeight = page.viewportSize()?.height ?? 900;
  const originalScrollY = await page.evaluate(() => window.scrollY);
  const positions = [
    { label: "top", top: 0 },
    {
      label: "middle",
      top: Math.max(0, Math.round((pageHeight - viewportHeight) / 2)),
    },
    {
      label: "bottom",
      top: Math.max(0, pageHeight - viewportHeight),
    },
  ];
  for (const position of positions) {
    await page.evaluate((top) => {
      document.documentElement.style.scrollBehavior = "auto";
      window.scrollTo(0, top);
    }, position.top);
    await testInfo.attach(`${name}-${position.label}`, {
      body: await page.screenshot({
        animations: "disabled",
        fullPage: false,
      }),
      contentType: "image/png",
    });
  }
  await page.evaluate((top) => window.scrollTo(0, top), originalScrollY);
}

async function expectVisualState(
  page: Page,
  browserName: string,
  name: string,
): Promise<void> {
  if (browserName !== "chromium") return;
  await expect(page).toHaveScreenshot(`${name}.png`, {
    animations: "disabled",
    caret: "hide",
    fullPage: false,
    stylePath: path.join(here, "snapshot.css"),
  });
}

async function expectNoRunningAnimations(page: Page): Promise<void> {
  const running = await page.evaluate(() =>
    document
      .getAnimations()
      .filter((animation) => animation.playState === "running")
      .map((animation) => {
        const effect = animation.effect as KeyframeEffect | null;
        return {
          className:
            effect?.target instanceof Element ? effect.target.className : "",
          duration: effect?.getTiming().duration,
        };
      }),
  );
  expect(running, "reduced motion left active animations").toEqual([]);
}

function expectCleanAudit(audit: Audit): void {
  expect(audit.consoleErrors, "browser console errors").toEqual([]);
  expect(audit.pageErrors, "uncaught page errors").toEqual([]);
  expect(audit.externalRequests, "unexpected external requests").toEqual([]);
  expect(audit.failedRequests, "failed browser requests").toEqual([]);
  expect(audit.responseErrors, "first-party HTTP errors").toEqual([]);
  expect(audit.unexpectedMethods, "unexpected HTTP methods").toEqual([]);
  expect(audit.webSockets, "unexpected WebSocket connections").toEqual([]);
}

function walkArtifact(
  directory: string,
  relative = "",
): { files: string[]; symlinks: string[] } {
  const files: string[] = [];
  const symlinks: string[] = [];
  for (const name of fs.readdirSync(directory).sort()) {
    const absolute = path.join(directory, name);
    const itemPath = relative ? `${relative}/${name}` : name;
    const stat = fs.lstatSync(absolute);
    if (stat.isSymbolicLink()) {
      symlinks.push(itemPath);
    } else if (stat.isDirectory()) {
      const nested = walkArtifact(absolute, itemPath);
      files.push(...nested.files);
      symlinks.push(...nested.symlinks);
    } else if (stat.isFile()) {
      files.push(itemPath);
    }
  }
  return { files, symlinks };
}

for (const surface of surfaces) {
  test.describe(surface.name, () => {
    for (const viewport of surface.viewports) {
      for (const theme of surface.themes) {
        test(`${viewport.name} · ${theme}`, async ({
          baseURL,
          browserName,
          page,
        }, testInfo) => {
          await page.setViewportSize(viewport);
          await prepareTheme(page, theme);
          const audit = monitorPage(page, baseURL);
          const response = await page.goto(surface.path, {
            waitUntil: "domcontentloaded",
          });
          expect(response?.status()).toBe(200);
          await page.waitForTimeout(250);

          await expect(page.locator("html")).toHaveAttribute("lang", /.+/);
          await expect(page).toHaveTitle(/\S+/);
          await expect(page.locator("h1")).toHaveCount(1);
          await expect(page.locator("main")).toBeVisible();
          await page.evaluate(() => document.fonts.ready);
          await expect
            .poll(() => page.evaluate(() => document.fonts.status))
            .toBe("loaded");
          await expectThemeInitialized(page, surface, theme);
          await expectNoPageOverflow(page);

          if (browserName === "chromium") {
            if (surface.engine === "website") {
              const colorScheme =
                theme === "dark" || theme === "system-dark"
                  ? "dark"
                  : "light";
              await page.emulateMedia({
                colorScheme,
                reducedMotion: "reduce",
              });
              await expect(page.locator("[data-rotating-word]")).toHaveText(
                "hardware.",
              );
              await expect(page.locator("[data-phi-count]")).toContainText(
                "masked",
              );
              const phiBody = page.locator("[data-phi-body]");
              for (const token of [
                "[NAME]",
                "[ID]",
                "[DATE]",
                "[PHONE]",
                "[HOSPITAL]",
              ]) {
                await expect(phiBody).toContainText(token);
              }
            }
            await expect(page).toHaveScreenshot(
              `${surface.name}-${viewport.name}-${theme}.png`,
              {
                animations: "disabled",
                caret: "hide",
                fullPage: false,
                stylePath: path.join(here, "snapshot.css"),
              },
            );
            await attachVisual(
              page,
              testInfo,
              `${surface.name}-${viewport.name}-${theme}-default`,
            );
            if (surface.engine === "website") {
              await page.emulateMedia({
                colorScheme:
                  theme === "dark" || theme === "system-dark"
                    ? "dark"
                    : "light",
                reducedMotion: "no-preference",
              });
            }
          }

          await expectVisibleKeyboardFocus(page);
          await expectAccessible(page);
          if (surface.engine === "website") {
            await page.emulateMedia({
              colorScheme:
                theme === "dark" || theme === "system-dark"
                  ? "dark"
                  : "light",
              reducedMotion: "reduce",
            });
            await page.locator("[data-rotating-word]").evaluate((word) => {
              word.textContent = "air-gapped box.";
            });
          }
          await expectTextSpacingReflow(page);
          expectCleanAudit(audit);
        });
      }
    }
  });
}

for (const surface of surfaces) {
  test(`${surface.name} retains structure in forced colors`, async ({
    baseURL,
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.emulateMedia({
      colorScheme: "dark",
      forcedColors: "active",
      reducedMotion: "reduce",
    });
    const audit = monitorPage(page, baseURL);
    const response = await page.goto(surface.path, {
      waitUntil: "domcontentloaded",
    });
    expect(response?.status()).toBe(200);
    await expect(page.locator("main")).toBeVisible();
    await expectNoPageOverflow(page);
    await expectVisibleKeyboardFocus(page);
    expectCleanAudit(audit);
    await attachVisual(page, testInfo, `${surface.name}-forced-colors`);
  });
}

test("website reduced motion keeps rotating and PHI demos static", async ({
  baseURL,
  page,
}) => {
  await page.emulateMedia({
    colorScheme: "light",
    reducedMotion: "reduce",
  });
  const audit = monitorPage(page, baseURL);
  await page.goto("/", { waitUntil: "load" });
  const before = await page.locator("[data-rotating-word], [data-phi-demo]")
    .allInnerTexts();
  await page.waitForTimeout(3_000);
  expect(await page.locator("[data-rotating-word], [data-phi-demo]").allInnerTexts())
    .toEqual(before);
  await expectNoRunningAnimations(page);
  expectCleanAudit(audit);
});

test("website default motion advances every rotating word", async ({
  baseURL,
  page,
}) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await page.emulateMedia({
    colorScheme: "light",
    reducedMotion: "no-preference",
  });
  await page.clock.install();
  const audit = monitorPage(page, baseURL);
  await page.goto("/", { waitUntil: "load" });
  const rotatingWord = page.locator("[data-rotating-word]");
  await expect(rotatingWord).toHaveText("hardware.");
  const initialPhiState = await page.locator("[data-phi-count]").innerText();

  for (const expectedWord of [
    "laptop.",
    "iPhone.",
    "GPU server.",
    "air-gapped box.",
    "hardware.",
  ]) {
    await page.clock.fastForward(2_690);
    await expect(rotatingWord).toHaveText(expectedWord);
    await expectNoPageOverflow(page);
  }
  expect(await page.locator("[data-phi-count]").innerText()).not.toBe(
    initialPhiState,
  );
  expectCleanAudit(audit);
});

for (const printable of [
  { name: "website", path: "/" },
  { name: "docs", path: "/docs/" },
]) {
  test(`${printable.name} print layout`, async ({
    baseURL,
    browserName,
    page,
  }) => {
    test.skip(browserName !== "chromium", "Canonical print baselines use Chromium");
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.emulateMedia({ colorScheme: "light", media: "print" });
    const audit = monitorPage(page, baseURL);
    await page.goto(printable.path, { waitUntil: "load" });
    await expect(page.locator("main")).toBeVisible();
    await expectNoPageOverflow(page);
    await expectVisualState(page, browserName, `${printable.name}-print`);
    expectCleanAudit(audit);
  });

  test(`${printable.name} reflows at a 400% zoom proxy`, async ({
    baseURL,
    browserName,
    context,
    page,
  }) => {
    test.skip(browserName !== "chromium", "CSS zoom proxy is Chromium-based");
    await page.setViewportSize({ width: 1440, height: 900 });
    const audit = monitorPage(page, baseURL);
    await page.goto(printable.path, { waitUntil: "load" });
    await page.setViewportSize({ width: 360, height: 900 });
    const devtools = await context.newCDPSession(page);
    await devtools.send("Emulation.setPageScaleFactor", {
      pageScaleFactor: 4,
    });
    await expect(page.locator("main")).toBeVisible();
    await expectNoPageOverflow(page);
    expectCleanAudit(audit);
  });
}

test("responsive source contract uses only shared breakpoints and CSS indicators", async ({
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Source assertions are browser-independent");
  const websiteHtml = fs.readFileSync(
    path.join(root, "docs/website/index.html"),
    "utf8",
  );
  const websiteScript = fs.readFileSync(
    path.join(root, "docs/website/assets/script.js"),
    "utf8",
  );
  for (const glyph of ["◐", "☀", "☾", "↗", "→", "✓", "⧉", "●", "−", "❯"]) {
    expect(`${websiteHtml}${websiteScript}`).not.toContain(glyph);
  }
  expect((websiteHtml.match(/<pre\b/gu) ?? []).length).toBe(2);
  expect((websiteHtml.match(/<pre\b[^>]*\btabindex="0"/gu) ?? []).length).toBe(2);

  const stylesheetPaths = [
    "docs/website/assets/style.css",
    "docs/stylesheets/openmed-brand.css",
    "docs/stylesheets/openmed-standalone.css",
  ];
  for (const stylesheet of stylesheetPaths) {
    const css = fs.readFileSync(path.join(root, stylesheet), "utf8");
    expect(css).not.toMatch(/@media[^{]*max-height/iu);
    for (const match of css.matchAll(/@media[^{]*max-width\s*:\s*([^)]+)/giu)) {
      expect(
        ["900px", "1080px"],
        `${stylesheet} has an unapproved max-width breakpoint`,
      ).toContain(match[1].trim());
    }
  }
});

for (const viewport of [
  { name: "mobile-390", width: 390, height: 844 },
  { name: "desktop-1440", width: 1440, height: 900 },
]) {
  test(`website without JavaScript · ${viewport.name}`, async ({
    browser,
    baseURL,
  }) => {
    const context = await browser.newContext({
      baseURL,
      colorScheme: "dark",
      javaScriptEnabled: false,
      reducedMotion: "reduce",
      viewport,
    });
    const page = await context.newPage();
    const audit = monitorPage(page, baseURL);
    const response = await page.goto("/", { waitUntil: "domcontentloaded" });
    expect(response?.status()).toBe(200);
    await expect(page.locator("main")).toContainText("pip install openmed");
    await expect(page.locator("main")).toContainText("synthetic input");
    await expect(page.locator("#year")).toHaveText("2026");
    const answers = page.locator('[id^="faq-answer-"]');
    expect(await answers.count()).toBeGreaterThan(0);
    for (const answer of await answers.all()) {
      await expect(answer).toBeVisible();
    }
    await expectNoPageOverflow(page);
    expectCleanAudit(audit);
    await context.close();
  });
}

test("website interactions persist and expose states", async ({
  baseURL,
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const audit = monitorPage(page, baseURL);
  await page.goto("/", { waitUntil: "domcontentloaded" });

  const theme = page.locator("#themeToggle");
  await theme.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload({ waitUntil: "domcontentloaded" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");

  const menu = page.locator("#navToggle");
  await menu.click();
  await expect(menu).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator("#primaryNav")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(menu).toHaveAttribute("aria-expanded", "false");

  await page.locator("#tab-install").click();
  await expect(page.locator("#tab-install")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await expect(page.locator("#panel-install")).toBeVisible();

  const copy = page.locator("[data-copy-text]").first();
  await copy.click();
  await expect(page.locator("#copyStatus")).toContainText(/copied/i);

  await page.locator('[data-filter="chemicals"]').click();
  await expect(page.locator('[data-filter="chemicals"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator(".model-grid [data-category]:visible")).toHaveCount(2);

  const closedQuestion = page.locator("#faq-question-2");
  await expect(closedQuestion).toHaveAttribute("aria-expanded", "false");
  await closedQuestion.click();
  await expect(closedQuestion).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator("#faq-answer-2")).toBeVisible();

  const hashTargets = await page
    .locator('a[href^="#"]')
    .evaluateAll((links) =>
      links.map((link) => (link as HTMLAnchorElement).hash.slice(1)),
    );
  expect(hashTargets.length).toBeGreaterThan(0);
  for (const target of hashTargets) {
    expect(
      await page.evaluate((id) => document.getElementById(id) !== null, target),
      `missing #${target}`,
    ).toBe(true);
  }
  await page.locator('.hero a[href="#compare"]').click();
  await expect(page).toHaveURL(/#compare$/);
  await expect(page.locator("#compare")).toBeInViewport();

  const blankExternal = page.locator(
    'a[href^="https://"][target="_blank"]',
  );
  expect(await blankExternal.count()).toBeGreaterThan(0);
  for (const link of await blankExternal.all()) {
    await expect(link).toHaveAttribute("rel", /\bnoopener\b/);
  }
  expectCleanAudit(audit);
});

for (const viewport of [
  { name: "mobile-320", width: 320, height: 800 },
  { name: "landscape-667", width: 667, height: 320 },
]) {
  for (const theme of ["light", "dark"] as const) {
    test(`website open menu is keyboard-complete · ${viewport.name} · ${theme}`, async ({
      baseURL,
      browserName,
      page,
    }) => {
      await page.setViewportSize(viewport);
      await prepareTheme(page, theme);
      await page.emulateMedia({
        colorScheme: theme,
        reducedMotion: "reduce",
      });
      const audit = monitorPage(page, baseURL);
      await page.goto("/", { waitUntil: "load" });

      const menu = page.locator("#navToggle");
      const navigation = page.locator("#primaryNav");
      const links = navigation.locator("a[href]");
      await menu.focus();
      await menu.press("Enter");
      await expect(menu).toHaveAttribute("aria-expanded", "true");
      await expect(navigation).toBeVisible();
      const linkCount = await links.count();
      expect(linkCount).toBeGreaterThan(1);
      await expect(links.first()).toBeFocused();

      for (let index = 1; index < linkCount; index += 1) {
        await page.keyboard.press(forwardLinkTabKey(browserName));
        await expect(links.nth(index)).toBeFocused();
      }
      const menuLayout = await navigation.evaluate((region) => {
        const navigationBox = region.getBoundingClientRect();
        const lastBox = [...region.querySelectorAll("a[href]")]
          .at(-1)
          ?.getBoundingClientRect();
        return {
          bottom: navigationBox.bottom,
          clientHeight: region.clientHeight,
          lastBottom: lastBox?.bottom ?? Number.POSITIVE_INFINITY,
          lastTop: lastBox?.top ?? Number.NEGATIVE_INFINITY,
          overflowY: getComputedStyle(region).overflowY,
          scrollHeight: region.scrollHeight,
          scrollTop: region.scrollTop,
          top: navigationBox.top,
          viewportHeight: window.innerHeight,
        };
      });
      expect(menuLayout.top).toBeGreaterThanOrEqual(0);
      expect(menuLayout.bottom).toBeLessThanOrEqual(
        menuLayout.viewportHeight + 1,
      );
      expect(menuLayout.lastTop).toBeGreaterThanOrEqual(menuLayout.top - 1);
      expect(menuLayout.lastBottom).toBeLessThanOrEqual(menuLayout.bottom + 1);
      expect(menuLayout.overflowY).toBe("auto");
      if (viewport.name === "landscape-667") {
        expect(menuLayout.scrollHeight).toBeGreaterThan(menuLayout.clientHeight);
        expect(menuLayout.scrollTop).toBeGreaterThan(0);
      }

      await expectNoPageOverflow(page);
      await expectAccessible(page);
      if (browserName === "chromium") {
        await expectVisualState(
          page,
          browserName,
          `website-menu-keyboard-${viewport.name}-${theme}`,
        );
      }
      await page.keyboard.press("Escape");
      await expect(menu).toHaveAttribute("aria-expanded", "false");
      await expect(menu).toBeFocused();
      expectCleanAudit(audit);
    });
  }
}

for (const theme of ["light", "dark"] as const) {
  test(`website visual interaction states · ${theme}`, async ({
    baseURL,
    browserName,
    page,
  }) => {
    test.skip(
      browserName !== "chromium",
      "Canonical visual baselines use Chromium",
    );
    const visualName = (name: string): string =>
      theme === "light" ? name : `${name}-dark`;
    await page.setViewportSize({ width: 1440, height: 900 });
    await prepareTheme(page, theme);
    await page.emulateMedia({
      colorScheme: theme,
      reducedMotion: "reduce",
    });
    const audit = monitorPage(page, baseURL);
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await page.evaluate(() => document.fonts.ready);
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    await expect(page.locator("[data-rotating-word]")).toHaveText("hardware.");
    await expect(page.locator("[data-phi-count]")).toHaveText("5/5 · masked");

    const primaryAction = page.locator(".hero .button").first();
    await primaryAction.hover();
    await expectVisualState(page, browserName, visualName("website-hover"));
    await primaryAction.focus();
    await expectAccessible(page);
    await expectVisualState(page, browserName, visualName("website-focus"));

    const installTab = page.locator("#tab-install");
    await installTab.focus();
    await installTab.press("Enter");
    await installTab.scrollIntoViewIfNeeded();
    await expect(installTab).toHaveAttribute("aria-selected", "true");
    await expectAccessible(page);
    await expectVisualState(
      page,
      browserName,
      visualName("website-tab-selected"),
    );

    const copy = page.locator("[data-copy-text]").first();
    await copy.focus();
    await copy.press("Enter");
    await copy.scrollIntoViewIfNeeded();
    await expect(page.locator("#copyStatus")).toContainText(/copied/i);
    await expectAccessible(page);
    await expectVisualState(page, browserName, visualName("website-copied"));

    const secondQuestion = page
      .locator('[data-faq-list] button[aria-controls]')
      .nth(1);
    await secondQuestion.focus();
    await secondQuestion.press("Enter");
    await secondQuestion.scrollIntoViewIfNeeded();
    await expect(secondQuestion).toHaveAttribute("aria-expanded", "true");
    await expectAccessible(page);
    await expectNoPageOverflow(page);
    expect(await page.evaluate(() => window.scrollX)).toBe(0);
    const stickyBrandBox = await page.locator(".site-header .brand").boundingBox();
    const faqIntroBox = await page
      .locator(".faq-grid > :first-child")
      .boundingBox();
    expect(stickyBrandBox).not.toBeNull();
    expect(faqIntroBox).not.toBeNull();
    expect(
      Math.abs((stickyBrandBox?.x ?? -1) - (faqIntroBox?.x ?? -2)),
    ).toBeLessThanOrEqual(1);
    await expectVisualState(page, browserName, visualName("website-faq-open"));

    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => window.scrollTo(0, 0));
    const menu = page.locator("#navToggle");
    await menu.focus();
    await menu.press("Enter");
    await expect(menu).toHaveAttribute("aria-expanded", "true");
    await expectAccessible(page);
    await expectVisualState(page, browserName, visualName("website-menu-open"));
    expectCleanAudit(audit);
  });
}

test("docs drawer traps focus and returns it on Escape", async ({
  baseURL,
  browserName,
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await prepareTheme(page, "light");
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/", { waitUntil: "domcontentloaded" });
  const trigger = page.locator("[data-openmed-drawer]");
  await trigger.click();
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  const navigation = page.locator(".md-sidebar--primary");
  await expect(navigation).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(() =>
        Boolean(document.activeElement?.closest(".md-sidebar--primary")),
      ),
    )
    .toBe(true);

  await page.keyboard.press("Shift+Tab");
  await expect(trigger).toBeFocused();
  await page.keyboard.press("Tab");
  await expect
    .poll(() =>
      page.evaluate(() =>
        Boolean(document.activeElement?.closest(".md-sidebar--primary")),
      ),
    )
    .toBe(true);
  await expectAccessible(page);
  await expectVisualState(page, browserName, "docs-drawer-open");
  await page.keyboard.press("Escape");
  await expect(trigger).toHaveAttribute("aria-expanded", "false");
  await expect(trigger).toBeFocused();
  expectCleanAudit(audit);
});

test("docs search, locale, theme, and code copy controls operate", async ({
  baseURL,
  browserName,
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await prepareTheme(page, "system-light");
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/", { waitUntil: "networkidle" });

  const languageLinks = page.locator(".md-select__link[hreflang]");
  await expect(languageLinks).toHaveCount(3);
  const localeMap = await languageLinks.evaluateAll((links) =>
    Object.fromEntries(
      links.map((link) => [
        link.getAttribute("hreflang"),
        new URL((link as HTMLAnchorElement).href).pathname,
      ]),
    ),
  );
  expect(localeMap).toEqual({
    en: "/docs/",
    hi: "/docs/hi/",
    zh: "/docs/zh/",
  });
  const languageButton = page.getByRole("button", { name: /select language/i });
  await languageButton.focus();
  await expectVisualState(page, browserName, "docs-locale-focus");

  await languageButton.evaluate((button) => button.blur());
  const search = page.locator('input[data-md-component="search-query"]');
  const searchMeta = page.locator(".md-search-result__meta");
  await expect(search).toBeVisible();
  await expect(searchMeta).toContainText(/type to start searching/i, {
    timeout: 20_000,
  });
  await search.click();
  await expect(search).toBeFocused();
  await search.pressSequentially("OpenMedSpan");
  await expect(searchMeta).toContainText(
    /matching documents?/i,
    { timeout: 20_000 },
  );
  await expectAccessible(page);
  await expectVisualState(page, browserName, "docs-search-open");
  await page.keyboard.press("Escape");
  await expect(page.locator("#__search")).not.toBeChecked();

  const palette = page.locator('form[data-md-component="palette"] label:visible');
  await palette.click();
  await expect(page.locator("body")).toHaveAttribute(
    "data-md-color-scheme",
    "default",
  );
  await page.locator('form[data-md-component="palette"] label:visible').click();
  await expect(page.locator("body")).toHaveAttribute(
    "data-md-color-scheme",
    "slate",
  );
  await page.reload({ waitUntil: "networkidle" });
  await expect(page.locator("body")).toHaveAttribute(
    "data-md-color-scheme",
    "slate",
  );

  await page.goto("/docs/getting-started/", { waitUntil: "networkidle" });
  const copy = page.getByRole("button", { name: "Copy to clipboard" }).first();
  await expect(copy).toBeVisible();
  await copy.click();
  await expect(page.locator('[data-md-component="dialog"]')).toContainText(
    /copied/i,
  );
  await copy.scrollIntoViewIfNeeded();
  await expectAccessible(page);
  await expectVisualState(page, browserName, "docs-code-copied");

  await page.goto("/docs/zh/", { waitUntil: "networkidle" });
  await expect(page.locator("html")).toHaveAttribute("lang", "zh");
  await page.goto("/docs/hi/", { waitUntil: "networkidle" });
  await expect(page.locator("html")).toHaveAttribute("lang", "hi");
  expectCleanAudit(audit);
});

test("docs deep links, admonitions, tabs, and tables are explicit", async ({
  baseURL,
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const audit = monitorPage(page, baseURL);

  const apiResponse = await page.goto(
    "/docs/api-reference/#openmed.PIIEntity",
    { waitUntil: "networkidle" },
  );
  expect(apiResponse?.status()).toBe(200);
  await expect(page.locator("#openmed\\.PIIEntity")).toBeAttached();
  await expect(page).toHaveURL(/#openmed\.PIIEntity$/);

  await page.goto("/docs/onboarding-india/", {
    waitUntil: "networkidle",
  });
  await expect(page.locator(".admonition.important")).toBeVisible();
  await expect(page.locator(".admonition-title")).toContainText(
    /synthetic walkthrough only/i,
  );
  await expect(page.locator(".md-typeset table").first()).toBeVisible();

  await page.goto("/docs/getting-started/", {
    waitUntil: "networkidle",
  });
  const tabSet = page.locator(".tabbed-set").first();
  await expect(tabSet).toBeVisible();
  expect(await tabSet.locator('input[type="radio"]').count()).toBeGreaterThan(1);
  await expect(tabSet.locator('input[type="radio"]:checked')).toHaveCount(1);
  await expectNoPageOverflow(page);
  await expectAccessible(page);
  expectCleanAudit(audit);
});

test("localized pages expose only real alternates and no false fallback", async ({
  baseURL,
  page,
  request,
}) => {
  const audit = monitorPage(page, baseURL);
  const response = await page.goto("/docs/zh/onboarding-china/", {
    waitUntil: "domcontentloaded",
  });
  expect(response?.status()).toBe(200);
  const alternates = await page
    .locator('link[rel~="alternate"][hreflang]')
    .evaluateAll((links) =>
      links.map((link) => ({
        href: new URL((link as HTMLLinkElement).href).pathname,
        language: link.getAttribute("hreflang"),
      })).sort((left, right) =>
        String(left.language).localeCompare(String(right.language)),
      ),
    );
  expect(alternates).toEqual([
    { href: "/docs/onboarding-china/", language: "en" },
    { href: "/docs/zh/onboarding-china/", language: "zh" },
  ]);
  const untranslated = await request.get("/docs/zh/analyze-text/");
  expect(untranslated.status()).toBe(404);
  expectCleanAudit(audit);
});

test("synthetic RTL fixture applies logical direction", async ({
  baseURL,
  page,
}) => {
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/demo/rtl/", { waitUntil: "domcontentloaded" });
  await expect(page.locator("html")).toHaveAttribute("lang", "ar");
  await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
  expect(
    await page.locator("main").evaluate((node) => getComputedStyle(node).direction),
  ).toBe("rtl");
  await expect(page.locator(".om-site-nav")).toHaveCSS("direction", "rtl");
  await expect(page.locator('bdi[dir="ltr"]').first()).toHaveCSS("direction", "ltr");
  await expect(page.locator("pre[dir=\"ltr\"]")).toHaveCSS("direction", "ltr");
  await expect(page.locator(".table-wrap")).toBeVisible();
  await expectNoPageOverflow(page);
  await expectAccessible(page);
  expectCleanAudit(audit);
});

test("leaderboard keeps wide evidence tables in a local scroll container", async ({
  baseURL,
  page,
}) => {
  await page.setViewportSize({ width: 320, height: 800 });
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/eval/benchmark-leaderboard/", {
    waitUntil: "domcontentloaded",
  });
  const tableWrap = page.locator(".table-wrap").first();
  await expect(tableWrap).toBeVisible();
  const dimensions = await tableWrap.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeGreaterThan(dimensions.clientWidth);
  await tableWrap.evaluate((element) => {
    element.scrollLeft = element.scrollWidth;
  });
  expect(await tableWrap.evaluate((element) => element.scrollLeft)).toBeGreaterThan(
    0,
  );
  await expectNoPageOverflow(page);
  await expectAccessible(page);
  expectCleanAudit(audit);
});

test("leaderboard keyboard tabs expose exactly one panel", async ({
  baseURL,
  page,
}) => {
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/eval/benchmark-leaderboard/", {
    waitUntil: "domcontentloaded",
  });
  const tabs = page.getByRole("tab");
  expect(await tabs.count()).toBeGreaterThan(0);
  await tabs.first().focus();
  await page.keyboard.press("ArrowRight");
  const selected = page.getByRole("tab", { selected: true });
  await expect(selected).toHaveCount(1);
  await expect(page.locator('[role="tabpanel"]:visible')).toHaveCount(1);
  expectCleanAudit(audit);
});

test("browser demo benchmarks both synthetic WASM and WebGPU branches", async ({
  baseURL,
  page,
}) => {
  const runtimePath = "/docs/demo/web/test-dual-runtime.js";
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", {
      configurable: true,
      value: {},
    });
  });
  await page.route(`**${runtimePath}`, async (route) => {
    await route.fulfill({
      body: `
        export async function createOpenMedPipeline(options) {
          if (!["wasm", "webgpu"].includes(options.backend)) {
            throw new Error("unsupported synthetic backend");
          }
          return async function detect(text) {
            const value = "John Doe";
            const start = text.indexOf(value);
            return [{
              entity_group: "NAME",
              score: options.backend === "webgpu" ? 0.98 : 0.99,
              start,
              end: start + value.length,
              word: value,
            }];
          };
        }
      `,
      contentType: "text/javascript; charset=utf-8",
      status: 200,
    });
  });
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/demo/web/", { waitUntil: "domcontentloaded" });
  await expect(page.locator("#webgpu-support")).toContainText(
    "WebGPU is available",
  );
  await page.locator("#runtime-module").fill("./test-dual-runtime.js");
  await page.locator("#repo-id").fill("./models/synthetic/");
  await page
    .locator("#input-text")
    .fill("John Doe visited the synthetic clinic.");
  await page.locator("#benchmark-both").click();
  await expect(page.locator("#status")).toHaveText(
    "WASM and WebGPU benchmarks completed.",
  );
  for (const backend of ["wasm", "webgpu"]) {
    await expect(page.locator(`#${backend}-load`)).not.toHaveText("—");
    await expect(page.locator(`#${backend}-first`)).not.toHaveText("—");
  }
  await expect(page.locator("#results mark")).toHaveText("John Doe");
  await expectAccessible(page);
  expectCleanAudit(audit);
});

test("browser demo model error visual state", async ({
  baseURL,
  browserName,
  page,
}) => {
  test.skip(browserName !== "chromium", "Canonical visual baselines use Chromium");
  const audit = monitorPage(page, baseURL);
  await page.setViewportSize({ width: 1440, height: 900 });
  await prepareTheme(page, "light");
  await page.goto("/docs/demo/web/", { waitUntil: "domcontentloaded" });
  await page
    .locator("#runtime-module")
    .fill("https://example.invalid/runtime.js");
  await page.locator("#repo-id").fill("./models/synthetic/");
  await page
    .locator("#input-text")
    .fill("Synthetic demo text used only for the error-state visual baseline.");
  await page.locator("#run-selected").click();
  await expect(page.locator("#status")).toHaveAttribute("data-kind", "error");
  await expectAccessible(page);
  await expectVisualState(page, browserName, "browser-demo-error");
  expectCleanAudit(audit);
});

test("staged artifact manifest owns and hashes every published file", async ({
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Artifact bytes are browser-independent");
  const manifest = JSON.parse(
    fs.readFileSync(path.join(artifactRoot, "pages-manifest.json"), "utf8"),
  );
  const actualArtifact = walkArtifact(artifactRoot);
  expect(actualArtifact.symlinks, "staged artifact contains symlinks").toEqual(
    [],
  );
  expect(
    actualArtifact.files
      .filter((file) => file !== "pages-manifest.json")
      .sort(),
    "manifest must enumerate the exact staged filesystem",
  ).toEqual(manifest.files.map((file: { path: string }) => file.path).sort());
  const routes = new Map<string, { owner: string; path: string }>();
  for (const file of manifest.files) {
    const absolute = path.resolve(artifactRoot, file.path);
    expect(absolute.startsWith(`${artifactRoot}${path.sep}`)).toBe(true);
    expect(fs.statSync(absolute).isFile()).toBe(true);
    expect(fs.statSync(absolute).size).toBe(file.bytes);
    expect(
      crypto.createHash("sha256").update(fs.readFileSync(absolute)).digest("hex"),
    ).toBe(file.sha256);
    expect(routes.has(file.route), `duplicate route ${file.route}`).toBe(false);
    routes.set(file.route, file);
  }
  for (const expected of manifest.expected_paths) {
    expect(fs.statSync(path.join(artifactRoot, expected)).isFile()).toBe(true);
  }
  expect(routes.get("/")?.owner).toBe("marketing");
  expect(routes.get("/docs/")?.owner).toBe("mkdocs");
  expect(
    routes.get("/docs/eval/benchmark-leaderboard/")?.owner,
  ).toBe("leaderboard");
  expect(routes.get("/docs/demo/web/")?.owner).toBe("browser-demo");
});

test("staged artifact respects recorded byte budgets", async ({
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Artifact bytes are browser-independent");
  const manifest = JSON.parse(
    fs.readFileSync(path.join(artifactRoot, "pages-manifest.json"), "utf8"),
  );
  const budgets = JSON.parse(
    fs.readFileSync(path.join(here, "budgets.json"), "utf8"),
  );
  expect(budgets.schema_version).toBe(2);
  const categoryPatterns: Record<string, RegExp> = {
    css: /\.css$/i,
    font: /\.(?:woff2?|ttf|otf)$/i,
    html: /\.html$/i,
    image: /\.(?:avif|gif|jpe?g|png|svg|webp)$/i,
    javascript: /\.(?:mjs|js)$/i,
    json: /\.json$/i,
  };
  const totalBytes = manifest.files.reduce(
    (total: number, file: { bytes: number }) => total + file.bytes,
    0,
  );
  const uniquePayloads = new Map<string, number>();
  for (const file of manifest.files as Array<{
    bytes: number;
    path: string;
    sha256: string;
  }>) {
    uniquePayloads.set(file.sha256, file.bytes);
  }
  const uniqueBytes = [...uniquePayloads.values()].reduce(
    (total, bytes) => total + bytes,
    0,
  );
  const sourceMapFiles = manifest.files.filter(
    (file: { path: string }) => file.path.endsWith(".map"),
  );
  expect(manifest.files.length).toBeLessThanOrEqual(
    budgets.artifact.maximum_files,
  );
  expect(totalBytes).toBeLessThanOrEqual(
    budgets.artifact.maximum_total_bytes,
  );
  expect(uniqueBytes).toBeLessThanOrEqual(
    budgets.artifact.maximum_unique_payload_bytes,
  );
  expect(totalBytes - uniqueBytes).toBeLessThanOrEqual(
    budgets.artifact.maximum_duplicate_payload_bytes,
  );
  expect(sourceMapFiles).toHaveLength(
    budgets.artifact.maximum_source_map_files,
  );
  for (const [category, maximum] of Object.entries(
    budgets.largest_file_bytes,
  )) {
    const governed = manifest.files.filter((file: { path: string }) =>
      categoryPatterns[category].test(file.path),
    );
    expect(governed.length, `${category} budget is exercised`).toBeGreaterThan(0);
    expect(
      Math.max(...governed.map((file: { bytes: number }) => file.bytes)),
      `largest ${category} payload`,
    ).toBeLessThanOrEqual(maximum as number);
  }
  for (const [governedPath, maximum] of Object.entries(
    budgets.governed_payload_bytes,
  )) {
    const file = manifest.files.find(
      (entry: { path: string }) => entry.path === governedPath,
    );
    expect(file, `missing governed payload ${governedPath}`).toBeDefined();
    expect(file.bytes, governedPath).toBeLessThanOrEqual(maximum as number);
  }
});

test("representative routes respect first-party transfer budgets", async ({
  baseURL,
  browser,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "Transfer budgets use Chromium");
  const routeBudgets = JSON.parse(
    fs.readFileSync(path.join(here, "budgets.json"), "utf8"),
  ).route_transfer_bytes;
  const origin = new URL(baseURL ?? "http://127.0.0.1:4173").origin;
  for (const [route, maximum] of Object.entries(routeBudgets)) {
    const context = await browser.newContext({ baseURL });
    const page = await context.newPage();
    let firstPartyBytes = 0;
    const responseErrors: string[] = [];
    page.on("response", (response) => {
      if (new URL(response.url()).origin !== origin) return;
      if (response.status() >= 400) {
        responseErrors.push(`${response.status()} ${response.url()}`);
      }
      const contentLength = Number(response.headers()["content-length"] ?? 0);
      if (Number.isFinite(contentLength)) firstPartyBytes += contentLength;
    });
    await page.goto(route, { waitUntil: "load" });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(500);
    if (route.startsWith("/docs/")) {
      await expect
        .poll(() =>
          page.locator('link[rel~="alternate"][hreflang]').count(),
        )
        .toBeGreaterThan(0);
      expect(
        await page
          .locator("link[data-openmed-hreflang-rel]")
          .count(),
        `${route} hreflang bootstrap markers`,
      ).toBe(0);
    }
    expect(firstPartyBytes, `${route} transfer bytes`).toBeGreaterThan(0);
    expect(firstPartyBytes, `${route} transfer bytes`).toBeLessThanOrEqual(
      maximum as number,
    );
    expect(responseErrors, `${route} first-party response errors`).toEqual([]);
    await context.close();
  }
});

for (const viewport of [
  { name: "mobile-320", width: 320, height: 800 },
  { name: "desktop-1440", width: 1440, height: 900 },
]) {
  test(`website stays within layout-shift and paint budgets · ${viewport.name}`, async ({
    browserName,
    page,
  }) => {
    test.skip(browserName !== "chromium", "Web Vitals observer is Chromium-based");
    const budgets = JSON.parse(
      fs.readFileSync(path.join(here, "budgets.json"), "utf8"),
    ).page_metrics;
    await page.setViewportSize(viewport);
    await page.addInitScript(() => {
      const state = { cls: 0, lcp: 0 };
      (window as Window & { __openmedVitals?: typeof state }).__openmedVitals =
        state;
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const shift = entry as PerformanceEntry & {
            hadRecentInput?: boolean;
            value?: number;
          };
          if (!shift.hadRecentInput) state.cls += shift.value ?? 0;
        }
      }).observe({ type: "layout-shift", buffered: true });
      new PerformanceObserver((list) => {
        state.lcp = list.getEntries().at(-1)?.startTime ?? state.lcp;
      }).observe({ type: "largest-contentful-paint", buffered: true });
    });
    await page.goto("/", { waitUntil: "load" });
    await page.evaluate(() => document.fonts.ready);
    await expect
      .poll(() => page.evaluate(() => document.fonts.status))
      .toBe("loaded");
    await page.waitForTimeout(1_000);
    const vitals = await page.evaluate(
      () =>
        (
          window as Window & {
            __openmedVitals: { cls: number; lcp: number };
          }
        ).__openmedVitals,
    );
    expect(vitals.lcp).toBeGreaterThan(0);
    expect(vitals.lcp).toBeLessThanOrEqual(
      budgets.largest_contentful_paint_ms,
    );
    expect(vitals.cls).toBeLessThanOrEqual(budgets.cumulative_layout_shift);
    const interactionLatency = await page.evaluate(
      () =>
        new Promise<number>((resolve) => {
          const start = performance.now();
          (document.querySelector("#tab-install") as HTMLButtonElement).click();
          requestAnimationFrame(() =>
            requestAnimationFrame(() => resolve(performance.now() - start)),
          );
        }),
    );
    expect(interactionLatency).toBeLessThanOrEqual(
      budgets.representative_interaction_latency_ms,
    );
  });
}

test("docs preload only the above-fold local IBM Plex face", async ({
  browserName,
  page,
}) => {
  test.skip(browserName !== "chromium", "Preload contract is browser-independent");
  await page.goto("/docs/", { waitUntil: "domcontentloaded" });
  const preloads = await page.locator('link[rel="preload"][as="font"]').evaluateAll(
    (links) =>
      links.map((link) => ({
        crossorigin: link.hasAttribute("crossorigin"),
        path: new URL((link as HTMLLinkElement).href).pathname,
        type: link.getAttribute("type"),
      })),
  );
  expect(preloads).toEqual([
    {
      crossorigin: true,
      path: "/docs/assets/fonts/IBMPlexSans-Regular.woff2",
      type: "font/woff2",
    },
  ]);
  expect(JSON.stringify(preloads)).not.toMatch(/Newsreader/iu);
});
