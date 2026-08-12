import { ConsoleMessage, Page, expect, test } from "@playwright/test";

const criticalSurfaces = [
  { name: "website", path: "/" },
  { name: "docs landing", path: "/docs/" },
  { name: "docs guide", path: "/docs/getting-started/" },
  { name: "docs API", path: "/docs/api-reference/" },
  { name: "docs Chinese", path: "/docs/zh/" },
  { name: "docs Hindi", path: "/docs/hi/" },
  { name: "leaderboard", path: "/docs/eval/benchmark-leaderboard/" },
  { name: "browser demo", path: "/docs/demo/web/" },
  { name: "RTL fixture", path: "/docs/demo/rtl/" },
];
const GITHUB_METADATA_URLS = [
  "https://api.github.com/repos/maziyarpanahi/openmed",
  "https://api.github.com/repos/maziyarpanahi/openmed/releases/latest",
] as const;

test.beforeEach(async ({ page }) => {
  for (const url of GITHUB_METADATA_URLS) {
    await page.route(url, (route) =>
      route.fulfill({
        body: "{}",
        contentType: "application/json",
        status: 503,
      }),
    );
  }
});

type PageAudit = {
  consoleErrors: string[];
  externalRequests: string[];
  failedRequests: string[];
  pageErrors: string[];
  responseErrors: string[];
  unexpectedMethods: string[];
};

function isGitHubMetadataRequest(url: URL): boolean {
  return GITHUB_METADATA_URLS.includes(
    url.href as (typeof GITHUB_METADATA_URLS)[number],
  );
}

function isGitHubMetadataConsoleError(message: ConsoleMessage): boolean {
  const source = message.location().url;
  return Boolean(source) && isGitHubMetadataRequest(new URL(source));
}

function monitorPage(page: Page, baseURL: string | undefined): PageAudit {
  const expectedOrigin = new URL(
    baseURL ?? "http://127.0.0.1:4173",
  ).origin;
  const audit: PageAudit = {
    consoleErrors: [],
    externalRequests: [],
    failedRequests: [],
    pageErrors: [],
    responseErrors: [],
    unexpectedMethods: [],
  };
  page.on("console", (message) => {
    if (
      message.type() === "error"
      && !isGitHubMetadataConsoleError(message)
    ) {
      audit.consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => audit.pageErrors.push(error.message));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (!["GET", "HEAD"].includes(request.method())) {
      audit.unexpectedMethods.push(`${request.method()} ${url.href}`);
    }
    if (
      (url.protocol === "http:" || url.protocol === "https:") &&
      url.origin !== expectedOrigin
      && !isGitHubMetadataRequest(url)
    ) {
      audit.externalRequests.push(`${request.method()} ${url.href}`);
    }
  });
  page.on("requestfailed", (request) => {
    if (isGitHubMetadataRequest(new URL(request.url()))) return;
    audit.failedRequests.push(
      `${request.method()} ${request.url()} ${
        request.failure()?.errorText ?? ""
      }`,
    );
  });
  page.on("response", (response) => {
    if (response.status() >= 400 && response.url().startsWith(expectedOrigin)) {
      audit.responseErrors.push(`${response.status()} ${response.url()}`);
    }
  });
  return audit;
}

function expectCleanAudit(audit: PageAudit): void {
  expect(audit.consoleErrors, "browser console errors").toEqual([]);
  expect(audit.pageErrors, "uncaught page errors").toEqual([]);
  expect(audit.externalRequests, "unexpected external requests").toEqual([]);
  expect(audit.failedRequests, "failed browser requests").toEqual([]);
  expect(audit.responseErrors, "first-party HTTP errors").toEqual([]);
  expect(audit.unexpectedMethods, "unexpected HTTP methods").toEqual([]);
}

async function expectNoHorizontalPageOverflow(
  page: Page,
): Promise<void> {
  const overflow = await page.evaluate(() => ({
    body: document.body.scrollWidth - document.body.clientWidth,
    document:
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  }));
  expect(overflow.body, JSON.stringify(overflow)).toBeLessThanOrEqual(1);
  expect(overflow.document, JSON.stringify(overflow)).toBeLessThanOrEqual(1);
}

for (const surface of criticalSurfaces) {
  test(`${surface.name} honors reduced motion`, async ({ baseURL, page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.emulateMedia({
      colorScheme: "dark",
      reducedMotion: "reduce",
    });
    const audit = monitorPage(page, baseURL);
    const response = await page.goto(surface.path, { waitUntil: "load" });
    expect(response?.status()).toBe(200);
    await expect(page.locator("main")).toBeVisible();
    expect(
      await page.evaluate(() =>
        window.matchMedia("(prefers-reduced-motion: reduce)").matches,
      ),
    ).toBe(true);
    await page.waitForTimeout(250);
    const runningAnimations = await page.evaluate(() =>
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
    expect(runningAnimations, "reduced motion left active animations").toEqual(
      [],
    );
    await expectNoHorizontalPageOverflow(page);
    expectCleanAudit(audit);
  });

  test(`${surface.name} reflows at a 400% zoom proxy`, async ({
    baseURL,
    browserName,
    context,
    page,
  }) => {
    test.skip(browserName !== "chromium", "Page-scale emulation requires CDP");
    await page.setViewportSize({ width: 360, height: 900 });
    const audit = monitorPage(page, baseURL);
    const response = await page.goto(surface.path, { waitUntil: "load" });
    expect(response?.status()).toBe(200);
    const devtools = await context.newCDPSession(page);
    await devtools.send("Emulation.setPageScaleFactor", {
      pageScaleFactor: 4,
    });
    await expect(page.locator("main")).toBeVisible();
    expect(
      await page.evaluate(() => window.visualViewport?.scale ?? 1),
    ).toBeGreaterThan(1);
    await expectNoHorizontalPageOverflow(page);
    expectCleanAudit(audit);
  });
}

for (const theme of ["light", "dark"] as const) {
  for (const viewport of [
    { name: "mobile", width: 390, height: 844 },
    { name: "desktop", width: 1440, height: 900 },
  ]) {
    test(`website interactions · ${viewport.name} · ${theme}`, async ({
      baseURL,
      page,
    }) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ colorScheme: theme });
      await page.addInitScript((selectedTheme) => {
        const marker = "__openmed_coverage_theme_prepared";
        if (sessionStorage.getItem(marker) !== "true") {
          sessionStorage.setItem(marker, "true");
          localStorage.setItem("openmed-theme", selectedTheme);
        }
      }, theme);
      const audit = monitorPage(page, baseURL);
      await page.goto("/", { waitUntil: "load" });
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);

      if (viewport.name === "mobile") {
        const menu = page.locator("#navToggle");
        await menu.focus();
        await menu.press("Enter");
        await expect(menu).toHaveAttribute("aria-expanded", "true");
        await expect(page.locator("#primaryNav")).toBeVisible();
        await page.keyboard.press("Escape");
        await expect(menu).toHaveAttribute("aria-expanded", "false");
        await expect(menu).toBeFocused();
      }

      await expect(page.locator(".terminal-install")).toContainText(
        'uv pip install "openmed[hf]"',
      );

      const copy = page.locator("[data-copy-text]").first();
      await copy.focus();
      await copy.press("Enter");
      await expect(page.locator("#copyStatus")).toContainText(/copied/i);
      await expect(copy).toHaveAttribute("aria-label", /copied/i);

      const allFilter = page.locator('[data-filter="all"]');
      const chemicalsFilter = page.locator('[data-filter="chemicals"]');
      await allFilter.focus();
      await allFilter.press("ArrowRight");
      await page.locator('[data-filter="disease"]').press("ArrowRight");
      await expect(chemicalsFilter).toBeFocused();
      await expect(chemicalsFilter).toHaveAttribute("aria-pressed", "true");
      await expect(
        page.locator(".model-grid [data-category]:visible"),
      ).toHaveCount(2);
      await expect(page.locator("[data-filter-status]")).toContainText(
        "Showing 2 model examples.",
      );

      const question = page.locator("#faq-question-2");
      await question.focus();
      await question.press("Enter");
      await expect(question).toHaveAttribute("aria-expanded", "true");
      await expect(page.locator("#faq-answer-2")).toBeVisible();
      await expect(page.locator("#faq-question-1")).toHaveAttribute(
        "aria-expanded",
        "false",
      );
      await expect(page.locator("#faq-answer-1")).toBeHidden();

      const nextTheme = theme === "dark" ? "light" : "dark";
      const themeToggle = page.locator("#themeToggle");
      await themeToggle.focus();
      await themeToggle.press("Enter");
      await expect(page.locator("html")).toHaveAttribute(
        "data-theme-preference",
        nextTheme,
      );
      await expect(page.locator("html")).toHaveAttribute(
        "data-theme",
        nextTheme,
      );
      await page.reload({ waitUntil: "load" });
      await expect(page.locator("html")).toHaveAttribute(
        "data-theme-preference",
        nextTheme,
      );
      await expect(page.locator("html")).toHaveAttribute(
        "data-theme",
        nextTheme,
      );
      await expectNoHorizontalPageOverflow(page);
      expectCleanAudit(audit);
    });
  }
}

test("docs theme control cycles through only light and dark", async ({
  baseURL,
  page,
}) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.addInitScript(() => {
    localStorage.removeItem("/docs/.__palette");
  });
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/", { waitUntil: "networkidle" });

  const palette = page.locator('form[data-md-component="palette"]');
  await expect(palette.locator('input[type="radio"]')).toHaveCount(2);
  await palette.locator("label:visible").click();
  await expect(page.locator("body")).toHaveAttribute(
    "data-md-color-scheme",
    "slate",
  );
  await palette.locator("label:visible").click();
  await expect(page.locator("body")).toHaveAttribute(
    "data-md-color-scheme",
    "default",
  );
  expectCleanAudit(audit);
});

test("standalone theme control cycles through only light and dark", async ({
  baseURL,
  page,
}) => {
  await page.addInitScript(() => {
    localStorage.setItem("openmed-theme", "light");
  });
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/eval/benchmark-leaderboard/", {
    waitUntil: "load",
  });

  const toggle = page.locator("[data-openmed-theme]");
  await expect(toggle).toHaveText("Theme: light");
  await toggle.click();
  await expect(toggle).toHaveText("Theme: dark");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await toggle.click();
  await expect(toggle).toHaveText("Theme: light");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  expectCleanAudit(audit);
});

test("browser demo completes a same-origin synthetic local inference", async ({
  baseURL,
  page,
}) => {
  const runtimePath = "/docs/demo/web/test-runtime.js";
  await page.route(`**${runtimePath}`, async (route) => {
    await route.fulfill({
      body: `
        export async function createOpenMedPipeline(options) {
          if (options.backend !== "wasm") {
            throw new Error("test runtime only exposes WASM");
          }
          return async function detect(text) {
            const value = "John Doe";
            const start = text.indexOf(value);
            return [{
              entity_group: "NAME",
              score: 0.99,
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
  const syntheticInput =
    "John Doe visited the synthetic clinic on 2026-07-29.";
  await page.locator("#runtime-module").fill(`.${runtimePath.split("/web")[1]}`);
  await page.locator("#repo-id").fill("./models/synthetic/");
  await page.locator("#input-text").fill(syntheticInput);
  await page.locator("#run-selected").click();

  await expect(page.locator("#status")).toContainText(
    /WASM inference completed/i,
  );
  await expect(page.locator("#results mark")).toHaveText("John Doe");
  await expect(page.locator("#entities")).toContainText("NAME: John Doe");
  await expect(page.locator("#wasm-load")).not.toHaveText("—");
  await expect(page.locator("#wasm-first")).not.toHaveText("—");
  expectCleanAudit(audit);

  const persistedInput = await page.evaluate((marker) => {
    const values = [
      ...Object.values(localStorage),
      ...Object.values(sessionStorage),
      document.cookie,
      ...performance.getEntriesByType("resource").map((entry) => entry.name),
    ];
    return values.some((value) => value.includes(marker));
  }, syntheticInput);
  expect(persistedInput, "synthetic input escaped into browser persistence").toBe(
    false,
  );
});

test("leaderboard filter exposes matching and empty states", async ({
  baseURL,
  page,
}) => {
  const audit = monitorPage(page, baseURL);
  await page.goto("/docs/eval/benchmark-leaderboard/", {
    waitUntil: "load",
  });
  const filter = page.locator("#leaderboard-filter");
  await expect(filter).toBeVisible();
  await filter.focus();
  await expect(filter).toBeFocused();
  const rows = page.locator("[data-leaderboard-row]");
  const initialCount = await rows.count();
  const initialVisibleCount = await page
    .locator("[data-leaderboard-row]:visible")
    .count();
  expect(initialCount).toBeGreaterThan(0);
  expect(initialVisibleCount).toBeGreaterThan(0);

  await filter.fill("SuperClinical");
  const positiveVisibleCount = await page
    .locator("[data-leaderboard-row]:visible")
    .count();
  expect(positiveVisibleCount).toBeGreaterThan(0);
  await expect(page.locator("#leaderboard-filter-status")).toContainText(
    new RegExp(`Showing ${positiveVisibleCount} of ${initialCount}`),
  );
  await expect(page.locator("[data-leaderboard-group]:visible")).toHaveCount(1);

  await filter.selectText();
  await filter.pressSequentially("no-synthetic-benchmark-matches-this");
  await expect(page.locator("[data-leaderboard-row]:visible")).toHaveCount(0);
  await expect(page.locator("[data-leaderboard-group]:visible")).toHaveCount(0);
  await expect(page.locator("#leaderboard-filter-status")).toContainText(
    /No benchmark rows match/i,
  );

  await filter.press("Escape");
  await expect(filter).toHaveValue("");
  await expect(page.locator("[data-leaderboard-row]:visible")).toHaveCount(
    initialVisibleCount,
  );
  await expect(page.locator("#leaderboard-filter-status")).toContainText(
    new RegExp(`Showing ${initialCount} of ${initialCount}`),
  );
  await expect(page.locator("[data-leaderboard-group]:visible")).toHaveCount(1);
  expectCleanAudit(audit);
});
