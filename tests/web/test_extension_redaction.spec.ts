import { createServer, type Server } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import {
  chromium,
  expect,
  test,
  type BrowserContext,
} from "../../web/extension/tests/playwright";

import { bundledPhiPipeline } from "../../web/extension/src/detector";

const repositoryDir = resolve(process.cwd(), "../..");
const extensionDir = resolve(repositoryDir, "web/extension/dist");
const syntheticNote =
  "Patient Alice Nguyen, DOB 1979-04-12, email alice@example.org.";

test("extension matches local model spans, masks without network, and honors site controls", async () => {
  const manifest = JSON.parse(
    await readFile(resolve(extensionDir, "manifest.json"), "utf8"),
  ) as {
    background: { scripts: string[]; service_worker: string };
    browser_specific_settings: {
      gecko: { data_collection_permissions: { required: string[] } };
    };
    host_permissions: string[];
  };
  expect(manifest.background).toMatchObject({
    scripts: ["worker.js"],
    service_worker: "worker.js",
  });
  expect(
    manifest.browser_specific_settings.gecko.data_collection_permissions,
  ).toEqual({ required: ["none"] });
  expect(manifest.host_permissions).toEqual([]);

  const server = await startFixtureServer();
  const userDataDir = await mkdtemp(join(tmpdir(), "openmed-extension-"));
  let context: BrowserContext | undefined;

  try {
    context = await chromium.launchPersistentContext(userDataDir, {
      channel: "chromium",
      headless: true,
      args: [
        `--disable-extensions-except=${extensionDir}`,
        `--load-extension=${extensionDir}`,
      ],
    });
    await waitForExtensionWorker(context);

    const page = await context.newPage();
    await page.goto(server.url);
    const panel = page.locator("#openmed-phi-guard");
    const status = panel.locator("[data-testid='openmed-status']");
    const mask = panel.locator("[data-testid='openmed-mask']");
    const toggle = panel.locator("[data-testid='openmed-site-toggle']");
    const policy = panel.locator("[data-testid='openmed-policy']");
    const note = page.locator("#clinical-note");
    await expect(status).toHaveText("OpenMed PHI Guard is ready.");
    await expect(mask).toBeDisabled();

    const outboundRequests: string[] = [];
    page.on("request", (request) => {
      if (/^https?:/.test(request.url())) {
        outboundRequests.push(request.url());
      }
    });

    const rawModelOutput = await bundledPhiPipeline(syntheticNote);
    const modelOutput = rawModelOutput.flatMap((span) =>
      Array.isArray(span) ? span : [span],
    );
    const expectedLabels = modelOutput.map((span) =>
      (span.entity ?? "").replace(/^S-/, ""),
    );
    await note.fill(syntheticNote);
    await expect(note).toHaveAttribute(
      "data-openmed-phi-count",
      String(modelOutput.length),
    );
    await expect(note).toHaveAttribute(
      "data-openmed-phi-labels",
      expectedLabels.join(","),
    );
    await expect(status).toContainText("PHI spans detected on-device");
    await expect(mask).toBeEnabled();
    expect(outboundRequests).toEqual([]);

    await page.locator("button[type='submit']").click();
    await expect(status).toContainText("Mask it before submitting");
    expect(outboundRequests).toEqual([]);

    await mask.click();
    await expect(note).toHaveValue(
      "Patient [PERSON], DOB [DATE_OF_BIRTH], email [EMAIL].",
    );
    await expect(status).toContainText("PHI masked on-device");

    await page.locator("#copy").evaluate((element, text) => {
      element.textContent = text;
    }, syntheticNote);
    const textMarks = page.locator(
      "[data-openmed-text-redaction] mark[data-openmed-phi]",
    );
    await expect(textMarks).toHaveCount(modelOutput.length);
    await expect(page.locator("#copy")).toContainText("[PERSON]");
    expect(outboundRequests).toEqual([]);

    await toggle.click();
    await expect(status).toContainText("disabled on this site");
    await expect(page.locator("[data-openmed-text-redaction]")).toHaveCount(0);
    await note.fill(syntheticNote);
    await page.waitForTimeout(350);
    await expect(note).not.toHaveAttribute("data-openmed-phi-count");
    await expect(mask).toBeDisabled();
    expect(outboundRequests).toEqual([]);

    await page.reload();
    await expect(status).toContainText("disabled on this site");
    outboundRequests.length = 0;
    await note.fill(syntheticNote);
    await page.waitForTimeout(350);
    await expect(note).not.toHaveAttribute("data-openmed-phi-count");

    await toggle.click();
    await expect(status).toContainText("ready");
    await policy.selectOption("clinical_minimal_redaction");
    await note.fill("Seen on 2026-07-19.");
    await expect(note).toHaveAttribute("data-openmed-phi-count", "0");
    await expect(status).toContainText("No PHI detected");

    await policy.selectOption("hipaa_safe_harbor");
    await expect(note).toHaveAttribute("data-openmed-phi-count", "1");
    expect(outboundRequests).toEqual([]);
  } finally {
    await context?.close();
    await server.close();
    await rm(userDataDir, { recursive: true, force: true });
  }
});

async function waitForExtensionWorker(context: BrowserContext): Promise<void> {
  if (context.serviceWorkers().length > 0) {
    return;
  }
  await context.waitForEvent("serviceworker");
}

async function startFixtureServer(): Promise<{
  url: string;
  close: () => Promise<void>;
}> {
  const server = createServer((_request, response) => {
    response.writeHead(200, {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
    });
    response.end(`<!doctype html>
      <html lang="en">
        <head><meta charset="utf-8"><title>OpenMed synthetic fixture</title></head>
        <body>
          <form>
            <label for="clinical-note">Synthetic clinical note</label>
            <textarea id="clinical-note"></textarea>
            <button type="submit">Submit</button>
          </form>
          <p id="copy"></p>
        </body>
      </html>`);
  });
  await new Promise<void>((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("Fixture server did not bind to a TCP port");
  }
  return {
    url: `http://127.0.0.1:${address.port}/`,
    close: () => closeServer(server),
  };
}

async function closeServer(server: Server): Promise<void> {
  await new Promise<void>((resolveClose, reject) => {
    server.close((error) => (error === undefined ? resolveClose() : reject(error)));
  });
}
