/**
 * Main process of the OpenMedKit Electron qualification harness.
 *
 * The harness is a minimal app around `js/openmedkit-electron`. It loads the
 * shared, TEST-ONLY signed portable bundles (verified inside the utility
 * process by the production verifier), opens hardened windows, drives the
 * qualification scenarios through the production renderer bridge, and writes
 * a value-free JSON report: check names, counts, codes, offsets and digests.
 * It never logs request text.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import {
  BrowserWindow,
  app,
  ipcMain,
  protocol,
  session,
  utilityProcess,
  type IpcMainEvent,
  type IpcMainInvokeEvent,
  type WebContents,
} from "electron";
import { deidentify } from "openmed";

import {
  ElectronBundleActivationError,
  ElectronDeidentifyService,
  ElectronHardeningError,
  OPENMED_CANCEL_CHANNEL,
  OPENMED_DEIDENTIFY_CHANNEL,
  OPENMED_ELECTRON_SCHEMA_VERSION,
  OPENMED_RENDERER_CSP,
  createHardenedWebPreferences,
  hardenOpenMedSession,
  hardenOpenMedWebContents,
  redactTextWithSpans,
  registerElectronDeidentifyIpc,
  type ElectronBundleSource,
  type ElectronBundleVerification,
  type ElectronDeidentifyServiceEvent,
  type ElectronHardeningEvent,
  type ElectronOfflineSessionLike,
  type RendererOpenMedSpan,
  type UtilityProcessModuleLike,
} from "../../src/index";
import {
  CRASH_MARKER,
  NETWORK_PROBE_MARKER,
  NPM_GOLDEN,
  OFFSET_CASES,
  SLOW_MARKER,
  createSyntheticPipeline,
  utf16ToScalar,
} from "./synthetic-model";

const APP_ORIGIN = "app://openmed";
const HARNESS_COMMAND = "openmed-harness:command";
const HARNESS_REPLY = "openmed-harness:reply";
const WATCHDOG_MS = 240_000;
const REPLY_TIMEOUT_MS = 60_000;
const RENDERER_FILES: Record<string, string> = {
  "/index.html": "text/html; charset=utf-8",
  "/renderer.js": "text/javascript; charset=utf-8",
};

interface QualificationBundle {
  directory: string;
  understoodExtensions: string[];
  expected: ElectronBundleVerification;
}

interface Qualification {
  now: string;
  trustPolicy: unknown;
  target: NonNullable<ElectronBundleSource["target"]>;
  bundles: { golden: QualificationBundle; extension: QualificationBundle; tampered: QualificationBundle };
}

interface Check {
  name: string;
  ok: boolean;
  details: Record<string, unknown>;
}

interface Outcome {
  ok: boolean;
  requestId?: string;
  spans?: RendererOpenMedSpan[];
  bundle?: { generation: number; contentId: string } | null;
  errorCode?: string;
}

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.find((item) => item.startsWith(prefix))?.slice(prefix.length);
}

const reportPath = argument("harness-report");
const inputPath = argument("harness-input");
const userDataPath = argument("harness-user-data");
if (userDataPath !== undefined) {
  app.setPath("userData", userDataPath);
  app.setPath("sessionData", userDataPath);
  app.setPath("crashDumps", join(userDataPath, "Crashpad"));
  app.setPath("logs", join(userDataPath, "logs"));
}
app.disableHardwareAcceleration();
protocol.registerSchemesAsPrivileged([
  { scheme: "app", privileges: { standard: true, secure: true } },
]);

const checks: Check[] = [];
const events: ElectronDeidentifyServiceEvent[] = [];
const blocked: Record<ElectronHardeningEvent, number> = {
  navigation_blocked: 0,
  window_open_blocked: 0,
  webview_blocked: 0,
  permission_denied: 0,
  request_blocked: 0,
};
const replyWaiters = new Map<string, (message: Record<string, unknown>) => void>();
const harnessContents = new Set<WebContents>();
const trustedContents = new Set<WebContents>();
let commandSequence = 0;
let finished = false;

function check(name: string, ok: boolean, details: Record<string, unknown> = {}): void {
  checks.push({ name, ok, details });
}

function finish(code: number, note?: string): void {
  if (finished) {
    return;
  }
  finished = true;
  const report = {
    schema: "openmed.electron_harness.report",
    schema_version: 1,
    ok: code === 0 && checks.length > 0 && checks.every((item) => item.ok),
    note: note ?? null,
    packaged: app.isPackaged,
    electron: process.versions.electron,
    platform: process.platform,
    checks,
    blocked,
    event_counts: events.reduce<Record<string, number>>((counts, event) => {
      counts[event.event] = (counts[event.event] ?? 0) + 1;
      return counts;
    }, {}),
  };
  if (reportPath !== undefined) {
    writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  }
  app.exit(report.ok ? 0 : 1);
}

function resourcesRoot(): string {
  return app.isPackaged
    ? join(process.resourcesPath, "openmed-harness")
    : join(app.getAppPath(), "..", "resources", "openmed-harness");
}

function waitReply(id: string, op: string): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      replyWaiters.delete(`${id}:${op}`);
      reject(new Error(`no ${op} reply`));
    }, REPLY_TIMEOUT_MS);
    replyWaiters.set(`${id}:${op}`, (message) => {
      clearTimeout(timer);
      resolve(message);
    });
  });
}

function send(window: BrowserWindow, command: Record<string, unknown>): void {
  window.webContents.send(HARNESS_COMMAND, command);
}

function nextId(prefix: string): string {
  commandSequence += 1;
  return `${prefix}-${commandSequence}`;
}

async function submit(window: BrowserWindow, text: string) {
  const id = nextId("submit");
  const submitted = waitReply(id, "submitted");
  const result = waitReply(id, "result") as Promise<Outcome & Record<string, unknown>>;
  send(window, { id, op: "submit", text });
  const accepted = await submitted;
  return { id, requestId: String(accepted.requestId), result };
}

async function run(window: BrowserWindow, text: string): Promise<Outcome> {
  return (await submit(window, text)).result;
}

async function cancel(window: BrowserWindow, target: string): Promise<boolean> {
  const id = nextId("cancel");
  const reply = waitReply(id, "cancelled");
  send(window, { id, op: "cancel", target });
  return (await reply).acknowledged === true;
}

async function attack(window: BrowserWindow, channel: string, payload: unknown) {
  const id = nextId("attack");
  const reply = waitReply(id, "attack");
  send(window, { id, op: "attack", channel, payload });
  return (await reply) as {
    threw: boolean;
    response: { ok: unknown; errorCode: unknown; cancelled: unknown } | null;
  };
}

async function until(predicate: () => boolean, attempts = 400): Promise<boolean> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (predicate()) {
      return true;
    }
    await new Promise<void>((resolve) => setTimeout(resolve, 25));
  }
  return predicate();
}

async function openWindow(
  name: string,
  preload: string,
  options: { relax?: Record<string, unknown>; trusted: boolean },
): Promise<{ window: BrowserWindow; hardeningError: ElectronHardeningError | null }> {
  const webPreferences = {
    ...createHardenedWebPreferences({
      preload: join(__dirname, preload),
      partition: "openmed-harness-renderer",
    }),
    backgroundThrottling: false,
    ...(options.relax ?? {}),
  };
  const window = new BrowserWindow({ show: false, width: 640, height: 480, webPreferences });
  let hardeningError: ElectronHardeningError | null = null;
  try {
    hardenOpenMedWebContents(window.webContents, {
      allowedOrigins: [APP_ORIGIN],
      onBlocked: (event) => {
        blocked[event] += 1;
      },
    });
  } catch (error) {
    if (!(error instanceof ElectronHardeningError)) {
      throw error;
    }
    hardeningError = error;
  }
  harnessContents.add(window.webContents);
  if (options.trusted) {
    trustedContents.add(window.webContents);
  }
  const ready = waitReply("ready", `ready-${name}`);
  const readyOnce = (event: IpcMainEvent, message: Record<string, unknown>) => {
    if (event.sender === window.webContents && message?.id === "ready") {
      replyWaiters.get(`ready:ready-${name}`)?.(message);
      replyWaiters.delete(`ready:ready-${name}`);
      ipcMain.removeListener(HARNESS_REPLY, readyOnce);
    }
  };
  ipcMain.on(HARNESS_REPLY, readyOnce);
  await window.loadURL(`${APP_ORIGIN}/index.html`);
  await ready;
  return { window, hardeningError };
}

function project(spans: readonly RendererOpenMedSpan[]): RendererOpenMedSpan[] {
  return spans.map((span) => ({
    schema_version: span.schema_version,
    start: span.start,
    end: span.end,
    entity_type: span.entity_type,
    canonical_label: span.canonical_label,
    policy_label: span.policy_label,
    score: span.score,
  }));
}

async function direct(text: string): Promise<RendererOpenMedSpan[]> {
  const result = await deidentify(text, {
    pipeline: createSyntheticPipeline(),
    docId: "electron-document",
    detector: "electron-utility-process",
  });
  return project(result.spans);
}

function same(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function bundleSource(bundle: QualificationBundle, qualification: Qualification) {
  return {
    root: join(resourcesRoot(), "bundles", bundle.directory),
    trustPolicy: qualification.trustPolicy,
    target: qualification.target,
    understoodExtensions: bundle.understoodExtensions,
  } satisfies ElectronBundleSource;
}

function verificationOf(value: ElectronBundleVerification | null | undefined) {
  return value === null || value === undefined
    ? null
    : { ...value, eligibleRuntimes: [...value.eligibleRuntimes] };
}

async function qualify(): Promise<void> {
  if (inputPath === undefined || reportPath === undefined) {
    throw new Error("harness arguments missing");
  }
  const input = JSON.parse(readFileSync(inputPath, "utf8")) as {
    given: string;
    family: string;
  };
  const sentinelNote = `Patient ${input.given} ${input.family} reviewed today.`;
  const qualification = JSON.parse(
    readFileSync(join(resourcesRoot(), "qualification.json"), "utf8"),
  ) as Qualification;
  const { golden, extension, tampered } = qualification.bundles;
  check("app_packaging", true, { packaged: app.isPackaged });

  const rendererDir = join(__dirname, "renderer");
  const rendererSession = session.fromPartition("openmed-harness-renderer");
  rendererSession.protocol.handle("app", (request) => {
    const url = new URL(request.url);
    const contentType = RENDERER_FILES[url.pathname];
    if (url.host !== "openmed" || contentType === undefined) {
      return new Response(null, { status: 404 });
    }
    const body = readFileSync(join(rendererDir, url.pathname.slice(1)));
    return new Response(body, {
      headers: {
        "content-type": contentType,
        "content-security-policy": OPENMED_RENDERER_CSP,
      },
    });
  });
  hardenOpenMedSession(rendererSession, {
    allowedOrigins: [APP_ORIGIN],
    onBlocked: (event) => {
      blocked[event] += 1;
    },
  });

  const offlineSession = session.fromPartition("openmed-offline-inference", {
    cache: false,
  });
  const service = new ElectronDeidentifyService({
    utilityProcess: utilityProcess as unknown as UtilityProcessModuleLike,
    offlineSession: offlineSession as unknown as ElectronOfflineSessionLike,
    workerPath: join(__dirname, "worker.cjs"),
    bundle: bundleSource(golden, qualification),
    now: () => new Date(qualification.now),
    logger: (event) => events.push(event),
  });
  const unregister = registerElectronDeidentifyIpc<IpcMainInvokeEvent>(ipcMain, service, {
    isTrustedSender: (event) =>
      trustedContents.has(event.sender) && event.senderFrame === event.sender.mainFrame,
  });
  ipcMain.on(HARNESS_REPLY, (event, message: Record<string, unknown>) => {
    if (!harnessContents.has(event.sender) || typeof message?.id !== "string") {
      return;
    }
    const key = `${message.id}:${String(message.op)}`;
    const waiter = replyWaiters.get(key);
    if (waiter !== undefined) {
      replyWaiters.delete(key);
      waiter(message);
    }
  });

  try {
    // Verified bundle inside the utility process.
    const first = await service.whenReady();
    check("bundle_verified_in_utility", same(verificationOf(first.verification), golden.expected) &&
      first.generation === 1, {
      generation: first.generation,
      content_id: first.contentId,
      runtime_id: first.runtimeId,
      verification_matches_vector: same(verificationOf(first.verification), golden.expected),
    });

    // Window hardening, observed from inside a real sandboxed renderer.
    const { window: main } = await openWindow("main", "preload.cjs", { trusted: true });
    const probeId = nextId("probe");
    const probeReply = waitReply(probeId, "probe");
    send(main, { id: probeId, op: "probe" });
    const probe = (await probeReply).probe as Record<string, unknown>;
    check(
      "window_hardening",
      probe.requireType === "undefined" &&
        probe.processType === "undefined" &&
        probe.moduleType === "undefined" &&
        probe.bufferType === "undefined" &&
        probe.ipcRendererExposed === false &&
        same(probe.bridgeKeys, ["deidentify", "submit"]) &&
        probe.sandboxed === true &&
        probe.contextIsolated === true &&
        probe.evalBlocked === true &&
        probe.functionBlocked === true &&
        probe.inlineScriptBlocked === true &&
        probe.fetchBlocked === true &&
        probe.windowOpenBlocked === true &&
        (probe.cspViolations as number) > 0 &&
        probe.origin === APP_ORIGIN,
      { probe, window_open_blocked: blocked.window_open_blocked },
    );

    // Parity with the shared offset contract and the direct reference path.
    const failures: string[] = [];
    for (const item of OFFSET_CASES) {
      const outcome = await run(main, item.text);
      const spans = outcome.spans ?? [];
      const span = spans[0];
      const scalarStart = span === undefined ? -1 : utf16ToScalar(item.text, span.start);
      const scalarEnd = span === undefined ? -1 : utf16ToScalar(item.text, span.end);
      const redacted =
        span === undefined
          ? ""
          : item.text.slice(0, span.start) + item.replacement + item.text.slice(span.end);
      if (
        !outcome.ok ||
        spans.length !== 1 ||
        scalarStart !== item.expected_start ||
        scalarEnd !== item.expected_end ||
        redacted !== item.expected_redacted ||
        !same(spans, await direct(item.text)) ||
        !same(outcome.bundle, { generation: 1, contentId: first.contentId })
      ) {
        failures.push(item.id);
      }
    }
    check("parity_offset_contract", failures.length === 0 && OFFSET_CASES.length >= 40, {
      cases: OFFSET_CASES.length,
      failed_case_ids: failures,
    });

    const goldenOutcome = await run(main, NPM_GOLDEN.text);
    const goldenSpans = goldenOutcome.spans ?? [];
    check(
      "parity_npm_golden",
      goldenOutcome.ok &&
        same(goldenSpans, await direct(NPM_GOLDEN.text)) &&
        same(
          goldenSpans.map((span) => [span.start, span.end, span.canonical_label]),
          NPM_GOLDEN.spans.map((span) => [span.start, span.end, span.canonical_label]),
        ) &&
        redactTextWithSpans(NPM_GOLDEN.text, goldenSpans) === NPM_GOLDEN.deidentifiedText,
      { spans: goldenSpans.length },
    );

    const sentinelOutcome = await run(main, sentinelNote);
    check(
      "parity_sentinel_note",
      sentinelOutcome.ok && same(sentinelOutcome.spans, await direct(sentinelNote)) &&
        (sentinelOutcome.spans ?? []).length === 1,
      { spans: (sentinelOutcome.spans ?? []).length },
    );

    // Offline: the utility-process network guard blocks fetch, TCP and DNS.
    const offline = await run(main, `${NETWORK_PROBE_MARKER} ${sentinelNote}`);
    check("utility_network_blocked", offline.ok, {
      error_code: offline.errorCode ?? null,
      renderer_requests_blocked: blocked.request_blocked,
    });

    // A compromised renderer cannot widen capabilities.
    const { window: attacker } = await openWindow("attack", "attack-preload.cjs", {
      trusted: true,
    });
    const base = {
      schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
      requestId: "attack-request",
      text: sentinelNote,
    };
    const smuggled = [
      { modelPath: "/synthetic/elsewhere" },
      { bundle: { root: "/synthetic/elsewhere" } },
      { bundleRoot: "/synthetic/elsewhere" },
      { trustPolicy: { keys: [] } },
      { generation: 99 },
      { runtime: "pytorch" },
      { policy: { allow: "all" } },
    ];
    const smuggledCodes: unknown[] = [];
    for (const extra of smuggled) {
      const reply = await attack(attacker, OPENMED_DEIDENTIFY_CHANNEL, { ...base, ...extra });
      smuggledCodes.push(reply.response?.errorCode ?? null);
    }
    const unknownChannel = await attack(attacker, "openmed:activate-bundle", base);
    const victim = await submit(main, `${SLOW_MARKER} ${sentinelNote}`);
    const stolen = await attack(attacker, OPENMED_CANCEL_CHANNEL, {
      schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
      requestId: victim.requestId,
    });
    const victimOutcome = await victim.result;
    check(
      "ipc_capabilities_constrained",
      smuggledCodes.every((code) => code === "INVALID_REQUEST") &&
        unknownChannel.threw &&
        stolen.response?.cancelled === false &&
        victimOutcome.ok,
      {
        smuggled_payloads: smuggled.length,
        smuggled_codes: smuggledCodes,
        unknown_channel_rejected: unknownChannel.threw,
        cross_renderer_cancel: stolen.response?.cancelled ?? null,
      },
    );

    // Cancellation of a running and a queued request.
    const startsBeforeCancel = service.diagnostics().utilityProcessStarts;
    const running = await submit(main, `${SLOW_MARKER} ${sentinelNote}`);
    const queued = await submit(main, sentinelNote);
    await until(() => service.diagnostics().pendingRequests === 2);
    const queuedAck = await cancel(main, queued.id);
    const runningAck = await cancel(main, running.id);
    const [runningOutcome, queuedOutcome] = await Promise.all([running.result, queued.result]);
    const afterCancel = await run(main, sentinelNote);
    const drained = await until(() => service.diagnostics().cancellingRequests === 0);
    check(
      "cancellation",
      runningOutcome.errorCode === "CANCELLED" &&
        queuedOutcome.errorCode === "CANCELLED" &&
        queuedAck &&
        runningAck &&
        afterCancel.ok &&
        drained &&
        service.diagnostics().utilityProcessStarts === startsBeforeCancel,
      {
        running_code: runningOutcome.errorCode ?? null,
        queued_code: queuedOutcome.errorCode ?? null,
        acknowledged: [runningAck, queuedAck],
        cancel_acks_drained: drained,
      },
    );

    // Bundle generation change while a request is in flight.
    const inFlight = await submit(main, `${SLOW_MARKER} ${sentinelNote}`);
    await until(() => service.diagnostics().pendingRequests === 1);
    const activation = service.activateBundle(bundleSource(extension, qualification));
    const afterChange = await submit(main, sentinelNote);
    const [oldOutcome, handle, newOutcome] = await Promise.all([
      inFlight.result,
      activation,
      afterChange.result,
    ]);
    check(
      "bundle_generation_change",
      same(oldOutcome.bundle, { generation: 1, contentId: first.contentId }) &&
        same(newOutcome.bundle, { generation: handle.generation, contentId: handle.contentId }) &&
        handle.generation === 2 &&
        handle.contentId !== first.contentId &&
        same(verificationOf(handle.verification), extension.expected),
      {
        in_flight: oldOutcome.bundle ?? null,
        after_change: newOutcome.bundle ?? null,
        verification_matches_vector: same(verificationOf(handle.verification), extension.expected),
      },
    );

    // A tampered generation is rejected; work fails closed until recovery.
    let rejectedReason: string | null = null;
    let rejectedVerification: unknown = null;
    try {
      await service.activateBundle(bundleSource(tampered, qualification));
    } catch (error) {
      if (error instanceof ElectronBundleActivationError) {
        rejectedReason = error.reason;
        rejectedVerification = verificationOf(error.verification);
      }
    }
    const whileRejected = await run(main, sentinelNote);
    const recovered = await service.activateBundle({
      ...bundleSource(golden, qualification),
      expectedContentId: first.contentId,
    });
    const afterRecovery = await run(main, sentinelNote);
    check(
      "bundle_rejection_fails_closed",
      rejectedReason === "file_digest_mismatch" &&
        same(rejectedVerification, tampered.expected) &&
        whileRejected.errorCode === "BUNDLE_UNAVAILABLE" &&
        same(afterRecovery.bundle, {
          generation: recovered.generation,
          contentId: first.contentId,
        }),
      {
        reason: rejectedReason,
        verification_matches_vector: same(rejectedVerification, tampered.expected),
        while_rejected: whileRejected.errorCode ?? null,
        recovered_generation: recovered.generation,
      },
    );

    // Renderer teardown cancels only that renderer's work.
    const startsBeforeTeardown = service.diagnostics().utilityProcessStarts;
    const { window: secondary } = await openWindow("secondary", "preload.cjs", {
      trusted: true,
    });
    await submit(secondary, `${SLOW_MARKER} ${sentinelNote}`);
    await until(() => service.diagnostics().pendingRequests === 1);
    const pendingBefore = service.diagnostics().pendingRequests;
    secondary.destroy();
    const releasedOnTeardown = await until(() => service.diagnostics().pendingRequests === 0);
    const survivor = await run(main, sentinelNote);
    const teardownDrained = await until(() => service.diagnostics().cancellingRequests === 0);
    check(
      "renderer_teardown",
      pendingBefore === 1 &&
        releasedOnTeardown &&
        survivor.ok &&
        teardownDrained &&
        service.diagnostics().utilityProcessStarts === startsBeforeTeardown,
      {
        pending_before: pendingBefore,
        released_on_teardown: releasedOnTeardown,
        cancel_acks_drained: teardownDrained,
      },
    );

    // Utility-process crash and verified restart.
    const startsBeforeCrash = service.diagnostics().utilityProcessStarts;
    const verifiedBeforeCrash = events.filter((event) => event.event === "bundle_verified").length;
    const crashed = await run(main, `${CRASH_MARKER} ${sentinelNote}`);
    const pendingAfterCrash = service.diagnostics().pendingRequests;
    const restarted = await run(main, sentinelNote);
    const verifiedAfterCrash = events.filter((event) => event.event === "bundle_verified").length;
    check(
      "utility_crash_restart",
      crashed.errorCode === "UTILITY_PROCESS_EXITED" &&
        pendingAfterCrash === 0 &&
        restarted.ok &&
        same(restarted.bundle, { generation: recovered.generation, contentId: first.contentId }) &&
        service.diagnostics().utilityProcessStarts === startsBeforeCrash + 1 &&
        verifiedAfterCrash === verifiedBeforeCrash + 1,
      {
        crash_code: crashed.errorCode ?? null,
        restarts: service.diagnostics().utilityProcessStarts - startsBeforeCrash,
        reverified: verifiedAfterCrash - verifiedBeforeCrash,
      },
    );

    // A relaxed window is refused even when the app marks it trusted.
    const { window: relaxedWindow, hardeningError } = await openWindow(
      "relaxed",
      "preload.cjs",
      { trusted: true, relax: { sandbox: false } },
    );
    const relaxedOutcome = await run(relaxedWindow, sentinelNote);
    check(
      "relaxed_window_refused",
      hardeningError !== null &&
        same(hardeningError.relaxations, ["sandbox_disabled"]) &&
        relaxedOutcome.errorCode === "UNTRUSTED_SENDER",
      {
        relaxations: hardeningError?.relaxations ?? [],
        error_code: relaxedOutcome.errorCode ?? null,
      },
    );

    // Remote navigation is refused and the window keeps working.
    const navigationsBefore = blocked.navigation_blocked;
    const navigateId = nextId("navigate");
    const navigating = waitReply(navigateId, "navigating");
    send(main, { id: navigateId, op: "navigate" });
    await navigating;
    const navigationBlocked = await until(() => blocked.navigation_blocked > navigationsBefore);
    const stillWorking = await run(main, sentinelNote);
    check(
      "remote_navigation_blocked",
      navigationBlocked &&
        main.webContents.getURL() === `${APP_ORIGIN}/index.html` &&
        stillWorking.ok,
      { navigation_blocked: blocked.navigation_blocked - navigationsBefore },
    );

    // Diagnostics are value-free.
    const allowed: Record<string, string[]> = {
      request_started: [],
      request_completed: ["spanCount"],
      request_failed: ["errorCode"],
      request_cancelled: [],
      utility_process_started: [],
      utility_process_exited: [],
      bundle_verified: ["contentId", "generation"],
      bundle_rejected: ["generation", "reason"],
    };
    const serialized = JSON.stringify(events);
    check(
      "diagnostics_value_free",
      events.every((event) => {
        const { event: name, ...fields } = event as { event: string } & Record<string, unknown>;
        return (
          name in allowed &&
          same(Object.keys(fields).sort(), allowed[name]) &&
          Object.values(fields).every(
            (value) =>
              typeof value === "number" ||
              (typeof value === "string" && /^(?:[A-Za-z_]+|sha256:[0-9a-f]{64})$/.test(value)),
          )
        );
      }) &&
        !serialized.includes(input.given) &&
        !serialized.includes(input.family),
      { events: events.length },
    );
    check("service_diagnostics", true, { diagnostics: service.diagnostics() });
  } finally {
    unregister();
    service.dispose();
  }
}

app.on("window-all-closed", () => {
  // The harness exits explicitly once the report is written.
});

const watchdog = setTimeout(() => finish(3, "watchdog"), WATCHDOG_MS);
app
  .whenReady()
  .then(qualify)
  .then(
    () => {
      clearTimeout(watchdog);
      finish(0);
    },
    (error: unknown) => {
      clearTimeout(watchdog);
      // Harness errors use fixed messages; anything else is reported by name.
      const message = error instanceof Error ? error.message : "";
      check("harness_completed", false, {
        error: /^[a-z0-9_ -]{1,64}$/i.test(message) ? message : "unexpected_error",
      });
      finish(2, "harness_error");
    },
  );
