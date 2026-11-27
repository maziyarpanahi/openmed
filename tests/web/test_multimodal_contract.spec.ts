// Shared multimodal packet parity: every case of the synthetic fixture that
// Python generates and Swift validates (tests/fixtures/parity/
// multimodal_contract_v1.json). Assertion messages name case ids only.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  MULTIMODAL_ABSTENTION_CODES,
  MULTIMODAL_CLOCK_REASONS,
  MULTIMODAL_MAX_WIRE_BYTES,
  MULTIMODAL_PACKET_SCHEMA,
  MULTIMODAL_PROVIDER_OUTCOMES,
  MULTIMODAL_REFUSALS,
  MultimodalContractError,
  MultimodalDigestUnavailableError,
  MultimodalPacket,
  admitMultimodalPacket,
  type MultimodalDigestProvider,
} from "../../js/openmedkit-web/src/multimodal/index";
import * as mainEntry from "../../js/openmedkit-web/src/index";

interface FixtureCase {
  id: string;
  wire: string;
  digest: string;
  expected: Record<string, unknown> & { kind: string };
}

interface NegativeCase {
  id: string;
  wire: string;
  reason: string;
}

interface Fixture {
  version: number;
  synthetic_only: boolean;
  cases: FixtureCase[];
  negative_cases: NegativeCase[];
  provider_outcomes: string[];
  abstention_codes: string[];
  clock_reasons: string[];
}

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const fixturePath = join(
  rootDir,
  "tests",
  "fixtures",
  "parity",
  "multimodal_contract_v1.json",
);
const distDir = join(rootDir, "js", "openmedkit-web", "dist");

let fixtureCache: Promise<Fixture> | undefined;

function fixture(): Promise<Fixture> {
  fixtureCache ??= readFile(fixturePath, "utf8").then(
    (text) => JSON.parse(text) as Fixture,
  );
  return fixtureCache;
}

async function positive(id: string): Promise<FixtureCase> {
  const row = (await fixture()).cases.find((candidate) => candidate.id === id);
  assert.ok(row, id);
  return row;
}

function refusal(code: string, id: string) {
  return (error: unknown) => {
    assert.ok(error instanceof MultimodalContractError, id);
    assert.equal(error.code, code, id);
    // Diagnostics carry the code only, never packet values.
    assert.equal(error.message, code, id);
    return true;
  };
}

/** Python's canonical identity digest, recomputed with Node's own SHA-256. */
function identityDigest(wire: string): string {
  const envelope = wire.replace(/^\{"evidence_digest":"[0-9a-f]{64}",/, "{");
  return createHash("sha256").update(envelope, "ascii").digest("hex");
}

test("fixture is the synthetic v1 corpus", async () => {
  const corpus = await fixture();
  assert.equal(corpus.version, 1);
  assert.equal(corpus.synthetic_only, true);
  assert.ok(corpus.cases.length > 0);
  assert.ok(corpus.negative_cases.length > 0);
});

test("every positive case admits with the Python digest and exact bytes", async () => {
  for (const row of (await fixture()).cases) {
    const packet = await admitMultimodalPacket(row.wire);
    assert.equal(packet.evidenceDigest, row.digest, row.id);
    assert.equal(identityDigest(row.wire), row.digest, row.id);
    assert.equal(packet.schemaVersion, MULTIMODAL_PACKET_SCHEMA, row.id);
    assert.equal(packet.kind, row.expected.kind, row.id);
    assert.equal(packet.toText(), row.wire, row.id);
    assert.deepEqual(packet.toBytes(), new TextEncoder().encode(row.wire), row.id);

    const fromBytes = await MultimodalPacket.admit(new TextEncoder().encode(row.wire));
    assert.deepEqual(fromBytes.projection, packet.projection, row.id);
  }
});

test("typed projections expose the fixture's expected values", async () => {
  for (const row of (await fixture()).cases) {
    const { expected } = row;
    const { projection } = await admitMultimodalPacket(row.wire);
    switch (projection.kind) {
      case "asset": {
        const { asset } = projection;
        if (expected.width !== undefined) assert.equal(asset.width, expected.width, row.id);
        if (expected.height !== undefined) {
          assert.equal(asset.height, expected.height, row.id);
        }
        if (expected.duration !== undefined) {
          assert.equal(asset.durationSeconds, expected.duration, row.id);
        }
        if (expected.bytes !== undefined) {
          assert.equal(asset.byteSize.toString(), expected.bytes, row.id);
        }
        assert.equal(typeof asset.byteSize, "bigint", row.id);
        break;
      }
      case "visual": {
        const { visual } = projection;
        if (expected.box === null) {
          assert.equal(visual.displayBox, null, row.id);
          assert.equal(visual.sourceBox, null, row.id);
        } else {
          const box = visual.displayBox;
          assert.ok(box, row.id);
          assert.deepEqual([box.left, box.top, box.right, box.bottom], expected.box, row.id);
        }
        if (expected.scalar_span !== undefined) {
          assert.deepEqual(
            [visual.scalarSpan?.start, visual.scalarSpan?.end],
            expected.scalar_span,
            row.id,
          );
        }
        if (typeof expected.synthetic_text === "string" && visual.scalarSpan) {
          // Spans are Unicode code points, not UTF-16 code units.
          const scalars = [...expected.synthetic_text];
          assert.equal(
            scalars.slice(visual.scalarSpan.start, visual.scalarSpan.end).join(""),
            "\u{1F600}é",
            row.id,
          );
        }
        const quarter = visual.rotation === 90 || visual.rotation === 270;
        assert.equal(visual.displayWidth, quarter ? visual.height : visual.width, row.id);
        assert.equal(visual.displayHeight, quarter ? visual.width : visual.height, row.id);
        break;
      }
      case "transcript": {
        const { transcript } = projection;
        assert.equal(transcript.finalized, expected.finalized, row.id);
        const span = transcript.currentScalarSpan;
        assert.deepEqual(
          span === null ? null : [span.start, span.end],
          expected.current_span ?? null,
          row.id,
        );
        const samples = transcript.sampleBounds;
        assert.deepEqual(
          samples === null ? null : [samples.start, samples.end],
          expected.sample_bounds,
          row.id,
        );
        if (expected.time_ms !== undefined) {
          assert.deepEqual(
            [transcript.relativeMilliseconds?.start, transcript.relativeMilliseconds?.end],
            expected.time_ms,
            row.id,
          );
        }
        if (expected.invalidation !== undefined) {
          assert.equal(transcript.invalidation, expected.invalidation, row.id);
          assert.equal(transcript.relativeMilliseconds, null, row.id);
        }
        break;
      }
      case "provider": {
        const { provider } = projection;
        assert.equal(provider.outcome, expected.outcome, row.id);
        assert.equal(provider.abstention, expected.reason, row.id);
        if (provider.outcome !== "success") assert.equal(provider.outputDigest, null, row.id);
        else assert.match(provider.outputDigest ?? "", /^[0-9a-f]{64}$/, row.id);
        for (const value of Object.values(provider.counts)) {
          assert.equal(typeof value, "bigint", row.id);
        }
        break;
      }
      case "clock": {
        const { clock } = projection;
        assert.equal(clock.state, expected.state, row.id);
        const overlap = clock.guaranteedOverlap;
        if (Array.isArray(expected.overlap)) {
          assert.ok(overlap, row.id);
          assert.deepEqual([Number(overlap.start), Number(overlap.end)], expected.overlap, row.id);
        } else if (Array.isArray(expected.overlap_strings)) {
          assert.ok(overlap, row.id);
          assert.deepEqual(
            [overlap.start.toString(), overlap.end.toString()],
            expected.overlap_strings,
            row.id,
          );
        } else {
          assert.equal(overlap, null, row.id);
          assert.equal(clock.targetStart, null, row.id);
          assert.equal(clock.targetEnd, null, row.id);
          assert.equal(clock.uncertaintyNanoseconds, null, row.id);
          assert.ok(clock.reasons.length > 0, row.id);
        }
        break;
      }
    }
  }
});

test("nanosecond and byte counts above 2^53 survive without precision loss", async () => {
  const clock = await admitMultimodalPacket((await positive("clock_large_integer")).wire);
  assert.equal(clock.projection.kind, "clock");
  if (clock.projection.kind !== "clock") return;
  const { guaranteedOverlap, targetStart, targetEnd } = clock.projection.clock;
  assert.ok(guaranteedOverlap && targetStart && targetEnd);
  assert.equal(guaranteedOverlap.start, 4000000000000000001n);
  assert.equal(guaranteedOverlap.end, 4000000000000000101n);
  assert.ok(guaranteedOverlap.start > BigInt(Number.MAX_SAFE_INTEGER));
  // Number() would collapse these neighbours; bigint keeps them distinct.
  assert.equal(Number(guaranteedOverlap.start), Number(guaranteedOverlap.start + 1n));
  assert.notEqual(guaranteedOverlap.start, guaranteedOverlap.start + 1n);
  assert.ok(targetStart.upper <= guaranteedOverlap.start);
  assert.ok(guaranteedOverlap.end <= targetEnd.lower);

  const asset = await admitMultimodalPacket((await positive("asset_large_integer")).wire);
  assert.equal(asset.projection.kind, "asset");
  if (asset.projection.kind === "asset") {
    assert.equal(asset.projection.asset.byteSize, 9223372036854775807n);
  }
});

test("every negative case refuses with the fixture's code", async () => {
  for (const row of (await fixture()).negative_cases) {
    await assert.rejects(admitMultimodalPacket(row.wire), refusal(row.reason, row.id));
  }
});

test("reason vocabularies match the Python and Swift enumerations", async () => {
  const corpus = await fixture();
  assert.deepEqual([...MULTIMODAL_PROVIDER_OUTCOMES], corpus.provider_outcomes);
  assert.deepEqual([...MULTIMODAL_ABSTENTION_CODES], corpus.abstention_codes);
  assert.deepEqual([...MULTIMODAL_CLOCK_REASONS], corpus.clock_reasons);
  const refusals = new Set(corpus.negative_cases.map((row) => row.reason));
  for (const code of refusals) {
    assert.ok((MULTIMODAL_REFUSALS as readonly string[]).includes(code), code);
  }
  assert.equal(MULTIMODAL_REFUSALS.length, 5);
});

test("framing, size and encoding faults refuse as invalid contracts", async () => {
  const { wire } = await positive("asset_image");
  const visual = (await positive("visual_0")).wire;
  const mutate = (source: string, from: string, to: string) => {
    assert.ok(source.includes(from), from);
    return source.replace(from, to);
  };
  const cases: Array<[string, Uint8Array | string]> = [
    ["empty", ""],
    ["trailing_space", `${wire} `],
    ["leading_newline", `\n${wire}`],
    ["trailing_data", `${wire}{}`],
    ["not_object", "[]"],
    ["truncated", wire.slice(0, -1)],
    ["invalid_utf8", new Uint8Array([0x7b, 0xff, 0x7d])],
    ["oversized", `${wire}${" ".repeat(MULTIMODAL_MAX_WIRE_BYTES)}`],
    ["deep_nesting", `${"[".repeat(5000)}${"]".repeat(5000)}`],
    ["escaped_ascii", mutate(wire, '"kind":"asset"', '"kind":"\\u0061sset"')],
    ["escaped_solidus", mutate(wire, "image/png", "image\\/png")],
    ["exponent_integer", mutate(wire, '"byte_size":128', '"byte_size":1.28e2')],
    ["negative_zero_integer", mutate(visual, '"page":0', '"page":-0')],
  ];
  for (const [id, input] of cases) {
    await assert.rejects(
      admitMultimodalPacket(input),
      refusal("invalid_multimodal_contract", id),
    );
  }
});

test("a changed digest or payload breaks provenance", async () => {
  const { wire } = await positive("asset_image");
  const tampered = wire.replace('"height":50', '"height":51');
  await assert.rejects(
    admitMultimodalPacket(tampered),
    refusal("incomplete_multimodal_provenance", "tampered_payload"),
  );
  const clock = (await positive("clock_aligned")).wire;
  const resultDigest = /"source_result_digest":"([0-9a-f]{64})"/.exec(clock)?.[1];
  assert.ok(resultDigest);
  await assert.rejects(
    admitMultimodalPacket(clock.replace(resultDigest, "0".repeat(64))),
    refusal("incomplete_multimodal_provenance", "clock_result_digest"),
  );
});

test("admission hashes through the injected WebCrypto provider", async () => {
  const { wire, digest } = await positive("clock_aligned");
  const calls: number[] = [];
  const recording: MultimodalDigestProvider = {
    digest: async (algorithm, data) => {
      assert.equal(algorithm, "SHA-256");
      calls.push(data.length);
      return globalThis.crypto.subtle.digest("SHA-256", new Uint8Array(data));
    },
  };
  const packet = await admitMultimodalPacket(wire, { subtle: recording });
  assert.equal(packet.evidenceDigest, digest);
  // The clock source result digest and the envelope identity.
  assert.equal(calls.length, 2);

  const wrong: MultimodalDigestProvider = {
    digest: async () => new Uint8Array(32).buffer,
  };
  await assert.rejects(
    admitMultimodalPacket(wire, { subtle: wrong }),
    refusal("incomplete_multimodal_provenance", "wrong_digest"),
  );
});

test("admission fails closed without a SHA-256 provider", async () => {
  const { wire } = await positive("asset_image");
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  assert.ok(descriptor?.configurable);
  Object.defineProperty(globalThis, "crypto", { value: undefined, configurable: true });
  try {
    await assert.rejects(admitMultimodalPacket(wire), MultimodalDigestUnavailableError);
  } finally {
    Object.defineProperty(globalThis, "crypto", descriptor);
  }
});

test("packets are immutable and cannot be constructed unchecked", async () => {
  const { wire } = await positive("visual_90");
  const packet = await admitMultimodalPacket(wire);
  const bytes = packet.toBytes();
  bytes.fill(0);
  assert.equal(packet.toText(), wire);
  assert.ok(Object.isFrozen(packet));
  assert.ok(Object.isFrozen(packet.projection));
  if (packet.projection.kind === "visual") {
    assert.ok(Object.isFrozen(packet.projection.visual));
    assert.ok(Object.isFrozen(packet.projection.visual.displayBox));
  }
  const Unchecked = MultimodalPacket as unknown as new (...args: unknown[]) => unknown;
  assert.throws(
    () => new Unchecked(Symbol("forged"), new Uint8Array(), MULTIMODAL_PACKET_SCHEMA),
    refusal("invalid_multimodal_contract", "forged_constructor"),
  );
});

test("multimodal admission stays off the main entry and builds as a subpath", async () => {
  assert.equal(Object.hasOwn(mainEntry, "admitMultimodalPacket"), false);
  const built = await import(pathToFileURL(join(distDir, "multimodal", "index.js")).href);
  const { wire, digest } = await positive("transcript_final");
  const packet = await built.admitMultimodalPacket(wire);
  assert.equal(packet.evidenceDigest, digest);
  const mainBundle = await readFile(join(distDir, "index.js"), "utf8");
  assert.equal(mainBundle.includes(MULTIMODAL_PACKET_SCHEMA), false);
  assert.equal(mainBundle.includes("incomplete_multimodal_provenance"), false);
});
