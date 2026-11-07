import assert from "node:assert/strict";
import test from "node:test";

import {
  OpenMedSidecarError,
  OpenMedTauriClient,
  type OpenMedSpan,
  type SidecarDeidentifyOptions,
  type SidecarDeidentifyResult,
  type TauriInvoke,
} from "../../js/openmedkit-tauri/src/client";

test("typed client sends de-identification options to the Tauri command", async () => {
  const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
  const expected: SidecarDeidentifyResult = {
    deidentifiedText: "Patient [NAME] called [PHONE].",
    spans: [],
  };
  const invoke: TauriInvoke = async <T>(
    command: string,
    args?: Record<string, unknown>,
  ): Promise<T> => {
    calls.push(args === undefined ? { command } : { command, args });
    return expected as T;
  };
  const client = new OpenMedTauriClient(invoke);

  const result = await client.deidentify("Patient Rowan Hale called 555-0100.", {
    policy: "hipaa_safe_harbor",
    deterministicOnly: true,
  });

  assert.deepEqual(result, expected);
  assert.deepEqual(calls, [
    {
      command: "openmed_sidecar_deidentify",
      args: {
        request: {
          text: "Patient Rowan Hale called 555-0100.",
          options: {
            policy: "hipaa_safe_harbor",
            deterministicOnly: true,
          },
        },
      },
    },
  ]);
});

test("a sidecar killed mid-request becomes a clean host error", async () => {
  const client = new OpenMedTauriClient(async () => {
    throw {
      code: "SIDECAR_TERMINATED",
      message: "Synthetic patient Rowan Hale was in the failed request.",
    };
  });

  await assert.rejects(
    client.deidentify("Synthetic patient note"),
    (error: unknown) => {
      assert.ok(error instanceof OpenMedSidecarError);
      assert.equal(error.code, "SIDECAR_TERMINATED");
      assert.equal(
        error.message,
        "The OpenMed sidecar terminated before responding.",
      );
      return true;
    },
  );
});

test("renderer input cannot select a model path", async () => {
  let invocationCount = 0;
  const client = new OpenMedTauriClient(async () => {
    invocationCount += 1;
    throw new Error("must not invoke");
  });
  const untrustedOptions = {
    modelName: "/private/rowan@example.test/model",
  } as unknown as SidecarDeidentifyOptions;

  await assert.rejects(
    client.deidentify("Synthetic note", untrustedOptions),
    (error: unknown) => {
      assert.ok(error instanceof OpenMedSidecarError);
      assert.equal(error.code, "INVALID_REQUEST");
      assert.ok(!error.message.includes("rowan@example.test"));
      return true;
    },
  );
  assert.equal(invocationCount, 0);
});

test("malformed or overlapping span responses fail closed", async () => {
  const overlapping = [
    span({ start: 0, end: 3 }),
    span({ start: 2, end: 4 }),
  ];
  const client = new OpenMedTauriClient(async <T>(): Promise<T> => {
    return {
      deidentifiedText: "[NAME]",
      spans: overlapping,
    } as T;
  });

  await assert.rejects(client.deidentify("ABCD"), (error: unknown) => {
    assert.ok(error instanceof OpenMedSidecarError);
    assert.equal(error.code, "SIDECAR_PROTOCOL");
    return true;
  });
});

test("span offsets use the documented Unicode code-point coordinate space", async () => {
  const expected = {
    deidentifiedText: "🧬[NAME]",
    spans: [span({ start: 1, end: 2 })],
  };
  const client = new OpenMedTauriClient(async <T>(): Promise<T> => expected as T);

  const result = await client.deidentify("🧬A");

  assert.deepEqual(result, expected);
});

test("ping and shutdown responses are validated", async () => {
  const client = new OpenMedTauriClient(async <T>(command: string): Promise<T> => {
    return (command === "openmed_sidecar_ping"
      ? { offline: false, protocolVersion: 1 }
      : { shutdown: false }) as T;
  });

  await assert.rejects(client.ping(), { code: "SIDECAR_PROTOCOL" });
  await assert.rejects(client.shutdown(), { code: "SIDECAR_PROTOCOL" });
});

function span(overrides: Partial<OpenMedSpan> = {}): OpenMedSpan {
  return {
    schema_version: 1,
    doc_id: "synthetic-tauri-test",
    start: 0,
    end: 1,
    text_hash:
      "hmac-sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    entity_type: "person",
    canonical_label: "NAME",
    policy_label: "DIRECT_IDENTIFIER",
    regulatory_tags: [],
    score: 0.9,
    detector: "synthetic-test",
    evidence: {},
    action: "mask",
    replacement: "[NAME]",
    reversible_id: null,
    section: null,
    metadata: {},
    ...overrides,
  };
}
