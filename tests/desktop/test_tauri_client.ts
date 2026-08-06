import assert from "node:assert/strict";
import test from "node:test";

import {
  OpenMedSidecarError,
  OpenMedTauriClient,
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
      message: "The OpenMed sidecar terminated before responding.",
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
