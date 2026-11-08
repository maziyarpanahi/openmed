// Offline checks for the packed React Native bridge tarball. The bridge ships
// TypeScript sources plus native module files, so the verifier runs this file
// with a TypeScript loader resolved outside the consumer's dependency graph.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  OPENMED_SPAN_SCHEMA_VERSION,
  deidentify,
  loadModel,
  setOpenMedKitNativeModuleForTests,
  type OpenMedKitNativeModule,
} from "@openmed/openmedkit-react-native";

interface ContractEntity {
  label: string;
  start: number;
  end: number;
  confidence: number;
}

interface RedactionCase {
  id: string;
  profile: string;
  text: string;
  entities: ContractEntity[];
  expected_redacted: string;
}

const consumerRoot = new URL("../", import.meta.url);
const contract = JSON.parse(
  readFileSync(new URL("consumer_contract.json", consumerRoot), "utf8"),
) as { redaction_cases: RedactionCase[] };
const hipaaCase = contract.redaction_cases.find(
  (candidate) => candidate.profile === "hipaa_safe_harbor",
) as RedactionCase;
const packageRoot = new URL("node_modules/@openmed/openmedkit-react-native/", consumerRoot);

test("resolves the bridge from the consumer install with native sources", () => {
  const resolved = fileURLToPath(import.meta.resolve("@openmed/openmedkit-react-native"));
  assert.ok(resolved.startsWith(fileURLToPath(packageRoot)), "resolved_outside_install");
  for (const file of [
    "package.json",
    "src/index.ts",
    "android/OpenMedKitRnModule.kt",
    "ios/OpenMedKitRN.swift",
  ]) {
    assert.ok(existsSync(new URL(file, packageRoot)), `missing:${file}`);
  }
});

test("a missing native module is a documented incompatibility", async () => {
  setOpenMedKitNativeModuleForTests(null);
  await assert.rejects(deidentify(hipaaCase.text), /OpenMedKitRN is not installed/);
});

test("a fake native session keeps policy and span contracts", async () => {
  const policies: string[] = [];
  const nativeModule: OpenMedKitNativeModule = {
    loadModel: async (source) => ({ ...source, platform: "android", loaded: true }),
    analyzeText: async () => [],
    extractPii: async () => [],
    deidentify: async (text, options) => {
      policies.push(options?.policy ?? "missing");
      return {
        text,
        redactedText: hipaaCase.expected_redacted,
        spans: hipaaCase.entities.map((entity) => ({
          start: entity.start,
          end: entity.end,
          label: entity.label,
          confidence: entity.confidence,
          action: "mask",
          text_hash: `sha256:${"0".repeat(64)}`,
        })),
      };
    },
  };
  setOpenMedKitNativeModuleForTests(nativeModule);
  try {
    const loaded = await loadModel({ modelPath: "/models/installed-consumer", backend: "onnx" });
    assert.equal(loaded.cacheKey, "onnx:/models/installed-consumer");
    const result = await deidentify(hipaaCase.text, { hashSecret: "fixture-secret" });
    const explicit = await deidentify(hipaaCase.text, { policy: "clinical_minimal_redaction" });

    assert.deepEqual(policies, ["hipaa_safe_harbor", "clinical_minimal_redaction"]);
    assert.equal(result.deidentifiedText, hipaaCase.expected_redacted, hipaaCase.id);
    assert.equal(explicit.spans.length, hipaaCase.entities.length);
    assert.deepEqual(
      result.spans.map((span) => [span.start, span.end, span.canonical_label, span.policy_label]),
      [
        [8, 12, "PERSON", "DIRECT_IDENTIFIER"],
        [18, 28, "DATE", "QUASI_IDENTIFIER"],
        [33, 39, "CONDITION", "CLINICAL_CONCEPT"],
      ],
    );
    assert.ok(result.spans.every((span) => span.schema_version === OPENMED_SPAN_SCHEMA_VERSION));
  } finally {
    setOpenMedKitNativeModuleForTests(null);
  }
});
