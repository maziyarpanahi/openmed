// Model-fixture manifest shared by the CI staging script and the native probe.
//
// A staged fixture directory holds `openmed-parity-fixture.json` plus a
// `model/` tree. The manifest lists every file under `model/` with its sha256
// and byte size. The bundle digest is sha256 over `path\tsha256\tbytes\n`
// lines sorted by path; the Swift and Kotlin probes recompute it the same way
// and refuse to load a bundle with a missing, extra, or altered file.

import { sha256Hex } from "./sha256";

export const FIXTURE_MANIFEST_SCHEMA = "openmed.rn.parity.model_fixture";
export const FIXTURE_MANIFEST_FILE = "openmed-parity-fixture.json";
export const FIXTURE_DIRECTORY_NAME = "openmed-parity-fixture";

export interface FixtureFile {
  path: string;
  sha256: string;
  bytes: number;
}

export interface FixtureManifest {
  schema: typeof FIXTURE_MANIFEST_SCHEMA;
  version: 1;
  platform: "ios" | "android";
  backend: "coreml" | "onnx";
  model_path: string;
  id2label_path: string;
  tokenizer_folder_path: string | null;
  tokenizer_name: string | null;
  source: { repo: string; revision: string };
  files: FixtureFile[];
  bundle_digest: string;
}

/** Digest over the sorted file list; mirrored by the native probes. */
export function bundleDigest(files: readonly FixtureFile[]): string {
  const lines = [...files]
    .sort((left, right) => (left.path < right.path ? -1 : left.path > right.path ? 1 : 0))
    .map((file) => `${file.path}\t${file.sha256}\t${file.bytes}\n`)
    .join("");
  return `sha256:${sha256Hex(lines)}`;
}

/** Validate a manifest and recompute its bundle digest. */
export function validateFixtureManifest(value: unknown): FixtureManifest {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("fixture_manifest_invalid");
  }
  const manifest = value as FixtureManifest;
  const pathPattern = /^model(\/[A-Za-z0-9._-]+)+$/;
  if (
    manifest.schema !== FIXTURE_MANIFEST_SCHEMA ||
    manifest.version !== 1 ||
    (manifest.platform !== "ios" && manifest.platform !== "android") ||
    (manifest.backend !== "coreml" && manifest.backend !== "onnx") ||
    !Array.isArray(manifest.files) ||
    manifest.files.length === 0
  ) {
    throw new Error("fixture_manifest_invalid");
  }
  const seen = new Set<string>();
  for (const file of manifest.files) {
    if (
      typeof file.path !== "string" ||
      !pathPattern.test(file.path) ||
      file.path.split("/").some((segment) => segment === "." || segment === "..") ||
      typeof file.sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(file.sha256) ||
      !Number.isInteger(file.bytes) ||
      file.bytes < 0 ||
      seen.has(file.path)
    ) {
      throw new Error("fixture_manifest_file_invalid");
    }
    seen.add(file.path);
  }
  for (const key of ["model_path", "id2label_path"] as const) {
    if (typeof manifest[key] !== "string" || !/^model(\/[A-Za-z0-9._-]+)*$/.test(manifest[key])) {
      throw new Error("fixture_manifest_invalid");
    }
  }
  if (bundleDigest(manifest.files) !== manifest.bundle_digest) {
    throw new Error("fixture_bundle_digest_mismatch");
  }
  return manifest;
}
