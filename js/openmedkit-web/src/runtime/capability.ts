export type OrtWebBackend = "webgpu" | "wasm-simd-threads" | "wasm-basic";

export interface OrtWebCapabilityProfile {
  webgpu: boolean;
  wasm: boolean;
  wasmSimd: boolean;
  sharedArrayBuffer: boolean;
  crossOriginIsolated: boolean;
  hardwareConcurrency: number;
}

export interface OrtWebBackendChoice {
  backend: OrtWebBackend;
  executionProviders: ("webgpu" | "wasm")[];
  wasm: {
    simd: boolean;
    threads: boolean;
    numThreads: number;
  };
  reason: string;
}

export interface OrtCapabilityGlobalScope {
  WebAssembly?: Pick<typeof WebAssembly, "validate">;
  SharedArrayBuffer?: typeof SharedArrayBuffer;
  crossOriginIsolated?: boolean;
  navigator?: {
    gpu?: unknown;
    hardwareConcurrency?: number;
  };
}

export interface CapabilityDetectionOptions {
  globalScope?: OrtCapabilityGlobalScope;
  overrides?: Partial<OrtWebCapabilityProfile>;
}

const MAX_WASM_THREADS = 4;
const MIN_THREADED_WASM_CONCURRENCY = 2;

const WASM_SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60,
  0x00, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
  0x41, 0x00, 0xfd, 0x0f, 0xfd, 0x62, 0x0b,
]);

export function detectOrtWebCapabilities(
  options: CapabilityDetectionOptions = {},
): OrtWebCapabilityProfile {
  const scope = options.globalScope ?? globalThis;
  const navigatorLike = scope.navigator as
    | { gpu?: unknown; hardwareConcurrency?: number }
    | undefined;
  const wasm = scope.WebAssembly;
  const hardwareConcurrency = Math.max(
    1,
    Math.floor(navigatorLike?.hardwareConcurrency ?? 1),
  );
  const detected: OrtWebCapabilityProfile = {
    webgpu: navigatorLike?.gpu !== undefined,
    wasm: wasm !== undefined,
    wasmSimd: detectWasmSimd(wasm),
    sharedArrayBuffer: scope.SharedArrayBuffer !== undefined,
    crossOriginIsolated: scope.crossOriginIsolated === true,
    hardwareConcurrency,
  };

  return {
    ...detected,
    ...(options.overrides ?? {}),
  };
}

export function selectOrtWebBackend(
  profile: OrtWebCapabilityProfile,
): OrtWebBackendChoice {
  const wasmThreadsAvailable =
    profile.wasm &&
    profile.wasmSimd &&
    profile.sharedArrayBuffer &&
    profile.crossOriginIsolated &&
    profile.hardwareConcurrency >= MIN_THREADED_WASM_CONCURRENCY;
  const numThreads = wasmThreadsAvailable
    ? Math.min(MAX_WASM_THREADS, profile.hardwareConcurrency)
    : 1;

  if (profile.webgpu) {
    return {
      backend: "webgpu",
      executionProviders: profile.wasm ? ["webgpu", "wasm"] : ["webgpu"],
      wasm: {
        simd: profile.wasmSimd,
        threads: wasmThreadsAvailable,
        numThreads,
      },
      reason: "navigator.gpu is available",
    };
  }

  if (!profile.wasm) {
    throw new Error("ONNX Runtime Web requires WebAssembly when WebGPU is absent.");
  }

  if (wasmThreadsAvailable) {
    return {
      backend: "wasm-simd-threads",
      executionProviders: ["wasm"],
      wasm: {
        simd: true,
        threads: true,
        numThreads,
      },
      reason:
        "WebAssembly SIMD, SharedArrayBuffer, and cross-origin isolation are available",
    };
  }

  return {
    backend: "wasm-basic",
    executionProviders: ["wasm"],
    wasm: {
      simd: false,
      threads: false,
      numThreads: 1,
    },
    reason:
      "WebGPU or threaded wasm prerequisites are unavailable; using single-threaded wasm",
  };
}

function detectWasmSimd(
  wasm: Pick<typeof WebAssembly, "validate"> | undefined,
): boolean {
  if (wasm === undefined) {
    return false;
  }
  try {
    return wasm.validate(WASM_SIMD_PROBE);
  } catch {
    return false;
  }
}
