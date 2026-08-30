import {
  probeOrtWebCapabilities,
  type CapabilityDetectionOptions,
  type OrtWebBackend,
  type OrtWebCapabilityProbeResult,
  type OrtWebCapabilityProfile,
} from "./capability";
import { CLASSIFY_WGSL_SOURCE } from "./classify-wgsl";
import {
  assertOfflineAssetPath,
  loadOrtWebSession,
  type OrtFeeds,
  type OrtResults,
  type OrtSessionCreateOptions,
  type OrtTensorLike,
  type OrtWebLoadedSession,
  type OrtWebRuntimeProvider,
  type OrtWebSessionCache,
} from "./ort-web-loader";

export { CLASSIFY_WGSL_SOURCE } from "./classify-wgsl";

export const WEBGPU_BENCHMARK_SUITE =
  "webgpu-token-classification-runtime" as const;
export const DEFAULT_WEBGPU_LOGIT_TOLERANCE = 1e-3;
export const DEFAULT_WEBGPU_MAX_RECALL_DELTA = 0;

const GPU_BUFFER_USAGE_MAP_READ = 0x0001;
const GPU_BUFFER_USAGE_COPY_SRC = 0x0004;
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_UNIFORM = 0x0040;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;
const GPU_MAP_MODE_READ = 0x0001;
const CLASSIFY_WORKGROUP_X = 8;
const CLASSIFY_WORKGROUP_Y = 8;
const MIN_INT64 = -(2n ** 63n);
const MAX_INT64 = 2n ** 63n - 1n;

export type TokenIdData =
  | readonly number[]
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array;

export interface WebGpuTokenBatch {
  inputIds: TokenIdData;
  attentionMask?: TokenIdData;
  tokenTypeIds?: TokenIdData;
  batchSize: number;
  sequenceLength: number;
  additionalFeeds?: OrtFeeds;
}

export interface TokenClassificationLogits {
  data: Float32Array;
  dims: readonly [number, number, number];
  outputName: string;
}

export interface WebGpuModelPaths {
  webgpu: string;
  wasm: string;
}

export interface WebGpuInputNames {
  inputIds: string;
  attentionMask: string;
  tokenTypeIds: string;
}

export interface WebGpuTokenClassificationSessionOptions {
  modelPath: string | WebGpuModelPaths;
  assetPath: string;
  runtime?: OrtWebRuntimeProvider;
  capabilities?: Partial<OrtWebCapabilityProfile>;
  globalScope?: CapabilityDetectionOptions["globalScope"];
  sessionOptions?: OrtSessionCreateOptions;
  cache?: OrtWebSessionCache;
  inputNames?: Partial<WebGpuInputNames>;
  outputName?: string;
  labelCount?: number;
  modelName?: string;
  deviceName?: string;
  tier?: string;
  canonicalTier?: string;
  precision?: string;
  runOptions?: Record<string, unknown>;
  clock?: () => number;
  generatedAt?: string | null;
  benchmarkSink?: (report: WebGpuBenchmarkReport) => void | Promise<void>;
  fallbackOnWebGpuError?: boolean;
  probeAdapter?: boolean;
  powerPreference?: "low-power" | "high-performance";
}

export interface WebGpuBenchmarkOptions {
  iterations?: number;
  warmupIterations?: number;
  deviceName?: string;
  tier?: string;
  canonicalTier?: string;
  generatedAt?: string | null;
  sink?: (report: WebGpuBenchmarkReport) => void | Promise<void>;
}

export interface WebGpuDeviceBenchmarkMetrics {
  backend: OrtWebBackend;
  precision: string;
  batch_size: number;
  sequence_length: number;
  latency: {
    cold_load_ms: number;
    first_inference_ms: number;
    cold_ms: number;
    warm_ms: number;
    p50_ms: number;
    p95_ms: number;
    count: number;
  };
  throughput: {
    tokens_per_second: number;
  };
}

export interface WebGpuBenchmarkReport {
  suite: typeof WEBGPU_BENCHMARK_SUITE;
  model_name: string;
  device: string;
  fixture_count: number;
  generated_at: string | null;
  metadata: {
    runtime: "onnxruntime-web";
    local_files_only: true;
    tier: string;
    canonical_tier: string;
    warmup_iterations: number;
    token_count: number;
    fallback_used: boolean;
  };
  metrics: {
    devices: Record<string, WebGpuDeviceBenchmarkMetrics>;
  };
}

export interface WebGpuClassificationHeadConfig {
  weights: Float32Array | readonly number[];
  bias: Float32Array | readonly number[];
  hiddenSize: number;
  labelCount: number;
  shaderSource?: string;
  label?: string;
}

export interface WebGpuTokenSpan {
  batch_index: number;
  label: string;
  start_token: number;
  end_token: number;
}

export interface WebGpuRecallGate {
  reference_count: number;
  candidate_count: number;
  matched_count: number;
  recall: number;
  recall_delta: number;
  max_recall_delta: number;
  critical_missed_count: number;
  per_label_recall: Record<string, number>;
  passed: boolean;
}

export interface WebGpuReferenceCertification {
  tolerance: number;
  max_abs_logit_delta: number;
  span_tolerance: number;
  span_parity_passed: boolean;
  reference_token_spans: readonly WebGpuTokenSpan[];
  candidate_token_spans: readonly WebGpuTokenSpan[];
  recall_gate: WebGpuRecallGate;
  passed: true;
}

export interface WebGpuReferenceCertificationOptions {
  referenceLogits: TokenClassificationLogits;
  candidateLogits: TokenClassificationLogits;
  id2label: Readonly<Record<string, string>> | ReadonlyMap<number, string>;
  attentionMask?: TokenIdData;
  tolerance?: number;
  spanTolerance?: number;
  maxRecallDelta?: number;
  criticalLabels?: readonly string[];
}

export interface WebGpuRecallGateOptions {
  spanTolerance?: number;
  maxRecallDelta?: number;
  criticalLabels?: readonly string[];
}

export class WebGpuVerificationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebGpuVerificationError";
  }
}

export class WebGpuClassificationHead {
  readonly hiddenSize: number;
  readonly labelCount: number;

  private readonly device: GpuDeviceLike;
  private readonly weights: Float32Array;
  private readonly bias: Float32Array;
  private readonly shaderSource: string;
  private readonly label: string;
  private pipeline: GpuComputePipelineLike | undefined;
  private weightsBuffer: GpuBufferLike | undefined;
  private biasBuffer: GpuBufferLike | undefined;
  private disposed = false;

  private constructor(device: GpuDeviceLike, config: WebGpuClassificationHeadConfig) {
    this.device = device;
    this.hiddenSize = positiveInteger(config.hiddenSize, "hiddenSize");
    this.labelCount = positiveInteger(config.labelCount, "labelCount");
    this.weights = finiteFloat32Array(config.weights, "weights");
    this.bias = finiteFloat32Array(config.bias, "bias");
    this.shaderSource = config.shaderSource ?? CLASSIFY_WGSL_SOURCE;
    this.label = config.label ?? "OpenMed token-classification head";

    if (this.weights.length !== this.hiddenSize * this.labelCount) {
      throw new RangeError(
        "classification weights must contain hiddenSize * labelCount values",
      );
    }
    if (this.bias.length !== this.labelCount) {
      throw new RangeError("classification bias must contain labelCount values");
    }
  }

  static async create(
    device: unknown,
    config: WebGpuClassificationHeadConfig,
  ): Promise<WebGpuClassificationHead> {
    const head = new WebGpuClassificationHead(asGpuDevice(device), config);
    await head.initialize();
    return head;
  }

  async run(
    hiddenStates: Float32Array | readonly number[],
    batchSize: number,
    sequenceLength: number,
  ): Promise<TokenClassificationLogits> {
    this.ensureActive();
    const batch = positiveInteger(batchSize, "batchSize");
    const sequence = positiveInteger(sequenceLength, "sequenceLength");
    const hidden = finiteFloat32Array(hiddenStates, "hiddenStates");
    const expectedHiddenLength = batch * sequence * this.hiddenSize;
    if (hidden.length !== expectedHiddenLength) {
      throw new RangeError(
        `hiddenStates must contain ${expectedHiddenLength} values, got ${hidden.length}`,
      );
    }

    const outputLength = batch * sequence * this.labelCount;
    const outputBytes = outputLength * Float32Array.BYTES_PER_ELEMENT;
    const hiddenBuffer = this.device.createBuffer({
      label: `${this.label} hidden states`,
      size: hidden.byteLength,
      usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
    });
    const outputBuffer = this.device.createBuffer({
      label: `${this.label} logits`,
      size: outputBytes,
      usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
    });
    const readbackBuffer = this.device.createBuffer({
      label: `${this.label} readback`,
      size: outputBytes,
      usage: GPU_BUFFER_USAGE_MAP_READ | GPU_BUFFER_USAGE_COPY_DST,
    });
    const shapeBuffer = this.device.createBuffer({
      label: `${this.label} shape`,
      size: 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPU_BUFFER_USAGE_UNIFORM | GPU_BUFFER_USAGE_COPY_DST,
    });

    try {
      this.device.queue.writeBuffer(hiddenBuffer, 0, hidden);
      this.device.queue.writeBuffer(
        shapeBuffer,
        0,
        new Uint32Array([
          batch,
          sequence,
          this.hiddenSize,
          this.labelCount,
        ]),
      );
      const pipeline = this.pipeline;
      const weightsBuffer = this.weightsBuffer;
      const biasBuffer = this.biasBuffer;
      if (
        pipeline === undefined ||
        weightsBuffer === undefined ||
        biasBuffer === undefined
      ) {
        throw new Error("WebGPU classification head was not initialized");
      }
      const bindGroup = this.device.createBindGroup({
        label: `${this.label} bindings`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: hiddenBuffer } },
          { binding: 1, resource: { buffer: weightsBuffer } },
          { binding: 2, resource: { buffer: biasBuffer } },
          { binding: 3, resource: { buffer: outputBuffer } },
          { binding: 4, resource: { buffer: shapeBuffer } },
        ],
      });
      const encoder = this.device.createCommandEncoder({
        label: `${this.label} commands`,
      });
      const pass = encoder.beginComputePass({
        label: `${this.label} compute`,
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(
        Math.ceil(sequence / CLASSIFY_WORKGROUP_X),
        Math.ceil(this.labelCount / CLASSIFY_WORKGROUP_Y),
        batch,
      );
      pass.end();
      encoder.copyBufferToBuffer(
        outputBuffer,
        0,
        readbackBuffer,
        0,
        outputBytes,
      );
      this.device.queue.submit([encoder.finish()]);
      await readbackBuffer.mapAsync(GPU_MAP_MODE_READ, 0, outputBytes);
      const mapped = readbackBuffer.getMappedRange(0, outputBytes);
      const data = new Float32Array(mapped.slice(0));
      readbackBuffer.unmap();
      ensureFinite(data, "WebGPU classification logits");
      return {
        data,
        dims: [batch, sequence, this.labelCount],
        outputName: "logits",
      };
    } finally {
      hiddenBuffer.destroy();
      outputBuffer.destroy();
      readbackBuffer.destroy();
      shapeBuffer.destroy();
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.weightsBuffer?.destroy();
    this.biasBuffer?.destroy();
    this.weightsBuffer = undefined;
    this.biasBuffer = undefined;
    this.pipeline = undefined;
  }

  private async initialize(): Promise<void> {
    const module = this.device.createShaderModule({
      code: this.shaderSource,
      label: `${this.label} shader`,
    });
    if (module.getCompilationInfo !== undefined) {
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === "error");
      if (errors.length > 0) {
        const locations = errors
          .map((message) =>
            message.lineNum === undefined
              ? message.message
              : `line ${message.lineNum}: ${message.message}`,
          )
          .join("; ");
        throw new Error(`WebGPU classification shader compilation failed: ${locations}`);
      }
    }
    const descriptor = {
      label: `${this.label} pipeline`,
      layout: "auto",
      compute: { module, entryPoint: "classify" },
    };
    this.pipeline =
      this.device.createComputePipelineAsync === undefined
        ? this.device.createComputePipeline(descriptor)
        : await this.device.createComputePipelineAsync(descriptor);
    this.weightsBuffer = this.device.createBuffer({
      label: `${this.label} weights`,
      size: this.weights.byteLength,
      usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
    });
    this.biasBuffer = this.device.createBuffer({
      label: `${this.label} bias`,
      size: this.bias.byteLength,
      usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST,
    });
    this.device.queue.writeBuffer(this.weightsBuffer, 0, this.weights);
    this.device.queue.writeBuffer(this.biasBuffer, 0, this.bias);
  }

  private ensureActive(): void {
    if (this.disposed) {
      throw new Error("WebGPU classification head has been disposed");
    }
  }
}

export async function createWebGpuClassificationHead(
  device: unknown,
  config: WebGpuClassificationHeadConfig,
): Promise<WebGpuClassificationHead> {
  return WebGpuClassificationHead.create(device, config);
}

export class WebGpuTokenClassificationSession {
  readonly backend: OrtWebBackend;
  readonly capabilityProbe: OrtWebCapabilityProbeResult;
  readonly modelName: string;
  readonly modelPath: string;
  readonly coldLoadMs: number;
  readonly fallbackReason: string | null;

  private readonly loaded: OrtWebLoadedSession;
  private readonly inputNames: WebGpuInputNames;
  private readonly outputName: string;
  private readonly labelCount: number | undefined;
  private readonly runOptions: Record<string, unknown> | undefined;
  private readonly clock: () => number;
  private readonly deviceName: string;
  private readonly tier: string;
  private readonly canonicalTier: string;
  private readonly precision: string;
  private readonly generatedAt: string | null;
  private readonly benchmarkSink:
    | ((report: WebGpuBenchmarkReport) => void | Promise<void>)
    | undefined;
  private readonly releaseOnDispose: boolean;
  private firstInferenceMs: number | null = null;
  private lastInferenceMs = 0;
  private disposed = false;

  private constructor(
    loaded: OrtWebLoadedSession,
    probe: OrtWebCapabilityProbeResult,
    options: WebGpuTokenClassificationSessionOptions,
    modelPath: string,
    coldLoadMs: number,
    fallbackReason: string | null,
    releaseOnDispose: boolean,
  ) {
    this.loaded = loaded;
    this.backend = loaded.backend.backend;
    this.capabilityProbe = probe;
    this.modelName = options.modelName ?? "local-token-classifier";
    this.modelPath = modelPath;
    this.coldLoadMs = coldLoadMs;
    this.fallbackReason = fallbackReason;
    this.inputNames = {
      inputIds: options.inputNames?.inputIds ?? "input_ids",
      attentionMask: options.inputNames?.attentionMask ?? "attention_mask",
      tokenTypeIds: options.inputNames?.tokenTypeIds ?? "token_type_ids",
    };
    this.outputName = options.outputName ?? "logits";
    this.labelCount =
      options.labelCount === undefined
        ? undefined
        : positiveInteger(options.labelCount, "labelCount");
    this.runOptions = options.runOptions;
    this.clock = options.clock ?? defaultClock;
    this.deviceName = options.deviceName ?? "browser";
    this.tier = options.tier ?? "unspecified";
    this.canonicalTier = options.canonicalTier ?? this.tier;
    this.precision = options.precision ?? "float32";
    this.generatedAt = options.generatedAt ?? null;
    this.benchmarkSink = options.benchmarkSink;
    this.releaseOnDispose = releaseOnDispose;
  }

  static async load(
    options: WebGpuTokenClassificationSessionOptions,
  ): Promise<WebGpuTokenClassificationSession> {
    const clock = options.clock ?? defaultClock;
    const modelPaths = validateModelPaths(options.modelPath);
    assertOfflineAssetPath(options.assetPath, "wasm asset path");
    const probeOptions: Parameters<typeof probeOrtWebCapabilities>[0] = {};
    if (options.globalScope !== undefined) {
      probeOptions.globalScope = options.globalScope;
    }
    if (options.capabilities !== undefined) {
      probeOptions.overrides = options.capabilities;
    }
    if (options.probeAdapter !== undefined) {
      probeOptions.probeAdapter = options.probeAdapter;
    }
    if (options.powerPreference !== undefined) {
      probeOptions.powerPreference = options.powerPreference;
    }
    let probe = await probeOrtWebCapabilities(probeOptions);
    const sessionCache = options.cache ?? new Map();
    const started = clock();
    let selectedPath = probe.profile.webgpu
      ? modelPaths.webgpu
      : modelPaths.wasm;
    let fallbackReason: string | null = null;
    let loaded: OrtWebLoadedSession;

    try {
      loaded = await loadSessionForProfile(
        options,
        selectedPath,
        probe.profile,
        sessionCache,
      );
    } catch (webGpuError) {
      const canFallback =
        probe.profile.webgpu &&
        probe.profile.wasm &&
        options.fallbackOnWebGpuError !== false;
      if (!canFallback) throw webGpuError;
      const fallbackProfile = { ...probe.profile, webgpu: false };
      selectedPath = modelPaths.wasm;
      try {
        loaded = await loadSessionForProfile(
          options,
          selectedPath,
          fallbackProfile,
          sessionCache,
        );
      } catch (wasmError) {
        throw new AggregateError(
          [webGpuError, wasmError],
          "WebGPU and wasm token-classification session creation failed",
        );
      }
      fallbackReason = "WebGPU session creation failed; wasm fallback selected";
      probe = {
        ...probe,
        profile: fallbackProfile,
        reason: fallbackReason,
      };
    }

    return new WebGpuTokenClassificationSession(
      loaded,
      probe,
      options,
      selectedPath,
      elapsedMilliseconds(started, clock()),
      fallbackReason,
      options.cache === undefined,
    );
  }

  async run(tokens: WebGpuTokenBatch): Promise<TokenClassificationLogits> {
    this.ensureActive();
    const normalized = normalizeTokenBatch(tokens);
    const ownedInputs: OrtTensorLike[] = [];
    const feeds: OrtFeeds = { ...(tokens.additionalFeeds ?? {}) };
    const inputIds = createInt64Tensor(
      this.loaded,
      normalized.inputIds,
      normalized.dims,
    );
    const attentionMask = createInt64Tensor(
      this.loaded,
      normalized.attentionMask,
      normalized.dims,
    );
    ownedInputs.push(inputIds, attentionMask);
    feeds[this.inputNames.inputIds] = inputIds;
    feeds[this.inputNames.attentionMask] = attentionMask;
    if (normalized.tokenTypeIds !== undefined) {
      const tokenTypeIds = createInt64Tensor(
        this.loaded,
        normalized.tokenTypeIds,
        normalized.dims,
      );
      ownedInputs.push(tokenTypeIds);
      feeds[this.inputNames.tokenTypeIds] = tokenTypeIds;
    }

    const started = this.clock();
    let outputs: OrtResults | undefined;
    try {
      outputs = await this.loaded.session.run(feeds, this.runOptions);
      const logits = extractLogits(
        outputs,
        this.outputName,
        normalized.batchSize,
        normalized.sequenceLength,
        this.labelCount,
      );
      this.lastInferenceMs = elapsedMilliseconds(started, this.clock());
      if (this.firstInferenceMs === null) {
        this.firstInferenceMs = this.lastInferenceMs;
      }
      return logits;
    } finally {
      for (const input of ownedInputs) input.dispose?.();
      if (outputs !== undefined) {
        for (const output of Object.values(outputs)) {
          if (isOrtTensorLike(output)) output.dispose?.();
        }
      }
    }
  }

  async benchmark(
    tokens: WebGpuTokenBatch,
    options: WebGpuBenchmarkOptions = {},
  ): Promise<WebGpuBenchmarkReport> {
    this.ensureActive();
    const iterations = positiveInteger(options.iterations ?? 5, "iterations");
    const warmupIterations = positiveInteger(
      options.warmupIterations ?? 1,
      "warmupIterations",
    );

    if (this.firstInferenceMs === null) {
      await this.run(tokens);
    }
    for (let index = 1; index < warmupIterations; index += 1) {
      await this.run(tokens);
    }

    const warmLatencies: number[] = [];
    for (let index = 0; index < iterations; index += 1) {
      await this.run(tokens);
      warmLatencies.push(this.lastInferenceMs);
    }
    const totalWarmMs = warmLatencies.reduce((total, value) => total + value, 0);
    const tokenCount = countActiveTokens(tokens);
    const tokensPerSecond =
      totalWarmMs > 0 ? (tokenCount * iterations * 1000) / totalWarmMs : 0;
    const deviceName = options.deviceName ?? this.deviceName;
    const firstInferenceMs = this.firstInferenceMs ?? 0;
    const deviceMetrics: WebGpuDeviceBenchmarkMetrics = {
      backend: this.backend,
      precision: this.precision,
      batch_size: tokens.batchSize,
      sequence_length: tokens.sequenceLength,
      latency: {
        cold_load_ms: this.coldLoadMs,
        first_inference_ms: firstInferenceMs,
        cold_ms: this.coldLoadMs + firstInferenceMs,
        warm_ms: mean(warmLatencies),
        p50_ms: percentile(warmLatencies, 0.5),
        p95_ms: percentile(warmLatencies, 0.95),
        count: iterations,
      },
      throughput: {
        tokens_per_second: tokensPerSecond,
      },
    };
    const report: WebGpuBenchmarkReport = {
      suite: WEBGPU_BENCHMARK_SUITE,
      model_name: this.modelName,
      device: `${this.backend}:${deviceName}`,
      fixture_count: 1,
      generated_at: options.generatedAt ?? this.generatedAt,
      metadata: {
        runtime: "onnxruntime-web",
        local_files_only: true,
        tier: options.tier ?? this.tier,
        canonical_tier: options.canonicalTier ?? this.canonicalTier,
        warmup_iterations: warmupIterations,
        token_count: tokenCount,
        fallback_used: this.fallbackReason !== null,
      },
      metrics: {
        devices: { [deviceName]: deviceMetrics },
      },
    };
    const sink = options.sink ?? this.benchmarkSink;
    if (sink !== undefined) await sink(report);
    return report;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    if (this.releaseOnDispose) await this.loaded.session.release?.();
  }

  private ensureActive(): void {
    if (this.disposed) {
      throw new Error("WebGPU token-classification session has been disposed");
    }
  }
}

export async function loadWebGpuTokenClassificationSession(
  options: WebGpuTokenClassificationSessionOptions,
): Promise<WebGpuTokenClassificationSession> {
  return WebGpuTokenClassificationSession.load(options);
}

export function decodeWebGpuTokenSpans(
  logits: TokenClassificationLogits,
  id2label: Readonly<Record<string, string>> | ReadonlyMap<number, string>,
  attentionMask?: TokenIdData,
): WebGpuTokenSpan[] {
  const [batchSize, sequenceLength, labelCount] = logits.dims;
  const expected = batchSize * sequenceLength * labelCount;
  if (logits.data.length !== expected) {
    throw new WebGpuVerificationError(
      `logits contain ${logits.data.length} values; expected ${expected}`,
    );
  }
  const mask =
    attentionMask === undefined
      ? undefined
      : int64Array(attentionMask, expectedMaskLength(batchSize, sequenceLength));
  const labels = normalizeLabelMap(id2label, labelCount);
  const spans: WebGpuTokenSpan[] = [];

  for (let batch = 0; batch < batchSize; batch += 1) {
    let active: { label: string; start: number } | undefined;
    const close = (end: number) => {
      if (active === undefined) return;
      spans.push({
        batch_index: batch,
        label: active.label,
        start_token: active.start,
        end_token: end,
      });
      active = undefined;
    };

    for (let token = 0; token < sequenceLength; token += 1) {
      const flatToken = batch * sequenceLength + token;
      if (mask !== undefined && mask[flatToken] === 0n) {
        close(token);
        continue;
      }
      const offset = flatToken * labelCount;
      let bestLabel = 0;
      let bestValue = logits.data[offset] ?? Number.NEGATIVE_INFINITY;
      for (let label = 1; label < labelCount; label += 1) {
        const value = logits.data[offset + label] ?? Number.NEGATIVE_INFINITY;
        if (value > bestValue) {
          bestValue = value;
          bestLabel = label;
        }
      }
      const parsed = parseTokenLabel(labels[bestLabel] ?? `LABEL_${bestLabel}`);
      if (parsed.kind === "outside") {
        close(token);
      } else if (parsed.kind === "single") {
        close(token);
        spans.push({
          batch_index: batch,
          label: parsed.label,
          start_token: token,
          end_token: token + 1,
        });
      } else if (parsed.kind === "begin") {
        close(token);
        active = { label: parsed.label, start: token };
      } else if (parsed.kind === "inside") {
        if (active?.label !== parsed.label) {
          close(token);
          active = { label: parsed.label, start: token };
        }
      } else {
        if (active?.label !== parsed.label) {
          close(token);
          active = { label: parsed.label, start: token };
        }
        close(token + 1);
      }
    }
    close(sequenceLength);
  }
  return spans;
}

export function evaluateWebGpuRecallGate(
  reference: readonly WebGpuTokenSpan[],
  candidate: readonly WebGpuTokenSpan[],
  options: WebGpuRecallGateOptions = {},
): WebGpuRecallGate {
  const spanTolerance = nonNegativeInteger(
    options.spanTolerance ?? 0,
    "spanTolerance",
  );
  const maxRecallDelta = boundedFraction(
    options.maxRecallDelta ?? DEFAULT_WEBGPU_MAX_RECALL_DELTA,
    "maxRecallDelta",
  );
  const criticalLabels = new Set(options.criticalLabels ?? []);
  const used = new Set<number>();
  const referenceByLabel = new Map<string, number>();
  const matchedByLabel = new Map<string, number>();
  let matched = 0;
  let criticalMissed = 0;

  for (const expected of reference) {
    referenceByLabel.set(
      expected.label,
      (referenceByLabel.get(expected.label) ?? 0) + 1,
    );
    const index = candidate.findIndex(
      (actual, candidateIndex) =>
        !used.has(candidateIndex) &&
        spansMatch(expected, actual, spanTolerance),
    );
    if (index >= 0) {
      used.add(index);
      matched += 1;
      matchedByLabel.set(
        expected.label,
        (matchedByLabel.get(expected.label) ?? 0) + 1,
      );
    } else if (criticalLabels.has(expected.label)) {
      criticalMissed += 1;
    }
  }

  const recall = reference.length === 0 ? 1 : matched / reference.length;
  const recallDelta = Math.max(0, 1 - recall);
  const perLabelRecall = Object.fromEntries(
    [...referenceByLabel.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([label, count]) => [
        label,
        (matchedByLabel.get(label) ?? 0) / count,
      ]),
  );
  return {
    reference_count: reference.length,
    candidate_count: candidate.length,
    matched_count: matched,
    recall,
    recall_delta: recallDelta,
    max_recall_delta: maxRecallDelta,
    critical_missed_count: criticalMissed,
    per_label_recall: perLabelRecall,
    passed:
      recallDelta <= maxRecallDelta + Number.EPSILON && criticalMissed === 0,
  };
}

export function certifyWebGpuReference(
  options: WebGpuReferenceCertificationOptions,
): WebGpuReferenceCertification {
  const tolerance = nonNegativeFinite(
    options.tolerance ?? DEFAULT_WEBGPU_LOGIT_TOLERANCE,
    "tolerance",
  );
  const spanTolerance = nonNegativeInteger(
    options.spanTolerance ?? 0,
    "spanTolerance",
  );
  if (!sameDimensions(options.referenceLogits.dims, options.candidateLogits.dims)) {
    throw new WebGpuVerificationError(
      "WebGPU logits shape does not match the Python reference",
    );
  }
  ensureFinite(options.referenceLogits.data, "reference logits");
  ensureFinite(options.candidateLogits.data, "candidate logits");
  if (
    options.referenceLogits.data.length !== options.candidateLogits.data.length
  ) {
    throw new WebGpuVerificationError(
      "WebGPU logits length does not match the Python reference",
    );
  }

  let maxAbsLogitDelta = 0;
  for (let index = 0; index < options.referenceLogits.data.length; index += 1) {
    maxAbsLogitDelta = Math.max(
      maxAbsLogitDelta,
      Math.abs(
        (options.referenceLogits.data[index] ?? 0) -
          (options.candidateLogits.data[index] ?? 0),
      ),
    );
  }
  if (maxAbsLogitDelta > tolerance) {
    throw new WebGpuVerificationError(
      `WebGPU logits exceeded tolerance ${tolerance}: max abs delta ${maxAbsLogitDelta}`,
    );
  }

  const referenceSpans = decodeWebGpuTokenSpans(
    options.referenceLogits,
    options.id2label,
    options.attentionMask,
  );
  const candidateSpans = decodeWebGpuTokenSpans(
    options.candidateLogits,
    options.id2label,
    options.attentionMask,
  );
  const recallGateOptions: WebGpuRecallGateOptions = {
    spanTolerance,
  };
  if (options.maxRecallDelta !== undefined) {
    recallGateOptions.maxRecallDelta = options.maxRecallDelta;
  }
  if (options.criticalLabels !== undefined) {
    recallGateOptions.criticalLabels = options.criticalLabels;
  }
  const recallGate = evaluateWebGpuRecallGate(
    referenceSpans,
    candidateSpans,
    recallGateOptions,
  );
  const spanParityPassed =
    referenceSpans.length === candidateSpans.length &&
    recallGate.matched_count === referenceSpans.length;
  if (!spanParityPassed) {
    throw new WebGpuVerificationError(
      "WebGPU decoded token spans do not match the Python reference",
    );
  }
  if (!recallGate.passed) {
    throw new WebGpuVerificationError(
      "WebGPU token-classification recall gate failed",
    );
  }
  return {
    tolerance,
    max_abs_logit_delta: maxAbsLogitDelta,
    span_tolerance: spanTolerance,
    span_parity_passed: true,
    reference_token_spans: referenceSpans,
    candidate_token_spans: candidateSpans,
    recall_gate: recallGate,
    passed: true,
  };
}

async function loadSessionForProfile(
  options: WebGpuTokenClassificationSessionOptions,
  modelPath: string,
  profile: OrtWebCapabilityProfile,
  cache: OrtWebSessionCache,
): Promise<OrtWebLoadedSession> {
  const loaderOptions: Parameters<typeof loadOrtWebSession>[0] = {
    modelPath,
    assetPath: options.assetPath,
    capabilities: profile,
    cache,
  };
  if (options.runtime !== undefined) loaderOptions.runtime = options.runtime;
  if (options.globalScope !== undefined) {
    loaderOptions.globalScope = options.globalScope;
  }
  if (options.sessionOptions !== undefined) {
    loaderOptions.sessionOptions = options.sessionOptions;
  }
  return loadOrtWebSession(loaderOptions);
}

function validateModelPaths(value: string | WebGpuModelPaths): WebGpuModelPaths {
  if (typeof value === "string") {
    const path = assertOfflineAssetPath(value, "model path");
    return { webgpu: path, wasm: path };
  }
  return {
    webgpu: assertOfflineAssetPath(value.webgpu, "WebGPU model path"),
    wasm: assertOfflineAssetPath(value.wasm, "wasm model path"),
  };
}

function normalizeTokenBatch(tokens: WebGpuTokenBatch): {
  inputIds: BigInt64Array;
  attentionMask: BigInt64Array;
  tokenTypeIds?: BigInt64Array;
  batchSize: number;
  sequenceLength: number;
  dims: readonly [number, number];
} {
  const batchSize = positiveInteger(tokens.batchSize, "batchSize");
  const sequenceLength = positiveInteger(
    tokens.sequenceLength,
    "sequenceLength",
  );
  const length = batchSize * sequenceLength;
  const inputIds = int64Array(tokens.inputIds, length);
  const attentionMask =
    tokens.attentionMask === undefined
      ? new BigInt64Array(length).fill(1n)
      : int64Array(tokens.attentionMask, length);
  const normalized: {
    inputIds: BigInt64Array;
    attentionMask: BigInt64Array;
    tokenTypeIds?: BigInt64Array;
    batchSize: number;
    sequenceLength: number;
    dims: readonly [number, number];
  } = {
    inputIds,
    attentionMask,
    batchSize,
    sequenceLength,
    dims: [batchSize, sequenceLength],
  };
  if (tokens.tokenTypeIds !== undefined) {
    normalized.tokenTypeIds = int64Array(tokens.tokenTypeIds, length);
  }
  return normalized;
}

function int64Array(values: TokenIdData, expectedLength: number): BigInt64Array {
  if (values.length !== expectedLength) {
    throw new RangeError(
      `token tensor must contain ${expectedLength} values, got ${values.length}`,
    );
  }
  const result = new BigInt64Array(expectedLength);
  for (let index = 0; index < expectedLength; index += 1) {
    const value = values[index];
    if (value === undefined) {
      throw new RangeError("token tensor contains a missing value");
    }
    let bigintValue: bigint;
    if (typeof value === "bigint") {
      bigintValue = value;
    } else {
      if (!Number.isSafeInteger(value)) {
        throw new RangeError("token ids and masks must be safe integers");
      }
      bigintValue = BigInt(value);
    }
    if (bigintValue < MIN_INT64 || bigintValue > MAX_INT64) {
      throw new RangeError("token ids and masks must fit signed int64");
    }
    result[index] = bigintValue;
  }
  return result;
}

function createInt64Tensor(
  loaded: OrtWebLoadedSession,
  data: BigInt64Array,
  dims: readonly [number, number],
): OrtTensorLike {
  const Tensor = loaded.runtime?.Tensor;
  return Tensor === undefined
    ? { type: "int64", data, dims }
    : new Tensor("int64", data, dims);
}

function extractLogits(
  outputs: OrtResults,
  requestedOutputName: string,
  batchSize: number,
  sequenceLength: number,
  configuredLabelCount: number | undefined,
): TokenClassificationLogits {
  const outputName =
    requestedOutputName in outputs
      ? requestedOutputName
      : Object.keys(outputs).length === 1
        ? (Object.keys(outputs)[0] ?? requestedOutputName)
        : requestedOutputName;
  const tensor = outputs[outputName];
  if (!isOrtTensorLike(tensor) || tensor.data === undefined || tensor.dims === undefined) {
    throw new Error(`ONNX Runtime output ${JSON.stringify(outputName)} is not a tensor`);
  }
  if (tensor.dims.length !== 3) {
    throw new RangeError("token-classification logits must have rank 3");
  }
  const dims = tensor.dims.map((value) => positiveInteger(value, "logits dimension"));
  const [actualBatch, actualSequence, labelCount] = dims;
  if (
    actualBatch !== batchSize ||
    actualSequence !== sequenceLength ||
    labelCount === undefined
  ) {
    throw new RangeError(
      "token-classification logits do not match the input batch and sequence shape",
    );
  }
  if (configuredLabelCount !== undefined && labelCount !== configuredLabelCount) {
    throw new RangeError(
      `token-classification logits expose ${labelCount} labels; expected ${configuredLabelCount}`,
    );
  }
  const data = float32TensorData(tensor);
  const expectedLength = actualBatch * actualSequence * labelCount;
  if (data.length !== expectedLength) {
    throw new RangeError(
      `token-classification logits contain ${data.length} values; expected ${expectedLength}`,
    );
  }
  ensureFinite(data, "token-classification logits");
  return {
    data,
    dims: [actualBatch, actualSequence, labelCount],
    outputName,
  };
}

function float32TensorData(tensor: OrtTensorLike): Float32Array {
  const data = tensor.data;
  if (tensor.type === "float16" && data instanceof Uint16Array) {
    return Float32Array.from(data, decodeFloat16);
  }
  if (
    Array.isArray(data) ||
    data instanceof Float32Array ||
    data instanceof Float64Array ||
    data instanceof Int8Array ||
    data instanceof Uint8Array ||
    data instanceof Int16Array ||
    data instanceof Uint16Array ||
    data instanceof Int32Array ||
    data instanceof Uint32Array
  ) {
    return Float32Array.from(data);
  }
  throw new TypeError("token-classification logits must contain numeric data");
}

function decodeFloat16(value: number): number {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function countActiveTokens(tokens: WebGpuTokenBatch): number {
  if (tokens.attentionMask === undefined) {
    return tokens.batchSize * tokens.sequenceLength;
  }
  let count = 0;
  for (const value of tokens.attentionMask) {
    if (typeof value === "bigint" ? value !== 0n : value !== 0) count += 1;
  }
  return count;
}

function normalizeLabelMap(
  value: Readonly<Record<string, string>> | ReadonlyMap<number, string>,
  labelCount: number,
): string[] {
  const labels = new Array<string>(labelCount);
  if (value instanceof Map) {
    for (const [index, label] of value.entries()) labels[index] = label;
  } else {
    for (const [index, label] of Object.entries(value)) {
      const parsed = Number(index);
      if (Number.isInteger(parsed)) labels[parsed] = label;
    }
  }
  for (let index = 0; index < labelCount; index += 1) {
    if (labels[index] === undefined) {
      throw new WebGpuVerificationError(`id2label is missing label ${index}`);
    }
  }
  return labels;
}

function parseTokenLabel(value: string):
  | { kind: "outside" }
  | { kind: "single" | "begin" | "inside" | "end"; label: string } {
  if (value.toUpperCase() === "O") return { kind: "outside" };
  const match = /^([BIESUL])[-_](.+)$/i.exec(value);
  if (match === null) return { kind: "single", label: value };
  const prefix = match[1]?.toUpperCase();
  const label = match[2] ?? value;
  if (prefix === "B") return { kind: "begin", label };
  if (prefix === "I") return { kind: "inside", label };
  if (prefix === "E" || prefix === "L") return { kind: "end", label };
  return { kind: "single", label };
}

function spansMatch(
  expected: WebGpuTokenSpan,
  actual: WebGpuTokenSpan,
  tolerance: number,
): boolean {
  return (
    expected.batch_index === actual.batch_index &&
    expected.label === actual.label &&
    Math.abs(expected.start_token - actual.start_token) <= tolerance &&
    Math.abs(expected.end_token - actual.end_token) <= tolerance
  );
}

function sameDimensions(
  left: readonly number[],
  right: readonly number[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function finiteFloat32Array(
  values: Float32Array | readonly number[],
  label: string,
): Float32Array {
  const result = Float32Array.from(values);
  ensureFinite(result, label);
  return result;
}

function ensureFinite(values: ArrayLike<number>, label: string): void {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) {
      throw new RangeError(`${label} must contain only finite numbers`);
    }
  }
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return value;
}

function nonNegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative integer`);
  }
  return value;
}

function nonNegativeFinite(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative finite number`);
  }
  return value;
}

function boundedFraction(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${label} must be between 0 and 1`);
  }
  return value;
}

function elapsedMilliseconds(started: number, finished: number): number {
  const elapsed = finished - started;
  return Number.isFinite(elapsed) && elapsed > 0 ? elapsed : 0;
}

function defaultClock(): number {
  return globalThis.performance?.now?.() ?? Date.now();
}

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function percentile(values: readonly number[], fraction: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(fraction * sorted.length) - 1),
  );
  return sorted[index] ?? 0;
}

function expectedMaskLength(batchSize: number, sequenceLength: number): number {
  return batchSize * sequenceLength;
}

function isOrtTensorLike(value: unknown): value is OrtTensorLike {
  return typeof value === "object" && value !== null;
}

interface GpuBufferLike {
  destroy(): void;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  unmap(): void;
}

interface GpuShaderModuleLike {
  getCompilationInfo?: () => Promise<{
    messages: readonly {
      type: string;
      message: string;
      lineNum?: number;
    }[];
  }>;
}

interface GpuComputePipelineLike {
  getBindGroupLayout(index: number): unknown;
}

interface GpuComputePassLike {
  setPipeline(pipeline: GpuComputePipelineLike): void;
  setBindGroup(index: number, bindGroup: unknown): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface GpuCommandEncoderLike {
  beginComputePass(descriptor?: Record<string, unknown>): GpuComputePassLike;
  copyBufferToBuffer(
    source: GpuBufferLike,
    sourceOffset: number,
    destination: GpuBufferLike,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}

interface GpuDeviceLike {
  queue: {
    writeBuffer(
      buffer: GpuBufferLike,
      bufferOffset: number,
      data: ArrayBufferView,
    ): void;
    submit(commandBuffers: readonly unknown[]): void;
  };
  createBuffer(descriptor: {
    label?: string;
    size: number;
    usage: number;
  }): GpuBufferLike;
  createShaderModule(descriptor: {
    code: string;
    label?: string;
  }): GpuShaderModuleLike;
  createComputePipeline(descriptor: Record<string, unknown>): GpuComputePipelineLike;
  createComputePipelineAsync?: (
    descriptor: Record<string, unknown>,
  ) => Promise<GpuComputePipelineLike>;
  createBindGroup(descriptor: Record<string, unknown>): unknown;
  createCommandEncoder(descriptor?: Record<string, unknown>): GpuCommandEncoderLike;
}

function asGpuDevice(value: unknown): GpuDeviceLike {
  if (
    typeof value !== "object" ||
    value === null ||
    !("queue" in value) ||
    !("createBuffer" in value) ||
    !("createShaderModule" in value) ||
    !("createComputePipeline" in value) ||
    !("createBindGroup" in value) ||
    !("createCommandEncoder" in value)
  ) {
    throw new TypeError("a WebGPU GPUDevice is required");
  }
  return value as GpuDeviceLike;
}
