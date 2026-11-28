// Portable task packs for the web runtime (docs/models/task-packs.md).
//
// A pack names a job, not a model. Its ordered `web` candidates come from the
// same manifest the Python resolver reads (openmed/core/data/task_packs.json),
// joined at build time with the web ONNX model catalog and bundled as
// task-packs.generated.ts, so the catalog rows stay off this entry. Loading
// tries the candidates in rank order through the OpenMed ONNX loader. Without
// `allowRemoteModels: true` it reads local files only, refuses remote runtime
// assets and never fetches. Inference stays on the device, nothing is logged,
// and errors carry codes only.
import { loadOnnxModel, loadOnnxModelWithLocalAssets } from "./model-loader";
import { OpenMedOfflineError } from "./offline";
import { TASK_PACK_NOTICES, WEB_TASK_PACKS } from "./task-packs.generated";
import type {
  DeidentifyOptions,
  ExtractPiiOptions,
  LoadOnnxModelOptions,
  OpenMedDeidentifyResult,
  OpenMedSpan,
  RawTokenClassificationPipeline,
  TokenClassificationPipeline,
} from "./types";

/** One candidate as the bundled manifest records it for the web runtime. */
export interface WebTaskPackCandidateData {
  readonly rank: number;
  readonly repo_id: string;
  readonly param_count: number;
  readonly license: string;
  /** The candidate's `reproducibility_hash` in the web ONNX model catalog. */
  readonly reproducibility_hash: string;
}

/** A pack's web runtime entry, with the fields of the Python `resolve()`. */
export interface WebTaskPackData {
  readonly pack_id: string;
  readonly pack_version: string;
  readonly task: string;
  readonly runtime: string;
  readonly format: string;
  readonly status: string;
  readonly candidates: readonly WebTaskPackCandidateData[];
  readonly canonical_labels: readonly string[];
  readonly notice_key: string;
}

/** A resolved candidate: the manifest row joined with its catalog row. */
export type TaskPackCandidate = WebTaskPackCandidateData;

/** The ordered web candidates of a pack. */
export interface TaskPackResolution {
  readonly pack_id: string;
  readonly pack_version: string;
  readonly task: string;
  readonly runtime: "web";
  readonly format: "onnx";
  readonly status: "resolved" | "no_candidate";
  /** Candidates, best first; empty when `status` is `no_candidate`. */
  readonly candidates: readonly TaskPackCandidate[];
  /** Labels every candidate is guaranteed to emit. */
  readonly canonical_labels: readonly string[];
  /** Key of the notice to show with the pack's output. */
  readonly notice_key: string;
  /** Default English text of that notice. */
  readonly notice: string;
}

export type TaskPackErrorCode =
  | "unknown_pack"
  | "catalog_mismatch"
  | "no_candidate"
  | "local_model_unavailable"
  | "candidates_unavailable"
  | "remote_model_refused"
  | "remote_asset_refused"
  | "pipeline_disposal_failed";

/** A value-free task pack failure: the message names the code only. */
export class TaskPackError extends Error {
  readonly code: TaskPackErrorCode;

  constructor(code: TaskPackErrorCode) {
    super(`OpenMed task pack failed: ${code}`);
    this.name = "TaskPackError";
    this.code = code;
  }
}

export interface TaskPackLoadOptions
  extends Omit<LoadOnnxModelOptions, "localFilesOnly" | "allowRemoteModels" | "revision"> {
  /**
   * Let the loader download a candidate (and remote runtime assets) when it is
   * not available locally. Off by default: the pack then reads local files
   * only and fails with `local_model_unavailable` instead of fetching.
   */
  allowRemoteModels?: boolean;
  /** Use this pipeline instead of loading a candidate; the caller owns it. */
  pipeline?: RawTokenClassificationPipeline;
}

/** Extraction options a loaded pack accepts; it supplies model and pipeline. */
export type TaskPackExtractOptions = Omit<
  ExtractPiiOptions,
  "pipeline" | "model" | "modelLoader" | "loaderOptions" | "offlineOnly"
>;
export type TaskPackDeidentifyOptions = Omit<
  DeidentifyOptions,
  "pipeline" | "model" | "modelLoader" | "loaderOptions" | "offlineOnly"
>;

/** A pack with a pipeline ready on this device. */
export interface LoadedTaskPack {
  readonly resolution: TaskPackResolution;
  /** The candidate that loaded, or null when a pipeline was supplied. */
  readonly candidate: TaskPackCandidate | null;
  readonly notice_key: string;
  readonly notice: string;
  /** `extractPii()` with the pack's pipeline. */
  extract(text: string, options?: TaskPackExtractOptions): Promise<OpenMedSpan[]>;
  /** `deidentify()` with the pack's pipeline. */
  deidentify(text: string, options?: TaskPackDeidentifyOptions): Promise<OpenMedDeidentifyResult>;
  /** Release a pipeline the pack loaded; a supplied pipeline stays the caller's. */
  dispose(): Promise<void>;
}

export interface TaskPack {
  readonly id: string;
  /** Ordered web candidates from the bundled manifest and catalog. No I/O. */
  resolve(): TaskPackResolution;
  /** Load the first available candidate, or wrap a supplied pipeline. */
  load(options?: TaskPackLoadOptions): Promise<LoadedTaskPack>;
}

type ModelLoad = (
  model: string,
  options: LoadOnnxModelOptions,
) => Promise<TokenClassificationPipeline>;

/** Package internals a pack runs through, supplied by the package entry. */
export interface TaskPackRuntime {
  extractPii(text: string, options: ExtractPiiOptions): Promise<OpenMedSpan[]>;
  deidentify(text: string, options: DeidentifyOptions): Promise<OpenMedDeidentifyResult>;
  /** Revision to request for a candidate when the package pins one. */
  pinnedRevisions?: Readonly<Record<string, string>>;
}

/** Bundled data a pack reads; tests replace it with synthetic rows. */
export interface TaskPackSources {
  packs?: readonly WebTaskPackData[];
  notices?: Readonly<Record<string, string>>;
  /** Loader used with remote consent; defaults to `loadOnnxModel`. */
  loadRemote?: ModelLoad;
  /** Loader used without it; defaults to the local-assets-only ONNX loader. */
  loadLocal?: ModelLoad;
}

const CATALOG_HASH = /^sha256:[0-9a-f]{64}$/;

function resolvePack(
  packId: string,
  sources: Required<Pick<TaskPackSources, "packs" | "notices">>,
): TaskPackResolution {
  const pack = sources.packs.find((item) => item.pack_id === packId);
  if (pack === undefined) {
    throw new TaskPackError("unknown_pack");
  }
  const notice = Object.hasOwn(sources.notices, pack.notice_key)
    ? sources.notices[pack.notice_key]
    : undefined;
  if (
    pack.runtime !== "web" ||
    pack.format !== "onnx" ||
    notice === undefined ||
    (pack.status !== "resolved" && pack.status !== "no_candidate") ||
    (pack.status === "resolved") !== pack.candidates.length > 0
  ) {
    throw new TaskPackError("catalog_mismatch");
  }
  const candidates = [...pack.candidates]
    .sort((left, right) => left.rank - right.rank)
    .map((candidate, index): TaskPackCandidate => {
      // Ranks are sequential and every candidate was joined with its catalog
      // row; anything else means the bundled data were edited or corrupted.
      if (candidate.rank !== index + 1 || !CATALOG_HASH.test(candidate.reproducibility_hash)) {
        throw new TaskPackError("catalog_mismatch");
      }
      return Object.freeze({ ...candidate });
    });
  return Object.freeze({
    pack_id: pack.pack_id,
    pack_version: pack.pack_version,
    task: pack.task,
    runtime: "web",
    format: "onnx",
    status: pack.status,
    candidates: Object.freeze(candidates),
    canonical_labels: Object.freeze([...pack.canonical_labels]),
    notice_key: pack.notice_key,
    notice,
  });
}

function loadedPack(
  runtime: TaskPackRuntime,
  resolution: TaskPackResolution,
  candidate: TaskPackCandidate | null,
  pipeline: RawTokenClassificationPipeline & { dispose?: () => Promise<void> | void },
): LoadedTaskPack {
  const owned = candidate !== null;
  // A loaded candidate is named in span metadata; a supplied pipeline is not,
  // so its spans equal a direct extractPii() call with that pipeline.
  const model = owned ? { model: candidate.repo_id } : {};
  let disposed = false;
  return Object.freeze({
    resolution,
    candidate,
    notice_key: resolution.notice_key,
    notice: resolution.notice,
    extract: (text: string, options: TaskPackExtractOptions = {}) =>
      runtime.extractPii(text, { ...options, ...model, pipeline }),
    deidentify: (text: string, options: TaskPackDeidentifyOptions = {}) =>
      runtime.deidentify(text, { ...options, ...model, pipeline }),
    async dispose() {
      if (!owned || disposed) {
        return;
      }
      disposed = true;
      try {
        await pipeline.dispose?.();
      } catch {
        throw new TaskPackError("pipeline_disposal_failed");
      }
    },
  });
}

/**
 * Create a task pack over the bundled manifest and catalog.
 *
 * Args:
 *   packId: Pack identifier, for example `pii`.
 *   runtime: The package functions the pack runs through.
 *   sources: Replacement data or loaders, for tests.
 *
 * Returns:
 *   The pack. Nothing is resolved or loaded until `resolve()` or `load()`.
 */
export function createTaskPack(
  packId: string,
  runtime: TaskPackRuntime,
  sources: TaskPackSources = {},
): TaskPack {
  const data = {
    packs: sources.packs ?? WEB_TASK_PACKS,
    notices: sources.notices ?? TASK_PACK_NOTICES,
  };
  const loadRemote = sources.loadRemote ?? loadOnnxModel;
  const loadLocal = sources.loadLocal ?? loadOnnxModelWithLocalAssets;
  const resolve = () => resolvePack(packId, data);
  return Object.freeze({
    id: packId,
    resolve,
    async load(options: TaskPackLoadOptions = {}): Promise<LoadedTaskPack> {
      const resolution = resolve();
      const { allowRemoteModels, pipeline, ...loaderOptions } = options;
      if (pipeline !== undefined) {
        return loadedPack(runtime, resolution, null, pipeline);
      }
      if (resolution.status === "no_candidate") {
        throw new TaskPackError("no_candidate");
      }
      const consent = allowRemoteModels === true;
      const load = consent ? loadRemote : loadLocal;
      for (const candidate of resolution.candidates) {
        const revision = runtime.pinnedRevisions?.[candidate.repo_id];
        let loaded: TokenClassificationPipeline;
        try {
          loaded = await load(candidate.repo_id, {
            ...loaderOptions,
            ...(revision === undefined ? {} : { revision }),
            localFilesOnly: !consent,
            allowRemoteModels: consent,
          });
        } catch (error) {
          // Every candidate shares the runtime settings, so an offline-policy
          // refusal ends the search. Other failures mean "not available here":
          // fall back to the next candidate without recording why.
          if (error instanceof OpenMedOfflineError && error.code !== "local_assets_unavailable") {
            throw new TaskPackError(error.code);
          }
          continue;
        }
        return loadedPack(runtime, resolution, candidate, loaded);
      }
      throw new TaskPackError(consent ? "candidates_unavailable" : "local_model_unavailable");
    },
  });
}
