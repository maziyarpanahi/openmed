export {
  ElectronDeidentifyService,
  type ElectronDeidentifyServiceEvent,
  type ElectronDeidentifyServiceOptions,
  type UtilityProcessLike,
  type UtilityProcessModuleLike,
} from "./main-service";
export {
  OPENMED_DEIDENTIFY_CHANNEL,
  OPENMED_ELECTRON_SCHEMA_VERSION,
  createElectronDeidentifyClient,
  redactTextWithSpans,
  registerElectronDeidentifyIpc,
  type ElectronDeidentifyRequest,
  type ElectronDeidentifyResponse,
  type ElectronDeidentifyServiceLike,
  type IpcMainLike,
  type IpcRendererLike,
  type RendererOpenMedSpan,
} from "./ipc";
