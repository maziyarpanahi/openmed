/**
 * Default OpenMed utility-process entrypoint.
 *
 * Bundle this module as the app's utility-process script. When it runs inside
 * an Electron utility process it starts the OpenMed loop with the default
 * `openmed` loaders. Apps that need a custom pipeline loader should build
 * their own entrypoint around `startUtilityProcess` from
 * `@openmed/openmedkit-electron/utility-runtime` instead, so only one loop
 * starts.
 */
import { startUtilityProcess, type UtilityParentPortLike } from "./utility-runtime";

export {
  createUtilityDeidentifyHandler,
  installOfflineNetworkGuard,
  loadVerifiedBundlePipeline,
  startUtilityProcess,
  type UtilityHandlerOptions,
  type UtilityMessageEvent,
  type UtilityMessageHandler,
  type UtilityParentPortLike,
  type VerifiedBundleRuntime,
  type VerifiedElectronBundle,
} from "./utility-runtime";

const electronProcess = process as NodeJS.Process & {
  parentPort?: UtilityParentPortLike | null;
};
if (electronProcess.parentPort) {
  startUtilityProcess(electronProcess.parentPort);
}
