const path = require("path");
const { getDefaultConfig, mergeConfig } = require("@react-native/metro-config");

const exampleRoot = __dirname;
const packageRoot = path.resolve(exampleRoot, "..");
const parityFixtures = path.resolve(packageRoot, "..", "..", "tests", "fixtures", "parity");
const exampleModules = path.join(exampleRoot, "node_modules");

function escapeForRegExp(value) {
  return value.replace(/[/\\^$*+?.()|[\]{}]/g, "\\$&");
}

/**
 * Metro configuration for the parity app.
 *
 * The bridge package source (`..`) and the shared parity corpora under
 * `tests/fixtures/parity` live outside this app, so both are watched. React and
 * React Native always resolve from this app's node_modules so the bridge and
 * the app share one copy.
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  watchFolders: [packageRoot, parityFixtures],
  resolver: {
    nodeModulesPaths: [exampleModules],
    extraNodeModules: {
      react: path.join(exampleModules, "react"),
      "react-native": path.join(exampleModules, "react-native"),
    },
    blockList: [
      new RegExp(`^${escapeForRegExp(path.join(packageRoot, "node_modules"))}\\/.*$`),
    ],
  },
};

module.exports = mergeConfig(getDefaultConfig(exampleRoot), config);
