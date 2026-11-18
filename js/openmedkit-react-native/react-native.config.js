// React Native CLI autolinking configuration for the OpenMedKit bridge.
//
// The CLI loads this file synchronously with require(), so it stays CommonJS.
// iOS: the CLI discovers openmedkit-react-native.podspec at the package root.
// Android: sourceDir is resolved relative to the package root.
module.exports = {
  dependency: {
    platforms: {
      ios: {},
      android: {
        sourceDir: "android",
        packageImportPath:
          "import com.openmed.openmedkit.reactnative.OpenMedKitRnPackage;",
        packageInstance: "new OpenMedKitRnPackage()",
      },
    },
  },
};
