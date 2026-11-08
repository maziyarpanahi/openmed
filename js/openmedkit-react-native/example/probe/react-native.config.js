// React Native CLI autolinking for the example-only parity probe.
module.exports = {
  dependency: {
    platforms: {
      ios: {},
      android: {
        sourceDir: "android",
        packageImportPath:
          "import com.openmed.openmedkit.parityprobe.OpenMedKitParityProbePackage;",
        packageInstance: "new OpenMedKitParityProbePackage()",
      },
    },
  },
};
