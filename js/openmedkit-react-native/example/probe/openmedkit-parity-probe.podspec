require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

# Example-only pod: the direct OpenMedKit probe for the parity app. It links the
# same OpenMedKit Swift package product as the OpenMedKitRN bridge pod (#4478),
# so CocoaPods resolves one OpenMedKit version for both. Override the source
# with OPENMEDKIT_SPM_URL / OPENMEDKIT_SPM_MINIMUM_VERSION when the bridge pod
# is pointed at another revision.
openmedkit_swift_package_url = ENV.fetch(
  "OPENMEDKIT_SPM_URL", "https://github.com/maziyarpanahi/openmed.git"
)
openmedkit_swift_package_minimum_version = ENV.fetch(
  "OPENMEDKIT_SPM_MINIMUM_VERSION", "3.2.0"
)

Pod::Spec.new do |s|
  s.name          = "openmedkit-parity-probe"
  s.version       = package["version"]
  s.summary       = package["description"]
  s.homepage      = "https://github.com/maziyarpanahi/openmed"
  s.license       = package["license"]
  s.authors       = "OpenMed"
  s.source        = { :path => "." }
  s.platforms     = { :ios => "17.0" }
  s.swift_version = "5.0"
  s.source_files  = "ios/**/*.{h,m,mm,swift}"
  s.pod_target_xcconfig = { "DEFINES_MODULE" => "YES" }

  if respond_to?(:install_modules_dependencies, true)
    install_modules_dependencies(s)
  else
    s.dependency "React-Core"
  end

  unless respond_to?(:spm_dependency, true)
    raise "openmedkit-parity-probe needs React Native 0.77 or later (spm_dependency)."
  end

  spm_dependency(
    s,
    url: openmedkit_swift_package_url,
    requirement: {
      kind: "upToNextMajorVersion",
      minimumVersion: openmedkit_swift_package_minimum_version,
    },
    products: ["OpenMedKit"]
  )
end
