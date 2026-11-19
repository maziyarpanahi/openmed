require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

# OpenMedKit is distributed as a Swift package, not as a CocoaPods pod.
# React Native 0.75 and later expose `spm_dependency`, which adds the Swift
# package product to this pod's target during `pod install`.
openmedkit_swift_package_url = "https://github.com/maziyarpanahi/openmed.git"
openmedkit_swift_package_minimum_version = "3.2.0"

Pod::Spec.new do |s|
  s.name         = "openmedkit-react-native"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = "https://github.com/maziyarpanahi/openmed"
  s.license      = package["license"]
  s.author       = { "Maziyar Panahi" => "maziyar.panahi@iscpif.fr" }
  s.source       = { :path => "." }
  s.platforms    = { :ios => "17.0" }
  s.swift_version = "5.0"
  s.source_files = "ios/**/*.{h,m,mm,swift}"
  s.pod_target_xcconfig = { "DEFINES_MODULE" => "YES" }

  if respond_to?(:install_modules_dependencies, true)
    install_modules_dependencies(s)
  else
    s.dependency "React-Core"
  end

  if respond_to?(:spm_dependency, true)
    spm_dependency(
      s,
      url: openmedkit_swift_package_url,
      requirement: {
        kind: "upToNextMajorVersion",
        minimumVersion: openmedkit_swift_package_minimum_version,
      },
      products: ["OpenMedKit"]
    )
  elsif respond_to?(:use_react_native!, true)
    raise "openmedkit-react-native requires React Native 0.77 or later: " \
          "OpenMedKit is linked through spm_dependency."
  end
end
